#!/usr/bin/env python3
"""Read STM32 telemetry and publish sensor and odometry data."""

import math
import time

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

try:
    import serial
except ImportError:  # pragma: no cover - reported clearly at runtime
    serial = None


FRAME_PREFIX = "$DATA,"
FRAME_END = "#"
FRAME_FIELDS = (
    "odom_linear",
    "odom_angular",
    "temp",
    "hum",
    "mq2",
    "pm25",
    "pm10",
    "hc",
)
MAX_BUFFER_LEN = 4096


def format_velocity_command(linear_x, angular_z):
    """Format one velocity command using the existing STM32 protocol."""
    return f"{linear_x:.3f},0.000,{angular_z:.3f},0.000,0.000\n"


class VelocityCommandState:
    """Validate, limit, and time out the latest velocity command."""

    def __init__(self, cmd_timeout, max_linear_x, max_angular_z):
        self.cmd_timeout = max(0.0, float(cmd_timeout))
        self.max_linear_x = abs(float(max_linear_x))
        self.max_angular_z = abs(float(max_angular_z))
        self.stop()

    def accept(self, linear_x, angular_z, received_at):
        """Store a finite, limited command and return whether it was accepted."""
        if not math.isfinite(linear_x) or not math.isfinite(angular_z):
            return False

        self.linear_x = max(-self.max_linear_x, min(self.max_linear_x, linear_x))
        self.angular_z = max(
            -self.max_angular_z,
            min(self.max_angular_z, angular_z),
        )
        self.last_command_time = received_at
        return True

    def command_for(self, now, emergency_stop=False):
        """Return the active command or zero when stopped or timed out."""
        if emergency_stop:
            self.stop()
            return 0.0, 0.0

        if self.last_command_time is None:
            return 0.0, 0.0

        age = now - self.last_command_time
        if age < 0.0 or age > self.cmd_timeout:
            self.stop()
            return 0.0, 0.0

        return self.linear_x, self.angular_z

    def stop(self):
        """Clear the active command so it cannot resume without a new message."""
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.last_command_time = None


def parse_stm32_frame(frame):
    """Return the eight finite values from one complete STM32 frame."""
    if not frame.startswith(FRAME_PREFIX) or not frame.endswith(FRAME_END):
        raise ValueError("frame must start with '$DATA,' and end with '#'")

    parts = frame[len(FRAME_PREFIX):-1].split(",")
    if len(parts) != len(FRAME_FIELDS):
        raise ValueError(
            f"frame has {len(parts)} fields, expected {len(FRAME_FIELDS)}"
        )

    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("frame contains a non-numeric field") from exc

    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("frame contains a non-finite field")

    return dict(zip(FRAME_FIELDS, numbers))


def _partial_prefix_suffix(value):
    """Keep only a suffix that could become the next frame prefix."""
    max_length = min(len(value), len(FRAME_PREFIX) - 1)
    for length in range(max_length, 0, -1):
        if value.endswith(FRAME_PREFIX[:length]):
            return value[-length:]
    return ""


def extract_serial_frames(buffer):
    """Extract complete frames while preserving a possible partial prefix."""
    frames = []
    dropped_noise = False

    while buffer:
        start = buffer.find(FRAME_PREFIX)
        if start < 0:
            remainder = _partial_prefix_suffix(buffer)
            discarded = buffer[:-len(remainder)] if remainder else buffer
            dropped_noise = dropped_noise or bool(discarded.strip())
            return frames, remainder, dropped_noise

        if start > 0:
            dropped_noise = dropped_noise or bool(buffer[:start].strip())
            buffer = buffer[start:]

        end = buffer.find(FRAME_END, len(FRAME_PREFIX))
        if end < 0:
            return frames, buffer, dropped_noise

        frames.append(buffer[:end + 1])
        buffer = buffer[end + 1:]

    return frames, "", dropped_noise


class PlanarOdometry:
    """Integrate planar velocity samples using their receive timestamps."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = None

    def update(self, linear, angular, stamp_seconds, max_integration_dt):
        """Update pose and return whether this sample was integrated."""
        if self.last_time is None:
            self.last_time = stamp_seconds
            return False

        dt = stamp_seconds - self.last_time
        self.last_time = stamp_seconds
        if dt <= 0.0 or dt > max_integration_dt:
            return False

        self.yaw += angular * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.x += linear * math.cos(self.yaw) * dt
        self.y += linear * math.sin(self.yaw) * dt
        return True

    def reset_time(self):
        """Make the first sample after a reconnect establish a new baseline."""
        self.last_time = None


def quaternion_from_yaw(yaw):
    """Return an ROS quaternion representing a planar yaw angle."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw * 0.5),
        w=math.cos(yaw * 0.5),
    )


class Stm32BridgeNode(Node):
    """Own the STM32 telemetry and optional command serial connection."""

    def __init__(self):
        super().__init__("stm32_bridge_node")

        self.declare_parameter("serial_port", "/dev/ttyS2")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("read_period", 0.02)
        self.declare_parameter("sensor_publish_rate", 10.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("max_integration_dt", 0.1)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("enable_cmd_vel", False)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("max_linear_x", 0.3)
        self.declare_parameter("max_angular_z", 1.2)
        self.declare_parameter("serial_command_rate", 20.0)
        self.declare_parameter("emergency_stop", False)

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.read_period = max(
            0.001, float(self.get_parameter("read_period").value)
        )
        self.sensor_publish_rate = max(
            0.1, float(self.get_parameter("sensor_publish_rate").value)
        )
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.max_integration_dt = max(
            0.0, float(self.get_parameter("max_integration_dt").value)
        )
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.enable_cmd_vel = bool(self.get_parameter("enable_cmd_vel").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_timeout = max(
            0.0, float(self.get_parameter("cmd_timeout").value)
        )
        self.max_linear_x = abs(
            float(self.get_parameter("max_linear_x").value)
        )
        self.max_angular_z = abs(
            float(self.get_parameter("max_angular_z").value)
        )
        self.serial_command_rate = max(
            0.1, float(self.get_parameter("serial_command_rate").value)
        )

        self.serial_handle = None
        self.serial_buffer = ""
        self.last_open_attempt = 0.0
        self.last_open_log = 0.0
        self.latest_values = None
        self.has_unpublished_sensor_data = False
        self.odom_state = PlanarOdometry()
        self.command_state = VelocityCommandState(
            self.cmd_timeout,
            self.max_linear_x,
            self.max_angular_z,
        )

        self.temp_pub = self.create_publisher(Float32, "/temp", 10)
        self.hum_pub = self.create_publisher(Float32, "/hum", 10)
        self.mq2_pub = self.create_publisher(Float32, "/mq2", 10)
        self.pm25_pub = self.create_publisher(Float32, "/pm25", 10)
        self.pm10_pub = self.create_publisher(Float32, "/pm10", 10)
        self.hc_pub = self.create_publisher(Float32, "/hc", 10)
        self.odom_linear_pub = self.create_publisher(Float32, "/odom_linear", 10)
        self.odom_angular_pub = self.create_publisher(Float32, "/odom_angular", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.read_timer = self.create_timer(self.read_period, self.read_serial)
        self.sensor_timer = self.create_timer(
            1.0 / self.sensor_publish_rate, self.publish_sensor_values
        )
        self.cmd_vel_subscription = None
        self.command_timer = None
        if self.enable_cmd_vel:
            self.cmd_vel_subscription = self.create_subscription(
                Twist,
                self.cmd_vel_topic,
                self.cmd_vel_callback,
                10,
            )
            self.command_timer = self.create_timer(
                1.0 / self.serial_command_rate,
                self.write_command_timer,
            )

        self.open_serial()
        mode = "control enabled" if self.enable_cmd_vel else "read-only"
        self.get_logger().info(
            f"STM32 bridge started ({mode}): "
            f"port={self.serial_port}, baudrate={self.baudrate}, "
            f"read_period={self.read_period:.3f}s, "
            f"sensor_publish_rate={self.sensor_publish_rate:.1f}Hz"
        )

    def open_serial(self):
        now = time.monotonic()
        if now - self.last_open_attempt < 1.0:
            return
        self.last_open_attempt = now

        if serial is None:
            self._log_open_error("pyserial is not installed")
            return
        if self.serial_handle and self.serial_handle.is_open:
            return

        try:
            self.serial_handle = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.0,
            )
            self.serial_buffer = ""
            self.odom_state.reset_time()
            self.command_state.stop()
            self.get_logger().info(f"Opened STM32 serial port: {self.serial_port}")
        except (serial.SerialException, OSError) as exc:
            self.serial_handle = None
            self._log_open_error(str(exc))

    def _log_open_error(self, message):
        now = time.monotonic()
        if now - self.last_open_log >= 5.0:
            self.last_open_log = now
            self.get_logger().error(
                f"Failed to open {self.serial_port} at {self.baudrate}: {message}"
            )

    def read_serial(self):
        if self.serial_handle is None or not self.serial_handle.is_open:
            self.open_serial()
            return

        try:
            waiting = self.serial_handle.in_waiting
            if waiting <= 0:
                return
            chunk = self.serial_handle.read(waiting)
        except (serial.SerialException, OSError) as exc:
            self.get_logger().error(f"STM32 serial read failed: {exc}")
            self.close_serial()
            return

        if not chunk:
            return

        self.serial_buffer += chunk.decode("utf-8", errors="ignore")
        frames, self.serial_buffer, dropped_noise = extract_serial_frames(
            self.serial_buffer
        )
        if dropped_noise:
            self.get_logger().warn("Dropped noise before an STM32 data frame")

        if len(self.serial_buffer) > MAX_BUFFER_LEN:
            self.serial_buffer = _partial_prefix_suffix(self.serial_buffer)
            self.get_logger().warn("STM32 serial buffer was too long and was trimmed")

        latest_valid = None
        for frame in frames:
            try:
                latest_valid = parse_stm32_frame(frame)
            except ValueError as exc:
                self.get_logger().warn(f"Discarded invalid STM32 frame: {exc}")

        if latest_valid is not None:
            self.accept_values(latest_valid)

    def accept_values(self, values):
        self.latest_values = values
        self.has_unpublished_sensor_data = True

        now = self.get_clock().now()
        stamp_seconds = now.nanoseconds * 1e-9
        self.odom_state.update(
            values["odom_linear"],
            values["odom_angular"],
            stamp_seconds,
            self.max_integration_dt,
        )
        self.publish_odometry(now, values)

    def publish_sensor_values(self):
        if self.latest_values is None or not self.has_unpublished_sensor_data:
            return

        data = self.latest_values
        self.odom_linear_pub.publish(Float32(data=data["odom_linear"]))
        self.odom_angular_pub.publish(Float32(data=data["odom_angular"]))
        self.temp_pub.publish(Float32(data=data["temp"]))
        self.hum_pub.publish(Float32(data=data["hum"]))
        self.mq2_pub.publish(Float32(data=data["mq2"]))
        self.pm25_pub.publish(Float32(data=data["pm25"]))
        self.pm10_pub.publish(Float32(data=data["pm10"]))
        self.hc_pub.publish(Float32(data=data["hc"]))
        self.has_unpublished_sensor_data = False

    def emergency_stop_active(self):
        """Read the emergency stop parameter so runtime changes take effect."""
        return bool(self.get_parameter("emergency_stop").value)

    def cmd_vel_callback(self, message):
        """Validate and retain the latest velocity command without writing yet."""
        if not self.enable_cmd_vel:
            return
        if self.emergency_stop_active():
            self.command_state.stop()
            return

        accepted = self.command_state.accept(
            message.linear.x,
            message.angular.z,
            time.monotonic(),
        )
        if not accepted:
            self.get_logger().warn("Discarded non-finite cmd_vel command")

    def write_command_timer(self):
        """Write the active command or a safety stop at the configured rate."""
        if not self.enable_cmd_vel:
            return

        linear_x, angular_z = self.command_state.command_for(
            time.monotonic(),
            emergency_stop=self.emergency_stop_active(),
        )
        self.write_velocity_command(linear_x, angular_z)

    def write_velocity_command(self, linear_x, angular_z):
        """Write one command only when control is explicitly enabled."""
        if not self.enable_cmd_vel:
            return False
        if self.serial_handle is None or not self.serial_handle.is_open:
            return False

        payload = format_velocity_command(linear_x, angular_z).encode("ascii")
        try:
            written = self.serial_handle.write(payload)
        except (serial.SerialException, OSError) as exc:
            self.get_logger().error(f"STM32 serial write failed: {exc}")
            self.close_serial()
            return False

        if written != len(payload):
            self.get_logger().error(
                f"Incomplete STM32 serial write: {written}/{len(payload)} bytes"
            )
            self.close_serial()
            return False
        return True

    def publish_odometry(self, now, values):
        orientation = quaternion_from_yaw(self.odom_state.yaw)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.odom_state.x
        odom.pose.pose.position.y = self.odom_state.y
        odom.pose.pose.orientation = orientation
        odom.twist.twist.linear.x = values["odom_linear"]
        odom.twist.twist.angular.z = values["odom_angular"]
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is None:
            return

        transform = TransformStamped()
        transform.header.stamp = odom.header.stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = self.odom_state.x
        transform.transform.translation.y = self.odom_state.y
        transform.transform.rotation = orientation
        self.tf_broadcaster.sendTransform(transform)

    def close_serial(self):
        if self.serial_handle is None:
            return
        try:
            if self.serial_handle.is_open:
                self.serial_handle.close()
        except (serial.SerialException, OSError) as exc:
            self.get_logger().warn(f"Failed to close STM32 serial port: {exc}")
        finally:
            self.serial_handle = None
            self.serial_buffer = ""
            self.odom_state.reset_time()
            self.command_state.stop()

    def destroy_node(self):
        if self.enable_cmd_vel:
            self.write_velocity_command(0.0, 0.0)
            self.command_state.stop()
        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Stm32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("Read-only STM32 bridge stopped by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
