#!/usr/bin/env python3
"""Read STM32 sensor frames from serial and publish ROS2 Float32 topics."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    import serial
except ImportError:  # pragma: no cover - reported clearly at runtime
    serial = None


FRAME_PREFIX = "$DATA,"
FRAME_END = "#"
MAX_BUFFER_LEN = 4096


class SerialSensorNode(Node):
    """Parse STM32 '$DATA,...#' frames without sending any control command."""

    def __init__(self):
        super().__init__("serial_sensor_node")

        self.declare_parameter("serial_port", "/dev/ttyS2")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("publish_rate", 10.0)

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.publish_rate = max(0.1, float(self.get_parameter("publish_rate").value))

        self.buffer = ""
        self.serial_handle = None
        self.last_open_log_time = 0.0
        self.latest_values = None

        self.temp_pub = self.create_publisher(Float32, "/temp", 10)
        self.hum_pub = self.create_publisher(Float32, "/hum", 10)
        self.mq2_pub = self.create_publisher(Float32, "/mq2", 10)
        self.pm25_pub = self.create_publisher(Float32, "/pm25", 10)
        self.pm10_pub = self.create_publisher(Float32, "/pm10", 10)
        self.hc_pub = self.create_publisher(Float32, "/hc", 10)

        self.odom_linear_pub = self.create_publisher(Float32, "/odom_linear", 10)
        self.odom_angular_pub = self.create_publisher(Float32, "/odom_angular", 10)

        self.read_timer = self.create_timer(0.02, self.read_serial)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_latest)

        self.open_serial()
        self.get_logger().info(
            "Serial sensor node started: "
            f"port={self.serial_port}, baudrate={self.baudrate}, "
            f"publish_rate={self.publish_rate:.1f}Hz"
        )

    def open_serial(self):
        if serial is None:
            self.get_logger().error(
                "pyserial is not installed. Install python3-serial on the RDK."
            )
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
            self.get_logger().info(f"Opened STM32 sensor serial port: {self.serial_port}")
        except serial.SerialException as exc:
            self.serial_handle = None
            now = time.monotonic()
            if now - self.last_open_log_time > 5.0:
                self.last_open_log_time = now
                self.get_logger().error(
                    f"Failed to open {self.serial_port} at {self.baudrate}: {exc}"
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
        except serial.SerialException as exc:
            self.get_logger().error(f"Serial read error: {exc}")
            self.close_serial()
            return

        if not chunk:
            return

        self.buffer += chunk.decode("utf-8", errors="ignore")
        self.trim_buffer()
        self.parse_buffer()

    def trim_buffer(self):
        if len(self.buffer) <= MAX_BUFFER_LEN:
            return

        prefix_index = self.buffer.rfind(FRAME_PREFIX)
        if prefix_index >= 0:
            self.buffer = self.buffer[prefix_index:]
        else:
            self.buffer = ""
        self.get_logger().warn("Serial buffer was too long and has been trimmed")

    def parse_buffer(self):
        while True:
            start = self.buffer.find(FRAME_PREFIX)
            if start < 0:
                end = self.buffer.find(FRAME_END)
                if end >= 0:
                    self.get_logger().warn("Dropped serial data before invalid frame end")
                    self.buffer = self.buffer[end + 1:]
                    continue
                if len(self.buffer) > len(FRAME_PREFIX):
                    self.buffer = self.buffer[-len(FRAME_PREFIX):]
                return

            if start > 0:
                self.buffer = self.buffer[start:]

            end = self.buffer.find(FRAME_END, len(FRAME_PREFIX))
            if end < 0:
                return

            frame = self.buffer[:end + 1]
            self.buffer = self.buffer[end + 1:]
            self.parse_frame(frame)

    def parse_frame(self, frame):
        if not frame.startswith(FRAME_PREFIX) or not frame.endswith(FRAME_END):
            self.get_logger().warn(f"Discarded invalid STM32 frame: {frame!r}")
            return

        payload = frame[len(FRAME_PREFIX):-1]
        parts = payload.split(",")
        if len(parts) != 8:
            self.get_logger().warn(
                f"Discarded STM32 frame with {len(parts)} fields, expected 8: {frame!r}"
            )
            return

        try:
            values = {
                "odom_linear": float(parts[0]),
                "odom_angular": float(parts[1]),
                "temp": float(parts[2]),
                "hum": float(parts[3]),
                "mq2": float(parts[4]),
                "pm25": float(parts[5]),
                "pm10": float(parts[6]),
                "hc": float(parts[7]),
            }
        except ValueError as exc:
            self.get_logger().warn(
                f"Discarded STM32 frame with non-numeric field: {frame!r}, error={exc}"
            )
            return

        self.latest_values = values

    def publish_latest(self):
        if self.latest_values is None:
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

    def close_serial(self):
        if self.serial_handle is None:
            return
        try:
            if self.serial_handle.is_open:
                self.serial_handle.close()
        finally:
            self.serial_handle = None

    def destroy_node(self):
        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Serial sensor node stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
