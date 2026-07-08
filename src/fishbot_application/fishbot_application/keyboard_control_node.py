#!/usr/bin/env python3
"""Publish keyboard velocity commands without accessing robot hardware."""

import errno
import os
import select
import sys
import termios
import time
import tty

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


KEY_POLL_TIMEOUT_SECONDS = 0.1
INPUT_RETRY_DELAY_SECONDS = 0.1
IDLE_LOG_INTERVAL_SECONDS = 30.0


class KeyboardControlNode(Node):
    """Map keyboard input to Twist messages on a configurable topic."""

    def __init__(self):
        """Initialize keyboard command parameters and the Twist publisher."""
        super().__init__("keyboard_control_node")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("angular_speed", 1.2)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.linear_speed = abs(float(self.get_parameter("linear_speed").value))
        self.angular_speed = abs(float(self.get_parameter("angular_speed").value))

        self.publisher_ = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.vx = 0.0
        self.wz = 0.0

        # This node only publishes Twist; stm32_bridge_node owns hardware writes.
        self.get_logger().info(
            "Keyboard publisher ready: "
            f"topic={self.cmd_vel_topic}, linear={self.linear_speed:.3f}, "
            f"angular={self.angular_speed:.3f}"
        )

    def process_key(self, key):
        """Publish the velocity mapped to one recognized movement key."""
        normalized = key.lower()
        if normalized == "w":
            self.vx = self.linear_speed
            self.wz = 0.0
        elif normalized == "s":
            self.vx = -self.linear_speed
            self.wz = 0.0
        elif normalized == "a":
            self.vx = 0.0
            self.wz = self.angular_speed
        elif normalized == "d":
            self.vx = 0.0
            self.wz = -self.angular_speed
        elif normalized == "x" or key == " ":
            self.vx = 0.0
            self.wz = 0.0
        else:
            return False

        key_name = "SPACE" if key == " " else normalized.upper()
        self.get_logger().info(f"Received key: {key_name}")
        self.publish_velocity()
        return True

    def publish_velocity(self):
        """Publish the current velocity as one Twist message."""
        message = Twist()
        message.linear.x = self.vx
        message.angular.z = self.wz
        self.publisher_.publish(message)
        self.get_logger().info(
            f"Published cmd_vel: linear.x={self.vx:.3f}, angular.z={self.wz:.3f}"
        )

    def publish_stop(self):
        """Publish one zero velocity command."""
        self.vx = 0.0
        self.wz = 0.0
        self.publish_velocity()

    def publish_exit_stop(self):
        """Best-effort stop used by every shutdown path."""
        try:
            self.publish_stop()
        except Exception as exc:  # Context may already be shutting down.
            self.get_logger().error(f"Failed to publish shutdown stop: {exc}")

    def wait_for_key(self, input_fd):
        """Return one unbuffered key, or None while the terminal is idle."""
        try:
            # Reassert raw mode without flushing pending input. This keeps
            # single-key reads working if another terminal operation changed it.
            tty.setraw(input_fd, termios.TCSANOW)
            readable, _, _ = select.select(
                [input_fd],
                [],
                [],
                KEY_POLL_TIMEOUT_SECONDS,
            )
            if not readable:
                return None

            key_bytes = os.read(input_fd, 1)
            if not key_bytes:
                raise OSError(errno.EIO, "terminal input returned EOF")
            return key_bytes.decode("utf-8", errors="ignore")
        except OSError as exc:
            if exc.errno == errno.EINTR:
                return None
            raise

    def run(self):
        """Read keys from an interactive terminal until Q or Ctrl+C."""
        input_fd = sys.stdin.fileno()
        old_settings = None
        last_input_time = time.monotonic()
        last_input_error_log = 0.0
        try:
            if not os.isatty(input_fd):
                raise RuntimeError("keyboard control requires an interactive terminal")

            old_settings = termios.tcgetattr(input_fd)
            tty.setraw(input_fd, termios.TCSANOW)
            self.get_logger().info(
                "Keyboard control started. Use WASD, X/space to stop, Q to quit."
            )

            while rclpy.ok():
                try:
                    key = self.wait_for_key(input_fd)
                except OSError as exc:
                    now = time.monotonic()
                    if now - last_input_error_log >= IDLE_LOG_INTERVAL_SECONDS:
                        self.get_logger().warning(
                            f"Terminal input error; retrying: {exc}"
                        )
                        last_input_error_log = now
                    time.sleep(INPUT_RETRY_DELAY_SECONDS)
                    rclpy.spin_once(self, timeout_sec=0.01)
                    continue

                if key:
                    last_input_time = time.monotonic()
                    if key.lower() == "q" or key == "\x03":
                        self.get_logger().info(
                            "Keyboard control exit requested; stopping robot."
                        )
                        break
                    self.process_key(key)
                elif time.monotonic() - last_input_time >= IDLE_LOG_INTERVAL_SECONDS:
                    self.get_logger().info(
                        "Keyboard idle; node is still running and waiting for input."
                    )
                    last_input_time = time.monotonic()

                rclpy.spin_once(self, timeout_sec=0.01)
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().error(f"Keyboard runtime error: {exc}")
        finally:
            try:
                self.publish_exit_stop()
            finally:
                if old_settings is not None:
                    termios.tcsetattr(
                        input_fd,
                        termios.TCSADRAIN,
                        old_settings,
                    )


def main(args=None):
    """Run keyboard control until the user exits or ROS shuts down."""
    rclpy.init(args=args)
    node = KeyboardControlNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard control interrupted; stopping robot.")
        node.publish_exit_stop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
