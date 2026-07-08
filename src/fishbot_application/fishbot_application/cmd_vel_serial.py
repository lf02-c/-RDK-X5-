#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import math


class CmdVelSerial(Node):
    def __init__(self):
        super().__init__('cmd_vel_serial')

        self.ser = serial.Serial('/dev/ttyS2', 115200, timeout=0.05)

        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        self.get_logger().info("cmd_vel_serial with rotation compensation started")

    def cmd_callback(self, msg: Twist):
        try:
            vx = msg.linear.x
            wz = msg.angular.z

            if abs(wz) < 0.15:
                wz_out = 0.0
            else:
                wz_out = wz


            line = f"{vx:.3f},0.000,{wz_out:.3f},0.000,0.000\n"
            self.ser.write(line.encode())

            # self.get_logger().info(f"cmd_vel in: wz={wz:.3f}, out: wz={wz_out:.3f}")

        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")


def main():
    rclpy.init()
    node = CmdVelSerial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, sending stop command...")
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        node.cmd_callback(stop_msg) 
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
