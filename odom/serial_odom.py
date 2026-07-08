#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

import serial
import math

from std_msgs.msg import Float32

def quat_from_yaw(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw * 0.5),
        w=math.cos(yaw * 0.5)
    )


class SerialOdom(Node):
    def __init__(self):
        super().__init__('serial_odom')

        self.ser = serial.Serial('/dev/ttyS2', 115200, timeout=0.05)

        self.temp_pub = self.create_publisher(Float32, '/temp', 10)
        self.hum_pub = self.create_publisher(Float32, '/hum', 10)
        self.mq2_pub = self.create_publisher(Float32, '/mq2', 10)
        self.pm25_pub = self.create_publisher(Float32, '/pm25', 10)
        self.pm10_pub = self.create_publisher(Float32, '/pm10', 10)
        self.hc_pub = self.create_publisher(Float32, '/hc', 10)

        # ===== ROS =====
        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===== ״̬ =====
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.v = 0.0
        self.w = 0.0

        self.last_time = self.get_clock().now()

        self.timer = self.create_timer(0.02, self.update)

    def update(self):
        line = self.ser.readline().decode(errors='ignore').strip()
        if not line:
            return
        #self.get_logger().info(f"{line}")
        if "$DATA" not in line:
            return

        try:
            line = line.replace("$DATA,", "").replace("#", "")
            data = line.split(',')
            

            if len(data) < 8:
                return

            #self.get_logger().info(f"{line}")

            v_real = float(data[0])
            w_real = float(data[1])

            self.temp = float(data[2])
            self.hum = float(data[3])
            self.mq2 = float(data[4])
            self.pm25 = float(data[5])
            self.pm10 = float(data[6])
            self.hc = float(data[7])

            self.temp_pub.publish(Float32(data=self.temp))
            self.hum_pub.publish(Float32(data=self.hum))
            self.mq2_pub.publish(Float32(data=self.mq2))
            self.pm25_pub.publish(Float32(data=self.pm25))
            self.pm10_pub.publish(Float32(data=self.pm10))
            self.hc_pub.publish(Float32(data=self.hc))

        except Exception:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9

        if dt <= 0.0 or dt > 0.1:
            self.last_time = now
            return

        self.last_time = now

        self.v = v_real
        self.w = w_real

        self.yaw = self.yaw + self.w * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        dx = self.v * math.cos(self.yaw) * dt
        dy = self.v * math.sin(self.yaw) * dt

        self.x += dx
        self.y += dy

        q = quat_from_yaw(self.yaw)

        # ===== Odometry =====
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        #self.get_logger().info(f"odom -> vx: {self.v:.3f}  wz: {self.w:.3f}  x:{self.x:.3f} y:{self.y:.3f} yaw:{self.yaw:.3f}")

        # ===== TF =====
        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = q

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = SerialOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
