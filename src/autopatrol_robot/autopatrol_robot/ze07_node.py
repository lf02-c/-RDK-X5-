#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float32

class ZE07CO_Node(Node):
    def __init__(self):
        super().__init__('ze07_node') 

        self.ser = serial.Serial('/dev/ttyS1', 9600, timeout=0.1)
        self.buffer = []

        self.co_pub = self.create_publisher(Float32, '/co', 10)

        self.timer = self.create_timer(0.1, self.read_serial)

        self.get_logger().info("ZE07-CO Node Started")

    def read_serial(self):
        while self.ser.in_waiting > 0:
            byte = self.ser.read(1)
            if not byte:
                return

            byte = byte[0]

            print(f"{byte:02X}", end=" ")

            if len(self.buffer) == 0:
                if byte == 0xFF:
                    self.buffer.append(byte)
            else:
                self.buffer.append(byte)
                if len(self.buffer) == 9:
                    self.parse_frame(self.buffer)
                    self.buffer.clear()

    def parse_frame(self, frame):
        if len(frame) != 9 or frame[0] != 0xFF:
            return

        checksum = self.calc_checksum(frame)
        if checksum != frame[8]:
            self.get_logger().warn("Checksum error")
            return

        gas_type = frame[1]
        unit = frame[2]
        decimal = frame[3]
        high = frame[4]
        low = frame[5]
        full_scale = (frame[6] << 8) | frame[7]

        fault = (high >> 7) & 0x01

        value_raw = ((high & 0x1F) << 8) + low
        co = value_raw / 10.0

        if fault == 1:
            self.get_logger().error("ZE07 Sensor Fault!")
            return

        self.get_logger().info(f"CO: {co:.1f} ppm")

        msg = Float32()
        msg.data = co
        self.co_pub.publish(msg)

    def calc_checksum(self, frame):
        s = sum(frame[1:8])
        return ((~s + 1) & 0xFF)

    def destroy_node(self):
        self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ZE07CO_Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()