#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class AlarmNode(Node):

    def __init__(self):
        super().__init__('alarm_node')

        self.get_logger().info("Alarm node started")

        self.last_voice_time = {}
        self.voice_interval = 3.0

        self.sub = self.create_subscription(
            String,
            '/patrol_result',
            self.callback,
            10
        )

    def speak(self, text, category="default"):

        now = time.time()

        if category in self.last_voice_time:
            if now - self.last_voice_time[category] < self.voice_interval:
                return

        cmd = f'espeak -s 150 -p 45 -a 200 -v en-us "{text}"'
        os.system(cmd)

        self.last_voice_time[category] = now

    def callback(self, msg):

        text = msg.data
        self.get_logger().info(f"Receive: {text}")

        if "Person" in text:
            speak_text = "Warning. Person detected"
            category = "person"

        elif "Fire" in text:
            speak_text = "Warning. Fire detected"
            category = "fire"

        elif "Accident" in text or "accident" in text:
            speak_text = "Emergency. Accident detected"
            category = "accident"

        else:
            speak_text = text
            category = "default"

        self.speak(speak_text, category)


def main():

    rclpy.init()

    node = AlarmNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()