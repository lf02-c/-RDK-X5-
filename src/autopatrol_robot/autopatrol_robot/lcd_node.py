#!/usr/bin/env python3
import time
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class HMISender(Node):
    ACTIVE_VALUES = {'true', '1', 'yes', 'detected'}
    INACTIVE_VALUES = {'false', '0', 'no', 'safe', 'none', 'clear'}
    ALERT_PRIORITY = ('fire', 'accident', 'person', 'congestion', 'water')
    FAILURE_LOG_INTERVAL = 10.0
    UNKNOWN_VALUE_LOG_INTERVAL = 10.0

    def __init__(self):
        super().__init__('hmi_sender_realtime_en')

        self.declare_parameter('esp32_ip', '10.86.10.24')
        self.declare_parameter('esp32_port', 80)
        self.declare_parameter('send_period', 0.5)
        self.declare_parameter('http_timeout', 0.3)

        self.esp32_ip = str(self.get_parameter('esp32_ip').value).strip()
        self.esp32_port = int(self.get_parameter('esp32_port').value)
        self.send_period = float(self.get_parameter('send_period').value)
        self.http_timeout = float(self.get_parameter('http_timeout').value)

        if not self.esp32_ip:
            raise ValueError('esp32_ip must not be empty')
        if not 1 <= self.esp32_port <= 65535:
            raise ValueError('esp32_port must be between 1 and 65535')
        if self.send_period <= 0.0:
            raise ValueError('send_period must be greater than zero')
        if self.http_timeout <= 0.0:
            raise ValueError('http_timeout must be greater than zero')

        self.url = f"http://{self.esp32_ip}:{self.esp32_port}/update"

        self.temp = None
        self.fire_alert = False
        self.accident_alert = False
        self.person_alert = False
        self.water_alert = False
        self.congestion_alert = False

        self._ever_connected = False
        self._connection_failed = False
        self._last_failure_log_time = 0.0
        self._last_unknown_log_times = {}

        self.create_subscription(Float32, '/temp', self.temp_cb, 10)
        self.create_subscription(
            String,
            '/fire_detected',
            lambda msg: self.alert_cb('fire', msg),
            10
        )
        self.create_subscription(
            String,
            '/accident_detected',
            lambda msg: self.alert_cb('accident', msg),
            10
        )
        self.create_subscription(
            String,
            '/person_detected',
            lambda msg: self.alert_cb('person', msg),
            10
        )
        self.create_subscription(
            String,
            '/water_detected',
            lambda msg: self.alert_cb('water', msg),
            10
        )
        self.create_subscription(
            String,
            '/congestion_detected',
            lambda msg: self.alert_cb('congestion', msg),
            10
        )

        self.timer = self.create_timer(self.send_period, self.send_to_esp32)

        self.get_logger().info(
            f"HMI sender started: url={self.url}, "
            f"send_period={self.send_period:.3f}s, "
            f"http_timeout={self.http_timeout:.3f}s"
        )

    def temp_cb(self, msg):
        self.temp = msg.data

    def alert_cb(self, alert_type, msg):
        value = (msg.data or '').strip().lower()
        if value in self.ACTIVE_VALUES or value == alert_type:
            active = True
        elif value in self.INACTIVE_VALUES:
            active = False
        else:
            self.log_unknown_alert_value(alert_type, value)
            return

        attribute = f'{alert_type}_alert'
        if getattr(self, attribute) == active:
            return

        setattr(self, attribute, active)
        state = 'active' if active else 'cleared'
        self.get_logger().info(
            f"{alert_type} alert {state}; "
            f"display status={self.get_current_road_status()}"
        )

    def log_unknown_alert_value(self, alert_type, value):
        now = time.monotonic()
        last_log_time = self._last_unknown_log_times.get(alert_type)
        if (
            last_log_time is not None
            and now - last_log_time < self.UNKNOWN_VALUE_LOG_INTERVAL
        ):
            return

        self._last_unknown_log_times[alert_type] = now
        self.get_logger().warn(
            f"Ignored unknown {alert_type} alert value: {value!r}"
        )

    def get_current_road_status(self):
        for alert_type in self.ALERT_PRIORITY:
            if getattr(self, f'{alert_type}_alert'):
                return alert_type
        return 'safe'

    def send_to_esp32(self):
        now = time.strftime("%H:%M:%S")

        if self.temp is None:
            temp_text = "--.-C"
        else:
            temp_text = f"{self.temp:.1f}C"

        safe_text = self.get_current_road_status()

        msg = (
            f"time={now}\n"
            f"temp={temp_text}\n"
            f"safe={safe_text}"
        )

        data = msg.encode("utf-8")

        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={"Content-Type": "text/plain"}
        )

        try:
            with urllib.request.urlopen(
                req,
                timeout=self.http_timeout
            ) as response:
                response.read()

            if not self._ever_connected:
                self.get_logger().info(
                    f"ESP32 connection established: {self.url}"
                )
            elif self._connection_failed:
                self.get_logger().info(
                    f"ESP32 connection recovered: {self.url}"
                )

            self._ever_connected = True
            self._connection_failed = False
        except Exception as e:
            now_monotonic = time.monotonic()
            if (
                not self._connection_failed
                or now_monotonic - self._last_failure_log_time
                >= self.FAILURE_LOG_INTERVAL
            ):
                self.get_logger().warn(f"ESP32 send failed: {e}")
                self._last_failure_log_time = now_monotonic
            self._connection_failed = True


def main():
    rclpy.init()
    node = HMISender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
