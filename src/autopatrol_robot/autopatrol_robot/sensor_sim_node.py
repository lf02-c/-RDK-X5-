#!/usr/bin/env python3
"""
sensor_sim_node.py — 模拟 ROS2 传感器数据发布节点

用途：
    在不依赖真实硬件（摄像头、YOLO、雷达、串口传感器）的情况下，
    向 web_bridge_node 订阅的所有话题发布模拟数据，验证整条链路：

        sensor_sim_node → web_bridge_node → Socket.IO → index.html

发布的话题（与 web_bridge_node.py 订阅列表完全一致）：
    /camera/image   (sensor_msgs/Image)   10 Hz   — 渐变测试图 + 帧号/时间叠加
    /temp           (std_msgs/Float32)     2 Hz    — 正弦波 20~35°C
    /hum            (std_msgs/Float32)     2 Hz    — 正弦波 45~75%，间歇冲高触发湿度告警
    /mq2            (std_msgs/Float32)     2 Hz    — 基础值 100~200，间歇脉冲到 400+ 触发烟雾告警
    /co             (std_msgs/Float32)     2 Hz    — 基础值 0.3~0.8，间歇脉冲到 1.5+ 触发 CO 告警
    /pm25           (std_msgs/Float32)     2 Hz    — 随机游走 30~120
    /pm10           (std_msgs/Float32)     2 Hz    — 随机游走 50~200
    /hc             (std_msgs/Float32)     2 Hz    — 模拟水位距离，从 15→3→15 循环
    /patrol_log     (std_msgs/String)     ~0.2 Hz   — 轮流发布含 person/fire/water 等关键词的日志

告警触发设计：
    每个传感器都有"故事线"——从正常→预警→危险→恢复的周期变化，
    确保前端能看到传感器卡片变色、环境等级切换、报警列表新增条目。

patrol_log 格式：
    "level|text x:X, y:Y"
    - level: "normal" / "warning" / "danger"
    - text 中必须包含 person/fire/accident/water 之一（触发 alarm_event 的条件）
    - x: / y: 用于 extract_position 解析坐标
"""

import math
import time
import random

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge


class SensorSimNode(Node):
    """发布模拟传感器数据、测试图像和巡检日志的 ROS2 节点。"""

    def __init__(self):
        super().__init__('sensor_sim_node')

        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()

        # ---- 各传感器模拟用相位 / 状态 ----
        self._temp_phase = 0.0
        self._hum_phase = 0.0
        self._mq2_phase = 0.0
        self._co_phase = 0.0
        self._pm25_state = 50.0
        self._pm10_state = 100.0
        self._hc_base = 15.0
        self._log_index = 0

        # ================= 9 个 Publisher（与 web_bridge_node 订阅完全一致） =================
        self.camera_pub = self.create_publisher(Image, '/camera/image', 10)
        self.patrol_log_pub = self.create_publisher(String, '/patrol_log', 10)

        self.temp_pub = self.create_publisher(Float32, '/temp', 10)
        self.hum_pub = self.create_publisher(Float32, '/hum', 10)
        self.mq2_pub = self.create_publisher(Float32, '/mq2', 10)
        self.co_pub = self.create_publisher(Float32, '/co', 10)
        self.pm25_pub = self.create_publisher(Float32, '/pm25', 10)
        self.pm10_pub = self.create_publisher(Float32, '/pm10', 10)
        self.hc_pub = self.create_publisher(Float32, '/hc', 10)

        # 主定时器：10 Hz 驱动全部发布
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Sensor Sim Node Started — 模拟数据发布中")

    # ================================================================
    #  主回调
    # ================================================================

    def timer_callback(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        # -- 每 tick 发布测试图像 (10 Hz) --
        self._publish_test_image(elapsed)

        # -- 每 5 ticks 发布传感器数据 (2 Hz) --
        if self.frame_count % 5 == 0:
            self._publish_sensors(elapsed)

        # -- 每 50 ticks 发布 patrol_log (约 0.2 Hz，每 5 秒一条) --
        if self.frame_count % 50 == 0:
            self._publish_patrol_log(elapsed)

    # ================================================================
    #  测试图像生成
    # ================================================================

    def _generate_test_image(self, elapsed):
        """生成 640×480 BGR 渐变测试图，叠加标记文字。"""
        height, width = 480, 640

        # 彩色渐变背景（水平渐变 R，垂直渐变 G，对角线渐变 B）
        grad_r = np.tile(
            np.linspace(40, 200, width, dtype=np.uint8), (height, 1)
        )
        grad_g = np.tile(
            np.linspace(30, 180, height, dtype=np.uint8), (width, 1)
        ).T
        grad_b = np.full((height, width), 60, dtype=np.uint8)

        img = np.stack([grad_b, grad_g, grad_r], axis=2)

        # 网格线（便于观察图像是否卡顿）
        for x in range(0, width, 80):
            cv2.line(img, (x, 0), (x, height), (80, 80, 80), 1)
        for y in range(0, height, 60):
            cv2.line(img, (0, y), (width, y), (80, 80, 80), 1)

        # 中心十字
        cv2.line(img, (width // 2, 0), (width // 2, height), (100, 100, 100), 1)
        cv2.line(img, (0, height // 2), (width, height // 2), (100, 100, 100), 1)

        # 顶部信息栏
        cv2.putText(
            img, "SIMULATION — 模拟测试画面",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            img, f"Frame: {self.frame_count}  Time: {elapsed:.1f}s",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        # 底部提示
        cv2.putText(
            img, "ROS2 Sim -> web_bridge_node -> Socket.IO -> index.html",
            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )

        # 四个角的彩色方块（验证颜色通道正确）
        cv2.rectangle(img, (0, 0), (30, 30), (0, 0, 255), -1)        # 左上 红
        cv2.rectangle(img, (width - 30, 0), (width, 30), (0, 255, 0), -1)  # 右上 绿
        cv2.rectangle(img, (0, height - 30), (30, height), (255, 0, 0), -1)  # 左下 蓝
        cv2.rectangle(
            img, (width - 30, height - 30), (width, height),
            (0, 255, 255), -1  # 右下 黄
        )

        return img

    def _publish_test_image(self, elapsed):
        img = self._generate_test_image(elapsed)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "sim_camera"
            self.camera_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"publish image error: {e}")

    # ================================================================
    #  传感器模拟
    # ================================================================

    @staticmethod
    def _sine_wave(phase, period, lo, hi):
        """返回基于 phase 的正弦波值，范围 [lo, hi]."""
        mid = (lo + hi) / 2.0
        amp = (hi - lo) / 2.0
        return mid + amp * math.sin(phase * 2.0 * math.pi / period)

    def _publish_sensors(self, elapsed):
        t_cycle = elapsed % 120.0  # 120 秒一个大周期

        # ---- 温度：20~35°C，周期性波动 ----
        temp = self._sine_wave(t_cycle, 30.0, 22.0, 33.0)
        # 在 60~70 秒区间叠加升温，触发高温预警
        if 60.0 < t_cycle < 75.0:
            temp = 32.0 + (t_cycle - 60.0) * 0.8  # 最高到 ~43°C，触发 danger
        temp = float(np.clip(temp, 15.0, 50.0))

        # ---- 湿度：45~70%，间歇冲高触发告警 ----
        hum = self._sine_wave(t_cycle, 25.0, 48.0, 68.0)
        if 80.0 < t_cycle < 95.0:
            hum = 65.0 + (t_cycle - 80.0) * 1.2  # 最高 ~83%，触发 danger
        hum = float(np.clip(hum, 30.0, 95.0))

        # ---- MQ2：基础 100~200，周期性脉冲 ----
        mq2 = self._sine_wave(t_cycle, 20.0, 120.0, 220.0)
        if 40.0 < t_cycle < 50.0:
            mq2 = 250.0 + (t_cycle - 40.0) * 30.0  # 最高 ~550，触发 danger
        mq2 = float(np.clip(mq2, 50.0, 600.0))

        # ---- CO：基础 0.3~0.8，间歇脉冲 ----
        co = self._sine_wave(t_cycle * 0.7, 18.0, 0.35, 0.75)
        if 100.0 < t_cycle < 110.0:
            co = 1.2 + (t_cycle - 100.0) * 0.12  # 最高 ~2.4，触发 danger
        co = float(np.clip(co, 0.1, 3.0))

        # ---- PM2.5：随机游走 ----
        self._pm25_state += random.gauss(0, 3.0)
        if 20.0 < t_cycle < 35.0:
            self._pm25_state += 5.0  # 冲高区间
        self._pm25_state = float(np.clip(self._pm25_state, 15.0, 200.0))

        # ---- PM10：随机游走 ----
        self._pm10_state += random.gauss(0, 5.0)
        if 20.0 < t_cycle < 35.0:
            self._pm10_state += 8.0
        self._pm10_state = float(np.clip(self._pm10_state, 30.0, 400.0))

        # ---- HC：模拟水位距离 15→3→15 循环（< 5 danger，< 8 warning） ----
        hc_phase = t_cycle % 80.0
        if hc_phase < 50.0:
            hc = 15.0 - hc_phase * 0.24  # 缓慢下降到 3.0
        else:
            hc = 3.0 + (hc_phase - 50.0) * 0.4  # 回升到 ~15
        hc = float(np.clip(hc, 2.0, 35.0))

        # ---- 发布 ----
        self.temp_pub.publish(Float32(data=temp))
        self.hum_pub.publish(Float32(data=hum))
        self.mq2_pub.publish(Float32(data=mq2))
        self.co_pub.publish(Float32(data=co))
        self.pm25_pub.publish(Float32(data=self._pm25_state))
        self.pm10_pub.publish(Float32(data=self._pm10_state))
        self.hc_pub.publish(Float32(data=hc))

        # 每 30 秒打印一次传感器摘要，方便终端观察
        if self.frame_count % 300 == 0:
            self.get_logger().info(
                f"SENSORS | T={temp:.1f}C H={hum:.1f}% "
                f"MQ2={mq2:.1f} CO={co:.2f} "
                f"PM2.5={self._pm25_state:.1f} PM10={self._pm10_state:.1f} "
                f"HC={hc:.1f}"
            )

    # ================================================================
    #  patrol_log 模拟（严格匹配 web_bridge_node.log_cb 解析逻辑）
    # ================================================================

    def _publish_patrol_log(self, elapsed):
        """
        log_cb 的解析规则（从 web_bridge_node.py 提取）：

            if "|" in data:
                level, text = data.split("|", 1)   # "danger|xxx" → level=danger, text=xxx
            else:
                text = data                          # 无 | 则 level=normal

        build_alarm_event 的触发规则：
            text.lower() 包含 "person" / "fire" / "accident" / "water" → 生成 alarm_event
            不包含以上关键词 → alarm_event = None，不写入报警日志

        extract_position 的解析规则：
            text 中需要包含 "x:X, y:Y" 格式才能解析出坐标

        下面的 patrol_log 模板确保：
            1. 包含 "|" 分隔符，指定 level
            2. text 中包含 alarm 关键词（触发报警）
            3. text 中包含 "x:X, y:Y"（生成有意义的位置）
        """

        # 轮流发布不同报警类型，覆盖 person / fire / water / accident
        log_templates = [
            # (level, text) — text 中 x: / y: 用于 extract_position 解析坐标
            (
                "danger",
                "person detected in tunnel area, x:3.20, y:1.50 confidence=0.92"
            ),
            (
                "warning",
                "fire suspected near equipment cabinet, x:7.80, y:2.10 temp_rise"
            ),
            (
                "danger",
                "water accumulation at low point, x:5.40, y:0.80 depth_warning"
            ),
            (
                "warning",
                "person entering restricted zone, x:9.10, y:1.30 intrusion_alert"
            ),
            (
                "danger",
                "accident vehicle collision detected, x:4.60, y:2.80 multi_car"
            ),
            (
                "warning",
                "fire smoke rising from ventilation shaft, x:6.30, y:1.90 low_visibility"
            ),
            (
                "danger",
                "water level exceeding threshold at drain, x:2.10, y:3.40 flood_risk"
            ),
            # 故意加入一条不含报警关键词的日志，验证不会生成 alarm_event
            (
                "normal",
                "patrol routine check passed, x:1.00, y:5.00 all_clear"
            ),
        ]

        level, text = log_templates[self._log_index % len(log_templates)]
        self._log_index += 1

        # 拼成 log_cb 能解析的格式
        msg_data = f"{level}|{text}"

        msg = String()
        msg.data = msg_data
        self.patrol_log_pub.publish(msg)

        # 预告下一条日志的内容
        next_level, next_text = log_templates[self._log_index % len(log_templates)]
        has_alarm = any(
            kw in next_text.lower() for kw in ("person", "fire", "accident", "water")
        )
        self.get_logger().info(
            f"PATROL_LOG [{level}] {text[:50]}... "
            f"→ next: [{next_level}] alarm={has_alarm}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SensorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Sensor Sim Node stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
