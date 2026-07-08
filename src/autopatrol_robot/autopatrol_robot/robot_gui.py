#!/usr/bin/env python3

import sys
import threading
import time
import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout,
    QTextEdit, QLabel
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QTextCharFormat
from PyQt5.QtCore import QTimer, pyqtSignal, Qt


class RobotGUI(Node, QWidget):

    log_signal = pyqtSignal(str)

    def __init__(self):

        Node.__init__(self, "robot_gui")
        QWidget.__init__(self)

        self.setWindowTitle("Robot Patrol GUI")
        self.resize(900, 600)

        layout = QHBoxLayout()  

        self.image_label = QLabel("Waiting for Camera...")
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setStyleSheet("background-color:black; color:white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.camera_label = self.image_label
        layout.addWidget(self.camera_label, 2) 

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: Consolas; font-size: 13px; background-color:#0c0c0c; color:white;"
        )
        layout.addWidget(self.log_text, 1)  

        self.setLayout(layout)

        self.bridge = CvBridge()
        self.frame = None
        self.frame_lock = threading.Lock()
        self.frame_id = 0
        self.displayed_frame_id = 0
        self.has_new_frame = False
        self.first_frame_logged = False
        self.display_count = 0
        self.last_perf_time = time.monotonic()
        self.last_debug_times = {}
        self.shutting_down = False

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            Image,
            "/camera/image",
            self.image_callback,
            image_qos
        )

        self.create_subscription(
            String,
            "/patrol_log",
            self.raw_log_callback,
            10
        )

        self.log_signal.connect(self.update_log_gui)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

        self.get_logger().info("Robot GUI Started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not self.first_frame_logged:
                self.first_frame_logged = True
                self.get_logger().info(
                    "[GUI_DEBUG] frame shape=%s, dtype=%s, min=%s, max=%s" %
                    (frame.shape, frame.dtype, frame.min(), frame.max())
                )
            with self.frame_lock:
                self.frame = frame
                self.frame_id += 1
                self.has_new_frame = True
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def raw_log_callback(self, msg):
        self.log_signal.emit(msg.data)

    def update_log_gui(self, text):
        if self.shutting_down:
            return

        cursor = self.log_text.textCursor()
        fmt = QTextCharFormat()

        if "PERSON DETECTED" in text:
            fmt.setForeground(QColor("red"))
            fmt.setFontWeight(75)
        elif "FIRE DETECTED" in text:
            fmt.setForeground(QColor("orange"))
            fmt.setFontWeight(75)
        elif "PATROL REPORT" in text:
            fmt.setForeground(QColor("cyan"))
            fmt.setFontWeight(75)
        elif "ACCIDENT" in text:
            fmt.setForeground(QColor("magenta"))  
            fmt.setFontWeight(75)

        cursor.insertText(text + "\n", fmt)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def debug_throttled(self, key, text, interval=2.0):
        now = time.monotonic()
        last_time = self.last_debug_times.get(key, 0.0)
        if now - last_time >= interval:
            self.get_logger().warn(text)
            self.last_debug_times[key] = now

    def update_gui(self):
        if self.shutting_down:
            return

        with self.frame_lock:
            if self.frame is None:
                self.debug_throttled("no_frame", "[GUI_DEBUG] no frame")
                return
            if not self.has_new_frame or self.frame_id == self.displayed_frame_id:
                return
            frame = self.frame
            frame_id = self.frame_id
            self.has_new_frame = False

        if frame.dtype != np.uint8 or len(frame.shape) != 3 or frame.shape[2] != 3:
            self.debug_throttled(
                "bad_frame",
                f"[GUI_DEBUG] bad frame shape={frame.shape}, dtype={frame.dtype}"
            )
            self.displayed_frame_id = frame_id
            return

        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            self.debug_throttled("label_size", "[GUI_DEBUG] label size is 0")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qimg = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        if qimg.isNull():
            self.debug_throttled("qimage_null", "[GUI_DEBUG] QImage is null")
            return

        qimg = qimg.copy()
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            self.debug_throttled("pixmap_null", "[GUI_DEBUG] pixmap is null")
            return

        pixmap = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        if pixmap.isNull():
            self.debug_throttled("scaled_pixmap_null", "[GUI_DEBUG] pixmap is null")
            return

        self.image_label.setPixmap(pixmap)
        self.displayed_frame_id = frame_id
        self.display_count += 1

        now = time.monotonic()
        elapsed = now - self.last_perf_time
        if elapsed >= 1.0:
            display_fps = self.display_count / elapsed
            self.get_logger().info(f"[GUI_PERF] display_fps={display_fps:.1f}")
            self.display_count = 0
            self.last_perf_time = now

    def closeEvent(self, event):
        self.shutting_down = True
        self.timer.stop()
        event.accept()


def ros_spin(node):
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass


def main():
    rclpy.init()

    app = QApplication(sys.argv)

    gui = RobotGUI()

    thread = threading.Thread(
        target=ros_spin,
        args=(gui,),
        daemon=True
    )
    thread.start()

    gui.show()

    exit_code = 0
    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        gui.shutting_down = True
        gui.timer.stop()
        gui.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2.0)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
