#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
from time import time, perf_counter
import logging
import math
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

CLASS_CONF = {
    "car": 0.12,
    "accident": 0.60,
    "fire": 0.35,
    "person": 0.18,
    "water": 0.18,
}
RAW_SCORE_THRESHOLD = 0.08
DEFAULT_CLASS_CONF = 0.25
VALID_CLASSES = ["car", "accident", "fire", "person", "water"]

ENABLE_OPENCV_PREPROCESS = True
ENABLE_GAMMA_CORRECTION = True
ENABLE_HIGHLIGHT_SUPPRESS = True
ENABLE_GAUSSIAN_BLUR = False
ENABLE_CLAHE = False

ENABLE_CONGESTION_DETECT = True
TRACK_HISTORY_LEN = 30
TRACK_MATCH_DISTANCE = 60
MAX_MISSING_FRAMES = 8
CONGESTION_WINDOW = 20
MOVE_PIXEL_THRES = 10
CONGESTION_CONFIRM_FRAMES = 10
MIN_STATIONARY_CARS = 1

_GAMMA_LUT_CACHE = {}
_CLAHE = None

PERF_WINDOW = 60
PERF_KEYS = [
    "camera_read_ms",
    "opencv_preprocess_ms",
    "model_preprocess_ms",
    "bpu_forward_ms",
    "c2numpy_ms",
    "postprocess_ms",
    "draw_ms",
    "ros_msg_ms",
    "publish_ms",
    "total_ms",
]
PERF_LABELS = {
    "camera_read_ms": "camera",
    "opencv_preprocess_ms": "opencv",
    "model_preprocess_ms": "preprocess",
    "bpu_forward_ms": "bpu",
    "c2numpy_ms": "c2numpy",
    "postprocess_ms": "post",
    "draw_ms": "draw",
    "ros_msg_ms": "ros_msg",
    "publish_ms": "publish",
    "total_ms": "total",
}
DEBUG_PANEL_WIDTH = 426


def gamma_correction(img, gamma=1.0):
    gamma = max(gamma, 0.1)
    key = round(gamma, 3)
    table = _GAMMA_LUT_CACHE.get(key)
    if table is None:
        values = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((values ** key) * 255.0, 0, 255).astype("uint8")
        _GAMMA_LUT_CACHE[key] = table
    return cv2.LUT(img, table)


def clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = _CLAHE.apply(y)
    enhanced = cv2.merge([y, cr, cb])
    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


def suppress_highlight(img, threshold=235, compress_ratio=0.35):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright_mask = v > threshold

    v_new = v.astype(np.float32)
    v_new[bright_mask] = threshold + (v_new[bright_mask] - threshold) * compress_ratio
    v_new = np.clip(v_new, 0, 255).astype(np.uint8)

    hsv_new = cv2.merge([h, s, v_new])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


def light_denoise(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def get_brightness_info(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_count = gray.size
    mean_brightness = float(gray.mean())
    dark_ratio = np.count_nonzero(gray < 45) / pixel_count
    bright_ratio = np.count_nonzero(gray > 235) / pixel_count
    return mean_brightness, dark_ratio, bright_ratio


def preprocess_for_yolo(img):
    if not ENABLE_OPENCV_PREPROCESS:
        return img

    mean_brightness, dark_ratio, bright_ratio = get_brightness_info(img)
    result = img

    if ENABLE_HIGHLIGHT_SUPPRESS and bright_ratio > 0.03:
        result = suppress_highlight(result, threshold=235, compress_ratio=0.35)

    if ENABLE_GAMMA_CORRECTION and (mean_brightness < 80 or dark_ratio > 0.35):
        result = gamma_correction(result, gamma=0.75)

    if ENABLE_CLAHE:
        result = clahe_enhance(result, clip_limit=1.8)

    if ENABLE_GAUSSIAN_BLUR:
        result = light_denoise(result)

    return result


# ============================================================
# ======================= ROS2 ===========================
# ============================================================

class DoorDetectNode(Node):
    def __init__(self):
        super().__init__('door_detect_node_debug_compare')

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image', 10)

        self.accident_pub = self.create_publisher(String, '/accident_detected', 10)
        self.person_pub = self.create_publisher(String, '/person_detected', 10)
        self.fire_pub = self.create_publisher(String, '/fire_detected', 10)

        class Opt:
            model_path = '/home/sunrise/yolov8s_5class_768_modified.bin'  # ���޸�Ϊʵ��·��
            nms_thres = 0.7
            raw_score_thres = RAW_SCORE_THRESHOLD
            reg = 16
        self.opt = Opt()

        self.accident_counter = 0
        self.ACCIDENT_THRESHOLD = 5 


        self.model = YOLOv8_Detect(self.opt)

        self.declare_parameter('video_path', '')
        self.declare_parameter('output_video_path', '')
        self.declare_parameter('loop_video', True)
        video_path = self.get_parameter('video_path').value or ''
        self.output_video_path = self.get_parameter('output_video_path').value or ''
        self.loop_video = self._to_bool(self.get_parameter('loop_video').value)
        self.use_video_file = bool(video_path)
        self.stop_requested = False
        self.video_writer = None
        self.video_writer_failed = False

        if self.use_video_file:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            camera_fourcc = getattr(cv2, "VideoWriter_fourcc")('M', 'J', 'P', 'G')
            self.cap.set(cv2.CAP_PROP_FOURCC, camera_fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            if self.use_video_file:
                self.get_logger().error(f"Cannot open video file: {video_path}")
                raise RuntimeError(f"Video open failed: {video_path}")
            self.get_logger().error("Cannot open camera")
            raise RuntimeError("Camera open failed")

        output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not output_fps or output_fps <= 1:
            output_fps = 25
        self.output_video_fps = output_fps

        self.frame_count = 0
        self.fps_list = []
        self.next_track_id = 1
        self.car_tracks = {}
        self.perf_samples = {key: [] for key in PERF_KEYS}

        self.timer = self.create_timer(0.001, self.timer_callback)
        if self.use_video_file:
            self.get_logger().info(f"Using video file input: {video_path}")
        else:
            self.get_logger().info("Using camera input: 0")
        self.get_logger().info(f"loop_video={self.loop_video}")
        if self.output_video_path:
            self.get_logger().info(
                f"Output compare video enabled: {self.output_video_path}, "
                f"fps={self.output_video_fps:.2f}"
            )
        self.get_logger().info("Door Detect Debug Compare Node Started")

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _read_frame(self):
        ret, frame = self.cap.read()
        if not ret and self.use_video_file:
            if self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self.stop_requested = True
        return ret, frame

    def _record_perf(self, timings):
        for key in PERF_KEYS:
            self.perf_samples[key].append(timings.get(key, 0.0))

        if len(self.perf_samples["total_ms"]) < PERF_WINDOW:
            return

        avg = {
            key: sum(values) / len(values)
            for key, values in self.perf_samples.items()
        }
        max_values = {
            key: max(values)
            for key, values in self.perf_samples.items()
        }
        fps = 1000.0 / avg["total_ms"] if avg["total_ms"] > 0 else 0.0

        avg_parts = [
            f"{PERF_LABELS[key]}={avg[key]:.2f} ms"
            for key in PERF_KEYS
        ]
        max_parts = [
            f"{PERF_LABELS[key]}={max_values[key]:.2f} ms"
            for key in PERF_KEYS
        ]

        self.get_logger().info(
            f"[PERF] avg over {PERF_WINDOW} frames: "
            + ", ".join(avg_parts)
            + f", fps={fps:.2f}"
        )
        self.get_logger().info(
            f"[PERF_MAX] over {PERF_WINDOW} frames: "
            + ", ".join(max_parts)
        )

        self.perf_samples = {key: [] for key in PERF_KEYS}

    def _resize_panel(self, img):
        height, width = img.shape[:2]
        if width <= 0:
            return img
        scale = DEBUG_PANEL_WIDTH / float(width)
        panel_height = max(1, int(height * scale))
        return cv2.resize(img, (DEBUG_PANEL_WIDTH, panel_height))

    def _draw_panel_title(self, img, title):
        cv2.putText(img, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _init_video_writer(self, frame):
        if not self.output_video_path or self.video_writer or self.video_writer_failed:
            return

        try:
            height, width = frame.shape[:2]
            output_dir = os.path.dirname(self.output_video_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            ext = os.path.splitext(self.output_video_path)[1].lower()
            if ext == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                self.output_video_fps,
                (width, height)
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to initialize output compare video writer: "
                f"{self.output_video_path}, error={e}"
            )
            self.video_writer = None
            self.video_writer_failed = True
            return

        if not self.video_writer.isOpened():
            self.get_logger().error(
                f"Cannot open output compare video writer: {self.output_video_path}, "
                f"size={width}x{height}, fps={self.output_video_fps:.2f}"
            )
            self.video_writer.release()
            self.video_writer = None
            self.video_writer_failed = True
            return

        self.get_logger().info(
            f"Saving compare video to: {self.output_video_path}, "
            f"size={width}x{height}, fps={self.output_video_fps:.2f}"
        )

    def _match_car_track(self, center, used_track_ids):
        best_track_id = None
        best_dist = TRACK_MATCH_DISTANCE

        for track_id, track in self.car_tracks.items():
            if track_id in used_track_ids:
                continue
            if not track["points"]:
                continue
            last_center = track["points"][-1]
            dist = math.hypot(center[0] - last_center[0], center[1] - last_center[1])
            if dist < best_dist:
                best_dist = dist
                best_track_id = track_id

        return best_track_id

    def update_car_tracks(self, car_detections):
        tracked_cars = []
        used_track_ids = set()

        for score, x1, y1, x2, y2 in car_detections:
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_id = self._match_car_track(center, used_track_ids)

            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.car_tracks[track_id] = {
                    "track_id": track_id,
                    "points": [],
                    "missing_count": 0,
                    "stationary_count": 0,
                    "bbox": None,
                    "score": 0.0,
                }
            else:
                used_track_ids.add(track_id)

            track = self.car_tracks[track_id]
            track["points"].append(center)
            if len(track["points"]) > TRACK_HISTORY_LEN:
                track["points"].pop(0)
            track["missing_count"] = 0
            track["bbox"] = (x1, y1, x2, y2)
            track["score"] = score

            is_stationary = False
            if ENABLE_CONGESTION_DETECT and len(track["points"]) >= CONGESTION_WINDOW:
                old_center = track["points"][-CONGESTION_WINDOW]
                move_dist = math.hypot(center[0] - old_center[0], center[1] - old_center[1])
                if move_dist < MOVE_PIXEL_THRES:
                    track["stationary_count"] += 1
                else:
                    track["stationary_count"] = max(0, track["stationary_count"] - 1)

                is_stationary = track["stationary_count"] >= CONGESTION_CONFIRM_FRAMES

            tracked_cars.append((track_id, score, x1, y1, x2, y2, is_stationary))
            used_track_ids.add(track_id)

        for track_id in list(self.car_tracks.keys()):
            if track_id in used_track_ids:
                continue
            self.car_tracks[track_id]["missing_count"] += 1
            if self.car_tracks[track_id]["missing_count"] > MAX_MISSING_FRAMES:
                del self.car_tracks[track_id]

        if ENABLE_CONGESTION_DETECT:
            stationary_count = sum(
                1
                for track in self.car_tracks.values()
                if track["missing_count"] == 0 and track["stationary_count"] >= CONGESTION_CONFIRM_FRAMES
            )
            congestion = stationary_count >= MIN_STATIONARY_CARS
        else:
            congestion = False
        return tracked_cars, congestion

    def timer_callback(self):
        total_start = perf_counter()
        timings = {key: 0.0 for key in PERF_KEYS}

        stage_start = perf_counter()
        ret, frame = self._read_frame()
        timings["camera_read_ms"] = (perf_counter() - stage_start) * 1000.0
        if not ret:
            if self.stop_requested:
                self.get_logger().info("Video ended and loop_video=False, stopping debug compare node")
                self.timer.cancel()
                if rclpy.ok():
                    rclpy.shutdown()
                return
            self.get_logger().warn("Failed to read frame")
            return

        self.frame_count += 1
        start_time = time()

        try:
            raw_frame = frame.copy()

            stage_start = perf_counter()
            preprocessed_frame = preprocess_for_yolo(frame)
            preprocess_view = preprocessed_frame.copy()
            detect_view = preprocessed_frame.copy()
            timings["opencv_preprocess_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            input_tensor = self.model.preprocess_yuv420sp(preprocessed_frame)
            timings["model_preprocess_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            dnn_outputs = self.model.forward(input_tensor)
            timings["bpu_forward_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            outputs = self.model.c2numpy(dnn_outputs)
            timings["c2numpy_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            results = self.model.postProcess(outputs)
            timings["postprocess_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()

            person_detected = False
            fire_detected = False
            accident_in_frame = False
            car_detections = []
            display_count = 0

            for class_id, score, x1, y1, x2, y2 in results:
                class_name = self.model.class_names[class_id]

                if class_name not in VALID_CLASSES:
                    continue

                threshold = CLASS_CONF.get(class_name, DEFAULT_CLASS_CONF)
                if score < threshold:
                    continue

                if class_name == "car":
                    car_detections.append((score, x1, y1, x2, y2))
                    continue

                draw_detection(detect_view, (x1, y1, x2, y2), score, class_name)
                display_count += 1

                if class_name == "accident":
                    accident_in_frame = True
                
                if class_name == "person":
                    person_detected = True
                elif class_name == "fire":
                    fire_detected = True

                        # ===== accident ����֡���� =====
            tracked_cars, congestion = self.update_car_tracks(car_detections)
            display_count += len(tracked_cars)
            for track_id, score, x1, y1, x2, y2, _ in tracked_cars:
                label = f"car ID:{track_id} {score:.2f}"
                draw_detection(detect_view, (x1, y1, x2, y2), score, "car", label=label)

            if congestion:
                cv2.putText(detect_view, "CONGESTION", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if accident_in_frame:
                self.accident_counter += 1
            else:
                self.accident_counter = 0

            # ����ͼ��
            if person_detected and (not hasattr(self, "last_person_time") or time() - self.last_person_time > 2):
                person_msg = String()
                person_msg.data = "person"
                self.person_pub.publish(person_msg)
                self.last_person_time = time()

            if fire_detected and (not hasattr(self, "last_fire_time") or time() - self.last_fire_time > 2):
                fire_msg = String()
                fire_msg.data = "fire"
                self.fire_pub.publish(fire_msg)
                self.last_fire_time = time()

            if self.accident_counter >= self.ACCIDENT_THRESHOLD:
                if not hasattr(self, "last_accident_time") or time() - self.last_accident_time > 2:
                    accident_msg = String()
                    accident_msg.data = "accident"
                    self.accident_pub.publish(accident_msg)
                    self.last_accident_time = time()

                self.accident_counter = 0

            # FPS ��ʾ
            inference_time = time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_list.append(fps)
            if len(self.fps_list) > 30:
                self.fps_list.pop(0)
            avg_fps = sum(self.fps_list) / len(self.fps_list)

            cv2.putText(detect_view, f"FPS: {avg_fps:.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(detect_view, f"Detections: {display_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            timings["draw_ms"] = (perf_counter() - stage_start) * 1000.0

            raw_panel = self._resize_panel(raw_frame)
            preprocess_panel = self._resize_panel(preprocess_view)
            detect_panel = self._resize_panel(detect_view)
            self._draw_panel_title(raw_panel, "RAW")
            self._draw_panel_title(preprocess_panel, "OPENCV PREPROCESS")
            self._draw_panel_title(detect_panel, "DETECTION RESULT")
            compare_frame = np.hstack([raw_panel, preprocess_panel, detect_panel])

            if self.output_video_path:
                self._init_video_writer(compare_frame)
                if self.video_writer:
                    try:
                        self.video_writer.write(compare_frame)
                    except Exception as e:
                        self.get_logger().error(
                            f"Failed to write compare video frame: {self.output_video_path}, error={e}"
                        )
                        self.video_writer.release()
                        self.video_writer = None
                        self.video_writer_failed = True

            stage_start = perf_counter()
            img_msg = self.bridge.cv2_to_imgmsg(compare_frame, "bgr8")
            timings["ros_msg_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            self.image_pub.publish(img_msg)
            timings["publish_ms"] = (perf_counter() - stage_start) * 1000.0
            timings["total_ms"] = (perf_counter() - total_start) * 1000.0
            self._record_perf(timings)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def destroy_node(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.cap.release()
        super().destroy_node()


# ============================================================
# ======================= YOLO ==========================
# ============================================================

class YOLOv8_Detect:
    def __init__(self, opt):
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.info(f"Model loaded in {1000 * (time() - begin_time):.2f} ms")
        except Exception as e:
            logger.error(f"Failed to load model file: {opt.model_path}")
            logger.error(e)
            exit(1)

        logger.info("-> output tensors")
        for i, out in enumerate(self.quantize_model[0].outputs):
            shape = out.properties.shape
            logger.info(f"output[{i}], name={out.name}, type={out.properties.dtype}, shape={shape}")

        self.bbox_out_idx = [0, 1, 2]
        self.cls_out_idx = [3, 4, 5]

        self.s_bboxes_scale = self.quantize_model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]

        actual_classes = self.quantize_model[0].outputs[3].properties.shape[-1]
        self.CLASSES_NUM = actual_classes
        logger.info(f"Detected number of classes: {self.CLASSES_NUM}")

        full_names = ["car", "accident", "fire", "person", "water"]
        if self.CLASSES_NUM <= len(full_names):
            self.class_names = full_names[:self.CLASSES_NUM]
        else:
            self.class_names = [f"class_{i}" for i in range(self.CLASSES_NUM)]
        logger.info(f"Using class names: {self.class_names}")

        self.weights_static = np.arange(16, dtype=np.float32)[np.newaxis, np.newaxis, :]

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"Input size: {self.input_H} x {self.input_W}")

        self.s_anchor = self._make_anchor(self.input_W // 8, self.input_H // 8)

        self.m_anchor = self._make_anchor(self.input_W // 16, self.input_H // 16)

        self.l_anchor = self._make_anchor(self.input_W // 32, self.input_H // 32)

        self.RAW_SCORE_THRESHOLD = getattr(opt, "raw_score_thres", RAW_SCORE_THRESHOLD)
        self.NMS_THRESHOLD = opt.nms_thres
        self.REG = opt.reg


        self.ACCIDENT_SCORE_THRESH = self.RAW_SCORE_THRESHOLD
        self.MAX_ACCIDENT_AREA_RATIO = 0.8
        self.MIN_ACCIDENT_AREA_RATIO = 0.005

    def _make_anchor(self, feat_w, feat_h):
        return np.stack([
            np.tile(np.linspace(0.5, feat_w - 0.5, feat_w), feat_h),
            np.repeat(np.arange(0.5, feat_h + 0.5, 1), feat_w)
        ], axis=0).transpose(1, 0)

    def preprocess_yuv420sp(self, img):
        self.img_h, self.img_w = img.shape[:2]

        self.x_scale = min(self.input_W / self.img_w, self.input_H / self.img_h)
        self.y_scale = self.x_scale

        new_w = int(self.img_w * self.x_scale)
        new_h = int(self.img_h * self.y_scale)

        self.x_shift = (self.input_W - new_w) // 2
        self.y_shift = (self.input_H - new_h) // 2
        x_other = self.input_W - new_w - self.x_shift
        y_other = self.input_H - new_h - self.y_shift

        resized = cv2.resize(img, (new_w, new_h))
        padded = cv2.copyMakeBorder(resized, self.y_shift, y_other, self.x_shift, x_other,
                                    cv2.BORDER_CONSTANT, value=[127, 127, 127])
        return self.bgr2nv12(padded)

    def bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[:2]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor):
        return self.quantize_model[0].forward(input_tensor)

    def c2numpy(self, outputs):
        return [out.buffer for out in outputs]

    def postProcess(self, outputs):
        results = []
        strides = [8, 16, 32]
        anchors = [self.s_anchor, self.m_anchor, self.l_anchor]
        bbox_scales = [self.s_bboxes_scale, self.m_bboxes_scale, self.l_bboxes_scale]

        all_boxes = []
        all_scores = []
        all_ids = []

        for i in range(3):
            bbox = outputs[self.bbox_out_idx[i]].reshape(-1, self.REG * 4) * bbox_scales[i]
            cls = outputs[self.cls_out_idx[i]].reshape(-1, self.CLASSES_NUM)

            scores = 1 / (1 + np.exp(-cls))
            class_ids = np.argmax(scores, axis=1)
            class_scores = np.max(scores, axis=1)

            mask = class_scores > self.RAW_SCORE_THRESHOLD
            bbox = bbox[mask]
            class_ids = class_ids[mask]
            class_scores = class_scores[mask]

            if len(bbox) == 0:
                continue

            bbox = bbox.reshape(-1, 4, self.REG)
            bbox = softmax(bbox, axis=2)
            bbox = np.sum(bbox * self.weights_static, axis=2)

            anchor = anchors[i][mask]

            x1 = (anchor[:, 0] - bbox[:, 0]) * strides[i]
            y1 = (anchor[:, 1] - bbox[:, 1]) * strides[i]
            x2 = (anchor[:, 0] + bbox[:, 2]) * strides[i]
            y2 = (anchor[:, 1] + bbox[:, 3]) * strides[i]

            x1 = (x1 - self.x_shift) / self.x_scale
            y1 = (y1 - self.y_shift) / self.y_scale
            x2 = (x2 - self.x_shift) / self.x_scale
            y2 = (y2 - self.y_shift) / self.y_scale

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            all_boxes.append(boxes)
            all_scores.append(class_scores)
            all_ids.append(class_ids)

        if not all_boxes:
            return results

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        temp_results = []
        for class_id in range(self.CLASSES_NUM):
            mask = ids == class_id
            if not np.any(mask):
                continue

            boxes_cls = boxes[mask]
            scores_cls = scores[mask]

            if self.class_names[class_id] == "accident":
                score_thresh = self.ACCIDENT_SCORE_THRESH
                nms_thresh = self.NMS_THRESHOLD
            else:
                score_thresh = self.RAW_SCORE_THRESHOLD
                nms_thresh = self.NMS_THRESHOLD

            keep = scores_cls >= score_thresh
            boxes_cls = boxes_cls[keep]
            scores_cls = scores_cls[keep]

            if len(boxes_cls) == 0:
                continue

            wh = boxes_cls[:, 2:4] - boxes_cls[:, 0:2]
            valid_box = (wh[:, 0] > 0) & (wh[:, 1] > 0)
            boxes_cls = boxes_cls[valid_box]
            scores_cls = scores_cls[valid_box]
            wh = wh[valid_box]

            if len(boxes_cls) == 0:
                continue

            nms_boxes = np.concatenate([boxes_cls[:, 0:2], wh], axis=1)

            indices = cv2.dnn.NMSBoxes(
                nms_boxes.tolist(),
                scores_cls.tolist(),
                score_thresh,
                nms_thresh
            )
            if len(indices) == 0:
                continue

            for idx in indices.flatten():
                x1, y1, x2, y2 = boxes_cls[idx]
                score = scores_cls[idx]
                temp_results.append((class_id, score, x1, y1, x2, y2))

        final_results = []
        for res in temp_results:
            class_id, score, x1, y1, x2, y2 = res
            if self.class_names[class_id] == "accident" and self.CLASSES_NUM > 4:
                area = (x2-x1)*(y2-y1)
                img_area = self.img_w * self.img_h
                area_ratio = area / img_area
                if area_ratio > self.MAX_ACCIDENT_AREA_RATIO or area_ratio < self.MIN_ACCIDENT_AREA_RATIO:
                    continue
                width = x2 - x1
                height = y2 - y1
                if width > 0 and height > 0:
                    aspect = width / height
                    if aspect > 5 or aspect < 0.2:
                        continue
            final_results.append(res)


        for class_id, score, x1, y1, x2, y2 in final_results:
            x1 = int(max(0, min(x1, self.img_w)))
            y1 = int(max(0, min(y1, self.img_h)))
            x2 = int(max(0, min(x2, self.img_w)))
            y2 = int(max(0, min(y2, self.img_h)))
            results.append((class_id, score, x1, y1, x2, y2))

        return results



rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
]

def draw_detection(img, bbox, score, class_name, label=None):
    x1, y1, x2, y2 = bbox

    if class_name == "car":
        color = (255, 0, 0)
    elif class_name == "accident":
        color = (255, 0, 255)
    elif class_name == "fire":
        color = (0, 165, 255)
    elif class_name == "person":
        color = (0, 0, 255)
    elif class_name == "water":
        color = (255, 255, 0)
    else:
        color = (255, 255, 255)  

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if label is None:
        label = f"{class_name}: {score:.2f}"

    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED
    )

    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )
    
def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
