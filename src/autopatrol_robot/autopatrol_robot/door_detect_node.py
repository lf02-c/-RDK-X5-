#!/usr/bin/env python3

import os
import sys

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

DIST_PACKAGES_PATH = "/usr/lib/python3/dist-packages"
if DIST_PACKAGES_PATH not in sys.path:
    sys.path.insert(0, DIST_PACKAGES_PATH)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
from time import time, perf_counter
import logging
import math
import queue
import subprocess
import threading

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

CLASS_CONF = {
    "car": 0.20,
    "accident": 0.55,
    "fire": 0.45,
    "person": 0.35,
    "water": 0.50,
}
RAW_SCORE_THRESHOLD = 0.08
DEFAULT_CLASS_CONF = 0.25
VALID_CLASSES = ["car", "accident", "fire", "person", "water"]
DEFAULT_MODEL_PATH = '/home/sunrise/yolov8n_5class_640_modified.bin'
POSTPROCESS_TOPK_PER_CLASS = 200
PIPELINE_QUEUE_SIZE = 1
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 60
DEFAULT_CAMERA_FOURCC = "MJPG"

ENABLE_OPENCV_PREPROCESS = True
ENABLE_GAMMA_CORRECTION = True
ENABLE_HIGHLIGHT_SUPPRESS = True
ENABLE_GAUSSIAN_BLUR = False
ENABLE_CLAHE = False
BRIGHTNESS_SAMPLE_SIZE = (160, 120)
HIGHLIGHT_PIXEL_THRESHOLD = 245
HIGHLIGHT_RATIO_THRESHOLD = 0.08
HIGHLIGHT_P95_THRESHOLD = 245
GAMMA_MEAN_THRESHOLD = 60
GAMMA_DARK_RATIO_THRESHOLD = 0.55
GAMMA_PIXEL_THRESHOLD = 45
PREPROCESS_STATS_KEYS = [
    "disabled",
    "fast_path",
    "highlight",
    "gamma",
    "clahe",
    "blur",
]
POSTPROCESS_STAGE_KEYS = [
    "cls_filter_ms",
    "dfl_decode_ms",
    "concat_ms",
    "nms_ms",
    "final_filter_ms",
]

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

TIMING_KEYS = [
    "camera_read_ms",
    "opencv_preprocess_ms",
    "model_preprocess_ms",
    "bpu_forward_ms",
    "c2numpy_ms",
    "postprocess_ms",
    "draw_ms",
    "ros_msg_ms",
    "publish_ms",
    "queue_wait_ms",
    "end_to_end_ms",
    "total_ms",
]


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


def get_brightness_info(
    img,
    sample_size=BRIGHTNESS_SAMPLE_SIZE,
    highlight_pixel_threshold=HIGHLIGHT_PIXEL_THRESHOLD,
    gamma_pixel_threshold=GAMMA_PIXEL_THRESHOLD
):
    if sample_size:
        sample_w, sample_h = sample_size
        height, width = img.shape[:2]
        if width > sample_w or height > sample_h:
            img = cv2.resize(img, (sample_w, sample_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_count = gray.size
    mean_brightness = float(gray.mean())
    dark_ratio = np.count_nonzero(gray < gamma_pixel_threshold) / pixel_count
    highlight_ratio = np.count_nonzero(gray > highlight_pixel_threshold) / pixel_count
    p95_val = float(np.percentile(gray, 95))
    return mean_brightness, dark_ratio, highlight_ratio, p95_val


def _bump_counter(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _record_brightness_stats(stats, mean_brightness, p95_val, dark_ratio, highlight_ratio):
    if stats is None:
        return
    stats["count"] += 1
    stats["mean_sum"] += mean_brightness
    stats["p95_sum"] += p95_val
    stats["dark_ratio_sum"] += dark_ratio
    stats["highlight_ratio_sum"] += highlight_ratio


def format_ms(label, value):
    return f"{label}={float(value):.2f} ms"


def fourcc_to_string(value):
    try:
        fourcc_int = int(value)
    except (TypeError, ValueError):
        return "UNKNOWN"

    if fourcc_int <= 0:
        return "UNKNOWN"

    chars = []
    for i in range(4):
        char_code = (fourcc_int >> (8 * i)) & 0xFF
        if char_code < 32 or char_code > 126:
            return "UNKNOWN"
        chars.append(chr(char_code))
    return "".join(chars)


def normalize_fourcc(value):
    fourcc = str(value or DEFAULT_CAMERA_FOURCC).strip().upper()
    if len(fourcc) != 4:
        return DEFAULT_CAMERA_FOURCC
    return fourcc


def preprocess_for_yolo(
    img,
    enabled=ENABLE_OPENCV_PREPROCESS,
    stats=None,
    sample_size=BRIGHTNESS_SAMPLE_SIZE,
    highlight_pixel_threshold=HIGHLIGHT_PIXEL_THRESHOLD,
    highlight_ratio_threshold=HIGHLIGHT_RATIO_THRESHOLD,
    highlight_p95_threshold=HIGHLIGHT_P95_THRESHOLD,
    gamma_mean_threshold=GAMMA_MEAN_THRESHOLD,
    gamma_dark_ratio_threshold=GAMMA_DARK_RATIO_THRESHOLD,
    gamma_pixel_threshold=GAMMA_PIXEL_THRESHOLD,
    brightness_stats=None
):
    if not enabled:
        _bump_counter(stats, "disabled")
        return img

    mean_brightness, dark_ratio, highlight_ratio, p95_val = get_brightness_info(
        img,
        sample_size=sample_size,
        highlight_pixel_threshold=highlight_pixel_threshold,
        gamma_pixel_threshold=gamma_pixel_threshold
    )
    _record_brightness_stats(brightness_stats, mean_brightness, p95_val, dark_ratio, highlight_ratio)
    result = img
    need_highlight = (
        ENABLE_HIGHLIGHT_SUPPRESS
        and highlight_ratio >= highlight_ratio_threshold
        and p95_val >= highlight_p95_threshold
    )
    need_gamma = (
        ENABLE_GAMMA_CORRECTION
        and mean_brightness < gamma_mean_threshold
        and dark_ratio > gamma_dark_ratio_threshold
    )

    if not need_highlight and not need_gamma and not ENABLE_CLAHE and not ENABLE_GAUSSIAN_BLUR:
        _bump_counter(stats, "fast_path")
        return img

    if need_highlight:
        _bump_counter(stats, "highlight")
        result = suppress_highlight(result, threshold=235, compress_ratio=0.35)

    if need_gamma:
        _bump_counter(stats, "gamma")
        result = gamma_correction(result, gamma=0.75)

    if ENABLE_CLAHE:
        _bump_counter(stats, "clahe")
        result = clahe_enhance(result, clip_limit=1.8)

    if ENABLE_GAUSSIAN_BLUR:
        _bump_counter(stats, "blur")
        result = light_denoise(result)

    return result

class DoorDetectNode(Node):
    def __init__(self):
        super().__init__('door_detect_node')

        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('enable_opencv_preprocess', ENABLE_OPENCV_PREPROCESS)
        self.declare_parameter('image_queue_depth', 2)
        self.declare_parameter('postprocess_topk_per_class', POSTPROCESS_TOPK_PER_CLASS)
        self.declare_parameter('brightness_sample_width', BRIGHTNESS_SAMPLE_SIZE[0])
        self.declare_parameter('brightness_sample_height', BRIGHTNESS_SAMPLE_SIZE[1])
        self.declare_parameter('highlight_pixel_threshold', HIGHLIGHT_PIXEL_THRESHOLD)
        self.declare_parameter('highlight_ratio_threshold', HIGHLIGHT_RATIO_THRESHOLD)
        self.declare_parameter('highlight_p95_threshold', HIGHLIGHT_P95_THRESHOLD)
        self.declare_parameter('gamma_mean_threshold', GAMMA_MEAN_THRESHOLD)
        self.declare_parameter('gamma_dark_ratio_threshold', GAMMA_DARK_RATIO_THRESHOLD)
        self.declare_parameter('gamma_pixel_threshold', GAMMA_PIXEL_THRESHOLD)
        self.declare_parameter('enable_pipeline_threads', True)
        self.declare_parameter('pipeline_queue_size', PIPELINE_QUEUE_SIZE)
        self.declare_parameter('camera_index', DEFAULT_CAMERA_INDEX)
        self.declare_parameter('camera_width', DEFAULT_CAMERA_WIDTH)
        self.declare_parameter('camera_height', DEFAULT_CAMERA_HEIGHT)
        self.declare_parameter('camera_fps', DEFAULT_CAMERA_FPS)
        self.declare_parameter('camera_fourcc', DEFAULT_CAMERA_FOURCC)
        self.declare_parameter('car_conf', CLASS_CONF["car"])
        self.declare_parameter('accident_conf', CLASS_CONF["accident"])
        self.declare_parameter('fire_conf', CLASS_CONF["fire"])
        self.declare_parameter('person_conf', CLASS_CONF["person"])
        self.declare_parameter('water_conf', CLASS_CONF["water"])

        model_path = self.get_parameter('model_path').value or DEFAULT_MODEL_PATH
        self.enable_opencv_preprocess = bool(self.get_parameter('enable_opencv_preprocess').value)
        image_queue_depth = max(1, int(self.get_parameter('image_queue_depth').value))
        postprocess_topk_per_class = max(1, int(self.get_parameter('postprocess_topk_per_class').value))
        sample_w = int(self.get_parameter('brightness_sample_width').value)
        sample_h = int(self.get_parameter('brightness_sample_height').value)
        self.brightness_sample_size = (sample_w, sample_h) if sample_w > 0 and sample_h > 0 else None
        self.highlight_pixel_threshold = int(self.get_parameter('highlight_pixel_threshold').value)
        self.highlight_ratio_threshold = float(self.get_parameter('highlight_ratio_threshold').value)
        self.highlight_p95_threshold = float(self.get_parameter('highlight_p95_threshold').value)
        self.gamma_mean_threshold = float(self.get_parameter('gamma_mean_threshold').value)
        self.gamma_dark_ratio_threshold = float(self.get_parameter('gamma_dark_ratio_threshold').value)
        self.gamma_pixel_threshold = int(self.get_parameter('gamma_pixel_threshold').value)
        self.enable_pipeline_threads = bool(self.get_parameter('enable_pipeline_threads').value)
        self.pipeline_queue_size = max(1, int(self.get_parameter('pipeline_queue_size').value))
        self.camera_index = max(0, int(self.get_parameter('camera_index').value))
        self.camera_width = max(1, int(self.get_parameter('camera_width').value))
        self.camera_height = max(1, int(self.get_parameter('camera_height').value))
        self.camera_fps = max(1, int(self.get_parameter('camera_fps').value))
        self.camera_fourcc = normalize_fourcc(self.get_parameter('camera_fourcc').value)
        self.class_conf = {
            "car": max(0.0, float(self.get_parameter('car_conf').value)),
            "accident": max(0.0, float(self.get_parameter('accident_conf').value)),
            "fire": max(0.0, float(self.get_parameter('fire_conf').value)),
            "person": max(0.0, float(self.get_parameter('person_conf').value)),
            "water": max(0.0, float(self.get_parameter('water_conf').value)),
        }

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image', image_queue_depth)

        self.accident_pub = self.create_publisher(String, '/accident_detected', 10)
        self.person_pub = self.create_publisher(String, '/person_detected', 10)
        self.fire_pub = self.create_publisher(String, '/fire_detected', 10)
        self.water_pub = self.create_publisher(String, '/water_detected', 10)
        self.congestion_pub = self.create_publisher(String, '/congestion_detected', 10)

        # ---------- ģ�Ͳ��� ----------
        class Opt:
            nms_thres = 0.7
            raw_score_thres = RAW_SCORE_THRESHOLD
            reg = 16
        self.opt = Opt()
        self.opt.model_path = model_path
        self.opt.postprocess_topk_per_class = postprocess_topk_per_class
        self.opt.class_conf = self.class_conf

        self.accident_counter = 0
        self.ACCIDENT_THRESHOLD = 5 


        self.model = YOLOv8_Detect(self.opt)

        self.declare_parameter('video_path', '')
        self.declare_parameter('output_video_path', '')
        video_path = self.get_parameter('video_path').value or ''
        self.output_video_path = self.get_parameter('output_video_path').value or ''
        self.use_video_file = bool(video_path)
        self.video_writer = None
        self.video_writer_failed = False
        self.resources_released = False
        self.shutdown_requested = False
        self.video_capture_finished = False
        self.video_infer_finished = False
        self.video_eof_logged = False
        self.total_processed_frames = 0
        self.cap = self._open_capture(video_path)

        if not self.cap.isOpened():
            if self.use_video_file:
                self.get_logger().error(f"Cannot open video file: {video_path}")
                raise RuntimeError(f"Video open failed: {video_path}")
            camera_device = self._camera_device()
            self.get_logger().error(f"Cannot open camera: {camera_device}")
            raise RuntimeError(f"Camera open failed: {camera_device}")

        output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not output_fps or output_fps <= 1:
            output_fps = 25
        self.output_video_fps = output_fps

        self.frame_count = 0
        self.infer_fps_list = []
        self.node_fps_list = []
        self.last_publish_time = None
        self.next_track_id = 1
        self.car_tracks = {}
        self.timing_samples = {key: [] for key in TIMING_KEYS}
        self.postprocess_perf_samples = {key: [] for key in POSTPROCESS_STAGE_KEYS}
        self.preprocess_branch_counts = {key: 0 for key in PREPROCESS_STATS_KEYS}
        self.preprocess_brightness_stats = {
            "count": 0,
            "mean_sum": 0.0,
            "p95_sum": 0.0,
            "dark_ratio_sum": 0.0,
            "highlight_ratio_sum": 0.0,
        }
        self.raw_queue_dropped = 0
        self.infer_queue_dropped = 0
        self.pipeline_capture_frames = 0
        self.pipeline_infer_frames = 0
        self.pipeline_processed_frames = 0
        self.pipeline_report_start = perf_counter()
        self.stop_event = threading.Event()
        self.pipeline_threads = []
        self.raw_queue = None
        self.infer_queue = None

        # ��ʱ�� 10Hz
        if self.enable_pipeline_threads:
            self.timer = None
            self._start_pipeline_threads()
        else:
            self.timer = self.create_timer(0.001, self.timer_callback)
        if self.use_video_file:
            self.get_logger().info(f"Using video file input: {video_path}")
        else:
            self.get_logger().info(f"Using camera input: /dev/video{self.camera_index}")
        self.get_logger().info(
            f"Vision params: model_path={model_path}, "
            f"enable_opencv_preprocess={self.enable_opencv_preprocess}, "
            f"image_queue_depth={image_queue_depth}, "
            f"postprocess_topk_per_class={postprocess_topk_per_class}, "
            f"brightness_sample_size={self.brightness_sample_size}, "
            f"highlight_pixel_threshold={self.highlight_pixel_threshold}, "
            f"highlight_ratio_threshold={self.highlight_ratio_threshold}, "
            f"highlight_p95_threshold={self.highlight_p95_threshold}, "
            f"gamma_mean_threshold={self.gamma_mean_threshold}, "
            f"gamma_dark_ratio_threshold={self.gamma_dark_ratio_threshold}, "
            f"gamma_pixel_threshold={self.gamma_pixel_threshold}, "
            f"enable_pipeline_threads={self.enable_pipeline_threads}, "
            f"pipeline_queue_size={self.pipeline_queue_size}, "
            f"camera_index={self.camera_index}, "
            f"camera_width={self.camera_width}, "
            f"camera_height={self.camera_height}, "
            f"camera_fps={self.camera_fps}, "
            f"camera_fourcc={self.camera_fourcc}, "
            f"class_conf={self.class_conf}"
        )
        if self.output_video_path:
            self.get_logger().info(
                f"Output video enabled: {self.output_video_path}, fps={self.output_video_fps:.2f}"
            )
        self.get_logger().info("Door Detect Node Started")

    def _set_v4l2_control(self, control_name, value):
        camera_device = self._camera_device()
        command = [
            "v4l2-ctl",
            "-d",
            camera_device,
            f"--set-ctrl={control_name}={value}",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.get_logger().info(f"[CAMERA_CTRL] {camera_device} {control_name}={value}")
        except FileNotFoundError:
            self.get_logger().warn("[CAMERA_CTRL] v4l2-ctl not found, skip camera controls")
        except subprocess.CalledProcessError as e:
            error_text = (e.stderr or e.stdout or "").strip()
            self.get_logger().warn(
                f"[CAMERA_CTRL] failed to set {camera_device} {control_name}={value}: {error_text}"
            )

    def _apply_camera_v4l2_controls(self):
        controls = [
            ("auto_exposure", 3),
            ("exposure_dynamic_framerate", 0),
            ("gain", 64),
            ("gamma", 300),
        ]
        for control_name, value in controls:
            self._set_v4l2_control(control_name, value)

    def _camera_device(self):
        return f"/dev/video{self.camera_index}"

    def _open_capture(self, video_path):
        if self.use_video_file:
            return cv2.VideoCapture(video_path)

        self._apply_camera_v4l2_controls()
        camera_device = self._camera_device()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        requested_fourcc = self.camera_fourcc
        self.get_logger().info(
            f"[CAMERA] requested: {camera_device} {self.camera_width}x{self.camera_height} "
            f"{requested_fourcc} {self.camera_fps}fps"
        )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fourcc = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"[CAMERA] actual: {camera_device} {actual_width}x{actual_height} "
            f"{actual_fourcc} {actual_fps:.0f}fps"
        )
        return cap

    def _read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def _mark_video_eof(self):
        if not self.use_video_file or self.video_eof_logged:
            return
        self.video_eof_logged = True
        self.video_capture_finished = True
        self.get_logger().info("[VIDEO] end of file")

    def _release_io_resources(self):
        if self.resources_released:
            return
        self.resources_released = True
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            if self.output_video_path:
                self.get_logger().info(f"[VIDEO] output saved: {self.output_video_path}")
        if self.cap:
            self.cap.release()

    def _finish_video_input(self):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.stop_event.set()
        if self.timer is not None:
            self.timer.cancel()
        self._mark_video_eof()
        self._release_io_resources()
        self.get_logger().info(f"[VIDEO] total frames processed: {self.total_processed_frames}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def _record_perf(self, timings):
        return

    def _init_video_writer(self, frame):
        if not self.output_video_path or self.video_writer or self.video_writer_failed:
            return

        height, width = frame.shape[:2]
        output_dir = os.path.dirname(self.output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        ext = os.path.splitext(self.output_video_path)[1].lower()
        if ext == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.output_video_fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            self.get_logger().error(
                f"Cannot open output video writer: {self.output_video_path}, "
                f"size={width}x{height}, fps={self.output_video_fps:.2f}"
            )
            self.video_writer.release()
            self.video_writer = None
            self.video_writer_failed = True
            return

        self.get_logger().info(
            f"Saving output video to: {self.output_video_path}, "
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

    def _preprocess_frame(self, frame, timings):
        stage_start = perf_counter()
        preprocessed_frame = preprocess_for_yolo(
            frame,
            enabled=self.enable_opencv_preprocess,
            stats=self.preprocess_branch_counts,
            sample_size=self.brightness_sample_size,
            highlight_pixel_threshold=self.highlight_pixel_threshold,
            highlight_ratio_threshold=self.highlight_ratio_threshold,
            highlight_p95_threshold=self.highlight_p95_threshold,
            gamma_mean_threshold=self.gamma_mean_threshold,
            gamma_dark_ratio_threshold=self.gamma_dark_ratio_threshold,
            gamma_pixel_threshold=self.gamma_pixel_threshold,
            brightness_stats=self.preprocess_brightness_stats
        )
        timings["opencv_preprocess_ms"] = (perf_counter() - stage_start) * 1000.0
        return preprocessed_frame

    def _run_inference(self, preprocessed_frame, timings):
        stage_start = perf_counter()
        input_tensor = self.model.preprocess_yuv420sp(preprocessed_frame)
        timings["model_preprocess_ms"] = (perf_counter() - stage_start) * 1000.0

        stage_start = perf_counter()
        dnn_outputs = self.model.forward(input_tensor)
        timings["bpu_forward_ms"] = (perf_counter() - stage_start) * 1000.0

        stage_start = perf_counter()
        outputs = self.model.c2numpy(dnn_outputs)
        timings["c2numpy_ms"] = (perf_counter() - stage_start) * 1000.0
        context = {
            "img_h": self.model.img_h,
            "img_w": self.model.img_w,
            "x_scale": self.model.x_scale,
            "y_scale": self.model.y_scale,
            "x_shift": self.model.x_shift,
            "y_shift": self.model.y_shift,
        }
        return outputs, context

    def _run_postprocess(self, outputs, timings, context=None):
        stage_start = perf_counter()
        results = self.model.postProcess(outputs, context=context)
        timings["postprocess_ms"] = (perf_counter() - stage_start) * 1000.0
        return results

    def _run_model(self, preprocessed_frame, timings):
        outputs, context = self._run_inference(preprocessed_frame, timings)
        return self._run_postprocess(outputs, timings, context=context)

    def _process_results_and_publish(self, preprocessed_frame, results, timings, total_start, capture_time=None):
        stage_start = perf_counter()

        person_detected = False
        fire_detected = False
        water_detected = False
        accident_in_frame = False
        car_detections = []
        display_count = 0

        for class_id, score, x1, y1, x2, y2 in results:
            class_name = self.model.class_names[class_id]

            if class_name not in VALID_CLASSES:
                continue

            threshold = self.class_conf.get(class_name, DEFAULT_CLASS_CONF)
            if score < threshold:
                continue

            if class_name == "car":
                car_detections.append((score, x1, y1, x2, y2))
                continue

            draw_detection(preprocessed_frame, (x1, y1, x2, y2), score, class_name)
            display_count += 1

            if class_name == "accident":
                accident_in_frame = True
            if class_name == "person":
                person_detected = True
            elif class_name == "fire":
                fire_detected = True
            elif class_name == "water":
                water_detected = True

        tracked_cars, congestion = self.update_car_tracks(car_detections)
        display_count += len(tracked_cars)
        for track_id, score, x1, y1, x2, y2, _ in tracked_cars:
            label = f"car ID:{track_id} {score:.2f}"
            draw_detection(preprocessed_frame, (x1, y1, x2, y2), score, "car", label=label)

        if congestion:
            cv2.putText(preprocessed_frame, "CONGESTION", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if not hasattr(self, "last_congestion_time") or time() - self.last_congestion_time > 2:
                congestion_msg = String()
                congestion_msg.data = "congestion"
                self.congestion_pub.publish(congestion_msg)
                self.last_congestion_time = time()

        if accident_in_frame:
            self.accident_counter += 1
        else:
            self.accident_counter = 0

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

        if water_detected and (not hasattr(self, "last_water_time") or time() - self.last_water_time > 2):
            water_msg = String()
            water_msg.data = "water"
            self.water_pub.publish(water_msg)
            self.last_water_time = time()

        if self.accident_counter >= self.ACCIDENT_THRESHOLD:
            if not hasattr(self, "last_accident_time") or time() - self.last_accident_time > 2:
                accident_msg = String()
                accident_msg.data = "accident"
                self.accident_pub.publish(accident_msg)
                self.last_accident_time = time()
            self.accident_counter = 0

        infer_ms = (
            timings["opencv_preprocess_ms"]
            + timings["model_preprocess_ms"]
            + timings["bpu_forward_ms"]
            + timings["c2numpy_ms"]
            + timings["postprocess_ms"]
        )
        infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
        self.infer_fps_list.append(infer_fps)
        if len(self.infer_fps_list) > 30:
            self.infer_fps_list.pop(0)
        avg_infer_fps = sum(self.infer_fps_list) / len(self.infer_fps_list)
        avg_node_fps = sum(self.node_fps_list) / len(self.node_fps_list) if self.node_fps_list else 0.0

        cv2.putText(preprocessed_frame, f"Node FPS: {avg_node_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(preprocessed_frame, f"Infer FPS: {avg_infer_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(preprocessed_frame, f"Detections: {display_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        timings["draw_ms"] = (perf_counter() - stage_start) * 1000.0

        if self.output_video_path:
            self._init_video_writer(preprocessed_frame)
            if self.video_writer:
                self.video_writer.write(preprocessed_frame)

        stage_start = perf_counter()
        img_msg = self.bridge.cv2_to_imgmsg(preprocessed_frame, "bgr8")
        timings["ros_msg_ms"] = (perf_counter() - stage_start) * 1000.0

        stage_start = perf_counter()
        self.image_pub.publish(img_msg)
        publish_done = perf_counter()
        timings["publish_ms"] = (publish_done - stage_start) * 1000.0
        if self.last_publish_time is not None:
            publish_interval = publish_done - self.last_publish_time
            if publish_interval > 0:
                self.node_fps_list.append(1.0 / publish_interval)
                if len(self.node_fps_list) > 30:
                    self.node_fps_list.pop(0)
        self.last_publish_time = publish_done
        timings["total_ms"] = (perf_counter() - total_start) * 1000.0
        if capture_time is not None:
            timings["end_to_end_ms"] = (publish_done - capture_time) * 1000.0
        self.total_processed_frames += 1
        self.pipeline_processed_frames += 1
        self._record_perf(timings)

    def _put_latest(self, target_queue, item, drop_counter_name):
        if self.use_video_file:
            while not self.stop_event.is_set():
                try:
                    target_queue.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            return

        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            target_queue.get_nowait()
            setattr(self, drop_counter_name, getattr(self, drop_counter_name) + 1)
        except queue.Empty:
            pass

        try:
            target_queue.put_nowait(item)
        except queue.Full:
            setattr(self, drop_counter_name, getattr(self, drop_counter_name) + 1)

    def _start_pipeline_threads(self):
        self.raw_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self.infer_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self.pipeline_threads = [
            threading.Thread(target=self._capture_loop, name="door_capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="door_bpu_inference", daemon=True),
            threading.Thread(target=self._postprocess_publish_loop, name="door_postprocess_publish", daemon=True),
        ]
        for thread in self.pipeline_threads:
            thread.start()
        self.get_logger().info("Pipeline threads started")

    def _capture_loop(self):
        while rclpy.ok() and not self.stop_event.is_set():
            total_start = perf_counter()
            timings = {key: 0.0 for key in TIMING_KEYS}

            stage_start = perf_counter()
            ret, frame = self._read_frame()
            timings["camera_read_ms"] = (perf_counter() - stage_start) * 1000.0
            if not ret:
                if self.use_video_file:
                    self._mark_video_eof()
                    break
                self.get_logger().warn(f"Failed to read frame from {self._camera_device()}")
                continue

            self.frame_count += 1
            self.pipeline_capture_frames += 1
            now = perf_counter()
            self._put_latest(self.raw_queue, {
                "frame": frame,
                "timings": timings,
                "total_start": total_start,
                "capture_time": total_start,
                "ready_time": now,
            }, "raw_queue_dropped")

    def _inference_loop(self):
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                item = self.raw_queue.get(timeout=0.1)
            except queue.Empty:
                if self.use_video_file and self.video_capture_finished:
                    self.video_infer_finished = True
                    break
                continue

            timings = item["timings"]
            start_wait = perf_counter()
            timings["queue_wait_ms"] += (start_wait - item["ready_time"]) * 1000.0
            try:
                preprocessed_frame = self._preprocess_frame(item["frame"], timings)
                item["frame"] = preprocessed_frame
                item["outputs"], item["postprocess_context"] = self._run_inference(preprocessed_frame, timings)
                self.pipeline_infer_frames += 1
            except Exception as e:
                self.get_logger().error(f"Pipeline inference error: {e}")
                continue
            item["ready_time"] = perf_counter()
            self._put_latest(self.infer_queue, item, "infer_queue_dropped")

    def _postprocess_publish_loop(self):
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                item = self.infer_queue.get(timeout=0.1)
            except queue.Empty:
                if self.use_video_file and self.video_infer_finished:
                    self._finish_video_input()
                    break
                continue

            timings = item["timings"]
            timings["queue_wait_ms"] += (perf_counter() - item["ready_time"]) * 1000.0
            try:
                results = self._run_postprocess(
                    item["outputs"],
                    timings,
                    context=item.get("postprocess_context")
                )
                self._process_results_and_publish(
                    item["frame"],
                    results,
                    timings,
                    item["total_start"],
                    capture_time=item["capture_time"]
                )
            except Exception as e:
                self.get_logger().error(f"Pipeline postprocess/publish error: {e}")

    def timer_callback(self):
        total_start = perf_counter()
        timings = {key: 0.0 for key in TIMING_KEYS}

        stage_start = perf_counter()
        ret, frame = self._read_frame()
        timings["camera_read_ms"] = (perf_counter() - stage_start) * 1000.0
        if not ret:
            if self.use_video_file:
                self._finish_video_input()
                return
            self.get_logger().warn(f"Failed to read frame from {self._camera_device()}")
            return

        self.frame_count += 1

        try:
            preprocessed_frame = self._preprocess_frame(frame, timings)
            results = self._run_model(preprocessed_frame, timings)
            self._process_results_and_publish(preprocessed_frame, results, timings, total_start)
            return

            stage_start = perf_counter()
            preprocessed_frame = preprocess_for_yolo(
                frame,
                enabled=self.enable_opencv_preprocess,
                stats=self.preprocess_branch_counts,
                sample_size=self.brightness_sample_size,
                highlight_pixel_threshold=self.highlight_pixel_threshold,
                highlight_ratio_threshold=self.highlight_ratio_threshold,
                highlight_p95_threshold=self.highlight_p95_threshold,
                gamma_mean_threshold=self.gamma_mean_threshold,
                gamma_dark_ratio_threshold=self.gamma_dark_ratio_threshold,
                gamma_pixel_threshold=self.gamma_pixel_threshold,
                brightness_stats=self.preprocess_brightness_stats
            )
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

                threshold = self.class_conf.get(class_name, DEFAULT_CLASS_CONF)
                if score < threshold:
                    continue

                if class_name == "car":
                    car_detections.append((score, x1, y1, x2, y2))
                    continue

                draw_detection(preprocessed_frame, (x1, y1, x2, y2), score, class_name)
                display_count += 1

                if class_name == "accident":
                    accident_in_frame = True
                
                if class_name == "person":
                    person_detected = True
                elif class_name == "fire":
                    fire_detected = True

            tracked_cars, congestion = self.update_car_tracks(car_detections)
            display_count += len(tracked_cars)
            for track_id, score, x1, y1, x2, y2, _ in tracked_cars:
                label = f"car ID:{track_id} {score:.2f}"
                draw_detection(preprocessed_frame, (x1, y1, x2, y2), score, "car", label=label)

            if congestion:
                cv2.putText(preprocessed_frame, "CONGESTION", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if accident_in_frame:
                self.accident_counter += 1
            else:
                self.accident_counter = 0

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

            infer_ms = (
                timings["opencv_preprocess_ms"]
                + timings["model_preprocess_ms"]
                + timings["bpu_forward_ms"]
                + timings["c2numpy_ms"]
                + timings["postprocess_ms"]
            )
            infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            self.infer_fps_list.append(infer_fps)
            if len(self.infer_fps_list) > 30:
                self.infer_fps_list.pop(0)
            avg_infer_fps = sum(self.infer_fps_list) / len(self.infer_fps_list)
            avg_node_fps = sum(self.node_fps_list) / len(self.node_fps_list) if self.node_fps_list else 0.0

            cv2.putText(preprocessed_frame, f"Node FPS: {avg_node_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preprocessed_frame, f"Infer FPS: {avg_infer_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preprocessed_frame, f"Detections: {display_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            timings["draw_ms"] = (perf_counter() - stage_start) * 1000.0

            if self.output_video_path:
                self._init_video_writer(preprocessed_frame)
                if self.video_writer:
                    self.video_writer.write(preprocessed_frame)

            stage_start = perf_counter()
            img_msg = self.bridge.cv2_to_imgmsg(preprocessed_frame, "bgr8")
            timings["ros_msg_ms"] = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            self.image_pub.publish(img_msg)
            publish_done = perf_counter()
            timings["publish_ms"] = (publish_done - stage_start) * 1000.0
            if self.last_publish_time is not None:
                publish_interval = publish_done - self.last_publish_time
                if publish_interval > 0:
                    self.node_fps_list.append(1.0 / publish_interval)
                    if len(self.node_fps_list) > 30:
                        self.node_fps_list.pop(0)
            self.last_publish_time = publish_done
            timings["total_ms"] = (perf_counter() - total_start) * 1000.0
            self._record_perf(timings)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def destroy_node(self):
        self.stop_event.set()
        current_thread = threading.current_thread()
        for thread in self.pipeline_threads:
            if thread.is_alive() and thread is not current_thread:
                thread.join(timeout=1.0)
        self._release_io_resources()
        super().destroy_node()

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
        self.POSTPROCESS_TOPK_PER_CLASS = getattr(opt, "postprocess_topk_per_class", POSTPROCESS_TOPK_PER_CLASS)
        self.class_conf = getattr(opt, "class_conf", CLASS_CONF)
        self.class_score_thresholds = np.array(
            [
                max(self.RAW_SCORE_THRESHOLD, self.class_conf.get(name, DEFAULT_CLASS_CONF))
                for name in self.class_names
            ],
            dtype=np.float32
        )


        self.ACCIDENT_SCORE_THRESH = max(
            self.RAW_SCORE_THRESHOLD,
            self.class_conf.get("accident", DEFAULT_CLASS_CONF)
        )
        self.MAX_ACCIDENT_AREA_RATIO = 0.8
        self.MIN_ACCIDENT_AREA_RATIO = 0.005
        self.last_postprocess_timings = {key: 0.0 for key in POSTPROCESS_STAGE_KEYS}

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
        nv12 = np.empty_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor):
        return self.quantize_model[0].forward(input_tensor)

    def c2numpy(self, outputs):
        return [out.buffer for out in outputs]

    def postProcess(self, outputs, context=None):
        if context is None:
            img_h = self.img_h
            img_w = self.img_w
            x_scale = self.x_scale
            y_scale = self.y_scale
            x_shift = self.x_shift
            y_shift = self.y_shift
        else:
            img_h = context["img_h"]
            img_w = context["img_w"]
            x_scale = context["x_scale"]
            y_scale = context["y_scale"]
            x_shift = context["x_shift"]
            y_shift = context["y_shift"]

        results = []
        stage_timings = {key: 0.0 for key in POSTPROCESS_STAGE_KEYS}
        strides = [8, 16, 32]
        anchors = [self.s_anchor, self.m_anchor, self.l_anchor]
        bbox_scales = [self.s_bboxes_scale, self.m_bboxes_scale, self.l_bboxes_scale]

        all_boxes = []
        all_scores = []
        all_ids = []

        for i in range(3):
            stage_start = perf_counter()
            bbox = outputs[self.bbox_out_idx[i]].reshape(-1, self.REG * 4) * bbox_scales[i]
            cls = outputs[self.cls_out_idx[i]].reshape(-1, self.CLASSES_NUM)

            class_ids = np.argmax(cls, axis=1)
            class_logits = cls[np.arange(cls.shape[0]), class_ids]
            class_scores = 1 / (1 + np.exp(-class_logits))
            score_thresholds = self.class_score_thresholds[class_ids]

            mask = class_scores >= score_thresholds
            bbox = bbox[mask]
            class_ids = class_ids[mask]
            class_scores = class_scores[mask]
            stage_timings["cls_filter_ms"] += (perf_counter() - stage_start) * 1000.0

            if len(bbox) == 0:
                continue

            stage_start = perf_counter()
            bbox = bbox.reshape(-1, 4, self.REG)
            bbox = softmax(bbox, axis=2)
            bbox = np.sum(bbox * self.weights_static, axis=2)

            anchor = anchors[i][mask]

            x1 = (anchor[:, 0] - bbox[:, 0]) * strides[i]
            y1 = (anchor[:, 1] - bbox[:, 1]) * strides[i]
            x2 = (anchor[:, 0] + bbox[:, 2]) * strides[i]
            y2 = (anchor[:, 1] + bbox[:, 3]) * strides[i]

            x1 = (x1 - x_shift) / x_scale
            y1 = (y1 - y_shift) / y_scale
            x2 = (x2 - x_shift) / x_scale
            y2 = (y2 - y_shift) / y_scale

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            all_boxes.append(boxes)
            all_scores.append(class_scores)
            all_ids.append(class_ids)
            stage_timings["dfl_decode_ms"] += (perf_counter() - stage_start) * 1000.0

        if not all_boxes:
            self.last_postprocess_timings = stage_timings
            return results

        stage_start = perf_counter()
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)
        stage_timings["concat_ms"] = (perf_counter() - stage_start) * 1000.0

        stage_start = perf_counter()
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
                score_thresh = self.class_score_thresholds[class_id]
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

            if len(scores_cls) > self.POSTPROCESS_TOPK_PER_CLASS:
                topk_idx = np.argpartition(
                    scores_cls,
                    -self.POSTPROCESS_TOPK_PER_CLASS
                )[-self.POSTPROCESS_TOPK_PER_CLASS:]
                boxes_cls = boxes_cls[topk_idx]
                scores_cls = scores_cls[topk_idx]
                wh = wh[topk_idx]

            nms_boxes = np.concatenate([boxes_cls[:, 0:2], wh], axis=1)

            indices = cv2.dnn.NMSBoxes(
                nms_boxes.tolist(),
                scores_cls.tolist(),
                score_thresh,
                nms_thresh
            )
            if len(indices) == 0:
                continue

            for idx in np.array(indices).reshape(-1):
                x1, y1, x2, y2 = boxes_cls[idx]
                score = scores_cls[idx]
                temp_results.append((class_id, score, x1, y1, x2, y2))

        stage_timings["nms_ms"] = (perf_counter() - stage_start) * 1000.0
        stage_start = perf_counter()
        if temp_results:
            result_array = np.asarray(temp_results, dtype=np.float32)
            class_ids = result_array[:, 0].astype(np.int32)
            x1 = result_array[:, 2]
            y1 = result_array[:, 3]
            x2 = result_array[:, 4]
            y2 = result_array[:, 5]
            valid = np.ones(result_array.shape[0], dtype=bool)

            if self.CLASSES_NUM > 4 and "accident" in self.class_names:
                accident_id = self.class_names.index("accident")
                accident_mask = class_ids == accident_id
                if np.any(accident_mask):
                    width = x2 - x1
                    height = y2 - y1
                    area_ratio = (width * height) / max(1, img_w * img_h)
                    aspect = np.divide(
                        width,
                        height,
                        out=np.zeros_like(width),
                        where=height > 0
                    )
                    valid_accident = (
                        (area_ratio >= self.MIN_ACCIDENT_AREA_RATIO)
                        & (area_ratio <= self.MAX_ACCIDENT_AREA_RATIO)
                        & (aspect >= 0.2)
                        & (aspect <= 5.0)
                    )
                    valid[accident_mask] = valid_accident[accident_mask]

            result_array = result_array[valid]
            if len(result_array) > 0:
                result_array[:, 2] = np.clip(result_array[:, 2], 0, img_w)
                result_array[:, 3] = np.clip(result_array[:, 3], 0, img_h)
                result_array[:, 4] = np.clip(result_array[:, 4], 0, img_w)
                result_array[:, 5] = np.clip(result_array[:, 5], 0, img_h)
                for class_id, score, x1, y1, x2, y2 in result_array:
                    results.append((int(class_id), float(score), int(x1), int(y1), int(x2), int(y2)))

        stage_timings["final_filter_ms"] = (perf_counter() - stage_start) * 1000.0
        self.last_postprocess_timings = stage_timings
        return results

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
    node = None
    try:
        node = DoorDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
