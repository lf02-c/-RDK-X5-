#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from cv_bridge import CvBridge

import cv2
import ast
import base64
import hashlib
import threading
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from time import monotonic, time

from flask import abort, Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO

from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from autopatrol_robot.keepout_mask import generate_keepout_mask

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


def quaternion_to_yaw(x, y, z, w):
    """Convert a quaternion to planar yaw without external helpers."""
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


# ================= Flask =================
def find_source_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "web" / "index.html").exists():
            return parent
    return None


def get_installed_share_dir():
    if get_package_share_directory is None:
        return None

    try:
        return Path(get_package_share_directory("autopatrol_robot"))
    except Exception:
        return None


def resolve_template_dir():
    project_root = find_source_project_root()
    if project_root is not None:
        return project_root / "src" / "web"

    share_dir = get_installed_share_dir()
    if share_dir is not None:
        return share_dir / "web"

    return Path(__file__).resolve().parent


def resolve_data_dir():
    env_dir = os.environ.get("LD_ALARM_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    project_root = find_source_project_root()
    if project_root is not None:
        return project_root / "data"

    share_dir = get_installed_share_dir()
    if share_dir is not None:
        return share_dir / "data"

    return Path(__file__).resolve().parent / "data"


def resolve_maps_dir():
    env_dir = os.environ.get("LD_MAPS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    return Path("/home/sunrise/LD/maps")


TEMPLATE_DIR = resolve_template_dir()
DATA_DIR = resolve_data_dir()
ALARM_LOG_PATH = DATA_DIR / "alarm_logs.json"
EVIDENCE_DIR = DATA_DIR / "alarm_evidence"
PATROL_POINTS_PATH = DATA_DIR / "patrol_points.json"
MAPS_DIR = resolve_maps_dir()
MAP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAP_FREE_MAX = 25
MAP_OCCUPIED_MIN = 65
MAP_FREE_THRESHOLD = 0.196
MAP_OCCUPIED_THRESHOLD = 0.65
MAP_FINGERPRINT_DECIMALS = 6
ALARM_COOLDOWN_SECONDS = {
    "person": 30.0,
    "fire": 30.0,
    "accident": 30.0,
    "water": 30.0,
    "congestion": 60.0,
    "sensor": 30.0,
}
SENSOR_THRESHOLDS = {
    "temp_warn": 35.0,
    "temp_danger": 45.0,

    "mq2_warn": 500.0,
    "mq2_danger": 600.0,

    "co_warn": 35.0,
    "co_danger": 100.0,

    "pm25_warn": 75.0,
    "pm25_danger": 150.0,

    "pm10_warn": 150.0,
    "pm10_danger": 300.0,

    "hc_low_warn": 1.0,
    "hc_low_danger": 8.0,

    "hc_high_warn": 38.0,
    "hc_high_danger": 45.0,
}
alarm_log_lock = threading.Lock()
patrol_points_lock = threading.Lock()
maps_lock = threading.RLock()
saved_map_fingerprint_cache = {}
web_bridge_node_instance = None
PATROL_COMMANDS = {
    "start": "/patrol/start",
    "pause": "/patrol/pause",
    "resume": "/patrol/resume",
    "cancel": "/patrol/cancel",
    "return_home": "/patrol/return_home",
}
PATROL_STATES = {
    "idle": "未巡检",
    "running": "准备巡检",
    "navigating": "巡检中",
    "pausing": "正在暂停",
    "paused": "已暂停",
    "arrived": "已到达",
    "canceling": "正在取消",
    "canceled": "已取消",
    "returning": "返回起点中",
    "completed": "巡检完成",
    "error": "导航失败",
}
PATROL_ACTIVE_STATES = {
    "running", "navigating", "pausing", "paused",
    "canceling", "returning", "arrived",
}
MANUAL_LINEAR_X = 0.08
MANUAL_ANGULAR_Z = 0.6
MANUAL_WATCHDOG_SECONDS = 0.5
MANUAL_DIRECTION_VELOCITIES = {
    "stop": (0.0, 0.0),
    "forward": (MANUAL_LINEAR_X, 0.0),
    "backward": (-MANUAL_LINEAR_X, 0.0),
    "left": (0.0, MANUAL_ANGULAR_Z),
    "right": (0.0, -MANUAL_ANGULAR_Z),
    "forward_left": (MANUAL_LINEAR_X, MANUAL_ANGULAR_Z),
    "forward_right": (MANUAL_LINEAR_X, -MANUAL_ANGULAR_Z),
    "backward_left": (-MANUAL_LINEAR_X, MANUAL_ANGULAR_Z),
    "backward_right": (-MANUAL_LINEAR_X, -MANUAL_ANGULAR_Z),
}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False
)


@socketio.on('connect', namespace='/')
def handle_socket_connect():
    node = web_bridge_node_instance
    if node is None:
        return

    latest_map, latest_robot_pose = node.get_navigation_cache()
    if latest_map is not None:
        socketio.emit('map', latest_map, to=request.sid, namespace='/')
    if latest_robot_pose is not None:
        socketio.emit(
            'robot_pose',
            latest_robot_pose,
            to=request.sid,
            namespace='/'
        )
    socketio.emit(
        'patrol_status', node.patrol_data, to=request.sid, namespace='/'
    )
    socketio.emit(
        'patrol_task_state',
        node.get_patrol_task(),
        to=request.sid,
        namespace='/'
    )
    socketio.emit(
        'patrol_safety_state',
        node.get_safety_state(),
        to=request.sid,
        namespace='/'
    )
    socketio.emit(
        'manual_control_state',
        node.get_manual_control_state(),
        to=request.sid,
        namespace='/'
    )


@socketio.on('manual_control_enter', namespace='/')
def handle_manual_control_enter(_data=None):
    node = web_bridge_node_instance
    if node is None:
        return {
            "success": False,
            "error": "bridge_unavailable",
            "message": "ROS bridge 尚未就绪",
        }
    return node.enter_manual_control(request.sid)


@socketio.on('manual_control_exit', namespace='/')
def handle_manual_control_exit(data=None):
    node = web_bridge_node_instance
    if node is None:
        return {
            "success": False,
            "error": "bridge_unavailable",
            "message": "ROS bridge 尚未就绪",
        }
    reason = data.get("reason") if isinstance(data, dict) else "button"
    return node.exit_manual_control(request.sid, reason=reason)


@socketio.on('manual_control_cmd', namespace='/')
def handle_manual_control_command(data=None):
    node = web_bridge_node_instance
    if node is None:
        return {
            "success": False,
            "error": "bridge_unavailable",
            "message": "ROS bridge 尚未就绪",
        }
    return node.handle_manual_control_command(request.sid, data)


@socketio.on('disconnect', namespace='/')
def handle_socket_disconnect(_reason=None):
    node = web_bridge_node_instance
    if node is not None:
        node.handle_manual_control_disconnect(request.sid)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/alarms', methods=['GET'])
def api_get_alarms():
    return jsonify({"alarms": load_alarm_logs()})


@app.route('/api/alarms/clear', methods=['POST'])
def api_clear_alarms():
    clear_alarm_logs()
    if web_bridge_node_instance is not None:
        web_bridge_node_instance.clear_alarm_cooldowns()
    socketio.emit(
        'alarms_cleared',
        {},
        namespace='/'
    )
    return jsonify({"alarms": []})


@app.route('/api/alarm-evidence/<filename>', methods=['GET'])
def api_get_alarm_evidence(filename):
    evidence_name = Path(filename).name
    if evidence_name != filename or not evidence_name.lower().endswith(".jpg"):
        abort(404)

    evidence_path = EVIDENCE_DIR / evidence_name
    if not evidence_path.exists() or not evidence_path.is_file():
        abort(404)

    return send_from_directory(str(EVIDENCE_DIR), evidence_name)


@app.route('/api/alarms/<alarm_id>/status', methods=['POST'])
def api_update_alarm_status(alarm_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")

    if status not in ("未处理", "已确认", "已忽略", "已处理"):
        return jsonify({"error": "invalid status"}), 400

    alarm = update_alarm_status(alarm_id, status)
    if alarm is None:
        return jsonify({"error": "alarm not found"}), 404

    socketio.emit(
        'alarm_updated',
        alarm,
        namespace='/'
    )

    return jsonify({"alarm": alarm})


@app.route('/api/maps', methods=['GET'])
def api_get_maps():
    try:
        return jsonify({"maps": list_map_assets()})
    except OSError as exc:
        return api_error("maps_read_failed", f"读取地图目录失败：{exc}", 500)


@app.route('/api/maps/current_match', methods=['GET'])
def api_get_current_map_match():
    try:
        return jsonify(resolve_current_map_match())
    except OSError as exc:
        return api_error("maps_read_failed", f"读取地图目录失败：{exc}", 500)


@app.route('/api/maps/current_match/debug', methods=['GET'])
def api_get_current_map_match_debug():
    node = web_bridge_node_instance
    if node is None:
        return jsonify({
            "current": {"has_map": False},
            "candidates": [],
        })

    current_map, _ = node.get_current_map_snapshot()
    if current_map is None:
        return jsonify({
            "current": {"has_map": False},
            "candidates": [],
        })

    try:
        return jsonify(build_current_map_match_debug(current_map))
    except ValueError as exc:
        return api_error("invalid_map_data", str(exc), 422)
    except OSError as exc:
        return api_error("maps_read_failed", f"读取地图目录失败：{exc}", 500)


@app.route('/api/maps/save_current', methods=['POST'])
def api_save_current_map():
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return api_error("invalid_json", "请求内容必须是 JSON 对象", 400)

    raw_name = data.get("map_name")
    map_name = str(raw_name or datetime.now().strftime("map_%Y%m%d_%H%M%S"))
    try:
        map_name = validate_map_name(map_name)
    except ValueError as exc:
        return api_error("invalid_map_name", str(exc), 400)

    node = web_bridge_node_instance
    if node is None:
        return api_error("map_unavailable", "Web Bridge 尚未就绪", 503)
    latest_map, _ = node.get_current_map_snapshot()
    if latest_map is None:
        return api_error("map_unavailable", "尚未收到 /map 数据", 503)

    try:
        asset = save_current_map_asset(map_name, latest_map)
    except FileExistsError:
        return api_error("map_already_exists", "同名地图已存在", 409)
    except ValueError as exc:
        return api_error("invalid_map_data", str(exc), 422)
    except OSError as exc:
        return api_error("map_save_failed", f"地图保存失败：{exc}", 500)

    return jsonify({"map": asset}), 201


@app.route('/api/maps/<map_name>', methods=['GET'])
def api_get_map_detail(map_name):
    try:
        map_name = validate_map_name(map_name)
    except ValueError as exc:
        return api_error("invalid_map_name", str(exc), 400)
    try:
        asset = get_map_asset_summary(map_name)
        if not any((
            asset["yaml_exists"],
            asset["pgm_exists"],
            asset["zones_exists"],
        )):
            return api_error("map_not_found", "地图不存在", 404)
        if not asset["complete"]:
            return api_error("map_incomplete", "地图 YAML 或 PGM 不完整", 409)
        yaml_data, map_payload = load_saved_map(map_name)
    except ValueError as exc:
        return api_error("invalid_map", str(exc), 422)
    except OSError as exc:
        return api_error("map_read_failed", f"读取地图失败：{exc}", 500)

    return jsonify({
        "asset": asset,
        "yaml": yaml_data,
        "map": map_payload,
    })


@app.route('/api/maps/<map_name>', methods=['DELETE'])
def api_delete_map(map_name):
    try:
        map_name = validate_map_name(map_name)
    except ValueError as exc:
        return api_error("invalid_map_name", str(exc), 400)

    try:
        deleted = delete_map_asset(map_name)
    except FileNotFoundError:
        return api_error("map_not_found", "地图不存在", 404)
    except OSError as exc:
        return api_error("map_delete_failed", f"地图删除失败：{exc}", 500)
    return jsonify({"map_name": map_name, "deleted": deleted})


@app.route('/api/maps/<map_name>/zones', methods=['GET'])
def api_get_map_zones(map_name):
    try:
        map_name = validate_map_name(map_name)
    except ValueError as exc:
        return api_error("invalid_map_name", str(exc), 400)
    try:
        _, map_payload = load_saved_map(map_name)
    except FileNotFoundError:
        return api_error("map_not_found", "地图不存在", 404)
    except ValueError as exc:
        return api_error("invalid_map", str(exc), 422)
    except OSError as exc:
        return api_error("map_read_failed", f"读取地图失败：{exc}", 500)
    try:
        zones = load_zones_document(map_name, map_payload)
    except ValueError as exc:
        return api_error("invalid_zones", str(exc), 422)
    except OSError as exc:
        return api_error("zones_read_failed", f"读取禁区失败：{exc}", 500)
    return jsonify(zones)


@app.route('/api/maps/<map_name>/zones', methods=['POST'])
def api_save_map_zones(map_name):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error("invalid_json", "请求内容必须是 JSON 对象", 400)

    try:
        map_name = validate_map_name(map_name)
    except ValueError as exc:
        return api_error("invalid_map_name", str(exc), 400)
    try:
        body_map_name = data.get("map_name")
        if body_map_name not in (None, "", map_name):
            return api_error(
                "map_name_mismatch",
                "请求中的 map_name 与 URL 不一致",
                400,
            )
        map_metadata, map_payload = load_saved_map(map_name)
    except FileNotFoundError:
        return api_error("map_not_found", "地图不存在", 404)
    except ValueError as exc:
        return api_error("invalid_map", str(exc), 422)
    except OSError as exc:
        return api_error("map_read_failed", f"读取地图失败：{exc}", 500)
    try:
        document = validate_zones_document(
            data,
            map_name,
            map_payload,
            assign_ids=True,
            update_time=True,
        )
        save_zones_document(map_name, document)
    except ValueError as exc:
        return api_error("invalid_zones", str(exc), 422)
    except OSError as exc:
        return api_error("zones_save_failed", f"禁区保存失败：{exc}", 500)
    try:
        keepout = generate_keepout_mask(
            MAPS_DIR,
            map_name,
            map_metadata,
            map_payload,
            document,
            logger=app.logger,
        )
    except Exception as exc:
        keepout_dir = MAPS_DIR / "keepout"
        keepout = {
            "generated": False,
            "yaml": str(keepout_dir / f"{map_name}_keepout.yaml"),
            "pgm": str(keepout_dir / f"{map_name}_keepout.pgm"),
            "restart_required": True,
            "warning": f"导航禁区 mask 生成失败：{exc}",
        }
        app.logger.exception("keepout mask generation failed")

    response_document = dict(document)
    response_document["keepout"] = keepout
    return jsonify(response_document)


@app.route('/api/patrol_points', methods=['GET'])
def api_get_patrol_points():
    return jsonify({"points": load_patrol_points()})


@app.route('/api/patrol_points/validate', methods=['POST'])
def api_validate_patrol_points():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error("invalid_json", "请求内容必须是 JSON 对象", 400)
    points = data.get("points")
    if not isinstance(points, list):
        return api_error("invalid_points", "points 必须是数组", 400)

    try:
        context = get_patrol_validation_context(data.get("map_name"))
    except PatrolValidationContextError as exc:
        return jsonify({
            "valid": False,
            "map_name": data.get("map_name"),
            "reason_code": exc.code,
            "message": str(exc),
            "results": [],
        }), exc.status_code

    results = validate_patrol_points(points, context)
    return jsonify({
        "valid": bool(points) and all(result["valid"] for result in results),
        "map_name": context["map_name"],
        "message": (
            "所有巡检点校验通过"
            if points and all(result["valid"] for result in results)
            else "存在不合法巡检点"
        ),
        "results": results,
    })


@app.route('/api/patrol_points', methods=['POST'])
def api_create_patrol_point():
    node = web_bridge_node_instance
    if node is not None and node.patrol_point_mutation_blocked():
        return api_error(
            "patrol_active", "巡检执行期间不能修改巡检点", 409
        )
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error("invalid_json", "请求内容必须是 JSON 对象", 400)
    if any(data.get(key) in (None, "") for key in ("x", "y", "yaw")):
        return api_error(
            "invalid_coordinate", "x、y、yaw 均不能为空", 400
        )

    try:
        point = build_patrol_point(data)
    except ValueError as exc:
        return api_error("invalid_coordinate", str(exc), 400)

    try:
        context = get_patrol_validation_context(data.get("map_name"))
    except PatrolValidationContextError as exc:
        return api_error(exc.code, str(exc), exc.status_code)

    validation = validate_patrol_point(point, context, 0)
    if not validation["valid"]:
        return jsonify({
            "error": "invalid_patrol_point",
            "message": "巡检点不合法",
            "validation": validation,
        }), 422

    with patrol_points_lock:
        if node is not None and node.patrol_point_mutation_blocked():
            return api_error(
                "patrol_active", "巡检执行期间不能修改巡检点", 409
            )
        points = load_patrol_points()
        points.append(point)
        save_patrol_points(points)

    task = node.invalidate_prepared_task(
        "巡检点已变更，请重新发送巡检任务"
    ) if node is not None else None
    return jsonify({"point": point, "points": points, "task": task})


@app.route('/api/patrol_points/<point_id>', methods=['DELETE'])
def api_delete_patrol_point(point_id):
    node = web_bridge_node_instance
    if node is not None and node.patrol_point_mutation_blocked():
        return api_error(
            "patrol_active", "巡检执行期间不能修改巡检点", 409
        )
    with patrol_points_lock:
        if node is not None and node.patrol_point_mutation_blocked():
            return api_error(
                "patrol_active", "巡检执行期间不能修改巡检点", 409
            )
        points = load_patrol_points()
        next_points = [point for point in points if point.get("id") != point_id]
        if len(next_points) == len(points):
            return jsonify({"error": "point not found"}), 404
        save_patrol_points(next_points)

    task = node.invalidate_prepared_task(
        "巡检点已变更，请重新发送巡检任务"
    ) if node is not None else None
    return jsonify({"points": next_points, "task": task})


@app.route('/api/patrol_points/clear', methods=['POST'])
def api_clear_patrol_points():
    node = web_bridge_node_instance
    if node is not None and node.patrol_point_mutation_blocked():
        return api_error(
            "patrol_active", "巡检执行期间不能修改巡检点", 409
        )
    with patrol_points_lock:
        if node is not None and node.patrol_point_mutation_blocked():
            return api_error(
                "patrol_active", "巡检执行期间不能修改巡检点", 409
            )
        save_patrol_points([])
    task = node.invalidate_prepared_task(
        "巡检点已变更，请重新发送巡检任务"
    ) if node is not None else None
    return jsonify({"points": [], "task": task})


@app.route('/api/patrol/task', methods=['GET'])
def api_get_patrol_task():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    return jsonify({"task": node.get_patrol_task()})


@app.route('/api/manual_control/state', methods=['GET'])
def api_get_manual_control_state():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    return jsonify(node.get_manual_control_state())


@app.route('/api/patrol/task/prepare', methods=['POST'])
def api_prepare_patrol_task():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    if node.emergency_stop_latched:
        return api_error("emergency_stop", "急停已锁定，不能装载任务", 423)
    if node.patrol_point_mutation_blocked():
        return api_error("patrol_active", "巡检正在执行，不能装载新任务", 409)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error("invalid_json", "请求内容必须是 JSON 对象", 400)
    with patrol_points_lock:
        points = load_patrol_points()
    if not points:
        return api_error("empty_patrol_points", "当前没有巡检点", 422)

    try:
        context = get_patrol_validation_context(data.get("map_name"))
    except PatrolValidationContextError as exc:
        return api_error(exc.code, str(exc), exc.status_code)
    results = validate_patrol_points(points, context)
    if not all(result["valid"] for result in results):
        return jsonify({
            "error": "invalid_patrol_points",
            "message": "存在不合法巡检点",
            "results": results,
        }), 422

    with patrol_points_lock:
        latest_points = load_patrol_points()
    if (
        calculate_patrol_points_hash(latest_points)
        != calculate_patrol_points_hash(points)
    ):
        return api_error(
            "patrol_points_changed",
            "巡检点在装载过程中发生变化，请重新发送任务",
            409,
        )

    task = node.prepare_patrol_task(points, context)
    return jsonify({
        "success": True,
        "message": "任务已装载，等待开始巡检",
        "task": task,
    })


@app.route('/api/patrol/task/discard', methods=['POST'])
def api_discard_patrol_task():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    if node.emergency_stop_latched:
        return api_error("emergency_stop", "急停已锁定", 423)
    task, discarded = node.discard_prepared_task()
    if not discarded:
        return api_error("task_not_prepared", "当前没有待开始的已装载任务", 409)
    return jsonify({
        "success": True,
        "message": "已取消装载任务",
        "task": task,
    })


@app.route('/api/patrol/control/<command>', methods=['POST'])
def api_patrol_control(command):
    if command not in PATROL_COMMANDS:
        return jsonify({"error": "unknown patrol command"}), 404
    if web_bridge_node_instance is None:
        return jsonify({"error": "ROS bridge is not ready"}), 503

    payload, status_code = web_bridge_node_instance.execute_patrol_control(command)
    return jsonify(payload), status_code


@app.route('/api/patrol/emergency_stop', methods=['POST'])
def api_patrol_emergency_stop():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    payload, status_code = node.engage_emergency_stop()
    return jsonify(payload), status_code


@app.route('/api/patrol/emergency_stop/release', methods=['POST'])
def api_release_patrol_emergency_stop():
    node = web_bridge_node_instance
    if node is None:
        return api_error("bridge_unavailable", "ROS bridge 尚未就绪", 503)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("confirm") != "解除急停":
        return api_error(
            "confirmation_required", "必须明确确认解除急停", 400
        )
    payload, status_code = node.release_emergency_stop()
    return jsonify(payload), status_code


def api_error(code, message, status_code):
    return jsonify({"error": code, "message": message}), status_code


def build_unmatched_map_payload(reason):
    return {
        "matched": False,
        "map_name": None,
        "zones_exists": False,
        "reason": reason,
    }


def resolve_map_match_for_fingerprint(current_fingerprint):
    if not current_fingerprint:
        return build_unmatched_map_payload("no_current_map_received")

    candidates = []
    for asset in list_map_assets():
        if not asset.get("complete"):
            continue
        try:
            fingerprint = get_saved_map_fingerprint(asset["map_name"])
        except (OSError, ValueError):
            continue
        if fingerprint == current_fingerprint:
            candidates.append(asset["map_name"])

    candidates.sort()
    if len(candidates) == 1:
        map_name = candidates[0]
        return {
            "matched": True,
            "map_name": map_name,
            "zones_exists": zones_path_for(map_name).is_file(),
            "reason": "matched_by_fingerprint",
        }
    if len(candidates) > 1:
        payload = build_unmatched_map_payload("ambiguous_match")
        payload["candidates"] = candidates
        return payload
    return build_unmatched_map_payload("no_saved_map_matches_current_map")


def resolve_current_map_match():
    node = web_bridge_node_instance
    if node is None:
        return build_unmatched_map_payload("no_current_map_received")
    return resolve_map_match_for_fingerprint(
        node.get_current_map_fingerprint()
    )


class PatrolValidationContextError(ValueError):
    def __init__(self, code, message, status_code=409):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


PATROL_POINT_REASON_MESSAGES = {
    "invalid_coordinate": "坐标或朝向不是有效数字",
    "outside_map": "点位超出当前地图范围",
    "occupied_cell": "点位位于障碍物区域",
    "unknown_cell": "点位位于未知或不确定区域",
    "enabled_keepout": "点位位于已启用禁区内",
    "map_unmatched": "当前地图未匹配到唯一的已保存地图",
    "zones_unavailable": "当前地图禁区数据不可用",
}


def get_patrol_validation_context(requested_map_name):
    try:
        map_name = validate_map_name(requested_map_name)
    except ValueError as exc:
        raise PatrolValidationContextError(
            "map_unmatched", "请求缺少有效的当前地图名称", 409
        ) from exc

    node = web_bridge_node_instance
    if node is None:
        raise PatrolValidationContextError(
            "map_unmatched", "Web Bridge 尚未收到当前地图", 503
        )
    current_map, current_fingerprint = node.get_current_map_snapshot()
    if current_map is None or not current_fingerprint:
        raise PatrolValidationContextError(
            "map_unmatched", "尚未收到 /map 数据", 409
        )

    try:
        match = resolve_map_match_for_fingerprint(current_fingerprint)
    except OSError as exc:
        raise PatrolValidationContextError(
            "map_unmatched", f"当前地图匹配失败：{exc}", 503
        ) from exc
    if not match.get("matched"):
        reason = match.get("reason")
        message = (
            "当前地图匹配到多个历史地图，无法确定禁区"
            if reason == "ambiguous_match"
            else "当前地图未匹配到唯一的已保存地图"
        )
        raise PatrolValidationContextError("map_unmatched", message, 409)
    if match.get("map_name") != map_name:
        raise PatrolValidationContextError(
            "map_unmatched", "页面地图上下文已过期，请等待重新匹配", 409
        )

    zones = []
    if match.get("zones_exists"):
        try:
            _, saved_map = load_saved_map(map_name)
            document = load_zones_document(map_name, saved_map)
            zones = document.get("zones", [])
        except (OSError, ValueError) as exc:
            raise PatrolValidationContextError(
                "zones_unavailable", f"当前地图禁区读取失败：{exc}", 503
            ) from exc

    return {
        "map_name": map_name,
        "map_fingerprint": current_fingerprint,
        "map": current_map,
        "zones": zones,
    }


def point_on_segment(x, y, start, end, tolerance):
    start_x = float(start["x"])
    start_y = float(start["y"])
    end_x = float(end["x"])
    end_y = float(end["y"])
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= tolerance * tolerance:
        return math.hypot(x - start_x, y - start_y) <= tolerance
    projection = (
        (x - start_x) * delta_x + (y - start_y) * delta_y
    ) / length_squared
    if projection < 0.0 or projection > 1.0:
        return False
    nearest_x = start_x + projection * delta_x
    nearest_y = start_y + projection * delta_y
    return math.hypot(x - nearest_x, y - nearest_y) <= tolerance


def point_in_polygon_including_boundary(x, y, points, tolerance):
    if not isinstance(points, list) or len(points) < 3:
        return False
    inside = False
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        if point_on_segment(x, y, start, end, tolerance):
            return True
        start_y = float(start["y"])
        end_y = float(end["y"])
        if (start_y > y) == (end_y > y):
            continue
        intersection_x = (
            float(start["x"])
            + (y - start_y)
            * (float(end["x"]) - float(start["x"]))
            / (end_y - start_y)
        )
        if x < intersection_x:
            inside = not inside
    return inside


def validate_patrol_point(raw_point, context, index):
    point_id = ""
    point_name = f"巡检点 {index + 1}"
    if isinstance(raw_point, dict):
        point_id = str(raw_point.get("id") or "")
        point_name = str(raw_point.get("name") or point_id or point_name)

    result = {
        "id": point_id,
        "name": point_name,
        "index": index,
        "valid": False,
        "reason_codes": [],
        "reasons": [],
    }
    if not isinstance(raw_point, dict):
        result["reason_codes"].append("invalid_coordinate")
        result["reasons"].append(
            PATROL_POINT_REASON_MESSAGES["invalid_coordinate"]
        )
        return result

    try:
        x = float(raw_point["x"])
        y = float(raw_point["y"])
        yaw = float(raw_point["yaw"])
    except (KeyError, TypeError, ValueError):
        x = y = yaw = math.nan
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        result["reason_codes"].append("invalid_coordinate")
        result["reasons"].append(
            PATROL_POINT_REASON_MESSAGES["invalid_coordinate"]
        )
        return result

    map_payload = context["map"]
    grid_x, grid_y = world_to_map_grid(map_payload, x, y)
    width = int(map_payload["width"])
    height = int(map_payload["height"])
    if not (0 <= grid_x < width and 0 <= grid_y < height):
        result["reason_codes"].append("outside_map")
        result["reasons"].append(PATROL_POINT_REASON_MESSAGES["outside_map"])
        return result

    cell_x = math.floor(grid_x)
    cell_y = math.floor(grid_y)
    try:
        occupancy = int(map_payload["data"][cell_y * width + cell_x])
    except (IndexError, TypeError, ValueError):
        occupancy = -1
    if occupancy >= MAP_OCCUPIED_MIN:
        result["reason_codes"].append("occupied_cell")
        result["reasons"].append(PATROL_POINT_REASON_MESSAGES["occupied_cell"])
    elif occupancy < 0 or occupancy > MAP_FREE_MAX:
        result["reason_codes"].append("unknown_cell")
        result["reasons"].append(PATROL_POINT_REASON_MESSAGES["unknown_cell"])

    tolerance = max(1e-9, float(map_payload["resolution"]) * 1e-6)
    for zone in context["zones"]:
        if zone.get("enabled") is False:
            continue
        if point_in_polygon_including_boundary(
            x, y, zone.get("points"), tolerance
        ):
            result["reason_codes"].append("enabled_keepout")
            result["reasons"].append(
                f"位于已启用禁区“{zone.get('name') or zone.get('id')}”内"
            )
            break

    result["valid"] = not result["reason_codes"]
    return result


def validate_patrol_points(points, context):
    return [
        validate_patrol_point(point, context, index)
        for index, point in enumerate(points)
    ]


def calculate_patrol_points_hash(points):
    serialized = json.dumps(
        points,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_map_name(map_name):
    value = str(map_name or "").strip()
    if not MAP_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "地图名称只能包含字母、数字、下划线和短横线，长度为 1～64"
        )
    return value


def yaml_path_for(map_name):
    return MAPS_DIR / f"{validate_map_name(map_name)}.yaml"


def pgm_path_for(map_name):
    return MAPS_DIR / f"{validate_map_name(map_name)}.pgm"


def zones_path_for(map_name):
    return MAPS_DIR / f"{validate_map_name(map_name)}_zones.json"


def ensure_path_in_maps_dir(path):
    root = MAPS_DIR.resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("地图文件路径超出 maps_dir") from exc
    return resolved


def parse_yaml_scalar(raw_value):
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in ("'", '"'):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("YAML 字符串格式无效") from exc
        return str(parsed)
    return value


def parse_map_yaml(yaml_path):
    values = {}
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("地图 YAML 不是 UTF-8 编码") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("地图 YAML 只支持标准扁平 Map Server 格式")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key in {
            "image", "mode", "resolution", "origin", "negate",
            "occupied_thresh", "free_thresh",
        }:
            values[key] = raw_value.strip()

    missing = [key for key in ("image", "resolution", "origin") if key not in values]
    if missing:
        raise ValueError(f"地图 YAML 缺少字段：{', '.join(missing)}")

    image = parse_yaml_scalar(values["image"])
    if not image:
        raise ValueError("地图 YAML 的 image 不能为空")
    mode = parse_yaml_scalar(values.get("mode", "trinary")).lower()
    if mode != "trinary":
        raise ValueError("当前只支持 trinary 地图")

    try:
        resolution = float(values["resolution"])
        origin = ast.literal_eval(values["origin"])
        negate = int(values.get("negate", "0"))
        occupied_thresh = float(
            values.get("occupied_thresh", str(MAP_OCCUPIED_THRESHOLD))
        )
        free_thresh = float(values.get("free_thresh", str(MAP_FREE_THRESHOLD)))
    except (TypeError, ValueError, SyntaxError) as exc:
        raise ValueError("地图 YAML 数值字段无效") from exc

    if (
        not math.isfinite(resolution) or resolution <= 0
        or not isinstance(origin, (list, tuple)) or len(origin) < 3
    ):
        raise ValueError("地图 YAML 的 resolution 或 origin 无效")
    origin_values = [float(origin[index]) for index in range(3)]
    if not all(math.isfinite(value) for value in origin_values):
        raise ValueError("地图 YAML 的 origin 必须是有限数字")
    if negate not in (0, 1):
        raise ValueError("地图 YAML 的 negate 只能为 0 或 1")
    if not 0 <= free_thresh < occupied_thresh <= 1:
        raise ValueError("地图 YAML 的占用阈值无效")

    return {
        "image": image,
        "mode": mode,
        "resolution": resolution,
        "origin": origin_values,
        "negate": negate,
        "occupied_thresh": occupied_thresh,
        "free_thresh": free_thresh,
    }


def image_path_from_yaml(yaml_path, metadata):
    image_path = Path(metadata["image"])
    if image_path.is_absolute():
        candidate = image_path
    else:
        candidate = yaml_path.parent / image_path
    resolved = ensure_path_in_maps_dir(candidate)
    if resolved.suffix.lower() != ".pgm":
        raise ValueError("当前只支持 PGM 地图图像")
    return resolved


def get_map_asset_summary(map_name):
    map_name = validate_map_name(map_name)
    yaml_path = yaml_path_for(map_name)
    zones_path = zones_path_for(map_name)
    default_pgm_path = pgm_path_for(map_name)
    errors = []
    image_path = default_pgm_path
    yaml_metadata = None

    if yaml_path.is_file():
        try:
            yaml_metadata = parse_map_yaml(yaml_path)
            image_path = image_path_from_yaml(yaml_path, yaml_metadata)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    pgm_exists = image_path.is_file()
    paths = [path for path in (yaml_path, image_path, zones_path) if path.exists()]
    modified_at = None
    if paths:
        timestamp = max(path.stat().st_mtime for path in paths)
        modified_at = datetime.fromtimestamp(timestamp).astimezone().isoformat(
            timespec="seconds"
        )

    return {
        "map_name": map_name,
        "yaml_exists": yaml_path.is_file(),
        "pgm_exists": pgm_exists,
        "zones_exists": zones_path.is_file(),
        "complete": yaml_path.is_file() and pgm_exists and not errors,
        "yaml_file": yaml_path.name,
        "pgm_file": image_path.name,
        "zones_file": zones_path.name,
        "modified_at": modified_at,
        "errors": errors,
    }


def list_map_assets():
    with maps_lock:
        if not MAPS_DIR.exists():
            return []
        names = set()
        for path in MAPS_DIR.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            name = None
            if path.suffix.lower() in (".yaml", ".pgm"):
                name = path.stem
            elif path.name.endswith("_zones.json"):
                name = path.name[:-len("_zones.json")]
            if name and MAP_NAME_PATTERN.fullmatch(name):
                names.add(name)
        return [get_map_asset_summary(name) for name in sorted(names)]


def decode_pgm_to_occupancy(image, metadata):
    height, width = image.shape[:2]
    data = []
    negate = metadata["negate"]
    occupied_thresh = metadata["occupied_thresh"]
    free_thresh = metadata["free_thresh"]

    for grid_y in range(height):
        image_y = height - 1 - grid_y
        for grid_x in range(width):
            pixel = int(image[image_y, grid_x])
            probability = pixel / 255.0 if negate else (255 - pixel) / 255.0
            if probability > occupied_thresh:
                data.append(100)
            elif probability < free_thresh:
                data.append(0)
            else:
                data.append(-1)
    return width, height, data


def load_saved_map(map_name):
    map_name = validate_map_name(map_name)
    with maps_lock:
        yaml_path = yaml_path_for(map_name)
        if not yaml_path.is_file():
            raise FileNotFoundError(map_name)
        metadata = parse_map_yaml(yaml_path)
        image_path = image_path_from_yaml(yaml_path, metadata)
        if not image_path.is_file():
            raise FileNotFoundError(image_path.name)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or len(image.shape) != 2:
            raise ValueError("PGM 地图图像无法读取")
        width, height, data = decode_pgm_to_occupancy(image, metadata)
        if width <= 0 or height <= 0:
            raise ValueError("PGM 地图尺寸无效")

        map_payload = {
            "map_name": map_name,
            "source": "saved_map",
            "frame_id": "map",
            "timestamp": yaml_path.stat().st_mtime,
            "width": width,
            "height": height,
            "resolution": metadata["resolution"],
            "origin": {
                "x": metadata["origin"][0],
                "y": metadata["origin"][1],
                "yaw": metadata["origin"][2],
            },
            "data": data,
        }
        return metadata, map_payload


def normalize_map_number(value):
    # ROS MapMetaData.resolution is float32.  Values such as YAML ``0.05``
    # arrive over /map as 0.050000000745..., so nanometre-level comparison
    # makes otherwise identical maps produce different fingerprints.
    number = round(float(value), MAP_FINGERPRINT_DECIMALS)
    return 0.0 if number == 0 else number


def format_map_number(value):
    text = f"{normalize_map_number(value):.9f}".rstrip("0").rstrip(".")
    return text or "0"


def canonicalize_map_payload(map_payload):
    if not isinstance(map_payload, dict):
        raise ValueError("地图数据格式无效")
    try:
        width = int(map_payload["width"])
        height = int(map_payload["height"])
        resolution = normalize_map_number(map_payload["resolution"])
        origin = map_payload["origin"]
        origin_values = {
            "x": normalize_map_number(origin["x"]),
            "y": normalize_map_number(origin["y"]),
            "yaw": normalize_map_number(origin["yaw"]),
        }
        raw_data = map_payload["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("地图数据缺少必要字段") from exc

    if width <= 0 or height <= 0 or resolution <= 0:
        raise ValueError("地图尺寸或分辨率无效")
    if not all(math.isfinite(value) for value in (
        resolution,
        origin_values["x"],
        origin_values["y"],
        origin_values["yaw"],
    )):
        raise ValueError("地图元数据必须是有限数字")
    if not isinstance(raw_data, (list, tuple)) or len(raw_data) < width * height:
        raise ValueError("地图栅格数据长度不足")

    canonical_data = []
    for raw_value in raw_data[:width * height]:
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("地图栅格包含非法数据") from exc
        if value < 0:
            canonical_data.append(-1)
        elif value <= MAP_FREE_MAX:
            canonical_data.append(0)
        elif value >= MAP_OCCUPIED_MIN:
            canonical_data.append(100)
        else:
            canonical_data.append(-1)

    metadata = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin": origin_values,
    }
    return metadata, canonical_data


def calculate_map_fingerprint(map_payload):
    metadata, canonical_data = canonicalize_map_payload(map_payload)
    data_hash = calculate_canonical_data_hash(canonical_data)
    fingerprint_payload = {
        **metadata,
        "data_sha256": data_hash,
    }
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calculate_canonical_data_hash(canonical_data):
    data_bytes = bytes(255 if value < 0 else value for value in canonical_data)
    return hashlib.sha256(data_bytes).hexdigest()


def count_canonical_cells(canonical_data):
    return {
        "free": sum(1 for value in canonical_data if value == 0),
        "occupied": sum(1 for value in canonical_data if value == 100),
        "unknown": sum(1 for value in canonical_data if value < 0),
    }


def flip_canonical_data(canonical_data, width, height):
    flipped = []
    for row in range(height - 1, -1, -1):
        start = row * width
        flipped.extend(canonical_data[start:start + width])
    return flipped


def map_debug_summary(metadata, canonical_data, has_map=True):
    origin = metadata["origin"]
    return {
        "has_map": has_map,
        "width": metadata["width"],
        "height": metadata["height"],
        "resolution": metadata["resolution"],
        "origin_x": origin["x"],
        "origin_y": origin["y"],
        "origin_yaw": origin["yaw"],
        "data_hash": calculate_canonical_data_hash(canonical_data),
        "normalized_counts": count_canonical_cells(canonical_data),
    }


def count_data_differences(left, right):
    if len(left) != len(right):
        return None, None
    difference_count = sum(
        1 for left_value, right_value in zip(left, right)
        if left_value != right_value
    )
    ratio = difference_count / len(left) if left else 0.0
    return difference_count, ratio


def build_current_map_match_debug(current_map):
    current_metadata, current_data = canonicalize_map_payload(current_map)
    current_summary = map_debug_summary(current_metadata, current_data)
    current_hash = current_summary["data_hash"]
    candidates = []

    for asset in list_map_assets():
        map_name = asset["map_name"]
        if not asset.get("complete"):
            candidates.append({
                "map_name": map_name,
                "meta_equal": False,
                "data_hash_equal": False,
                "flipped_data_hash_equal": False,
                "diff_count": None,
                "diff_ratio": None,
                "reason": "incomplete_or_invalid_map",
                "errors": asset.get("errors", []),
            })
            continue

        try:
            _, candidate_map = load_saved_map(map_name)
            candidate_metadata, candidate_data = canonicalize_map_payload(
                candidate_map
            )
        except (OSError, ValueError) as exc:
            candidates.append({
                "map_name": map_name,
                "meta_equal": False,
                "data_hash_equal": False,
                "flipped_data_hash_equal": False,
                "diff_count": None,
                "diff_ratio": None,
                "reason": "map_read_error",
                "error": str(exc),
            })
            continue

        candidate_hash = calculate_canonical_data_hash(candidate_data)
        data_hash_equal = current_hash == candidate_hash
        same_dimensions = (
            current_metadata["width"] == candidate_metadata["width"]
            and current_metadata["height"] == candidate_metadata["height"]
        )
        if same_dimensions:
            flipped_data = flip_canonical_data(
                candidate_data,
                candidate_metadata["width"],
                candidate_metadata["height"],
            )
            flipped_hash_equal = (
                current_hash == calculate_canonical_data_hash(flipped_data)
            )
            diff_count, diff_ratio = count_data_differences(
                current_data,
                candidate_data,
            )
            flipped_diff_count, flipped_diff_ratio = count_data_differences(
                current_data,
                flipped_data,
            )
        else:
            flipped_hash_equal = False
            diff_count = diff_ratio = None
            flipped_diff_count = flipped_diff_ratio = None

        meta_equal = current_metadata == candidate_metadata
        if meta_equal and data_hash_equal:
            reason = "matched_by_fingerprint"
        elif not meta_equal:
            reason = "metadata_mismatch"
        else:
            reason = "data_hash_mismatch"

        candidate_summary = map_debug_summary(
            candidate_metadata,
            candidate_data,
        )
        candidates.append({
            "map_name": map_name,
            "width": candidate_summary["width"],
            "height": candidate_summary["height"],
            "resolution": candidate_summary["resolution"],
            "origin_x": candidate_summary["origin_x"],
            "origin_y": candidate_summary["origin_y"],
            "origin_yaw": candidate_summary["origin_yaw"],
            "data_hash": candidate_hash,
            "normalized_counts": candidate_summary["normalized_counts"],
            "meta_equal": meta_equal,
            "data_hash_equal": data_hash_equal,
            "flipped_data_hash_equal": flipped_hash_equal,
            "diff_count": diff_count,
            "diff_ratio": diff_ratio,
            "flipped_diff_count": flipped_diff_count,
            "flipped_diff_ratio": flipped_diff_ratio,
            "reason": reason,
        })

    return {
        "current": current_summary,
        "candidates": candidates,
    }


def get_saved_map_fingerprint(map_name):
    map_name = validate_map_name(map_name)
    with maps_lock:
        yaml_path = yaml_path_for(map_name)
        if not yaml_path.is_file():
            raise FileNotFoundError(map_name)
        metadata = parse_map_yaml(yaml_path)
        image_path = image_path_from_yaml(yaml_path, metadata)
        if not image_path.is_file():
            raise FileNotFoundError(image_path.name)
        yaml_stat = yaml_path.stat()
        image_stat = image_path.stat()
        signature = (
            yaml_stat.st_mtime_ns,
            yaml_stat.st_size,
            image_stat.st_mtime_ns,
            image_stat.st_size,
        )
        cached = saved_map_fingerprint_cache.get(map_name)
        if cached and cached[0] == signature:
            return cached[1]
        _, map_payload = load_saved_map(map_name)
        fingerprint = calculate_map_fingerprint(map_payload)
        saved_map_fingerprint_cache[map_name] = (signature, fingerprint)
        return fingerprint


def write_temporary_file(target_path, content, binary=False):
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8"}
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        delete=False,
        **kwargs,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
        if not binary:
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary_path


def build_pgm_bytes(metadata, canonical_data):
    width = metadata["width"]
    height = metadata["height"]
    pixels = bytearray()
    for image_y in range(height):
        grid_y = height - 1 - image_y
        row_offset = grid_y * width
        for grid_x in range(width):
            value = canonical_data[row_offset + grid_x]
            pixels.append(205 if value < 0 else (0 if value >= 100 else 254))
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    return header + bytes(pixels)


def save_current_map_asset(map_name, map_payload):
    map_name = validate_map_name(map_name)
    metadata, canonical_data = canonicalize_map_payload(map_payload)
    yaml_path = yaml_path_for(map_name)
    pgm_path = pgm_path_for(map_name)
    zones_path = zones_path_for(map_name)
    with maps_lock:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (yaml_path, pgm_path, zones_path)):
            raise FileExistsError(map_name)

        yaml_content = "\n".join((
            f"image: {pgm_path.name}",
            "mode: trinary",
            f"resolution: {format_map_number(metadata['resolution'])}",
            "origin: ["
            f"{format_map_number(metadata['origin']['x'])}, "
            f"{format_map_number(metadata['origin']['y'])}, "
            f"{format_map_number(metadata['origin']['yaw'])}]",
            "negate: 0",
            f"occupied_thresh: {MAP_OCCUPIED_THRESHOLD}",
            f"free_thresh: {MAP_FREE_THRESHOLD}",
        ))
        pgm_temporary = None
        yaml_temporary = None
        pgm_published = False
        try:
            pgm_temporary = write_temporary_file(
                pgm_path,
                build_pgm_bytes(metadata, canonical_data),
                binary=True,
            )
            yaml_temporary = write_temporary_file(yaml_path, yaml_content)
            os.replace(pgm_temporary, pgm_path)
            pgm_temporary = None
            pgm_published = True
            os.replace(yaml_temporary, yaml_path)
            yaml_temporary = None
        except Exception:
            if pgm_published and pgm_path.exists() and not yaml_path.exists():
                pgm_path.unlink()
            raise
        finally:
            for temporary in (pgm_temporary, yaml_temporary):
                if temporary is not None and temporary.exists():
                    temporary.unlink()

        saved_map_fingerprint_cache.pop(map_name, None)
        return get_map_asset_summary(map_name)


def delete_map_asset(map_name):
    map_name = validate_map_name(map_name)
    with maps_lock:
        yaml_path = yaml_path_for(map_name)
        keepout_dir = MAPS_DIR / "keepout"
        targets = {
            yaml_path,
            pgm_path_for(map_name),
            zones_path_for(map_name),
            keepout_dir / f"{map_name}_keepout.yaml",
            keepout_dir / f"{map_name}_keepout.pgm",
        }
        if yaml_path.is_file():
            try:
                metadata = parse_map_yaml(yaml_path)
                targets.add(image_path_from_yaml(yaml_path, metadata))
            except (OSError, ValueError):
                pass
        existing = sorted((path for path in targets if path.exists()), key=lambda p: p.name)
        if not existing:
            raise FileNotFoundError(map_name)
        deleted = []
        for path in existing:
            path.unlink()
            deleted.append(path.name)
        saved_map_fingerprint_cache.pop(map_name, None)
        return deleted


def world_to_map_grid(map_payload, x, y):
    origin = map_payload["origin"]
    dx = x - float(origin["x"])
    dy = y - float(origin["y"])
    yaw = float(origin["yaw"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    resolution = float(map_payload["resolution"])
    return (
        (cos_yaw * dx + sin_yaw * dy) / resolution,
        (-sin_yaw * dx + cos_yaw * dy) / resolution,
    )


def polygon_area(points):
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point["x"] * next_point["y"]
        total -= next_point["x"] * point["y"]
    return abs(total) / 2.0


def validate_zones_document(
    data,
    map_name,
    map_payload,
    assign_ids=False,
    update_time=False,
):
    if not isinstance(data, dict):
        raise ValueError("zones.json 必须是 JSON 对象")
    body_map_name = data.get("map_name")
    if body_map_name not in (None, "", map_name):
        raise ValueError("zones.json 的 map_name 与地图不一致")
    raw_zones = data.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ValueError("zones 必须是数组")

    width = int(map_payload["width"])
    height = int(map_payload["height"])
    resolution = float(map_payload["resolution"])
    minimum_area = max(1e-6, resolution * resolution / 2.0)
    zone_ids = set()
    zones = []

    for raw_zone in raw_zones:
        if not isinstance(raw_zone, dict):
            raise ValueError("每个禁区必须是 JSON 对象")
        name = str(raw_zone.get("name") or "").strip()
        if not name:
            raise ValueError("禁区名称不能为空")
        if raw_zone.get("type", "keepout") != "keepout":
            raise ValueError(f"禁区“{name}”的 type 必须是 keepout")
        enabled = raw_zone.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"禁区“{name}”的 enabled 必须是布尔值")

        zone_id = str(raw_zone.get("id") or "").strip()
        if not zone_id and assign_ids:
            zone_id = f"zone_{uuid.uuid4().hex[:8]}"
        if not MAP_NAME_PATTERN.fullmatch(zone_id):
            raise ValueError(f"禁区“{name}”的 id 无效")
        if zone_id in zone_ids:
            raise ValueError(f"禁区 ID 重复：{zone_id}")
        zone_ids.add(zone_id)

        raw_points = raw_zone.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(f"禁区“{name}”至少需要 3 个顶点")
        points = []
        distinct_points = set()
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                raise ValueError(f"禁区“{name}”包含无效顶点")
            try:
                x = float(raw_point["x"])
                y = float(raw_point["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"禁区“{name}”包含无效坐标") from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"禁区“{name}”的坐标必须是有限数字")
            grid_x, grid_y = world_to_map_grid(map_payload, x, y)
            if not (0 <= grid_x < width and 0 <= grid_y < height):
                raise ValueError(f"禁区“{name}”的顶点超出地图范围")
            points.append({"x": x, "y": y})
            distinct_points.add((round(x, 9), round(y, 9)))
        if len(distinct_points) < 3:
            raise ValueError(f"禁区“{name}”至少需要 3 个不同顶点")
        if polygon_area(points) < minimum_area:
            raise ValueError(f"禁区“{name}”的面积过小")

        zones.append({
            "id": zone_id,
            "name": name,
            "type": "keepout",
            "enabled": enabled,
            "points": points,
        })

    updated_at = (
        datetime.now().astimezone().isoformat(timespec="seconds")
        if update_time
        else data.get("updated_at")
    )
    return {
        "map_name": map_name,
        "updated_at": updated_at,
        "zones": zones,
    }


def load_zones_document(map_name, map_payload):
    map_name = validate_map_name(map_name)
    path = zones_path_for(map_name)
    with maps_lock:
        if not path.exists():
            return {"map_name": map_name, "updated_at": None, "zones": []}
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError("zones.json 不是有效 JSON") from exc
        return validate_zones_document(data, map_name, map_payload)


def save_zones_document(map_name, document):
    map_name = validate_map_name(map_name)
    path = zones_path_for(map_name)
    content = json.dumps(document, ensure_ascii=False, indent=2)
    with maps_lock:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            temporary = write_temporary_file(path, content)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def load_alarm_logs():
    if not ALARM_LOG_PATH.exists():
        return []

    try:
        with ALARM_LOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("alarms"), list):
        return data["alarms"]

    return []


def load_patrol_points():
    if not PATROL_POINTS_PATH.exists():
        return []

    try:
        with PATROL_POINTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    return data if isinstance(data, list) else []


def save_patrol_points(points):
    target_path = PATROL_POINTS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(points, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_patrol_point(data):
    allowed_types = {"normal", "danger", "sensor", "manual"}
    raw_name = str(data.get("name") or "").strip()
    name = raw_name or f"巡检点 {datetime.now().strftime('%H%M%S')}"

    point_type = str(data.get("type") or "manual").strip()
    if point_type not in allowed_types:
        point_type = "manual"

    try:
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        yaw = float(data.get("yaw", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid coordinate") from exc
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError("coordinate must be finite")

    now = datetime.now()
    return {
        "id": f"point_{now.strftime('%Y%m%d%H%M%S%f')}",
        "name": name,
        "type": point_type,
        "x": x,
        "y": y,
        "yaw": yaw,
        "note": str(data.get("note") or "").strip(),
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S")
    }


def save_alarm_logs(logs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ALARM_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_alarm_log(alarm):
    with alarm_log_lock:
        logs = load_alarm_logs()
        logs = [item for item in logs if item.get("id") != alarm.get("id")]
        logs.insert(0, alarm)
        save_alarm_logs(logs)


def clear_alarm_logs():
    with alarm_log_lock:
        save_alarm_logs([])
        clear_alarm_evidence_files()


def clear_alarm_evidence_files():
    if not EVIDENCE_DIR.exists():
        return

    for evidence_path in EVIDENCE_DIR.glob("*.jpg"):
        try:
            if evidence_path.is_file():
                evidence_path.unlink()
        except OSError:
            pass


def update_alarm_status(alarm_id, status):
    with alarm_log_lock:
        logs = load_alarm_logs()
        updated_alarm = None

        for alarm in logs:
            if alarm.get("id") == alarm_id:
                alarm["status"] = status
                alarm["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated_alarm = alarm
                break

        if updated_alarm is None:
            return None

        save_alarm_logs(logs)
        return updated_alarm


# ================= ROS2 NODE =================
class WebBridge(Node):

    def __init__(self):
        super().__init__('web_bridge_node')
        global MAPS_DIR, PATROL_POINTS_PATH

        self.sensor_data = {
            "temp": 0.0,
            "hum": 0.0,
            "mq2": 0.0,
            "pm25": 0.0,
            "pm10": 0.0,
            "hc": 0.0,
            "co": 0.0
        }

        self.bridge = CvBridge()
        self.latest_frame = None
        self.navigation_cache_lock = threading.Lock()
        self.latest_map = None
        self.latest_map_fingerprint = None
        self.latest_robot_pose = None
        self.tf_available = None
        self.last_tf_failure_emit_time = 0.0
        self.last_tf_warning_time = 0.0
        self.alarm_seq = 0
        self.last_alarm_times = {}
        self.patrol_status = "未巡检"
        self.patrol_data = {
            "version": 1,
            "session_id": "",
            "state": "idle",
            "current_point": None,
            "completed_point_ids": [],
            "failed_points": [],
            "message": "等待巡检任务",
            "updated_at": "",
        }
        self.patrol_task_lock = threading.RLock()
        self.patrol_task = {
            "state": "draft",
            "task_id": "",
            "map_name": None,
            "map_fingerprint": None,
            "points_hash": "",
            "point_count": 0,
            "prepared_at": "",
            "message": "待发送巡检任务",
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
        self.prepared_points_snapshot = None
        self.emergency_stop_latched = False
        self.emergency_stop_parameter_confirmed = False
        self.safety_sync_complete = False
        self.safety_message = "正在确认 STM32 急停状态"
        self.manual_control_lock = threading.RLock()
        self.manual_mode_enabled = False
        self.manual_control_owner_sid = None
        self.manual_last_command_time = None
        self.manual_last_command_at = None
        self.manual_current_direction = "stop"
        self.manual_control_message = "自动模式，未接管手动控制"
        self.manual_state_revision = 0
        self.automatic_motion_pending = False
        self.declare_parameter('web_stream_fps', 25.0)
        self.declare_parameter('web_jpeg_quality', 55)
        self.declare_parameter('web_frame_width', 640)
        self.declare_parameter('web_frame_height', 480)
        self.declare_parameter(
            'patrol_points_file',
            str(PATROL_POINTS_PATH)
        )
        self.declare_parameter('maps_dir', str(MAPS_DIR))
        self.web_stream_fps = max(1.0, float(self.get_parameter('web_stream_fps').value))
        self.web_jpeg_quality = min(
            100,
            max(1, int(self.get_parameter('web_jpeg_quality').value))
        )
        self.web_frame_width = max(1, int(self.get_parameter('web_frame_width').value))
        self.web_frame_height = max(1, int(self.get_parameter('web_frame_height').value))
        PATROL_POINTS_PATH = Path(
            str(self.get_parameter('patrol_points_file').value)
        ).expanduser()
        self.patrol_points_file = str(PATROL_POINTS_PATH)
        MAPS_DIR = Path(str(self.get_parameter('maps_dir').value)).expanduser()
        self.maps_dir = str(MAPS_DIR)
        with maps_lock:
            saved_map_fingerprint_cache.clear()

        # ================= ����ͷ���� =================
        self.create_subscription(
            Image,
            '/camera/image',
            self.image_cb,
            10
        )

        # ================= log���� =================
        self.create_subscription(
            String,
            '/patrol_log',
            self.log_cb,
            10
        )

        patrol_status_qos = QoSProfile(depth=1)
        patrol_status_qos.reliability = ReliabilityPolicy.RELIABLE
        patrol_status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            '/patrol/status',
            self.patrol_status_cb,
            patrol_status_qos
        )
        self.patrol_clients = {
            command: self.create_client(Trigger, service_name)
            for command, service_name in PATROL_COMMANDS.items()
        }
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.manual_watchdog_timer = self.create_timer(
            0.05, self.manual_control_watchdog
        )
        self.stm32_get_parameters_client = self.create_client(
            GetParameters, "/stm32_bridge_node/get_parameters"
        )
        self.stm32_set_parameters_client = self.create_client(
            SetParameters, "/stm32_bridge_node/set_parameters"
        )
        self.safety_sync_pending = False
        self.safety_sync_timer = self.create_timer(
            1.0, self.sync_stm32_emergency_stop
        )

        self.create_subscription(
            String,
            '/person_detected',
            lambda msg: self.vision_alarm_cb(msg, "person", "warning"),
            10
        )
        self.create_subscription(
            String,
            '/fire_detected',
            lambda msg: self.vision_alarm_cb(msg, "fire", "danger"),
            10
        )
        self.create_subscription(
            String,
            '/accident_detected',
            lambda msg: self.vision_alarm_cb(msg, "accident", "danger"),
            10
        )
        self.create_subscription(
            String,
            '/water_detected',
            lambda msg: self.vision_alarm_cb(msg, "water", "warning"),
            10
        )
        self.create_subscription(
            String,
            '/congestion_detected',
            lambda msg: self.vision_alarm_cb(msg, "congestion", "warning"),
            10
        )

        self.create_subscription(Float32, '/temp', self.temp_cb, 10)
        self.create_subscription(Float32, '/hum', self.hum_cb, 10)
        self.create_subscription(Float32, '/mq2', self.mq2_cb, 10)
        self.create_subscription(Float32, '/pm25', self.pm25_cb, 10)
        self.create_subscription(Float32, '/pm10', self.pm10_cb, 10)
        self.create_subscription(Float32, '/hc', self.hc_cb, 10)

        self.create_subscription(
            Float32,
            '/co',
            self.co_cb,
            10
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            map_qos
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sensor_timer = self.create_timer(0.5, self.push_sensor)

        self.timer = self.create_timer(1.0 / self.web_stream_fps, self.push_frame)
        self.robot_pose_timer = self.create_timer(0.2, self.push_robot_pose)

        self.get_logger().info(
            "Web Bridge Started, "
            f"web_stream_fps={self.web_stream_fps:.1f}, "
            f"web_jpeg_quality={self.web_jpeg_quality}, "
            f"web_frame_size={self.web_frame_width}x{self.web_frame_height}, "
            f"patrol_points_file={self.patrol_points_file}, "
            f"maps_dir={self.maps_dir}"
        )

    def get_navigation_cache(self):
        with self.navigation_cache_lock:
            return self.latest_map, self.latest_robot_pose

    def get_current_map_snapshot(self):
        with self.navigation_cache_lock:
            if self.latest_map is None:
                return None, None
            snapshot = dict(self.latest_map)
            snapshot["origin"] = dict(self.latest_map["origin"])
            snapshot["data"] = list(self.latest_map["data"])
            return snapshot, self.latest_map_fingerprint

    def get_current_map_fingerprint(self):
        with self.navigation_cache_lock:
            return self.latest_map_fingerprint

    def map_cb(self, msg):
        origin = msg.info.origin
        orientation = origin.orientation
        payload = {
            "frame_id": msg.header.frame_id or "map",
            "timestamp": (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) / 1_000_000_000.0
            ),
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "resolution": float(msg.info.resolution),
            "origin": {
                "x": float(origin.position.x),
                "y": float(origin.position.y),
                "yaw": quaternion_to_yaw(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w
                )
            },
            "data": list(msg.data)
        }

        try:
            fingerprint = calculate_map_fingerprint(payload)
        except ValueError as exc:
            self.get_logger().warning(f"invalid /map fingerprint data: {exc}")
            fingerprint = None

        with self.navigation_cache_lock:
            self.latest_map = payload
            self.latest_map_fingerprint = fingerprint

        socketio.emit('map', payload, namespace='/')

    def push_robot_pose(self):
        transform = None
        base_frame = None
        lookup_errors = []

        for candidate in ("base_footprint", "base_link"):
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    candidate,
                    Time()
                )
                base_frame = candidate
                break
            except TransformException as exc:
                lookup_errors.append(f"{candidate}: {exc}")

        if transform is None:
            self.emit_tf_unavailable(lookup_errors)
            return

        try:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            payload = {
                "x": float(translation.x),
                "y": float(translation.y),
                "yaw": quaternion_to_yaw(
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w
                ),
                "source": f"tf_map_{base_frame}",
                "frame_id": transform.header.frame_id or "map",
                "base_frame": base_frame,
                "timestamp": (
                    float(transform.header.stamp.sec)
                    + float(transform.header.stamp.nanosec)
                    / 1_000_000_000.0
                ),
                "valid": True
            }
        except Exception as exc:
            self.emit_tf_unavailable([f"invalid transform: {exc}"])
            return

        with self.navigation_cache_lock:
            self.latest_robot_pose = payload

        self.tf_available = True
        socketio.emit('robot_pose', payload, namespace='/')

    def emit_tf_unavailable(self, lookup_errors):
        now = time()
        message = "; ".join(lookup_errors) or "map TF is unavailable"
        payload = {
            "valid": False,
            "source": "tf_unavailable",
            "frame_id": "map",
            "message": message
        }

        with self.navigation_cache_lock:
            self.latest_robot_pose = payload

        if (
            self.tf_available is not False
            or now - self.last_tf_failure_emit_time >= 1.0
        ):
            socketio.emit('robot_pose', payload, namespace='/')
            self.last_tf_failure_emit_time = now

        if (
            self.tf_available is not False
            or now - self.last_tf_warning_time >= 5.0
        ):
            self.get_logger().warning(f"robot pose unavailable: {message}")
            self.last_tf_warning_time = now

        self.tf_available = False

    def _task_timestamp(self):
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def get_patrol_task(self):
        with self.patrol_task_lock:
            return dict(self.patrol_task)

    def emit_patrol_task(self):
        payload = self.get_patrol_task()
        socketio.emit("patrol_task_state", payload, namespace="/")
        return payload

    def _update_patrol_task(self, state, message, **updates):
        with self.patrol_task_lock:
            self.patrol_task.update(updates)
            self.patrol_task["state"] = state
            self.patrol_task["message"] = message
            self.patrol_task["updated_at"] = self._task_timestamp()
        return self.emit_patrol_task()

    def reset_cached_patrol_status(self, message):
        self.patrol_status = PATROL_STATES["idle"]
        self.patrol_data = {
            "version": 1,
            "session_id": "",
            "state": "idle",
            "current_point": None,
            "completed_point_ids": [],
            "failed_points": [],
            "message": message,
            "updated_at": self._task_timestamp(),
        }
        socketio.emit("patrol_status", self.patrol_data, namespace="/")

    def prepare_patrol_task(self, points, context):
        points_snapshot = json.loads(json.dumps(points, ensure_ascii=False))
        task_id = (
            f"task-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        with self.patrol_task_lock:
            self.prepared_points_snapshot = points_snapshot
            self.patrol_task = {
                "state": "prepared",
                "task_id": task_id,
                "map_name": context["map_name"],
                "map_fingerprint": context["map_fingerprint"],
                "points_hash": calculate_patrol_points_hash(points_snapshot),
                "point_count": len(points_snapshot),
                "prepared_at": self._task_timestamp(),
                "message": "任务已装载，等待开始巡检",
                "updated_at": self._task_timestamp(),
            }
        self.reset_cached_patrol_status("任务已装载，等待开始巡检")
        return self.emit_patrol_task()

    def invalidate_prepared_task(self, message):
        with self.patrol_task_lock:
            if self.patrol_task.get("state") != "prepared":
                return dict(self.patrol_task)
            self.prepared_points_snapshot = None
        self.reset_cached_patrol_status(message)
        return self._update_patrol_task("invalidated", message)

    def discard_prepared_task(self):
        with self.patrol_task_lock:
            if self.patrol_task.get("state") != "prepared":
                return dict(self.patrol_task), False
            self.prepared_points_snapshot = None
        self.reset_cached_patrol_status("已取消装载任务")
        return self._update_patrol_task(
            "canceled", "已取消装载任务"
        ), True

    def patrol_point_mutation_blocked(self):
        return (
            self.get_patrol_task().get("state") in {"active", "emergency_stop"}
            or str(self.patrol_data.get("state") or "idle") in PATROL_ACTIVE_STATES
        )

    def _invalidate_task_for_start(self, message):
        with self.patrol_task_lock:
            self.prepared_points_snapshot = None
        return self._update_patrol_task("invalidated", message)

    def _start_prepared_patrol(self):
        task = self.get_patrol_task()
        if task.get("state") != "prepared":
            return {
                "error": "task_not_prepared",
                "message": "请先发送并装载巡检任务",
                "task": task,
            }, 409

        self._update_patrol_task(
            "active", "正在复核任务并启动巡检"
        )

        with patrol_points_lock:
            points = load_patrol_points()
        if calculate_patrol_points_hash(points) != task.get("points_hash"):
            invalidated = self._invalidate_task_for_start(
                "巡检点已变更，请重新发送巡检任务"
            )
            return {
                "error": "task_invalidated",
                "message": invalidated["message"],
                "task": invalidated,
            }, 409

        try:
            context = get_patrol_validation_context(task.get("map_name"))
        except PatrolValidationContextError as exc:
            invalidated = self._invalidate_task_for_start(str(exc))
            return {
                "error": exc.code,
                "message": str(exc),
                "task": invalidated,
            }, exc.status_code
        if context.get("map_fingerprint") != task.get("map_fingerprint"):
            invalidated = self._invalidate_task_for_start(
                "当前地图已变化，请重新发送巡检任务"
            )
            return {
                "error": "task_invalidated",
                "message": invalidated["message"],
                "task": invalidated,
            }, 409

        results = validate_patrol_points(points, context)
        if not points or not all(result["valid"] for result in results):
            invalidated = self._invalidate_task_for_start(
                "巡检点校验已失效，请检查后重新发送任务"
            )
            return {
                "error": "invalid_patrol_points",
                "message": invalidated["message"],
                "results": results,
                "task": invalidated,
            }, 422

        payload, status_code = self.call_patrol_service("start")
        if status_code == 200 and payload.get("success"):
            task = self._update_patrol_task(
                "active", "巡检已启动，等待巡检状态更新"
            )
            payload["task"] = task
        else:
            restored = self._update_patrol_task(
                "prepared", "启动失败，任务仍保持已装载"
            )
            payload["task"] = restored
        return payload, status_code

    def execute_patrol_control(self, command):
        if self.emergency_stop_latched:
            return {
                "error": "emergency_stop",
                "message": "急停已锁定，禁止巡检控制",
                "task": self.get_patrol_task(),
            }, 423
        if (
            command in {"start", "resume", "return_home"}
            and not self.safety_sync_complete
        ):
            return {
                "error": "safety_state_unavailable",
                "message": "尚未确认 STM32 急停状态，禁止启动运动",
                "task": self.get_patrol_task(),
            }, 503
        motion_command = command in {"start", "resume", "return_home"}
        if motion_command:
            with self.manual_control_lock:
                if self.manual_mode_enabled:
                    return {
                        "error": "manual_control_active",
                        "message": "手动模式已接管，禁止启动自动运动",
                        "task": self.get_patrol_task(),
                    }, 409
                self.automatic_motion_pending = True

        try:
            if command == "start":
                return self._start_prepared_patrol()
            if command == "cancel" and self.get_patrol_task().get("state") == "prepared":
                task, _ = self.discard_prepared_task()
                return {
                    "command": "discard",
                    "success": True,
                    "message": "已取消装载任务",
                    "task": task,
                }, 200
            return self.call_patrol_service(command)
        finally:
            if motion_command:
                with self.manual_control_lock:
                    self.automatic_motion_pending = False

    def _wait_for_ros_future(self, future, timeout_sec):
        completed = threading.Event()
        holder = {}

        def store_result(done_future):
            try:
                holder["response"] = done_future.result()
            except Exception as exc:
                holder["error"] = str(exc)
            finally:
                completed.set()

        future.add_done_callback(store_result)
        if not completed.wait(timeout_sec):
            future.cancel()
            return None, "ROS 请求超时"
        if "error" in holder:
            return None, holder["error"]
        return holder.get("response"), None

    def publish_velocity_command(self, linear_x, angular_z):
        try:
            command = Twist()
            command.linear.x = float(linear_x)
            command.angular.z = float(angular_z)
            self.cmd_vel_publisher.publish(command)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def publish_zero_command(self):
        return self.publish_velocity_command(0.0, 0.0)

    def _manual_state_unlocked(self):
        return {
            "version": 1,
            "revision": self.manual_state_revision,
            "mode": "manual" if self.manual_mode_enabled else "automatic",
            "enabled": self.manual_mode_enabled,
            "owner_sid": self.manual_control_owner_sid,
            "direction": self.manual_current_direction,
            "last_command_at": self.manual_last_command_at,
            "watchdog_sec": MANUAL_WATCHDOG_SECONDS,
            "limits": {
                "linear_x": MANUAL_LINEAR_X,
                "angular_z": MANUAL_ANGULAR_Z,
            },
            "message": self.manual_control_message,
        }

    def get_manual_control_state(self):
        with self.manual_control_lock:
            return self._manual_state_unlocked()

    def emit_manual_control_state(self, payload=None):
        state = payload or self.get_manual_control_state()
        socketio.emit("manual_control_state", state, namespace="/")
        return state

    def _manual_patrol_conflict_unlocked(self):
        task_state = self.get_patrol_task().get("state")
        patrol_state = str(self.patrol_data.get("state") or "idle")
        return (
            self.automatic_motion_pending
            or task_state in {"active", "emergency_stop"}
            or patrol_state in PATROL_ACTIVE_STATES
        )

    def enter_manual_control(self, owner_sid):
        with self.manual_control_lock:
            if self.manual_mode_enabled:
                state = self._manual_state_unlocked()
                if self.manual_control_owner_sid == owner_sid:
                    return {
                        "success": True,
                        "message": "本页面已接管手动控制",
                        "state": state,
                    }
                return {
                    "success": False,
                    "error": "manual_control_busy",
                    "message": "手动控制已被其他页面接管",
                    "state": state,
                }
            if not self.safety_sync_complete:
                return {
                    "success": False,
                    "error": "safety_state_unavailable",
                    "message": "尚未确认 STM32 急停状态，禁止手动控制",
                    "state": self._manual_state_unlocked(),
                }
            if self.emergency_stop_latched:
                return {
                    "success": False,
                    "error": "emergency_stop",
                    "message": "急停已锁定，禁止手动控制",
                    "state": self._manual_state_unlocked(),
                }
            if self._manual_patrol_conflict_unlocked():
                return {
                    "success": False,
                    "error": "patrol_active",
                    "message": "巡检正在执行或暂停中，请先取消巡检",
                    "state": self._manual_state_unlocked(),
                }

            zero_success, zero_error = self.publish_zero_command()
            if not zero_success:
                return {
                    "success": False,
                    "error": "zero_publish_failed",
                    "message": f"进入手动模式前发布零速失败：{zero_error}",
                    "state": self._manual_state_unlocked(),
                }
            self.manual_mode_enabled = True
            self.manual_control_owner_sid = owner_sid
            self.manual_last_command_time = None
            self.manual_last_command_at = None
            self.manual_current_direction = "stop"
            self.manual_control_message = "手动模式已接管，等待 WASD 指令"
            self.manual_state_revision += 1
            state = self._manual_state_unlocked()

        self.emit_manual_control_state(state)
        return {
            "success": True,
            "message": "已进入手动模式",
            "state": state,
        }

    def _release_manual_control(
        self,
        message,
        always_publish_zero=False,
        expected_owner_sid=None,
    ):
        with self.manual_control_lock:
            if expected_owner_sid is not None and (
                not self.manual_mode_enabled
                or self.manual_control_owner_sid != expected_owner_sid
            ):
                return {
                    "success": False,
                    "zero_success": True,
                    "zero_error": "",
                    "was_enabled": self.manual_mode_enabled,
                    "owner_mismatch": True,
                    "state": self._manual_state_unlocked(),
                }
            was_enabled = self.manual_mode_enabled
            zero_success = True
            zero_error = ""
            if was_enabled or always_publish_zero:
                zero_success, zero_error = self.publish_zero_command()
            self.manual_mode_enabled = False
            self.manual_control_owner_sid = None
            self.manual_last_command_time = None
            self.manual_last_command_at = None
            self.manual_current_direction = "stop"
            self.manual_control_message = message
            if was_enabled:
                self.manual_state_revision += 1
            state = self._manual_state_unlocked()

        if was_enabled:
            self.emit_manual_control_state(state)
        return {
            "success": zero_success,
            "zero_success": zero_success,
            "zero_error": zero_error,
            "was_enabled": was_enabled,
            "owner_mismatch": False,
            "state": state,
        }

    def exit_manual_control(self, owner_sid, reason="button"):
        allowed_reasons = {
            "button", "blur", "hidden", "page_change", "unload"
        }
        safe_reason = reason if reason in allowed_reasons else "button"
        with self.manual_control_lock:
            if not self.manual_mode_enabled:
                return {
                    "success": True,
                    "message": "当前已是自动模式",
                    "state": self._manual_state_unlocked(),
                }
            if self.manual_control_owner_sid != owner_sid:
                return {
                    "success": False,
                    "error": "not_manual_control_owner",
                    "message": "当前页面没有手动控制权",
                    "state": self._manual_state_unlocked(),
                }

        result = self._release_manual_control(
            f"已退出手动模式（{safe_reason}）",
            expected_owner_sid=owner_sid,
        )
        if result["owner_mismatch"]:
            return {
                "success": False,
                "error": "not_manual_control_owner",
                "message": "手动控制权已发生变化",
                "state": result["state"],
            }
        return {
            "success": result["success"],
            "error": None if result["success"] else "zero_publish_failed",
            "message": (
                "已退出手动模式，车辆保持停止"
                if result["success"]
                else f"已退出手动模式；零速发布失败：{result['zero_error']}"
            ),
            "state": result["state"],
        }

    def handle_manual_control_command(self, owner_sid, data):
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "invalid_payload",
                "message": "手动控制命令必须是 JSON 对象",
                "state": self.get_manual_control_state(),
            }
        direction = data.get("direction")
        if direction not in MANUAL_DIRECTION_VELOCITIES:
            return {
                "success": False,
                "error": "invalid_direction",
                "message": "未知的手动控制方向",
                "state": self.get_manual_control_state(),
            }

        conflict_state = None
        with self.manual_control_lock:
            if not self.manual_mode_enabled:
                return {
                    "success": False,
                    "error": "manual_control_disabled",
                    "message": "未进入手动模式",
                    "state": self._manual_state_unlocked(),
                }
            if self.manual_control_owner_sid != owner_sid:
                return {
                    "success": False,
                    "error": "not_manual_control_owner",
                    "message": "当前页面没有手动控制权",
                    "state": self._manual_state_unlocked(),
                }
            if (
                not self.safety_sync_complete
                or self.emergency_stop_latched
                or self._manual_patrol_conflict_unlocked()
            ):
                conflict_state = True

        if conflict_state:
            result = self._release_manual_control(
                "安全状态变化，已强制退出手动模式",
                always_publish_zero=True,
                expected_owner_sid=owner_sid,
            )
            return {
                "success": False,
                "error": "manual_control_interlock",
                "message": "安全状态不允许手动控制，已发布零速",
                "state": result["state"],
            }

        with self.manual_control_lock:
            if (
                not self.manual_mode_enabled
                or self.manual_control_owner_sid != owner_sid
            ):
                return {
                    "success": False,
                    "error": "manual_control_lost",
                    "message": "手动控制权已释放",
                    "state": self._manual_state_unlocked(),
                }
            if (
                not self.safety_sync_complete
                or self.emergency_stop_latched
                or self._manual_patrol_conflict_unlocked()
            ):
                result = self._release_manual_control(
                    "安全状态变化，已强制退出手动模式",
                    always_publish_zero=True,
                    expected_owner_sid=owner_sid,
                )
                return {
                    "success": False,
                    "error": "manual_control_interlock",
                    "message": "安全状态不允许手动控制，已发布零速",
                    "state": result["state"],
                }
            linear_x, angular_z = MANUAL_DIRECTION_VELOCITIES[direction]
            publish_success, publish_error = self.publish_velocity_command(
                linear_x, angular_z
            )
            if not publish_success:
                return {
                    "success": False,
                    "error": "cmd_vel_publish_failed",
                    "message": f"速度指令发布失败：{publish_error}",
                    "state": self._manual_state_unlocked(),
                }
            direction_changed = direction != self.manual_current_direction
            self.manual_last_command_time = monotonic()
            self.manual_last_command_at = self._task_timestamp()
            self.manual_current_direction = direction
            self.manual_control_message = (
                "手动控制已停止"
                if direction == "stop"
                else f"手动控制方向：{direction}"
            )
            if direction_changed:
                self.manual_state_revision += 1
            state = self._manual_state_unlocked()

        if direction_changed:
            self.emit_manual_control_state(state)
        return {
            "success": True,
            "message": "手动控制命令已接受",
            "state": state,
        }

    def handle_manual_control_disconnect(self, owner_sid):
        with self.manual_control_lock:
            is_owner = (
                self.manual_mode_enabled
                and self.manual_control_owner_sid == owner_sid
            )
        if is_owner:
            self._release_manual_control(
                "控制页面已断开，已发布零速",
                expected_owner_sid=owner_sid,
            )

    def manual_control_watchdog(self):
        with self.manual_control_lock:
            timed_out = (
                self.manual_mode_enabled
                and self.manual_current_direction != "stop"
                and self.manual_last_command_time is not None
                and monotonic() - self.manual_last_command_time
                >= MANUAL_WATCHDOG_SECONDS
            )
            if not timed_out:
                return
            zero_success, zero_error = self.publish_zero_command()
            self.manual_last_command_time = None
            self.manual_last_command_at = self._task_timestamp()
            self.manual_current_direction = "stop"
            self.manual_control_message = (
                "手动控制超时，已自动停止"
                if zero_success
                else f"手动控制超时；零速发布失败：{zero_error}"
            )
            self.manual_state_revision += 1
            state = self._manual_state_unlocked()

        self.emit_manual_control_state(state)

    def sync_stm32_emergency_stop(self):
        if self.safety_sync_pending:
            return
        client = self.stm32_get_parameters_client
        if not client.service_is_ready():
            return
        self.safety_sync_pending = True
        request_object = GetParameters.Request()
        request_object.names = ["emergency_stop"]
        future = client.call_async(request_object)

        def apply_result(done_future):
            self.safety_sync_pending = False
            try:
                response = done_future.result()
                values = list(getattr(response, "values", []))
                if not values or values[0].type != ParameterType.PARAMETER_BOOL:
                    return
                enabled = bool(values[0].bool_value)
            except Exception as exc:
                self.get_logger().warning(
                    f"failed to read STM32 emergency_stop parameter: {exc}"
                )
                return

            task_state = self.get_patrol_task().get("state")
            if not enabled and (
                self.emergency_stop_latched
                or task_state == "emergency_stop"
            ):
                self.emergency_stop_latched = True
                self.emergency_stop_parameter_confirmed = False
                self.safety_sync_complete = True
                self.safety_message = (
                    "STM32 急停参数为 False，但 Web 急停状态仍锁定；"
                    "请执行解除急停"
                )
                self.emit_safety_state()
                self.safety_sync_timer.cancel()
                return

            self.emergency_stop_latched = enabled
            self.emergency_stop_parameter_confirmed = True
            self.safety_sync_complete = True
            self.safety_message = (
                "检测到 STM32 急停已锁定" if enabled else "急停未触发"
            )
            if enabled:
                with self.patrol_task_lock:
                    self.prepared_points_snapshot = None
                self._update_patrol_task(
                    "emergency_stop", "检测到 STM32 急停已锁定"
                )
            self.emit_safety_state()
            self.safety_sync_timer.cancel()

        future.add_done_callback(apply_result)

    def get_stm32_emergency_stop(self, timeout_sec=2.0):
        client = self.stm32_get_parameters_client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=timeout_sec
        ):
            return False, None, "STM32 参数读取服务不可用"

        request_object = GetParameters.Request()
        request_object.names = ["emergency_stop"]
        future = client.call_async(request_object)
        response, error = self._wait_for_ros_future(future, timeout_sec)
        if error:
            return False, None, error
        values = list(getattr(response, "values", []))
        if not values or values[0].type != ParameterType.PARAMETER_BOOL:
            return False, None, "emergency_stop 参数不存在或类型不是 bool"
        return True, bool(values[0].bool_value), ""

    def set_stm32_emergency_stop(self, enabled, timeout_sec=2.0):
        client = self.stm32_set_parameters_client
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=timeout_sec
        ):
            return False, False, "STM32 参数设置服务不可用"

        parameter_value = ParameterValue()
        parameter_value.type = ParameterType.PARAMETER_BOOL
        parameter_value.bool_value = bool(enabled)
        parameter = Parameter()
        parameter.name = "emergency_stop"
        parameter.value = parameter_value
        request_object = SetParameters.Request()
        request_object.parameters = [parameter]
        set_future = client.call_async(request_object)
        set_response, error = self._wait_for_ros_future(set_future, timeout_sec)
        if error:
            return False, False, error
        results = list(getattr(set_response, "results", []))
        set_success = bool(results) and all(result.successful for result in results)
        if not set_success:
            reason = "; ".join(
                result.reason for result in results if result.reason
            ) or "STM32 拒绝修改 emergency_stop 参数"
            return False, False, reason

        read_success, current_value, error = self.get_stm32_emergency_stop(
            timeout_sec=timeout_sec
        )
        if not read_success:
            return True, False, error
        confirmed = current_value == bool(enabled)
        return True, confirmed, "" if confirmed else "急停参数回读不一致"

    def get_safety_state(self):
        return {
            "emergency_stop": self.emergency_stop_latched,
            "parameter_confirmed": self.emergency_stop_parameter_confirmed,
            "sync_complete": self.safety_sync_complete,
            "message": self.safety_message,
            "updated_at": self._task_timestamp(),
        }

    def emit_safety_state(self):
        payload = self.get_safety_state()
        socketio.emit("patrol_safety_state", payload, namespace="/")
        return payload

    def engage_emergency_stop(self):
        self.emergency_stop_latched = True
        self.emergency_stop_parameter_confirmed = False
        manual_exit = self._release_manual_control(
            "急停已触发，手动控制已强制退出",
            always_publish_zero=True,
        )
        with self.patrol_task_lock:
            self.prepared_points_snapshot = None
        task = self._update_patrol_task("emergency_stop", "已进入急停锁定")

        zero_success = manual_exit["zero_success"]
        zero_error = manual_exit["zero_error"]
        patrol_state = str(self.patrol_data.get("state") or "idle")
        cancel_required = patrol_state in {
            "running", "navigating", "pausing", "paused", "returning"
        }
        cancel_success = True
        cancel_error = ""
        if cancel_required:
            cancel_payload, cancel_status = self.call_patrol_service("cancel")
            cancel_success = cancel_status == 200 and bool(
                cancel_payload.get("success")
            )
            if not cancel_success:
                cancel_error = cancel_payload.get(
                    "message", cancel_payload.get("error", "取消巡检失败")
                )

        set_success, confirmed, parameter_error = (
            self.set_stm32_emergency_stop(True)
        )
        # Keep the Web-side latch fail-safe even if an earlier asynchronous
        # startup sync completed with a stale False value during this request.
        self.emergency_stop_latched = True
        self.emergency_stop_parameter_confirmed = confirmed
        self.safety_sync_complete = set_success and confirmed
        success = zero_success and cancel_success and set_success and confirmed
        errors = {}
        if not zero_success:
            errors["publish_zero_cmd"] = zero_error
        if not cancel_success:
            errors["cancel_patrol"] = cancel_error
        if not set_success or not confirmed:
            errors["set_emergency_stop"] = parameter_error
        self.safety_message = "已急停" if success else "急停未完全确认"
        safety = self.emit_safety_state()
        return {
            "success": success,
            "state": "emergency_stop",
            "steps": {
                "publish_zero_cmd": zero_success,
                "cancel_patrol": cancel_success,
                "set_emergency_stop": set_success,
                "confirmed_emergency_stop": confirmed,
            },
            "errors": errors,
            "message": self.safety_message,
            "task": task,
            "safety": safety,
        }, 200 if success else 503

    def release_emergency_stop(self):
        task_before_release = self.get_patrol_task()
        web_emergency_state = (
            self.emergency_stop_latched
            or task_before_release.get("state") == "emergency_stop"
        )
        if not web_emergency_state:
            return {
                "success": False,
                "state": task_before_release.get("state"),
                "message": "当前未处于急停状态",
                "steps": {},
            }, 409

        manual_release = self._release_manual_control(
            "正在解除急停；手动控制保持关闭",
            always_publish_zero=True,
        )
        zero_success = manual_release["zero_success"]
        zero_error = manual_release["zero_error"]
        set_success, confirmed, parameter_error = (
            self.set_stm32_emergency_stop(False)
        )
        self.emergency_stop_parameter_confirmed = confirmed
        self.safety_sync_complete = set_success and confirmed
        success = zero_success and set_success and confirmed
        errors = {}
        if not zero_success:
            errors["publish_zero_cmd"] = zero_error
        if not set_success or not confirmed:
            errors["set_emergency_stop"] = parameter_error

        if success:
            self.emergency_stop_latched = False
            self.emergency_stop_parameter_confirmed = True
            with self.patrol_task_lock:
                self.prepared_points_snapshot = None
            has_previous_task = bool(
                task_before_release.get("task_id")
                or task_before_release.get("map_name")
                or int(task_before_release.get("point_count") or 0) > 0
                or self.patrol_data.get("session_id")
            )
            if has_previous_task:
                self.safety_message = (
                    "急停已解除，任务已取消，请重新发送巡检任务"
                )
                task = self._update_patrol_task(
                    "invalidated", self.safety_message
                )
            else:
                self.safety_message = "急停已解除，任务保持未运行状态"
                task = self._update_patrol_task(
                    "draft",
                    self.safety_message,
                    task_id="",
                    map_name=None,
                    map_fingerprint=None,
                    points_hash="",
                    point_count=0,
                    prepared_at="",
                )
            with self.manual_control_lock:
                self.manual_mode_enabled = False
                self.manual_control_owner_sid = None
                self.manual_last_command_time = None
                self.manual_last_command_at = None
                self.manual_current_direction = "stop"
                self.manual_control_message = (
                    "急停已解除，保持自动模式和停止状态"
                )
                self.manual_state_revision += 1
                manual_state = self._manual_state_unlocked()
        else:
            self.emergency_stop_latched = True
            self.safety_message = "解除急停失败，继续保持锁定"
            task = self._update_patrol_task(
                "emergency_stop", self.safety_message
            )
            with self.manual_control_lock:
                self.manual_mode_enabled = False
                self.manual_control_owner_sid = None
                self.manual_last_command_time = None
                self.manual_last_command_at = None
                self.manual_current_direction = "stop"
                self.manual_control_message = (
                    "解除急停失败，手动控制保持关闭"
                )
                self.manual_state_revision += 1
                manual_state = self._manual_state_unlocked()
        self.emit_manual_control_state(manual_state)
        safety = self.emit_safety_state()
        return {
            "success": success,
            "state": task.get("state"),
            "steps": {
                "publish_zero_cmd": zero_success,
                "set_emergency_stop": set_success,
                "confirmed_emergency_stop_released": confirmed,
            },
            "errors": errors,
            "message": self.safety_message,
            "task": task,
            "safety": safety,
        }, 200 if success else 503

    # ================= ͼ��ص�?=================
    def patrol_status_cb(self, message):
        """Forward validated structured patrol status to Web clients."""
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("status payload is not an object")
            state = str(payload.get("state") or "")
            if state not in PATROL_STATES:
                raise ValueError(f"unknown patrol state: {state}")
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().warn(f"invalid patrol status ignored: {exc}")
            return

        self.patrol_data = payload
        if state in PATROL_ACTIVE_STATES:
            self._release_manual_control(
                "巡检状态已进入执行中，手动控制已强制退出"
            )
        self.patrol_status = PATROL_STATES[state]
        if not self.emergency_stop_latched:
            if state in PATROL_ACTIVE_STATES:
                self._update_patrol_task(
                    "active", payload.get("message") or PATROL_STATES[state]
                )
            elif state == "completed":
                self._update_patrol_task(
                    "finished", payload.get("message") or "巡检完成"
                )
            elif state == "canceled":
                self._update_patrol_task(
                    "canceled", payload.get("message") or "巡检已取消"
                )
            elif state == "error":
                self._update_patrol_task(
                    "finished", payload.get("message") or "导航失败"
                )
        socketio.emit("patrol_status", payload, namespace="/")

    def call_patrol_service(self, command, timeout_sec=2.0):
        """Call one patrol Trigger service from a Flask request thread."""
        client = self.patrol_clients.get(command)
        if client is None:
            return {"error": "unknown patrol command"}, 404
        if not client.service_is_ready():
            return {"error": "patrol service is unavailable"}, 503

        future = client.call_async(Trigger.Request())
        completed = threading.Event()
        result_holder = {}

        def store_result(done_future):
            try:
                result_holder["response"] = done_future.result()
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                completed.set()

        future.add_done_callback(store_result)
        if not completed.wait(timeout_sec):
            future.cancel()
            return {"error": "patrol service timed out"}, 504
        if "error" in result_holder:
            return {"error": result_holder["error"]}, 503

        service_response = result_holder["response"]
        payload = {
            "command": command,
            "success": bool(service_response.success),
            "message": service_response.message,
        }
        return payload, 200 if service_response.success else 409

    def image_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    # ================= ������Ƶ =================
    def push_frame(self):
        if self.latest_frame is None:
            return

        try:
            frame = self.latest_frame
            height, width = frame.shape[:2]
            if width != self.web_frame_width or height != self.web_frame_height:
                frame = cv2.resize(
                    frame,
                    (self.web_frame_width, self.web_frame_height),
                    interpolation=cv2.INTER_AREA
                )

            # ѹ��ͼƬ���ؼ���
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.web_jpeg_quality]
            )

            jpg_base64 = base64.b64encode(buffer).decode('utf-8')

            # ?? ע�⣺�°� socketio ��Ҫ broadcast
            socketio.emit(
                'camera',
                jpg_base64,
                namespace='/'
            )

        except Exception as e:
            self.get_logger().error(f"camera emit error: {e}")

    # ================= LOG =================
    # ================= LOG =================
    def log_cb(self, msg):
        data = msg.data
        level = "normal"
        
        if "|" in data:
            level, text = data.split("|", 1)
        else:
            text = data

        try:
            alarm_event = self.build_alarm_event(text, level)
            socketio.emit(
                'log',
                {
                    "text": text,
                    "level": level,
                    "has_alarm_event": alarm_event is not None
                },  # ���� JSON
                namespace='/'
            )
            if alarm_event is not None:
                self.emit_alarm_event(alarm_event)
        except Exception as e:
            self.get_logger().error(f"log emit error: {e}")

    def vision_alarm_cb(self, msg, alarm_type, level):
        text = (msg.data or "").strip()
        if not text:
            return

        # TODO: add same-type alarm cooldown to avoid excessive repeated logs.
        if alarm_type not in text.lower():
            text = f"{alarm_type}: {text}"

        try:
            alarm_event = self.build_alarm_event(
                text,
                level,
                position="camera_view"
            )
            if alarm_event is not None:
                self.emit_alarm_event(alarm_event)
        except Exception as e:
            self.get_logger().error(f"vision alarm emit error: {e}")

    def emit_alarm_event(self, alarm_event):
        if self.is_alarm_in_cooldown(alarm_event):
            return

        alarm_event.pop("_cooldown_key", None)
        alarm_event.pop("_cooldown_seconds", None)
        self.attach_alarm_evidence(alarm_event)
        append_alarm_log(alarm_event)
        socketio.emit(
            'alarm_event',
            alarm_event,
            namespace='/'
        )

    def is_alarm_in_cooldown(self, alarm_event):
        alarm_type = alarm_event.get("type") or "unknown"
        position = alarm_event.get("position") or "unknown"
        cooldown_seconds = float(
            alarm_event.get(
                "_cooldown_seconds",
                ALARM_COOLDOWN_SECONDS.get(alarm_type, 30.0)
            )
        )
        cooldown_key = alarm_event.get("_cooldown_key") or (alarm_type, position)
        now = time()
        last_time = self.last_alarm_times.get(cooldown_key)

        if last_time is not None and now - last_time < cooldown_seconds:
            return True

        self.last_alarm_times[cooldown_key] = now
        return False

    def clear_alarm_cooldowns(self):
        self.last_alarm_times.clear()

    def attach_alarm_evidence(self, alarm_event):
        if self.latest_frame is None:
            return

        alarm_id = alarm_event.get("id")
        if not alarm_id:
            return

        try:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            frame = self.latest_frame.copy()
            filename = f"{alarm_id}.jpg"
            evidence_path = EVIDENCE_DIR / filename
            ok = cv2.imwrite(
                str(evidence_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if ok:
                alarm_event["evidence_image"] = f"/api/alarm-evidence/{filename}"
        except Exception as e:
            self.get_logger().error(f"save alarm evidence error: {e}")

    def build_alarm_event(self, text, level, position=None):
        lower_text = text.lower()
        alarm_type = None

        if "person" in lower_text:
            alarm_type = "person"
        elif "fire" in lower_text:
            alarm_type = "fire"
        elif "accident" in lower_text:
            alarm_type = "accident"
        elif "water" in lower_text:
            alarm_type = "water"
        elif "congestion" in lower_text:
            alarm_type = "congestion"

        if alarm_type is None:
            return None

        self.alarm_seq += 1
        alarm_level = "高" if level == "danger" else "中"
        if alarm_type in ("fire", "accident"):
            alarm_level = "高"

        now = datetime.now()
        return {
            "id": f"alarm-{now.strftime('%Y%m%d%H%M%S%f')}-{self.alarm_seq}",
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "type": alarm_type,
            "level": alarm_level,
            "position": position if position is not None else self.extract_position(text),
            "status": "未处理",
            "sensor": self.sensor_summary(),
            "text": text
        }

    def build_sensor_alarm_event(self, alert):
        self.alarm_seq += 1
        now = datetime.now()
        level = "高" if alert["level"] == "danger" else "中"
        return {
            "id": f"alarm-{now.strftime('%Y%m%d%H%M%S%f')}-{self.alarm_seq}",
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "sensor",
            "level": level,
            "position": alert["position"],
            "status": "未处理",
            "sensor": self.sensor_summary(),
            "sensor_snapshot": self.sensor_snapshot(),
            "text": alert["text"],
            "_cooldown_key": ("sensor", alert["cooldown_key"]),
            "_cooldown_seconds": alert["cooldown_seconds"],
        }

    def extract_position(self, text):
        if "x:" not in text or "y:" not in text:
            return "位置未知"

        try:
            x_part = text.split("x:", 1)[1].split(",", 1)[0].strip()
            y_part = text.split("y:", 1)[1].split()[0].strip()
            return f"x={float(x_part):.2f}, y={float(y_part):.2f}"
        except Exception:
            return "位置未知"

    def sensor_summary(self):
        d = self.sensor_data
        return (
            f"T {d['temp']:.1f}C / H {d['hum']:.1f}% / "
            f"MQ2 {d['mq2']:.1f} / CO {d['co']:.1f}ppm / "
            f"PM2.5 {d['pm25']:.1f} / PM10 {d['pm10']:.1f} / "
            f"HC {d['hc']:.1f}cm"
        )

    def sensor_snapshot(self):
        return {
            "temp": float(self.sensor_data["temp"]),
            "hum": float(self.sensor_data["hum"]),
            "mq2": float(self.sensor_data["mq2"]),
            "co": float(self.sensor_data["co"]),
            "pm25": float(self.sensor_data["pm25"]),
            "pm10": float(self.sensor_data["pm10"]),
            "hc": float(self.sensor_data["hc"]),
        }
    
    def temp_cb(self, msg):
        self.sensor_data["temp"] = msg.data

    def hum_cb(self, msg):
        self.sensor_data["hum"] = msg.data

    def mq2_cb(self, msg):
        self.sensor_data["mq2"] = msg.data

    def pm25_cb(self, msg):
        self.sensor_data["pm25"] = msg.data

    def pm10_cb(self, msg):
        self.sensor_data["pm10"] = msg.data

    def hc_cb(self, msg):
        self.sensor_data["hc"] = msg.data

    def co_cb(self, msg):
        self.sensor_data["co"] = msg.data

    def evaluate_sensor_alerts(self):
        d = self.sensor_data
        alerts = []
        level = "normal"
        sensor_alarm_alerts = []

        def add_alert(key, level_name, msg, text, position, cooldown_key,
                      cooldown_seconds=30.0):
            nonlocal level
            alerts.append({"key": key, "msg": msg})
            sensor_alarm_alerts.append({
                "key": key,
                "level": level_name,
                "text": text,
                "position": position,
                "cooldown_key": cooldown_key,
                "cooldown_seconds": cooldown_seconds,
            })
            if level_name == "danger":
                level = "danger"
            elif level != "danger":
                level = "warning"

        thresholds = SENSOR_THRESHOLDS

        if d["co"] >= thresholds["co_danger"]:
            add_alert(
                "co",
                "danger",
                "CO DANGER",
                f"CO 危险：当前 {d['co']:.1f} ppm",
                "传感器:CO",
                "co_danger",
                60.0,
            )
        elif d["co"] >= thresholds["co_warn"]:
            add_alert(
                "co",
                "warning",
                "CO HIGH",
                f"CO 预警：当前 {d['co']:.1f} ppm",
                "传感器:CO",
                "co_warn",
                30.0,
            )

        if d["mq2"] >= thresholds["mq2_danger"]:
            add_alert(
                "mq2",
                "danger",
                "MQ2 DANGER",
                f"MQ2 烟雾危险：当前 {d['mq2']:.1f}",
                "传感器:MQ2",
                "mq2_danger",
                60.0,
            )
        elif d["mq2"] >= thresholds["mq2_warn"]:
            add_alert(
                "mq2",
                "warning",
                "MQ2 HIGH",
                f"MQ2 烟雾预警：当前 {d['mq2']:.1f}",
                "传感器:MQ2",
                "mq2_warn",
                30.0,
            )

        if d["temp"] >= thresholds["temp_danger"]:
            add_alert(
                "temp",
                "danger",
                "Temperature DANGER",
                f"温度危险：当前 {d['temp']:.1f} ℃",
                "传感器:温度",
                "temp_danger",
                30.0,
            )
        elif d["temp"] >= thresholds["temp_warn"]:
            add_alert(
                "temp",
                "warning",
                "Temperature HIGH",
                f"温度预警：当前 {d['temp']:.1f} ℃",
                "传感器:温度",
                "temp_warn",
                30.0,
            )

        if d["pm25"] >= thresholds["pm25_danger"]:
            add_alert(
                "pm25",
                "danger",
                "PM2.5 DANGER",
                f"PM2.5 危险：当前 {d['pm25']:.1f}",
                "传感器:PM2.5",
                "pm25_danger",
                30.0,
            )
        elif d["pm25"] >= thresholds["pm25_warn"]:
            add_alert(
                "pm25",
                "warning",
                "PM2.5 HIGH",
                f"PM2.5 预警：当前 {d['pm25']:.1f}",
                "传感器:PM2.5",
                "pm25_warn",
                30.0,
            )

        if d["pm10"] >= thresholds["pm10_danger"]:
            add_alert(
                "pm10",
                "danger",
                "PM10 DANGER",
                f"PM10 危险：当前 {d['pm10']:.1f}",
                "传感器:PM10",
                "pm10_danger",
                30.0,
            )
        elif d["pm10"] >= thresholds["pm10_warn"]:
            add_alert(
                "pm10",
                "warning",
                "PM10 HIGH",
                f"PM10 预警：当前 {d['pm10']:.1f}",
                "传感器:PM10",
                "pm10_warn",
                30.0,
            )

        if d["hc"] <= thresholds["hc_low_danger"]:
            add_alert(
                "hc",
                "danger",
                "HC LOW DANGER",
                f"测距状态过低危险：当前 {d['hc']:.1f} cm",
                "传感器:测距状态过低",
                "hc_low_danger",
                60.0,
            )
        elif d["hc"] <= thresholds["hc_low_warn"]:
            add_alert(
                "hc",
                "warning",
                "HC LOW WARNING",
                f"测距状态过低预警：当前 {d['hc']:.1f} cm",
                "传感器:测距状态过低",
                "hc_low_warn",
                30.0,
            )
        elif d["hc"] >= thresholds["hc_high_danger"]:
            add_alert(
                "hc",
                "danger",
                "HC HIGH DANGER",
                f"测距状态过高危险：当前 {d['hc']:.1f} cm",
                "传感器:测距状态过高",
                "hc_high_danger",
                60.0,
            )
        elif d["hc"] >= thresholds["hc_high_warn"]:
            add_alert(
                "hc",
                "warning",
                "HC HIGH WARNING",
                f"测距状态过高预警：当前 {d['hc']:.1f} cm",
                "传感器:测距状态过高",
                "hc_high_warn",
                30.0,
            )

        return level, alerts, sensor_alarm_alerts

    def push_sensor(self):

        d = self.sensor_data
        level, alerts, sensor_alarm_alerts = self.evaluate_sensor_alerts()

        for alert in sensor_alarm_alerts:
            self.emit_alarm_event(self.build_sensor_alarm_event(alert))

        socketio.emit("sensor", d)

        socketio.emit("sensor_alert", {
            "level": level,
            "alerts": alerts
        })

        socketio.emit("robot_status", {
            "patrol_status": self.patrol_status,
            "patrol": self.patrol_data,
            "sensor_level": level,
            "battery": None
        })

# ================= ROS�߳� =================
def ros_spin(node):
    rclpy.spin(node)


# ================= MAIN =================
def main():
    global web_bridge_node_instance
    rclpy.init()

    node = WebBridge()
    web_bridge_node_instance = node

    t = threading.Thread(
        target=ros_spin,
        args=(node,),
        daemon=True
    )
    t.start()

    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    main()
