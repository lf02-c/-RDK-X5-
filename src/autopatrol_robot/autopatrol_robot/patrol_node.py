#!/usr/bin/env python3
"""Coordinate Web-managed patrol points through Nav2 NavigateToPose."""

import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import uuid

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


ACTIVE_STATES = {
    "running",
    "navigating",
    "pausing",
    "canceling",
    "returning",
}
NAV2_NO_PATH_MARKERS = (
    "no path",
    "no valid path",
    "failed to create plan",
    "planning failed",
    "planner failed",
    "无法规划",
    "无可行路径",
)


def find_source_project_root():
    """Find the LD project root when running from a source workspace."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "web" / "index.html").exists():
            return parent
    return None


def get_installed_share_dir():
    """Return the installed package share directory when available."""
    if get_package_share_directory is None:
        return None
    try:
        return Path(get_package_share_directory("autopatrol_robot"))
    except Exception:
        return None


def resolve_data_dir():
    """Resolve the same persistent data directory used by web_bridge_node."""
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


def resolve_patrol_points_file(parameter_value=""):
    """Prefer an explicit parameter, otherwise use the Web data path."""
    configured_path = str(parameter_value or "").strip()
    if configured_path:
        return Path(configured_path).expanduser()
    return resolve_data_dir() / "patrol_points.json"


def load_patrol_points(path):
    """Load and validate one immutable patrol-point snapshot."""
    points_path = Path(path).expanduser()
    try:
        file_content = points_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"巡检点文件不存在：{points_path}") from exc
    except OSError as exc:
        raise ValueError(f"巡检点文件读取失败：{exc}") from exc

    if not file_content.strip():
        raise ValueError("巡检点为空 (patrol points file is empty)")

    try:
        raw_points = json.loads(file_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"巡检点文件解析失败：{exc.msg}") from exc

    if not isinstance(raw_points, list):
        raise ValueError("巡检点文件解析失败：根节点必须是数组 (array)")
    if not raw_points:
        raise ValueError("巡检点为空 (patrol points list is empty)")

    normalized = []
    seen_ids = set()
    for index, raw_point in enumerate(raw_points, start=1):
        if not isinstance(raw_point, dict):
            raise ValueError(f"patrol point {index} must be an object")

        point_id = str(raw_point.get("id") or "").strip()
        if not point_id:
            raise ValueError(f"patrol point {index} has no id")
        if point_id in seen_ids:
            raise ValueError(f"duplicate patrol point id: {point_id}")

        coordinates = {}
        for key in ("x", "y", "yaw"):
            try:
                value = float(raw_point[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"patrol point {point_id} has invalid {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"patrol point {point_id} has non-finite {key}")
            coordinates[key] = value

        seen_ids.add(point_id)
        normalized.append({
            "id": point_id,
            "name": str(raw_point.get("name") or point_id).strip(),
            "type": str(raw_point.get("type") or "manual").strip(),
            "x": coordinates["x"],
            "y": coordinates["y"],
            "yaw": coordinates["yaw"],
            "note": str(raw_point.get("note") or "").strip(),
            "created_at": str(raw_point.get("created_at") or "").strip(),
        })

    return normalized


def quaternion_from_yaw(yaw):
    """Return planar quaternion components for a yaw angle in radians."""
    return math.sin(yaw * 0.5), math.cos(yaw * 0.5)


class PatrolNode(Node):
    """Expose patrol services and serialize patrol goals through Nav2."""

    def __init__(self):
        super().__init__("patrol_node")

        default_patrol_points_file = str(resolve_patrol_points_file())
        self.declare_parameter(
            "patrol_points_file", default_patrol_points_file
        )
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("amcl_pose_topic", "/amcl_pose")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")
        self.declare_parameter("pose_max_age_sec", 2.0)
        self.declare_parameter("enable_navigation", True)

        self.patrol_points_file = str(
            resolve_patrol_points_file(
                self.get_parameter("patrol_points_file").value
            )
        )
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.amcl_pose_topic = str(
            self.get_parameter("amcl_pose_topic").value
        )
        self.navigate_action_name = str(
            self.get_parameter("navigate_action_name").value
        )
        self.pose_max_age_sec = max(
            0.0, float(self.get_parameter("pose_max_age_sec").value)
        )
        self.enable_navigation = bool(
            self.get_parameter("enable_navigation").value
        )

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(
            String, "/patrol/status", status_qos
        )
        self.log_pub = self.create_publisher(String, "/patrol_log", 10)

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.pose_callback,
            10,
        )

        self.action_client = ActionClient(
            self, NavigateToPose, self.navigate_action_name
        )
        self.create_service(Trigger, "/patrol/start", self.start_callback)
        self.create_service(Trigger, "/patrol/pause", self.pause_callback)
        self.create_service(Trigger, "/patrol/resume", self.resume_callback)
        self.create_service(Trigger, "/patrol/cancel", self.cancel_callback)
        self.create_service(
            Trigger, "/patrol/return_home", self.return_home_callback
        )

        self.latest_pose = None
        self.latest_pose_received_at = None
        self.home_pose = None
        self.session_id = ""
        self.state = "idle"
        self.message = "等待巡检任务"
        self.points = []
        self.current_index = 0
        self.completed_point_ids = []
        self.failed_points = []
        self.current_goal_handle = None
        self.active_goal_token = None
        self.active_target_kind = None
        self.goal_sequence = 0
        self.pending_cancel_intent = None

        self.publish_status("idle", "等待巡检任务")
        self.get_logger().info(
            "Patrol coordinator ready: "
            f"points={self.patrol_points_file}, "
            f"action={self.navigate_action_name}"
        )

    def pose_callback(self, message):
        """Keep the latest AMCL pose for the next mission home position."""
        self.latest_pose = copy.deepcopy(message.pose.pose)
        self.latest_pose_received_at = self.get_clock().now()

    def start_callback(self, _request, response):
        """Load a point snapshot and start one patrol round."""
        if self.state in ACTIVE_STATES or self.state == "paused":
            return self._reject(response, f"patrol is already {self.state}")
        if not self.enable_navigation:
            return self._reject(response, "navigation is disabled")

        try:
            points = load_patrol_points(self.patrol_points_file)
        except ValueError as exc:
            return self._reject(response, str(exc))

        if not self.action_client.server_is_ready():
            return self._reject(response, "Nav2 NavigateToPose is not ready")

        pose_error = self._pose_validation_error()
        if pose_error:
            return self._reject(response, pose_error)

        self._invalidate_active_goal()
        self.session_id = self._new_session_id()
        self.points = points
        self.current_index = 0
        self.completed_point_ids = []
        self.failed_points = []
        self.home_pose = self._pose_stamped_from_amcl()
        self.publish_status("running", f"已加载 {len(points)} 个巡检点")
        self.log(f"Patrol started: {self.session_id}, points={len(points)}")
        self._dispatch_current_point()
        response.success = True
        response.message = "patrol start accepted"
        return response

    def pause_callback(self, _request, response):
        """Cancel the active goal and retain the unfinished point."""
        if self.state not in {"running", "navigating"}:
            return self._reject(response, f"cannot pause from {self.state}")

        self._begin_cancel("pause")
        response.success = True
        response.message = "patrol pause accepted"
        return response

    def resume_callback(self, _request, response):
        """Resend the point that was unfinished when patrol paused."""
        if self.state != "paused":
            return self._reject(response, f"cannot resume from {self.state}")
        if not self.action_client.server_is_ready():
            return self._reject(response, "Nav2 NavigateToPose is not ready")

        self.publish_status("running", "正在恢复当前巡检点")
        self._dispatch_current_point()
        response.success = True
        response.message = "patrol resume accepted"
        return response

    def cancel_callback(self, _request, response):
        """Cancel the active patrol without returning home."""
        if self.state == "paused":
            self.publish_status("canceled", "巡检任务已取消")
            self.log("Patrol canceled while paused")
        elif self.state in {"running", "navigating", "returning"}:
            self._begin_cancel("cancel")
        else:
            return self._reject(response, f"cannot cancel from {self.state}")

        response.success = True
        response.message = "patrol cancel accepted"
        return response

    def return_home_callback(self, _request, response):
        """Cancel any active goal and navigate to the mission start pose."""
        if self.home_pose is None:
            return self._reject(response, "no mission start pose is available")
        if self.state in {"pausing", "canceling", "returning"}:
            return self._reject(response, f"cannot return home from {self.state}")
        if not self.enable_navigation:
            return self._reject(response, "navigation is disabled")
        if not self.action_client.server_is_ready():
            return self._reject(response, "Nav2 NavigateToPose is not ready")

        if self.state in {"running", "navigating"}:
            self._begin_cancel("return_home")
        else:
            self._dispatch_home_goal()

        response.success = True
        response.message = "return home accepted"
        return response

    def _pose_validation_error(self):
        if self.latest_pose is None or self.latest_pose_received_at is None:
            return "no AMCL pose is available"
        if self.pose_max_age_sec <= 0.0:
            return None

        age = (
            self.get_clock().now() - self.latest_pose_received_at
        ).nanoseconds / 1_000_000_000.0
        if age > self.pose_max_age_sec:
            return f"AMCL pose is stale ({age:.1f}s)"
        return None

    def _pose_stamped_from_amcl(self):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose = copy.deepcopy(self.latest_pose)
        return pose

    def _pose_stamped_from_point(self, point):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = point["x"]
        pose.pose.position.y = point["y"]
        pose.pose.position.z = 0.0
        z, w = quaternion_from_yaw(point["yaw"])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        return pose

    def _dispatch_current_point(self):
        if self.current_index >= len(self.points):
            self.publish_status("completed", "单轮巡检已完成")
            self.log("Patrol completed")
            return

        point = self.points[self.current_index]
        self._send_goal(
            self._pose_stamped_from_point(point),
            target_kind="patrol",
        )

    def _dispatch_home_goal(self):
        self.publish_status("running", "正在发送返回起点目标")
        home_pose = copy.deepcopy(self.home_pose)
        home_pose.header.stamp = self.get_clock().now().to_msg()
        self._send_goal(home_pose, target_kind="home")

    def _send_goal(self, pose, target_kind):
        self.goal_sequence += 1
        token = (self.session_id, self.goal_sequence)
        self.active_goal_token = token
        self.active_target_kind = target_kind
        self.current_goal_handle = None

        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done, goal_token=token: self._goal_response(done, goal_token)
        )

    def _goal_response(self, future, token):
        if token != self.active_goal_token:
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._fail_active_goal(f"failed to send Nav2 goal: {exc}")
            return

        if not goal_handle.accepted:
            self._fail_active_goal(
                "Nav2 拒绝目标，当前地图或禁区约束下无可行路径，"
                "巡检任务已停止",
                code="nav2_goal_rejected",
            )
            return

        self.current_goal_handle = goal_handle
        if self.active_target_kind == "home":
            self.publish_status("returning", "Nav2 已接受返回起点目标")
        elif self.pending_cancel_intent == "pause":
            self.publish_status("pausing", "正在暂停巡检")
        elif self.pending_cancel_intent is not None:
            self.publish_status("canceling", "正在取消当前导航目标")
        else:
            point = self.points[self.current_index]
            self.publish_status(
                "navigating", f"正在前往 {point['name']}"
            )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, goal_token=token: self._goal_result(done, goal_token)
        )

        if self.pending_cancel_intent is not None:
            self._cancel_current_goal(token)

    def _goal_result(self, future, token):
        if token != self.active_goal_token:
            return
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
            result = getattr(wrapped_result, "result", None)
        except Exception as exc:
            self._fail_active_goal(
                f"Nav2 结果读取失败，巡检任务已停止：{exc}",
                code="nav2_navigation_failed",
            )
            return

        target_kind = self.active_target_kind
        cancel_intent = self.pending_cancel_intent
        self.current_goal_handle = None
        self.active_goal_token = None
        self.active_target_kind = None
        self.pending_cancel_intent = None

        if cancel_intent is not None:
            if status == GoalStatus.STATUS_SUCCEEDED and target_kind == "patrol":
                self._record_current_point_arrival()
            elif status not in {
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_SUCCEEDED,
            }:
                self._record_failure(
                    f"Nav2 ended with status {status}", target_kind
                )
                self.publish_status("error", "取消过程中导航异常结束")
                return
            self._finish_cancel_intent(cancel_intent)
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            if target_kind == "home":
                self.publish_status("arrived", "已返回任务起点")
                self.log("Returned to mission start pose")
                return

            self._record_current_point_arrival()
            self._dispatch_current_point()
        elif status == GoalStatus.STATUS_CANCELED:
            self.publish_status("canceled", "导航目标被外部取消")
            self.log("Nav2 goal canceled externally")
        else:
            failure_code, failure_message = self._nav2_failure_details(
                status,
                result,
            )
            self._record_failure(
                failure_message,
                target_kind,
                code=failure_code,
            )
            self.publish_status("error", failure_message)
            self.log(f"Nav2 goal failed: {failure_message}")

    def _record_current_point_arrival(self):
        if self.current_index >= len(self.points):
            return
        point = self.points[self.current_index]
        if point["id"] not in self.completed_point_ids:
            self.completed_point_ids.append(point["id"])
        self.publish_status("arrived", f"已到达 {point['name']}")
        self.log(f"Arrived at patrol point: {point['id']}")
        self.current_index += 1

    @staticmethod
    def _nav2_failure_details(status, result):
        error_code = getattr(result, "error_code", None)
        error_message = str(getattr(result, "error_msg", "") or "").strip()
        normalized_message = error_message.lower()
        no_path = any(
            marker in normalized_message
            for marker in NAV2_NO_PATH_MARKERS
        )
        if no_path:
            message = "当前地图与禁区约束下无可行路径，巡检任务已停止"
            if error_message:
                message = f"{message}：{error_message}"
            return "nav2_no_feasible_path", message

        detail_parts = []
        if error_code not in (None, 0):
            detail_parts.append(f"error_code={error_code}")
        if error_message:
            detail_parts.append(error_message)
        if not detail_parts:
            detail_parts.append(f"status={status}")
        return (
            "nav2_navigation_failed",
            "Nav2 规划或导航失败，当前巡检任务已停止："
            + "，".join(detail_parts),
        )

    def _record_failure(
        self,
        reason,
        target_kind=None,
        code="nav2_navigation_failed",
    ):
        goal_kind = target_kind or self.active_target_kind
        if goal_kind != "patrol":
            return
        if self.current_index >= len(self.points):
            return
        point_id = self.points[self.current_index]["id"]
        self.failed_points.append({
            "id": point_id,
            "code": code,
            "reason": reason,
        })

    def _begin_cancel(self, intent):
        self.pending_cancel_intent = intent
        if intent == "pause":
            self.publish_status("pausing", "正在暂停巡检")
        else:
            self.publish_status("canceling", "正在取消当前导航目标")

        if self.current_goal_handle is not None:
            self._cancel_current_goal(self.active_goal_token)

    def _cancel_current_goal(self, token):
        if self.current_goal_handle is None or token is None:
            return
        future = self.current_goal_handle.cancel_goal_async()
        future.add_done_callback(
            lambda done, goal_token=token: self._cancel_response(
                done, goal_token
            )
        )

    def _cancel_response(self, future, token):
        if token != self.active_goal_token:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._cancel_request_failed(
                f"failed to cancel Nav2 goal: {exc}"
            )
            return
        if not response.goals_canceling:
            self._cancel_request_failed("Nav2 refused to cancel the goal")

    def _cancel_request_failed(self, reason):
        self.pending_cancel_intent = None
        fallback_state = (
            "returning"
            if self.active_target_kind == "home"
            else "navigating"
        )
        self.publish_status(fallback_state, reason)
        self.log(f"Patrol cancel warning: {reason}")

    def _finish_cancel_intent(self, intent):
        if intent == "pause":
            if self.current_index >= len(self.points):
                self.publish_status("completed", "单轮巡检已完成")
            else:
                self.publish_status("paused", "巡检已暂停")
            self.log("Patrol paused")
        elif intent == "return_home":
            self._dispatch_home_goal()
        else:
            self.publish_status("canceled", "巡检任务已取消")
            self.log("Patrol canceled")

    def _fail_active_goal(self, reason, code="nav2_navigation_failed"):
        self._record_failure(reason, code=code)
        self._invalidate_active_goal()
        self.publish_status("error", reason)
        self.log(f"Patrol error: {reason}")

    def _invalidate_active_goal(self):
        self.goal_sequence += 1
        self.current_goal_handle = None
        self.active_goal_token = None
        self.active_target_kind = None
        self.pending_cancel_intent = None

    def publish_status(self, state, message=""):
        """Publish the latest structured status for Web consumers."""
        self.state = state
        self.message = message
        payload = {
            "version": 1,
            "session_id": self.session_id,
            "state": self.state,
            "current_point": self._current_point_payload(),
            "completed_point_ids": list(self.completed_point_ids),
            "failed_points": copy.deepcopy(self.failed_points),
            "message": self.message,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
        message_object = String()
        message_object.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(message_object)

    def _current_point_payload(self):
        if self.current_index >= len(self.points):
            return None
        point = self.points[self.current_index]
        return {
            "id": point["id"],
            "name": point["name"],
            "index": self.current_index + 1,
            "total": len(self.points),
            "x": point["x"],
            "y": point["y"],
            "yaw": point["yaw"],
        }

    def log(self, text):
        message = String()
        message.data = text
        self.log_pub.publish(message)

    @staticmethod
    def _new_session_id():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"patrol-{timestamp}-{uuid.uuid4().hex[:6]}"

    @staticmethod
    def _reject(response, message):
        response.success = False
        response.message = message
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Patrol coordinator interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
