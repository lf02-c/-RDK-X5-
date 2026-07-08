"""Unit tests for patrol point validation and status serialization."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def _load_patrol_module():
    class DummyNode:
        pass

    class DummyMessage:
        def __init__(self, **kwargs):
            self.data = ""
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummyQoS:
        def __init__(self, depth=1):
            self.depth = depth
            self.reliability = None
            self.durability = None

    class DummyNavigateToPose:
        class Goal:
            def __init__(self):
                self.pose = None

    class DummyTrigger:
        class Request:
            pass

    class DummyGoalStatus:
        STATUS_CANCELED = 5
        STATUS_SUCCEEDED = 4

    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy_action = types.ModuleType("rclpy.action")
    fake_rclpy_action.ActionClient = object
    fake_rclpy_node = types.ModuleType("rclpy.node")
    fake_rclpy_node.Node = DummyNode
    fake_rclpy_qos = types.ModuleType("rclpy.qos")
    fake_rclpy_qos.DurabilityPolicy = types.SimpleNamespace(
        TRANSIENT_LOCAL="transient"
    )
    fake_rclpy_qos.QoSProfile = DummyQoS
    fake_rclpy_qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="reliable"
    )

    fake_action_msgs = types.ModuleType("action_msgs")
    fake_action_msgs_msg = types.ModuleType("action_msgs.msg")
    fake_action_msgs_msg.GoalStatus = DummyGoalStatus

    fake_geometry = types.ModuleType("geometry_msgs")
    fake_geometry_msg = types.ModuleType("geometry_msgs.msg")
    fake_geometry_msg.PoseStamped = DummyMessage
    fake_geometry_msg.PoseWithCovarianceStamped = DummyMessage

    fake_nav2 = types.ModuleType("nav2_msgs")
    fake_nav2_action = types.ModuleType("nav2_msgs.action")
    fake_nav2_action.NavigateToPose = DummyNavigateToPose

    fake_std = types.ModuleType("std_msgs")
    fake_std_msg = types.ModuleType("std_msgs.msg")
    fake_std_msg.String = DummyMessage

    fake_std_srvs = types.ModuleType("std_srvs")
    fake_std_srvs_srv = types.ModuleType("std_srvs.srv")
    fake_std_srvs_srv.Trigger = DummyTrigger

    stubs = {
        "rclpy": fake_rclpy,
        "rclpy.action": fake_rclpy_action,
        "rclpy.node": fake_rclpy_node,
        "rclpy.qos": fake_rclpy_qos,
        "action_msgs": fake_action_msgs,
        "action_msgs.msg": fake_action_msgs_msg,
        "geometry_msgs": fake_geometry,
        "geometry_msgs.msg": fake_geometry_msg,
        "nav2_msgs": fake_nav2,
        "nav2_msgs.action": fake_nav2_action,
        "std_msgs": fake_std,
        "std_msgs.msg": fake_std_msg,
        "std_srvs": fake_std_srvs,
        "std_srvs.srv": fake_std_srvs_srv,
    }

    module_path = (
        Path(__file__).parents[1]
        / "autopatrol_robot"
        / "patrol_node.py"
    )
    spec = importlib.util.spec_from_file_location(
        "patrol_node_tested", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


PATROL = _load_patrol_module()


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class PatrolPointValidationTests(unittest.TestCase):
    def _write_json(self, value):
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with handle:
            json.dump(value, handle, ensure_ascii=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_valid_points_preserve_order_and_metadata(self):
        path = self._write_json([
            {
                "id": "p1",
                "name": "入口",
                "type": "normal",
                "x": "1.25",
                "y": 2,
                "yaw": "1.57",
                "note": "first",
                "created_at": "2026-06-20 10:00:00",
            },
            {
                "id": "p2",
                "x": 3,
                "y": 4,
                "yaw": 0,
            },
        ])

        points = PATROL.load_patrol_points(path)

        self.assertEqual([point["id"] for point in points], ["p1", "p2"])
        self.assertEqual(points[0]["x"], 1.25)
        self.assertEqual(points[0]["name"], "入口")
        self.assertEqual(points[1]["name"], "p2")

    def test_empty_list_is_rejected(self):
        path = self._write_json([])
        with self.assertRaisesRegex(ValueError, "empty"):
            PATROL.load_patrol_points(path)

    def test_non_array_is_rejected(self):
        path = self._write_json({"points": []})
        with self.assertRaisesRegex(ValueError, "array"):
            PATROL.load_patrol_points(path)

    def test_duplicate_ids_are_rejected(self):
        point = {"id": "p1", "x": 0, "y": 0, "yaw": 0}
        path = self._write_json([point, dict(point)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PATROL.load_patrol_points(path)

    def test_missing_id_is_rejected(self):
        path = self._write_json([{"x": 0, "y": 0, "yaw": 0}])
        with self.assertRaisesRegex(ValueError, "no id"):
            PATROL.load_patrol_points(path)

    def test_non_finite_coordinate_is_rejected(self):
        path = self._write_json([
            {"id": "p1", "x": math.nan, "y": 0, "yaw": 0}
        ])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            PATROL.load_patrol_points(path)


class PatrolStateHelperTests(unittest.TestCase):
    def test_quaternion_from_yaw(self):
        z, w = PATROL.quaternion_from_yaw(math.pi)
        self.assertAlmostEqual(z, 1.0)
        self.assertAlmostEqual(w, 0.0, places=7)

    def test_status_contains_progress_and_failure_lists(self):
        node = PATROL.PatrolNode.__new__(PATROL.PatrolNode)
        node.session_id = "patrol-test"
        node.state = "idle"
        node.message = ""
        node.points = [{
            "id": "p1",
            "name": "入口",
            "x": 1.0,
            "y": 2.0,
            "yaw": 0.5,
        }]
        node.current_index = 0
        node.completed_point_ids = []
        node.failed_points = []
        node.status_pub = FakePublisher()

        node.publish_status("navigating", "test")

        payload = json.loads(node.status_pub.messages[-1].data)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["state"], "navigating")
        self.assertEqual(payload["current_point"]["index"], 1)
        self.assertEqual(payload["current_point"]["total"], 1)
        self.assertEqual(payload["completed_point_ids"], [])
        self.assertEqual(payload["failed_points"], [])

    def test_stale_goal_response_is_ignored(self):
        node = PATROL.PatrolNode.__new__(PATROL.PatrolNode)
        node.active_goal_token = ("current", 2)

        class BombFuture:
            def result(self):
                raise AssertionError("stale future must not be read")

        node._goal_response(BombFuture(), ("old", 1))

    def test_failed_patrol_point_records_reason(self):
        node = PATROL.PatrolNode.__new__(PATROL.PatrolNode)
        node.active_target_kind = None
        node.current_index = 0
        node.points = [{"id": "p1"}]
        node.failed_points = []

        node._record_failure("aborted", target_kind="patrol")

        self.assertEqual(
            node.failed_points,
            [{
                "id": "p1",
                "code": "nav2_navigation_failed",
                "reason": "aborted",
            }],
        )

    def test_no_path_result_stops_task_with_structured_reason(self):
        node = PATROL.PatrolNode.__new__(PATROL.PatrolNode)
        node.active_goal_token = ("session", 1)
        node.active_target_kind = "patrol"
        node.pending_cancel_intent = None
        node.current_goal_handle = object()
        node.current_index = 0
        node.points = [{"id": "p1", "name": "入口"}]
        node.failed_points = []
        node.log = lambda _text: None

        def publish_status(state, message):
            node.state = state
            node.message = message

        node.publish_status = publish_status
        wrapped_result = types.SimpleNamespace(
            status=6,
            result=types.SimpleNamespace(
                error_code=9000,
                error_msg="Failed to create plan: no valid path",
            ),
        )
        future = types.SimpleNamespace(result=lambda: wrapped_result)

        node._goal_result(future, ("session", 1))

        self.assertEqual(node.state, "error")
        self.assertIn("无可行路径", node.message)
        self.assertEqual(
            node.failed_points[0]["code"],
            "nav2_no_feasible_path",
        )
        self.assertIsNone(node.active_goal_token)

    def test_cancel_refusal_keeps_goal_retryable(self):
        node = PATROL.PatrolNode.__new__(PATROL.PatrolNode)
        goal_handle = object()
        node.current_goal_handle = goal_handle
        node.active_target_kind = "patrol"
        node.pending_cancel_intent = "pause"
        node.log = lambda _text: None

        def publish_status(state, message):
            node.state = state
            node.message = message

        node.publish_status = publish_status
        node._cancel_request_failed("cancel refused")

        self.assertIs(node.current_goal_handle, goal_handle)
        self.assertIsNone(node.pending_cancel_intent)
        self.assertEqual(node.state, "navigating")


if __name__ == "__main__":
    unittest.main()
