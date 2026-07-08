"""Unit tests for Web patrol storage and control endpoints."""

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def _load_web_module():
    class DummyNode:
        pass

    class DummyMessage:
        pass

    class DummyQoS:
        def __init__(self, depth=1):
            self.depth = depth

    class DummyTrigger:
        class Request:
            pass

    class DummySocketIO:
        def __init__(self, *args, **kwargs):
            self.events = []

        def emit(self, *args, **kwargs):
            self.events.append((args, kwargs))

        def on(self, *args, **kwargs):
            def register(function):
                return function
            return register

    class DummyFlask:
        def __init__(self, *args, **kwargs):
            self.config = {}
            self.logger = types.SimpleNamespace(exception=lambda *args: None)

        def route(self, *args, **kwargs):
            def register(function):
                return function
            return register

    fake_flask = types.ModuleType("flask")
    fake_flask.Flask = DummyFlask
    fake_flask.abort = lambda status: status
    fake_flask.jsonify = lambda payload: payload
    fake_flask.render_template = lambda *args, **kwargs: ""
    fake_flask.request = types.SimpleNamespace(
        get_json=lambda **kwargs: {}
    )
    fake_flask.send_from_directory = lambda *args, **kwargs: None

    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy_node = types.ModuleType("rclpy.node")
    fake_rclpy_node.Node = DummyNode
    fake_rclpy_qos = types.ModuleType("rclpy.qos")
    fake_rclpy_qos.DurabilityPolicy = types.SimpleNamespace(
        TRANSIENT_LOCAL="transient"
    )
    fake_rclpy_qos.HistoryPolicy = types.SimpleNamespace(
        KEEP_LAST="keep_last"
    )
    fake_rclpy_qos.QoSProfile = DummyQoS
    fake_rclpy_qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="reliable"
    )
    fake_rclpy_time = types.ModuleType("rclpy.time")
    fake_rclpy_time.Time = DummyMessage

    fake_geometry = types.ModuleType("geometry_msgs")
    fake_geometry_msg = types.ModuleType("geometry_msgs.msg")
    fake_geometry_msg.Twist = DummyMessage

    fake_interfaces = types.ModuleType("rcl_interfaces")
    fake_interfaces_msg = types.ModuleType("rcl_interfaces.msg")
    fake_interfaces_msg.Parameter = DummyMessage
    fake_interfaces_msg.ParameterType = DummyMessage
    fake_interfaces_msg.ParameterValue = DummyMessage
    fake_interfaces_srv = types.ModuleType("rcl_interfaces.srv")
    fake_interfaces_srv.GetParameters = DummyMessage
    fake_interfaces_srv.SetParameters = DummyMessage

    fake_nav = types.ModuleType("nav_msgs")
    fake_nav_msg = types.ModuleType("nav_msgs.msg")
    fake_nav_msg.OccupancyGrid = DummyMessage

    fake_sensor = types.ModuleType("sensor_msgs")
    fake_sensor_msg = types.ModuleType("sensor_msgs.msg")
    fake_sensor_msg.Image = DummyMessage

    fake_std = types.ModuleType("std_msgs")
    fake_std_msg = types.ModuleType("std_msgs.msg")
    fake_std_msg.String = DummyMessage
    fake_std_msg.Float32 = DummyMessage

    fake_std_srvs = types.ModuleType("std_srvs")
    fake_std_srvs_srv = types.ModuleType("std_srvs.srv")
    fake_std_srvs_srv.Trigger = DummyTrigger

    fake_cv_bridge = types.ModuleType("cv_bridge")
    fake_cv_bridge.CvBridge = object

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMWRITE_JPEG_QUALITY = 1
    fake_cv2.INTER_AREA = 1

    fake_socketio = types.ModuleType("flask_socketio")
    fake_socketio.SocketIO = DummySocketIO

    fake_tf = types.ModuleType("tf2_ros")
    fake_tf.Buffer = DummyMessage
    fake_tf.TransformException = Exception
    fake_tf.TransformListener = DummyMessage

    fake_package = types.ModuleType("autopatrol_robot")
    fake_package.__path__ = []
    fake_keepout = types.ModuleType("autopatrol_robot.keepout_mask")
    fake_keepout.generate_keepout_mask = lambda *args, **kwargs: {}

    stubs = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_rclpy_node,
        "rclpy.qos": fake_rclpy_qos,
        "rclpy.time": fake_rclpy_time,
        "geometry_msgs": fake_geometry,
        "geometry_msgs.msg": fake_geometry_msg,
        "rcl_interfaces": fake_interfaces,
        "rcl_interfaces.msg": fake_interfaces_msg,
        "rcl_interfaces.srv": fake_interfaces_srv,
        "nav_msgs": fake_nav,
        "nav_msgs.msg": fake_nav_msg,
        "sensor_msgs": fake_sensor,
        "sensor_msgs.msg": fake_sensor_msg,
        "std_msgs": fake_std,
        "std_msgs.msg": fake_std_msg,
        "std_srvs": fake_std_srvs,
        "std_srvs.srv": fake_std_srvs_srv,
        "cv_bridge": fake_cv_bridge,
        "cv2": fake_cv2,
        "flask": fake_flask,
        "flask_socketio": fake_socketio,
        "tf2_ros": fake_tf,
        "autopatrol_robot": fake_package,
        "autopatrol_robot.keepout_mask": fake_keepout,
    }

    module_path = (
        Path(__file__).parents[1]
        / "autopatrol_robot"
        / "web_bridge_node.py"
    )
    spec = importlib.util.spec_from_file_location(
        "web_bridge_node_tested", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


WEB = _load_web_module()


class FakeFuture:
    def __init__(self, response):
        self.response = response
        self.canceled = False

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self.response

    def cancel(self):
        self.canceled = True


class FakeClient:
    def __init__(self, ready=True, success=True):
        self.ready = ready
        self.success = success

    def service_is_ready(self):
        return self.ready

    def call_async(self, _request):
        response = types.SimpleNamespace(
            success=self.success,
            message="accepted" if self.success else "rejected",
        )
        return FakeFuture(response)


class FakeBridge:
    def __init__(self, success=True):
        self.success = success
        self.commands = []

    def call_patrol_service(self, command):
        self.commands.append(command)
        status = 200 if self.success else 409
        return {
            "command": command,
            "success": self.success,
            "message": "accepted" if self.success else "rejected",
        }, status

    def execute_patrol_control(self, command):
        return self.call_patrol_service(command)


class PatrolStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_path = WEB.PATROL_POINTS_PATH
        WEB.PATROL_POINTS_PATH = (
            Path(self.temp_dir.name) / "patrol_points.json"
        )
        self.addCleanup(self._restore_path)

    def _restore_path(self):
        WEB.PATROL_POINTS_PATH = self.original_path

    def test_atomic_save_and_load(self):
        points = [{"id": "p1", "x": 1.0, "y": 2.0, "yaw": 0.0}]

        WEB.save_patrol_points(points)

        self.assertEqual(WEB.load_patrol_points(), points)
        self.assertEqual(
            list(Path(self.temp_dir.name).glob("*.tmp")),
            [],
        )

    def test_non_finite_coordinate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            WEB.build_patrol_point({
                "id": "ignored",
                "x": math.inf,
                "y": 0,
                "yaw": 0,
            })


class KeepoutApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_maps_dir = WEB.MAPS_DIR
        self.original_get_json = WEB.request.get_json
        WEB.MAPS_DIR = Path(self.temp_dir.name)
        WEB.request.get_json = lambda **kwargs: {
            "map_name": "326",
            "zones": [],
        }
        self.addCleanup(self._restore_globals)

    def _restore_globals(self):
        WEB.MAPS_DIR = self.original_maps_dir
        WEB.request.get_json = self.original_get_json

    def _call_save(self, generator):
        metadata = {
            "resolution": 0.05,
            "origin": [0.0, 0.0, 0.0],
        }
        payload = {
            "width": 10,
            "height": 10,
            "resolution": 0.05,
            "origin": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        }
        document = {
            "map_name": "326",
            "updated_at": "2026-06-22T12:00:00+08:00",
            "zones": [],
        }
        with mock.patch.object(
            WEB,
            "load_saved_map",
            return_value=(metadata, payload),
        ), mock.patch.object(
            WEB,
            "validate_zones_document",
            return_value=document,
        ), mock.patch.object(
            WEB,
            "save_zones_document",
        ), mock.patch.object(
            WEB,
            "generate_keepout_mask",
            side_effect=generator,
        ):
            return WEB.api_save_map_zones("326")

    def test_zone_save_returns_generated_mask_status(self):
        result = self._call_save(lambda *args: {
            "generated": True,
            "yaml": "/maps/keepout/326_keepout.yaml",
            "pgm": "/maps/keepout/326_keepout.pgm",
            "restart_required": True,
            "warning": None,
        })

        self.assertTrue(result["keepout"]["generated"])
        self.assertTrue(result["keepout"]["restart_required"])

    def test_zone_save_keeps_success_and_returns_mask_warning(self):
        def fail_generation(*args):
            raise OSError("disk full")

        result = self._call_save(fail_generation)

        self.assertEqual(result["map_name"], "326")
        self.assertFalse(result["keepout"]["generated"])
        self.assertIn("disk full", result["keepout"]["warning"])


class PatrolServiceClientTests(unittest.TestCase):
    def test_ready_service_response_is_returned(self):
        node = WEB.WebBridge.__new__(WEB.WebBridge)
        node.patrol_clients = {
            "start": FakeClient(ready=True, success=True)
        }

        payload, status = node.call_patrol_service("start")

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])

    def test_unavailable_service_returns_503(self):
        node = WEB.WebBridge.__new__(WEB.WebBridge)
        node.patrol_clients = {
            "start": FakeClient(ready=False)
        }

        payload, status = node.call_patrol_service("start")

        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "patrol service is unavailable")


class PatrolControlApiTests(unittest.TestCase):
    def setUp(self):
        WEB.web_bridge_node_instance = None
        self.addCleanup(
            setattr, WEB, "web_bridge_node_instance", None
        )

    def test_unknown_command_returns_404(self):
        payload, status = WEB.api_patrol_control("unknown")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "unknown patrol command")

    def test_missing_ros_bridge_returns_503(self):
        payload, status = WEB.api_patrol_control("start")
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "ROS bridge is not ready")

    def test_start_is_forwarded_to_ros_bridge(self):
        bridge = FakeBridge(success=True)
        WEB.web_bridge_node_instance = bridge

        payload, status = WEB.api_patrol_control("start")

        self.assertEqual(status, 200)
        self.assertEqual(bridge.commands, ["start"])
        self.assertTrue(payload["success"])

    def test_rejected_service_command_returns_409(self):
        WEB.web_bridge_node_instance = FakeBridge(success=False)

        payload, status = WEB.api_patrol_control("pause")

        self.assertEqual(status, 409)
        self.assertFalse(payload["success"])


if __name__ == "__main__":
    unittest.main()
