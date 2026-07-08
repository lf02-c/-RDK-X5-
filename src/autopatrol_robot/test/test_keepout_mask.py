"""Unit tests for static Nav2 keepout mask generation."""

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).parents[1]
    / "autopatrol_robot"
    / "keepout_mask.py"
)
SPEC = importlib.util.spec_from_file_location(
    "keepout_mask_tested",
    MODULE_PATH,
)
KEEPOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KEEPOUT)


def map_fixture(width=8, height=6, resolution=1.0, yaw=0.0):
    metadata = {
        "resolution": resolution,
        "origin": [0.0, 0.0, yaw],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    payload = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin": {"x": 0.0, "y": 0.0, "yaw": yaw},
    }
    return metadata, payload


class KeepoutRasterTests(unittest.TestCase):
    def test_world_to_grid_respects_rotated_origin(self):
        grid_x, grid_y = KEEPOUT.world_to_grid(
            (0.0, 0.0, math.pi / 2.0),
            1.0,
            -1.0,
            2.0,
        )

        self.assertAlmostEqual(grid_x, 2.0)
        self.assertAlmostEqual(grid_y, 1.0)

    def test_enabled_polygon_is_black_and_y_axis_is_flipped(self):
        _, payload = map_fixture()
        zones = [{
            "enabled": True,
            "points": [
                {"x": 1.0, "y": 1.0},
                {"x": 3.0, "y": 1.0},
                {"x": 3.0, "y": 3.0},
                {"x": 1.0, "y": 3.0},
            ],
        }]

        image = KEEPOUT.build_keepout_image(payload, zones)

        self.assertEqual((image["height"], image["width"]), (6, 8))
        pixels = image["pixels"]
        self.assertEqual(pixels[4 * 8 + 1], 0)
        self.assertEqual(pixels[2 * 8 + 3], 0)
        self.assertEqual(pixels[0], 254)

    def test_disabled_polygon_is_not_rasterized(self):
        _, payload = map_fixture()
        zones = [{
            "enabled": False,
            "points": [
                {"x": 1.0, "y": 1.0},
                {"x": 3.0, "y": 1.0},
                {"x": 3.0, "y": 3.0},
            ],
        }]

        image = KEEPOUT.build_keepout_image(payload, zones)

        self.assertTrue(all(pixel == 254 for pixel in image["pixels"]))


class KeepoutFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.metadata, self.payload = map_fixture()
        self.document = {"zones": []}

    def test_generate_writes_expected_yaml_and_pgm(self):
        result = KEEPOUT.generate_keepout_mask(
            self.temp_dir.name,
            "326",
            self.metadata,
            self.payload,
            self.document,
        )

        yaml_path = Path(result["yaml"])
        pgm_path = Path(result["pgm"])
        self.assertEqual(yaml_path.name, "326_keepout.yaml")
        self.assertEqual(pgm_path.name, "326_keepout.pgm")
        yaml_content = yaml_path.read_text()
        self.assertIn("image: 326_keepout.pgm", yaml_content)
        self.assertIn("negate: 0", yaml_content)
        self.assertTrue(pgm_path.read_bytes().startswith(b"P5\n8 6\n255\n"))

    def test_publish_failure_restores_previous_mask(self):
        keepout_dir = Path(self.temp_dir.name) / "keepout"
        keepout_dir.mkdir()
        pgm_path = keepout_dir / "326_keepout.pgm"
        yaml_path = keepout_dir / "326_keepout.yaml"
        pgm_path.write_bytes(b"old-pgm")
        yaml_path.write_bytes(b"old-yaml")
        real_replace = KEEPOUT.os.replace
        call_count = 0

        def fail_yaml_replace(source, target):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("simulated yaml replace failure")
            return real_replace(source, target)

        with mock.patch.object(
            KEEPOUT.os,
            "replace",
            side_effect=fail_yaml_replace,
        ):
            with self.assertRaisesRegex(OSError, "simulated"):
                KEEPOUT.generate_keepout_mask(
                    self.temp_dir.name,
                    "326",
                    self.metadata,
                    self.payload,
                    self.document,
                )

        self.assertEqual(pgm_path.read_bytes(), b"old-pgm")
        self.assertEqual(yaml_path.read_bytes(), b"old-yaml")


if __name__ == "__main__":
    unittest.main()
