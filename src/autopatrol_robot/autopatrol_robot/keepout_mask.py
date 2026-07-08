"""Generate a static Nav2 keepout mask from Web-managed polygons."""

import math
import os
from pathlib import Path
import tempfile
import logging

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


DEFAULT_KEEPOUT_PADDING_M = 0.18
LOGGER = logging.getLogger(__name__)


def _format_number(value):
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text or "0"


def calculate_padding_pixels(resolution, keepout_padding_m=DEFAULT_KEEPOUT_PADDING_M):
    """Convert keepout padding from metres to a conservative pixel radius."""
    resolution = float(resolution)
    keepout_padding_m = float(keepout_padding_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("map resolution must be a positive number")
    if not math.isfinite(keepout_padding_m) or keepout_padding_m < 0.0:
        raise ValueError("keepout_padding_m must be a non-negative number")
    if keepout_padding_m == 0.0:
        return 0
    return int(math.ceil(keepout_padding_m / resolution))


def _dilate_pixels_with_cv2(width, height, pixels, padding_px):
    image = np.frombuffer(bytes(pixels), dtype=np.uint8).reshape((height, width))
    occupied = np.where(image == 0, 255, 0).astype(np.uint8)
    kernel_size = 2 * padding_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(occupied, kernel, iterations=1)
    return bytearray(np.where(dilated > 0, 0, 254).astype(np.uint8).tobytes())


def _dilate_pixels_without_cv2(width, height, pixels, padding_px):
    occupied_indices = [
        divmod(index, width)
        for index, value in enumerate(pixels)
        if value == 0
    ]
    if not occupied_indices:
        return pixels

    offsets = [
        (dy, dx)
        for dy in range(-padding_px, padding_px + 1)
        for dx in range(-padding_px, padding_px + 1)
        if dx * dx + dy * dy <= padding_px * padding_px
    ]
    dilated = bytearray([254]) * (width * height)
    for row, col in occupied_indices:
        for dy, dx in offsets:
            target_row = row + dy
            target_col = col + dx
            if 0 <= target_row < height and 0 <= target_col < width:
                dilated[target_row * width + target_col] = 0
    return dilated


def dilate_keepout_pixels(width, height, pixels, padding_px):
    """Expand occupied mask cells by padding_px pixels."""
    if padding_px <= 0 or not any(value == 0 for value in pixels):
        return pixels
    if cv2 is not None and np is not None:
        return _dilate_pixels_with_cv2(width, height, pixels, padding_px)
    return _dilate_pixels_without_cv2(width, height, pixels, padding_px)


def world_to_grid(origin, resolution, x, y):
    """Convert one map-frame point to continuous OccupancyGrid coordinates."""
    delta_x = float(x) - float(origin[0])
    delta_y = float(y) - float(origin[1])
    yaw = float(origin[2])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        (cos_yaw * delta_x + sin_yaw * delta_y) / resolution,
        (-sin_yaw * delta_x + cos_yaw * delta_y) / resolution,
    )


def _point_on_segment(x, y, start, end, tolerance=1e-9):
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= tolerance * tolerance:
        return math.hypot(x - start[0], y - start[1]) <= tolerance
    projection = (
        (x - start[0]) * delta_x + (y - start[1]) * delta_y
    ) / length_squared
    if projection < 0.0 or projection > 1.0:
        return False
    nearest_x = start[0] + projection * delta_x
    nearest_y = start[1] + projection * delta_y
    return math.hypot(x - nearest_x, y - nearest_y) <= tolerance


def _point_in_polygon(x, y, points):
    inside = False
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        if _point_on_segment(x, y, start, end):
            return True
        if (start[1] > y) == (end[1] > y):
            continue
        intersection_x = (
            start[0]
            + (y - start[1]) * (end[0] - start[0])
            / (end[1] - start[1])
        )
        if x < intersection_x:
            inside = not inside
    return inside


def _orientation(start, end, point):
    return (
        (end[0] - start[0]) * (point[1] - start[1])
        - (end[1] - start[1]) * (point[0] - start[0])
    )


def _segments_intersect(first_start, first_end, second_start, second_end):
    tolerance = 1e-9
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if (
        orientations[0] * orientations[1] < -tolerance
        and orientations[2] * orientations[3] < -tolerance
    ):
        return True
    return any((
        abs(orientations[0]) <= tolerance
        and _point_on_segment(*second_start, first_start, first_end),
        abs(orientations[1]) <= tolerance
        and _point_on_segment(*second_end, first_start, first_end),
        abs(orientations[2]) <= tolerance
        and _point_on_segment(*first_start, second_start, second_end),
        abs(orientations[3]) <= tolerance
        and _point_on_segment(*first_end, second_start, second_end),
    ))


def _cell_intersects_polygon(cell_x, cell_y, points):
    corners = (
        (cell_x, cell_y),
        (cell_x + 1.0, cell_y),
        (cell_x + 1.0, cell_y + 1.0),
        (cell_x, cell_y + 1.0),
    )
    if _point_in_polygon(cell_x + 0.5, cell_y + 0.5, points):
        return True
    if any(_point_in_polygon(*corner, points) for corner in corners):
        return True
    if any(
        cell_x <= point[0] <= cell_x + 1.0
        and cell_y <= point[1] <= cell_y + 1.0
        for point in points
    ):
        return True
    cell_edges = tuple(
        (corners[index], corners[(index + 1) % len(corners)])
        for index in range(len(corners))
    )
    return any(
        _segments_intersect(
            points[index],
            points[(index + 1) % len(points)],
            edge_start,
            edge_end,
        )
        for index in range(len(points))
        for edge_start, edge_end in cell_edges
    )


def build_keepout_image(
    map_payload,
    zones,
    keepout_padding_m=DEFAULT_KEEPOUT_PADDING_M,
    keepout_padding_px=None,
):
    """Rasterize enabled keepout polygons into a free/occupied mask image."""
    width = int(map_payload["width"])
    height = int(map_payload["height"])
    resolution = float(map_payload["resolution"])
    origin_payload = map_payload["origin"]
    origin = (
        float(origin_payload["x"]),
        float(origin_payload["y"]),
        float(origin_payload["yaw"]),
    )
    if width <= 0 or height <= 0 or resolution <= 0.0:
        raise ValueError("地图尺寸或分辨率无效")

    if keepout_padding_px is None:
        keepout_padding_px = calculate_padding_pixels(
            resolution,
            keepout_padding_m,
        )

    pixels = bytearray([254]) * (width * height)
    for zone in zones:
        if zone.get("enabled") is False:
            continue
        points = zone.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("禁区至少需要 3 个顶点")

        grid_points = []
        for point in points:
            grid_x, grid_y = world_to_grid(
                origin,
                resolution,
                float(point["x"]),
                float(point["y"]),
            )
            if not math.isfinite(grid_x) or not math.isfinite(grid_y):
                raise ValueError("禁区包含无效坐标")
            grid_points.append((grid_x, grid_y))

        minimum_x = max(0, math.floor(min(point[0] for point in grid_points)))
        maximum_x = min(
            width - 1,
            math.floor(max(point[0] for point in grid_points)),
        )
        minimum_y = max(0, math.floor(min(point[1] for point in grid_points)))
        maximum_y = min(
            height - 1,
            math.floor(max(point[1] for point in grid_points)),
        )
        for grid_y in range(minimum_y, maximum_y + 1):
            image_y = height - 1 - grid_y
            row_offset = image_y * width
            for grid_x in range(minimum_x, maximum_x + 1):
                if _cell_intersects_polygon(grid_x, grid_y, grid_points):
                    pixels[row_offset + grid_x] = 0

    pixels = dilate_keepout_pixels(width, height, pixels, keepout_padding_px)
    return {"width": width, "height": height, "pixels": pixels}


def _build_pgm_bytes(image):
    width = image["width"]
    height = image["height"]
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    return header + bytes(image["pixels"])


def _build_yaml(map_name, map_metadata):
    origin = map_metadata["origin"]
    return "\n".join((
        f"image: {map_name}_keepout.pgm",
        "mode: trinary",
        f"resolution: {_format_number(map_metadata['resolution'])}",
        "origin: ["
        f"{_format_number(origin[0])}, "
        f"{_format_number(origin[1])}, "
        f"{_format_number(origin[2])}]",
        "negate: 0",
        "occupied_thresh: "
        f"{_format_number(map_metadata.get('occupied_thresh', 0.65))}",
        "free_thresh: "
        f"{_format_number(map_metadata.get('free_thresh', 0.196))}",
    )) + "\n"


def _write_temporary(path, content, binary):
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8"}
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        **kwargs,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _restore_file(path, previous_content, binary):
    if previous_content is None:
        path.unlink(missing_ok=True)
        return
    temporary = _write_temporary(path, previous_content, binary)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_keepout_mask(
    maps_dir,
    map_name,
    map_metadata,
    map_payload,
    zones_document,
    keepout_padding_m=DEFAULT_KEEPOUT_PADDING_M,
    logger=None,
):
    """Generate and atomically publish one map's keepout PGM and YAML."""
    keepout_dir = Path(maps_dir) / "keepout"
    keepout_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = keepout_dir / f"{map_name}_keepout.pgm"
    yaml_path = keepout_dir / f"{map_name}_keepout.yaml"

    resolution = float(map_payload["resolution"])
    padding_px = calculate_padding_pixels(resolution, keepout_padding_m)
    active_logger = logger or LOGGER
    active_logger.info(
        "keepout mask padding: resolution=%.6f m/pixel, "
        "keepout_padding_m=%.3f, padding_px=%d",
        resolution,
        keepout_padding_m,
        padding_px,
    )

    image = build_keepout_image(
        map_payload,
        zones_document.get("zones", []),
        keepout_padding_m=keepout_padding_m,
        keepout_padding_px=padding_px,
    )
    pgm_content = _build_pgm_bytes(image)
    yaml_content = _build_yaml(map_name, map_metadata)
    previous_pgm = pgm_path.read_bytes() if pgm_path.is_file() else None
    previous_yaml = yaml_path.read_bytes() if yaml_path.is_file() else None
    pgm_temporary = None
    yaml_temporary = None
    pgm_replaced = False
    yaml_replaced = False

    try:
        pgm_temporary = _write_temporary(pgm_path, pgm_content, True)
        yaml_temporary = _write_temporary(yaml_path, yaml_content, False)
        os.replace(pgm_temporary, pgm_path)
        pgm_temporary = None
        pgm_replaced = True
        os.replace(yaml_temporary, yaml_path)
        yaml_temporary = None
        yaml_replaced = True
    except Exception:
        if pgm_replaced:
            _restore_file(pgm_path, previous_pgm, True)
        if yaml_replaced:
            _restore_file(yaml_path, previous_yaml, True)
        raise
    finally:
        if pgm_temporary is not None:
            pgm_temporary.unlink(missing_ok=True)
        if yaml_temporary is not None:
            yaml_temporary.unlink(missing_ok=True)

    return {
        "generated": True,
        "yaml": str(yaml_path),
        "pgm": str(pgm_path),
        "restart_required": True,
        "warning": None,
    }
