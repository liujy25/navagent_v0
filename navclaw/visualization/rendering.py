from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from navclaw.mapping.exploration.bev_visuals import (
    BEV_GRAPH_EDGE_COLOR,
    draw_numbered_circle_marker,
    numbered_circle_marker_style,
    place_node_marker_style,
)
from navclaw.mapping.routing import _node_id_sort_key


VISUALIZE_AGENT_MARKER_RADIUS_PX = 10
VISUALIZE_FRONTIER_MARKER_RADIUS_PX = 5
VISUALIZE_FOCUS_CROP_MARGIN_PX = 80


def compose_contact_sheet(images: list[np.ndarray], gap_px: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("contact sheet requires at least one image")
    normalized = [np.asarray(image, dtype=np.uint8) for image in images]
    if len(normalized) == 1:
        return normalized[0]
    max_height = max(int(image.shape[0]) for image in normalized)
    max_width = max(int(image.shape[1]) for image in normalized)
    columns = min(3, len(normalized))
    rows = int(np.ceil(len(normalized) / float(columns)))
    canvas = np.zeros(
        (
            rows * max_height + max(0, rows - 1) * int(gap_px),
            columns * max_width + max(0, columns - 1) * int(gap_px),
            3,
        ),
        dtype=np.uint8,
    )
    for index, image in enumerate(normalized):
        row, column = divmod(index, columns)
        y0 = row * (max_height + int(gap_px))
        x0 = column * (max_width + int(gap_px))
        height, width = image.shape[:2]
        canvas[y0 : y0 + height, x0 : x0 + width] = image
    return canvas


def _normalize_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("rgb image must be HxWxC")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth, dtype=np.float32)
    finite_mask = np.isfinite(depth_array) & (depth_array > 0.0)
    if not np.any(finite_mask):
        return np.zeros((depth_array.shape[0], depth_array.shape[1], 3), dtype=np.uint8)
    valid_values = depth_array[finite_mask]
    min_depth = float(np.min(valid_values))
    max_depth = float(np.max(valid_values))
    normalized = np.zeros_like(depth_array, dtype=np.float32)
    if max_depth > min_depth:
        normalized[finite_mask] = (depth_array[finite_mask] - min_depth) / (max_depth - min_depth)
    uint8_image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([uint8_image, uint8_image, uint8_image], axis=2)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash_length: float = 12.0,
    gap_length: float = 8.0,
) -> None:
    x0, y0 = start_xy
    x1, y1 = end_xy
    total_length = float(np.hypot(x1 - x0, y1 - y0))
    if total_length <= 1e-6:
        return
    direction_x = (x1 - x0) / total_length
    direction_y = (y1 - y0) / total_length
    cursor = 0.0
    while cursor < total_length:
        dash_end = min(total_length, cursor + float(dash_length))
        segment_start = (
            x0 + direction_x * cursor,
            y0 + direction_y * cursor,
        )
        segment_end = (
            x0 + direction_x * dash_end,
            y0 + direction_y * dash_end,
        )
        draw.line([segment_start, segment_end], fill=fill, width=int(width))
        cursor += float(dash_length) + float(gap_length)


def _draw_dashed_ellipse_outline(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    width: int,
    dash_degrees: int = 20,
    gap_degrees: int = 12,
) -> None:
    start_degree = 0
    while start_degree < 360:
        end_degree = min(360, start_degree + int(dash_degrees))
        draw.arc(bbox, start=start_degree, end=end_degree, fill=fill, width=int(width))
        start_degree += int(dash_degrees) + int(gap_degrees)


def _draw_dashed_rectangle_outline(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    left, top, right, bottom = bbox
    _draw_dashed_line(draw, (left, top), (right, top), fill=fill, width=width)
    _draw_dashed_line(draw, (right, top), (right, bottom), fill=fill, width=width)
    _draw_dashed_line(draw, (right, bottom), (left, bottom), fill=fill, width=width)
    _draw_dashed_line(draw, (left, bottom), (left, top), fill=fill, width=width)


def _fit_image_to_canvas(
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    array = _normalize_rgb(image)
    image_height, image_width = array.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError("cannot fit empty image to canvas")
    scale = min(float(canvas_width) / float(image_width), float(canvas_height) / float(image_height))
    resized_width = max(1, int(round(float(image_width) * scale)))
    resized_height = max(1, int(round(float(image_height) * scale)))
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((int(canvas_height), int(canvas_width), 3), fill_color, dtype=np.uint8)
    offset_x = (int(canvas_width) - resized_width) // 2
    offset_y = (int(canvas_height) - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def _compute_focus_crop_bounds(
    image_shape: tuple[int, int, int] | tuple[int, int],
    explored_mask: np.ndarray | None,
    extra_points_px: list[tuple[float, float]] | None = None,
    margin_px: int = VISUALIZE_FOCUS_CROP_MARGIN_PX,
) -> tuple[int, int, int, int]:
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    crop_points: list[np.ndarray] = []
    if explored_mask is not None:
        explored = np.asarray(explored_mask, dtype=bool)
        if explored.shape != (image_height, image_width):
            raise ValueError("focus crop explored_mask shape must match image shape")
        if np.any(explored):
            explored_y, explored_x = np.nonzero(explored)
            crop_points.append(np.stack([explored_x, explored_y], axis=1).astype(np.float64))
    if extra_points_px is not None and extra_points_px != []:
        crop_points.append(np.asarray(extra_points_px, dtype=np.float64).reshape(-1, 2))
    if crop_points == []:
        return (0, 0, image_width, image_height)
    stacked = np.concatenate(crop_points, axis=0)
    min_x = max(0, int(np.floor(np.min(stacked[:, 0]))) - int(margin_px))
    max_x = min(image_width, int(np.ceil(np.max(stacked[:, 0]))) + int(margin_px) + 1)
    min_y = max(0, int(np.floor(np.min(stacked[:, 1]))) - int(margin_px))
    max_y = min(image_height, int(np.ceil(np.max(stacked[:, 1]))) + int(margin_px) + 1)
    return (min_x, min_y, max_x, max_y)


def _crop_image_to_bounds(
    image: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    array = _normalize_rgb(image)
    min_x, min_y, max_x, max_y = [int(value) for value in crop_bounds]
    if min_x < 0 or min_y < 0 or max_x > array.shape[1] or max_y > array.shape[0]:
        raise ValueError("crop bounds must lie within image")
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("crop bounds must define a non-empty region")
    return array[min_y:max_y, min_x:max_x]


def _draw_agent_marker(
    image: np.ndarray,
    px_x: int,
    px_y: int,
    radius_px: int = VISUALIZE_AGENT_MARKER_RADIUS_PX,
) -> None:
    cv2.circle(
        image,
        (int(px_x), int(px_y)),
        int(radius_px),
        (80, 160, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        image,
        (int(px_x), int(px_y)),
        int(radius_px),
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def _draw_frontier_points(
    image: np.ndarray,
    frontier_points_xy: list[tuple[float, float]] | None,
    xy_to_px,
) -> None:
    if frontier_points_xy is None or frontier_points_xy == []:
        return
    frontier_points = np.asarray(frontier_points_xy, dtype=np.float64).reshape(-1, 2)
    frontier_px = xy_to_px(frontier_points)
    for px_x, px_y in frontier_px:
        cv2.circle(
            image,
            (int(px_x), int(px_y)),
            int(VISUALIZE_FRONTIER_MARKER_RADIUS_PX),
            (255, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def _is_dashed_node_kind(node_kind: str) -> bool:
    return str(node_kind).strip() != "place"


def _graph_payload_nodes(
    graph_payload: dict[str, Any],
    floor_id: str | None = None,
) -> list[dict[str, Any]]:
    floor_id_str = None if floor_id is None else str(floor_id)
    raw_nodes = graph_payload.get("nodes")
    if isinstance(raw_nodes, list):
        nodes = [node for node in raw_nodes if isinstance(node, dict)]
        if floor_id_str is None:
            return nodes
        return [node for node in nodes if str(node.get("floor_id", "")) == floor_id_str]
    raw_floors = graph_payload.get("floors")
    if not isinstance(raw_floors, list):
        return []
    nodes: list[dict[str, Any]] = []
    for floor in raw_floors:
        if not isinstance(floor, dict):
            continue
        floor_id = str(floor.get("id", ""))
        if floor_id_str is not None and floor_id != floor_id_str:
            continue
        raw_floor_nodes = floor.get("nodes", [])
        if not isinstance(raw_floor_nodes, list):
            continue
        for node in raw_floor_nodes:
            if not isinstance(node, dict):
                continue
            node_payload = dict(node)
            if floor_id != "" and "floor_id" not in node_payload:
                node_payload["floor_id"] = floor_id
            nodes.append(node_payload)
    return nodes


def _graph_payload_edges(
    graph_payload: dict[str, Any],
    *,
    include_vertical: bool,
    floor_id: str | None = None,
) -> list[dict[str, Any]]:
    floor_id_str = None if floor_id is None else str(floor_id)
    raw_edges = graph_payload.get("edges")
    if isinstance(raw_edges, list):
        edges = [edge for edge in raw_edges if isinstance(edge, dict)]
        if floor_id_str is None:
            return edges
        return [edge for edge in edges if str(edge.get("floor_id", "")) == floor_id_str]
    edges: list[dict[str, Any]] = []
    raw_floors = graph_payload.get("floors")
    if isinstance(raw_floors, list):
        for floor in raw_floors:
            if not isinstance(floor, dict):
                continue
            if floor_id_str is not None and str(floor.get("id", "")) != floor_id_str:
                continue
            raw_floor_edges = floor.get("edges", [])
            if isinstance(raw_floor_edges, list):
                edges.extend(edge for edge in raw_floor_edges if isinstance(edge, dict))
    if include_vertical and floor_id_str is None:
        raw_vertical_edges = graph_payload.get("vertical_edges", [])
        if isinstance(raw_vertical_edges, list):
            edges.extend(edge for edge in raw_vertical_edges if isinstance(edge, dict))
    return edges


def _render_graph_image(graph_payload: dict[str, Any], current_node_id: str | None) -> np.ndarray:
    width = 1200
    height = 900
    background = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)

    nodes = _graph_payload_nodes(graph_payload)
    edges = _graph_payload_edges(graph_payload, include_vertical=True)
    if nodes == []:
        draw.text((40, 40), "empty graph", fill=(0, 0, 0))
        return np.asarray(image, dtype=np.uint8)

    xs = [float(node["position"][0]) for node in nodes]
    ys = [float(node["position"][1]) for node in nodes]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad = 100.0

    def project(position: tuple[float, float, float] | list[float]) -> tuple[float, float]:
        x = float(position[0])
        y = float(position[1])
        if max_x == min_x:
            px = width / 2.0
        else:
            px = pad + (x - min_x) / (max_x - min_x) * (width - 2.0 * pad)
        if max_y == min_y:
            py = height / 2.0
        else:
            py = height - pad - (y - min_y) / (max_y - min_y) * (height - 2.0 * pad)
        return float(px), float(py)

    node_kinds: dict[str, str] = {}
    for node in nodes:
        node_kinds[str(node["id"])] = str(node.get("node_kind", "place")).strip() or "place"

    def draw_node_badge(
        px: float,
        py: float,
        node_id: str,
        category: str,
        fill: tuple[int, int, int],
        dashed: bool,
    ) -> None:
        label = f"{node_id}\n{category}"
        bbox = draw.multiline_textbbox((0.0, 0.0), label, spacing=2, align="center")
        text_width = float(bbox[2] - bbox[0])
        text_height = float(bbox[3] - bbox[1])
        radius = int(max(24.0, max(text_width, text_height) / 2.0 + 12.0))
        outline_bbox = (
            px - radius,
            py - radius,
            px + radius,
            py + radius,
        )
        draw.ellipse(
            outline_bbox,
            fill=fill,
        )
        if dashed:
            _draw_dashed_ellipse_outline(draw, outline_bbox, fill=(0, 0, 0), width=3)
        else:
            draw.ellipse(outline_bbox, outline=(0, 0, 0), width=3)
        draw.multiline_text(
            (px, py),
            label,
            fill=(0, 0, 0),
            anchor="mm",
            align="center",
            spacing=2,
        )

    node_positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        node_positions[str(node["id"])] = project(node["position"])

    for edge in edges:
        src_id = str(edge["src_id"])
        dst_id = str(edge["dst_id"])
        if src_id not in node_positions or dst_id not in node_positions:
            continue
        src_xy = node_positions[src_id]
        dst_xy = node_positions[dst_id]
        dashed = _is_dashed_node_kind(node_kinds.get(src_id, "place")) or _is_dashed_node_kind(
            node_kinds.get(dst_id, "place")
        )
        if dashed:
            _draw_dashed_line(draw, src_xy, dst_xy, fill=(0, 0, 0), width=4)
        else:
            draw.line([src_xy, dst_xy], fill=(0, 0, 0), width=4)

    for node in nodes:
        node_id = str(node["id"])
        px, py = node_positions[node_id]
        color = (255, 215, 0)
        if current_node_id is not None and node_id == current_node_id:
            color = (80, 140, 255)
        name = str(node.get("category", "") or "unlabeled")
        draw_node_badge(
            px=px,
            py=py,
            node_id=node_id,
            category=name,
            fill=color,
            dashed=_is_dashed_node_kind(node_kinds.get(node_id, "place")),
        )

    return np.asarray(image, dtype=np.uint8)


def _render_bev_graph_image(
    graph_payload: dict[str, Any],
    current_node_id: str | None,
    bev_background: np.ndarray,
    xy_to_px,
    explored_mask: np.ndarray | None = None,
    point_markers: list[dict[str, Any]] | None = None,
    crop: bool = True,
    floor_id: str | None = None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(bev_background, dtype=np.uint8).copy()).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    marker_style = numbered_circle_marker_style(
        image_width=int(image.size[0]),
        image_height=int(image.size[1]),
        mode="bev",
    )
    node_marker_style = place_node_marker_style(
        image_width=int(image.size[0]),
        image_height=int(image.size[1]),
        mode="bev",
    )

    nodes = _graph_payload_nodes(graph_payload, floor_id=floor_id)
    markers = [] if point_markers is None else list(point_markers)
    if nodes == [] and markers == []:
        draw.text((40, 40), "empty graph", fill=(0, 0, 0))
        return np.asarray(image, dtype=np.uint8)

    node_positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        node_id = str(node["id"])
        position_xy = np.asarray([[float(node["position"][0]), float(node["position"][1])]], dtype=np.float64)
        node_px = xy_to_px(position_xy)[0]
        node_positions[node_id] = (float(node_px[0]), float(node_px[1]))

    def draw_node_badge(
        px: float,
        py: float,
        label: str,
    ) -> None:
        draw_numbered_circle_marker(
            image,
            (float(px), float(py)),
            str(label),
            style=node_marker_style,
        )

    crop_points: list[np.ndarray] = []
    if explored_mask is not None and np.any(explored_mask):
        explored_y, explored_x = np.nonzero(explored_mask)
        crop_points.append(np.stack([explored_x, explored_y], axis=1).astype(np.float64))

    edges = _graph_payload_edges(graph_payload, include_vertical=False, floor_id=floor_id)
    for edge in edges:
        src_id = str(edge.get("src_id", ""))
        dst_id = str(edge.get("dst_id", ""))
        if src_id not in node_positions or dst_id not in node_positions:
            continue
        src_xy = node_positions[src_id]
        dst_xy = node_positions[dst_id]
        draw.line([src_xy, dst_xy], fill=BEV_GRAPH_EDGE_COLOR, width=6)
        crop_points.append(np.asarray([[src_xy[0], src_xy[1]], [dst_xy[0], dst_xy[1]]], dtype=np.float64))

    ordered_nodes = sorted(nodes, key=lambda node: _node_id_sort_key(str(node["id"])))
    for index, node in enumerate(ordered_nodes):
        node_id = str(node["id"])
        px, py = node_positions[node_id]
        draw_node_badge(
            px=px,
            py=py,
            label=str(index),
        )
        crop_points.append(np.asarray([[px, py]], dtype=np.float64))

    for marker in markers:
        x = float(marker["x"])
        y = float(marker["y"])
        label = str(marker.get("label", ""))
        color_raw = marker.get("color", (255, 0, 255))
        if not isinstance(color_raw, (list, tuple)) or len(color_raw) != 3:
            raise ValueError(f"point marker color must be a length-3 list/tuple, got {color_raw!r}")
        color = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2]))
        radius = int(marker.get("radius", 10))
        marker_px_array = xy_to_px(np.asarray([[x, y]], dtype=np.float64))[0]
        marker_px = (float(marker_px_array[0]), float(marker_px_array[1]))
        if label.isdigit() and len(label) <= 2:
            draw_numbered_circle_marker(
                image,
                marker_px,
                label,
                style=marker_style,
            )
            crop_points.append(np.asarray([[marker_px[0], marker_px[1]]], dtype=np.float64))
            continue
        draw.ellipse(
            (
                marker_px[0] - radius,
                marker_px[1] - radius,
                marker_px[0] + radius,
                marker_px[1] + radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        if label != "":
            draw.text(
                (marker_px[0] + radius + 4.0, marker_px[1] - radius - 4.0),
                label,
                fill=(255, 255, 255),
            )
        crop_points.append(np.asarray([[marker_px[0], marker_px[1]]], dtype=np.float64))

    rendered = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if crop_points == [] or not bool(crop):
        return rendered

    stacked = np.concatenate(crop_points, axis=0)
    margin = 120
    min_x = max(0, int(np.floor(np.min(stacked[:, 0]))) - margin)
    max_x = min(rendered.shape[1], int(np.ceil(np.max(stacked[:, 0]))) + margin + 1)
    min_y = max(0, int(np.floor(np.min(stacked[:, 1]))) - margin)
    max_y = min(rendered.shape[0], int(np.ceil(np.max(stacked[:, 1]))) + margin + 1)
    return rendered[min_y:max_y, min_x:max_x]


@dataclass
class CacheSnapshot:
    observation_ids: set[str]
    detection_ids: set[str]
    image_ids: set[str]


@dataclass
class StepArtifactObservationGroups:
    panorama_obs_ids: set[str]
    step_obs_ids: set[str]
