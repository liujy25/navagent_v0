from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw
from frontier_exploration.frontier_detection import (
    contour_to_frontiers,
    filter_out_small_unexplored,
    interpolate_contour,
)
from skimage.morphology import medial_axis

from navclaw.agent.visual_grounding import VisualWaypoint
from navclaw.mapping.exploration.bev_map import _split_frontier_by_turn_angle
from navclaw.mapping.exploration.bev_visuals import draw_numbered_circle_marker
from navclaw.mapping.exploration.bev_visuals import add_cardinal_direction_border
from navclaw.mapping.exploration.bev_visuals import numbered_circle_marker_style
from navclaw.mapping.exploration.bev_visuals import place_node_marker_style
from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.llm.image_preprocessing import scale_xy
from navclaw.mapping.exploration.manager import _compute_unoccluded_mask
from navclaw.mapping.exploration.manager import _project_points
from navclaw.mapping.exploration.overlay_drawing import _clamp_frontier_circle_marker_center
from navclaw.mapping.exploration.overlay_drawing import _draw_frontier_circle_marker
from navclaw.mapping.exploration.overlay_drawing import _rgb_frontier_circle_style
from navclaw.visualization.action_mode_overlays import FRONTIER_MARKER_FILL
from navclaw.visualization.action_mode_overlays import BevOverlayTransform
from navclaw.visualization.action_mode_overlays import _render_scaled_overlay

if TYPE_CHECKING:
    from navclaw.agent.visual_action_context import VisualViewContext
    from navclaw.mapping.exploration.bev_map import GlobalBEVMap
    from navclaw.mapping.exploration.manager import ExplorationManager
    from navclaw.runtime.cache import RuntimeCache
    from navclaw.types import LocalmapFrontierRecord


VLN_WAYPOINT_SAMPLE_SPACING_M = 0.8
VLN_WAYPOINT_MIN_DISTANCE_M = 0.4
VLN_WAYPOINT_MAX_DISTANCE_M = 4.5
VLN_WAYPOINT_MAX_CANDIDATES = 20
VLN_STOP_WAYPOINT_SAMPLE_SPACING_M = 0.6
VLN_STOP_WAYPOINT_MAX_DISTANCE_M = 3.0
VLN_WAYPOINT_OCCLUSION_TOLERANCE_M = 0.10
VLN_WAYPOINT_VIEW_HALF_ANGLE_DEG = 75.0
VLN_WAYPOINT_ANCHOR_NMS_DISTANCE_M = 1.0
VLN_WAYPOINT_ANCHOR_COMBINE_NMS_DISTANCE_M = 1.0
VLN_WAYPOINT_FINAL_MERGE_DISTANCE_M = 1.0
VLN_WAYPOINT_FRONTIER_BACKOFF_M = 0.45
VLN_WAYPOINT_VISIBLE_BACKTRACK_STEP_M = 0.10
VLN_WAYPOINT_SKELETON_MIN_CLEARANCE_M = 0.18
VLN_WAYPOINT_FRONTIER_ANCHOR_LIMIT = 30
VLN_WAYPOINT_SKELETON_ANCHOR_LIMIT = 30
VLN_WAYPOINT_COMBINED_ANCHOR_LIMIT = 60
VLN_WAYPOINT_NODE_DEDUP_RADIUS_M = 0.5
VLN_WAYPOINT_GLOBAL_NODE_DEDUP_RADIUS_M = 2.0


def _sampling_density_distance(
    default_distance_m: float,
    sample_spacing_m: float,
) -> float:
    spacing = float(sample_spacing_m)
    if spacing <= 0.0:
        raise ValueError("VLN waypoint sample spacing must be positive")
    return min(float(default_distance_m), spacing)


@dataclass(frozen=True)
class _VlnWaypointAnchor:
    xy: tuple[float, float]
    source: str
    score: float


@dataclass(frozen=True)
class VlnSampledWaypointCandidate:
    label: int
    goal_xy: tuple[float, float]
    path_xy: list[tuple[float, float]]
    point_pixel: tuple[float, float]
    point_2d: tuple[float, float]
    euclidean_distance_m: float
    path_length_m: float
    projected_depth_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": int(self.label),
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "point_pixel": [float(self.point_pixel[0]), float(self.point_pixel[1])],
            "point_2d": [float(self.point_2d[0]), float(self.point_2d[1])],
            "euclidean_distance_m": float(self.euclidean_distance_m),
            "path_length_m": float(self.path_length_m),
            "projected_depth_m": float(self.projected_depth_m),
        }


@dataclass(frozen=True)
class VlnProjectedSampledWaypointCandidate:
    view: "VisualViewContext"
    candidate: VlnSampledWaypointCandidate
    source: str


def _forward_xy_from_yaw(yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    return np.asarray([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)


def _angle_mask(
    xy: np.ndarray,
    robot_xy: np.ndarray,
    forward_xy: np.ndarray,
    half_angle_deg: float,
) -> np.ndarray:
    offsets = np.asarray(xy, dtype=np.float64).reshape(-1, 2) - np.asarray(robot_xy, dtype=np.float64).reshape(1, 2)
    distances = np.linalg.norm(offsets, axis=1)
    unit_offsets = offsets / np.clip(distances.reshape(-1, 1), 1e-8, None)
    cos_values = np.clip(unit_offsets @ np.asarray(forward_xy, dtype=np.float64).reshape(2, 1), -1.0, 1.0).reshape(-1)
    angles = np.degrees(np.arccos(cos_values))
    return angles <= float(half_angle_deg)


def _project_waypoint_points(
    *,
    xy: np.ndarray,
    world_z: float,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: np.ndarray,
    occlusion_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    z = np.full((len(xy), 1), float(world_z), dtype=np.float32)
    points_odom = np.concatenate([xy.astype(np.float32), z], axis=1)
    projected_points, projected_depths, valid = _project_points(
        points_odom=points_odom,
        T_cam_odom=T_cam_odom,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
    )
    unoccluded = _compute_unoccluded_mask(
        projected_points=projected_points,
        projected_depths=projected_depths,
        valid_projection=valid,
        depth_image=depth,
        depth_tolerance_m=float(occlusion_tolerance_m),
    )
    return valid & unoccluded, projected_points, projected_depths


def _visible_traversable_pixels(
    *,
    map_obj,
    robot_xy: np.ndarray,
    observation,
    world_z: float,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: np.ndarray,
    min_distance_m: float,
    occlusion_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traversable = map_obj._traversable_map()
    local_ys, local_xs = np.nonzero(traversable)
    if len(local_xs) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=bool),
        )

    pixel_xy = np.stack([local_xs, local_ys], axis=1).astype(np.int32)
    xy = map_obj.px_to_xy(pixel_xy.astype(np.float64))
    distances = np.linalg.norm(xy - np.asarray(robot_xy, dtype=np.float64).reshape(1, 2), axis=1)
    distance_mask = distances >= float(min_distance_m)
    forward_xy = _forward_xy_from_yaw(float(getattr(observation.pose, "yaw", 0.0)))
    view_mask = _angle_mask(
        xy,
        robot_xy,
        forward_xy,
        half_angle_deg=VLN_WAYPOINT_VIEW_HALF_ANGLE_DEG,
    )
    visible_mask, _projected, _depths = _project_waypoint_points(
        xy=xy,
        world_z=world_z,
        T_cam_odom=T_cam_odom,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
        depth=depth,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )
    keep = distance_mask & view_mask & visible_mask
    return pixel_xy, xy, keep


def _frontier_polyline_midpoint(points_px: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((2,), dtype=np.float64)
    if len(points) == 1:
        return points[0]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-6:
        return points[len(points) // 2]
    target = total_length * 0.5
    cumulative = 0.0
    for index, segment_length in enumerate(lengths):
        next_cumulative = cumulative + float(segment_length)
        if next_cumulative >= target:
            ratio = (target - cumulative) / max(float(segment_length), 1e-6)
            return points[index] * (1.0 - ratio) + points[index + 1] * ratio
        cumulative = next_cumulative
    return points[-1]


def _local_frontier_segments_px(map_obj) -> list[np.ndarray]:
    if hasattr(map_obj, "map1_free_map") and hasattr(map_obj, "map2_free_map"):
        full_map = (~np.asarray(map_obj.obstacle_map, dtype=bool)).astype(np.uint8)
        explored_mask = np.asarray(map_obj.map1_free_map, dtype=bool).astype(np.uint8)
        explored_mask = explored_mask.copy()
        explored_mask[full_map == 0] = 0
        filtered_explored_mask = filter_out_small_unexplored(
            full_map,
            explored_mask,
            int(map_obj.area_thresh_in_pixels),
        )
        contours, _hierarchy = cv2.findContours(
            filtered_explored_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
        unexplored_mask = cv2.blur(
            np.where(unexplored_mask > 0, 255, unexplored_mask),
            (3, 3),
        )
        frontier_point_mask = np.asarray(unexplored_mask, dtype=np.uint8).copy()
        frontier_point_mask[~np.asarray(map_obj.map2_free_map, dtype=bool)] = 0

        raw_frontiers: list[np.ndarray] = []
        for contour in contours:
            raw_frontiers.extend(
                contour_to_frontiers(
                    interpolate_contour(contour),
                    frontier_point_mask,
                )
            )
        segments: list[np.ndarray] = []
        for frontier in raw_frontiers:
            segments.extend(_split_frontier_by_turn_angle(frontier))
        return [np.asarray(segment, dtype=np.float64).reshape(-1, 2) for segment in segments]

    _frontiers_px, clusters = map_obj.detect_frontier_clusters_px()
    segments = []
    for cluster in clusters:
        segments.extend(_split_frontier_by_turn_angle(np.asarray(cluster, dtype=np.float64).reshape(-1, 2)))
    return [np.asarray(segment, dtype=np.float64).reshape(-1, 2) for segment in segments]


def _nms_anchors(
    anchors: list[_VlnWaypointAnchor],
    *,
    min_distance_m: float,
    limit: int,
) -> list[_VlnWaypointAnchor]:
    selected: list[_VlnWaypointAnchor] = []
    for anchor in sorted(anchors, key=lambda item: (-float(item.score), str(item.source))):
        xy = np.asarray(anchor.xy, dtype=np.float64)
        if any(float(np.linalg.norm(xy - np.asarray(item.xy, dtype=np.float64))) < float(min_distance_m) for item in selected):
            continue
        selected.append(anchor)
        if len(selected) >= int(limit):
            break
    return selected


def _frontier_anchors(
    *,
    map_obj,
    robot_xy: np.ndarray,
    observation,
    min_distance_m: float,
    anchor_spacing_m: float,
) -> list[_VlnWaypointAnchor]:
    forward_xy = _forward_xy_from_yaw(float(getattr(observation.pose, "yaw", 0.0)))
    anchors: list[_VlnWaypointAnchor] = []
    for segment in _local_frontier_segments_px(map_obj):
        if len(segment) < 2:
            continue
        midpoint_px = _frontier_polyline_midpoint(segment)
        midpoint_xy = map_obj.px_to_xy(midpoint_px.reshape(1, 2))[0]
        distance = float(np.linalg.norm(midpoint_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
        if distance < float(min_distance_m):
            continue
        if not bool(_angle_mask(midpoint_xy.reshape(1, 2), robot_xy, forward_xy, VLN_WAYPOINT_VIEW_HALF_ANGLE_DEG)[0]):
            continue
        segment_length_m = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()) / float(map_obj.pixels_per_meter)
        anchors.append(
            _VlnWaypointAnchor(
                xy=(float(midpoint_xy[0]), float(midpoint_xy[1])),
                source="frontier",
                score=1000.0 + segment_length_m,
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=VLN_WAYPOINT_FRONTIER_ANCHOR_LIMIT,
    )


def _skeleton_anchors(
    *,
    map_obj,
    visible_px: np.ndarray,
    visible_keep: np.ndarray,
    anchor_spacing_m: float,
) -> list[_VlnWaypointAnchor]:
    if len(visible_px) == 0 or not bool(np.any(visible_keep)):
        return []
    kept_px = np.asarray(visible_px, dtype=np.int32)[np.asarray(visible_keep, dtype=bool)]
    x_min = max(0, int(np.min(kept_px[:, 0])) - 2)
    x_max = min(map_obj.size - 1, int(np.max(kept_px[:, 0])) + 2)
    y_min = max(0, int(np.min(kept_px[:, 1])) - 2)
    y_max = min(map_obj.size - 1, int(np.max(kept_px[:, 1])) + 2)

    visible_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    visible_mask[kept_px[:, 1] - y_min, kept_px[:, 0] - x_min] = True
    if not bool(np.any(visible_mask)):
        return []

    skeleton, distance = medial_axis(visible_mask, return_distance=True)
    ys, xs = np.nonzero(skeleton)
    min_clearance_px = float(VLN_WAYPOINT_SKELETON_MIN_CLEARANCE_M) * float(map_obj.pixels_per_meter)
    anchors: list[_VlnWaypointAnchor] = []
    for x, y in zip(xs, ys):
        if float(distance[y, x]) < min_clearance_px:
            continue
        px = np.asarray([[x + x_min, y + y_min]], dtype=np.float64)
        xy = map_obj.px_to_xy(px)[0]
        anchors.append(
            _VlnWaypointAnchor(
                xy=(float(xy[0]), float(xy[1])),
                source="skeleton",
                score=500.0 + float(distance[y, x]) / float(map_obj.pixels_per_meter),
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=VLN_WAYPOINT_SKELETON_ANCHOR_LIMIT,
    )


def _registered_frontier_anchors_unified(
    *,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    robot_xy: np.ndarray,
    min_distance_m: float,
    anchor_spacing_m: float,
) -> list[_VlnWaypointAnchor]:
    anchors: list[_VlnWaypointAnchor] = []
    for record in list((frontier_records or {}).values()):
        goal_xy = np.asarray(record.goal_xy, dtype=np.float64).reshape(2)
        if not bool(np.all(np.isfinite(goal_xy))):
            continue
        distance = float(np.linalg.norm(goal_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
        if distance < float(min_distance_m):
            continue
        anchors.append(
            _VlnWaypointAnchor(
                xy=(float(goal_xy[0]), float(goal_xy[1])),
                source="registered_frontier",
                score=1000.0 + float(record.score_mean),
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=VLN_WAYPOINT_FRONTIER_ANCHOR_LIMIT,
    )


def _skeleton_anchors_unified(
    *,
    map_obj,
    robot_xy: np.ndarray,
    min_distance_m: float,
    anchor_spacing_m: float,
) -> list[_VlnWaypointAnchor]:
    traversable = np.asarray(map_obj._traversable_map(), dtype=bool)
    traversable_ys, traversable_xs = np.nonzero(traversable)
    if len(traversable_xs) == 0:
        return []
    y_min = max(0, int(np.min(traversable_ys)) - 2)
    y_max = min(map_obj.size - 1, int(np.max(traversable_ys)) + 2)
    x_min = max(0, int(np.min(traversable_xs)) - 2)
    x_max = min(map_obj.size - 1, int(np.max(traversable_xs)) + 2)
    crop = traversable[y_min : y_max + 1, x_min : x_max + 1].copy()
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return []
    pixel_xy = np.stack([xs + x_min, ys + y_min], axis=1).astype(np.float64)
    xy = map_obj.px_to_xy(pixel_xy)
    distances = np.linalg.norm(xy - np.asarray(robot_xy, dtype=np.float64).reshape(1, 2), axis=1)
    keep = distances >= float(min_distance_m)
    filtered = np.zeros_like(crop, dtype=bool)
    filtered[ys[keep], xs[keep]] = True
    if not bool(np.any(filtered)):
        return []

    skeleton, distance = medial_axis(filtered, return_distance=True)
    skeleton_ys, skeleton_xs = np.nonzero(skeleton)
    min_clearance_px = float(VLN_WAYPOINT_SKELETON_MIN_CLEARANCE_M) * float(map_obj.pixels_per_meter)
    anchors: list[_VlnWaypointAnchor] = []
    for x, y in zip(skeleton_xs, skeleton_ys):
        if float(distance[y, x]) < min_clearance_px:
            continue
        point_px = np.asarray([[x + x_min, y + y_min]], dtype=np.float64)
        point_xy = map_obj.px_to_xy(point_px)[0]
        anchors.append(
            _VlnWaypointAnchor(
                xy=(float(point_xy[0]), float(point_xy[1])),
                source="skeleton",
                score=500.0 + float(distance[y, x]) / float(map_obj.pixels_per_meter),
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=VLN_WAYPOINT_SKELETON_ANCHOR_LIMIT,
    )


def _path_length_m(path_xy: np.ndarray) -> float:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _sample_path_point(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) == 0:
        raise ValueError("path must not be empty")
    if len(path) == 1:
        return path[0]
    remaining = float(distance_m)
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            continue
        if remaining <= segment_length:
            return start + segment * (remaining / segment_length)
        remaining -= segment_length
    return path[-1]


def _path_prefix_to_distance(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) <= 1:
        return path
    remaining = float(distance_m)
    pieces = [path[0]]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            continue
        if remaining <= segment_length:
            pieces.append(start + segment * (remaining / segment_length))
            return np.asarray(pieces, dtype=np.float64)
        pieces.append(end)
        remaining -= segment_length
    return path


def _select_visible_path_waypoint(
    *,
    path_xy: np.ndarray,
    preferred_distance_m: float,
    world_z: float,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: np.ndarray,
    min_distance_m: float,
    max_distance_m: float,
    occlusion_tolerance_m: float,
) -> tuple[np.ndarray, tuple[float, float], float, float] | None:
    path_length = _path_length_m(path_xy)
    distance = min(float(preferred_distance_m), path_length, float(max_distance_m))
    while distance >= float(min_distance_m):
        waypoint_xy = _sample_path_point(path_xy, distance)
        visible, projected, projected_depths = _project_waypoint_points(
            xy=waypoint_xy.reshape(1, 2),
            world_z=world_z,
            T_cam_odom=T_cam_odom,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
            depth=depth,
            occlusion_tolerance_m=occlusion_tolerance_m,
        )
        if len(visible) > 0 and bool(visible[0]):
            return (
                waypoint_xy,
                (float(projected[0, 0]), float(projected[0, 1])),
                float(projected_depths[0]),
                float(distance),
            )
        distance -= float(VLN_WAYPOINT_VISIBLE_BACKTRACK_STEP_M)
    return None


def _candidate_from_anchor(
    *,
    map_obj,
    robot_xy: np.ndarray,
    anchor: _VlnWaypointAnchor,
    raw_label: int,
    cost_map: np.ndarray,
    world_z: float,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: np.ndarray,
    min_distance_m: float,
    max_distance_m: float,
    occlusion_tolerance_m: float,
) -> VlnSampledWaypointCandidate | None:
    path_xy_arr = map_obj.compute_astar_path(
        robot_xy,
        np.asarray(anchor.xy, dtype=np.float64),
        cost_map=cost_map,
    )
    if path_xy_arr is None or len(path_xy_arr) < 2:
        return None
    path_length = _path_length_m(path_xy_arr)
    if path_length < float(min_distance_m):
        return None
    if anchor.source == "frontier":
        preferred_distance_m = max(float(min_distance_m), path_length - float(VLN_WAYPOINT_FRONTIER_BACKOFF_M))
    else:
        preferred_distance_m = path_length

    selected = _select_visible_path_waypoint(
        path_xy=path_xy_arr,
        preferred_distance_m=preferred_distance_m,
        world_z=world_z,
        T_cam_odom=T_cam_odom,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
        depth=depth,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )
    if selected is None:
        return None
    waypoint_xy, px, projected_depth, distance_along_path = selected
    euclidean = float(np.linalg.norm(waypoint_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
    if euclidean < float(min_distance_m) or euclidean > float(max_distance_m):
        return None
    path_prefix = _path_prefix_to_distance(path_xy_arr, distance_along_path)
    return VlnSampledWaypointCandidate(
        label=int(raw_label),
        goal_xy=(float(waypoint_xy[0]), float(waypoint_xy[1])),
        path_xy=[(float(item[0]), float(item[1])) for item in np.asarray(path_prefix, dtype=np.float64)],
        point_pixel=px,
        point_2d=(
            float(px[0] / float(max(1, image_width - 1))),
            float(px[1] / float(max(1, image_height - 1))),
        ),
        euclidean_distance_m=euclidean,
        path_length_m=float(distance_along_path),
        projected_depth_m=float(projected_depth),
    )


def _anchor_source_priority(source: str) -> int:
    if source in {"frontier", "registered_frontier"}:
        return 0
    if source == "skeleton":
        return 1
    return 9


def _is_near_avoid_node(
    xy: tuple[float, float] | np.ndarray,
    avoid_node_xys: list[tuple[float, float]] | None,
    radius_m: float,
    node_dedup_map: "GlobalBEVMap | None" = None,
) -> bool:
    nodes = [] if avoid_node_xys is None else list(avoid_node_xys)
    if nodes == []:
        return False
    point = np.asarray(xy, dtype=np.float64).reshape(2)
    for node_xy in nodes:
        node_point = np.asarray(node_xy, dtype=np.float64).reshape(2)
        distance_m = float(np.linalg.norm(point - node_point))
        if node_dedup_map is None:
            if distance_m < float(radius_m):
                return True
            continue
        if distance_m > float(radius_m):
            continue
        stats = node_dedup_map.segment_known_free_stats(
            start_xy=node_point,
            end_xy=point,
        )
        if stats is None:
            continue
        sample_count = int(stats["sample_count"])
        if (
            sample_count > 0
            and int(stats["known_free_count"]) == sample_count
            and int(stats["obstacle_hit_count"]) == 0
        ):
            return True
    return False


def generate_vln_sampled_waypoint_candidates(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    selected_view: "VisualViewContext",
    robot_xy: np.ndarray,
    world_z: float,
    sample_spacing_m: float = VLN_WAYPOINT_SAMPLE_SPACING_M,
    min_distance_m: float = VLN_WAYPOINT_MIN_DISTANCE_M,
    max_distance_m: float = VLN_WAYPOINT_MAX_DISTANCE_M,
    max_candidates: int = VLN_WAYPOINT_MAX_CANDIDATES,
    occlusion_tolerance_m: float = VLN_WAYPOINT_OCCLUSION_TOLERANCE_M,
) -> list[VlnSampledWaypointCandidate]:
    observation = cache.get_observation(str(selected_view.obs_id)).observation
    if observation.rgb is None:
        raise ValueError("sampled waypoint generation requires observation.rgb")
    if observation.depth is None:
        raise ValueError("sampled waypoint generation requires observation.depth")
    if observation.intrinsics is None:
        raise ValueError("sampled waypoint generation requires observation.intrinsics")
    if observation.T_cam_odom is None:
        raise ValueError("sampled waypoint generation requires observation.T_cam_odom")

    rgb = np.asarray(observation.rgb, dtype=np.uint8)
    depth = np.asarray(observation.depth)
    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    image_height, image_width = rgb.shape[:2]

    map_obj = exploration.map
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    anchor_spacing_m = _sampling_density_distance(
        VLN_WAYPOINT_ANCHOR_NMS_DISTANCE_M,
        sample_spacing_m,
    )
    combined_anchor_spacing_m = _sampling_density_distance(
        VLN_WAYPOINT_ANCHOR_COMBINE_NMS_DISTANCE_M,
        sample_spacing_m,
    )
    final_merge_distance_m = _sampling_density_distance(
        VLN_WAYPOINT_FINAL_MERGE_DISTANCE_M,
        sample_spacing_m,
    )
    visible_px, _visible_xy, visible_keep = _visible_traversable_pixels(
        map_obj=map_obj,
        robot_xy=robot_xy,
        observation=observation,
        world_z=world_z,
        T_cam_odom=T_cam_odom,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
        depth=depth,
        min_distance_m=min_distance_m,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )
    anchors: list[_VlnWaypointAnchor] = []
    anchors.extend(
        _frontier_anchors(
            map_obj=map_obj,
            robot_xy=robot_xy,
            observation=observation,
            min_distance_m=min_distance_m,
            anchor_spacing_m=anchor_spacing_m,
        )
    )
    anchors.extend(
        _skeleton_anchors(
            map_obj=map_obj,
            visible_px=visible_px,
            visible_keep=visible_keep,
            anchor_spacing_m=anchor_spacing_m,
        )
    )
    anchors = _nms_anchors(
        anchors,
        min_distance_m=combined_anchor_spacing_m,
        limit=VLN_WAYPOINT_COMBINED_ANCHOR_LIMIT,
    )
    cost_map = map_obj._build_astar_cost_map()

    raw_candidates: list[VlnSampledWaypointCandidate] = []
    source_by_label: dict[int, str] = {}
    score_by_label: dict[int, float] = {}
    for raw_label, anchor in enumerate(anchors, start=1):
        candidate = _candidate_from_anchor(
            map_obj=map_obj,
            robot_xy=robot_xy,
            anchor=anchor,
            raw_label=raw_label,
            cost_map=cost_map,
            world_z=world_z,
            T_cam_odom=T_cam_odom,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
            depth=depth,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            occlusion_tolerance_m=occlusion_tolerance_m,
        )
        if candidate is None:
            continue
        raw_candidates.append(candidate)
        source_by_label[int(candidate.label)] = str(anchor.source)
        score_by_label[int(candidate.label)] = float(anchor.score)

    selected: list[VlnSampledWaypointCandidate] = []
    for candidate in sorted(
        raw_candidates,
        key=lambda item: (
            _anchor_source_priority(source_by_label.get(int(item.label), "")),
            -float(score_by_label.get(int(item.label), 0.0)),
            float(item.path_length_m),
            abs(float(item.point_2d[0]) - 0.5),
        ),
    ):
        goal_xy = np.asarray(candidate.goal_xy, dtype=np.float64)
        if any(
            float(np.linalg.norm(goal_xy - np.asarray(existing.goal_xy, dtype=np.float64)))
            < float(final_merge_distance_m)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(max_candidates):
            break

    selected = sorted(selected, key=_candidate_label_sort_key)
    return [
        VlnSampledWaypointCandidate(
            label=index,
            goal_xy=candidate.goal_xy,
            path_xy=candidate.path_xy,
            point_pixel=candidate.point_pixel,
            point_2d=candidate.point_2d,
            euclidean_distance_m=candidate.euclidean_distance_m,
            path_length_m=candidate.path_length_m,
            projected_depth_m=candidate.projected_depth_m,
        )
        for index, candidate in enumerate(selected, start=1)
    ]


def _projection_view_score(
    *,
    point_pixel: tuple[float, float],
    image_width: int,
    image_height: int,
    projected_depth_m: float,
) -> float:
    x = float(point_pixel[0])
    y = float(point_pixel[1])
    border_margin = min(x, y, float(image_width - 1) - x, float(image_height - 1) - y)
    center_bonus = 1.0 - abs((x / max(1.0, float(image_width - 1))) - 0.5)
    lower_half_bonus = y / max(1.0, float(image_height - 1))
    depth_penalty = 0.01 * float(projected_depth_m)
    return float(border_margin * 0.01 + center_bonus + 0.5 * lower_half_bonus - depth_penalty)


def _projected_views_for_waypoint(
    *,
    waypoint_xy: np.ndarray,
    views: list["VisualViewContext"],
    cache: "RuntimeCache",
    world_z: float,
    occlusion_tolerance_m: float,
) -> list[tuple[float, "VisualViewContext", tuple[float, float], float]]:
    projections: list[
        tuple[float, "VisualViewContext", tuple[float, float], float]
    ] = []
    for view in views:
        observation = cache.get_observation(str(view.obs_id)).observation
        if observation.rgb is None:
            raise ValueError("unified sampled waypoint generation requires observation.rgb")
        if observation.depth is None:
            raise ValueError("unified sampled waypoint generation requires observation.depth")
        if observation.intrinsics is None:
            raise ValueError("unified sampled waypoint generation requires observation.intrinsics")
        if observation.T_cam_odom is None:
            raise ValueError("unified sampled waypoint generation requires observation.T_cam_odom")

        rgb = np.asarray(observation.rgb, dtype=np.uint8)
        depth = np.asarray(observation.depth)
        image_height, image_width = rgb.shape[:2]
        visible, projected_points, projected_depths = _project_waypoint_points(
            xy=np.asarray(waypoint_xy, dtype=np.float64).reshape(1, 2),
            world_z=float(world_z),
            T_cam_odom=np.asarray(observation.T_cam_odom, dtype=np.float32),
            intrinsics=np.asarray(observation.intrinsics, dtype=np.float32),
            image_width=int(image_width),
            image_height=int(image_height),
            depth=depth,
            occlusion_tolerance_m=float(occlusion_tolerance_m),
        )
        if len(visible) == 0 or not bool(visible[0]):
            continue
        point_pixel = (float(projected_points[0, 0]), float(projected_points[0, 1]))
        projected_depth = float(projected_depths[0])
        score = _projection_view_score(
            point_pixel=point_pixel,
            image_width=int(image_width),
            image_height=int(image_height),
            projected_depth_m=projected_depth,
        )
        projections.append((score, view, point_pixel, projected_depth))
    return projections


def _best_projected_view_for_waypoint(
    *,
    waypoint_xy: np.ndarray,
    views: list["VisualViewContext"],
    cache: "RuntimeCache",
    world_z: float,
    occlusion_tolerance_m: float,
) -> tuple["VisualViewContext", tuple[float, float], float] | None:
    projections = _projected_views_for_waypoint(
        waypoint_xy=waypoint_xy,
        views=views,
        cache=cache,
        world_z=float(world_z),
        occlusion_tolerance_m=float(occlusion_tolerance_m),
    )
    if projections == []:
        return None
    _score, view, point_pixel, projected_depth = max(
        projections,
        key=lambda item: float(item[0]),
    )
    return view, point_pixel, projected_depth


def _candidate_from_anchor_best_view(
    *,
    map_obj,
    robot_xy: np.ndarray,
    anchor: _VlnWaypointAnchor,
    raw_label: int,
    cost_map: np.ndarray,
    views: list["VisualViewContext"],
    cache: "RuntimeCache",
    world_z: float,
    min_distance_m: float,
    max_distance_m: float,
    occlusion_tolerance_m: float,
) -> tuple[VlnSampledWaypointCandidate, "VisualViewContext"] | None:
    path_xy_arr = map_obj.compute_astar_path(
        robot_xy,
        np.asarray(anchor.xy, dtype=np.float64),
        cost_map=cost_map,
    )
    if path_xy_arr is None or len(path_xy_arr) < 2:
        return None
    path_length = _path_length_m(path_xy_arr)
    if path_length < float(min_distance_m):
        return None
    if anchor.source == "frontier":
        preferred_distance_m = max(float(min_distance_m), path_length - float(VLN_WAYPOINT_FRONTIER_BACKOFF_M))
    else:
        preferred_distance_m = path_length

    distance = min(float(preferred_distance_m), path_length, float(max_distance_m))
    while distance >= float(min_distance_m):
        waypoint_xy = _sample_path_point(path_xy_arr, distance)
        best_projection = _best_projected_view_for_waypoint(
            waypoint_xy=waypoint_xy,
            views=views,
            cache=cache,
            world_z=float(world_z),
            occlusion_tolerance_m=float(occlusion_tolerance_m),
        )
        if best_projection is not None:
            view, point_pixel, projected_depth = best_projection
            observation = cache.get_observation(str(view.obs_id)).observation
            image_height, image_width = np.asarray(observation.rgb).shape[:2]
            euclidean = float(np.linalg.norm(waypoint_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
            if euclidean < float(min_distance_m) or euclidean > float(max_distance_m):
                return None
            path_prefix = _path_prefix_to_distance(path_xy_arr, distance)
            candidate = VlnSampledWaypointCandidate(
                label=int(raw_label),
                goal_xy=(float(waypoint_xy[0]), float(waypoint_xy[1])),
                path_xy=[(float(item[0]), float(item[1])) for item in np.asarray(path_prefix, dtype=np.float64)],
                point_pixel=point_pixel,
                point_2d=(
                    float(point_pixel[0] / float(max(1, image_width - 1))),
                    float(point_pixel[1] / float(max(1, image_height - 1))),
                ),
                euclidean_distance_m=euclidean,
                path_length_m=float(distance),
                projected_depth_m=float(projected_depth),
            )
            return candidate, view
        distance -= float(VLN_WAYPOINT_VISIBLE_BACKTRACK_STEP_M)
    return None


def generate_vln_unified_sampled_waypoint_candidates(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    views: list["VisualViewContext"],
    robot_xy: np.ndarray,
    world_z: float,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None = None,
    sample_spacing_m: float = VLN_WAYPOINT_SAMPLE_SPACING_M,
    min_distance_m: float = VLN_WAYPOINT_MIN_DISTANCE_M,
    max_distance_m: float = VLN_WAYPOINT_MAX_DISTANCE_M,
    max_candidates: int = VLN_WAYPOINT_MAX_CANDIDATES,
    occlusion_tolerance_m: float = VLN_WAYPOINT_OCCLUSION_TOLERANCE_M,
    avoid_node_xys: list[tuple[float, float]] | None = None,
    node_dedup_radius_m: float = VLN_WAYPOINT_NODE_DEDUP_RADIUS_M,
    node_dedup_map: "GlobalBEVMap | None" = None,
) -> list[VlnProjectedSampledWaypointCandidate]:
    if views == []:
        raise ValueError("unified sampled waypoint generation requires at least one view")
    map_obj = exploration.map
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    anchor_spacing_m = _sampling_density_distance(
        VLN_WAYPOINT_ANCHOR_NMS_DISTANCE_M,
        sample_spacing_m,
    )
    combined_anchor_spacing_m = _sampling_density_distance(
        VLN_WAYPOINT_ANCHOR_COMBINE_NMS_DISTANCE_M,
        sample_spacing_m,
    )
    final_merge_distance_m = _sampling_density_distance(
        VLN_WAYPOINT_FINAL_MERGE_DISTANCE_M,
        sample_spacing_m,
    )
    anchors = [
        *_registered_frontier_anchors_unified(
            frontier_records=frontier_records,
            robot_xy=robot_xy,
            min_distance_m=min_distance_m,
            anchor_spacing_m=anchor_spacing_m,
        ),
        *_skeleton_anchors_unified(
            map_obj=map_obj,
            robot_xy=robot_xy,
            min_distance_m=min_distance_m,
            anchor_spacing_m=anchor_spacing_m,
        ),
    ]
    anchors = _nms_anchors(
        anchors,
        min_distance_m=combined_anchor_spacing_m,
        limit=VLN_WAYPOINT_COMBINED_ANCHOR_LIMIT,
    )
    cost_map = map_obj._build_astar_cost_map()

    raw_candidates: list[tuple[VlnSampledWaypointCandidate, "VisualViewContext", str, float]] = []
    for raw_label, anchor in enumerate(anchors, start=1):
        projected = _candidate_from_anchor_best_view(
            map_obj=map_obj,
            robot_xy=robot_xy,
            anchor=anchor,
            raw_label=raw_label,
            cost_map=cost_map,
            views=views,
            cache=cache,
            world_z=float(world_z),
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            occlusion_tolerance_m=occlusion_tolerance_m,
        )
        if projected is None:
            continue
        candidate, view = projected
        raw_candidates.append((candidate, view, str(anchor.source), float(anchor.score)))

    selected: list[tuple[VlnSampledWaypointCandidate, "VisualViewContext", str, float]] = []
    for candidate, view, source, score in sorted(
        raw_candidates,
        key=lambda item: (
            _anchor_source_priority(item[2]),
            -float(item[3]),
            float(item[0].path_length_m),
            abs(float(item[0].point_2d[0]) - 0.5),
        ),
    ):
        goal_xy = np.asarray(candidate.goal_xy, dtype=np.float64)
        if _is_near_avoid_node(
            goal_xy,
            avoid_node_xys=avoid_node_xys,
            radius_m=float(node_dedup_radius_m),
            node_dedup_map=node_dedup_map,
        ):
            continue
        if any(
            float(np.linalg.norm(goal_xy - np.asarray(existing[0].goal_xy, dtype=np.float64)))
            < float(final_merge_distance_m)
            for existing in selected
        ):
            continue
        selected.append((candidate, view, source, score))
        if len(selected) >= int(max_candidates):
            break

    view_order = {int(view.angle_deg): index for index, view in enumerate(views)}
    selected = sorted(
        selected,
        key=lambda item: (
            view_order.get(int(item[1].angle_deg), 999),
            _candidate_label_sort_key(item[0]),
        ),
    )
    projected_selected: list[VlnProjectedSampledWaypointCandidate] = []
    for label, (candidate, _best_view, source, _score) in enumerate(
        selected,
        start=1,
    ):
        projections = _projected_views_for_waypoint(
            waypoint_xy=np.asarray(candidate.goal_xy, dtype=np.float64),
            views=views,
            cache=cache,
            world_z=float(world_z),
            occlusion_tolerance_m=float(occlusion_tolerance_m),
        )
        for _projection_score, view, point_pixel, projected_depth in projections:
            observation = cache.get_observation(str(view.obs_id)).observation
            image_height, image_width = np.asarray(observation.rgb).shape[:2]
            projected_selected.append(
                VlnProjectedSampledWaypointCandidate(
                    view=view,
                    candidate=VlnSampledWaypointCandidate(
                        label=int(label),
                        goal_xy=candidate.goal_xy,
                        path_xy=candidate.path_xy,
                        point_pixel=point_pixel,
                        point_2d=(
                            float(point_pixel[0] / float(max(1, image_width - 1))),
                            float(point_pixel[1] / float(max(1, image_height - 1))),
                        ),
                        euclidean_distance_m=candidate.euclidean_distance_m,
                        path_length_m=candidate.path_length_m,
                        projected_depth_m=float(projected_depth),
                    ),
                    source=source,
                )
            )
    return sorted(
        projected_selected,
        key=lambda item: (
            view_order.get(int(item.view.angle_deg), 999),
            _candidate_label_sort_key(item.candidate),
        ),
    )


def draw_vln_sampled_waypoint_rgb_overlay(
    *,
    cache: "RuntimeCache",
    selected_view: "VisualViewContext",
    candidates: list[VlnSampledWaypointCandidate],
) -> np.ndarray:
    observation = cache.get_observation(str(selected_view.obs_id)).observation
    rgb = np.asarray(observation.rgb, dtype=np.uint8)
    resized = resize_rgb_to_fit(rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE)
    node_overlay_image_id = str(getattr(selected_view, "node_overlay_image_id", "")).strip()
    if node_overlay_image_id != "":
        render_rgb = np.asarray(cache.get_image(node_overlay_image_id).image, dtype=np.uint8)
    else:
        render_rgb = np.asarray(resized.image, dtype=np.uint8)
    scale_x = float(resized.metadata["scale_x"])
    scale_y = float(resized.metadata["scale_y"])
    canvas = Image.fromarray(render_rgb.copy()).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    render_height, render_width = render_rgb.shape[:2]
    marker_font, marker_radius_px = _rgb_frontier_circle_style(
        image_width=render_width,
        image_height=render_height,
    )
    for candidate in candidates:
        label = str(candidate.label)
        scaled_xy = scale_xy(candidate.point_pixel, scale_x=scale_x, scale_y=scale_y)
        marker_xy = _clamp_frontier_circle_marker_center(
            draw=draw,
            center_xy=scaled_xy,
            label=label,
            font=marker_font,
            radius_px=marker_radius_px,
            image_size=(render_width, render_height),
        )
        _draw_frontier_circle_marker(
            draw=draw,
            center_xy=marker_xy,
            label=label,
            font=marker_font,
            radius_px=marker_radius_px,
            image=canvas,
        )
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def draw_vln_sampled_waypoint_bev_overlay(
    *,
    exploration: "ExplorationManager",
    robot_xy: np.ndarray,
    candidates: list[VlnSampledWaypointCandidate],
    landmark_markers: list[dict[str, object]] | None = None,
    base_overlay: np.ndarray | None = None,
    base_transform: BevOverlayTransform | None = None,
    coordinate_exploration: "ExplorationManager | None" = None,
    reference_node_marker: dict[str, object] | None = None,
    show_cardinal_border: bool = False,
) -> np.ndarray:
    if base_overlay is not None:
        if base_transform is None or coordinate_exploration is None:
            raise ValueError(
                "sampled waypoint base BEV requires its coordinate transform and exploration"
            )
        image = Image.fromarray(
            np.asarray(base_overlay, dtype=np.uint8).copy()
        ).convert("RGBA")
        image_width, image_height = int(image.size[0]), int(image.size[1])
        node_style = place_node_marker_style(
            image_width=image_width,
            image_height=image_height,
            mode="bev",
        )
        if reference_node_marker is not None:
            reference_xy = reference_node_marker.get("xy")
            if isinstance(reference_xy, (list, tuple)) and len(reference_xy) == 2:
                reference_px = coordinate_exploration.map.xy_to_px(
                    np.asarray(
                        [float(reference_xy[0]), float(reference_xy[1])],
                        dtype=np.float64,
                    ).reshape(1, 2)
                )[0]
                draw_numbered_circle_marker(
                    image,
                    center_xy=base_transform.transform_px(
                        (float(reference_px[0]), float(reference_px[1]))
                    ),
                    label=str(reference_node_marker.get("label", "")),
                    style=node_style,
                )
        candidate_style = numbered_circle_marker_style(
            image_width=image_width,
            image_height=image_height,
            mode="bev",
        )
        for candidate in candidates:
            goal_px = coordinate_exploration.map.xy_to_px(
                np.asarray(candidate.goal_xy, dtype=np.float64).reshape(1, 2)
            )[0]
            draw_numbered_circle_marker(
                image,
                center_xy=base_transform.transform_px(
                    (float(goal_px[0]), float(goal_px[1]))
                ),
                label=str(candidate.label),
                style=candidate_style,
            )
        rendered = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if bool(show_cardinal_border):
            rendered, _metadata = add_cardinal_direction_border(rendered)
        return rendered

    background = np.asarray(exploration.render_global_bev_background(), dtype=np.uint8)
    map_obj = exploration.map
    robot_px = map_obj.xy_to_px(np.asarray(robot_xy, dtype=np.float64).reshape(1, 2))[0]
    marker_specs: list[tuple[tuple[float, float], str, tuple[int, ...]]] = []
    for candidate in candidates:
        goal_px = map_obj.xy_to_px(np.asarray(candidate.goal_xy, dtype=np.float64).reshape(1, 2))[0]
        marker_specs.append(
            (
                (float(goal_px[0]), float(goal_px[1])),
                str(candidate.label),
                FRONTIER_MARKER_FILL,
            )
        )
    landmark_marker_specs: list[tuple[tuple[float, float], str]] = []
    for marker in list(landmark_markers or []):
        xy = marker.get("xy")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            continue
        landmark_px = map_obj.xy_to_px(
            np.asarray([float(xy[0]), float(xy[1])], dtype=np.float64).reshape(1, 2)
        )[0]
        landmark_marker_specs.append(
            (
                (float(landmark_px[0]), float(landmark_px[1])),
                str(marker.get("label", "")),
            )
        )
    rendered = _render_scaled_overlay(
        background=background,
        feasible_mask=np.asarray(map_obj.free_map, dtype=bool),
        marker_specs=marker_specs,
        robot_center=(float(robot_px[0]), float(robot_px[1])),
        landmark_marker_specs=landmark_marker_specs,
    )
    if bool(show_cardinal_border):
        rendered, _metadata = add_cardinal_direction_border(rendered)
    return rendered


def visual_waypoint_from_sampled_candidate(
    *,
    selected_view: "VisualViewContext",
    candidate: VlnSampledWaypointCandidate,
    target: str = "",
    raw_world_z: float = 0.0,
) -> VisualWaypoint:
    return VisualWaypoint(
        obs_id=str(selected_view.obs_id),
        angle_deg=int(selected_view.angle_deg),
        point_2d=(float(candidate.point_2d[0]), float(candidate.point_2d[1])),
        point_pixel=(float(candidate.point_pixel[0]), float(candidate.point_pixel[1])),
        raw_world_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        goal_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        goal_yaw=0.0,
        path_xy=[(float(x), float(y)) for x, y in candidate.path_xy],
        waypoint_target=str(target),
        target=str(target),
        depth_m=float(candidate.projected_depth_m),
        raw_world_z=float(raw_world_z),
    )


def _candidate_pool_sort_key(candidate: VlnSampledWaypointCandidate) -> tuple[float, float, int]:
    return (
        float(candidate.path_length_m),
        abs(float(candidate.point_2d[0]) - 0.5),
        int(candidate.label),
    )


def _candidate_label_sort_key(candidate: VlnSampledWaypointCandidate) -> tuple[float, float, int]:
    return (
        float(candidate.point_2d[1]),
        float(candidate.point_2d[0]),
        int(candidate.label),
    )
