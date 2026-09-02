from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navclaw.env.interface import RawObservation
from navclaw.graph.graph import Graph
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.llm.image_preprocessing import scale_xy
from navclaw.runtime.cache import RuntimeCache

from navclaw.mapping.exploration.bev_map import FrontierCandidate, GlobalBEVMap
from navclaw.mapping.exploration.bev_visuals import (
    BEV_DILATED_OBSTACLE_COLOR,
    BEV_EXPLORED_FREE_COLOR,
    BEV_OBSTACLE_COLOR,
    BEV_UNKNOWN_BACKGROUND_COLOR,
    draw_numbered_circle_marker,
    place_node_marker_style,
)
from navclaw.mapping.exploration.frontier_buffer import FrontierBuffer
from navclaw.mapping.exploration.overlay_drawing import FRONTIER_SCORE_FONT
from navclaw.mapping.exploration.overlay_drawing import RGB_FRONTIER_BADGE_BACKOFF_M
from navclaw.mapping.exploration.overlay_drawing import RGB_FRONTIER_BADGE_FONT_SIZE_MAX_PX
from navclaw.mapping.exploration.overlay_drawing import _clamp_frontier_circle_marker_center
from navclaw.mapping.exploration.overlay_drawing import _crop_bev_overlay
from navclaw.mapping.exploration.overlay_drawing import _draw_dashed_line
from navclaw.mapping.exploration.overlay_drawing import _draw_frontier_badge
from navclaw.mapping.exploration.overlay_drawing import _draw_frontier_circle_marker
from navclaw.mapping.exploration.overlay_drawing import _draw_place_node_circle
from navclaw.mapping.exploration.overlay_drawing import _frontier_badge_font
from navclaw.mapping.exploration.overlay_drawing import _frontier_label_text
from navclaw.mapping.exploration.overlay_drawing import _rgb_frontier_circle_style


RGB_OVERLAY_MAX_REMAINING_PATH_RATIO = 0.25
RGB_FRONTIER_PROJECTION_HEIGHTS_M = (0.03, 0.07, 0.11, 0.15)
RGB_PLACE_NODE_PROJECTION_HEIGHTS_M = (0.03, 0.07, 0.11, 0.15)
RGB_PLACE_NODE_OCCLUSION_TOLERANCE_M = 0.10
FALLBACK_FRONTIER_MIN_DISTANCE_TO_AGENT_M = 0.5
ROOM_CATEGORY_VERTICAL_OFFSET_PX = 8
NODE_LABEL_DOT_RADIUS_PX = 4
BEV_SELECTION_LABEL_PADDING_X_PX = 8
BEV_SELECTION_LABEL_PADDING_Y_PX = 5
BEV_SELECTION_LABEL_OUTLINE_WIDTH_PX = 3


def _is_dashed_node_kind(node_kind: str) -> bool:
    return str(node_kind).strip() != "place"


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    stripped = str(node_id).strip()
    if stripped.startswith("n") and stripped[1:].isdigit():
        return (int(stripped[1:]), stripped)
    digits = "".join(char for char in stripped if char.isdigit())
    if digits != "":
        return (int(digits), stripped)
    return (10**9, stripped)


def _densify_path_xy(path_xy: list[tuple[float, float]], step_m: float = 0.05) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if len(path) <= 1:
        return path

    pieces: list[np.ndarray] = [path[0:1]]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        subdivisions = max(1, int(np.ceil(distance / step_m)))
        interpolation = np.linspace(start, end, subdivisions + 1, dtype=np.float32)[1:]
        pieces.append(interpolation)
    return np.concatenate(pieces, axis=0)


def _remaining_path_ratio_from_index(path_xy: np.ndarray, point_index: int) -> float:
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path_xy must have shape Nx2, got {path.shape}")
    if len(path) <= 1:
        return 0.0
    clamped_index = int(np.clip(int(point_index), 0, len(path) - 1))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        return 0.0
    remaining_length = float(segment_lengths[clamped_index:].sum())
    return remaining_length / total_length


def _path_index_before_distance(
    path_xy: np.ndarray,
    visible_run: np.ndarray,
    anchor_index: int,
    distance_m: float,
) -> int:
    path = np.asarray(path_xy, dtype=np.float32)
    run = np.asarray(visible_run, dtype=np.int64)
    if len(run) == 0 or float(distance_m) <= 0.0:
        return int(anchor_index)
    start_index = int(run[0])
    current_index = int(np.clip(int(anchor_index), start_index, len(path) - 1))
    moved_m = 0.0
    while current_index > start_index and moved_m < float(distance_m):
        moved_m += float(np.linalg.norm(path[current_index] - path[current_index - 1]))
        current_index -= 1
    return int(current_index)


def _centered_room_category_anchor(
    center_x: int,
    center_y: int,
    text_width: int,
    text_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    anchor_x = int(round(float(center_x) - float(text_width) / 2.0))
    anchor_y = int(round(float(center_y) + float(ROOM_CATEGORY_VERTICAL_OFFSET_PX)))
    max_anchor_x = max(0, int(image_width) - int(text_width))
    max_anchor_y = max(0, int(image_height) - int(text_height))
    anchor_x = min(max(0, anchor_x), max_anchor_x)
    anchor_y = min(max(0, anchor_y), max_anchor_y)
    return anchor_x, anchor_y


def _centered_node_label_anchor(
    center_x: int,
    center_y: int,
    text_width: int,
    text_height: int,
    image_width: int,
    image_height: int,
    stack_index: int = 0,
) -> tuple[int, int]:
    anchor_x = int(round(float(center_x) - float(text_width) / 2.0))
    anchor_y = int(
        round(
            float(center_y)
            + float(ROOM_CATEGORY_VERTICAL_OFFSET_PX)
            + float(stack_index) * float(text_height + 4)
        )
    )
    max_anchor_x = max(0, int(image_width) - int(text_width))
    max_anchor_y = max(0, int(image_height) - int(text_height))
    anchor_x = min(max(0, anchor_x), max_anchor_x)
    anchor_y = min(max(0, anchor_y), max_anchor_y)
    return anchor_x, anchor_y


def _last_visible_run(valid_mask: np.ndarray) -> np.ndarray:
    visible_indices = np.flatnonzero(valid_mask)
    if len(visible_indices) == 0:
        return np.zeros((0,), dtype=np.int64)

    end_index = int(visible_indices[-1])
    start_index = end_index
    while start_index > 0 and bool(valid_mask[start_index - 1]):
        start_index -= 1
    return np.arange(start_index, end_index + 1, dtype=np.int64)


def _as_depth_meters(depth: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth)
    if np.issubdtype(depth_array.dtype, np.integer):
        depth_array = depth_array.astype(np.float32)
        if float(np.nanmax(depth_array)) > 100.0:
            depth_array = depth_array / 1000.0
        return depth_array
    return depth_array.astype(np.float32)


def _project_points(
    points_odom: np.ndarray,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_odom) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    ones = np.ones((points_odom.shape[0], 1), dtype=np.float32)
    homo_points = np.concatenate([points_odom.astype(np.float32), ones], axis=1)
    points_cam = (T_cam_odom.astype(np.float32) @ homo_points.T).T[:, :3]
    depths = points_cam[:, 2]
    valid = depths > 1e-3

    projected = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    projected[:, 0] = fx * points_cam[:, 0] / np.clip(depths, 1e-6, None) + cx
    projected[:, 1] = fy * points_cam[:, 1] / np.clip(depths, 1e-6, None) + cy

    valid &= projected[:, 0] >= 0.0
    valid &= projected[:, 0] < float(image_width)
    valid &= projected[:, 1] >= 0.0
    valid &= projected[:, 1] < float(image_height)
    return projected, depths.astype(np.float32), valid


def _compute_unoccluded_mask(
    projected_points: np.ndarray,
    projected_depths: np.ndarray,
    valid_projection: np.ndarray,
    depth_image: np.ndarray,
    depth_tolerance_m: float = 0.0,
) -> np.ndarray:
    if len(projected_points) == 0:
        return np.zeros((0,), dtype=bool)

    depth_m = _as_depth_meters(depth_image)
    image_height, image_width = depth_m.shape[:2]
    unoccluded = np.zeros((len(projected_points),), dtype=bool)
    valid_indices = np.flatnonzero(valid_projection)
    for index in valid_indices:
        u = int(round(float(projected_points[index, 0])))
        v = int(round(float(projected_points[index, 1])))
        u = int(np.clip(u, 0, image_width - 1))
        v = int(np.clip(v, 0, image_height - 1))
        u_min = max(0, u - 1)
        u_max = min(image_width, u + 2)
        v_min = max(0, v - 1)
        v_max = min(image_height, v + 2)
        window = depth_m[v_min:v_max, u_min:u_max]
        valid_window = window[np.isfinite(window) & (window > 1e-3)]
        if valid_window.size == 0:
            continue
        observed_depth = float(np.median(valid_window))
        if float(projected_depths[index]) <= observed_depth + float(depth_tolerance_m):
            unoccluded[index] = True
    return unoccluded


@dataclass
class ExploreView:
    obs_id: str
    overlay_id: str
    frontier_ids: list[str]
    frontier_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": self.obs_id,
            "overlay_id": self.overlay_id,
            "frontier_ids": list(self.frontier_ids),
            "frontier_sources": {
                str(frontier_id): str(source)
                for frontier_id, source in self.frontier_sources.items()
            },
        }


@dataclass
class PlaceNodeOverlayView:
    obs_id: str
    overlay_id: str
    node_ids: list[str]
    edge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": str(self.obs_id),
            "overlay_id": str(self.overlay_id),
            "node_ids": [str(node_id) for node_id in self.node_ids],
            "edge_ids": [str(edge_id) for edge_id in self.edge_ids],
        }


@dataclass
class _ProjectedFrontierOverlay:
    obs_id: str
    frontier_id: str
    frontier_xy: tuple[float, float]
    projected_path: np.ndarray
    visible_run: np.ndarray
    end_xy: tuple[float, float]
    center_distance_sq: float


@dataclass
class _ProjectedPlaceNodeOverlay:
    obs_id: str
    node_id: str
    node_xy: tuple[float, float]
    end_xy: tuple[float, float]
    center_distance_sq: float
    distance_m: float


@dataclass
class _FrontierVisualEvidence:
    frontier_id: str
    obs_id: str
    frontier_xy: tuple[float, float]
    end_xy: tuple[float, float]
    center_distance_sq: float
    source: str


def _select_best_projected_frontiers(
    projected_frontiers: list[_ProjectedFrontierOverlay],
) -> list[_ProjectedFrontierOverlay]:
    best_by_frontier: dict[str, _ProjectedFrontierOverlay] = {}
    for projected in projected_frontiers:
        current = best_by_frontier.get(projected.frontier_id)
        if current is None or float(projected.center_distance_sq) < float(current.center_distance_sq):
            best_by_frontier[projected.frontier_id] = projected
    return list(best_by_frontier.values())


class ExplorationManager:
    def __init__(self, bev_map: GlobalBEVMap | None = None) -> None:
        self.map = GlobalBEVMap() if bev_map is None else bev_map
        self.frontier_buffer = FrontierBuffer()
        self.integrated_obs_ids: set[str] = set()

    def reset(self) -> None:
        self.map.reset()
        self.frontier_buffer.clear()
        self.integrated_obs_ids = set()

    def observe_observation(self, cache: RuntimeCache, obs_id: str) -> dict[str, object]:
        if obs_id in self.integrated_obs_ids:
            return {
                "obs_id": obs_id,
                "integrated": False,
                "frontier_count": 0,
                "frontiers": [],
            }

        observation = cache.get_observation(obs_id).observation
        self.map.update_from_observation(observation)
        self.integrated_obs_ids.add(obs_id)
        return {
            "obs_id": obs_id,
            "integrated": True,
            "frontier_count": 0,
            "frontiers": [],
        }

    def observe_raw_observation(self, observation: RawObservation) -> None:
        self.map.update_from_observation(observation)

    def merge_raw_layers_from_local_exploration(
        self,
        local_exploration: "ExplorationManager",
        *,
        obs_ids: list[str] | None = None,
    ) -> dict[str, object]:
        merge_method = getattr(self.map, "merge_raw_layers_from_local_map", None)
        if not callable(merge_method):
            raise TypeError(
                f"{type(self.map).__name__} does not support raw-layer localmap merge"
            )
        details = dict(merge_method(local_exploration.map))
        merged_obs_ids = [] if obs_ids is None else [str(obs_id) for obs_id in obs_ids]
        self.integrated_obs_ids.update(merged_obs_ids)
        details["merged_obs_count"] = int(len(merged_obs_ids))
        return details

    def refresh_frontiers(self, robot_xy: np.ndarray) -> list[dict[str, object]]:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        frontiers_xy = self.map.detect_global_frontiers(robot_xy)
        self.frontier_buffer.update(frontiers_xy)
        return [entry.to_dict() for entry in self.frontier_buffer.selectable_entries()]

    def build_reachable_frontier_candidates(
        self,
        robot_xy: np.ndarray,
    ) -> list[FrontierCandidate]:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        self.refresh_frontiers(robot_xy)
        candidates = self.map.query_reachable_frontiers(
            robot_xy=robot_xy,
            frontier_entries=self.frontier_buffer.selectable_entries(),
        )
        self.frontier_buffer.update_navigation_goals(candidates)
        return candidates

    def render_bev_graph_overlay(
        self,
        graph: Graph,
        robot_xy: np.ndarray,
        candidates: list[FrontierCandidate],
        frontier_owner_node_id_by_frontier_id: dict[str, str] | None = None,
    ) -> np.ndarray:
        return self._render_bev_graph_frontier_overlay(
            graph=graph,
            robot_xy=np.asarray(robot_xy, dtype=np.float64).reshape(2),
            candidates=list(candidates),
            frontier_owner_node_id_by_frontier_id=(
                None if frontier_owner_node_id_by_frontier_id is None else dict(frontier_owner_node_id_by_frontier_id)
            ),
        )

    def render_frontier_raw_overlay(
        self,
        robot_xy: np.ndarray,
    ) -> np.ndarray:
        robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        waypoints_px, raw_frontiers = self.map.detect_frontier_clusters_px()
        image = self._render_bev_background().copy()
        image[self.map.explored_area & self.map.navigable_map] = np.asarray(BEV_EXPLORED_FREE_COLOR, dtype=np.uint8)
        image[self.map.dilated_obstacles] = np.asarray(BEV_DILATED_OBSTACLE_COLOR, dtype=np.uint8)
        image[self.map.obstacle_map.astype(bool)] = np.asarray(BEV_OBSTACLE_COLOR, dtype=np.uint8)

        palette = [
            (255, 80, 80),
            (80, 200, 120),
            (80, 160, 255),
            (255, 180, 70),
            (220, 110, 255),
            (255, 255, 80),
            (80, 230, 230),
        ]
        frontier_entries = self.frontier_buffer.selectable_entries()
        for index, frontier_px in enumerate(raw_frontiers):
            color = palette[index % len(palette)]
            frontier_px_int = np.asarray(frontier_px, dtype=np.int32)
            valid = (
                (frontier_px_int[:, 0] >= 0)
                & (frontier_px_int[:, 0] < self.map.size)
                & (frontier_px_int[:, 1] >= 0)
                & (frontier_px_int[:, 1] < self.map.size)
            )
            frontier_px_int = frontier_px_int[valid]
            if len(frontier_px_int) > 0:
                image[frontier_px_int[:, 1], frontier_px_int[:, 0]] = np.asarray(color, dtype=np.uint8)

        rendered = Image.fromarray(image).convert("RGBA")
        draw = ImageDraw.Draw(rendered, "RGBA")
        robot_px = self.map.xy_to_px(robot_xy.reshape(1, 2))[0]
        draw.ellipse(
            (
                float(robot_px[0]) - 8,
                float(robot_px[1]) - 8,
                float(robot_px[0]) + 8,
                float(robot_px[1]) + 8,
            ),
            fill=(80, 160, 255),
            outline=(255, 255, 255),
            width=2,
        )
        for index, _frontier_px in enumerate(raw_frontiers):
            if index >= len(waypoints_px):
                continue
            color = palette[index % len(palette)]
            frontier_id = f"raw_{index}"
            if index < len(frontier_entries):
                frontier_id = str(frontier_entries[index].id)
            badge_center = (float(waypoints_px[index][0]), float(waypoints_px[index][1]))
            _draw_frontier_badge(draw, badge_center, frontier_id, image=rendered, mode="bev")
        return np.asarray(rendered.convert("RGB"), dtype=np.uint8)

    def render_frontier_annotated_views(
        self,
        cache: RuntimeCache,
        obs_ids: list[str],
        candidates: list[FrontierCandidate],
        projection_base_z_m: float,
        frontier_label_map: dict[str, str] | None = None,
        base_image_ids_by_obs_id: dict[str, str] | None = None,
        update_visual_evidence: bool = True,
        image_max_size: tuple[int, int] | None = None,
    ) -> list[ExploreView]:
        return self._render_views(
            cache=cache,
            obs_ids=list(obs_ids),
            candidates=list(candidates),
            projection_base_z_m=float(projection_base_z_m),
            frontier_label_map=None if frontier_label_map is None else dict(frontier_label_map),
            base_image_ids_by_obs_id=dict(base_image_ids_by_obs_id or {}),
            update_visual_evidence=bool(update_visual_evidence),
            image_max_size=image_max_size,
        )

    def render_place_node_annotated_views(
        self,
        cache: RuntimeCache,
        graph: Graph,
        obs_ids: list[str],
        floor_id: str | None = None,
        image_max_size: tuple[int, int] | None = None,
    ) -> list[PlaceNodeOverlayView]:
        floor_id_str = None if floor_id is None else str(floor_id)
        with graph.lock:
            place_nodes = [
                node
                for node in sorted(
                    graph.iter_nodes(floor_id=floor_id_str),
                    key=lambda item: _node_id_sort_key(str(item.id)),
                )
                if node.node_kind == "place"
            ]
        if place_nodes == []:
            return []

        views: list[PlaceNodeOverlayView] = []
        for obs_id in obs_ids:
            observation = cache.get_observation(str(obs_id)).observation
            if (
                observation.rgb is None
                or observation.depth is None
                or observation.intrinsics is None
                or observation.T_cam_odom is None
            ):
                continue

            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            depth = np.asarray(observation.depth)
            intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
            T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
            image_height, image_width = rgb.shape[:2]
            image_center_x = (float(image_width) - 1.0) / 2.0
            image_center_y = (float(image_height) - 1.0) / 2.0
            robot_xy = self._robot_xy_from_observation(observation)
            resized = resize_rgb_to_fit(rgb, max_size=image_max_size)
            render_rgb = np.asarray(resized.image, dtype=np.uint8)
            render_height, render_width = render_rgb.shape[:2]
            scale_x = float(resized.metadata["scale_x"])
            scale_y = float(resized.metadata["scale_y"])
            image = Image.fromarray(render_rgb.copy()).convert("RGBA")
            draw = ImageDraw.Draw(image, "RGBA")

            projected_nodes: list[_ProjectedPlaceNodeOverlay] = []
            for node in place_nodes:
                node_xy = (
                    float(node.position[0]),
                    float(node.position[1]),
                )
                node_z = float(node.position[2])
                node_point = np.asarray(
                    [
                        [
                            float(node_xy[0]),
                            float(node_xy[1]),
                            float(node_z),
                        ]
                    ],
                    dtype=np.float32,
                )
                projected, projected_depths, valid = _project_points(
                    points_odom=node_point,
                    T_cam_odom=T_cam_odom,
                    intrinsics=intrinsics,
                    image_width=image_width,
                    image_height=image_height,
                )
                unoccluded = _compute_unoccluded_mask(
                    projected_points=projected,
                    projected_depths=projected_depths,
                    valid_projection=valid,
                    depth_image=depth,
                    depth_tolerance_m=RGB_PLACE_NODE_OCCLUSION_TOLERANCE_M,
                )
                if len(unoccluded) == 0 or not bool(unoccluded[0]):
                    continue
                best_point_xy = (
                    float(projected[0, 0]),
                    float(projected[0, 1]),
                )
                center_distance_sq = float(
                    (float(best_point_xy[0]) - image_center_x) ** 2
                    + (float(best_point_xy[1]) - image_center_y) ** 2
                )
                projected_nodes.append(
                    _ProjectedPlaceNodeOverlay(
                        obs_id=str(obs_id),
                        node_id=str(node.id),
                        node_xy=node_xy,
                        end_xy=best_point_xy,
                        center_distance_sq=center_distance_sq,
                        distance_m=float(np.linalg.norm(np.asarray(node_xy, dtype=np.float64) - robot_xy)),
                    )
                )
            if projected_nodes == []:
                continue

            node_font = _frontier_badge_font(RGB_FRONTIER_BADGE_FONT_SIZE_MAX_PX)
            for projected_node in sorted(
                projected_nodes,
                key=lambda item: (-float(item.distance_m), _node_id_sort_key(str(item.node_id))),
            ):
                _draw_place_node_circle(
                    draw=draw,
                    center_xy=scale_xy(projected_node.end_xy, scale_x=scale_x, scale_y=scale_y),
                    label=projected_node.node_id,
                    font=node_font,
                    image_size=(render_width, render_height),
                    image=image,
                )

            node_ids = sorted(
                {str(projected_node.node_id) for projected_node in projected_nodes},
                key=_node_id_sort_key,
            )
            overlay_record = cache.store_image(
                np.asarray(image.convert("RGB"), dtype=np.uint8),
                kind="place_node_overlay",
                metadata={
                    "obs_id": str(obs_id),
                    "floor_id": floor_id_str,
                    "node_ids": list(node_ids),
                    "edge_ids": [],
                },
            )
            views.append(
                PlaceNodeOverlayView(
                    obs_id=str(obs_id),
                    overlay_id=str(overlay_record.id),
                    node_ids=list(node_ids),
                    edge_ids=[],
                )
            )
        return views

    def update_frontier_scores(self, ranked_frontier_ids: list[str]) -> list[dict[str, object]]:
        return self.frontier_buffer.apply_ranking(ranked_frontier_ids)

    def list_frontier_ids(self) -> list[str]:
        return [str(entry.id) for entry in self.frontier_buffer.selectable_entries()]

    def has_frontier(self, frontier_id: str) -> bool:
        frontier_id_str = str(frontier_id)
        for entry in self.frontier_buffer.selectable_entries():
            if str(entry.id) == frontier_id_str:
                return True
        return False

    def blacklist_frontier(self, frontier_id: str, reason: str) -> dict[str, object]:
        return self.frontier_buffer.blacklist(frontier_id=frontier_id, reason=reason)

    def resolve_frontier_goal_pose(self, frontier_id: str) -> dict[str, float]:
        frontier_id_str = str(frontier_id)
        for entry in self.frontier_buffer.selectable_entries():
            if entry.id != frontier_id_str:
                continue
            if entry.goal_xy is None or entry.goal_yaw_deg is None:
                raise ValueError(f"Frontier {frontier_id_str} has no resolved goal pose")
            return {
                "x": float(entry.goal_xy[0]),
                "y": float(entry.goal_xy[1]),
                "yaw": float(entry.goal_yaw_deg),
            }
        raise ValueError(f"Unknown frontier_id for resolve_frontier_goal_pose: {frontier_id_str}")

    def _get_frontier_entry(self, frontier_id: str):
        frontier_id_str = str(frontier_id)
        for entry in self.frontier_buffer.entries:
            if str(entry.id) == frontier_id_str:
                return entry
        return None

    def _entry_visual_evidence(
        self,
        frontier_id: str,
    ) -> _FrontierVisualEvidence | None:
        entry = self._get_frontier_entry(frontier_id)
        if entry is None:
            return None
        if (
            entry.evidence_obs_id is None
            or entry.evidence_frontier_xy is None
            or entry.evidence_end_xy is None
            or entry.evidence_center_distance_sq is None
        ):
            return None
        return _FrontierVisualEvidence(
            frontier_id=str(frontier_id),
            obs_id=str(entry.evidence_obs_id),
            frontier_xy=(
                float(entry.evidence_frontier_xy[0]),
                float(entry.evidence_frontier_xy[1]),
            ),
            end_xy=(
                float(entry.evidence_end_xy[0]),
                float(entry.evidence_end_xy[1]),
            ),
            center_distance_sq=float(entry.evidence_center_distance_sq),
            source="history",
        )

    def _update_frontier_visual_evidence(
        self,
        evidence: _FrontierVisualEvidence,
    ) -> None:
        entry = self._get_frontier_entry(evidence.frontier_id)
        if entry is None:
            raise ValueError(f"Unknown frontier_id for visual evidence update: {evidence.frontier_id}")
        entry.evidence_obs_id = str(evidence.obs_id)
        entry.evidence_frontier_xy = (
            float(evidence.frontier_xy[0]),
            float(evidence.frontier_xy[1]),
        )
        entry.evidence_end_xy = (
            float(evidence.end_xy[0]),
            float(evidence.end_xy[1]),
        )
        entry.evidence_center_distance_sq = float(evidence.center_distance_sq)

    def query_global_scored_frontiers(self, robot_xy: np.ndarray) -> list[dict[str, object]]:
        robot_xy_array = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        candidates: list[dict[str, object]] = []
        for entry in self.frontier_buffer.selectable_entries():
            if int(entry.score_count) <= 0:
                continue
            frontier_xy = np.asarray(entry.xy, dtype=np.float64).reshape(2)
            distance = float(np.linalg.norm(frontier_xy - robot_xy_array))
            if distance < FALLBACK_FRONTIER_MIN_DISTANCE_TO_AGENT_M:
                continue
            candidates.append(
                {
                    "frontier_id": str(entry.id),
                    "score_mean": float(entry.score_mean),
                    "distance": distance,
                }
            )
        candidates.sort(
            key=lambda item: (-float(item["score_mean"]), float(item["distance"]), str(item["frontier_id"]))
        )
        return candidates

    def query_nearest_frontiers(self, robot_xy: np.ndarray) -> list[dict[str, object]]:
        robot_xy_array = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        candidates: list[dict[str, object]] = []
        for entry in self.frontier_buffer.selectable_entries():
            frontier_xy = np.asarray(entry.xy, dtype=np.float64).reshape(2)
            distance = float(np.linalg.norm(frontier_xy - robot_xy_array))
            if distance < FALLBACK_FRONTIER_MIN_DISTANCE_TO_AGENT_M:
                continue
            candidates.append(
                {
                    "frontier_id": str(entry.id),
                    "score_mean": float(entry.score_mean),
                    "distance": distance,
                }
            )
        candidates.sort(
            key=lambda item: (float(item["distance"]), str(item["frontier_id"]))
        )
        return candidates

    def update_node(self, graph: Graph, cache: RuntimeCache, node_id: str) -> dict[str, object]:
        node = graph.get_node(node_id)
        integrated_obs_ids: list[str] = []
        skipped_obs_ids: list[str] = []
        for obs_id in node.obs_ids:
            if obs_id in self.integrated_obs_ids:
                skipped_obs_ids.append(obs_id)
                continue
            self.observe_observation(cache=cache, obs_id=obs_id)
            integrated_obs_ids.append(obs_id)

        robot_xy = np.asarray(node.position[:2], dtype=np.float64)
        frontier_entries = self.refresh_frontiers(robot_xy)
        return {
            "node_id": node_id,
            "integrated_obs_ids": integrated_obs_ids,
            "skipped_obs_ids": skipped_obs_ids,
            "frontier_count": len(frontier_entries),
            "frontiers": frontier_entries,
        }

    def explore_pose(
        self,
        graph: Graph,
        cache: RuntimeCache,
        pose: dict[str, float],
        obs_ids: list[str],
    ) -> dict[str, object]:
        robot_xy = np.asarray([float(pose["x"]), float(pose["y"])], dtype=np.float64)
        self.refresh_frontiers(robot_xy)
        candidates = self.map.query_reachable_frontiers(
            robot_xy=robot_xy,
            frontier_entries=self.frontier_buffer.selectable_entries(),
        )
        self.frontier_buffer.update_navigation_goals(candidates)
        views = self._render_views(cache=cache, obs_ids=list(obs_ids), candidates=candidates)
        bev_overlay_record = cache.store_image(
            self._render_bev_overlay(robot_xy=robot_xy, candidates=candidates),
            kind="bev_overlay",
        )
        bev_graph_frontier_overlay_record = cache.store_image(
            self._render_bev_graph_frontier_overlay(
                graph=graph,
                robot_xy=robot_xy,
                candidates=candidates,
            ),
            kind="bev_graph_frontier_overlay",
        )
        return {
            "robot_xy": [float(robot_xy[0]), float(robot_xy[1])],
            "frontier_count": len(candidates),
            "frontiers": [candidate.to_dict() for candidate in candidates],
            "annotated_views": [view.to_dict() for view in views],
            "bev_overlay_id": bev_overlay_record.id,
            "bev_graph_frontier_overlay_id": bev_graph_frontier_overlay_record.id,
        }

    def render_global_bev_background(self) -> np.ndarray:
        return self._render_bev_background()

    def render_bev_graph_background_raw_obstacles(self) -> np.ndarray:
        return self._render_bev_background()

    def _render_bev_background(self) -> np.ndarray:
        base = np.full(
            (self.map.size, self.map.size, 3),
            BEV_UNKNOWN_BACKGROUND_COLOR,
            dtype=np.uint8,
        )
        base[self.map.explored_area & self.map.navigable_map] = np.asarray(BEV_EXPLORED_FREE_COLOR, dtype=np.uint8)
        base[self.map.dilated_obstacles] = np.asarray(BEV_DILATED_OBSTACLE_COLOR, dtype=np.uint8)
        base[self.map.obstacle_map] = np.asarray(BEV_OBSTACLE_COLOR, dtype=np.uint8)
        return base

    def _robot_xy_from_observation(self, observation) -> np.ndarray:
        if observation.T_odom_base is not None:
            T_odom_base = np.asarray(observation.T_odom_base, dtype=np.float64)
            return T_odom_base[:2, 3].copy()
        return np.asarray([observation.pose.x, observation.pose.y], dtype=np.float64)

    def _render_views(
        self,
        cache: RuntimeCache,
        obs_ids: list[str],
        candidates: list[FrontierCandidate],
        projection_base_z_m: float,
        frontier_label_map: dict[str, str] | None = None,
        base_image_ids_by_obs_id: dict[str, str] | None = None,
        update_visual_evidence: bool = True,
        image_max_size: tuple[int, int] | None = None,
    ) -> list[ExploreView]:
        projected_frontiers: list[_ProjectedFrontierOverlay] = []
        for obs_id in obs_ids:
            observation = cache.get_observation(obs_id).observation
            if observation.rgb is None:
                raise ValueError("explore requires observation.rgb")
            if observation.depth is None:
                raise ValueError("explore requires observation.depth")
            if observation.intrinsics is None:
                raise ValueError("explore requires observation.intrinsics")
            if observation.T_cam_odom is None:
                raise ValueError("explore requires observation.T_cam_odom")

            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            depth = np.asarray(observation.depth)
            intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
            T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
            image_height, image_width = rgb.shape[:2]
            image_center_x = (float(image_width) - 1.0) / 2.0
            image_center_y = (float(image_height) - 1.0) / 2.0

            for candidate in candidates:
                dense_path_xy = _densify_path_xy(candidate.path_xy)
                selected_projected_path: np.ndarray | None = None
                selected_visible_run: np.ndarray | None = None
                selected_anchor_index: int | None = None
                for projection_height_m in RGB_FRONTIER_PROJECTION_HEIGHTS_M:
                    world_z = float(projection_base_z_m) + float(projection_height_m)
                    path_points = np.asarray(
                        [[float(x), float(y), world_z] for x, y in dense_path_xy],
                        dtype=np.float32,
                    )
                    projected_path, projected_depths, valid_path = _project_points(
                        points_odom=path_points,
                        T_cam_odom=T_cam_odom,
                        intrinsics=intrinsics,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    unoccluded_path = _compute_unoccluded_mask(
                        projected_points=projected_path,
                        projected_depths=projected_depths,
                        valid_projection=valid_path,
                        depth_image=depth,
                    )
                    visible_indices = np.flatnonzero(unoccluded_path)
                    if len(visible_indices) == 0:
                        continue

                    anchor_index = int(visible_indices[-1])
                    remaining_ratio = _remaining_path_ratio_from_index(
                        path_xy=dense_path_xy,
                        point_index=anchor_index,
                    )
                    if remaining_ratio >= RGB_OVERLAY_MAX_REMAINING_PATH_RATIO:
                        continue

                    visible_run = _last_visible_run(unoccluded_path)
                    if len(visible_run) == 0:
                        continue
                    selected_projected_path = np.asarray(projected_path, dtype=np.float32)
                    selected_visible_run = np.asarray(visible_run, dtype=np.int64)
                    selected_anchor_index = int(anchor_index)
                    break
                if selected_projected_path is None or selected_visible_run is None or selected_anchor_index is None:
                    continue

                badge_index = _path_index_before_distance(
                    path_xy=dense_path_xy,
                    visible_run=selected_visible_run,
                    anchor_index=selected_anchor_index,
                    distance_m=RGB_FRONTIER_BADGE_BACKOFF_M,
                )
                end_xy = (
                    float(selected_projected_path[badge_index][0]),
                    float(selected_projected_path[badge_index][1]),
                )
                center_distance_sq = float(
                    (float(end_xy[0]) - image_center_x) ** 2 + (float(end_xy[1]) - image_center_y) ** 2
                )
                projected_frontiers.append(
                    _ProjectedFrontierOverlay(
                        obs_id=str(obs_id),
                        frontier_id=str(candidate.frontier_id),
                        frontier_xy=(float(candidate.xy[0]), float(candidate.xy[1])),
                        projected_path=selected_projected_path,
                        visible_run=selected_visible_run,
                        end_xy=(float(end_xy[0]), float(end_xy[1])),
                        center_distance_sq=center_distance_sq,
                    )
                )
        selected_frontiers = _select_best_projected_frontiers(projected_frontiers)
        frontier_order = {str(candidate.frontier_id): index for index, candidate in enumerate(candidates)}
        selected_frontiers_by_id = {
            str(selected.frontier_id): selected
            for selected in selected_frontiers
        }
        selected_evidences: list[_FrontierVisualEvidence] = []
        if bool(update_visual_evidence):
            for candidate in candidates:
                frontier_id = str(candidate.frontier_id)
                current_projection = selected_frontiers_by_id.get(frontier_id)
                stored_evidence = self._entry_visual_evidence(frontier_id=frontier_id)
                if current_projection is None and stored_evidence is None:
                    continue
                if current_projection is None:
                    selected_evidences.append(stored_evidence)
                    continue
                current_evidence = _FrontierVisualEvidence(
                    frontier_id=frontier_id,
                    obs_id=str(current_projection.obs_id),
                    frontier_xy=(
                        float(current_projection.frontier_xy[0]),
                        float(current_projection.frontier_xy[1]),
                    ),
                    end_xy=(
                        float(current_projection.end_xy[0]),
                        float(current_projection.end_xy[1]),
                    ),
                    center_distance_sq=float(current_projection.center_distance_sq),
                    source="current",
                )
                if (
                    stored_evidence is not None
                    and float(stored_evidence.center_distance_sq) <= float(current_evidence.center_distance_sq)
                ):
                    selected_evidences.append(stored_evidence)
                    continue
                self._update_frontier_visual_evidence(current_evidence)
                selected_evidences.append(current_evidence)
        else:
            for candidate in candidates:
                frontier_id = str(candidate.frontier_id)
                current_projection = selected_frontiers_by_id.get(frontier_id)
                if current_projection is None:
                    continue
                selected_evidences.append(
                    _FrontierVisualEvidence(
                        frontier_id=frontier_id,
                        obs_id=str(current_projection.obs_id),
                        frontier_xy=(
                            float(current_projection.frontier_xy[0]),
                            float(current_projection.frontier_xy[1]),
                        ),
                        end_xy=(
                            float(current_projection.end_xy[0]),
                            float(current_projection.end_xy[1]),
                        ),
                        center_distance_sq=float(current_projection.center_distance_sq),
                        source="current",
                    )
                )

        selected_by_obs: dict[str, list[_FrontierVisualEvidence]] = {}
        for selected in selected_evidences:
            selected_by_obs.setdefault(str(selected.obs_id), []).append(selected)

        views: list[ExploreView] = []
        for obs_id, obs_frontiers in selected_by_obs.items():
            observation = cache.get_observation(obs_id).observation
            if observation.rgb is None:
                raise ValueError("explore requires observation.rgb")
            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            image_height, image_width = rgb.shape[:2]
            resized = resize_rgb_to_fit(rgb, max_size=image_max_size)
            render_rgb = np.asarray(resized.image, dtype=np.uint8)
            base_image_id = str(
                (base_image_ids_by_obs_id or {}).get(str(obs_id), "")
            ).strip()
            if base_image_id != "":
                base_rgb = np.asarray(
                    cache.get_image(base_image_id).image,
                    dtype=np.uint8,
                )
                if base_rgb.shape[:2] == render_rgb.shape[:2]:
                    render_rgb = base_rgb
            render_height, render_width = render_rgb.shape[:2]
            scale_x = float(resized.metadata["scale_x"])
            scale_y = float(resized.metadata["scale_y"])
            image = Image.fromarray(render_rgb.copy()).convert("RGBA")
            draw = ImageDraw.Draw(image, "RGBA")
            marker_font, marker_radius_px = _rgb_frontier_circle_style(
                image_width=render_width,
                image_height=render_height,
            )
            visible_frontier_ids: list[str] = []
            frontier_sources: dict[str, str] = {}
            obs_frontiers = sorted(
                obs_frontiers,
                key=lambda item: int(frontier_order.get(str(item.frontier_id), 10**9)),
            )
            for projected in obs_frontiers:
                label_text = None
                if frontier_label_map is not None:
                    label_text = frontier_label_map.get(str(projected.frontier_id))
                label = _frontier_label_text(str(projected.frontier_id)) if label_text is None else str(label_text)
                scaled_end_xy = scale_xy(projected.end_xy, scale_x=scale_x, scale_y=scale_y)
                marker_xy = _clamp_frontier_circle_marker_center(
                    draw=draw,
                    center_xy=scaled_end_xy,
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
                    image=image,
                )
                visible_frontier_ids.append(str(projected.frontier_id))
                frontier_sources[str(projected.frontier_id)] = str(projected.source)

            overlay_record = cache.store_image(np.asarray(image.convert("RGB"), dtype=np.uint8), kind="overlay")
            views.append(
                ExploreView(
                    obs_id=str(obs_id),
                    overlay_id=overlay_record.id,
                    frontier_ids=visible_frontier_ids,
                    frontier_sources=frontier_sources,
                )
            )
        return views

    def _render_bev_overlay(
        self,
        robot_xy: np.ndarray,
        candidates: list[FrontierCandidate],
        frontier_label_map: dict[str, str] | None = None,
    ) -> np.ndarray:
        image = Image.fromarray(self.render_global_bev_background()).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        robot_px = self.map.xy_to_px(np.asarray(robot_xy, dtype=np.float64).reshape(1, 2))[0]
        draw.ellipse(
            (
                int(robot_px[0]) - 6,
                int(robot_px[1]) - 6,
                int(robot_px[0]) + 6,
                int(robot_px[1]) + 6,
            ),
            fill=(80, 160, 255),
            outline=(255, 255, 255),
            width=2,
        )

        for candidate in candidates:
            frontier_px = self.map.xy_to_px(
                np.asarray(candidate.xy, dtype=np.float64).reshape(1, 2)
            )[0]

            label_text = None
            if frontier_label_map is not None:
                label_text = frontier_label_map.get(str(candidate.frontier_id))
            _draw_frontier_badge(
                draw,
                (float(frontier_px[0]), float(frontier_px[1])),
                candidate.frontier_id,
                label_text=label_text,
                image=image,
                mode="bev",
            )
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def render_frontier_registered_overlay(
        self,
        robot_xy: np.ndarray,
        candidates: list[FrontierCandidate],
        frontier_label_map: dict[str, str],
    ) -> np.ndarray:
        return self._render_bev_overlay(
            robot_xy=robot_xy,
            candidates=list(candidates),
            frontier_label_map=dict(frontier_label_map),
        )

    def _render_bev_graph_frontier_overlay(
        self,
        graph: Graph,
        robot_xy: np.ndarray,
        candidates: list[FrontierCandidate],
        frontier_owner_node_id_by_frontier_id: dict[str, str] | None = None,
    ) -> np.ndarray:
        image = Image.fromarray(self.render_global_bev_background()).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        focus_points: list[tuple[float, float]] = []

        node_positions: dict[str, tuple[float, float]] = {}
        for node in graph.iter_nodes():
            node_xy = np.asarray([[float(node.position[0]), float(node.position[1])]], dtype=np.float64)
            node_px = self.map.xy_to_px(node_xy)[0]
            node_positions[str(node.id)] = (float(node_px[0]), float(node_px[1]))

        place_nodes = [
            node
            for node in graph.iter_nodes()
            if node.node_kind == "place"
        ]
        ordered_place_nodes = sorted(place_nodes, key=lambda item: _node_id_sort_key(str(item.id)))
        place_index_by_node_id: dict[str, int] = {
            str(node.id): index for index, node in enumerate(ordered_place_nodes)
        }

        for edge in graph.iter_edges(include_vertical=False):
            src_id = str(edge.src_id)
            dst_id = str(edge.dst_id)
            src_xy = node_positions[src_id]
            dst_xy = node_positions[dst_id]
            src_node = graph.get_node(src_id)
            dst_node = graph.get_node(dst_id)
            dashed = _is_dashed_node_kind(src_node.node_kind) or _is_dashed_node_kind(dst_node.node_kind)
            if dashed:
                _draw_dashed_line(draw, src_xy, dst_xy, fill=(70, 70, 70, 180), width=3)
            else:
                draw.line([src_xy, dst_xy], fill=(60, 90, 130, 160), width=3)
            focus_points.extend([src_xy, dst_xy])

        frontier_fill = (230, 111, 81, 255)
        frontier_outline = (255, 255, 255, 255)
        frontier_radius_px = 6
        place_style = place_node_marker_style(
            image_width=int(image.size[0]),
            image_height=int(image.size[1]),
            mode="bev",
        )

        for node in ordered_place_nodes:
            px, py = node_positions[str(node.id)]
            place_label = str(place_index_by_node_id[str(node.id)])
            draw_numbered_circle_marker(
                image,
                (px, py),
                place_label,
                style=place_style,
            )
            focus_points.append((px, py))

        robot_px = self.map.xy_to_px(np.asarray(robot_xy, dtype=np.float64).reshape(1, 2))[0]
        robot_center = (float(robot_px[0]), float(robot_px[1]))
        draw.ellipse(
            (
                robot_center[0] - 8,
                robot_center[1] - 8,
                robot_center[0] + 8,
                robot_center[1] + 8,
            ),
            fill=(80, 160, 255),
            outline=(255, 255, 255),
            width=2,
        )
        focus_points.append(robot_center)

        for candidate in candidates:
            owner_node_id = None
            if frontier_owner_node_id_by_frontier_id is not None:
                owner_node_id = frontier_owner_node_id_by_frontier_id.get(str(candidate.frontier_id))
            frontier_goal_xy = np.asarray([[float(candidate.goal_xy[0]), float(candidate.goal_xy[1])]], dtype=np.float64)
            frontier_goal_px = self.map.xy_to_px(frontier_goal_xy)[0]
            frontier_center = (float(frontier_goal_px[0]), float(frontier_goal_px[1]))
            if owner_node_id is not None and owner_node_id in node_positions:
                _draw_dashed_line(
                    draw,
                    node_positions[str(owner_node_id)],
                    frontier_center,
                    fill=(230, 111, 81, 200),
                    width=3,
                )
            draw.ellipse(
                (
                    frontier_center[0] - 6,
                    frontier_center[1] - 6,
                    frontier_center[0] + 6,
                    frontier_center[1] + 6,
                ),
                fill=(255, 80, 80),
                outline=(255, 255, 255),
                width=2,
            )
            focus_points.append(frontier_center)

        rendered = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return _crop_bev_overlay(
            rendered,
            explored_mask=self.map.explored_area,
            extra_points_px=focus_points,
        )

    def render_frontier_score_debug(
        self,
        robot_xy: np.ndarray,
        selected_frontier_id: str | None = None,
    ) -> np.ndarray:
        image = Image.fromarray(self.render_global_bev_background())
        draw = ImageDraw.Draw(image)

        robot_px = self.map.xy_to_px(np.asarray(robot_xy, dtype=np.float64).reshape(1, 2))[0]
        draw.ellipse(
            (
                int(robot_px[0]) - 8,
                int(robot_px[1]) - 8,
                int(robot_px[0]) + 8,
                int(robot_px[1]) + 8,
            ),
            fill=(80, 160, 255),
            outline=(255, 255, 255),
            width=2,
        )

        for entry in self.frontier_buffer.entries:
            frontier_xy = np.asarray([[float(entry.xy[0]), float(entry.xy[1])]], dtype=np.float64)
            frontier_px = self.map.xy_to_px(frontier_xy)[0]
            is_selected = selected_frontier_id is not None and str(entry.id) == str(selected_frontier_id)
            color = (255, 80, 80) if is_selected else (255, 210, 0)
            radius = 7 if is_selected else 5
            draw.ellipse(
                (
                    int(frontier_px[0]) - radius,
                    int(frontier_px[1]) - radius,
                    int(frontier_px[0]) + radius,
                    int(frontier_px[1]) + radius,
                ),
                fill=color,
                outline=(0, 0, 0),
                width=2,
            )
            label = f"{entry.id}:{float(entry.score_mean):.2f}"
            text_anchor = (float(frontier_px[0]) + 10.0, float(frontier_px[1]) - 10.0)
            bbox = draw.textbbox(text_anchor, label, font=FRONTIER_SCORE_FONT)
            draw.rectangle(
                (
                    bbox[0] - 2,
                    bbox[1] - 2,
                    bbox[2] + 2,
                    bbox[3] + 2,
                ),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=1,
            )
            draw.text(text_anchor, label, fill=(0, 0, 0), font=FRONTIER_SCORE_FONT)

        return np.asarray(image, dtype=np.uint8)
