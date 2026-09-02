from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from frontier_exploration.frontier_detection import (
    contour_to_frontiers,
    detect_frontier_waypoints,
    filter_out_small_unexplored,
    frontier_waypoints,
    interpolate_contour,
)
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from skimage.graph import route_through_array

from navclaw.env.interface import RawObservation


DEFAULT_VIRTUAL_MAX_RANGE_PIXEL_STRIDE = 4
DEFAULT_VIRTUAL_MAX_RANGE_BLOCKER_DILATE_PX = 3
LOCAL_FRONTIER_MIN_DILATED_OBSTACLE_CLEARANCE_PX = 2.0
LOCAL_FRONTIER_SPLIT_TURN_ANGLE_DEG = 70.0
LOCAL_FRONTIER_SPLIT_TANGENT_WINDOW_PX = 6
LOCAL_FRONTIER_SPLIT_MIN_SEGMENT_LENGTH_PX = 12.0


def _frontier_polyline_lengths(points_px: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)])


def _split_frontier_by_turn_angle(
    frontier_px: np.ndarray,
    *,
    turn_angle_deg: float = LOCAL_FRONTIER_SPLIT_TURN_ANGLE_DEG,
    tangent_window_px: int = LOCAL_FRONTIER_SPLIT_TANGENT_WINDOW_PX,
    min_segment_length_px: float = LOCAL_FRONTIER_SPLIT_MIN_SEGMENT_LENGTH_PX,
) -> list[np.ndarray]:
    points = np.asarray(frontier_px, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return [points]

    cumulative = _frontier_polyline_lengths(points)
    total_length = float(cumulative[-1]) if len(cumulative) else 0.0
    if total_length < float(min_segment_length_px) * 2.0:
        return [points]

    window = max(1, int(tangent_window_px))
    min_length = float(min_segment_length_px)
    split_candidates: list[tuple[float, int]] = []
    for index in range(window, len(points) - window):
        if cumulative[index] < min_length or total_length - cumulative[index] < min_length:
            continue
        before = points[index] - points[index - window]
        after = points[index + window] - points[index]
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        if before_norm < 1e-6 or after_norm < 1e-6:
            continue
        cosine = float(np.dot(before, after) / (before_norm * after_norm))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if angle >= float(turn_angle_deg):
            split_candidates.append((angle, index))

    if split_candidates == []:
        return [points]

    selected_indices: list[int] = []
    for _angle, index in sorted(split_candidates, reverse=True):
        split_length = float(cumulative[index])
        if any(abs(split_length - float(cumulative[existing])) < min_length for existing in selected_indices):
            continue
        selected_indices.append(index)
    selected_indices.sort()

    segments: list[np.ndarray] = []
    start = 0
    for split_index in selected_indices:
        segment = points[start : split_index + 1]
        if len(segment) >= 2 and float(cumulative[split_index] - cumulative[start]) >= min_length:
            segments.append(segment)
            start = split_index
    if start < len(points) - 1:
        segment = points[start:]
        if len(segment) >= 2 and float(cumulative[-1] - cumulative[start]) >= min_length:
            segments.append(segment)
    return segments or [points]


def _get_point_cloud(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    v_coords, u_coords = np.nonzero(mask)
    z = depth[mask]
    x = (u_coords.astype(np.float32) - float(cx)) * z / float(fx)
    y = (v_coords.astype(np.float32) - float(cy)) * z / float(fy)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo_points = np.concatenate([points.astype(np.float32), ones], axis=1)
    transformed = (transform.astype(np.float32) @ homo_points.T).T
    return transformed[:, :3]


def _as_depth_array(depth: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth)
    if np.issubdtype(depth_array.dtype, np.integer):
        depth_array = depth_array.astype(np.float32)
        if float(np.nanmax(depth_array)) > 100.0:
            depth_array = depth_array / 1000.0
        return depth_array
    return depth_array.astype(np.float32)


def _fill_small_depth_holes(depth: np.ndarray, area_thresh: int, fill_value: float) -> np.ndarray:
    if int(area_thresh) == -1:
        filled_depth = np.array(depth, dtype=np.float32, copy=True)
        filled_depth[filled_depth == 0.0] = float(fill_value)
        return filled_depth
    binary_img = np.where(depth == 0.0, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_holes = np.zeros_like(binary_img)
    for contour in contours:
        if cv2.contourArea(contour) < int(area_thresh):
            cv2.drawContours(filled_holes, [contour], 0, 1, -1)
    return np.where(filled_holes == 1, float(fill_value), depth).astype(np.float32)


@dataclass
class FrontierCandidate:
    frontier_id: str
    xy: tuple[float, float]
    goal_xy: tuple[float, float]
    path_xy: list[tuple[float, float]]
    path_length: float
    score_mean: float

    def goal_yaw_degrees(self) -> float:
        if len(self.path_xy) >= 2:
            prev_xy = self.path_xy[-2]
            curr_xy = self.path_xy[-1]
            dx = float(curr_xy[0]) - float(prev_xy[0])
            dy = float(curr_xy[1]) - float(prev_xy[1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return float(math.degrees(math.atan2(dy, dx)))
        dx = float(self.goal_xy[0]) - float(self.xy[0])
        dy = float(self.goal_xy[1]) - float(self.xy[1])
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return float(math.degrees(math.atan2(dy, dx)))
        return 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "frontier_id": self.frontier_id,
            "xy": [float(self.xy[0]), float(self.xy[1])],
            "goal_pose": {
                "x": float(self.goal_xy[0]),
                "y": float(self.goal_xy[1]),
                "yaw": self.goal_yaw_degrees(),
            },
            "path_length": float(self.path_length),
            "score_mean": float(self.score_mean),
        }


class GlobalBEVMap:
    def __init__(
        self,
        size: int = 2000,
        pixels_per_meter: int = 20,
        min_height: float = 0.2,
        max_height: float = 1.0,
        agent_radius: float = 0.18,
        area_thresh: float = 1.5,
        hole_area_thresh: int = 100000,
        min_depth: float = 0.1,
        max_depth: float = 4.9,
        astar_clearance_radius: float = 0.35,
        astar_clearance_weight: float = 8.0,
        depth_is_normalized: bool = False,
        virtual_max_range_blocker: bool = True,
        virtual_max_range_pixel_stride: int = DEFAULT_VIRTUAL_MAX_RANGE_PIXEL_STRIDE,
        virtual_max_range_blocker_dilate_px: int = DEFAULT_VIRTUAL_MAX_RANGE_BLOCKER_DILATE_PX,
    ) -> None:
        self.size = int(size)
        self.pixels_per_meter = int(pixels_per_meter)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.agent_radius = float(agent_radius)
        self.area_thresh_in_pixels = float(area_thresh) * float(
            self.pixels_per_meter ** 2
        )
        self.hole_area_thresh = int(hole_area_thresh)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.astar_clearance_radius = float(astar_clearance_radius)
        self.astar_clearance_weight = float(astar_clearance_weight)
        self.depth_is_normalized = bool(depth_is_normalized)
        self.virtual_max_range_blocker = bool(virtual_max_range_blocker)
        self.virtual_max_range_pixel_stride = max(1, int(virtual_max_range_pixel_stride))
        self.virtual_max_range_blocker_dilate_px = max(1, int(virtual_max_range_blocker_dilate_px))

        self.obstacle_map = np.zeros((self.size, self.size), dtype=bool)
        self.dilated_obstacles = np.zeros((self.size, self.size), dtype=bool)
        self.navigable_map = np.ones((self.size, self.size), dtype=bool)
        self.explored_area = np.zeros((self.size, self.size), dtype=bool)
        self.known_map = np.zeros((self.size, self.size), dtype=bool)
        self.free_map = np.zeros((self.size, self.size), dtype=bool)
        self.frontiers_px = np.zeros((0, 2), dtype=np.float64)
        self.frontier_clusters_px: list[np.ndarray] = []
        self.episode_origin: np.ndarray | None = None

        kernel_size = self.pixels_per_meter * self.agent_radius * 2.0
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self.navigable_kernel = np.ones(
            (max(1, kernel_size), max(1, kernel_size)),
            dtype=np.uint8,
        )

    def reset(self) -> None:
        self.obstacle_map.fill(False)
        self.dilated_obstacles.fill(False)
        self.navigable_map.fill(True)
        self.explored_area.fill(False)
        self.known_map.fill(False)
        self.free_map.fill(False)
        self.frontiers_px = np.zeros((0, 2), dtype=np.float64)
        self.frontier_clusters_px = []
        self.episode_origin = None

    def update_from_observation(self, observation: RawObservation) -> bool:
        if observation.depth is None:
            raise ValueError("update requires observation.depth")
        if observation.intrinsics is None:
            raise ValueError("update requires observation.intrinsics")
        if observation.T_cam_odom is None:
            raise ValueError("update requires observation.T_cam_odom")
        if observation.T_odom_base is None:
            raise ValueError("update requires observation.T_odom_base")

        depth = _as_depth_array(np.asarray(observation.depth))
        intrinsic = np.asarray(observation.intrinsics, dtype=np.float32)
        T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
        T_odom_base = np.asarray(observation.T_odom_base, dtype=np.float32)

        if self.episode_origin is None:
            self.episode_origin = T_odom_base[:3, 3].copy()

        metric_depth = self._metric_depth_from_observation_depth(depth)
        valid_depth_mask = (
            np.isfinite(metric_depth)
            & (metric_depth > self.min_depth)
            & (metric_depth < self.max_depth)
        )

        if bool(np.any(valid_depth_mask)):
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            point_cloud_optical = _get_point_cloud(
                metric_depth,
                valid_depth_mask,
                fx,
                fy,
                cx,
                cy,
            )
            T_odom_cam = np.linalg.inv(T_cam_odom)
            point_cloud_odom = _transform_points(T_odom_cam, point_cloud_optical)
            floor_height = float(T_odom_base[2, 3])
            min_obstacle_height = floor_height + float(self.min_height)
            max_obstacle_height = floor_height + float(self.max_height)
            obstacle_cloud = point_cloud_odom[
                (point_cloud_odom[:, 2] >= min_obstacle_height)
                & (point_cloud_odom[:, 2] <= max_obstacle_height)
            ]
            self._update_obstacle_map(obstacle_cloud[:, :2])

        self._refresh_navigable_map()
        self._update_explored_area(
            T_odom_base=T_odom_base,
            raw_depth=depth,
            metric_depth=metric_depth,
            depth_shape=depth.shape,
            intrinsic=intrinsic,
            T_cam_odom=T_cam_odom,
        )
        self._refresh_compatibility_maps()
        self._update_frontiers()
        return True

    def detect_global_frontiers(self, robot_xy: np.ndarray) -> np.ndarray:
        frontiers_px, _ = self.detect_frontier_clusters_px()
        if len(frontiers_px) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return self.px_to_xy(frontiers_px)

    def detect_frontier_clusters_px(self) -> tuple[np.ndarray, list[np.ndarray]]:
        self._update_frontiers()
        return np.array(self.frontiers_px, dtype=np.float64, copy=True), [
            np.array(cluster, dtype=np.float64, copy=True)
            for cluster in self.frontier_clusters_px
        ]

    def query_reachable_frontiers(
        self,
        robot_xy: np.ndarray,
        frontier_entries: list[object],
    ) -> list[FrontierCandidate]:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        cost_map = self._build_astar_cost_map()
        candidates: list[FrontierCandidate] = []
        for entry in frontier_entries:
            goal_xy = np.asarray(entry.xy, dtype=np.float64)
            path_xy = self.compute_astar_path(robot_xy, goal_xy, cost_map=cost_map)
            if path_xy is None or len(path_xy) < 2:
                continue
            final_goal_xy = np.asarray(path_xy[-1], dtype=np.float64).reshape(2)
            if self._xy_in_dilated_obstacle(final_goal_xy):
                continue
            deltas = np.diff(path_xy, axis=0)
            path_length = float(np.linalg.norm(deltas, axis=1).sum())
            candidates.append(
                FrontierCandidate(
                    frontier_id=entry.id,
                    xy=(float(goal_xy[0]), float(goal_xy[1])),
                    goal_xy=(float(final_goal_xy[0]), float(final_goal_xy[1])),
                    path_xy=[(float(item[0]), float(item[1])) for item in path_xy],
                    path_length=path_length,
                    score_mean=float(entry.score_mean),
                )
            )
        candidates.sort(key=lambda item: item.path_length)
        return candidates

    def compute_astar_path(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        cost_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        active_cost_map = self._build_astar_cost_map() if cost_map is None else cost_map
        start_px = self.xy_to_px(np.asarray(start_xy, dtype=np.float64).reshape(1, 2))[0]
        goal_px = self.xy_to_px(np.asarray(goal_xy, dtype=np.float64).reshape(1, 2))[0]

        start_rc = self._find_nearest_traversable_pixel(start_px, active_cost_map)
        if start_rc is None:
            return None

        goal_rc = self._find_nearest_traversable_pixel(goal_px, active_cost_map)
        if goal_rc is None:
            return None

        path_rc, _ = route_through_array(
            active_cost_map,
            start_rc,
            goal_rc,
            fully_connected=True,
        )
        path_rows = np.asarray([row for row, _ in path_rc], dtype=np.int32)
        path_cols = np.asarray([col for _, col in path_rc], dtype=np.int32)
        if np.any(active_cost_map[path_rows, path_cols] >= 1e9):
            return None
        path_px = np.asarray(
            [[float(col), float(row)] for row, col in path_rc],
            dtype=np.float64,
        )
        return self.px_to_xy(path_px)

    def _metric_depth_from_observation_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if not bool(np.any(np.isfinite(depth))):
            return depth
        if self.depth_is_normalized:
            filled_depth = _fill_small_depth_holes(
                depth,
                self.hole_area_thresh,
                fill_value=1.0,
            )
            return filled_depth * (self.max_depth - self.min_depth) + self.min_depth
        return _fill_small_depth_holes(
            depth,
            self.hole_area_thresh,
            fill_value=self.max_depth,
        )

    def metric_depth_from_observation_depth(self, depth: np.ndarray) -> np.ndarray:
        return self._metric_depth_from_observation_depth(depth)

    def _update_obstacle_map(self, xy_points: np.ndarray) -> None:
        xy_points = np.asarray(xy_points, dtype=np.float64).reshape(-1, 2)
        if len(xy_points) == 0:
            return
        pixel_points = self.xy_to_px(xy_points)
        in_bounds = (
            (pixel_points[:, 0] >= 0)
            & (pixel_points[:, 0] < self.size)
            & (pixel_points[:, 1] >= 0)
            & (pixel_points[:, 1] < self.size)
        )
        pixel_points = pixel_points[in_bounds]
        if len(pixel_points) == 0:
            return
        self.obstacle_map[pixel_points[:, 1], pixel_points[:, 0]] = True

    def _refresh_navigable_map(self) -> None:
        self.dilated_obstacles = cv2.dilate(
            self.obstacle_map.astype(np.uint8),
            self.navigable_kernel,
            iterations=1,
        ).astype(bool)
        self.navigable_map = ~self.dilated_obstacles

    def _update_explored_area(
        self,
        T_odom_base: np.ndarray,
        raw_depth: np.ndarray,
        metric_depth: np.ndarray,
        depth_shape: tuple[int, ...],
        intrinsic: np.ndarray,
        T_cam_odom: np.ndarray,
    ) -> None:
        agent_xy_location = np.asarray(T_odom_base[:2, 3], dtype=np.float64).reshape(
            1,
            2,
        )
        agent_pixel_location = self.xy_to_px(agent_xy_location)[0]
        if not self._pixel_in_bounds(agent_pixel_location):
            return
        topdown_fov = self._topdown_fov_from_intrinsics(
            depth_shape=depth_shape,
            intrinsic=intrinsic,
        )
        virtual_blocker = self._virtual_max_range_blocker(
            raw_depth=raw_depth,
            metric_depth=metric_depth,
            intrinsic=intrinsic,
            T_cam_odom=T_cam_odom,
        )
        reveal_navigable_map = self.navigable_map
        if virtual_blocker is not None:
            reveal_navigable_map = self.navigable_map & ~virtual_blocker
        new_explored_area = reveal_fog_of_war(
            top_down_map=reveal_navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self.obstacle_map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=self._fog_of_war_angle_from_base_pose(T_odom_base),
            fov=np.rad2deg(topdown_fov),
            max_line_len=self.max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(
            new_explored_area,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        if virtual_blocker is not None:
            new_explored_area[virtual_blocker] = 0
        self.explored_area[new_explored_area > 0] = True
        self.explored_area[self.navigable_map == 0] = False
        self._keep_explored_component_connected_to_agent(agent_pixel_location)

    def _virtual_max_range_blocker(
        self,
        raw_depth: np.ndarray,
        metric_depth: np.ndarray,
        intrinsic: np.ndarray,
        T_cam_odom: np.ndarray,
    ) -> np.ndarray | None:
        if not self.virtual_max_range_blocker:
            return None
        raw_depth = np.asarray(raw_depth)
        metric_depth = np.asarray(metric_depth, dtype=np.float32)
        if raw_depth.shape != metric_depth.shape:
            raise ValueError("raw_depth and metric_depth shape mismatch")
        if self.depth_is_normalized:
            raw_valid = np.isfinite(raw_depth) & (raw_depth > 0.0)
        else:
            raw_valid = np.isfinite(raw_depth) & (raw_depth > self.min_depth)
        max_range_mask = raw_valid & np.isfinite(metric_depth) & (metric_depth >= self.max_depth)
        if not bool(np.any(max_range_mask)):
            return None

        sampled_mask = np.zeros_like(max_range_mask, dtype=bool)
        stride = int(self.virtual_max_range_pixel_stride)
        sampled_mask[::stride, ::stride] = max_range_mask[::stride, ::stride]
        if not bool(np.any(sampled_mask)):
            return None

        virtual_depth = np.full_like(metric_depth, float(self.max_depth), dtype=np.float32)
        point_cloud_optical = _get_point_cloud(
            virtual_depth,
            sampled_mask,
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
        )
        T_odom_cam = np.linalg.inv(T_cam_odom)
        point_cloud_odom = _transform_points(T_odom_cam, point_cloud_optical)
        pixel_points = self.xy_to_px(point_cloud_odom[:, :2])
        in_bounds = (
            (pixel_points[:, 0] >= 0)
            & (pixel_points[:, 0] < self.size)
            & (pixel_points[:, 1] >= 0)
            & (pixel_points[:, 1] < self.size)
        )
        pixel_points = pixel_points[in_bounds]
        if len(pixel_points) == 0:
            return None

        blocker = np.zeros_like(self.obstacle_map, dtype=np.uint8)
        blocker[pixel_points[:, 1], pixel_points[:, 0]] = 1
        blocker = cv2.dilate(
            blocker,
            np.ones(
                (
                    int(self.virtual_max_range_blocker_dilate_px),
                    int(self.virtual_max_range_blocker_dilate_px),
                ),
                dtype=np.uint8,
            ),
            iterations=1,
        )
        return blocker.astype(bool)

    def _keep_explored_component_connected_to_agent(self, agent_pixel_location: np.ndarray) -> None:
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) <= 1:
            return
        min_dist = math.inf
        best_idx = 0
        point = tuple([int(i) for i in agent_pixel_location])
        for idx, contour in enumerate(contours):
            dist = cv2.pointPolygonTest(contour, point, True)
            if dist >= 0:
                best_idx = idx
                break
            if abs(float(dist)) < min_dist:
                min_dist = abs(float(dist))
                best_idx = idx
        new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
        cv2.drawContours(new_area, contours, best_idx, 1, -1)
        self.explored_area = new_area.astype(bool)

    def _refresh_compatibility_maps(self) -> None:
        self.known_map = self.explored_area | self.obstacle_map
        self.free_map = self.explored_area & self.navigable_map

    def _update_frontiers(self) -> None:
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        waypoints, frontiers, _segments = detect_frontier_waypoints(
            self.navigable_map.astype(np.uint8),
            explored_area,
            int(self.area_thresh_in_pixels),
            return_frontiers=True,
        )
        if len(waypoints) == 0:
            self.frontiers_px = np.zeros((0, 2), dtype=np.float64)
            self.frontier_clusters_px = []
            return
        self.frontiers_px = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
        clusters: list[np.ndarray] = []
        for frontier in frontiers:
            frontier_px = np.asarray(frontier, dtype=np.float64).reshape(-1, 2)
            clusters.append(frontier_px)
        self.frontier_clusters_px = clusters

    def _build_astar_cost_map(self) -> np.ndarray:
        traversable = self._traversable_map()
        cost_map = np.full(traversable.shape, 1e10, dtype=np.float64)
        cost_map[traversable] = 1.0
        if self.astar_clearance_radius > 0.0 and self.astar_clearance_weight > 0.0:
            clearance_px = cv2.distanceTransform(
                traversable.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            clearance_m = clearance_px.astype(np.float64) / float(self.pixels_per_meter)
            shortfall = np.clip(
                (float(self.astar_clearance_radius) - clearance_m)
                / float(self.astar_clearance_radius),
                0.0,
                1.0,
            )
            clearance_penalty = 1.0 + float(self.astar_clearance_weight) * np.square(shortfall)
            traversable_mask = cost_map < 1e9
            cost_map[traversable_mask] *= clearance_penalty[traversable_mask]
        return cost_map

    def _traversable_map(self) -> np.ndarray:
        return self.explored_area & self.navigable_map

    def is_traversable_xy(
        self,
        target_xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> bool:
        target_px = self.xy_to_px(
            np.asarray(target_xy, dtype=np.float64).reshape(1, 2)
        )[0]
        if not self._pixel_in_bounds(target_px):
            return False
        traversable = self._traversable_map()
        return bool(traversable[int(target_px[1]), int(target_px[0])])

    def _find_nearest_traversable_pixel(
        self,
        target_px: np.ndarray,
        cost_map: np.ndarray,
        search_radius_px: int | None = None,
    ) -> tuple[int, int] | None:
        if search_radius_px is None:
            search_radius_px = max(2, int(np.ceil(0.25 * self.pixels_per_meter)))

        px_x = int(target_px[0])
        px_y = int(target_px[1])
        if not (0 <= px_x < self.size and 0 <= px_y < self.size):
            return None

        target_rc = (px_y, px_x)
        if cost_map[target_rc] < 1e9:
            return target_rc

        best_goal = None
        best_distance = None
        y_min = max(0, px_y - search_radius_px)
        y_max = min(self.size, px_y + search_radius_px + 1)
        x_min = max(0, px_x - search_radius_px)
        x_max = min(self.size, px_x + search_radius_px + 1)
        for candidate_y in range(y_min, y_max):
            for candidate_x in range(x_min, x_max):
                if cost_map[candidate_y, candidate_x] >= 1e9:
                    continue
                distance = float(np.hypot(candidate_x - px_x, candidate_y - px_y))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_goal = (candidate_y, candidate_x)
        return best_goal

    def find_nearest_navigable_xy(
        self,
        target_xy: tuple[float, float] | list[float] | np.ndarray,
        max_radius_m: float = 0.7,
    ) -> tuple[float, float] | None:
        target_xy_arr = np.asarray(target_xy, dtype=np.float64).reshape(1, 2)
        target_px = self.xy_to_px(target_xy_arr)[0]
        px_x = int(target_px[0])
        px_y = int(target_px[1])
        if not (0 <= px_x < self.size and 0 <= px_y < self.size):
            return None
        traversable = self._traversable_map()
        if traversable[px_y, px_x]:
            result_px = np.array([[px_x, px_y]], dtype=np.float64)
            result_xy = self.px_to_xy(result_px)[0]
            return (float(result_xy[0]), float(result_xy[1]))
        search_radius_px = max(1, int(max_radius_m * self.pixels_per_meter))
        best_goal = None
        best_distance = None
        y_min = max(0, px_y - search_radius_px)
        y_max = min(self.size, px_y + search_radius_px + 1)
        x_min = max(0, px_x - search_radius_px)
        x_max = min(self.size, px_x + search_radius_px + 1)
        radius_sq = float(search_radius_px * search_radius_px)
        for candidate_y in range(y_min, y_max):
            for candidate_x in range(x_min, x_max):
                dx = candidate_x - px_x
                dy = candidate_y - px_y
                dist_sq = float(dx * dx + dy * dy)
                if dist_sq > radius_sq:
                    continue
                if not traversable[candidate_y, candidate_x]:
                    continue
                if best_distance is None or dist_sq < best_distance:
                    best_distance = dist_sq
                    best_goal = (candidate_x, candidate_y)
        if best_goal is None:
            return None
        result_px = np.array([[best_goal[0], best_goal[1]]], dtype=np.float64)
        result_xy = self.px_to_xy(result_px)[0]
        return (float(result_xy[0]), float(result_xy[1]))

    def obstacle_free_segment(
        self,
        start_xy: tuple[float, float] | list[float] | np.ndarray,
        end_xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> bool:
        hit_count = self.obstacle_segment_hit_count(start_xy=start_xy, end_xy=end_xy)
        return hit_count is not None and int(hit_count) == 0

    def obstacle_segment_hit_count(
        self,
        start_xy: tuple[float, float] | list[float] | np.ndarray,
        end_xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> int | None:
        line_px = self._line_pixels_between_xy(start_xy=start_xy, end_xy=end_xy)
        if line_px is None:
            return None
        return int(np.count_nonzero(self.obstacle_map[line_px[:, 1], line_px[:, 0]]))

    def segment_known_free_stats(
        self,
        start_xy: tuple[float, float] | list[float] | np.ndarray,
        end_xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> dict[str, object] | None:
        line_px = self._line_pixels_between_xy(start_xy=start_xy, end_xy=end_xy)
        if line_px is None:
            return None
        sample_count = int(len(line_px))
        obstacle_mask = self.obstacle_map[line_px[:, 1], line_px[:, 0]]
        known_free_mask = self.free_map[line_px[:, 1], line_px[:, 0]] & ~obstacle_mask
        obstacle_hit_count = int(np.count_nonzero(obstacle_mask))
        known_free_count = int(np.count_nonzero(known_free_mask))
        return {
            "sample_count": int(sample_count),
            "obstacle_hit_count": obstacle_hit_count,
            "known_free_count": known_free_count,
            "known_free_ratio": float(known_free_count) / float(sample_count),
        }

    def _line_pixels_between_xy(
        self,
        start_xy: tuple[float, float] | list[float] | np.ndarray,
        end_xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> np.ndarray | None:
        points_xy = np.asarray([start_xy, end_xy], dtype=np.float64).reshape(2, 2)
        points_px = self.xy_to_px(points_xy)
        start_px = points_px[0]
        end_px = points_px[1]
        if not self._pixel_in_bounds(start_px) or not self._pixel_in_bounds(end_px):
            return None
        dx = int(end_px[0]) - int(start_px[0])
        dy = int(end_px[1]) - int(start_px[1])
        step_count = max(abs(dx), abs(dy)) + 1
        xs = np.rint(
            np.linspace(float(start_px[0]), float(end_px[0]), step_count)
        ).astype(np.int32)
        ys = np.rint(
            np.linspace(float(start_px[1]), float(end_px[1]), step_count)
        ).astype(np.int32)
        line_px = np.stack([xs, ys], axis=1)
        in_bounds = (
            (line_px[:, 0] >= 0)
            & (line_px[:, 0] < self.size)
            & (line_px[:, 1] >= 0)
            & (line_px[:, 1] < self.size)
        )
        if not bool(np.all(in_bounds)):
            return None
        return line_px

    def _xy_in_dilated_obstacle(
        self,
        xy: tuple[float, float] | list[float] | np.ndarray,
    ) -> bool:
        xy_array = np.asarray(xy, dtype=np.float64).reshape(1, 2)
        px = self.xy_to_px(xy_array)[0]
        if not self._pixel_in_bounds(px):
            return True
        return bool(self.dilated_obstacles[int(px[1]), int(px[0])])

    def xy_to_px(self, points: np.ndarray) -> np.ndarray:
        return self._xy_to_grid_index(points)

    def px_to_xy(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        origin = (
            np.array([0.0, 0.0], dtype=np.float64)
            if self.episode_origin is None
            else self.episode_origin[:2]
        )
        half_cells = float(self.size) / 2.0
        rel_points = np.zeros_like(px, dtype=np.float64)
        rel_points[:, 0] = (px[:, 0] - half_cells) / float(self.pixels_per_meter)
        rel_points[:, 1] = -(px[:, 1] - half_cells) / float(self.pixels_per_meter)
        return rel_points + origin

    def _xy_to_grid_float(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        origin = (
            np.array([0.0, 0.0], dtype=np.float64)
            if self.episode_origin is None
            else self.episode_origin[:2]
        )
        rel_points = points - origin
        half_cells = float(self.size) / 2.0
        grid = np.zeros_like(rel_points, dtype=np.float64)
        grid[:, 0] = half_cells + rel_points[:, 0] * float(self.pixels_per_meter)
        grid[:, 1] = half_cells - rel_points[:, 1] * float(self.pixels_per_meter)
        return grid

    def _xy_to_grid_index(self, points: np.ndarray) -> np.ndarray:
        return np.rint(self._xy_to_grid_float(points)).astype(np.int32)

    def _pixel_in_bounds(self, px: np.ndarray) -> bool:
        px = np.asarray(px, dtype=np.int32).reshape(2)
        return bool(0 <= int(px[0]) < self.size and 0 <= int(px[1]) < self.size)

    def _topdown_fov_from_intrinsics(self, depth_shape: tuple[int, ...], intrinsic: np.ndarray) -> float:
        width = int(depth_shape[1])
        fx = float(intrinsic[0, 0])
        return float(2.0 * math.atan((float(width) / 2.0) / fx))

    def _yaw_from_transform(self, transform: np.ndarray) -> float:
        return float(math.atan2(float(transform[1, 0]), float(transform[0, 0])))

    def _fog_of_war_angle_from_base_pose(self, transform: np.ndarray) -> float:
        return self._yaw_from_transform(transform) + math.pi / 2.0


class Map1Map2BEVMap(GlobalBEVMap):
    """BEV map using raw-free map1 and final map2 free space."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.raw_free_map = np.zeros((self.size, self.size), dtype=bool)
        self.map1_free_map = np.zeros((self.size, self.size), dtype=bool)
        self.map2_free_map = np.zeros((self.size, self.size), dtype=bool)
        self._dilated_obstacle_clearance_px = np.zeros((self.size, self.size), dtype=np.float32)
        self._latest_agent_xy: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.raw_free_map = np.zeros((self.size, self.size), dtype=bool)
        self.map1_free_map = np.zeros((self.size, self.size), dtype=bool)
        self.map2_free_map = np.zeros((self.size, self.size), dtype=bool)
        self._dilated_obstacle_clearance_px = np.zeros((self.size, self.size), dtype=np.float32)
        self._latest_agent_xy = None

    def merge_raw_layers_from_local_map(self, local_map: "Map1Map2BEVMap") -> dict[str, object]:
        if self.episode_origin is None and local_map.episode_origin is not None:
            self.episode_origin = np.asarray(local_map.episode_origin, dtype=np.float32).copy()
        if local_map._latest_agent_xy is not None:
            self._latest_agent_xy = np.asarray(
                local_map._latest_agent_xy,
                dtype=np.float64,
            ).copy()

        projected_obstacles = self._project_mask_from_map(
            source_map=local_map,
            source_mask=np.asarray(local_map.obstacle_map, dtype=bool),
        )
        projected_raw_free = self._project_mask_from_map(
            source_map=local_map,
            source_mask=np.asarray(local_map.raw_free_map, dtype=bool),
        )

        self.obstacle_map |= projected_obstacles
        self.raw_free_map |= projected_raw_free
        self.raw_free_map[self.obstacle_map] = False
        self._refresh_navigable_map()
        self._refresh_compatibility_maps()
        self._update_frontiers()
        return {
            "projected_obstacle_count": int(np.count_nonzero(projected_obstacles)),
            "projected_raw_free_count": int(np.count_nonzero(projected_raw_free)),
            "obstacle_count": int(np.count_nonzero(self.obstacle_map)),
            "raw_free_count": int(np.count_nonzero(self.raw_free_map)),
            "map2_free_count": int(np.count_nonzero(self.map2_free_map)),
        }

    def _project_mask_from_map(
        self,
        *,
        source_map: "Map1Map2BEVMap",
        source_mask: np.ndarray,
    ) -> np.ndarray:
        target_mask = np.zeros((self.size, self.size), dtype=bool)
        source_rows_cols = np.argwhere(np.asarray(source_mask, dtype=bool))
        if len(source_rows_cols) == 0:
            return target_mask
        source_px = np.asarray(
            [[float(col), float(row)] for row, col in source_rows_cols],
            dtype=np.float64,
        )
        xy = source_map.px_to_xy(source_px)
        target_px = self.xy_to_px(xy)
        in_bounds = (
            (target_px[:, 0] >= 0)
            & (target_px[:, 0] < int(self.size))
            & (target_px[:, 1] >= 0)
            & (target_px[:, 1] < int(self.size))
        )
        target_px = target_px[in_bounds]
        if len(target_px) > 0:
            target_mask[target_px[:, 1], target_px[:, 0]] = True
        return target_mask

    def _update_explored_area(
        self,
        T_odom_base: np.ndarray,
        raw_depth: np.ndarray,
        metric_depth: np.ndarray,
        depth_shape: tuple[int, ...],
        intrinsic: np.ndarray,
        T_cam_odom: np.ndarray,
    ) -> None:
        agent_xy_location = np.asarray(T_odom_base[:2, 3], dtype=np.float64).reshape(1, 2)
        self._latest_agent_xy = agent_xy_location.reshape(2).copy()
        agent_pixel_location = self.xy_to_px(agent_xy_location)[0]
        if not self._pixel_in_bounds(agent_pixel_location):
            return

        topdown_fov = self._topdown_fov_from_intrinsics(
            depth_shape=depth_shape,
            intrinsic=intrinsic,
        )
        virtual_blocker = self._virtual_max_range_blocker(
            raw_depth=raw_depth,
            metric_depth=metric_depth,
            intrinsic=intrinsic,
            T_cam_odom=T_cam_odom,
        )
        raw_free_space = ~self.obstacle_map
        reveal_map = raw_free_space if virtual_blocker is None else raw_free_space & ~virtual_blocker
        new_free = reveal_fog_of_war(
            top_down_map=reveal_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self.obstacle_map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=self._fog_of_war_angle_from_base_pose(T_odom_base),
            fov=np.rad2deg(topdown_fov),
            max_line_len=self.max_depth * self.pixels_per_meter,
        )
        new_free = cv2.dilate(
            new_free.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        if virtual_blocker is not None:
            new_free[virtual_blocker] = False
        new_free[raw_free_space == 0] = False
        self.raw_free_map[new_free] = True
        self.raw_free_map[raw_free_space == 0] = False

    def _refresh_compatibility_maps(self) -> None:
        self.map1_free_map = cv2.dilate(
            self.raw_free_map.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        self.map1_free_map[self.obstacle_map] = False

        map2 = self.map1_free_map & ~self.dilated_obstacles
        self.explored_area = map2.copy()
        if self._latest_agent_xy is not None:
            agent_pixel_location = self.xy_to_px(
                np.asarray(self._latest_agent_xy, dtype=np.float64).reshape(1, 2)
            )[0]
            self._keep_explored_component_connected_to_agent(agent_pixel_location)
            self.map2_free_map = self.explored_area.copy()
        else:
            self.map2_free_map = self._largest_component(map2)
        self.explored_area = self.map2_free_map.copy()
        self.free_map = self.map2_free_map.copy()
        self.navigable_map = self.map2_free_map.copy()
        self.known_map = self.map2_free_map | self.dilated_obstacles | self.obstacle_map
        self._dilated_obstacle_clearance_px = cv2.distanceTransform(
            (~self.dilated_obstacles).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

    def _traversable_map(self) -> np.ndarray:
        return self.map2_free_map.copy()

    def _xy_in_map2_free(self, xy: np.ndarray) -> bool:
        px = self.xy_to_px(np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
        if not self._pixel_in_bounds(px):
            return False
        return bool(self.map2_free_map[int(px[1]), int(px[0])])

    def _dilated_obstacle_clearance_at_xy(self, xy: np.ndarray) -> float:
        px = self.xy_to_px(np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
        if not self._pixel_in_bounds(px):
            return 0.0
        return float(self._dilated_obstacle_clearance_px[int(px[1]), int(px[0])])

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask, dtype=np.uint8)
        if not bool(np.any(binary)):
            return np.zeros_like(binary, dtype=bool)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        if component_count <= 1:
            return binary.astype(bool)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_label


class LocalFrontierBEVMap(Map1Map2BEVMap):
    """Local frontier map using map1 for detection and map2 for planning."""

    def _update_frontiers(self) -> None:
        full_map = (~self.obstacle_map).astype(np.uint8)
        explored_mask = self.map1_free_map.astype(np.uint8)
        explored_mask = explored_mask.copy()
        explored_mask[full_map == 0] = 0
        filtered_explored_mask = filter_out_small_unexplored(
            full_map,
            explored_mask,
            int(self.area_thresh_in_pixels),
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
        frontier_point_mask[~self.map2_free_map] = 0

        raw_frontiers = []
        for contour in contours:
            raw_frontiers.extend(
                contour_to_frontiers(
                    interpolate_contour(contour),
                    frontier_point_mask,
                )
            )

        frontiers: list[np.ndarray] = []
        for frontier in raw_frontiers:
            frontiers.extend(_split_frontier_by_turn_angle(frontier))

        waypoint_input = [
            np.asarray(frontier, dtype=np.float64).reshape(-1, 1, 2)
            for frontier in frontiers
        ]
        waypoints, _segments = frontier_waypoints(waypoint_input, None, return_segments=True)
        if len(waypoints) == 0:
            self.frontiers_px = np.zeros((0, 2), dtype=np.float64)
            self.frontier_clusters_px = []
            return
        self.frontiers_px = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
        self.frontier_clusters_px = [
            np.asarray(frontier, dtype=np.float64).reshape(-1, 2)
            for frontier in frontiers
        ]

    def query_reachable_frontiers(
        self,
        robot_xy: np.ndarray,
        frontier_entries: list[object],
    ) -> list[FrontierCandidate]:
        filtered_entries = []
        for entry in frontier_entries:
            frontier_xy = np.asarray(entry.xy, dtype=np.float64)
            if not self._xy_in_map2_free(frontier_xy):
                continue
            if self._dilated_obstacle_clearance_at_xy(frontier_xy) < LOCAL_FRONTIER_MIN_DILATED_OBSTACLE_CLEARANCE_PX:
                continue
            filtered_entries.append(entry)
        return super().query_reachable_frontiers(robot_xy=robot_xy, frontier_entries=filtered_entries)
