from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from navclaw.mapping.exploration.manager import ExplorationManager
    from navclaw.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class VisualWaypoint:
    obs_id: str
    angle_deg: int
    point_2d: tuple[float, float]
    point_pixel: tuple[float, float]
    raw_world_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    goal_yaw: float
    path_xy: list[tuple[float, float]] = field(default_factory=list)
    waypoint_target: str = ""
    target: str = ""
    depth_m: float = 0.0
    raw_world_z: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": str(self.obs_id),
            "angle_deg": int(self.angle_deg),
            "point_2d": [float(self.point_2d[0]), float(self.point_2d[1])],
            "point_pixel": [float(self.point_pixel[0]), float(self.point_pixel[1])],
            "raw_world_xy": [float(self.raw_world_xy[0]), float(self.raw_world_xy[1])],
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "goal_yaw": float(self.goal_yaw),
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "waypoint_target": str(self.waypoint_target),
            "target": str(self.target),
            "depth_m": float(self.depth_m),
            "raw_world_z": float(self.raw_world_z),
        }


def ground_visual_waypoint(
    *,
    cache: "RuntimeCache",
    exploration: "ExplorationManager",
    obs_id: str,
    angle_deg: int,
    point_2d: tuple[float, float],
    waypoint_target: str,
    target: str,
    use_navigation_snap: bool = True,
) -> VisualWaypoint:
    observation = cache.get_observation(str(obs_id)).observation
    if observation.depth is None:
        raise ValueError("visual grounding requires observation.depth")
    if observation.intrinsics is None:
        raise ValueError("visual grounding requires observation.intrinsics")
    if observation.T_cam_odom is None:
        raise ValueError("visual grounding requires observation.T_cam_odom")

    depth = np.asarray(observation.depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"visual grounding depth must be 2D, got shape {depth.shape}")
    height, width = depth.shape
    x_norm, y_norm = float(point_2d[0]), float(point_2d[1])
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        raise ValueError(f"point_2d must be in [0,1], got {point_2d!r}")
    u = float(x_norm * float(width - 1))
    v = float(y_norm * float(height - 1))
    depth_m = _median_valid_depth_near_pixel(depth=depth, u=u, v=v)
    if depth_m is None:
        raise ValueError(f"visual grounding found no valid depth near pixel ({u:.1f}, {v:.1f})")

    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
        raise ValueError("visual grounding received invalid camera intrinsics")

    x_cam = (float(u) - cx) * float(depth_m) / fx
    y_cam = (float(v) - cy) * float(depth_m) / fy
    z_cam = float(depth_m)
    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    point_cam = np.asarray([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    point_odom = T_odom_cam @ point_cam
    raw_world_xy = np.asarray([float(point_odom[0]), float(point_odom[1])], dtype=np.float64)
    start_xy = np.asarray([float(observation.pose.x), float(observation.pose.y)], dtype=np.float64)

    if use_navigation_snap:
        snapped_xy = exploration.map.find_nearest_navigable_xy(raw_world_xy, max_radius_m=0.7)
        target_xy = raw_world_xy if snapped_xy is None else np.asarray(snapped_xy, dtype=np.float64).reshape(2)
        path_xy = exploration.map.compute_astar_path(start_xy=start_xy, goal_xy=target_xy)
        if path_xy is None or len(path_xy) == 0:
            path = [(float(target_xy[0]), float(target_xy[1]))]
        else:
            path = [
                (float(point[0]), float(point[1]))
                for point in np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
            ]
    else:
        path = [(float(raw_world_xy[0]), float(raw_world_xy[1]))]
    goal_xy = path[-1]
    goal_yaw = _goal_yaw_from_path_or_target(path=path, start_xy=start_xy, raw_world_xy=raw_world_xy)
    return VisualWaypoint(
        obs_id=str(obs_id),
        angle_deg=int(angle_deg),
        point_2d=(float(x_norm), float(y_norm)),
        point_pixel=(float(u), float(v)),
        raw_world_xy=(float(raw_world_xy[0]), float(raw_world_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        goal_yaw=float(goal_yaw),
        path_xy=path,
        waypoint_target=str(waypoint_target),
        target=str(target),
        depth_m=float(depth_m),
        raw_world_z=float(point_odom[2]),
    )


def ground_visual_waypoint_along_ray(
    *,
    cache: "RuntimeCache",
    exploration: "ExplorationManager",
    obs_id: str,
    angle_deg: int,
    point_2d: tuple[float, float],
    waypoint_target: str,
    target: str,
    depth_step_m: float = 0.1,
) -> VisualWaypoint:
    observation = cache.get_observation(str(obs_id)).observation
    if observation.depth is None:
        raise ValueError("visual grounding requires observation.depth")
    if observation.intrinsics is None:
        raise ValueError("visual grounding requires observation.intrinsics")
    if observation.T_cam_odom is None:
        raise ValueError("visual grounding requires observation.T_cam_odom")

    depth = np.asarray(observation.depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"visual grounding depth must be 2D, got shape {depth.shape}")
    metric_depth = exploration.map.metric_depth_from_observation_depth(depth)
    height, width = metric_depth.shape
    x_norm, y_norm = float(point_2d[0]), float(point_2d[1])
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        raise ValueError(f"point_2d must be in [0,1], got {point_2d!r}")
    u = float(x_norm * float(width - 1))
    v = float(y_norm * float(height - 1))
    initial_depth_m = _median_valid_depth_near_pixel_lavira(
        depth=metric_depth,
        u=u,
        v=v,
    )
    if initial_depth_m is None:
        raise ValueError(
            f"visual grounding found no valid depth near pixel ({u:.1f}, {v:.1f})"
        )

    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
        raise ValueError("visual grounding received invalid camera intrinsics")
    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    start_xy = np.asarray(
        [float(observation.pose.x), float(observation.pose.y)],
        dtype=np.float64,
    )

    step_m = float(depth_step_m)
    if step_m <= 0.0:
        raise ValueError("depth_step_m must be positive")
    minimum_depth_m = max(0.1, float(exploration.map.min_depth))
    candidate_depth_m = float(initial_depth_m)
    while candidate_depth_m >= minimum_depth_m:
        point_odom = _project_pixel_to_odom(
            u=u,
            v=v,
            depth_m=candidate_depth_m,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            T_odom_cam=T_odom_cam,
        )
        raw_world_xy = np.asarray(
            [float(point_odom[0]), float(point_odom[1])],
            dtype=np.float64,
        )
        if exploration.map.is_traversable_xy(raw_world_xy):
            path_xy = exploration.map.compute_astar_path(
                start_xy=start_xy,
                goal_xy=raw_world_xy,
            )
            if path_xy is not None and len(path_xy) > 0:
                path = [
                    (float(point[0]), float(point[1]))
                    for point in np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
                ]
                goal_xy = path[-1]
                goal_yaw = _goal_yaw_from_path_or_target(
                    path=path,
                    start_xy=start_xy,
                    raw_world_xy=raw_world_xy,
                )
                return VisualWaypoint(
                    obs_id=str(obs_id),
                    angle_deg=int(angle_deg),
                    point_2d=(float(x_norm), float(y_norm)),
                    point_pixel=(float(u), float(v)),
                    raw_world_xy=(float(raw_world_xy[0]), float(raw_world_xy[1])),
                    goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                    goal_yaw=float(goal_yaw),
                    path_xy=path,
                    waypoint_target=str(waypoint_target),
                    target=str(target),
                    depth_m=float(candidate_depth_m),
                    raw_world_z=float(point_odom[2]),
                )
        candidate_depth_m -= step_m
    raise ValueError(
        "visual grounding found no traversable target along the selected pixel ray"
    )


def _project_pixel_to_odom(
    *,
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_odom_cam: np.ndarray,
) -> np.ndarray:
    point_cam = np.asarray(
        [
            (float(u) - float(cx)) * float(depth_m) / float(fx),
            (float(v) - float(cy)) * float(depth_m) / float(fy),
            float(depth_m),
            1.0,
        ],
        dtype=np.float32,
    )
    return np.asarray(T_odom_cam, dtype=np.float32) @ point_cam


def _median_valid_depth_near_pixel_lavira(
    *,
    depth: np.ndarray,
    u: float,
    v: float,
) -> float | None:
    height, width = depth.shape
    px = int(round(float(u)))
    py = int(round(float(v)))
    for window_size in (3, 5, 7, 9):
        radius = window_size // 2
        x_min = max(0, px - radius)
        x_max = min(width, px + radius + 1)
        y_min = max(0, py - radius)
        y_max = min(height, py + radius + 1)
        window = np.asarray(depth[y_min:y_max, x_min:x_max], dtype=np.float32)
        valid = window[np.isfinite(window) & (window > 0.0)]
        if valid.size > 0:
            return float(np.median(valid))
    return None


def _median_valid_depth_near_pixel(
    *,
    depth: np.ndarray,
    u: float,
    v: float,
) -> float | None:
    height, width = depth.shape
    px = int(round(float(u)))
    py = int(round(float(v)))
    for radius in (1, 3, 5, 9, 15):
        x_min = max(0, px - radius)
        x_max = min(width, px + radius + 1)
        y_min = max(0, py - radius)
        y_max = min(height, py + radius + 1)
        window = np.asarray(depth[y_min:y_max, x_min:x_max], dtype=np.float32)
        valid = window[np.isfinite(window) & (window > 0.0)]
        if valid.size > 0:
            return float(np.median(valid))
    return None


def _goal_yaw_from_path_or_target(
    *,
    path: list[tuple[float, float]],
    start_xy: np.ndarray,
    raw_world_xy: np.ndarray,
) -> float:
    if len(path) >= 2:
        prev_xy = path[-2]
        curr_xy = path[-1]
        dx = float(curr_xy[0]) - float(prev_xy[0])
        dy = float(curr_xy[1]) - float(prev_xy[1])
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return float(math.degrees(math.atan2(dy, dx)))
    delta = np.asarray(raw_world_xy, dtype=np.float64).reshape(2) - np.asarray(start_xy, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(delta)) > 1e-6:
        return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))
    return 0.0
