from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np
from skimage.morphology import medial_axis
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from PIL import Image, ImageDraw

from navprobe.agent.visual_grounding import VisualWaypoint
from navprobe.agent.vln_waypoint_sampling import draw_vln_sampled_waypoint_rgb_overlay
from navprobe.agent.vln_waypoint_sampling import _projection_view_score
from navprobe.agent.vln_waypoint_sampling import _projected_views_for_waypoint
from navprobe.agent.vln_waypoint_sampling import visual_waypoint_from_sampled_candidate
from navprobe.agent.vln_waypoint_sampling import VLN_WAYPOINT_MAX_CANDIDATES
from navprobe.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate

if TYPE_CHECKING:
    from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.runtime.cache import RuntimeCache


VERTICAL_FSS_PIXELS_PER_METER = 20
VERTICAL_FSS_LOCAL_RADIUS_M = 4.5
VERTICAL_FSS_MAX_VERTICAL_DELTA_M = 1.1
VERTICAL_FSS_MAX_NEIGHBOR_HEIGHT_DELTA_M = 0.24
VERTICAL_FSS_MIN_CANDIDATE_DISTANCE_M = 0.4
VERTICAL_FSS_MAX_CANDIDATE_XY_DISTANCE_M = 2.0
VERTICAL_FSS_MIN_CANDIDATE_CLEARANCE_M = 0.12
VERTICAL_FSS_CANDIDATE_NMS_DISTANCE_M = 0.6
VERTICAL_FSS_MIN_DIRECTIONAL_PROGRESS_M = 0.08
VERTICAL_FSS_OCCLUSION_TOLERANCE_M = 0.18


@dataclass(frozen=True)
class VerticalFssCandidate:
    label: int
    goal_xyz: tuple[float, float, float]
    path_xyz: list[tuple[float, float, float]]
    height_delta_m: float
    directional_progress_m: float
    clearance_m: float
    euclidean_distance_m: float
    path_length_m: float
    anchor_source: str = "skeleton"

    def to_dict(self) -> dict[str, object]:
        return {
            "label": int(self.label),
            "anchor_source": self.anchor_source,
            "goal_xyz": [float(value) for value in self.goal_xyz],
            "path_xyz": [
                [float(x), float(y), float(z)] for x, y, z in self.path_xyz
            ],
            "height_delta_m": float(self.height_delta_m),
            "directional_progress_m": float(self.directional_progress_m),
            "clearance_m": float(self.clearance_m),
            "euclidean_distance_m": float(self.euclidean_distance_m),
            "path_length_m": float(self.path_length_m),
        }


@dataclass(frozen=True)
class VerticalFssViewCandidateSet:
    view: "VisualViewContext"
    candidates: list[VlnSampledWaypointCandidate]
    rgb_overlay: np.ndarray


@dataclass(frozen=True)
class PreparedVerticalFssCandidates:
    direction: str
    robot_xyz: tuple[float, float, float]
    candidates: list[VerticalFssCandidate]
    view_candidate_sets: list[VerticalFssViewCandidateSet]
    height_connected_cell_count: int
    detected_stair_regions: list[dict[str, object]] = field(default_factory=list)
    temporary_stair_cell_count: int = 0
    bev_overlay: np.ndarray | None = None

    @property
    def available_labels_by_angle(self) -> dict[int, set[int]]:
        return {
            int(item.view.angle_deg): {
                int(candidate.label) for candidate in item.candidates
            }
            for item in self.view_candidate_sets
        }

    def selected(
        self,
        *,
        candidate_label: int,
    ) -> tuple["VisualViewContext", VerticalFssCandidate, VlnSampledWaypointCandidate]:
        visible_projections: list[
            tuple[float, VerticalFssViewCandidateSet, VlnSampledWaypointCandidate]
        ] = []
        for item in self.view_candidate_sets:
            image_height, image_width = np.asarray(item.rgb_overlay).shape[:2]
            for candidate in item.candidates:
                if int(candidate.label) != int(candidate_label):
                    continue
                visible_projections.append(
                    (
                        _projection_view_score(
                            point_pixel=candidate.point_pixel,
                            image_width=int(image_width),
                            image_height=int(image_height),
                            projected_depth_m=float(candidate.projected_depth_m),
                        ),
                        item,
                        candidate,
                    )
                )
        if visible_projections == []:
            raise ValueError(
                f"vertical FSS candidate label is unavailable: {candidate_label}"
            )
        _score, selected_view, projected_candidate = max(
            visible_projections,
            key=lambda item: float(item[0]),
        )
        world_candidate = next(
            (
                candidate
                for candidate in self.candidates
                if int(candidate.label) == int(candidate_label)
            ),
            None,
        )
        if world_candidate is None:
            raise ValueError(f"vertical FSS candidate label is unavailable: {candidate_label}")
        return selected_view.view, world_candidate, projected_candidate

    def selected_overlay(self, *, candidate_label: int, cache) -> np.ndarray:
        selected_view, _world_candidate, projected_candidate = self.selected(
            candidate_label=candidate_label,
        )
        return draw_vln_sampled_waypoint_rgb_overlay(
            cache=cache,
            selected_view=selected_view,
            candidates=[projected_candidate],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": str(self.direction),
            "robot_xyz": [float(value) for value in self.robot_xyz],
            "max_candidate_xy_distance_m": float(
                VERTICAL_FSS_MAX_CANDIDATE_XY_DISTANCE_M
            ),
            "height_connected_cell_count": int(self.height_connected_cell_count),
            "detected_stair_regions": self.detected_stair_regions,
            "temporary_stair_cell_count": self.temporary_stair_cell_count,
            "candidate_count": len(self.candidates),
            "available_labels_by_angle": {
                str(angle): sorted(labels)
                for angle, labels in self.available_labels_by_angle.items()
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def prepare_vertical_fss_candidates(
    *,
    cache: "RuntimeCache",
    visual_context: "VisualActionContext",
    exploration: "ExplorationManager",
    direction: str,
    detector,
    map_floor_height: float,
    frontier_only: bool = False,
) -> PreparedVerticalFssCandidates:
    direction_text = str(direction).strip().lower()
    if direction_text not in {"up", "down"}:
        raise ValueError(f"vertical FSS requires direction up or down, got {direction!r}")
    if detector is None:
        raise ValueError("vertical FSS requires the configured stair detector")
    if visual_context.views == []:
        raise ValueError("vertical FSS requires at least one panorama view")

    anchor_view = next(
        (
            view
            for view in visual_context.views
            if int(view.angle_deg) == 0
        ),
        visual_context.views[0],
    )
    anchor_observation = cache.get_observation(str(anchor_view.obs_id)).observation
    robot_xyz = np.asarray(
        [
            float(anchor_observation.pose.x),
            float(anchor_observation.pose.y),
            float(anchor_observation.pose.z),
        ],
        dtype=np.float64,
    )
    base_z = float(robot_xyz[2])
    grid_size = int(
        round(2.0 * VERTICAL_FSS_LOCAL_RADIUS_M * VERTICAL_FSS_PIXELS_PER_METER)
    ) + 1
    origin_xy = robot_xyz[:2] - VERTICAL_FSS_LOCAL_RADIUS_M
    robot_pixel = (grid_size // 2, grid_size // 2)
    support_heights = np.full(
        (grid_size, grid_size),
        -np.inf if direction_text == "up" else np.inf,
        dtype=np.float32,
    )
    support_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    observed = np.zeros((grid_size, grid_size), dtype=bool)
    stair_cells = np.zeros_like(observed)
    detected_regions: list[dict[str, object]] = []

    for view in visual_context.views:
        observation = cache.get_observation(str(view.obs_id)).observation
        if observation.depth is None:
            raise ValueError(f"vertical FSS observation {view.obs_id} has no depth")
        if observation.intrinsics is None:
            raise ValueError(f"vertical FSS observation {view.obs_id} has no intrinsics")
        if observation.T_cam_odom is None:
            raise ValueError(f"vertical FSS observation {view.obs_id} has no T_cam_odom")
        depth = exploration.map.metric_depth_from_observation_depth(
            np.asarray(observation.depth, dtype=np.float32)
        )
        intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
        T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
        valid_depth = (
            np.isfinite(depth)
            & (depth > 0.1)
            & (depth < float(exploration.map.max_depth))
        )
        points_cam, points_odom = _backproject(depth, intrinsics, T_cam_odom)
        height_delta = points_odom[:, :, 2] - base_z
        camera_z = float(np.linalg.inv(T_cam_odom)[2, 3])
        support_band = (
            (height_delta >= -0.12)
            & (height_delta <= VERTICAL_FSS_MAX_VERTICAL_DELTA_M)
            if direction_text == "up"
            else (height_delta <= 0.12)
            & (height_delta >= -VERTICAL_FSS_MAX_VERTICAL_DELTA_M)
        )
        support_mask = (
            _horizontal_support_mask(points_cam, T_cam_odom, valid_depth)
            & support_band
            & (points_odom[:, :, 2] < camera_z - 0.08)
        )
        stair_mask, regions = _detected_stair_mask(
            detector=detector, observation=observation, obs_id=str(view.obs_id)
        )
        detected_regions.extend(regions)
        # This local support map is discarded after grounding. Only the
        # detected stair region can add off-floor traversable cells.
        tx, ty, inside = _grid_indices(
            points_odom[support_mask & stair_mask, :2], origin_xy, grid_size
        )
        stair_cells[ty[inside], tx[inside]] = True
        ox, oy, inside = _grid_indices(
            points_odom[valid_depth, :2], origin_xy, grid_size
        )
        observed[oy[inside], ox[inside]] = True
        support_points = points_odom[support_mask]
        sx, sy, inside = _grid_indices(
            support_points[:, :2],
            origin_xy,
            grid_size,
        )
        sx, sy = sx[inside], sy[inside]
        support_z = support_points[:, 2][inside]
        np.add.at(support_count, (sy, sx), 1)
        if direction_text == "up":
            np.maximum.at(support_heights, (sy, sx), support_z)
        else:
            np.minimum.at(support_heights, (sy, sx), support_z)

    support = support_count > 0
    support_heights[~support] = base_z
    support_clearance = None
    support, support_heights, map_known = _fuse_vertical_support_with_map(
        map_obj=exploration.map, origin_xy=origin_xy,
        observed_support=support, observed_heights=support_heights,
        stair_cells=stair_cells,
        floor_height=float(map_floor_height),
    )
    observed |= map_known
    support_clearance = distance_transform_edt(support)
    connected, parents, seed = _connected_support(
        support,
        support_heights,
        robot_pixel,
        base_z,
        require_robot_seed=True,
        clearance_radius_m=float(exploration.map.agent_radius),
    )
    frontier_mask = None
    raw_frontier = support & binary_dilation(~observed)
    frontier_mask = binary_dilation(
        raw_frontier,
        iterations=max(1, int(math.ceil(float(exploration.map.agent_radius) * VERTICAL_FSS_PIXELS_PER_METER))),
    )
    sampled = _sample_vertical_candidates(
        connected=connected,
        heights=support_heights,
        parents=parents,
        seed=seed,
        robot_pixel=robot_pixel,
        base_z=base_z,
        direction=direction_text,
        observed=observed,
        frontier_only=frontier_only,
        support_clearance=support_clearance,
        frontier_mask=frontier_mask,
    )

    candidates: list[VerticalFssCandidate] = []
    projections_by_angle: dict[int, list[VlnSampledWaypointCandidate]] = {
        int(view.angle_deg): [] for view in visual_context.views
    }
    for sampled_candidate in sampled:
        # Each anchor has a collision-free support path. Pick a visible
        # prefix endpoint, rather than discarding an occluded/far anchor.
        path_pixels = list(sampled_candidate["path_pixels"])
        projections = []
        for path_index in range(len(path_pixels) - 1, -1, -1):
            x, y = path_pixels[path_index]
            goal_xyz = (
                float(origin_xy[0] + (float(x) + 0.5) / VERTICAL_FSS_PIXELS_PER_METER),
                float(origin_xy[1] + (float(y) + 0.5) / VERTICAL_FSS_PIXELS_PER_METER),
                float(support_heights[int(y), int(x)]),
            )
            distance = float(np.linalg.norm(np.asarray(goal_xyz[:2]) - robot_xyz[:2]))
            if not VERTICAL_FSS_MIN_CANDIDATE_DISTANCE_M <= distance <= VERTICAL_FSS_MAX_CANDIDATE_XY_DISTANCE_M:
                continue
            projections = _projected_views_for_waypoint(
                waypoint_xy=np.asarray(goal_xyz[:2], dtype=np.float64),
                views=visual_context.views,
                cache=cache,
                world_z=float(goal_xyz[2]),
                occlusion_tolerance_m=VERTICAL_FSS_OCCLUSION_TOLERANCE_M,
            )
            if projections:
                break
        if not projections:
            continue
        if any(np.linalg.norm(np.asarray(goal_xyz[:2]) - np.asarray(c.goal_xyz[:2])) < VERTICAL_FSS_CANDIDATE_NMS_DISTANCE_M for c in candidates):
            continue
        path_xyz = [tuple(float(value) for value in robot_xyz)] + [
            (
                float(origin_xy[0] + (float(px) + 0.5) / VERTICAL_FSS_PIXELS_PER_METER),
                float(origin_xy[1] + (float(py) + 0.5) / VERTICAL_FSS_PIXELS_PER_METER),
                float(support_heights[int(py), int(px)]),
            )
            for px, py in path_pixels[:path_index + 1]
        ]
        label = len(candidates) + 1
        path_array = np.asarray(path_xyz, dtype=np.float64).reshape(-1, 3)
        path_length = float(
            np.linalg.norm(np.diff(path_array, axis=0), axis=1).sum()
            if len(path_array) > 1
            else 0.0
        )
        candidate = VerticalFssCandidate(
            label=label,
            goal_xyz=goal_xyz,
            path_xyz=path_xyz,
            height_delta_m=float(goal_xyz[2] - base_z),
            directional_progress_m=float((goal_xyz[2] - base_z) * (1 if direction_text == "up" else -1)),
            clearance_m=float(sampled_candidate["clearance_m"]),
            euclidean_distance_m=float(
                np.linalg.norm(np.asarray(goal_xyz[:2], dtype=np.float64) - robot_xyz[:2])
            ),
            path_length_m=path_length,
            anchor_source=str(sampled_candidate.get("anchor_source", "skeleton")),
        )
        candidates.append(candidate)
        path_xy = [(float(px), float(py)) for px, py, _pz in path_xyz]
        for _score, view, point_pixel, projected_depth in projections:
            observation = cache.get_observation(str(view.obs_id)).observation
            image_height, image_width = np.asarray(observation.rgb).shape[:2]
            projections_by_angle[int(view.angle_deg)].append(
                VlnSampledWaypointCandidate(
                    label=label,
                    goal_xy=(float(goal_xyz[0]), float(goal_xyz[1])),
                    path_xy=path_xy,
                    point_pixel=(float(point_pixel[0]), float(point_pixel[1])),
                    point_2d=(
                        float(point_pixel[0] / float(max(1, image_width - 1))),
                        float(point_pixel[1] / float(max(1, image_height - 1))),
                    ),
                    euclidean_distance_m=float(candidate.euclidean_distance_m),
                    path_length_m=float(candidate.path_length_m),
                    projected_depth_m=float(projected_depth),
                )
            )

    view_candidate_sets = [
        VerticalFssViewCandidateSet(
            view=view,
            candidates=projections_by_angle[int(view.angle_deg)],
            rgb_overlay=draw_vln_sampled_waypoint_rgb_overlay(
                cache=cache,
                selected_view=view,
                candidates=projections_by_angle[int(view.angle_deg)],
            ),
        )
        for view in visual_context.views
    ]
    return PreparedVerticalFssCandidates(
        direction=direction_text,
        robot_xyz=tuple(float(value) for value in robot_xyz),
        candidates=candidates,
        view_candidate_sets=view_candidate_sets,
        height_connected_cell_count=int(np.count_nonzero(connected)),
        detected_stair_regions=detected_regions,
        temporary_stair_cell_count=int(np.count_nonzero(stair_cells & connected)),
        bev_overlay=_draw_vertical_fss_bev(
            connected=connected, stair_cells=stair_cells,
            robot_pixel=robot_pixel, origin_xy=origin_xy, candidates=candidates,
        ),
    )


def vertical_fss_visual_waypoint(
    *,
    selected_view: "VisualViewContext",
    world_candidate: VerticalFssCandidate,
    projected_candidate: VlnSampledWaypointCandidate,
) -> VisualWaypoint:
    return visual_waypoint_from_sampled_candidate(
        selected_view=selected_view,
        candidate=projected_candidate,
        target=f"vertical FSS candidate label {int(world_candidate.label)}",
        raw_world_z=float(world_candidate.goal_xyz[2]),
    )


def _backproject(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_cam_odom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    points_cam = np.stack(
        [
            (xs - float(intrinsics[0, 2])) * depth / float(intrinsics[0, 0]),
            (ys - float(intrinsics[1, 2])) * depth / float(intrinsics[1, 1]),
            depth,
        ],
        axis=-1,
    )
    T_odom_cam = np.linalg.inv(T_cam_odom)
    points_odom = points_cam @ T_odom_cam[:3, :3].T + T_odom_cam[:3, 3]
    return points_cam, points_odom


def _horizontal_support_mask(
    points_cam: np.ndarray,
    T_cam_odom: np.ndarray,
    valid_depth: np.ndarray,
) -> np.ndarray:
    delta_x = points_cam[:, 2:, :] - points_cam[:, :-2, :]
    delta_y = points_cam[2:, :, :] - points_cam[:-2, :, :]
    normals_cam = np.cross(delta_x[1:-1, :, :], delta_y[:, 1:-1, :])
    normal_norm = np.linalg.norm(normals_cam, axis=2)
    normals_odom = normals_cam @ np.linalg.inv(T_cam_odom)[:3, :3].T
    horizontal = np.zeros(valid_depth.shape, dtype=bool)
    horizontal[1:-1, 1:-1] = (
        (normal_norm > 1e-6)
        & (
            np.abs(normals_odom[:, :, 2]) / np.clip(normal_norm, 1e-6, None)
            >= np.cos(np.deg2rad(35.0))
        )
        & valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, :-2]
        & valid_depth[1:-1, 2:]
        & valid_depth[:-2, 1:-1]
        & valid_depth[2:, 1:-1]
        & (np.linalg.norm(delta_x[1:-1, :, :], axis=2) < 0.30)
        & (np.linalg.norm(delta_y[:, 1:-1, :], axis=2) < 0.30)
    )
    return horizontal


def _grid_indices(
    points_xy: np.ndarray,
    origin_xy: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.floor(
        (points_xy - origin_xy.reshape(1, 2)) * VERTICAL_FSS_PIXELS_PER_METER
    ).astype(np.int32)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < grid_size)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < grid_size)
    )
    return pixels[:, 0], pixels[:, 1], inside


def _temporary_stair_support(*, support, heights, stair_cells, base_z, include_base=True):
    # Observed same-floor support remains traversable. Detected stairs and
    # their directly adjoining level landing surfaces are a temporary override.
    allowed = support & (stair_cells | ((np.abs(heights - base_z) <= 0.12) & include_base))
    boundary = support & binary_dilation(stair_cells) & ~allowed
    for y, x in zip(*np.nonzero(boundary)):
        y0, y1 = max(0, y - 1), min(support.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(support.shape[1], x + 2)
        neighbors = heights[y0:y1, x0:x1][stair_cells[y0:y1, x0:x1]]
        if len(neighbors) and np.min(np.abs(neighbors - heights[y, x])) <= VERTICAL_FSS_MAX_NEIGHBOR_HEIGHT_DELTA_M:
            allowed[y, x] = True
    queue = deque((int(x), int(y), float(heights[y, x])) for y, x in zip(*np.nonzero(allowed & (stair_cells | boundary))))
    while queue:
        x, y, level_height = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < support.shape[1] and 0 <= ny < support.shape[0]):
                continue
            if allowed[ny, nx] or not support[ny, nx]:
                continue
            if abs(float(heights[ny, nx]) - level_height) <= 0.04:
                allowed[ny, nx] = True
                queue.append((nx, ny, level_height))
    return allowed


def _fuse_vertical_support_with_map(
    *, map_obj, origin_xy, observed_support, observed_heights, stair_cells, floor_height,
):
    """Copy Mt into the local grid, then override observed stair/landing cells."""
    ys, xs = np.indices(observed_support.shape)
    xy = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1) / VERTICAL_FSS_PIXELS_PER_METER + origin_xy
    pixels = map_obj.xy_to_px(xy)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < map_obj.obstacle_map.shape[1])
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < map_obj.obstacle_map.shape[0]))
    known = np.zeros(xs.size, dtype=bool)
    base_free = np.zeros(xs.size, dtype=bool)
    mx, my = pixels[inside, 0], pixels[inside, 1]
    known[inside] = map_obj.known_map[my, mx]
    # Recompute footprint clearance after the stair override; Mt's existing
    # dilation would otherwise keep a detected stair entrance blocked.
    base_free[inside] = map_obj.explored_area[my, mx] & ~map_obj.obstacle_map[my, mx]
    base_free = base_free.reshape(observed_support.shape)
    override = _temporary_stair_support(
        support=observed_support, heights=observed_heights,
        stair_cells=stair_cells, base_z=floor_height, include_base=False,
    )
    override &= stair_cells | (np.abs(observed_heights - floor_height) > 0.12)
    heights = np.full(observed_support.shape, floor_height, dtype=np.float32)
    measured_base = base_free & observed_support & (np.abs(observed_heights - floor_height) <= 0.12)
    heights[measured_base | override] = observed_heights[measured_base | override]
    return base_free | override, heights, known.reshape(observed_support.shape)


def _connected_support(
    support: np.ndarray,
    heights: np.ndarray,
    robot_pixel: tuple[int, int],
    base_z: float,
    require_robot_seed: bool = False,
    clearance_radius_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    if clearance_radius_m > 0.0:
        support = support & (
            distance_transform_edt(support)
            >= float(clearance_radius_m) * VERTICAL_FSS_PIXELS_PER_METER
        )
    ys, xs = np.nonzero(support & (np.abs(heights - base_z) <= 0.25))
    parents = np.full(support.shape + (2,), -1, dtype=np.int32)
    if len(xs) == 0:
        return np.zeros_like(support), parents, None
    distances = np.hypot(xs - robot_pixel[0], ys - robot_pixel[1])
    near = distances <= 0.8 * VERTICAL_FSS_PIXELS_PER_METER
    if not bool(np.any(near)):
        return np.zeros_like(support), parents, None
    near_indices = np.flatnonzero(near)
    seed_index = int(near_indices[np.argmin(distances[near])])
    seed = (int(xs[seed_index]), int(ys[seed_index]))
    if require_robot_seed:
        rx, ry = robot_pixel
        if not support[ry, rx] or abs(float(heights[ry, rx]) - base_z) > 0.25:
            return np.zeros_like(support), parents, None
        seed = robot_pixel
    connected = np.zeros_like(support)
    connected[seed[1], seed[0]] = True
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        x, y = queue.popleft()
        for dx, dy in (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= support.shape[1] or ny >= support.shape[0]:
                continue
            if connected[ny, nx] or not support[ny, nx]:
                continue
            if dx and dy and not (support[y, nx] and support[ny, x]):
                continue
            if (
                abs(float(heights[ny, nx] - heights[y, x]))
                > VERTICAL_FSS_MAX_NEIGHBOR_HEIGHT_DELTA_M
            ):
                continue
            connected[ny, nx] = True
            parents[ny, nx] = (x, y)
            queue.append((nx, ny))
    return connected, parents, seed


def _path_to_seed(
    point: tuple[int, int],
    parents: np.ndarray,
    seed: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [point]
    while path[-1] != seed:
        x, y = path[-1]
        parent = parents[y, x]
        if int(parent[0]) < 0:
            return []
        path.append((int(parent[0]), int(parent[1])))
    path.reverse()
    return path


def _sample_vertical_candidates(
    *,
    connected: np.ndarray,
    heights: np.ndarray,
    parents: np.ndarray,
    seed: tuple[int, int] | None,
    robot_pixel: tuple[int, int],
    base_z: float,
    direction: str,
    observed: np.ndarray | None = None,
    support_clearance: np.ndarray | None = None,
    frontier_mask: np.ndarray | None = None,
    frontier_only: bool = False,
) -> list[dict[str, object]]:
    if seed is None or not bool(np.any(connected)):
        return []
    skeleton, clearance = medial_axis(connected, return_distance=True, rng=0)
    if support_clearance is not None:
        clearance = support_clearance
    progress = heights - base_z if direction == "up" else base_z - heights
    ys, xs = np.nonzero(
        skeleton
        & (
            clearance
            >= VERTICAL_FSS_MIN_CANDIDATE_CLEARANCE_M
            * VERTICAL_FSS_PIXELS_PER_METER
        )
    )
    distances = (
        np.hypot(
            xs.astype(np.float64) + 0.5 - robot_pixel[0],
            ys.astype(np.float64) + 0.5 - robot_pixel[1],
        )
        / VERTICAL_FSS_PIXELS_PER_METER
    )
    keep = (
        (distances >= VERTICAL_FSS_MIN_CANDIDATE_DISTANCE_M)
        & (distances <= VERTICAL_FSS_MAX_CANDIDATE_XY_DISTANCE_M)
    )
    points = [
        (int(x), int(y), float(point_progress), float(point_clearance))
        for x, y, point_progress, point_clearance in zip(
            xs[keep],
            ys[keep],
            progress[ys[keep], xs[keep]],
            clearance[ys[keep], xs[keep]],
        )
    ]
    if frontier_only:
        points = []
    sources = {(x, y): "skeleton" for x, y, _, _ in points}
    if observed is not None:
        # Frontiers border unobserved cells. Move their anchors inward onto
        # support with robot clearance, retaining the frontier provenance.
        frontier = connected & (binary_dilation(~observed) if frontier_mask is None else frontier_mask)
        components, count = label(frontier)
        safe_y, safe_x = np.nonzero(
            connected & (clearance >= VERTICAL_FSS_MIN_CANDIDATE_CLEARANCE_M * VERTICAL_FSS_PIXELS_PER_METER)
        )
        for component in range(1, count + 1):
            fy, fx = np.nonzero(components == component)
            if len(fx) < 3 or not len(safe_x):
                continue
            center = np.array([np.mean(fx), np.mean(fy)])
            nearest = int(np.argmin((safe_x - center[0]) ** 2 + (safe_y - center[1]) ** 2))
            x, y = int(safe_x[nearest]), int(safe_y[nearest])
            distance = math.hypot(x + 0.5 - robot_pixel[0], y + 0.5 - robot_pixel[1]) / VERTICAL_FSS_PIXELS_PER_METER
            if distance >= VERTICAL_FSS_MIN_CANDIDATE_DISTANCE_M:
                points.append((x, y, float(progress[y, x]), float(clearance[y, x])))
                sources[(x, y)] = "frontier"
    points.sort(
        key=lambda item: (
            item[2] < VERTICAL_FSS_MIN_DIRECTIONAL_PROGRESS_M,
            -item[2],
            sources[(item[0], item[1])] != "frontier",
            -item[3],
        )
    )
    selected: list[tuple[int, int, float, float]] = []
    for point in points:
        if any(
            math.hypot(point[0] - other[0], point[1] - other[1])
            < VERTICAL_FSS_CANDIDATE_NMS_DISTANCE_M
            * VERTICAL_FSS_PIXELS_PER_METER
            for other in selected
        ):
            continue
        selected.append(point)
        if len(selected) >= VLN_WAYPOINT_MAX_CANDIDATES:
            break
    return [
        {
            "pixel": (x, y),
            "anchor_source": sources[(x, y)],
            "height_delta_m": float(heights[y, x] - base_z),
            "directional_progress_m": float(point_progress),
            "clearance_m": float(point_clearance / VERTICAL_FSS_PIXELS_PER_METER),
            "path_pixels": _path_to_seed((x, y), parents, seed),
        }
        for x, y, point_progress, point_clearance in selected
    ]


def _detected_stair_mask(*, detector, observation, obs_id):
    """Rasterize detector output only; simulator semantic masks are not used."""
    rgb = np.asarray(observation.rgb)
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    regions = []
    for detection in detector.detect(image_rgb=rgb, class_name="stairs"):
        x0, y0, x1, y1 = detection.bbox
        x0, x1 = np.clip([int(np.floor(x0)), int(np.ceil(x1))], 0, width)
        y0, y1 = np.clip([int(np.floor(y0)), int(np.ceil(y1))], 0, height)
        if x1 <= x0 or y1 <= y0:
            continue
        if detection.mask is not None:
            region_mask = np.asarray(detection.mask, dtype=bool)
            if region_mask.shape != mask.shape:
                raise ValueError("stair detector mask must match observation shape")
            mask |= region_mask
        else:
            mask[y0:y1, x0:x1] = True
        regions.append({"obs_id": obs_id, "bbox": [int(x0), int(y0), int(x1), int(y1)], "score": float(detection.score)})
    return mask, regions


def _draw_vertical_fss_bev(*, connected, stair_cells, robot_pixel, origin_xy, candidates):
    rgb = np.full((*connected.shape, 3), 50, dtype=np.uint8)
    rgb[connected] = (220, 220, 220)
    rgb[connected & stair_cells] = (160, 205, 240)
    scale = 3
    image = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    rx, ry = (int(value * scale) for value in robot_pixel)
    draw.ellipse((rx - 5, ry - 5, rx + 5, ry + 5), fill=(255, 120, 0))
    for candidate in candidates:
        points = (np.asarray(candidate.path_xyz)[:, :2] - origin_xy) * VERTICAL_FSS_PIXELS_PER_METER * scale
        draw.line([tuple(point) for point in points], fill=(90, 140, 190), width=2)
        x, y = points[-1]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(220, 30, 30))
        draw.text((x, y), str(candidate.label), fill="white", anchor="mm")
    return np.asarray(image)
