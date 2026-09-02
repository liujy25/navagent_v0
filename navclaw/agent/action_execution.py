from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from navclaw.schemas import ActionCall, ActionResult
from navclaw.perception.detectors.interface import DetectorInterface
from navclaw.env.interface import EnvInterface
from navclaw.env.interface import RawObservation
from navclaw.graph.graph import Graph
from navclaw.perception import geometry as perception
from navclaw.runtime.cache import RuntimeCache
from navclaw.perception.geometry import compute_object_approach

MOVE_TO_OBJECT_STOP_DISTANCE_M = 0.6


def _capture_step_observation_enabled(
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    step_observation_enabled: Callable[[], bool] | None,
) -> bool:
    if step_observation_handler is None:
        return False
    if step_observation_enabled is None:
        return True
    return bool(step_observation_enabled())


def _apply_step_observation(
    observation: RawObservation,
    source: str,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
) -> dict[str, object] | None:
    if step_observation_handler is None:
        return None
    return step_observation_handler(observation, source)


def _apply_intermediate_observations_to_bev(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
    step_observation_enabled: Callable[[], bool] | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
    rgb_history_sample_interval: int = 2,
) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    total_start = time.perf_counter()
    observations = env.pop_last_move_intermediate_observations()
    step_payloads: list[dict[str, object]] = []
    rgb_history_obs_ids: list[str] = []
    observed_path_xy: list[list[float]] = []
    capture_step_observations = _capture_step_observation_enabled(
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
    )
    sample_interval = max(1, int(rgb_history_sample_interval))
    timing = {
        "intermediate_observation_count": int(len(observations)),
        "capture_step_observations": bool(capture_step_observations),
        "rgb_history_sample_interval": int(sample_interval),
        "bev_update_seconds": 0.0,
        "visualization_emit_seconds": 0.0,
        "step_observation_handler_seconds": 0.0,
        "rgb_history_handler_seconds": 0.0,
        "step_observation_payload_count": 0,
        "rgb_history_obs_count": 0,
    }
    for move_obs_index, observation in enumerate(observations):
        _append_observation_path_xy(observed_path_xy, observation)
        payload: dict[str, object] | None = None
        bev_start = time.perf_counter()
        exploration.observe_raw_observation(observation)
        timing["bev_update_seconds"] = float(timing["bev_update_seconds"]) + (
            time.perf_counter() - bev_start
        )
        if hasattr(env, "emit_visualization_observation"):
            visualization_start = time.perf_counter()
            env.emit_visualization_observation(observation)
            timing["visualization_emit_seconds"] = float(timing["visualization_emit_seconds"]) + (
                time.perf_counter() - visualization_start
            )
        if capture_step_observations:
            handler_start = time.perf_counter()
            payload = _apply_step_observation(
                observation=observation,
                source="move_intermediate",
                step_observation_handler=step_observation_handler,
            )
            timing["step_observation_handler_seconds"] = float(
                timing["step_observation_handler_seconds"]
            ) + (time.perf_counter() - handler_start)
            if payload is not None:
                step_payloads.append(payload)
        if rgb_history_observation_handler is not None and int(move_obs_index) % sample_interval == 0:
            existing_obs_id = None if payload is None else payload.get("obs_id")
            if existing_obs_id is not None:
                rgb_history_obs_ids.append(str(existing_obs_id))
            else:
                history_start = time.perf_counter()
                history_payload = rgb_history_observation_handler(observation, "move_rgb_history")
                timing["rgb_history_handler_seconds"] = float(
                    timing["rgb_history_handler_seconds"]
                ) + (time.perf_counter() - history_start)
                if isinstance(history_payload, dict) and history_payload.get("obs_id") is not None:
                    rgb_history_obs_ids.append(str(history_payload["obs_id"]))
    timing["step_observation_payload_count"] = int(len(step_payloads))
    timing["rgb_history_obs_count"] = int(len(rgb_history_obs_ids))
    if observed_path_xy != []:
        timing["observed_path_xy"] = observed_path_xy
    timing["elapsed_seconds"] = float(time.perf_counter() - total_start)
    return step_payloads, rgb_history_obs_ids, timing


def _append_observation_path_xy(path_xy: list[list[float]], observation: RawObservation) -> None:
    point = [float(observation.pose.x), float(observation.pose.y)]
    if path_xy != [] and path_xy[-1] == point:
        return
    path_xy.append(point)


def _pop_last_move_timing(env: EnvInterface) -> dict[str, object]:
    pop_timing = getattr(env, "pop_last_move_timing", None)
    if not callable(pop_timing):
        return {}
    timing = pop_timing()
    if not isinstance(timing, dict):
        raise ValueError("pop_last_move_timing must return a dict")
    return dict(timing)


def _env_supports_move_along_path(env: EnvInterface) -> bool:
    supports = getattr(env, "supports_move_along_path", None)
    return callable(supports) and bool(supports())


def _move_with_agent_planned_path(
    *,
    env: EnvInterface,
    exploration,
    x: float,
    y: float,
    yaw: float,
    z: float | None = None,
) -> tuple[object, dict[str, object]]:
    planning_start = time.perf_counter()
    current_observation = env.get_obs()
    start_xy = np.asarray(
        [float(current_observation.pose.x), float(current_observation.pose.y)],
        dtype=np.float64,
    )
    goal_xy = np.asarray([float(x), float(y)], dtype=np.float64)
    path_xy = exploration.map.compute_astar_path(start_xy=start_xy, goal_xy=goal_xy)
    if path_xy is None or len(path_xy) == 0:
        raise ValueError(
            "Current global BEV map could not find a path "
            f"from ({start_xy[0]:.3f}, {start_xy[1]:.3f}) to ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})"
        )
    path_payload = [
        [float(point[0]), float(point[1])]
        for point in np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    ]
    move_along_path = getattr(env, "move_along_path")
    pose = move_along_path(path_xy=path_payload, final_yaw=float(yaw), final_z=z)
    executed_xy = _pose_xy_list(pose)
    return pose, {
        "planner": "agent_global_bev_astar",
        "elapsed_seconds": float(time.perf_counter() - planning_start),
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "requested_goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "planned_goal_xy": list(path_payload[-1]),
        "executed_goal_xy": executed_xy,
        "path_point_count": int(len(path_payload)),
        "path_xy": [[float(point[0]), float(point[1])] for point in path_payload],
    }


def _move_and_update_bev(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    step_observation_enabled: Callable[[], bool] | None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    x: float,
    y: float,
    yaw: float,
    z: float | None = None,
):
    total_start = time.perf_counter()
    env_move_start = time.perf_counter()
    agent_path_timing: dict[str, object] | None = None
    if _env_supports_move_along_path(env):
        pose, agent_path_timing = _move_with_agent_planned_path(
            env=env,
            exploration=exploration,
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            z=z,
        )
    else:
        pose = env.move(x=x, y=y, yaw=yaw, z=z)
    env_move_seconds = time.perf_counter() - env_move_start
    rpc_timing = _pop_last_move_timing(env)
    step_payloads, rgb_history_obs_ids, intermediate_processing_timing = _apply_intermediate_observations_to_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
    )
    timing: dict[str, object] = {
        "elapsed_seconds": float(time.perf_counter() - total_start),
        "env_move_call_seconds": float(env_move_seconds),
        "intermediate_processing": intermediate_processing_timing,
    }
    if rpc_timing != {}:
        timing["rpc"] = rpc_timing
    if agent_path_timing is not None:
        timing["agent_path_planning"] = agent_path_timing
    return pose, step_payloads, rgb_history_obs_ids, timing


def _path_xy_from_move_timing(timing: dict[str, object]) -> list[list[float]]:
    intermediate_processing = timing.get("intermediate_processing")
    if isinstance(intermediate_processing, dict):
        observed_path = _normalize_optional_path_xy(intermediate_processing.get("observed_path_xy"))
        if observed_path != []:
            return observed_path
    agent_path = timing.get("agent_path_planning")
    if not isinstance(agent_path, dict):
        return []
    return _normalize_optional_path_xy(agent_path.get("path_xy", []))


def _normalize_optional_path_xy(raw_path: object) -> list[list[float]]:
    if not isinstance(raw_path, list):
        return []
    return [
        [float(point[0]), float(point[1])]
        for point in raw_path
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]


def _path_xy_from_observed_motion(
    step_payloads: list[dict[str, object]],
    final_pose,
) -> list[list[float]]:
    path_xy: list[list[float]] = []

    def append_pose(raw_pose: object) -> None:
        if not isinstance(raw_pose, dict):
            return
        if raw_pose.get("x") is None or raw_pose.get("y") is None:
            return
        point = [float(raw_pose["x"]), float(raw_pose["y"])]
        if path_xy != [] and path_xy[-1] == point:
            return
        path_xy.append(point)

    for payload in step_payloads:
        if isinstance(payload, dict):
            append_pose(payload.get("pose"))
    if path_xy == []:
        return []
    append_pose(final_pose.to_dict())
    return path_xy


def _result_path_xy(
    *,
    step_payloads: list[dict[str, object]],
    final_pose,
    fallback_path_xy: list[list[float]] | list[tuple[float, float]] | None = None,
) -> list[list[float]]:
    observed_path = _path_xy_from_observed_motion(step_payloads, final_pose)
    if observed_path != []:
        return observed_path
    if fallback_path_xy is None or list(fallback_path_xy) == []:
        return []
    path_xy = _normalize_path_xy(fallback_path_xy)
    final_xy = _pose_xy_list(final_pose)
    if path_xy == [] or path_xy[-1] != final_xy:
        path_xy.append(final_xy)
    return path_xy


def _pose_xy_list(pose) -> list[float]:
    return [float(pose.x), float(pose.y)]


def _normalize_path_xy(path_xy: list[list[float]] | list[tuple[float, float]]) -> list[list[float]]:
    normalized = [
        [float(point[0]), float(point[1])]
        for point in list(path_xy)
    ]
    if normalized == []:
        raise ValueError("move_along_path requires at least one path point")
    return normalized


def _move_along_path_and_update_bev(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    step_observation_enabled: Callable[[], bool] | None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    path_xy: list[list[float]] | list[tuple[float, float]],
    final_yaw: float,
    final_z: float | None = None,
):
    total_start = time.perf_counter()
    path_payload = _normalize_path_xy(path_xy)
    final_xy = path_payload[-1]
    env_move_start = time.perf_counter()
    env_supports_path_move = _env_supports_move_along_path(env)
    if env_supports_path_move:
        pose = getattr(env, "move_along_path")(
            path_xy=path_payload,
            final_yaw=float(final_yaw),
            final_z=final_z,
        )
    else:
        pose = env.move(
            x=float(final_xy[0]),
            y=float(final_xy[1]),
            yaw=float(final_yaw),
            z=final_z,
        )
    env_move_seconds = time.perf_counter() - env_move_start
    rpc_timing = _pop_last_move_timing(env)
    step_payloads, rgb_history_obs_ids, intermediate_processing_timing = _apply_intermediate_observations_to_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
    )
    timing: dict[str, object] = {
        "elapsed_seconds": float(time.perf_counter() - total_start),
        "env_move_call_seconds": float(env_move_seconds),
        "intermediate_processing": intermediate_processing_timing,
        "stored_path_execution": {
            "planner": "stored_frontier_localmap_path",
            "env_supports_path_move": bool(env_supports_path_move),
            "path_point_count": int(len(path_payload)),
            "requested_goal_xy": [float(final_xy[0]), float(final_xy[1])],
            "executed_goal_xy": _pose_xy_list(pose),
            "final_yaw": float(final_yaw),
        },
    }
    if final_z is not None:
        timing["stored_path_execution"]["requested_goal_z"] = float(final_z)
    if rpc_timing != {}:
        timing["rpc"] = rpc_timing
    return pose, step_payloads, rgb_history_obs_ids, timing


def move(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None],
    step_observation_enabled: Callable[[], bool],
    x: float,
    y: float,
    yaw: float,
    z: float | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
) -> ActionResult:
    pose, step_payloads, rgb_history_obs_ids, move_timing = _move_and_update_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
        x=x,
        y=y,
        yaw=yaw,
        z=z,
    )
    data: dict[str, object] = {
        "pose": pose.to_dict(),
        "timing": move_timing,
    }
    if step_payloads != []:
        data["step_observations"] = step_payloads
    if rgb_history_obs_ids != []:
        data["rgb_history_obs_ids"] = [str(obs_id) for obs_id in rgb_history_obs_ids]
    path_xy = _result_path_xy(
        step_payloads=step_payloads,
        final_pose=pose,
        fallback_path_xy=_path_xy_from_move_timing(move_timing),
    )
    if path_xy != []:
        data["path_xy"] = path_xy
    return ActionResult(
        ok=True,
        data=data,
        message=f"Moved to ({x}, {y}, {yaw}).",
    )


def move_along_path(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None],
    step_observation_enabled: Callable[[], bool],
    path_xy: list[list[float]] | list[tuple[float, float]],
    final_yaw: float,
    final_z: float | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
) -> ActionResult:
    pose, step_payloads, rgb_history_obs_ids, move_timing = _move_along_path_and_update_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
        path_xy=path_xy,
        final_yaw=final_yaw,
        final_z=final_z,
    )
    data: dict[str, object] = {
        "pose": pose.to_dict(),
        "timing": move_timing,
    }
    if step_payloads != []:
        data["step_observations"] = step_payloads
    if rgb_history_obs_ids != []:
        data["rgb_history_obs_ids"] = [str(obs_id) for obs_id in rgb_history_obs_ids]
    fallback_path_xy = _path_xy_from_move_timing(move_timing)
    if fallback_path_xy == []:
        fallback_path_xy = path_xy
    result_path_xy = _result_path_xy(
        step_payloads=step_payloads,
        final_pose=pose,
        fallback_path_xy=fallback_path_xy,
    )
    if result_path_xy != []:
        data["path_xy"] = result_path_xy
    return ActionResult(
        ok=True,
        data=data,
        message=f"Moved along stored path with {len(path_xy)} point(s).",
    )


def move_to_node(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None],
    step_observation_enabled: Callable[[], bool],
    graph: Graph,
    node_id: str,
    yaw: float | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
) -> ActionResult:
    node = graph.get_node(node_id)
    target_yaw = float(node.yaw) if yaw is None else float(yaw)
    pose, step_payloads, rgb_history_obs_ids, move_timing = _move_and_update_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
        x=float(node.position[0]),
        y=float(node.position[1]),
        yaw=target_yaw,
    )
    data: dict[str, object] = {
        "pose": pose.to_dict(),
        "selected_node": {
            "node_id": str(node_id),
            "goal_pose": {
                "x": float(node.position[0]),
                "y": float(node.position[1]),
                "yaw": target_yaw,
            },
        },
        "timing": move_timing,
    }
    if step_payloads != []:
        data["step_observations"] = step_payloads
    if rgb_history_obs_ids != []:
        data["rgb_history_obs_ids"] = [str(obs_id) for obs_id in rgb_history_obs_ids]
    path_xy = _result_path_xy(
        step_payloads=step_payloads,
        final_pose=pose,
        fallback_path_xy=_path_xy_from_move_timing(move_timing),
    )
    if path_xy != []:
        data["path_xy"] = path_xy
    return ActionResult(
        ok=True,
        data=data,
        message=f"Moved to node {node_id}.",
    )


def move_to_object(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None],
    step_observation_enabled: Callable[[], bool],
    cache: RuntimeCache,
    det_id: str,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
) -> ActionResult:
    approach = compute_object_approach(
        cache=cache,
        det_id=det_id,
        stop_distance=MOVE_TO_OBJECT_STOP_DISTANCE_M,
    )
    goal_pose = approach["goal_pose"]
    depth_stats = dict(approach.get("depth_stats", {}))
    proxy_move_due_to_max_depth = bool(approach.get("proxy_move_due_to_max_depth", False))
    proxy_move_direction_available = bool(approach.get("proxy_move_direction_available", True))
    if proxy_move_due_to_max_depth and not proxy_move_direction_available:
        current_pose = dict(approach["current_pose"])
        return ActionResult(
            ok=False,
            data={
                "pose": current_pose,
                "selected_detection": {
                    "det_id": str(det_id),
                    "obs_id": str(approach["obs_id"]),
                    "class_name": str(approach["class_name"]),
                    "stop_distance": float(MOVE_TO_OBJECT_STOP_DISTANCE_M),
                    "proxy_move_due_to_max_depth": True,
                    "proxy_move_skipped": True,
                    "proxy_move_skip_reason": "direction_unavailable",
                    "proxy_move_step_m": approach.get("proxy_move_step_m"),
                    "planned_move_distance_m": approach.get("planned_move_distance_m"),
                    "estimated_distance_xy_m": approach.get("estimated_distance_xy_m"),
                    "depth_median": None if "median_depth" not in depth_stats else float(depth_stats["median_depth"]),
                    "depth_saturated_ratio": (
                        None if "saturated_ratio" not in depth_stats else float(depth_stats["saturated_ratio"])
                    ),
                    "max_depth_m": None if "max_depth_m" not in depth_stats else float(depth_stats["max_depth_m"]),
                    "approach_goal_pose": {
                        "x": float(goal_pose["x"]),
                        "y": float(goal_pose["y"]),
                        "yaw": float(goal_pose["yaw"]),
                    },
                    "aligned_goal_pose": {
                        "x": float(current_pose["x"]),
                        "y": float(current_pose["y"]),
                        "yaw": float(current_pose["yaw"]),
                    },
                },
                "proxy_move_due_to_max_depth": True,
                "proxy_move_skipped": True,
                "proxy_move_skip_reason": "direction_unavailable",
                "depth_stats": depth_stats,
                "target_position": approach["target_position"],
            },
            message=(
                f"Skipped proxy move for detected object {det_id} because the target direction "
                "could not be computed."
            ),
        )
    approach_pose, first_step_payloads, first_rgb_history_obs_ids, first_move_timing = _move_and_update_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
        x=float(goal_pose["x"]),
        y=float(goal_pose["y"]),
        yaw=float(goal_pose["yaw"]),
    )
    target_position = approach["target_position"]
    delta_x = float(target_position["x"]) - float(approach_pose.x)
    delta_y = float(target_position["y"]) - float(approach_pose.y)
    if math.hypot(delta_x, delta_y) <= 1e-6:
        corrected_yaw = float(approach_pose.yaw)
    else:
        corrected_yaw = float(math.degrees(math.atan2(delta_y, delta_x)))
    pose, second_step_payloads, second_rgb_history_obs_ids, second_move_timing = _move_and_update_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
        x=float(approach_pose.x),
        y=float(approach_pose.y),
        yaw=corrected_yaw,
    )
    if proxy_move_due_to_max_depth:
        message = (
            f"Moved toward detected object {det_id} using a max-depth proxy target; "
            "redetection is required from the new pose."
        )
    else:
        message = f"Moved toward detected object {det_id}."
    data: dict[str, object] = {
        "pose": pose.to_dict(),
        "selected_detection": {
            "det_id": str(det_id),
            "obs_id": str(approach["obs_id"]),
            "class_name": str(approach["class_name"]),
            "stop_distance": float(MOVE_TO_OBJECT_STOP_DISTANCE_M),
            "proxy_move_due_to_max_depth": bool(proxy_move_due_to_max_depth),
            "proxy_move_step_m": approach.get("proxy_move_step_m"),
            "proxy_move_direction_available": proxy_move_direction_available,
            "planned_move_distance_m": approach.get("planned_move_distance_m"),
            "estimated_distance_xy_m": approach.get("estimated_distance_xy_m"),
            "depth_median": None if "median_depth" not in depth_stats else float(depth_stats["median_depth"]),
            "depth_saturated_ratio": (
                None if "saturated_ratio" not in depth_stats else float(depth_stats["saturated_ratio"])
            ),
            "max_depth_m": None if "max_depth_m" not in depth_stats else float(depth_stats["max_depth_m"]),
            "approach_goal_pose": {
                "x": float(goal_pose["x"]),
                "y": float(goal_pose["y"]),
                "yaw": float(goal_pose["yaw"]),
            },
            "aligned_goal_pose": {
                "x": float(approach_pose.x),
                "y": float(approach_pose.y),
                "yaw": corrected_yaw,
            },
        },
        "proxy_move_due_to_max_depth": bool(proxy_move_due_to_max_depth),
        "depth_stats": depth_stats,
        "target_position": target_position,
        "timing": {
            "move_phase_count": 2,
            "move_phases": [
                {
                    "phase": "approach",
                    "timing": first_move_timing,
                },
                {
                    "phase": "align_yaw",
                    "timing": second_move_timing,
                },
            ],
        },
    }
    step_payloads = [*first_step_payloads, *second_step_payloads]
    if step_payloads != []:
        data["step_observations"] = step_payloads
    rgb_history_obs_ids = [*first_rgb_history_obs_ids, *second_rgb_history_obs_ids]
    if rgb_history_obs_ids != []:
        data["rgb_history_obs_ids"] = [str(obs_id) for obs_id in rgb_history_obs_ids]
    path_xy = _result_path_xy(step_payloads=step_payloads, final_pose=pose)
    if path_xy != []:
        data["path_xy"] = path_xy
    return ActionResult(
        ok=True,
        data=data,
        message=message,
    )


def done(env: EnvInterface) -> ActionResult:
    stop_result = env.stop()
    return ActionResult(
        ok=True,
        data={
            "done": True,
            "stop_result": stop_result,
        },
        message="Stop requested.",
    )


class LocalmapActionExecutor:
    """Executes the fixed action set used by the localmap runner."""

    def __init__(
        self,
        *,
        env: EnvInterface,
        graph: Graph,
        cache: RuntimeCache,
        detector: DetectorInterface,
        exploration,
    ) -> None:
        self.env = env
        self.graph = graph
        self.cache = cache
        self.detector = detector
        self.exploration = exploration
        self.post_execute_handler = None
        self.step_observation_handler = None
        self.step_observation_enabled_getter = None

    def should_capture_step_observations(self) -> bool:
        if self.step_observation_handler is None:
            return False
        if self.step_observation_enabled_getter is None:
            return True
        return bool(self.step_observation_enabled_getter())

    def handle_step_observation(
        self,
        observation: RawObservation,
        source: str,
    ) -> dict[str, object] | None:
        if self.step_observation_handler is None:
            return None
        return self.step_observation_handler(observation, source)

    def handle_rgb_history_observation(
        self,
        observation: RawObservation,
        source: str,
    ) -> dict[str, object]:
        record = self.cache.store_observation(observation)
        return {
            "obs_id": str(record.id),
            "rgb_id": f"{record.id}:rgb",
            "depth_id": f"{record.id}:depth",
            "pose": observation.pose.to_dict(),
            "source": str(source),
        }

    def execute(self, call: ActionCall) -> ActionResult:
        action_name = str(call.action)
        args = dict(call.args)
        if action_name == "move":
            result = move(
                self.env,
                self.exploration,
                self.handle_step_observation,
                self.should_capture_step_observations,
                rgb_history_observation_handler=self.handle_rgb_history_observation,
                **args,
            )
        elif action_name == "move_along_path":
            result = move_along_path(
                self.env,
                self.exploration,
                self.handle_step_observation,
                self.should_capture_step_observations,
                rgb_history_observation_handler=self.handle_rgb_history_observation,
                **args,
            )
        elif action_name == "move_to_node":
            result = move_to_node(
                self.env,
                self.exploration,
                self.handle_step_observation,
                self.should_capture_step_observations,
                self.graph,
                rgb_history_observation_handler=self.handle_rgb_history_observation,
                **args,
            )
        elif action_name == "move_to_object":
            result = move_to_object(
                self.env,
                self.exploration,
                self.handle_step_observation,
                self.should_capture_step_observations,
                self.cache,
                rgb_history_observation_handler=self.handle_rgb_history_observation,
                **args,
            )
        elif action_name == "get_obs":
            result = perception.get_obs(self.env, self.cache, self.exploration)
        elif action_name == "detect":
            result = perception.detect(self.detector, self.cache, **args)
        elif action_name == "done":
            result = done(self.env)
        else:
            raise ValueError(f"unsupported localmap action: {action_name!r}")
        if self.post_execute_handler is not None:
            self.post_execute_handler(call, result)
        return result
