from __future__ import annotations

import math

from navclaw.mapping.exploration.bev_map import FrontierCandidate
from navclaw.graph.graph import Graph
from navclaw.types import LocalmapFrontierRecord, LocalmapOverlayRecord


def _frontier_id_sort_key(frontier_id: str) -> tuple[int, int, str]:
    parts = str(frontier_id).split("-", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]), str(frontier_id))
    return (10**9, 10**9, str(frontier_id))


def _copy_frontier_candidate(candidate: FrontierCandidate, frontier_id: str) -> FrontierCandidate:
    return FrontierCandidate(
        frontier_id=str(frontier_id),
        xy=(float(candidate.xy[0]), float(candidate.xy[1])),
        goal_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        path_xy=[(float(x), float(y)) for x, y in candidate.path_xy],
        path_length=float(candidate.path_length),
        score_mean=float(candidate.score_mean),
    )


def _frontier_candidate_from_record(record: LocalmapFrontierRecord) -> FrontierCandidate:
    return FrontierCandidate(
        frontier_id=str(record.frontier_id),
        xy=(float(record.frontier_xy[0]), float(record.frontier_xy[1])),
        goal_xy=(float(record.goal_xy[0]), float(record.goal_xy[1])),
        path_xy=[(float(x), float(y)) for x, y in record.path_xy],
        path_length=float(record.path_length),
        score_mean=float(record.score_mean),
    )


def _active_frontier_ids_by_overlay(
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> dict[str, list[str]]:
    active_by_overlay: dict[str, list[str]] = {}
    for frontier_id, record in frontier_records.items():
        overlay_id = str(record.overlay_id)
        if overlay_id == "":
            continue
        active_by_overlay.setdefault(overlay_id, []).append(str(frontier_id))
    for frontier_ids in active_by_overlay.values():
        frontier_ids.sort(key=_frontier_id_sort_key)
    return active_by_overlay


def _sync_frontier_overlay_state(
    frontier_records: dict[str, LocalmapFrontierRecord],
    overlay_records: dict[str, LocalmapOverlayRecord],
) -> None:
    active_overlays = {
        str(record.overlay_id)
        for record in frontier_records.values()
        if str(record.overlay_id) != ""
    }
    stale_overlay_ids = [
        overlay_id
        for overlay_id in overlay_records
        if overlay_id not in active_overlays
    ]
    for overlay_id in stale_overlay_ids:
        del overlay_records[overlay_id]


def _set_node_frontier_record(
    *,
    graph: Graph,
    record: LocalmapFrontierRecord,
) -> None:
    node = graph.get_node(str(record.source_node_id))
    node.frontiers[str(record.frontier_id)] = record


def _remove_node_frontier_record(
    *,
    graph: Graph,
    record: LocalmapFrontierRecord,
) -> None:
    node = graph.get_node(str(record.source_node_id))
    node.frontiers.pop(str(record.frontier_id), None)


def _remove_frontier_record(
    frontier_records: dict[str, LocalmapFrontierRecord],
    frontier_id: str,
    graph: Graph | None = None,
) -> None:
    record = frontier_records.pop(str(frontier_id), None)
    if graph is not None and record is not None:
        _remove_node_frontier_record(graph=graph, record=record)


def _remove_owned_frontier_records(
    frontier_records: dict[str, LocalmapFrontierRecord],
    source_node_id: str,
    graph: Graph | None = None,
) -> None:
    stale_frontier_ids = [
        frontier_id
        for frontier_id, record in frontier_records.items()
        if str(record.source_node_id) == str(source_node_id)
    ]
    for frontier_id in stale_frontier_ids:
        record = frontier_records.pop(frontier_id)
        if graph is not None:
            _remove_node_frontier_record(graph=graph, record=record)


def _current_place_active_frontier_ids(
    current_place_node_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> list[str]:
    frontier_ids: list[str] = []
    for frontier_id, record in frontier_records.items():
        if str(record.source_node_id) != str(current_place_node_id):
            continue
        frontier_ids.append(str(frontier_id))
    frontier_ids.sort(key=_frontier_id_sort_key)
    return frontier_ids


def _current_place_active_overlay_frontier_ids(
    current_place_node_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> list[str]:
    frontier_ids: list[str] = []
    for frontier_id, record in frontier_records.items():
        if str(record.source_node_id) != str(current_place_node_id):
            continue
        if str(record.overlay_id) == "":
            continue
        frontier_ids.append(str(frontier_id))
    frontier_ids.sort(key=_frontier_id_sort_key)
    return frontier_ids


def _has_any_active_overlay_frontier(
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> bool:
    return any(str(record.overlay_id) != "" for record in frontier_records.values())


def _select_nearest_active_candidate(
    graph: Graph,
    current_place_node_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> dict[str, object] | None:
    current_place_node = graph.get_node(str(current_place_node_id))
    current_xy = (float(current_place_node.position[0]), float(current_place_node.position[1]))
    best_payload: dict[str, object] | None = None
    best_distance: float | None = None
    for frontier_id, record in frontier_records.items():
        distance = math.hypot(
            current_xy[0] - float(record.goal_xy[0]),
            current_xy[1] - float(record.goal_xy[1]),
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_payload = {
                "mode": "nearest_active_frontier_fallback",
                "frontier_id": str(frontier_id),
                "goal_pose": {
                    "x": float(record.goal_xy[0]),
                    "y": float(record.goal_xy[1]),
                    "yaw": float(record.goal_yaw_deg),
                },
                "source_node_id": str(record.source_node_id),
                "distance_m": float(distance),
            }
    return best_payload


def _resolve_localmap_frontier_goal_pose(
    frontier_records: dict[str, LocalmapFrontierRecord],
    frontier_id: str,
) -> dict[str, float]:
    frontier_id_str = str(frontier_id)
    record = frontier_records.get(frontier_id_str)
    if record is None:
        raise ValueError(f"unknown localmap frontier_id: {frontier_id_str!r}")
    return {
        "x": float(record.goal_xy[0]),
        "y": float(record.goal_xy[1]),
        "yaw": float(record.goal_yaw_deg),
    }
