from __future__ import annotations

from collections import deque
import math

from navclaw.schemas import ActionCall
from navclaw.graph.node import Node
from navclaw.graph.graph import Graph
from navclaw.types import (
    LocalmapFrontierRecord,
    LocalmapNavigationPlan,
    LocalmapRouteTarget,
)


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    stripped = str(node_id).strip()
    if stripped.startswith("n") and stripped[1:].isdigit():
        return (int(stripped[1:]), stripped)
    digits = "".join(char for char in stripped if char.isdigit())
    if digits != "":
        return (int(digits), stripped)
    return (10**9, stripped)


def _place_nodes(graph: Graph) -> list[Node]:
    return [node for node in graph.iter_nodes() if node.node_kind == "place"]


def _move_edge_place_adjacency(graph: Graph) -> dict[str, set[str]]:
    place_node_ids = {str(node.id) for node in _place_nodes(graph)}
    adjacency: dict[str, set[str]] = {
        node_id: set() for node_id in place_node_ids
    }
    for edge in graph.iter_edges(include_vertical=False):
        if str(edge.relation) != "move":
            continue
        src_id = str(edge.src_id)
        dst_id = str(edge.dst_id)
        if src_id not in place_node_ids or dst_id not in place_node_ids:
            raise ValueError(f"move edge references non-place node: {src_id!r}->{dst_id!r}")
        adjacency[src_id].add(dst_id)
        adjacency[dst_id].add(src_id)
    return adjacency


def _find_place_route_node_ids(
    graph: Graph,
    start_place_node_id: str,
    goal_place_node_id: str,
) -> list[str] | None:
    start_node_id = str(start_place_node_id)
    goal_node_id = str(goal_place_node_id)
    if start_node_id == goal_node_id:
        return [start_node_id]
    adjacency = _move_edge_place_adjacency(graph)
    if start_node_id not in adjacency or goal_node_id not in adjacency:
        return None
    parent: dict[str, str | None] = {start_node_id: None}
    queue: deque[str] = deque([start_node_id])
    while queue:
        node_id = queue.popleft()
        if node_id == goal_node_id:
            break
        for neighbor_node_id in sorted(adjacency[node_id], key=_node_id_sort_key):
            if neighbor_node_id in parent:
                continue
            parent[neighbor_node_id] = node_id
            queue.append(neighbor_node_id)
    if goal_node_id not in parent:
        return None
    route_node_ids: list[str] = []
    cursor: str | None = goal_node_id
    while cursor is not None:
        route_node_ids.append(cursor)
        cursor = parent[cursor]
    route_node_ids.reverse()
    return route_node_ids


def find_place_route_node_ids(
    graph: Graph,
    start_place_node_id: str,
    goal_place_node_id: str,
) -> list[str] | None:
    return _find_place_route_node_ids(
        graph=graph,
        start_place_node_id=start_place_node_id,
        goal_place_node_id=goal_place_node_id,
    )


def find_nearest_place_node_id(
    graph: Graph,
    xy: tuple[float, float],
) -> str | None:
    place_nodes = _place_nodes(graph)
    if place_nodes == []:
        return None
    x = float(xy[0])
    y = float(xy[1])
    nearest = min(
        place_nodes,
        key=lambda node: (
            math.hypot(float(node.position[0]) - x, float(node.position[1]) - y),
            _node_id_sort_key(str(node.id)),
        ),
    )
    return str(nearest.id)


def build_place_route_segment_plans(
    graph: Graph,
    start_place_node_id: str,
    goal_place_node_id: str,
    *,
    include_start_node_hop: bool = False,
    start_node_hop_segment_type: str = "route_start_node_hop",
    route_leg_segment_type: str = "place_route_leg",
    segment_index_start: int = 0,
) -> dict[str, object] | None:
    start_node_id = str(start_place_node_id)
    goal_node_id = str(goal_place_node_id)
    route_node_ids = _find_place_route_node_ids(
        graph=graph,
        start_place_node_id=start_node_id,
        goal_place_node_id=goal_node_id,
    )
    if route_node_ids is None:
        return None

    segment_plans: list[dict[str, object]] = []
    segment_index = int(segment_index_start)
    if include_start_node_hop:
        start_node = graph.get_node(start_node_id)
        start_goal_pose = {
            "x": float(start_node.position[0]),
            "y": float(start_node.position[1]),
            "yaw": float(start_node.yaw),
        }
        segment_plans.append(
            {
                "segment_index": int(segment_index),
                "segment_type": str(start_node_hop_segment_type),
                "from_place_node_id": None,
                "to_place_node_id": start_node_id,
                "goal_pose": dict(start_goal_pose),
                "move_call": ActionCall(action="move", args=dict(start_goal_pose)).to_dict(),
            }
        )
        segment_index += 1

    for from_place_node_id, to_place_node_id in zip(route_node_ids[:-1], route_node_ids[1:]):
        next_place_node = graph.get_node(str(to_place_node_id))
        goal_pose = {
            "x": float(next_place_node.position[0]),
            "y": float(next_place_node.position[1]),
            "yaw": float(next_place_node.yaw),
        }
        segment_plans.append(
            {
                "segment_index": int(segment_index),
                "segment_type": str(route_leg_segment_type),
                "from_place_node_id": str(from_place_node_id),
                "to_place_node_id": str(to_place_node_id),
                "goal_pose": dict(goal_pose),
                "move_call": ActionCall(action="move", args=dict(goal_pose)).to_dict(),
            }
        )
        segment_index += 1

    return {
        "route_node_ids": route_node_ids,
        "segment_plans": segment_plans,
    }


def _refresh_route_target(
    route_target: LocalmapRouteTarget,
    graph: Graph,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> LocalmapRouteTarget | None:
    frontier_id = str(route_target.frontier_id)
    frontier_record = frontier_records.get(frontier_id)
    if frontier_record is None:
        return None
    owner_node_id = str(frontier_record.source_node_id)
    graph.get_node(owner_node_id)
    return LocalmapRouteTarget(
        frontier_id=frontier_id,
        owner_node_id=owner_node_id,
        overlay_id=route_target.overlay_id,
        selection_mode=str(route_target.selection_mode),
        selection_thought=str(route_target.selection_thought),
    )


def _build_navigation_plan_for_route_target(
    graph: Graph,
    current_place_node_id: str,
    route_target: LocalmapRouteTarget,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> LocalmapNavigationPlan | None:
    frontier_id = str(route_target.frontier_id)
    frontier_record = frontier_records.get(frontier_id)
    if frontier_record is None:
        return None
    owner_node_id = None if route_target.owner_node_id is None else str(route_target.owner_node_id)
    frontier_goal_pose = {
        "x": float(frontier_record.goal_xy[0]),
        "y": float(frontier_record.goal_xy[1]),
        "yaw": float(frontier_record.goal_yaw_deg),
    }
    return LocalmapNavigationPlan(
        navigation_mode="frontier_path_move",
        target_frontier_id=frontier_id,
        target_owner_node_id=owner_node_id,
        place_route_node_ids=[],
        segment_plans=[
            {
                "segment_index": 0,
                "segment_type": "frontier_path_move",
                "from_place_node_id": str(current_place_node_id),
                "to_place_node_id": None,
                "goal_pose": dict(frontier_goal_pose),
                "move_call": ActionCall(action="move", args=dict(frontier_goal_pose)).to_dict(),
            }
        ],
        consume_frontier_after_move=True,
    )
