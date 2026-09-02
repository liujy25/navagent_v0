from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from navclaw.graph.edge import Edge
from navclaw.memory.entity_knowledge import EntityKnowledge
from navclaw.memory.entity_knowledge import next_entity_knowledge_id

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState, NavClawStepState


def set_pending_node_move(
    *,
    state: "NavClawAgentState",
    step_id: int,
    from_node_id: str | None,
    reason: str,
    move_mode: str = "move",
    backtrack_reference_node_id: str | None = None,
    backtrack_contexts: list[dict[str, object]] | None = None,
    rgb_history_obs_ids: list[str] | None = None,
    path_xy: list[list[float]] | list[tuple[float, float]] | None = None,
) -> None:
    if from_node_id is None or str(from_node_id).strip() == "":
        return
    pending: dict[str, object] = {
        "step_id": int(step_id),
        "from_node": str(from_node_id),
        "move_mode": str(move_mode),
        "reason": str(reason),
        "rgb_history_obs_ids": [] if rgb_history_obs_ids is None else [str(obs_id) for obs_id in rgb_history_obs_ids],
        "path_xy": _normalize_path_xy(path_xy),
    }
    backtrack_reference_node_text = (
        ""
        if backtrack_reference_node_id is None
        else str(backtrack_reference_node_id).strip()
    )
    if backtrack_reference_node_text != "":
        pending["backtrack_reference_node_id"] = backtrack_reference_node_text
    normalized_backtrack_contexts = [
        deepcopy(item)
        for item in list(backtrack_contexts or [])
        if isinstance(item, dict)
    ]
    if normalized_backtrack_contexts != []:
        pending["backtrack_contexts"] = normalized_backtrack_contexts
    state.pending_node_move = pending


def complete_pending_node_move(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> dict[str, object] | None:
    pending = getattr(state, "pending_node_move", None)
    if not isinstance(pending, dict):
        return None
    current_node_id = getattr(state, "current_place_node_id", None)
    if current_node_id is None or str(current_node_id).strip() == "":
        return None
    from_node_id = str(pending.get("from_node", "")).strip()
    to_node_id = str(current_node_id).strip()
    state.pending_node_move = None
    if from_node_id == "" or to_node_id == "" or from_node_id == to_node_id:
        return None
    return record_node_move(
        state=state,
        step_id=int(pending.get("step_id", getattr(step, "place_step_index", 0))),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        reason=str(pending.get("reason", "")),
        move_mode=str(pending.get("move_mode", "move")),
        backtrack_reference_node_id=str(
            pending.get("backtrack_reference_node_id", "")
        ).strip(),
        backtrack_contexts=[
            deepcopy(item)
            for item in list(pending.get("backtrack_contexts", []))
            if isinstance(item, dict)
        ],
        rgb_history_obs_ids=[
            str(obs_id)
            for obs_id in list(pending.get("rgb_history_obs_ids", []))
        ],
        path_xy=_normalize_path_xy(pending.get("path_xy", [])),
    )


def record_node_move(
    *,
    state: "NavClawAgentState",
    step_id: int,
    from_node_id: str | None,
    to_node_id: str | None,
    reason: str,
    move_mode: str = "move",
    backtrack_reference_node_id: str | None = None,
    backtrack_contexts: list[dict[str, object]] | None = None,
    rgb_history_obs_ids: list[str] | None = None,
    path_xy: list[list[float]] | list[tuple[float, float]] | None = None,
) -> dict[str, object] | None:
    if from_node_id is None or to_node_id is None:
        return None
    from_node_text = str(from_node_id).strip()
    to_node_text = str(to_node_id).strip()
    if from_node_text == "" or to_node_text == "" or from_node_text == to_node_text:
        return None
    rgb_history = [] if rgb_history_obs_ids is None else [str(obs_id) for obs_id in rgb_history_obs_ids]
    normalized_path_xy = _normalize_path_xy(path_xy)
    backtrack_reference_node_text = (
        ""
        if backtrack_reference_node_id is None
        else str(backtrack_reference_node_id).strip()
    )
    normalized_backtrack_contexts = [
        deepcopy(item)
        for item in list(backtrack_contexts or [])
        if isinstance(item, dict)
    ]
    edge: Edge | None = None
    if (
        rgb_history != []
        or normalized_path_xy != []
        or backtrack_reference_node_text != ""
        or normalized_backtrack_contexts != []
    ):
        edge = _attach_move_data_to_edge(
            state=state,
            from_node_id=from_node_text,
            to_node_id=to_node_text,
            rgb_history_obs_ids=rgb_history,
            path_xy=normalized_path_xy,
        )
    if edge is not None and (
        backtrack_reference_node_text != "" or normalized_backtrack_contexts != []
    ):
        attach_backtrack_edge_knowledge(
            state=state,
            edge=edge,
            update_node_id=to_node_text,
            backtrack_contexts=normalized_backtrack_contexts,
            fallback_trigger_node_id=from_node_text,
            fallback_reference_node_id=backtrack_reference_node_text,
            from_node_id=from_node_text,
            to_node_id=to_node_text,
        )
    record = {
        "step_id": int(step_id),
        "from_node": from_node_text,
        "to_node": to_node_text,
        "move_mode": str(move_mode),
        "reason": str(reason),
        "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history],
    }
    if backtrack_reference_node_text != "":
        record["backtrack_reference_node_id"] = backtrack_reference_node_text
    if normalized_backtrack_contexts != []:
        record["backtrack_contexts"] = normalized_backtrack_contexts
    history = list(getattr(state, "node_move_history", []))
    history.append(record)
    state.node_move_history = history
    return record


def _backtrack_knowledge_contents(
    *,
    backtrack_contexts: list[dict[str, object]],
    fallback_trigger_node_id: str,
    fallback_reference_node_id: str,
    from_node_id: str,
    to_node_id: str,
) -> list[str]:
    if backtrack_contexts == []:
        if str(fallback_reference_node_id).strip() == "":
            return []
        return [
            f"Backtrack was triggered at {fallback_trigger_node_id}, replanned at "
            f"reference node {fallback_reference_node_id}, and executed from "
            f"{from_node_id} to {to_node_id}."
        ]
    contents: list[str] = []
    for item in backtrack_contexts:
        trigger_node_id = str(
            item.get("trigger_planning_node_id", fallback_trigger_node_id)
        ).strip()
        anchor_node_id = str(
            item.get("anchor_node_id", fallback_reference_node_id)
        ).strip()
        reason = str(item.get("reason", "")).strip().rstrip(".;")
        reason_text = "" if reason == "" else f" because {reason}"
        contents.append(
            f"Backtrack was triggered at {trigger_node_id}{reason_text}; replanning "
            f"used node {anchor_node_id}, and the resulting action moved from "
            f"{from_node_id} to {to_node_id}."
        )
    return contents


def attach_backtrack_edge_knowledge(
    *,
    state: "NavClawAgentState",
    edge: Edge,
    update_node_id: str,
    backtrack_contexts: list[dict[str, object]],
    fallback_trigger_node_id: str,
    fallback_reference_node_id: str,
    from_node_id: str,
    to_node_id: str,
) -> None:
    contents = _backtrack_knowledge_contents(
        backtrack_contexts=backtrack_contexts,
        fallback_trigger_node_id=fallback_trigger_node_id,
        fallback_reference_node_id=fallback_reference_node_id,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
    )
    with state.graph.lock:
        existing_contents = {
            str(item.content).casefold() for item in edge.knowledge
        }
        for content in contents:
            if content.casefold() in existing_contents:
                continue
            edge.knowledge.append(
                EntityKnowledge(
                    knowledge_id=next_entity_knowledge_id(edge.knowledge),
                    update_node_id=str(update_node_id),
                    content=content,
                )
            )
            existing_contents.add(content.casefold())


def recent_node_moves(
    *,
    state: "NavClawAgentState",
    limit: int,
) -> list[dict[str, object]]:
    history = list(getattr(state, "node_move_history", []))
    if int(limit) <= 0:
        return []
    return [deepcopy(item) for item in history[-int(limit):] if isinstance(item, dict)]


def _attach_move_data_to_edge(
    *,
    state: "NavClawAgentState",
    from_node_id: str,
    to_node_id: str,
    rgb_history_obs_ids: list[str],
    path_xy: list[tuple[float, float]],
) -> Edge:
    with state.graph.lock:
        for edge in state.graph.iter_edges(include_vertical=False):
            if str(edge.relation) != "move":
                continue
            if str(edge.src_id) == str(from_node_id) and str(edge.dst_id) == str(to_node_id):
                if edge.rgb_history_obs_ids == []:
                    edge.rgb_history_obs_ids = [str(obs_id) for obs_id in rgb_history_obs_ids]
                if edge.path_xy == []:
                    edge.path_xy = list(path_xy)
                return edge
        return state.graph.add_edge(
            src_id=str(from_node_id),
            dst_id=str(to_node_id),
            relation="move",
            rgb_history_obs_ids=[str(obs_id) for obs_id in rgb_history_obs_ids],
            path_xy=list(path_xy),
        )


def _normalize_path_xy(
    path_xy: object,
) -> list[tuple[float, float]]:
    if path_xy is None:
        return []
    if not isinstance(path_xy, (list, tuple)):
        raise ValueError(f"node move path_xy must be list/tuple when present: {path_xy!r}")
    points: list[tuple[float, float]] = []
    for point in list(path_xy):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"node move path_xy point must be length-2 list/tuple: {point!r}")
        points.append((float(point[0]), float(point[1])))
    return points
