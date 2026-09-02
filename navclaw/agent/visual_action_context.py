from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING

from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState, NavClawStepState


PANORAMA_DIRECTION_ORDER = ("front", "back", "left", "right")
_DIRECTION_BY_ANGLE = {
    0: "front",
    180: "back",
    90: "left",
    270: "right",
}
_ANGLE_BY_DIRECTION = {
    direction: angle for angle, direction in _DIRECTION_BY_ANGLE.items()
}
VISUAL_ACTION_ANGLES = [
    _ANGLE_BY_DIRECTION[direction] for direction in PANORAMA_DIRECTION_ORDER
]


def visual_action_angles(views: list["VisualViewContext"]) -> list[int]:
    return [int(view.angle_deg) for view in views]


def direction_for_angle(angle: int) -> str:
    normalized = int(angle) % 360
    try:
        return _DIRECTION_BY_ANGLE[normalized]
    except KeyError as exc:
        raise ValueError(f"panorama angle has no cardinal direction: {angle}") from exc


def angle_for_direction(direction: object) -> int:
    normalized = str(direction).strip().lower()
    try:
        return int(_ANGLE_BY_DIRECTION[normalized])
    except KeyError as exc:
        raise ValueError(f"invalid panorama direction: {direction!r}") from exc


def ordered_panorama_angles(angles: list[int]) -> list[int]:
    angles_by_direction = {
        direction_for_angle(int(angle)): int(angle) % 360 for angle in angles
    }
    return [
        angles_by_direction[direction]
        for direction in PANORAMA_DIRECTION_ORDER
        if direction in angles_by_direction
    ]


def ordered_visual_views(views: list["VisualViewContext"]) -> list["VisualViewContext"]:
    views_by_angle = {int(view.angle_deg): view for view in views}
    return [
        views_by_angle[int(angle)]
        for angle in ordered_panorama_angles(list(views_by_angle))
    ]


def allowed_directions(angles: list[int]) -> list[str]:
    available = {direction_for_angle(angle) for angle in angles}
    return [
        direction for direction in PANORAMA_DIRECTION_ORDER if direction in available
    ]


def allowed_directions_text(angles: list[int]) -> str:
    return "[" + ", ".join(allowed_directions(angles)) + "]"


@dataclass(frozen=True)
class VisualViewContext:
    angle_deg: int
    obs_id: str
    rgb_id: str
    depth_id: str
    pose: dict[str, object] = field(default_factory=dict)
    node_overlay_image_id: str = ""
    visible_visited_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "angle_deg": int(self.angle_deg),
            "obs_id": str(self.obs_id),
            "rgb_id": str(self.rgb_id),
            "depth_id": str(self.depth_id),
            "pose": deepcopy(self.pose),
            "node_overlay_image_id": str(self.node_overlay_image_id),
            "visible_visited_nodes": [str(node_id) for node_id in self.visible_visited_nodes],
        }


@dataclass(frozen=True)
class VisualEdgeHistoryContext:
    step_id: int
    src_node_id: str
    dst_node_id: str
    rgb_history_obs_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": int(self.step_id),
            "src_node_id": str(self.src_node_id),
            "dst_node_id": str(self.dst_node_id),
            "rgb_history_obs_ids": [str(obs_id) for obs_id in self.rgb_history_obs_ids],
        }


@dataclass(frozen=True)
class VisualActionContext:
    current_node_id: str
    views: list[VisualViewContext]
    recent_edge_rgb_history_obs_ids: list[str] = field(default_factory=list)
    edge_rgb_history_blocks: list[VisualEdgeHistoryContext] = field(default_factory=list)
    context_evidence_text: str = ""
    previous_arrival_node_id: str = ""
    previous_arrival_angle_deg: int | None = None
    graph_context_visible: bool = True

    @property
    def available_angles(self) -> list[int]:
        return visual_action_angles(self.views)

    def view_for_angle(self, angle_deg: int) -> VisualViewContext:
        for view in self.views:
            if int(view.angle_deg) == int(angle_deg):
                return view
        raise ValueError(f"missing visual view for angle: {angle_deg}")

    def to_dict(self) -> dict[str, object]:
        return {
            "current_node_id": str(self.current_node_id),
            "views": [view.to_dict() for view in self.views],
            "recent_edge_rgb_history_obs_ids": [
                str(obs_id) for obs_id in self.recent_edge_rgb_history_obs_ids
            ],
            "edge_rgb_history_blocks": [
                block.to_dict() for block in self.edge_rgb_history_blocks
            ],
            "context_evidence_text": str(self.context_evidence_text),
            "previous_arrival_node_id": str(self.previous_arrival_node_id),
            "previous_arrival_angle_deg": None
            if self.previous_arrival_angle_deg is None
            else int(self.previous_arrival_angle_deg),
            "graph_context_visible": bool(self.graph_context_visible),
        }


def build_visual_action_context(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
    context_evidence_text: str,
) -> VisualActionContext:
    include_graph_context = True
    current_node_id = str(step.current_place_node_id or state.current_place_node_id or "")
    if current_node_id == "":
        raise ValueError("visual action context requires current_place_node_id")
    views = _views_from_step(step)
    if include_graph_context:
        views = _attach_place_node_overlays(state=state, views=views)
    return VisualActionContext(
        current_node_id=current_node_id,
        views=views,
        recent_edge_rgb_history_obs_ids=_recent_edge_rgb_history_obs_ids(
            state=state,
            current_node_id=current_node_id,
        ),
        edge_rgb_history_blocks=_edge_rgb_history_blocks(
            state=state,
            include_node_ids=include_graph_context,
        ),
        context_evidence_text=(str(context_evidence_text) if include_graph_context else ""),
        previous_arrival_node_id=(
            _previous_arrival_node_id(state=state, current_node_id=current_node_id)
            if include_graph_context
            else ""
        ),
        previous_arrival_angle_deg=(
            _previous_arrival_angle_deg(
                state=state,
                current_node_id=current_node_id,
                available_angles=visual_action_angles(views),
            )
            if include_graph_context
            else None
        ),
        graph_context_visible=include_graph_context,
    )


def build_visual_action_context_for_node(
    *,
    state: "NavClawAgentState",
    node_id: str,
    context_evidence_text: str,
) -> VisualActionContext:
    include_graph_context = True
    node_id_text = str(node_id).strip()
    if node_id_text == "":
        raise ValueError("visual action context requires node_id")
    node = state.graph.get_node(node_id_text)
    obs_ids = [str(obs_id) for obs_id in list(node.obs_ids)]
    if obs_ids == []:
        raise ValueError(f"node {node_id_text!r} has no stored panorama obs_ids")
    angle_to_obs_id = _angle_to_obs_id_from_ordered_obs_ids(obs_ids)
    views = _views_from_angle_to_obs_id(state=state, angle_to_obs_id=angle_to_obs_id)
    if include_graph_context:
        views = _attach_place_node_overlays(state=state, views=views)
    return VisualActionContext(
        current_node_id=node_id_text,
        views=views,
        recent_edge_rgb_history_obs_ids=_recent_edge_rgb_history_obs_ids(
            state=state,
            current_node_id=node_id_text,
        ),
        edge_rgb_history_blocks=_edge_rgb_history_blocks(
            state=state,
            include_node_ids=include_graph_context,
        ),
        context_evidence_text=(str(context_evidence_text) if include_graph_context else ""),
        previous_arrival_node_id=(
            _previous_arrival_node_id(state=state, current_node_id=node_id_text)
            if include_graph_context
            else ""
        ),
        previous_arrival_angle_deg=(
            _previous_arrival_angle_deg(
                state=state,
                current_node_id=node_id_text,
                available_angles=visual_action_angles(views),
            )
            if include_graph_context
            else None
        ),
        graph_context_visible=include_graph_context,
    )


def _views_from_step(step: "NavClawStepState") -> list[VisualViewContext]:
    angle_to_obs_id = {
        int(angle): str(obs_id)
        for angle, obs_id in dict(getattr(step, "angle_to_obs_id", {})).items()
    }
    if angle_to_obs_id == {} and list(getattr(step, "panorama_obs_ids", [])) != []:
        angle_to_obs_id = _angle_to_obs_id_from_ordered_obs_ids(
            [str(obs_id) for obs_id in list(step.panorama_obs_ids)]
        )
    if angle_to_obs_id == {}:
        raise ValueError("visual action context missing panorama observations")
    return _views_from_angle_to_obs_id_from_step(step=step, angle_to_obs_id=angle_to_obs_id)


def _views_from_angle_to_obs_id_from_step(
    *,
    step: "NavClawStepState",
    angle_to_obs_id: dict[int, str],
) -> list[VisualViewContext]:
    pose_by_obs_id = {}
    for raw_view in list(getattr(step, "panorama_views", [])):
        if not isinstance(raw_view, dict):
            continue
        obs_id = str(raw_view.get("obs_id", ""))
        if obs_id != "":
            pose_by_obs_id[obs_id] = deepcopy(raw_view.get("pose", {}))
    return [
        VisualViewContext(
            angle_deg=int(angle),
            obs_id=str(angle_to_obs_id[int(angle)]),
            rgb_id=f"{angle_to_obs_id[int(angle)]}:rgb",
            depth_id=f"{angle_to_obs_id[int(angle)]}:depth",
            pose=dict(pose_by_obs_id.get(str(angle_to_obs_id[int(angle)]), {})),
        )
        for angle in sorted(angle_to_obs_id)
    ]


def _views_from_angle_to_obs_id(
    *,
    state: "NavClawAgentState",
    angle_to_obs_id: dict[int, str],
) -> list[VisualViewContext]:
    views: list[VisualViewContext] = []
    for angle in sorted(angle_to_obs_id):
        obs_id = str(angle_to_obs_id[int(angle)])
        observation = state.cache.get_observation(obs_id).observation
        views.append(
            VisualViewContext(
                angle_deg=int(angle),
                obs_id=obs_id,
                rgb_id=f"{obs_id}:rgb",
                depth_id=f"{obs_id}:depth",
                pose=observation.pose.to_dict(),
            )
        )
    return views


def _angle_to_obs_id_from_ordered_obs_ids(obs_ids: list[str]) -> dict[int, str]:
    normalized_obs_ids = [str(obs_id) for obs_id in obs_ids]
    count = len(normalized_obs_ids)
    if count <= 0:
        return {}
    return {
        int(round(float(index) * 360.0 / float(count))) % 360: str(obs_id)
        for index, obs_id in enumerate(normalized_obs_ids)
    }


def _attach_place_node_overlays(
    *,
    state: "NavClawAgentState",
    views: list[VisualViewContext],
) -> list[VisualViewContext]:
    floor_id = str(state.system.current_floor_id)
    exploration = state.global_exploration_for_floor(floor_id)
    overlay_views = exploration.render_place_node_annotated_views(
        cache=state.cache,
        graph=state.graph,
        obs_ids=[str(view.obs_id) for view in views],
        floor_id=floor_id,
        image_max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
    )
    overlays_by_obs_id = {
        str(overlay.obs_id): overlay
        for overlay in overlay_views
    }
    annotated_views: list[VisualViewContext] = []
    for view in views:
        overlay = overlays_by_obs_id.get(str(view.obs_id))
        if overlay is None:
            annotated_views.append(view)
            continue
        annotated_views.append(
            replace(
                view,
                node_overlay_image_id=str(overlay.overlay_id),
                visible_visited_nodes=[str(node_id) for node_id in overlay.node_ids],
            )
        )
    return annotated_views


def _recent_edge_rgb_history_obs_ids(
    *,
    state: "NavClawAgentState",
    current_node_id: str,
) -> list[str]:
    for edge in reversed(state.graph.iter_edges(include_vertical=True)):
        if str(edge.relation) not in {"move", "stairs_up", "stairs_down"}:
            continue
        if str(edge.dst_id) != str(current_node_id):
            continue
        return [str(obs_id) for obs_id in list(edge.rgb_history_obs_ids)]
    return []


def _edge_rgb_history_blocks(
    *,
    state: "NavClawAgentState",
    include_node_ids: bool = True,
) -> list[VisualEdgeHistoryContext]:
    history = list(getattr(state, "node_move_history", []))
    blocks: list[VisualEdgeHistoryContext] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        obs_ids = [str(obs_id) for obs_id in list(item.get("rgb_history_obs_ids", []))]
        if obs_ids == []:
            continue
        blocks.append(
            VisualEdgeHistoryContext(
                step_id=int(item.get("step_id", 0)),
                src_node_id=(str(item.get("from_node", "")) if include_node_ids else ""),
                dst_node_id=(str(item.get("to_node", "")) if include_node_ids else ""),
                rgb_history_obs_ids=obs_ids,
            )
        )
    return blocks


def _previous_arrival_node_id(
    *,
    state: "NavClawAgentState",
    current_node_id: str,
) -> str:
    for item in reversed(list(getattr(state, "node_move_history", []))):
        if not isinstance(item, dict):
            continue
        if str(item.get("to_node", "")).strip() != str(current_node_id):
            continue
        previous_node_id = str(item.get("from_node", "")).strip()
        if previous_node_id != "":
            return previous_node_id
    previous_node_id = str(getattr(state, "previous_place_node_id", "") or "").strip()
    if previous_node_id == str(current_node_id):
        return ""
    return previous_node_id


def _previous_arrival_angle_deg(
    *,
    state: "NavClawAgentState",
    current_node_id: str,
    available_angles: list[int],
) -> int | None:
    if available_angles == []:
        return None
    previous_node_id = _previous_arrival_node_id(state=state, current_node_id=current_node_id)
    if previous_node_id == "":
        return None
    try:
        current_node = state.graph.get_node(str(current_node_id))
        previous_node = state.graph.get_node(str(previous_node_id))
    except KeyError:
        return None
    dx = float(previous_node.position[0]) - float(current_node.position[0])
    dy = float(previous_node.position[1]) - float(current_node.position[1])
    if math.hypot(dx, dy) < 1e-4:
        return None
    bearing_deg = math.degrees(math.atan2(dy, dx))
    relative_deg = (bearing_deg - float(current_node.yaw)) % 360.0
    return min(
        [int(angle) for angle in available_angles],
        key=lambda angle: min(
            abs(float(angle) - relative_deg),
            360.0 - abs(float(angle) - relative_deg),
        ),
    )
