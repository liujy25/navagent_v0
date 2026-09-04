from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.entity_knowledge import manage_retrieved_knowledge
from navclaw.agent.episodic_retrieval import RetrievalWorkspace
from navclaw.agent.episodic_retrieval import RetrieveRequest
from navclaw.agent.episodic_retrieval import build_memory_index
from navclaw.agent.episodic_retrieval import execute_retrieve_request
from navclaw.agent.episodic_retrieval import memory_index_text
from navclaw.agent.node_summary import NodeSummaryDecision, summarize_current_node
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext, build_visual_action_context
from navclaw.agent.visual_action_context import build_visual_action_context_for_node
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_grounding import VisualWaypoint
from navclaw.agent.visual_policy import decide_stop_confirmation
from navclaw.agent.visual_policy import decide_vln_navigation_step
from navclaw.agent.visual_policy import decide_vln_task_progress_step
from navclaw.agent.visual_policy import ensure_task_progress_memory
from navclaw.agent.visual_policy_decisions import LocalMovePlanDecision
from navclaw.agent.visual_policy_decisions import NavigationModeDecision
from navclaw.agent.visual_policy_decisions import TaskProgressDecision
from navclaw.agent.visual_policy_decisions import VisualActionPointDecision
from navclaw.agent.visual_policy_decisions import VisualWaypointVerificationDecision
from navclaw.agent.vln_landmark_context import build_vln_landmark_context
from navclaw.agent.vln_landmark_context import draw_vln_landmark_panorama_views
from navclaw.agent.vln_waypoint_policy import plan_vln_waypoint_loop
from navclaw.agent.vln_waypoint_sampling import VLN_STOP_WAYPOINT_MAX_DISTANCE_M
from navclaw.agent.vln_waypoint_sampling import VLN_STOP_WAYPOINT_SAMPLE_SPACING_M
from navclaw.agent.vln_waypoint_sampling import VLN_WAYPOINT_GLOBAL_NODE_DEDUP_RADIUS_M
from navclaw.agent.waypoint import GroundedWaypointTarget
from navclaw.agent.waypoint import WaypointPlanningContext
from navclaw.agent.waypoint import WaypointPolicyOutput
from navclaw.agent.waypoint import WaypointPolicyResult
from navclaw.agent.waypoint.types import FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY
from navclaw.graph.node import add_node_observation_knowledge
from navclaw.graph.node import has_node_observation_knowledge
from navclaw.memory.task_progress import TaskProgressUpdateResult
from navclaw.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION
from navclaw.schemas import ActionCall
from navclaw.types import LocalmapNavigationPlan, LocalmapRouteTarget
from navclaw.visualization.action_mode_overlays import (
    BevOverlayTransform,
    _node_labels_by_id,
    render_task_progress_bev_overlay_result,
)
from navclaw.visualization.waypoint_overlay import draw_waypoint_overlay_rgb

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState, NavClawStepState
    from navclaw.mapping.exploration.manager import ExplorationManager
    from navclaw.perception.goal import GoalSpec


VISUAL_WAYPOINT_VERIFICATION_MAX_ATTEMPTS = 4
NAVIGATION_REPLAN_MAX_ATTEMPTS = 2
NEED_MORE_EVIDENCE_WARNING_CONSECUTIVE_COUNT = 3


@dataclass(frozen=True)
class VisualWaypointAttempt:
    attempt_index: int
    selected_angle_deg: int
    selected_obs_id: str
    waypoint_target: str
    visual_action: VisualActionPointDecision | None
    verification: VisualWaypointVerificationDecision | None
    failure_reason: str = ""
    sampled_candidate: dict[str, object] = field(default_factory=dict)
    candidate_selection: dict[str, object] = field(default_factory=dict)
    candidate_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_index": int(self.attempt_index),
            "selected_angle_deg": int(self.selected_angle_deg),
            "selected_obs_id": str(self.selected_obs_id),
            "waypoint_target": str(self.waypoint_target),
            "visual_action": None if self.visual_action is None else self.visual_action.to_dict(),
            "verification": None if self.verification is None else self.verification.to_dict(),
            "failure_reason": str(self.failure_reason),
            "sampled_candidate": dict(self.sampled_candidate),
            "candidate_selection": dict(self.candidate_selection),
            "candidate_count": None if self.candidate_count is None else int(self.candidate_count),
        }


@dataclass(frozen=True)
class VisualNavigationDecision:
    visual_context: VisualActionContext
    task_progress_initialization: dict[str, object]
    navigation_mode: NavigationModeDecision | None
    task_progress_update_result: TaskProgressUpdateResult
    visual_action: VisualActionPointDecision | None
    waypoint: VisualWaypoint | None
    action_call: ActionCall | None
    local_move_plan: LocalMovePlanDecision | None = None
    route_target: LocalmapRouteTarget | None = None
    navigation_plan: LocalmapNavigationPlan | None = None
    waypoint_attempts: list[VisualWaypointAttempt] = field(default_factory=list)
    navigation_replan_feedback: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""
    context_evidence_text: str = ""
    waypoint_policy_result: WaypointPolicyResult | None = None
    grounded_waypoint_target: GroundedWaypointTarget | None = None
    terminal_check: dict[str, object] = field(default_factory=dict)
    physical_current_node_id: str = ""
    planning_current_node_id: str = ""
    backtrack_contexts: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "visual_context": self.visual_context.to_dict(),
            "task_progress_initialization": deepcopy(self.task_progress_initialization),
            "navigation_mode": (
                None if self.navigation_mode is None else self.navigation_mode.to_dict()
            ),
            "task_progress_update_result": self.task_progress_update_result.to_dict(),
            "visual_action": None if self.visual_action is None else self.visual_action.to_dict(),
            "waypoint": None if self.waypoint is None else self.waypoint.to_dict(),
            "action_call": None if self.action_call is None else self.action_call.to_dict(),
            "local_move_plan": None if self.local_move_plan is None else self.local_move_plan.to_dict(),
            "route_target": None if self.route_target is None else self.route_target.to_dict(),
            "navigation_plan": None if self.navigation_plan is None else self.navigation_plan.to_dict(),
            "waypoint_attempts": [attempt.to_dict() for attempt in self.waypoint_attempts],
            "navigation_replan_feedback": [dict(item) for item in self.navigation_replan_feedback],
            "failure_reason": str(self.failure_reason),
            "context_evidence_text": str(self.context_evidence_text),
            "waypoint_policy_result": (
                None if self.waypoint_policy_result is None else self.waypoint_policy_result.to_dict()
            ),
            "grounded_waypoint_target": (
                None if self.grounded_waypoint_target is None else self.grounded_waypoint_target.to_dict()
            ),
            "terminal_check": deepcopy(self.terminal_check),
            "physical_current_node_id": str(self.physical_current_node_id),
            "planning_current_node_id": str(self.planning_current_node_id),
            "backtrack_contexts": [
                deepcopy(item) for item in self.backtrack_contexts
            ],
        }


@dataclass(frozen=True)
class EpisodicRetrievalLoopResult:
    task_progress_initialization: dict[str, object]
    task_progress_decision: TaskProgressDecision
    task_progress_update_result: TaskProgressUpdateResult
    navigation_mode: NavigationModeDecision | None
    workspace: RetrievalWorkspace
    progress_update_count: int
    planning_visual_context: VisualActionContext | None = None
    backtrack_contexts: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""


@dataclass(frozen=True)
class _VlnWaypointCandidateBevBase:
    image: np.ndarray
    transform: BevOverlayTransform
    landmark_markers: list[dict[str, object]]
    reference_node_marker: dict[str, object] | None
    context_text: str


def summarize_current_node_for_visual_policy(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> dict[str, object]:
    visual_context = build_visual_action_context(
        state=state,
        step=step,
        context_evidence_text="",
    )
    decision = summarize_current_node(
        client=state.llm_client,
        cache=state.cache,
        visual_context=visual_context,
    )
    node_id = str(visual_context.current_node_id)
    node = state.graph.get_node(node_id)
    added_knowledge = add_node_observation_knowledge(
        node,
        node_summary=decision.node_summary,
        direction_summaries=decision.direction_summaries,
    )
    payload = _node_summary_payload(
        node_id=node_id,
        visual_context=visual_context,
        decision=decision,
    )
    payload["knowledge"] = [item.to_dict() for item in added_knowledge]
    step.node_summary = payload
    return payload


def ensure_current_node_summary_for_visual_policy(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> dict[str, object]:
    node_id = str(step.current_place_node_id or state.current_place_node_id or "").strip()
    if node_id == "":
        raise ValueError("node summary requires a current place node")
    node = state.graph.get_node(node_id)
    if has_node_observation_knowledge(node):
        payload = {
            "node_id": node_id,
            "source": "existing",
            "knowledge": [item.to_dict() for item in node.knowledge],
        }
        step.node_summary = payload
        return payload
    return summarize_current_node_for_visual_policy(
        state=state,
        step=step,
    )


def recent_outcome_text(state: "NavClawAgentState") -> str:
    action = getattr(state, "last_executed_action", None)
    if not isinstance(action, dict):
        return "none"
    summary = {
        "type": action.get("type"),
        "route_completed": action.get("route_completed"),
        "execution": action.get("execution"),
    }
    if isinstance(action.get("move_result"), dict):
        result = dict(action["move_result"])
        summary["move_ok"] = result.get("ok")
        summary["move_message"] = result.get("message")
    return json.dumps(summary, ensure_ascii=False)


def _node_summary_payload(
    *,
    node_id: str,
    visual_context: VisualActionContext,
    decision: NodeSummaryDecision,
) -> dict[str, object]:
    return {
        "node_id": str(node_id),
        "panorama_obs_ids": {
            str(view.angle_deg): str(view.obs_id)
            for view in visual_context.views
        },
        "edge_rgb_history_obs_ids": [
            str(obs_id) for obs_id in visual_context.recent_edge_rgb_history_obs_ids
        ],
        **decision.to_dict(),
    }




def _goal_kind(goal: "GoalSpec | None") -> str:
    if goal is None:
        return ""
    return str(getattr(goal, "goal_kind", "") or "").strip()




def _task_progress_overlay_robot_xy(visual_context: VisualActionContext) -> tuple[float, float] | None:
    try:
        pose = dict(visual_context.view_for_angle(0).pose)
    except (KeyError, ValueError):
        return None
    if pose.get("x") is None or pose.get("y") is None:
        return None
    return (float(pose["x"]), float(pose["y"]))


def _task_progress_overlay_robot_yaw_deg(visual_context: VisualActionContext) -> float | None:
    try:
        pose = dict(visual_context.view_for_angle(0).pose)
    except (KeyError, ValueError):
        return None
    if pose.get("yaw") is None:
        return None
    return float(pose["yaw"])


def _progress_updates_for_goal(
    raw_updates: list[dict[str, object]],
    *,
    goal: "GoalSpec | None",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    filtered_updates: list[dict[str, object]] = []
    skipped_updates: list[dict[str, object]] = []
    for update in raw_updates:
        if str(update.get("kind", "")).strip().lower() == "verify_candidate":
            skipped_updates.append(
                {
                    "reason": "verify_candidate_removed_from_task_progress",
                    "update": deepcopy(update),
                }
            )
            continue
        sanitized_update = deepcopy(update)
        sanitized_update.pop("candidate", None)
        filtered_updates.append(sanitized_update)
    return filtered_updates, skipped_updates




def _navigation_replan_feedback_text(feedback: list[dict[str, object]]) -> str:
    if feedback == []:
        return ""
    lines: list[str] = []
    for index, item in enumerate(feedback):
        fields = [f"action_mode={item.get('action_mode', '')}"]
        for output_name, input_name in (
            ("reference_node_id", "reference_node_id"),
            ("stop_objective", "stop_objective"),
            ("waypoint_target", "waypoint_target"),
            ("validation_result", "verifier_verdict"),
        ):
            value = item.get(input_name)
            if value is not None and str(value).strip() != "":
                fields.append(f"{output_name}={value}")
        selected_direction = str(item.get("selected_direction", "")).strip()
        selected_angle = item.get("selected_angle_deg")
        if selected_direction != "":
            fields.append(f"selected_direction={selected_direction}")
        elif selected_angle is not None:
            try:
                fields.append(
                    f"selected_direction={direction_for_angle(int(selected_angle))}"
                )
            except (TypeError, ValueError):
                pass
        if "target_not_visible" in item:
            fields.append(
                f"target_not_visible={bool(item.get('target_not_visible', False))}"
            )
        frontier_directions = str(
            item.get("available_frontier_directions", "")
        ).strip()
        if frontier_directions != "":
            fields.append(
                f"available_frontier_directions={frontier_directions}"
            )
        failure_summary = str(item.get("failure_summary", "")).strip()
        if failure_summary != "":
            fields.append(f"failure_summary={failure_summary}")
        lines.append(f"Attempt {index}: " + "; ".join(fields))
    return "\n".join(lines)


def _fallback_navigation_feedback(
    *,
    navigation_mode: NavigationModeDecision,
    local_move_plan: LocalMovePlanDecision,
    selected_view: VisualViewContext,
    waypoint_target: str,
    verification: VisualWaypointVerificationDecision,
) -> dict[str, object]:
    return {
        "action_mode": str(navigation_mode.action_mode),
        "stop_objective": str(navigation_mode.stop_objective),
        "selected_direction": direction_for_angle(selected_view.angle_deg),
        "selected_obs_id": str(selected_view.obs_id),
        "waypoint_target": str(waypoint_target),
        "local_move_reasoning": str(local_move_plan.reasoning),
        "verifier_verdict": str(verification.verdict),
        "target_not_visible": bool(verification.target_not_visible),
        "failure_summary": str(verification.critique),
    }


def _with_replan_log(
    decision: VisualNavigationDecision,
    *,
    waypoint_attempts: list[VisualWaypointAttempt],
    navigation_replan_feedback: list[dict[str, object]],
) -> VisualNavigationDecision:
    if waypoint_attempts == [] and navigation_replan_feedback == []:
        return decision
    return replace(
        decision,
        waypoint_attempts=[*waypoint_attempts, *decision.waypoint_attempts],
        navigation_replan_feedback=[
            *[dict(item) for item in navigation_replan_feedback],
            *[dict(item) for item in decision.navigation_replan_feedback],
        ],
    )






def _need_more_evidence_warning_text(state: "NavClawAgentState") -> str:
    count = int(getattr(state, "consecutive_need_more_evidence_count", 0) or 0)
    warning_start_count = int(NEED_MORE_EVIDENCE_WARNING_CONSECUTIVE_COUNT) - 1
    if count < warning_start_count:
        return ""
    return (
        f"The last {count} task progress updates returned route_status=need_more_evidence. "
        "Reassess whether the agent is still near the relevant decision point. "
        "Use need_more_evidence again only if one more local move or observation is likely to resolve the ambiguity. "
        "Use recovery_needed if the current branch is weak, exhausted, or no longer useful for the current active progress item; "
        "use continue_current if the current branch still has positive route evidence."
    )


def _record_route_status_for_warning(
    *,
    state: "NavClawAgentState",
    route_status: str,
) -> None:
    if str(route_status) == "need_more_evidence":
        state.consecutive_need_more_evidence_count = (
            int(getattr(state, "consecutive_need_more_evidence_count", 0) or 0) + 1
        )
        return
    state.consecutive_need_more_evidence_count = 0


def _vln_initial_orientation_note(*, goal_kind: str, step_index: int) -> str:
    if str(goal_kind) != GOAL_KIND_VLN_INSTRUCTION or int(step_index) != 0:
        return ""
    return "Step 0 note: no movement has been executed yet; do not assume any route progress has already happened."


def _vln_initial_progress_note(*, goal_kind: str, step_index: int) -> str:
    if str(goal_kind) != GOAL_KIND_VLN_INSTRUCTION or int(step_index) != 0:
        return ""
    return (
        "Step 0 note: no movement has been executed yet. Use the current observation "
        "to establish the starting state, and do not mark instruction actions such as "
        "exit, turn, walk, pass, or stop as completed."
    )


















def _uses_system_active_waypoint_grounding(
    *,
    goal_kind: str,
) -> bool:
    return str(goal_kind) == GOAL_KIND_VLN_INSTRUCTION


def _vln_backtrack_node_ids(
    *,
    state: "NavClawAgentState",
    current_node_id: str,
) -> set[str]:
    return {
        str(node.id)
        for node in state.graph.iter_nodes()
        if str(node.id) != str(current_node_id)
        and str(node.floor_id) == str(state.system.current_floor_id)
        and str(node.node_kind) == "place"
        and list(node.obs_ids) != []
    }














def _vln_backtrack_context(
    *,
    decision: NavigationModeDecision,
    planning_node_id: str,
    physical_node_id: str,
) -> dict[str, object]:
    return {
        "trigger_planning_node_id": str(planning_node_id),
        "anchor_node_id": str(decision.backtrack_anchor_node_id),
        "physical_robot_node_id": str(physical_node_id),
        "objective": str(decision.backtrack_objective or decision.action_objective),
        "reason": str(decision.backtrack_reason or decision.action_reason),
    }


def _vln_backtrack_context_text(
    backtrack_contexts: list[dict[str, object]],
) -> str:
    if backtrack_contexts == []:
        return ""
    lines = ["Backtrack context:"]
    for index, item in enumerate(backtrack_contexts):
        lines.extend(
            [
                f"Backtrack {int(index) + 1}:",
                "- Trigger planning reference node: "
                + str(item.get("trigger_planning_node_id", "")),
                "- Planning reference node: " + str(item.get("anchor_node_id", "")),
                "- Physical robot node: "
                + str(item.get("physical_robot_node_id", "")),
                "- Objective: " + str(item.get("objective", "")),
                "- Reason: " + str(item.get("reason", "")),
            ]
        )
        result = item.get("result")
        if isinstance(result, dict):
            lines.extend(
                [
                    "- result: " + str(result.get("status", "")),
                    "- resulting navigation action: "
                    + str(result.get("navigation_action_mode", "")),
                    "- result detail: " + str(result.get("detail", "")),
                ]
            )
    return "\n".join(lines)


def _vln_waypoint_inherited_agent_context(
    *,
    context_loop: EpisodicRetrievalLoopResult | None,
    task_progress: TaskProgressDecision,
    navigation_mode: NavigationModeDecision,
    active_progress_item: str = "",
    task_progress_text: str = "",
) -> list[dict[str, object]]:
    active_item = str(active_progress_item).strip()
    progress = context_loop.task_progress_decision if context_loop is not None else task_progress
    navigation_action: dict[str, object] = {
        "mode": str(navigation_mode.action_mode),
    }
    if str(navigation_mode.action_mode) == "approach_to_stop":
        navigation_action["movement"] = str(
            navigation_mode.approach_movement or "move"
        )
    direction = str(navigation_mode.direction).strip()
    if direction != "":
        navigation_action["direction"] = direction
    objective = str(navigation_mode.action_objective).strip()
    if objective != "":
        navigation_action["objective"] = objective
    navigation_reason = str(
        navigation_mode.action_reason or navigation_mode.reasoning_action
    ).strip()
    if navigation_reason != "":
        navigation_action["reason"] = navigation_reason
    if str(navigation_mode.action_mode) == "backtrack":
        navigation_action["anchor_node_id"] = str(
            navigation_mode.backtrack_anchor_node_id
        )
    elif str(navigation_mode.action_mode) == "vertical_transition":
        navigation_action["vertical_direction"] = str(
            navigation_mode.vertical_direction
        )
    progress_memory = str(task_progress_text).strip()
    progress_memory_section = (
        f"Current progress:\n{progress_memory}\n"
        if progress_memory != ""
        else ""
    )
    active_item_section = (
        "Active item and its read-only grounding conditions:\n"
        f"{active_item}\n"
        "An unconfirmed condition remains unresolved and is not an established fact.\n"
        if active_item != ""
        else ""
    )
    retrieval_section = ""
    if context_loop is not None and context_loop.workspace.rounds != []:
        retrieval_section = (
            "Retrieval conclusions:\n"
            + context_loop.workspace.retrieval_log_text()
            + "\n"
        )
    backtrack_section = ""
    context_backtrack_contexts = (
        []
        if context_loop is None
        else list(getattr(context_loop, "backtrack_contexts", []))
    )
    if context_backtrack_contexts != []:
        backtrack_section = (
            _vln_backtrack_context_text(context_backtrack_contexts) + "\n"
        )
    progress_updates_section = ""
    if progress.progress_updates:
        progress_updates_section = (
            "Progress updates:\n"
            + json.dumps(progress.progress_updates, ensure_ascii=False, indent=2)
            + "\n"
        )
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "High-level navigation action:\n"
                f"{json.dumps(navigation_action, ensure_ascii=False)}\n\n"
                "Navigation state:\n"
                f"{progress_memory_section}"
                f"{active_item_section}"
                f"{retrieval_section}"
                f"{backtrack_section}"
                f"{progress_updates_section}"
            ),
        }
    ]
    return content


def _current_active_task_progress_item(task_progress_memory) -> str:
    for item in list(task_progress_memory.items):
        if str(item.status) == "active" and str(item.kind) == "task":
            content = str(item.content).strip()
            unconfirmed = [
                str(condition.content).strip()
                for condition in item.conditions
                if str(condition.status) == "unconfirmed"
            ]
            if unconfirmed:
                return content + "\nUnconfirmed progress conditions:\n- " + "\n- ".join(unconfirmed)
            return content
    return ""


def _vln_waypoint_bev_landmark_markers(
    *,
    step_index: int,
    floor_id: str,
    context_loop: EpisodicRetrievalLoopResult | None,
    landmark_context,
) -> list[dict[str, object]]:
    if landmark_context is None:
        return []
    markers = [marker.to_dict() for marker in landmark_context.bev_markers]
    if int(step_index) == 0:
        return markers
    if context_loop is None:
        return []
    workspace = context_loop.workspace
    selected_ids = set(
        workspace.selected_landmark_ids_by_floor.get(str(floor_id), set())
    )
    labels_by_id = workspace.landmark_display_labels_by_floor.get(
        str(floor_id), {}
    )
    selected_labels = {
        int(labels_by_id[landmark_id])
        for landmark_id in selected_ids
        if landmark_id in labels_by_id
    }
    return [
        marker
        for marker in markers
        if int(marker.get("label", -1)) in selected_labels
    ]


def _vln_waypoint_candidate_bev_base(
    *,
    state: "NavClawAgentState",
    step_index: int,
    current_node_id: str,
    context_loop: EpisodicRetrievalLoopResult | None,
    landmark_context,
    include_graph_context: bool,
) -> _VlnWaypointCandidateBevBase:
    floor_id = str(state.system.current_floor_id)
    global_exploration = state.global_exploration_for_floor(floor_id)
    landmark_markers = _vln_waypoint_bev_landmark_markers(
        step_index=int(step_index),
        floor_id=floor_id,
        context_loop=context_loop,
        landmark_context=landmark_context,
    )
    floor_nodes = list(state.graph.iter_nodes(floor_id=floor_id))
    node_labels_by_id = _node_labels_by_id(floor_nodes)
    current_id = str(current_node_id)
    selected_node_ids: set[str] = set()
    selected_edge_ids: set[str] = set()
    base_image: np.ndarray | None = None
    base_transform: BevOverlayTransform | None = None
    if include_graph_context and context_loop is not None:
        workspace = context_loop.workspace
        base_image = workspace.shared_bev_by_floor.get(floor_id)
        base_transform = workspace.shared_bev_transform_by_floor.get(floor_id)
        selected_node_ids = set(
            workspace.selected_node_ids_by_floor.get(floor_id, set())
        )
        selected_edge_ids = set(
            workspace.selected_edge_ids_by_floor.get(floor_id, set())
        )
    if base_image is None or base_transform is None:
        selected_node_ids = {current_id} if include_graph_context else set()
        selected_edge_ids = set()
        render_result = render_task_progress_bev_overlay_result(
            graph=state.graph,
            global_exploration=global_exploration,
            floor_id=floor_id,
            landmark_markers=landmark_markers,
            show_graph=include_graph_context,
            node_ids=selected_node_ids,
            edge_ids=selected_edge_ids,
        )
        base_image = render_result.image
        base_transform = render_result.transform

    reference_node_marker = None
    if include_graph_context and current_id not in selected_node_ids:
        current_node = state.graph.get_node(current_id)
        reference_node_marker = {
            "label": int(node_labels_by_id[current_id]),
            "xy": [float(current_node.position[0]), float(current_node.position[1])],
        }
        selected_node_ids.add(current_id)

    context_lines: list[str] = []
    if include_graph_context:
        node_text = ", ".join(
            f"{int(node_labels_by_id[node_id])}={node_id}"
            for node_id in sorted(
                selected_node_ids,
                key=lambda node_id: int(node_labels_by_id[node_id]),
            )
            if node_id in node_labels_by_id
        )
        context_lines.append(
            f"Blue numbered circles are graph nodes: {node_text or 'none'}."
        )
        if current_id in node_labels_by_id:
            context_lines.append(
                "Current reference node: "
                f"{int(node_labels_by_id[current_id])}={current_id}."
            )
        edges_by_id = {
            str(edge.id): edge
            for edge in state.graph.iter_edges(
                floor_id=floor_id,
                include_vertical=False,
            )
        }
        edge_text = ", ".join(
            f"{edge_id} ({edges_by_id[edge_id].src_id} -> "
            f"{edges_by_id[edge_id].dst_id})"
            for edge_id in sorted(selected_edge_ids)
            if edge_id in edges_by_id
        )
        if edge_text != "":
            context_lines.append(
                "Highlighted retrieved edge trajectories: " + edge_text + "."
            )
    return _VlnWaypointCandidateBevBase(
        image=np.asarray(base_image, dtype=np.uint8),
        transform=base_transform,
        landmark_markers=landmark_markers,
        reference_node_marker=reference_node_marker,
        context_text="\n".join(context_lines),
    )


def run_episodic_retrieval_loop(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
    goal: "GoalSpec | None",
    goal_text: str,
    visual_context: VisualActionContext,
    vln_landmark_context,
) -> EpisodicRetrievalLoopResult:
    goal_kind = _goal_kind(goal)
    task_progress_memory = state.system.memory.task_progress
    physical_visual_context = visual_context
    planning_visual_context = visual_context
    planning_landmark_context = vln_landmark_context
    backtrack_contexts: list[dict[str, object]] = []
    pending_terminal_check = getattr(state, "pending_vln_terminal_check", None)
    terminal_check_required = isinstance(pending_terminal_check, dict)
    pending_initialization = getattr(
        state,
        "pending_vln_task_progress_initialization",
        None,
    )
    if pending_initialization is not None:
        task_progress_initialization = dict(pending_initialization)
        setattr(state, "pending_vln_task_progress_initialization", None)
        task_progress_memory.ensure_current_task_started(
            str(planning_visual_context.current_node_id)
        )
    else:
        task_progress_initialization = ensure_task_progress_memory(
            client=state.llm_client,
            task_progress=task_progress_memory,
            goal_text=goal_text,
            task_type=goal_kind,
            cache=state.cache,
            visual_context=planning_visual_context,
        )
    entries = build_memory_index(state)
    workspace = RetrievalWorkspace(
        entries=entries,
        max_retrieve_rounds=int(state.max_retrieve_rounds),
    )
    progress_update_count = 0
    terminal_check_payload: dict[str, object] | None = None
    knowledge_management_payload: dict[str, object] | None = None
    knowledge_management_complete = False
    task_progress_decision: TaskProgressDecision | None = None
    progress_update_history: list[dict[str, object]] = []

    def manage_knowledge_once() -> None:
        nonlocal knowledge_management_complete
        nonlocal knowledge_management_payload
        if knowledge_management_complete or workspace.rounds == []:
            return
        if task_progress_decision is None:
            raise ValueError("knowledge management requires final task progress")
        knowledge_management_payload = manage_retrieved_knowledge(
            client=state.llm_client,
            state=state,
            workspace=workspace,
            current_node_id=str(planning_visual_context.current_node_id),
            progress_updates=progress_update_history,
        )
        knowledge_management_complete = True

    def update_step_trace(
        *,
        final_navigation_mode: NavigationModeDecision | None = None,
        terminal_check: dict[str, object] | None = None,
        failure_reason: str = "",
    ) -> None:
        payload = workspace.to_dict()
        payload["progress_update_count"] = int(progress_update_count)
        payload["post_approach_terminal_check"] = bool(terminal_check_required)
        payload["physical_current_node_id"] = str(
            physical_visual_context.current_node_id
        )
        payload["planning_current_node_id"] = str(
            planning_visual_context.current_node_id
        )
        payload["backtrack_contexts"] = [
            deepcopy(item) for item in backtrack_contexts
        ]
        payload["memory_context_mode"] = "active_retrieval"
        if task_progress_decision is not None:
            payload["final_progress_update"] = task_progress_decision.to_dict()
        if progress_update_history != []:
            payload["progress_update_history"] = [
                deepcopy(item) for item in progress_update_history
            ]
        if final_navigation_mode is not None:
            payload["final_navigation_mode"] = final_navigation_mode.to_dict()
        if str(failure_reason).strip() != "":
            payload["failure_reason"] = str(failure_reason)
        resolved_terminal_check = (
            terminal_check_payload if terminal_check is None else terminal_check
        )
        if resolved_terminal_check is not None:
            payload["terminal_check"] = deepcopy(resolved_terminal_check)
        if knowledge_management_payload is not None:
            payload["knowledge_management"] = deepcopy(
                knowledge_management_payload
            )
        step.episodic_retrieval = payload

    update_step_trace()
    task_progress_update_result = TaskProgressUpdateResult()
    progress_is_current = False
    decision_count = 0
    max_decisions = (
        2 * int(workspace.max_retrieve_rounds)
        + 3
        + 2 * int(NAVIGATION_REPLAN_MAX_ATTEMPTS)
    )
    while decision_count < max_decisions:
        decision_count += 1
        has_landmark_evidence = (
            planning_landmark_context is not None
            and list(planning_landmark_context.evidences) != []
        )
        landmark_panorama_views = (
            draw_vln_landmark_panorama_views(
                cache=state.cache,
                visual_context=planning_visual_context,
                evidences=planning_landmark_context.evidences,
            )
            if has_landmark_evidence
            else None
        )
        detected_landmarks_text = (
            str(planning_landmark_context.text)
            if has_landmark_evidence
            else ""
        )
        planning_node_ref = str(planning_visual_context.current_node_id)
        provided_fields_by_ref = {
            planning_node_ref: [
                "rgb",
                *(["landmarks"] if has_landmark_evidence else []),
            ]
        }
        loop_fields_by_ref = workspace.fields_by_ref(
            provided_fields_by_ref=provided_fields_by_ref,
        )
        loop_index_text = memory_index_text(
            entries,
            provided_fields_by_ref=provided_fields_by_ref,
        )
        has_retrievable_fields = any(loop_fields_by_ref.values())
        all_backtrack_node_ids = _vln_backtrack_node_ids(
            state=state,
            current_node_id=str(planning_visual_context.current_node_id),
        )
        allowed_backtrack_node_ids = (
            all_backtrack_node_ids
            if int(step.place_step_index) > 0
            and len(backtrack_contexts) < int(NAVIGATION_REPLAN_MAX_ATTEMPTS)
            else set()
        )
        allow_retrieve = (
            not progress_is_current
            and int(step.place_step_index) > 0
            and workspace.can_retrieve
            and has_retrievable_fields
        )
        retrieval_context_content = (
            workspace.prompt_content()
            if not progress_is_current
            else workspace.conclusion_context_content()
        ) if workspace.rounds != [] else []
        backtrack_context_text = _vln_backtrack_context_text(backtrack_contexts)
        progress_context_text = "\n\n".join(
            item
            for item in (
                _vln_initial_progress_note(
                    goal_kind=goal_kind,
                    step_index=int(step.place_step_index),
                ),
            )
            if item != ""
        )
        planning_reference_panorama = (
            str(planning_visual_context.current_node_id)
            != str(physical_visual_context.current_node_id)
        )
        if progress_is_current:
            if task_progress_decision is None:
                raise ValueError(
                    "PCNP requires the latest task-progress update"
                )
            decision = decide_vln_navigation_step(
                client=state.llm_client,
                cache=state.cache,
                goal_kind=goal_kind,
                visual_context=planning_visual_context,
                task_progress=task_progress_memory,
                latest_task_progress=task_progress_decision,
                retrieval_workspace_content=retrieval_context_content,
                progress_context_text=progress_context_text,
                backtrack_context_text=backtrack_context_text,
                landmark_panorama_views=landmark_panorama_views,
                detected_landmarks_text=detected_landmarks_text,
                allowed_backtrack_node_ids=allowed_backtrack_node_ids,
                require_backtrack=False,
                system_owned_waypoint_objective=(
                    _uses_system_active_waypoint_grounding(
                        goal_kind=_goal_kind(goal),
                    )
                ),
                task_progress_bev_overlay=None,
                planning_reference_panorama=planning_reference_panorama,
            )
        else:
            decision = decide_vln_task_progress_step(
                client=state.llm_client,
                cache=state.cache,
                goal_kind=goal_kind,
                visual_context=planning_visual_context,
                task_progress=task_progress_memory,
                memory_index_text=loop_index_text if allow_retrieve else "",
                retrieval_workspace_content=retrieval_context_content,
                retrieve_max_rounds=(
                    int(workspace.max_retrieve_rounds)
                    if allow_retrieve
                    else 0
                ),
                retrieve_completed_rounds=(
                    int(workspace.retrieve_count) if allow_retrieve else 0
                ),
                retrieve_fields_by_ref=(
                    loop_fields_by_ref if allow_retrieve else {}
                ),
                retrieve_provided_fields_by_ref=(
                    provided_fields_by_ref if allow_retrieve else {}
                ),
                allow_retrieve=allow_retrieve,
                allow_update_progress=True,
                require_retrieval_conclusion=workspace.has_pending_evidence,
                progress_context_text=progress_context_text,
                backtrack_context_text=backtrack_context_text,
                landmark_panorama_views=landmark_panorama_views,
                detected_landmarks_text=detected_landmarks_text,
                terminal_check_context=(
                    pending_terminal_check if terminal_check_required else None
                ),
                task_progress_bev_overlay=None,
                planning_reference_panorama=planning_reference_panorama,
            )
        if isinstance(decision, RetrieveRequest):
            condition_update_result = task_progress_memory.apply_condition_updates(
                list(decision.progress_condition_updates),
                current_node_id=str(planning_visual_context.current_node_id),
            )
            task_progress_update_result.applied_updates.extend(
                condition_update_result.applied_updates
            )
            task_progress_update_result.skipped_updates.extend(
                condition_update_result.skipped_updates
            )
            if workspace.has_pending_evidence:
                workspace.conclude_latest_retrieval(decision.retrieval_conclusion)
                workspace.discard_concluded_raw_evidence()
            execute_retrieve_request(
                state=state,
                workspace=workspace,
                request=decision,
            )
            progress_is_current = False
            update_step_trace()
            continue
        if isinstance(decision, TaskProgressDecision):
            if workspace.has_pending_evidence:
                workspace.conclude_latest_retrieval(decision.retrieval_conclusion)
            progress_updates, skipped_progress_updates = _progress_updates_for_goal(
                decision.progress_updates,
                goal=goal,
            )
            item_update_result = task_progress_memory.apply_updates(
                progress_updates,
                current_node_id=str(planning_visual_context.current_node_id),
            )
            task_progress_update_result.applied_updates.extend(
                item_update_result.applied_updates
            )
            task_progress_update_result.skipped_updates.extend(
                item_update_result.skipped_updates
            )
            task_progress_update_result.skipped_updates.extend(skipped_progress_updates)
            condition_update_result = task_progress_memory.apply_condition_updates(
                decision.progress_condition_updates,
                current_node_id=str(planning_visual_context.current_node_id),
            )
            task_progress_update_result.applied_updates.extend(
                condition_update_result.applied_updates
            )
            task_progress_update_result.skipped_updates.extend(
                condition_update_result.skipped_updates
            )
            task_progress_decision = decision
            progress_update_history.extend(
                deepcopy(item) for item in decision.progress_updates
            )
            progress_update_count += 1
            progress_is_current = True
            if workspace.rounds != []:
                workspace.discard_concluded_raw_evidence()
            if terminal_check_required:
                terminal_check_payload = {
                    "decision": str(decision.terminal_check_decision),
                    "reason": str(decision.terminal_check_reasoning),
                    "missing_constraints": [
                        str(item)
                        for item in decision.terminal_check_missing_constraints
                    ],
                }
                has_active_task = any(
                    str(item.status) == "active"
                    for item in task_progress_memory.items
                    if str(item.kind) == "task"
                )
                if decision.terminal_check_decision == "done" and has_active_task:
                    raise ValueError(
                        "terminal_check done requires all task progress items to be done"
                    )
                if decision.terminal_check_decision == "continue" and not has_active_task:
                    raise ValueError(
                        "terminal_check continue requires an active task progress item"
                    )
                state.pending_vln_terminal_check = None
                if decision.terminal_check_decision == "done":
                    manage_knowledge_once()
                    update_step_trace(terminal_check=terminal_check_payload)
                    return EpisodicRetrievalLoopResult(
                        task_progress_initialization=task_progress_initialization,
                        task_progress_decision=task_progress_decision,
                        task_progress_update_result=task_progress_update_result,
                        navigation_mode=None,
                        workspace=workspace,
                        progress_update_count=int(progress_update_count),
                        planning_visual_context=planning_visual_context,
                        backtrack_contexts=[
                            deepcopy(item) for item in backtrack_contexts
                        ],
                    )
                terminal_check_required = False
                pending_terminal_check = None
                update_step_trace(terminal_check=terminal_check_payload)
            else:
                update_step_trace()
            continue
        if task_progress_decision is None or not progress_is_current:
            raise ValueError("navigation action preceded the required progress update")
        if str(decision.action_mode) == "backtrack":
            backtrack_contexts.append(
                _vln_backtrack_context(
                    decision=decision,
                    planning_node_id=str(planning_visual_context.current_node_id),
                    physical_node_id=str(physical_visual_context.current_node_id),
                )
            )
            planning_visual_context = build_visual_action_context_for_node(
                state=state,
                node_id=str(decision.backtrack_anchor_node_id),
                context_evidence_text=str(visual_context.context_evidence_text),
            )
            planning_landmark_context = build_vln_landmark_context(
                landmark_controller=state.landmark_controller,
                graph=state.graph,
                visual_context=planning_visual_context,
                floor_id=str(state.system.current_floor_id),
            )
            progress_is_current = False
            update_step_trace()
            continue
        manage_knowledge_once()
        update_step_trace(final_navigation_mode=decision)
        return EpisodicRetrievalLoopResult(
            task_progress_initialization=task_progress_initialization,
            task_progress_decision=task_progress_decision,
            task_progress_update_result=task_progress_update_result,
            navigation_mode=decision,
            workspace=workspace,
            progress_update_count=int(progress_update_count),
            planning_visual_context=planning_visual_context,
            backtrack_contexts=[deepcopy(item) for item in backtrack_contexts],
        )
    raise ValueError("VLN progress-navigation agent exceeded its bounded decision loop")


def plan_visual_navigation_action(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
    goal: "GoalSpec | None",
    context_evidence_text: str,
) -> VisualNavigationDecision:
    if _goal_kind(goal) != GOAL_KIND_VLN_INSTRUCTION:
        raise ValueError("the public agent only accepts VLN instructions")
    goal_text = _goal_text(state=state, goal=goal)
    task_progress_memory = state.system.memory.task_progress
    visual_context = build_visual_action_context(
        state=state,
        step=step,
        context_evidence_text=context_evidence_text,
    )
    physical_visual_context = visual_context
    planning_visual_context = visual_context
    backtrack_contexts: list[dict[str, object]] = []
    vln_landmark_context = build_vln_landmark_context(
        landmark_controller=state.landmark_controller,
        graph=state.graph,
        visual_context=visual_context,
        floor_id=str(state.system.current_floor_id),
    )
    include_task_progress = True
    include_graph_context = True
    initial_navigation_mode: NavigationModeDecision | None = None
    context_loop = run_episodic_retrieval_loop(
        state=state,
        step=step,
        goal=goal,
        goal_text=goal_text,
        visual_context=visual_context,
        vln_landmark_context=vln_landmark_context,
    )
    episodic_context_loop = context_loop
    task_progress_initialization = context_loop.task_progress_initialization
    task_progress_decision = context_loop.task_progress_decision
    task_progress_update_result = context_loop.task_progress_update_result
    initial_navigation_mode = context_loop.navigation_mode
    planning_visual_context = (
        context_loop.planning_visual_context
        if context_loop.planning_visual_context is not None
        else physical_visual_context
    )
    backtrack_contexts = [
        deepcopy(item) for item in context_loop.backtrack_contexts
    ]
    if str(planning_visual_context.current_node_id) != str(
        physical_visual_context.current_node_id
    ):
        vln_landmark_context = build_vln_landmark_context(
            landmark_controller=state.landmark_controller,
            graph=state.graph,
            visual_context=planning_visual_context,
            floor_id=str(state.system.current_floor_id),
        )
    if str(context_loop.failure_reason).strip() != "":
        return VisualNavigationDecision(
            visual_context=planning_visual_context,
            task_progress_initialization=task_progress_initialization,
            navigation_mode=None,
            task_progress_update_result=task_progress_update_result,
            visual_action=None,
            waypoint=None,
            action_call=None,
            failure_reason=str(context_loop.failure_reason),
            context_evidence_text=context_evidence_text,
            physical_current_node_id=str(
                physical_visual_context.current_node_id
            ),
            planning_current_node_id=str(
                planning_visual_context.current_node_id
            ),
            backtrack_contexts=backtrack_contexts,
        )
    if initial_navigation_mode is None:
        terminal_check = {
            "decision": str(task_progress_decision.terminal_check_decision),
            "reason": str(task_progress_decision.terminal_check_reasoning),
            "missing_constraints": [
                str(item)
                for item in task_progress_decision.terminal_check_missing_constraints
            ],
        }
        if terminal_check["decision"] != "done":
            raise ValueError(
                "missing navigation mode requires a completed terminal check"
            )
        return VisualNavigationDecision(
            visual_context=visual_context,
            task_progress_initialization=task_progress_initialization,
            navigation_mode=None,
            task_progress_update_result=task_progress_update_result,
            visual_action=None,
            waypoint=None,
            action_call=ActionCall(action="done", args={}),
            context_evidence_text=context_evidence_text,
            terminal_check=terminal_check,
            physical_current_node_id=str(
                physical_visual_context.current_node_id
            ),
            planning_current_node_id=str(
                planning_visual_context.current_node_id
            ),
            backtrack_contexts=backtrack_contexts,
        )
    _record_route_status_for_warning(
        state=state,
        route_status=(
            task_progress_decision.route_status
            if _goal_kind(goal) == GOAL_KIND_VLN_INSTRUCTION
            else ""
        ),
    )
    updated_task_progress_text = (
        task_progress_memory.format_for_prompt(
            include_node_bindings=visual_context.graph_context_visible,
            show_empty_progress_conditions=episodic_context_loop is not None,
        )
    )
    navigation_replan_feedback: list[dict[str, object]] = []
    rejected_backtrack_anchor_node_ids: set[str] = set()
    waypoint_attempts: list[VisualWaypointAttempt] = []
    visual_action: VisualActionPointDecision | None = None
    accepted_verification: VisualWaypointVerificationDecision | None = None
    local_move_plan: LocalMovePlanDecision | None = None
    selected_view: VisualViewContext | None = None
    navigation_mode: NavigationModeDecision | None = initial_navigation_mode
    waypoint_target = ""

    for navigation_attempt_index in range(NAVIGATION_REPLAN_MAX_ATTEMPTS):
        if navigation_mode is None:
            if episodic_context_loop is not None:
                workspace = episodic_context_loop.workspace
                replan_backtrack_node_ids = (
                    _vln_backtrack_node_ids(
                        state=state,
                        current_node_id=str(
                            planning_visual_context.current_node_id
                        ),
                    )
                    - rejected_backtrack_anchor_node_ids
                )
                navigation_mode = decide_vln_navigation_step(
                    client=state.llm_client,
                    cache=state.cache,
                    goal_kind=_goal_kind(goal),
                    visual_context=planning_visual_context,
                    task_progress=task_progress_memory,
                    retrieval_workspace_content=(
                        workspace.conclusion_context_content()
                        if workspace.rounds != []
                        else []
                    ),
                    latest_task_progress=task_progress_decision,
                    backtrack_context_text=_vln_backtrack_context_text(
                        backtrack_contexts
                    ),
                    navigation_replan_feedback_text=_navigation_replan_feedback_text(
                        navigation_replan_feedback
                    ),
                    landmark_panorama_views=(
                        draw_vln_landmark_panorama_views(
                            cache=state.cache,
                            visual_context=planning_visual_context,
                            evidences=vln_landmark_context.evidences,
                        )
                        if vln_landmark_context is not None
                        and list(vln_landmark_context.evidences) != []
                        else None
                    ),
                    detected_landmarks_text=(
                        str(vln_landmark_context.text)
                        if vln_landmark_context is not None
                        and list(vln_landmark_context.evidences) != []
                        else ""
                    ),
                    allowed_backtrack_node_ids=replan_backtrack_node_ids,
                    system_owned_waypoint_objective=_uses_system_active_waypoint_grounding(
                        goal_kind=_goal_kind(goal),
                    ),
                    planning_reference_panorama=(
                        str(planning_visual_context.current_node_id)
                        != str(physical_visual_context.current_node_id)
                    ),
                )
                step.episodic_retrieval["final_navigation_mode"] = (
                    navigation_mode.to_dict()
                )
        if (
            str(navigation_mode.action_mode) == "approach_to_stop"
            and str(navigation_mode.approach_movement) == "stay"
        ):
            planning_node_id = str(planning_visual_context.current_node_id)
            physical_node_id = str(physical_visual_context.current_node_id)
            if planning_node_id != physical_node_id:
                planning_node = state.graph.get_node(planning_node_id)
                goal_xy = (
                    planning_node.nav_goal_xy
                    if planning_node.nav_goal_xy is not None
                    else (
                        float(planning_node.position[0]),
                        float(planning_node.position[1]),
                    )
                )
                target = GroundedWaypointTarget(
                    goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                    goal_yaw=float(planning_node.yaw),
                    world_z=float(planning_node.position[2]),
                    policy_name=FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
                    source_type="planning_node",
                    source_id=planning_node_id,
                    obs_id=str(planning_visual_context.view_for_angle(0).obs_id),
                    angle_deg=0,
                )
                return VisualNavigationDecision(
                    visual_context=planning_visual_context,
                    task_progress_initialization=task_progress_initialization,
                    navigation_mode=navigation_mode,
                    task_progress_update_result=task_progress_update_result,
                    visual_action=None,
                    waypoint=None,
                    action_call=target.to_action_call(),
                    local_move_plan=LocalMovePlanDecision(
                        selected_angle_deg=0,
                        waypoint_target=(
                            f"return to planning node {planning_node_id}"
                        ),
                        reasoning=str(navigation_mode.action_reason),
                    ),
                    grounded_waypoint_target=target,
                    waypoint_policy_result=WaypointPolicyResult(
                        policy_name=FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
                        target=target,
                        reasoning=str(navigation_mode.action_reason),
                    ),
                    waypoint_attempts=waypoint_attempts,
                    navigation_replan_feedback=navigation_replan_feedback,
                    context_evidence_text=context_evidence_text,
                    physical_current_node_id=physical_node_id,
                    planning_current_node_id=planning_node_id,
                    backtrack_contexts=backtrack_contexts,
                )
            return VisualNavigationDecision(
                visual_context=planning_visual_context,
                task_progress_initialization=task_progress_initialization,
                navigation_mode=navigation_mode,
                task_progress_update_result=task_progress_update_result,
                visual_action=None,
                waypoint=None,
                action_call=None,
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
                context_evidence_text=context_evidence_text,
                physical_current_node_id=physical_node_id,
                planning_current_node_id=planning_node_id,
                backtrack_contexts=backtrack_contexts,
            )
        use_vln_fss_stop = str(navigation_mode.action_mode) in {
            "stop",
            "approach_to_stop",
        }
        if (
            navigation_mode.action_mode in {"go_to_waypoint", "backtrack"}
            or use_vln_fss_stop
        ):
            use_system_active_progress = (
                str(navigation_mode.action_mode) == "go_to_waypoint"
                and _uses_system_active_waypoint_grounding(
                    goal_kind=_goal_kind(goal),
                )
            )
            context_active_progress_item = _current_active_task_progress_item(
                task_progress_memory
            )
            active_progress_item = (
                context_active_progress_item if use_system_active_progress else ""
            )
            inherited_agent_context_content = _vln_waypoint_inherited_agent_context(
                context_loop=episodic_context_loop,
                task_progress=task_progress_decision,
                navigation_mode=navigation_mode,
                active_progress_item=context_active_progress_item,
                task_progress_text=updated_task_progress_text,
            )
            use_candidate_bev = True
            candidate_bev_base = None
            if use_candidate_bev:
                candidate_bev_reference_node_id = _vln_sampled_planner_node_id(
                    state=state,
                    navigation_mode=navigation_mode,
                    current_node_id=str(planning_visual_context.current_node_id),
                    floor_id=str(state.system.current_floor_id),
                )
                candidate_bev_base = _vln_waypoint_candidate_bev_base(
                    state=state,
                    step_index=int(step.place_step_index),
                    current_node_id=candidate_bev_reference_node_id,
                    context_loop=episodic_context_loop,
                    landmark_context=vln_landmark_context,
                    include_graph_context=visual_context.graph_context_visible,
                )
            policy_context = WaypointPlanningContext(
                state=state,
                step=step,
                goal_text=goal_text,
                goal_kind=_goal_kind(goal),
                visual_context=planning_visual_context,
                execution_visual_context=physical_visual_context,
                task_progress=task_progress_decision,
                task_progress_text=updated_task_progress_text,
                task_progress_initialization=task_progress_initialization,
                navigation_mode=navigation_mode,
                task_progress_update_result=task_progress_update_result,
                context_evidence_text=context_evidence_text,
                inherited_agent_context_content=inherited_agent_context_content,
                active_progress_item=active_progress_item,
                candidate_bev_landmark_markers=(
                    []
                    if candidate_bev_base is None
                    else candidate_bev_base.landmark_markers
                ),
                candidate_bev_base_image=(
                    None
                    if candidate_bev_base is None
                    else candidate_bev_base.image
                ),
                candidate_bev_transform=(
                    None
                    if candidate_bev_base is None
                    else candidate_bev_base.transform
                ),
                candidate_bev_reference_node_marker=(
                    None
                    if candidate_bev_base is None
                    else candidate_bev_base.reference_node_marker
                ),
                candidate_bev_context_text=(
                    ""
                    if candidate_bev_base is None
                    else candidate_bev_base.context_text
                ),
            )
            if use_vln_fss_stop:
                policy_output = _plan_frontier_skeleton_sample_stop_policy(
                    policy_context
                )
            else:
                policy_output = _plan_frontier_skeleton_sample_waypoint_policy(
                    policy_context
                )
            waypoint_decision = replace(
                policy_output.decision,
                waypoint_policy_result=policy_output.result,
                grounded_waypoint_target=policy_output.result.target,
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=[
                    deepcopy(item) for item in backtrack_contexts
                ],
            )
            if str(policy_output.result.failure_reason) in {
                "no_projected_sampled_waypoint_candidates",
                "go_to_waypoint_no_active_frontiers",
                "go_to_waypoint_no_active_frontier_labels",
            }:
                failed_backtrack_index: int | None = None
                planning_node_id = str(
                    planning_visual_context.current_node_id
                ).strip()
                for index in range(len(backtrack_contexts) - 1, -1, -1):
                    context = backtrack_contexts[index]
                    result = context.get("result")
                    if (
                        str(context.get("anchor_node_id", "")).strip()
                        == planning_node_id
                        and not isinstance(result, dict)
                    ):
                        failed_backtrack_index = int(index)
                        break
                if failed_backtrack_index is None:
                    return _with_replan_log(
                        waypoint_decision,
                        waypoint_attempts=waypoint_attempts,
                        navigation_replan_feedback=navigation_replan_feedback,
                    )
                failed_backtrack = deepcopy(
                    backtrack_contexts[failed_backtrack_index]
                )
                reference_node_id = str(
                    failed_backtrack.get("anchor_node_id", "")
                ).strip()
                failure_summary = (
                    "The previous backtrack failed because its anchor node "
                    "has no available reachable waypoint. The intended region "
                    "may be represented more closely by another visited node."
                )
                failed_backtrack["result"] = {
                    "status": "failed",
                    "navigation_action_mode": str(navigation_mode.action_mode),
                    "detail": failure_summary,
                }
                backtrack_contexts[failed_backtrack_index] = failed_backtrack
                if reference_node_id != "":
                    rejected_backtrack_anchor_node_ids.add(reference_node_id)
                trigger_node_id = str(
                    failed_backtrack.get("trigger_planning_node_id", "")
                ).strip()
                if trigger_node_id != "":
                    planning_visual_context = (
                        physical_visual_context
                        if trigger_node_id
                        == str(physical_visual_context.current_node_id)
                        else build_visual_action_context_for_node(
                            state=state,
                            node_id=trigger_node_id,
                            context_evidence_text=str(
                                visual_context.context_evidence_text
                            ),
                        )
                    )
                    vln_landmark_context = build_vln_landmark_context(
                        landmark_controller=state.landmark_controller,
                        graph=state.graph,
                        visual_context=planning_visual_context,
                        floor_id=str(state.system.current_floor_id),
                    )
                if episodic_context_loop is not None:
                    episodic_context_loop = replace(
                        episodic_context_loop,
                        planning_visual_context=planning_visual_context,
                        backtrack_contexts=[
                            deepcopy(item) for item in backtrack_contexts
                        ],
                    )
                step.episodic_retrieval["planning_current_node_id"] = str(
                    planning_visual_context.current_node_id
                )
                step.episodic_retrieval["backtrack_contexts"] = [
                    deepcopy(item) for item in backtrack_contexts
                ]
                if (
                    navigation_attempt_index + 1
                    < NAVIGATION_REPLAN_MAX_ATTEMPTS
                ):
                    navigation_mode = None
                    continue
            return _with_replan_log(
                waypoint_decision,
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
            )
        if navigation_mode.action_mode == "vertical_transition":
            return VisualNavigationDecision(
                visual_context=planning_visual_context,
                task_progress_initialization=task_progress_initialization,
                navigation_mode=navigation_mode,
                task_progress_update_result=task_progress_update_result,
                visual_action=None,
                waypoint=None,
                action_call=ActionCall(
                    action="vertical_transition",
                    args={"direction": str(navigation_mode.vertical_direction)},
                ),
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
                context_evidence_text=context_evidence_text,
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=[
                    deepcopy(item) for item in backtrack_contexts
                ],
            )
        raise ValueError(
            f"unsupported canonical VLN navigation mode: {navigation_mode.action_mode!r}"
        )
    raise ValueError("canonical VLN navigation replanning was exhausted")


def confirm_pending_visual_stop(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
    goal: "GoalSpec | None",
    pending_stop: dict[str, object],
) -> dict[str, object]:
    goal_text = _goal_text(state=state, goal=goal)
    visual_context = build_visual_action_context(
        state=state,
        step=step,
        context_evidence_text="",
    )
    decision = decide_stop_confirmation(
        client=state.llm_client,
        cache=state.cache,
        goal_text=goal_text,
        goal_kind=_goal_kind(goal),
        visual_context=visual_context,
        pending_stop=pending_stop,
    )
    payload = {
        "pending_stop": deepcopy(pending_stop),
        "visual_context": visual_context.to_dict(),
        "decision": decision.to_dict(),
    }
    step.stop_confirmation = payload
    state.stop_confirmation_history.append(payload)
    return payload


def _plan_vln_waypoint_navigation(
    *,
    state: "NavClawAgentState",
    goal_kind: str,
    visual_context: VisualActionContext,
    execution_visual_context: VisualActionContext | None,
    task_progress_initialization: dict[str, object],
    navigation_mode: NavigationModeDecision,
    task_progress_update_result: TaskProgressUpdateResult,
    context_evidence_text: str,
    inherited_agent_context_content: list[dict[str, object]],
    active_progress_item: str = "",
    candidate_bev_landmark_markers: list[dict[str, object]] | None = None,
    candidate_bev_base_image: object | None = None,
    candidate_bev_transform: BevOverlayTransform | None = None,
    candidate_bev_reference_node_marker: dict[str, object] | None = None,
    candidate_bev_context_text: str = "",
    step_index: int,
    waypoint_loop=plan_vln_waypoint_loop,
    selection_kind: str = "sampled_waypoint_candidate",
    no_action_failure: str = "vln_waypoint_no_action",
    sample_spacing_m: float | None = None,
    max_distance_m: float | None = None,
    enable_node_dedup: bool = True,
    use_global_node_dedup: bool = False,
) -> VisualNavigationDecision:
    action_origin_visual_context = (
        visual_context
        if execution_visual_context is None
        else execution_visual_context
    )
    planner_visual_context = _vln_sampled_planner_visual_context(
        state=state,
        visual_context=visual_context,
        navigation_mode=navigation_mode,
        context_evidence_text=context_evidence_text,
    )
    planner_context_text = _vln_sampled_planner_context_text(
        state=state,
        navigation_mode=navigation_mode,
        current_node_id=str(action_origin_visual_context.current_node_id),
        planner_node_id=str(planner_visual_context.current_node_id),
        initial_orientation_note=_vln_initial_orientation_note(
            goal_kind=goal_kind,
            step_index=int(step_index),
        ),
    )
    planner_landmark_context = build_vln_landmark_context(
        landmark_controller=state.landmark_controller,
        graph=state.graph,
        visual_context=planner_visual_context,
        floor_id=str(state.system.current_floor_id),
    )
    sampling_kwargs: dict[str, float] = {}
    if sample_spacing_m is not None:
        sampling_kwargs["sample_spacing_m"] = float(sample_spacing_m)
    if max_distance_m is not None:
        sampling_kwargs["max_distance_m"] = float(max_distance_m)
    waypoint_context_kwargs: dict[str, object] = {
        "active_progress_item": str(active_progress_item),
        "candidate_bev_landmark_markers": list(
            candidate_bev_landmark_markers or []
        ),
        "candidate_bev_base_image": candidate_bev_base_image,
        "candidate_bev_transform": candidate_bev_transform,
        "candidate_bev_reference_node_marker": (
            candidate_bev_reference_node_marker
        ),
        "candidate_bev_context_text": str(candidate_bev_context_text),
    }
    if waypoint_loop is plan_vln_waypoint_loop:
        waypoint_context_kwargs["heading_reference"] = (
            f"the stored heading of reference node "
            f"{planner_visual_context.current_node_id}"
            if str(planner_visual_context.current_node_id)
            != str(action_origin_visual_context.current_node_id)
            else "the current robot heading"
        )
    if candidate_bev_base_image is not None:
        waypoint_context_kwargs["candidate_bev_coordinate_exploration"] = (
            state.global_exploration_for_floor(
                str(state.system.current_floor_id)
            )
        )
    if (
        enable_node_dedup
        and use_global_node_dedup
        and str(goal_kind) == GOAL_KIND_VLN_INSTRUCTION
    ):
        waypoint_context_kwargs["node_dedup_map"] = state.global_exploration_for_floor(
            str(state.system.current_floor_id)
        ).map
        waypoint_context_kwargs["node_dedup_radius_m"] = float(
            VLN_WAYPOINT_GLOBAL_NODE_DEDUP_RADIUS_M
        )
    result = waypoint_loop(
        client=state.llm_client,
        cache=state.cache,
        goal_text="",
        visual_context=planner_visual_context,
        task_progress_analysis="",
        task_progress_text="",
        exploration=_vln_waypoint_local_exploration(
            state=state,
            node_id=str(planner_visual_context.current_node_id),
        ),
        floor_height_m=float(state.system.current_floor_height),
        goal_kind=goal_kind,
        planner_context_text=planner_context_text,
        planner_context_images=(
            _waypoint_selection_images_for_node(
                state,
                str(planner_visual_context.current_node_id),
                include_node_identity=planner_visual_context.graph_context_visible,
            )
            if str(planner_visual_context.current_node_id)
            != str(action_origin_visual_context.current_node_id)
            else []
        ),
        inherited_agent_context_content=inherited_agent_context_content,
        landmark_evidences=planner_landmark_context.evidences,
        frontier_records=state.frontier_records_for_floor(str(state.system.current_floor_id)),
        avoid_node_xys=(
            _vln_waypoint_avoid_node_xys(
                state=state,
                planner_node_id=str(planner_visual_context.current_node_id),
                floor_id=str(state.system.current_floor_id),
            )
            if enable_node_dedup
            else []
        ),
        **waypoint_context_kwargs,
        **sampling_kwargs,
    )
    waypoint_attempts = _vln_waypoint_attempts_from_records(result.attempt_records)
    if result.failure_reason != "" or result.waypoint is None:
        return _visual_navigation_failure(
            visual_context=visual_context,
            task_progress_initialization=task_progress_initialization,
            navigation_mode=navigation_mode,
            task_progress_update_result=task_progress_update_result,
            visual_action=result.visual_action,
            waypoint_attempts=waypoint_attempts,
            failure_reason=str(result.failure_reason or no_action_failure),
            context_evidence_text=context_evidence_text,
            local_move_plan=result.local_move_plan,
            navigation_replan_feedback=result.navigation_replan_feedback,
        )
    waypoint = replace(
        result.waypoint,
        goal_yaw=_vln_waypoint_start_to_goal_yaw(
            state=state,
            visual_context=planner_visual_context,
            waypoint=result.waypoint,
        ),
    )
    selected_label: int | None = None
    candidate_overlay: np.ndarray | None = None
    if result.attempt_records != []:
        record = result.attempt_records[-1]
        sampled_candidate = record.get("sampled_candidate", {})
        if isinstance(sampled_candidate, dict) and sampled_candidate.get("label") is not None:
            try:
                selected_label = int(sampled_candidate["label"])
            except (TypeError, ValueError):
                selected_label = None
        overlay_images = record.get("overlay_images", {})
        if isinstance(overlay_images, dict) and isinstance(
            overlay_images.get("sampled_candidate_rgb_overlay"),
            np.ndarray,
        ):
            candidate_overlay = np.asarray(overlay_images["sampled_candidate_rgb_overlay"], dtype=np.uint8)
    _store_last_visual_waypoint_selection_context(
        state=state,
        waypoint=waypoint,
        reasoning="" if result.local_move_plan is None else str(result.local_move_plan.reasoning),
        planner_node_id=str(planner_visual_context.current_node_id),
        selection_kind=selection_kind,
        selected_label=selected_label,
        overlay_image=candidate_overlay,
    )
    action_call = ActionCall(
        action="move",
        args={
            "x": float(waypoint.goal_xy[0]),
            "y": float(waypoint.goal_xy[1]),
            "yaw": float(waypoint.goal_yaw),
            "z": float(waypoint.raw_world_z),
        },
    )
    return VisualNavigationDecision(
        visual_context=visual_context,
        task_progress_initialization=task_progress_initialization,
        navigation_mode=navigation_mode,
        task_progress_update_result=task_progress_update_result,
        visual_action=result.visual_action,
        waypoint=waypoint,
        action_call=action_call,
        local_move_plan=result.local_move_plan,
        waypoint_attempts=waypoint_attempts,
        navigation_replan_feedback=result.navigation_replan_feedback,
        context_evidence_text=context_evidence_text,
    )


def _store_last_visual_waypoint_selection_context(
    *,
    state: "NavClawAgentState",
    waypoint: VisualWaypoint,
    reasoning: str,
    planner_node_id: str | None = None,
    selection_kind: str = "visual_waypoint",
    selected_label: int | None = None,
    overlay_image: np.ndarray | None = None,
) -> None:
    if overlay_image is None:
        observation = state.cache.get_observation(str(waypoint.obs_id)).observation
        image = draw_waypoint_overlay_rgb(
            observation.rgb,
            point_pixel=(float(waypoint.point_pixel[0]), float(waypoint.point_pixel[1])),
        )
    else:
        image = np.asarray(overlay_image, dtype=np.uint8)
    image_record = state.cache.store_image(
        image,
        kind="last_waypoint_selection_overlay",
        metadata={
            "selection_kind": str(selection_kind),
            "obs_id": str(waypoint.obs_id),
            "angle_deg": int(waypoint.angle_deg),
            "planner_node_id": "" if planner_node_id is None else str(planner_node_id),
            "selected_label": None if selected_label is None else int(selected_label),
        },
    )
    context = {
        "selection_kind": str(selection_kind),
        "selected_angle_deg": int(waypoint.angle_deg),
        "planner_node_id": "" if planner_node_id is None else str(planner_node_id),
        "selected_label": None if selected_label is None else int(selected_label),
        "reasoning": str(reasoning),
        "image_id": str(image_record.id),
    }
    state.last_waypoint_selection_context = context
    if planner_node_id is not None and str(planner_node_id).strip() != "":
        state.waypoint_selection_context_by_node_id[str(planner_node_id)] = dict(context)


def _waypoint_selection_context_for_node(
    state: "NavClawAgentState",
    planner_node_id: str,
) -> dict[str, object] | None:
    contexts = getattr(state, "waypoint_selection_context_by_node_id", {})
    if not isinstance(contexts, dict):
        return None
    context = contexts.get(str(planner_node_id))
    if not isinstance(context, dict) or context == {}:
        return None
    return context


def _waypoint_selection_text_for_node(
    *,
    state: "NavClawAgentState",
    planner_node_id: str,
    recovery_reason: str,
) -> str:
    context = _waypoint_selection_context_for_node(state, planner_node_id)
    if context is None:
        return f"Previous selection at reference_node_id {planner_node_id}: none."
    angle = context.get("selected_angle_deg")
    try:
        direction_text = direction_for_angle(int(angle))
    except (TypeError, ValueError):
        direction_text = "unknown direction"
    kind = str(context.get("selection_kind", "waypoint")).strip() or "waypoint"
    selected_label = context.get("selected_label")
    label_text = ""
    if selected_label is not None:
        try:
            label_text = f" label {int(selected_label)}"
        except (TypeError, ValueError):
            label_text = ""
    selection = f"{direction_text} {kind}{label_text}".strip()
    return (
        f"previous_selection: {selection}; "
        f"replan_reason: {str(recovery_reason).strip() or 'none'}"
    )


def _waypoint_selection_images_for_node(
    state: "NavClawAgentState",
    planner_node_id: str,
    *,
    include_node_identity: bool = True,
) -> list[tuple[str, np.ndarray]]:
    context = _waypoint_selection_context_for_node(state, planner_node_id)
    if context is None:
        return []
    image_id = str(context.get("image_id", "")).strip()
    if image_id == "":
        return []
    try:
        image = np.asarray(state.cache.get_image(image_id).image, dtype=np.uint8)
    except KeyError:
        return []
    label = context.get("selected_label")
    label_text = ""
    if label is not None:
        try:
            label_text = f" The previous selected label was {int(label)}."
        except (TypeError, ValueError):
            label_text = ""
    text = (
        f"Backtrack feedback RGB for reference_node_id {planner_node_id}: "
        "sampled labels from the previous decision are shown."
        f"{label_text}"
        if include_node_identity
        else (
            "Backtrack feedback RGB from the previous decision at the selected prior node: "
            f"sampled labels are shown.{label_text}"
        )
    )
    return [(text, image)]


def _vln_sampled_planner_node_id(
    *,
    state: "NavClawAgentState",
    navigation_mode: NavigationModeDecision,
    current_node_id: str,
    floor_id: str,
) -> str:
    if str(navigation_mode.action_mode) == "backtrack":
        anchor_node_id = str(navigation_mode.backtrack_anchor_node_id).strip()
    elif str(navigation_mode.route_status) == "recovery_needed":
        anchor_node_id = str(navigation_mode.recovery_anchor_node_id).strip()
    else:
        return str(current_node_id)
    if anchor_node_id == "" or anchor_node_id == str(current_node_id):
        return str(current_node_id)
    if not state.graph.has_node(anchor_node_id):
        return str(current_node_id)
    anchor_node = state.graph.get_node(anchor_node_id)
    if str(anchor_node.floor_id) != str(floor_id) or str(anchor_node.node_kind) != "place":
        return str(current_node_id)
    if list(anchor_node.obs_ids) == []:
        return str(current_node_id)
    return anchor_node_id


def _vln_navigation_uses_prior_node(
    navigation_mode: NavigationModeDecision,
) -> bool:
    return (
        str(navigation_mode.action_mode) == "backtrack"
        or str(navigation_mode.route_status) == "recovery_needed"
    )


def _vln_prior_node_reason(navigation_mode: NavigationModeDecision) -> str:
    if str(navigation_mode.action_mode) == "backtrack":
        return str(navigation_mode.backtrack_reason)
    return str(navigation_mode.recovery_reason)


def _vln_sampled_planner_visual_context(
    *,
    state: "NavClawAgentState",
    visual_context: VisualActionContext,
    navigation_mode: NavigationModeDecision,
    context_evidence_text: str,
) -> VisualActionContext:
    planner_node_id = _vln_sampled_planner_node_id(
        state=state,
        navigation_mode=navigation_mode,
        current_node_id=str(visual_context.current_node_id),
        floor_id=str(state.system.current_floor_id),
    )
    if str(planner_node_id) == str(visual_context.current_node_id):
        return visual_context
    return build_visual_action_context_for_node(
        state=state,
        node_id=planner_node_id,
        context_evidence_text=context_evidence_text,
    )


def _vln_sampled_planner_context_text(
    *,
    state: "NavClawAgentState",
    navigation_mode: NavigationModeDecision,
    current_node_id: str,
    planner_node_id: str,
    initial_orientation_note: str = "",
) -> str:
    orientation_note = str(initial_orientation_note).strip()
    orientation_lines = []
    if orientation_note != "":
        orientation_lines = [orientation_note]
    if str(planner_node_id) == str(current_node_id):
        lines = [
            f"reference_node_id: {current_node_id}",
            "Candidate waypoints are generated from the current robot panorama.",
            *orientation_lines,
        ]
        return "\n".join(lines).strip()
    lines = [
        "Waypoint reference:",
        f"- current_robot_node_id: {current_node_id}",
        f"- reference_node_id: {planner_node_id}",
        "- Candidate overlays are generated from the reference node panorama.",
        "- Execution starts from the current robot node and routes to the selected candidate.",
        f"- {_waypoint_selection_text_for_node(state=state, planner_node_id=planner_node_id, recovery_reason=_vln_prior_node_reason(navigation_mode))}",
        *[f"- {line}" for line in orientation_lines],
    ]
    return "\n".join(lines).strip()


def _vln_waypoint_start_to_goal_yaw(
    *,
    state: "NavClawAgentState",
    visual_context: VisualActionContext,
    waypoint: VisualWaypoint,
) -> float:
    start_xy = _vln_waypoint_start_xy(state=state, visual_context=visual_context)
    goal_xy = np.asarray(waypoint.goal_xy, dtype=np.float64).reshape(2)
    delta = goal_xy - start_xy
    if float(np.linalg.norm(delta)) <= 1e-6:
        return float(waypoint.goal_yaw)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def _vln_waypoint_start_xy(
    *,
    state: "NavClawAgentState",
    visual_context: VisualActionContext,
) -> np.ndarray:
    current_view = visual_context.view_for_angle(0)
    observation = state.cache.get_observation(str(current_view.obs_id)).observation
    if observation.T_odom_base is not None:
        transform = np.asarray(observation.T_odom_base, dtype=np.float64)
        return np.asarray([float(transform[0, 3]), float(transform[1, 3])], dtype=np.float64)
    return np.asarray([float(observation.pose.x), float(observation.pose.y)], dtype=np.float64)


def _vln_waypoint_local_exploration(
    *,
    state: "NavClawAgentState",
    node_id: str,
) -> "ExplorationManager":
    node_id_text = str(node_id).strip()
    if node_id_text == "":
        raise ValueError("VLN waypoint sampling requires a planner node id")
    try:
        return state.local_explorations_by_node_id[node_id_text]
    except KeyError as exc:
        raise ValueError(f"missing local exploration for VLN waypoint node {node_id_text!r}") from exc


def _vln_waypoint_avoid_node_xys(
    *,
    state: "NavClawAgentState",
    planner_node_id: str,
    floor_id: str,
) -> list[tuple[float, float]]:
    planner_node_id_text = str(planner_node_id).strip()
    floor_id_text = str(floor_id).strip()
    avoid_xys: list[tuple[float, float]] = []
    with state.graph.lock:
        for node in state.graph.iter_nodes(floor_id=floor_id_text):
            if str(node.node_kind) != "place":
                continue
            if str(node.id) == planner_node_id_text:
                continue
            avoid_xys.append((float(node.position[0]), float(node.position[1])))
    return avoid_xys


def _vln_waypoint_attempts_from_records(records: list[dict[str, object]]) -> list[VisualWaypointAttempt]:
    attempts: list[VisualWaypointAttempt] = []
    for index, record in enumerate(records):
        visual_action = record.get("visual_action")
        verification = record.get("verification")
        attempts.append(
            VisualWaypointAttempt(
                attempt_index=int(index),
                selected_angle_deg=int(record.get("selected_angle_deg", 0)),
                selected_obs_id=str(record.get("selected_obs_id", "")),
                waypoint_target=str(record.get("waypoint_target", "")),
                visual_action=visual_action if isinstance(visual_action, VisualActionPointDecision) else None,
                verification=verification if isinstance(verification, VisualWaypointVerificationDecision) else None,
                failure_reason=str(record.get("failure_reason", "")),
                sampled_candidate=(
                    dict(record.get("sampled_candidate", {}))
                    if isinstance(record.get("sampled_candidate"), dict)
                    else {}
                ),
                candidate_selection=(
                    dict(record.get("candidate_selection", {}))
                    if isinstance(record.get("candidate_selection"), dict)
                    else {}
                ),
                candidate_count=(
                    int(record["candidate_count"])
                    if record.get("candidate_count") is not None
                    else None
                ),
            )
        )
    return attempts














def _plan_sampled_waypoint_policy(
    context: WaypointPlanningContext,
    *,
    policy_name: str,
    source_type: str,
    waypoint_loop,
    selection_kind: str,
    no_action_failure: str,
    sample_spacing_m: float | None = None,
    max_distance_m: float | None = None,
    enable_node_dedup: bool = True,
    use_global_node_dedup: bool = False,
) -> WaypointPolicyOutput:
    decision = _plan_vln_waypoint_navigation(
        state=context.state,
        goal_kind=context.goal_kind,
        visual_context=context.visual_context,
        execution_visual_context=getattr(
            context,
            "execution_visual_context",
            None,
        ),
        task_progress_initialization=context.task_progress_initialization,
        navigation_mode=context.navigation_mode,
        task_progress_update_result=context.task_progress_update_result,
        context_evidence_text=context.context_evidence_text,
        inherited_agent_context_content=context.inherited_agent_context_content,
        active_progress_item=getattr(context, "active_progress_item", ""),
        candidate_bev_landmark_markers=getattr(
            context,
            "candidate_bev_landmark_markers",
            [],
        ),
        candidate_bev_base_image=getattr(
            context,
            "candidate_bev_base_image",
            None,
        ),
        candidate_bev_transform=getattr(
            context,
            "candidate_bev_transform",
            None,
        ),
        candidate_bev_reference_node_marker=getattr(
            context,
            "candidate_bev_reference_node_marker",
            None,
        ),
        candidate_bev_context_text=getattr(
            context,
            "candidate_bev_context_text",
            "",
        ),
        step_index=int(context.step.place_step_index),
        waypoint_loop=waypoint_loop,
        selection_kind=selection_kind,
        no_action_failure=no_action_failure,
        sample_spacing_m=sample_spacing_m,
        max_distance_m=max_distance_m,
        enable_node_dedup=enable_node_dedup,
        use_global_node_dedup=use_global_node_dedup,
    )
    target: GroundedWaypointTarget | None = None
    reasoning = "" if decision.local_move_plan is None else str(decision.local_move_plan.reasoning)
    if decision.waypoint is not None and decision.action_call is not None:
        waypoint = decision.waypoint
        source_id = ""
        if decision.waypoint_attempts != []:
            sampled_candidate = decision.waypoint_attempts[-1].sampled_candidate
            if sampled_candidate.get("label") is not None:
                source_id = str(sampled_candidate["label"])
        target = GroundedWaypointTarget(
            goal_xy=(float(waypoint.goal_xy[0]), float(waypoint.goal_xy[1])),
            goal_yaw=float(waypoint.goal_yaw),
            world_z=float(waypoint.raw_world_z),
            policy_name=policy_name,
            source_type=source_type,
            source_id=source_id,
            obs_id=str(waypoint.obs_id),
            angle_deg=int(waypoint.angle_deg),
            point_2d=(float(waypoint.point_2d[0]), float(waypoint.point_2d[1])),
            raw_world_xy=(float(waypoint.raw_world_xy[0]), float(waypoint.raw_world_xy[1])),
        )
        decision = replace(decision, action_call=target.to_action_call())
    result = WaypointPolicyResult(
        policy_name=policy_name,
        target=target,
        reasoning=reasoning,
        failure_reason=str(decision.failure_reason),
    )
    return WaypointPolicyOutput(result=result, decision=decision)


def _plan_frontier_skeleton_sample_waypoint_policy(
    context: WaypointPlanningContext,
) -> WaypointPolicyOutput:
    return _plan_sampled_waypoint_policy(
        context,
        policy_name=FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
        source_type="sample",
        waypoint_loop=plan_vln_waypoint_loop,
        selection_kind="sampled_waypoint_candidate",
        no_action_failure="vln_waypoint_no_action",
        use_global_node_dedup=True,
    )


def _plan_frontier_skeleton_sample_stop_policy(
    context: WaypointPlanningContext,
) -> WaypointPolicyOutput:
    return _plan_sampled_waypoint_policy(
        context,
        policy_name=FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
        source_type="stop_sample",
        waypoint_loop=plan_vln_waypoint_loop,
        selection_kind="stop_sampled_waypoint_candidate",
        no_action_failure="vln_stop_waypoint_no_action",
        sample_spacing_m=VLN_STOP_WAYPOINT_SAMPLE_SPACING_M,
        max_distance_m=VLN_STOP_WAYPOINT_MAX_DISTANCE_M,
        enable_node_dedup=False,
    )












def _visual_navigation_failure(
    *,
    visual_context: VisualActionContext,
    task_progress_initialization: dict[str, object],
    navigation_mode: NavigationModeDecision,
    task_progress_update_result: TaskProgressUpdateResult,
    visual_action: VisualActionPointDecision | None,
    waypoint_attempts: list[VisualWaypointAttempt],
    failure_reason: str,
    context_evidence_text: str,
    local_move_plan: LocalMovePlanDecision | None = None,
    navigation_replan_feedback: list[dict[str, object]] | None = None,
) -> VisualNavigationDecision:
    return VisualNavigationDecision(
        visual_context=visual_context,
        task_progress_initialization=task_progress_initialization,
        navigation_mode=navigation_mode,
        task_progress_update_result=task_progress_update_result,
        visual_action=visual_action,
        waypoint=None,
        action_call=None,
        local_move_plan=local_move_plan,
        waypoint_attempts=waypoint_attempts,
        navigation_replan_feedback=[dict(item) for item in (navigation_replan_feedback or [])],
        failure_reason=str(failure_reason),
        context_evidence_text=context_evidence_text,
    )


def _goal_text(
    *,
    state: "NavClawAgentState",
    goal: "GoalSpec | None",
) -> str:
    if goal is not None:
        navigation_goal_text = getattr(goal, "navigation_goal_text", None)
        if callable(navigation_goal_text):
            goal_text = str(navigation_goal_text()).strip()
            if goal_text != "":
                return goal_text
        description = str(getattr(goal, "description", "")).strip()
        if description != "":
            return description
    return str(state.system.goal.target)


def _robot_xy_from_step(
    *,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> tuple[float, float]:
    obs_id = str(step.current_obs_id or state.current_obs_id or "").strip()
    if obs_id != "" and obs_id in state.cache.observations:
        observation = state.cache.get_observation(obs_id).observation
        if observation.T_odom_base is not None:
            transform = np.asarray(observation.T_odom_base, dtype=np.float64)
            return (float(transform[0, 3]), float(transform[1, 3]))
        return (float(observation.pose.x), float(observation.pose.y))
    current_node_id = str(step.current_place_node_id or state.current_place_node_id or "").strip()
    if current_node_id == "":
        raise ValueError("cannot infer robot_xy without current observation or current node")
    node = state.graph.get_node(current_node_id)
    return (float(node.position[0]), float(node.position[1]))


def _candidate_source_view(
    *,
    step: "NavClawStepState",
    visual_context: VisualActionContext,
    candidate_view: VisualViewContext,
) -> dict[str, object]:
    return {
        "step_index": int(step.place_step_index),
        "node_id": str(visual_context.current_node_id),
        "angle_deg": int(candidate_view.angle_deg),
        "obs_id": str(candidate_view.obs_id),
        "image_id": str(candidate_view.node_overlay_image_id),
        "visible_visited_nodes": [str(node_id) for node_id in candidate_view.visible_visited_nodes],
    }
