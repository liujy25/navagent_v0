from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from navclaw.agent.actions import AgentAction, AgentActionType

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState, NavClawStepState


class AgentFeedbackType(str, Enum):
    VISUAL_WAYPOINT = "visual_waypoint_feedback"
    VERTICAL_TRANSITION = "vertical_transition_feedback"
    FINALIZE = "finalize_feedback"


@dataclass(frozen=True)
class AgentFeedback:
    feedback_type: AgentFeedbackType
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    state_delta: dict[str, Any] = field(default_factory=dict)
    action_result_refs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "feedback_type": str(self.feedback_type.value),
            "ok": bool(self.ok),
            "summary": deepcopy(self.summary),
            "state_delta": deepcopy(self.state_delta),
            "action_result_refs": deepcopy(self.action_result_refs),
        }


def apply_agent_feedback(
    *,
    action: AgentAction,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> AgentFeedback:
    if action.action_type == AgentActionType.VISUAL_WAYPOINT:
        feedback = _visual_waypoint_feedback(action=action, state=state, step=step)
    elif action.action_type == AgentActionType.VERTICAL_TRANSITION:
        feedback = _vertical_transition_feedback(action=action, state=state, step=step)
    elif action.action_type == AgentActionType.FINALIZE:
        feedback = _finalize_feedback(action=action, state=state, step=step)
    else:
        raise ValueError(f"unsupported agent action type: {action.action_type!r}")
    step.agent_feedback = feedback.to_dict()
    return feedback


def _execution_ok(step: "NavClawStepState") -> bool:
    if step.results == []:
        return step.executed_action is None
    return all(bool(result.ok) for _, result in step.results)


def _action_result_refs(step: "NavClawStepState") -> dict[str, Any]:
    return {
        "executed_action_type": (
            None
            if step.executed_action is None
            else str(step.executed_action.get("type", ""))
        ),
        "result_count": len(step.results),
        "navigation_segment_count": len(step.navigation_segment_timings),
    }


def _rgb_history_obs_ids_from_results(step: "NavClawStepState") -> list[str]:
    obs_ids: list[str] = []
    for _, result in list(step.results):
        for obs_id in list(result.data.get("rgb_history_obs_ids", [])):
            obs_id_str = str(obs_id)
            if obs_id_str not in obs_ids:
                obs_ids.append(obs_id_str)
    return obs_ids


def _visual_waypoint_feedback(
    *,
    action: AgentAction,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> AgentFeedback:
    visual = step.visual_decision if isinstance(step.visual_decision, dict) else {}
    waypoint = visual.get("waypoint") if isinstance(visual.get("waypoint"), dict) else {}
    return AgentFeedback(
        feedback_type=AgentFeedbackType.VISUAL_WAYPOINT,
        ok=_execution_ok(step),
        summary={
            "selected_angle_deg": action.args.get("selected_angle_deg"),
            "selected_obs_id": action.args.get("selected_obs_id"),
            "target": action.args.get("target"),
            "current_place_node_id": state.current_place_node_id,
            "previous_place_node_id": state.previous_place_node_id,
            "goal_xy": waypoint.get("goal_xy"),
            "rgb_history_obs_ids": _rgb_history_obs_ids_from_results(step),
            "finalize_reason": str(state.finalize_reason),
        },
        state_delta={
            "current_place_node_id": state.current_place_node_id,
            "previous_place_node_id": state.previous_place_node_id,
            "pending_node_move": deepcopy(state.pending_node_move),
        },
        action_result_refs=_action_result_refs(step),
    )


def _vertical_transition_feedback(
    *,
    action: AgentAction,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> AgentFeedback:
    executed = step.executed_action if isinstance(step.executed_action, dict) else {}
    return AgentFeedback(
        feedback_type=AgentFeedbackType.VERTICAL_TRANSITION,
        ok=_execution_ok(step),
        summary={
            "direction": str(action.args.get("direction", "")),
            "route_completed": bool(executed.get("route_completed", False)),
            "before_node_id": executed.get("before_node_id"),
            "after_node_id": executed.get("after_node_id"),
            "before_floor_id": executed.get("before_floor_id"),
            "after_floor_id": executed.get("after_floor_id"),
            "finalize_reason": str(state.finalize_reason),
        },
        state_delta={
            "current_floor_id": str(state.system.current_floor_id),
            "current_floor_height": float(state.system.current_floor_height),
            "current_place_node_id": state.current_place_node_id,
            "previous_place_node_id": state.previous_place_node_id,
        },
        action_result_refs=_action_result_refs(step),
    )


def _finalize_feedback(
    *,
    action: AgentAction,
    state: "NavClawAgentState",
    step: "NavClawStepState",
) -> AgentFeedback:
    return AgentFeedback(
        feedback_type=AgentFeedbackType.FINALIZE,
        ok=True,
        summary={
            "reason": str(action.args["reason"]),
            "current_place_node_id": state.current_place_node_id,
        },
        action_result_refs=_action_result_refs(step),
    )
