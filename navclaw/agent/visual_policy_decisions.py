from __future__ import annotations

from dataclasses import dataclass, field


NAVIGATION_STOP_APPROACH_ACTION_MODES = {
    "stop",
    "approach_to_stop",
    "approach_candidate",
}
@dataclass(frozen=True)
class TaskProgressDecision:
    progress_analysis: str
    progress_reasoning: str
    retrieval_conclusion: str = ""
    route_status: str = "continue_current"
    route_status_target_index: int | None = None
    status_reasoning: str = ""
    recovery_reason: str = ""
    recovery_anchor_node_id: str = ""
    progress_updates: list[dict[str, object]] = field(default_factory=list)
    progress_condition_updates: list[dict[str, object]] = field(default_factory=list)
    terminal_check_decision: str = ""
    terminal_check_reasoning: str = ""
    terminal_check_missing_constraints: list[str] = field(default_factory=list)
    # Retain an explicit empty update call without fabricating one for a no-op response.
    update_progress_called: bool = True
    standalone_terminal_check: bool = False

    def to_dict(self) -> dict[str, object]:
        tool_calls: list[dict[str, object]] = []
        if self.progress_condition_updates:
            tool_calls.append(
                {
                    "name": "update_predicates",
                    "arguments": {
                        "predicate_updates": [
                            dict(item) for item in self.progress_condition_updates
                        ]
                    },
                }
            )
        progress_arguments: dict[str, object] = {
            "agenda_updates": [dict(item) for item in self.progress_updates]
        }
        if str(self.terminal_check_decision).strip() != "":
            progress_arguments["terminal_check"] = {
                "decision": str(self.terminal_check_decision),
                "missing_constraints": [
                    str(item) for item in self.terminal_check_missing_constraints
                ],
            }
        if self.update_progress_called or self.progress_updates or (self.terminal_check_decision and not self.standalone_terminal_check):
            tool_calls.append(
                {"name": "update_task_state", "arguments": progress_arguments}
            )
        payload: dict[str, object] = {"tool_calls": tool_calls}
        if self.standalone_terminal_check and self.terminal_check_decision:
            payload["terminal_check"] = progress_arguments["terminal_check"]
        if str(self.progress_analysis).strip() != "":
            payload = {"task_state_assessment": str(self.progress_analysis), **payload}
        if str(self.progress_reasoning).strip() != "":
            payload["progress_reasoning"] = str(self.progress_reasoning)
        if str(self.route_status).strip() != "":
            payload.update(
                {
                    "route_status": str(self.route_status),
                    "route_status_target_index": (
                        None
                        if self.route_status_target_index is None
                        else int(self.route_status_target_index)
                    ),
                    "status_reasoning": str(self.status_reasoning),
                    "recovery_reason": str(self.recovery_reason),
                    "recovery_anchor_node_id": str(self.recovery_anchor_node_id),
                }
            )
        if str(self.retrieval_conclusion).strip() != "":
            payload = {
                "retrieval_conclusion": str(self.retrieval_conclusion),
                **payload,
            }
        return payload


@dataclass(frozen=True)
class NavigationModeDecision:
    action_mode: str
    stop_objective: str
    progress_analysis: str
    progress_reasoning: str
    reasoning_action: str
    approach_movement: str = ""
    direction: str = ""
    candidate_angle_deg: int | None = None
    progress_updates: list[dict[str, object]] = field(default_factory=list)
    vertical_direction: str = ""
    task_item_index: int | None = None
    subgoal_id: str | None = None
    subgoal_attempt: int | None = None
    waypoint_target: str = ""
    route_status: str = "continue_current"
    status_reasoning: str = ""
    recovery_reason: str = ""
    recovery_anchor_node_id: str = ""
    action_objective: str = ""
    action_reason: str = ""
    backtrack_reason: str = ""
    backtrack_anchor_node_id: str = ""
    backtrack_objective: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_mode": str(self.action_mode),
            "stop_objective": str(self.stop_objective),
            "reasoning_action": str(self.reasoning_action),
            "direction": str(self.direction),
            "candidate_angle_deg": (
                None
                if self.candidate_angle_deg is None
                else int(self.candidate_angle_deg)
            ),
            "vertical_direction": str(self.vertical_direction),
        }
        if self.action_mode == "vertical_transition" and self.task_item_index is not None:
            payload["task_item_index"] = self.task_item_index
        if self.subgoal_id is not None:
            payload["subgoal_id"] = self.subgoal_id
            if self.subgoal_attempt is not None:
                payload["subgoal_attempt"] = self.subgoal_attempt
        if self.waypoint_target:
            payload["waypoint_target"] = self.waypoint_target
        if self.progress_updates:
            payload["progress_updates"] = [
                dict(item) for item in self.progress_updates
            ]
        if str(self.progress_analysis).strip() != "":
            payload["progress_analysis"] = str(self.progress_analysis)
        if str(self.progress_reasoning).strip() != "":
            payload["progress_reasoning"] = str(self.progress_reasoning)
        if (
            str(self.action_mode).strip() == "approach_to_stop"
            and str(self.approach_movement).strip() != ""
        ):
            payload["approach_movement"] = str(self.approach_movement)
        if str(self.route_status).strip() != "":
            payload.update(
                {
                    "route_status": str(self.route_status),
                    "status_reasoning": str(self.status_reasoning),
                    "recovery_reason": str(self.recovery_reason),
                    "recovery_anchor_node_id": str(self.recovery_anchor_node_id),
                }
            )
        else:
            if str(self.action_objective).strip() != "":
                payload["action_objective"] = str(self.action_objective)
            if str(self.action_reason).strip() != "":
                payload["action_reason"] = str(self.action_reason)
            if str(self.action_mode).strip() == "backtrack":
                payload.update({
                    "backtrack_reason": str(self.backtrack_reason),
                    "backtrack_anchor_node_id": str(self.backtrack_anchor_node_id),
                    "backtrack_objective": str(self.backtrack_objective),
                })
        return payload


@dataclass(frozen=True)
class LocalMovePlanDecision:
    selected_angle_deg: int | None
    waypoint_target: str
    reasoning: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_angle_deg": None if self.selected_angle_deg is None else int(self.selected_angle_deg),
            "waypoint_target": str(self.waypoint_target),
            "reasoning": str(self.reasoning),
            "failure_reason": str(self.failure_reason),
        }


@dataclass(frozen=True)
class ExplorePlannerDecision:
    frontier_label: int
    frontier_id: str
    reasoning: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "frontier_label": int(self.frontier_label),
            "frontier_id": str(self.frontier_id),
            "reasoning": str(self.reasoning),
        }
        if str(self.failure_reason).strip() != "":
            payload["failure_reason"] = str(self.failure_reason)
        return payload


@dataclass(frozen=True)
class NodeFrontierOverlayPromptImage:
    node_reference: str
    frontier_labels: list[int]
    image_id: str
    angle_deg: int | None = None
    obs_id: str = ""


@dataclass(frozen=True)
class VisualActionPointDecision:
    point_2d: tuple[float, float] | None
    target: str
    reasoning: str
    failure_reason: str = ""
    status: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "point_2d": None if self.point_2d is None else [float(self.point_2d[0]), float(self.point_2d[1])],
            "target": str(self.target),
            "reasoning": str(self.reasoning),
            "failure_reason": str(self.failure_reason),
        }
        if str(self.status).strip() != "":
            payload = {"status": str(self.status), **payload}
        return payload


@dataclass(frozen=True)
class VisualWaypointVerificationDecision:
    verdict: str
    critique: str
    target_not_visible: bool = False
    transition_complete_after_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": str(self.verdict),
            "critique": str(self.critique),
            "target_not_visible": bool(self.target_not_visible),
            "transition_complete_after_execution": bool(
                self.transition_complete_after_execution
            ),
        }


@dataclass(frozen=True)
class StopConfirmationDecision:
    decision: str
    reasoning: str
    continue_objective: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": str(self.decision),
            "reasoning": str(self.reasoning),
            "continue_objective": str(self.continue_objective),
        }


def normalize_stop_confirmation(payload: dict[str, object]) -> StopConfirmationDecision:
    decision = str(payload.get("decision", "")).strip()
    if decision not in {"stop", "continue", "reject_candidate"}:
        raise ValueError(f"stop confirmation returned unsupported decision: {payload!r}")
    reasoning = str(payload.get("reasoning", "")).strip()
    if reasoning == "":
        raise ValueError(f"stop confirmation requires reasoning: {payload!r}")
    continue_objective = str(payload.get("continue_objective", "")).strip()
    if decision == "continue" and continue_objective == "":
        raise ValueError(f"stop confirmation continue requires continue_objective: {payload!r}")
    return StopConfirmationDecision(
        decision=decision,
        reasoning=reasoning,
        continue_objective=continue_objective,
    )
