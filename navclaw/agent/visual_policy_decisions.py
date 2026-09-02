from __future__ import annotations

from dataclasses import dataclass, field


NAVIGATION_STOP_APPROACH_ACTION_MODES = {"stop", "approach_to_stop"}


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

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "progress_updates": [dict(item) for item in self.progress_updates],
            "progress_condition_updates": [
                dict(item) for item in self.progress_condition_updates
            ],
        }
        if self.progress_analysis.strip():
            payload["progress_analysis"] = self.progress_analysis
        if self.progress_reasoning.strip():
            payload["progress_reasoning"] = self.progress_reasoning
        if self.route_status.strip():
            payload.update(
                {
                    "route_status": self.route_status,
                    "route_status_target_index": self.route_status_target_index,
                    "status_reasoning": self.status_reasoning,
                    "recovery_reason": self.recovery_reason,
                    "recovery_anchor_node_id": self.recovery_anchor_node_id,
                }
            )
        if self.retrieval_conclusion.strip():
            payload["retrieval_conclusion"] = self.retrieval_conclusion
        if self.terminal_check_decision.strip():
            payload["terminal_check"] = {
                "decision": self.terminal_check_decision,
                "reasoning": self.terminal_check_reasoning,
                "missing_constraints": list(self.terminal_check_missing_constraints),
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
    progress_updates: list[dict[str, object]] = field(default_factory=list)
    vertical_direction: str = ""
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
            "action_mode": self.action_mode,
            "stop_objective": self.stop_objective,
            "reasoning_action": self.reasoning_action,
            "vertical_direction": self.vertical_direction,
        }
        if self.progress_updates:
            payload["progress_updates"] = [
                dict(item) for item in self.progress_updates
            ]
        if self.progress_analysis.strip():
            payload["progress_analysis"] = self.progress_analysis
        if self.progress_reasoning.strip():
            payload["progress_reasoning"] = self.progress_reasoning
        if self.action_mode == "approach_to_stop" and self.approach_movement.strip():
            payload["approach_movement"] = self.approach_movement
        if self.route_status.strip():
            payload.update(
                {
                    "route_status": self.route_status,
                    "status_reasoning": self.status_reasoning,
                    "recovery_reason": self.recovery_reason,
                    "recovery_anchor_node_id": self.recovery_anchor_node_id,
                }
            )
        else:
            if self.action_objective.strip():
                payload["action_objective"] = self.action_objective
            if self.action_reason.strip():
                payload["action_reason"] = self.action_reason
            if self.action_mode == "backtrack":
                payload.update(
                    {
                        "backtrack_reason": self.backtrack_reason,
                        "backtrack_anchor_node_id": self.backtrack_anchor_node_id,
                        "backtrack_objective": self.backtrack_objective,
                    }
                )
        return payload


@dataclass(frozen=True)
class LocalMovePlanDecision:
    selected_angle_deg: int | None
    waypoint_target: str
    reasoning: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_angle_deg": self.selected_angle_deg,
            "waypoint_target": self.waypoint_target,
            "reasoning": self.reasoning,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class VisualActionPointDecision:
    point_2d: tuple[float, float] | None
    target: str
    reasoning: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "point_2d": (
                None
                if self.point_2d is None
                else [float(self.point_2d[0]), float(self.point_2d[1])]
            ),
            "target": self.target,
            "reasoning": self.reasoning,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class VisualWaypointVerificationDecision:
    verdict: str
    critique: str
    target_not_visible: bool = False
    transition_complete_after_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "critique": self.critique,
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
            "decision": self.decision,
            "reasoning": self.reasoning,
            "continue_objective": self.continue_objective,
        }


def normalize_visual_action_point(
    payload: dict[str, object],
) -> VisualActionPointDecision:
    failure_reason = str(payload.get("failure_reason", "")).strip()
    point = payload.get("point_2d")
    point_2d: tuple[float, float] | None = None
    if not failure_reason:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"visual action requires point_2d length-2 list: {payload!r}")
        point_2d = (float(point[0]), float(point[1]))
        if not all(0.0 <= item <= 1.0 for item in point_2d):
            raise ValueError(f"visual action point_2d outside [0,1]: {payload!r}")
    return VisualActionPointDecision(
        point_2d=point_2d,
        target=str(payload.get("target", "")).strip(),
        reasoning=str(payload.get("reasoning", "")).strip(),
        failure_reason=failure_reason,
    )


def normalize_visual_waypoint_verification(
    payload: dict[str, object],
) -> VisualWaypointVerificationDecision:
    verdict = str(payload.get("verdict", "")).strip()
    allowed_verdicts = {"execute", "revise_same_view", "fallback_navigation", "fail"}
    if verdict not in allowed_verdicts:
        raise ValueError(f"waypoint verifier returned unsupported verdict: {payload!r}")
    critique = str(payload.get("critique", "")).strip()
    if not critique:
        raise ValueError(f"waypoint verifier requires critique: {payload!r}")
    return VisualWaypointVerificationDecision(
        verdict=verdict,
        critique=critique,
        target_not_visible=bool(payload.get("target_not_visible", False)),
        transition_complete_after_execution=bool(
            payload.get("transition_complete_after_execution", False)
        ),
    )


def normalize_stop_confirmation(
    payload: dict[str, object],
) -> StopConfirmationDecision:
    decision = str(payload.get("decision", "")).strip()
    if decision not in {"stop", "continue"}:
        raise ValueError(f"stop confirmation returned unsupported decision: {payload!r}")
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError(f"stop confirmation requires reasoning: {payload!r}")
    continue_objective = str(payload.get("continue_objective", "")).strip()
    if decision == "continue" and not continue_objective:
        raise ValueError(f"stop confirmation continue requires continue_objective: {payload!r}")
    return StopConfirmationDecision(
        decision=decision,
        reasoning=reasoning,
        continue_objective=continue_objective,
    )
