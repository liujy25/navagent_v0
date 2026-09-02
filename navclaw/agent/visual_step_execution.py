from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING

from navclaw.agent.actions import AgentAction, AgentActionType
from navclaw.agent.decisions import act_decision
from navclaw.agent.feedback import apply_agent_feedback
from navclaw.agent.node_moves import set_pending_node_move
from navclaw.agent.visual_navigation import confirm_pending_visual_stop
from navclaw.agent.visual_navigation import plan_visual_navigation_action
from navclaw.agent.visual_policy_decisions import NAVIGATION_STOP_APPROACH_ACTION_MODES
from navclaw.agent.vertical_transition import execute_vertical_transition_action
from navclaw.schemas import ActionCall
from navclaw.types import LocalmapPlaceReuseState

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentContext, NavClawAgentState


def run_pending_stop_confirmation(
    *,
    context: "NavClawAgentContext",
    state: "NavClawAgentState",
    step,
) -> bool:
    pending_stop = state.pending_stop_confirmation
    if not isinstance(pending_stop, dict):
        return False
    payload = confirm_pending_visual_stop(
        state=state,
        step=step,
        goal=context.goal_spec,
        pending_stop=pending_stop,
    )
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise ValueError(f"stop confirmation payload missing decision: {payload!r}")
    decision_name = str(decision.get("decision", "")).strip()
    if decision_name != "stop":
        step.policy_decision = {
            "source": "visual_stop_confirmation_continue",
            "decision": deepcopy(payload),
        }
        state.pending_stop_confirmation = None
        return False

    _execute_terminal_stop(
        state=state,
        step=step,
        pending_stop=pending_stop,
        policy_decision={
            "source": "visual_stop_confirmation",
            "decision": deepcopy(payload),
        },
        reason=str(decision.get("reasoning", "")),
        executed_action_type="visual_stop_confirmed",
        extra_executed_fields={"confirmation": deepcopy(payload)},
    )
    return True


def _execute_terminal_stop(
    *,
    state: "NavClawAgentState",
    step,
    pending_stop: dict[str, object],
    policy_decision: dict[str, object],
    reason: str,
    executed_action_type: str,
    extra_executed_fields: dict[str, object],
) -> None:
    action = AgentAction(
        action_type=AgentActionType.FINALIZE,
        args={"reason": "done"},
        source=str(policy_decision.get("source", "visual_stop_confirmation")),
        reason=str(reason),
    )
    step.agent_action = action.to_dict()
    step.agent_decision = act_decision(action=action, reason=str(action.reason)).to_dict()
    step.policy_decision = deepcopy(policy_decision)
    action_call = ActionCall(action="done", args={})
    segment_timer_started = datetime.now().isoformat()
    result = state.action_executor.execute(action_call)
    step.results = [(action_call, result)]
    step.navigation_segment_timings = [
        {
            "segment_index": 0,
            "segment_type": "done",
            "move_ok": bool(result.ok),
            "started_at": segment_timer_started,
            "timing": deepcopy(result.data.get("timing", {})),
            "rgb_history_obs_ids": [],
            "path_point_count": 0,
        }
    ]
    stop_result = result.data.get("stop_result")
    if isinstance(stop_result, dict):
        state.stop_result = dict(stop_result)
    step.executed_action = {
        "type": str(executed_action_type),
        "pending_stop": deepcopy(pending_stop),
        **deepcopy(extra_executed_fields),
        "done_call": action_call.to_dict(),
        "done_result": result.to_dict(),
    }
    state.last_executed_action = step.executed_action
    state.pending_vln_terminal_check = None
    state.pending_stop_confirmation = None
    state.finalize_reason = "done"
    apply_agent_feedback(action=action, state=state, step=step)


def run_visual_waypoint_decision(
    *,
    context: "NavClawAgentContext",
    state: "NavClawAgentState",
    step,
    context_evidence_text: str,
) -> None:
    decision = plan_visual_navigation_action(
        state=state,
        step=step,
        goal=context.goal_spec,
        context_evidence_text=str(context_evidence_text),
    )
    decision_payload = decision.to_dict()
    if (
        isinstance(getattr(step, "episodic_retrieval", None), dict)
        and step.episodic_retrieval != {}
    ):
        decision_payload["episodic_retrieval"] = deepcopy(step.episodic_retrieval)
    step.visual_decision = decision_payload
    step.policy_decision = {
        "source": "visual_waypoint_policy",
        "decision": decision_payload,
    }
    if decision.navigation_mode is None:
        if str(decision.failure_reason).strip() != "":
            failure_reason = str(decision.failure_reason)
            step.agent_action = {
                "action_type": "visual_waypoint",
                "args": {
                    "selected_angle_deg": None,
                    "selected_obs_id": "",
                    "target": "",
                    "failure_reason": failure_reason,
                },
                "source": "visual_waypoint_policy",
                "reason": failure_reason,
            }
            step.agent_decision = {
                "action": deepcopy(step.agent_action),
                "reason": failure_reason,
            }
            state.finalize_reason = (
                f"visual_waypoint_failed:{failure_reason}"
            )
            return
        if (
            decision.action_call is None
            or decision.action_call.action != "done"
            or str(decision.terminal_check.get("decision", "")) != "done"
        ):
            raise ValueError("progress terminal decision requires a done action")
        _execute_terminal_stop(
            state=state,
            step=step,
            pending_stop={},
            policy_decision={
                "source": "vln_progress_terminal_check",
                "terminal_check": deepcopy(decision.terminal_check),
            },
            reason=str(decision.terminal_check.get("reasoning", "")),
            executed_action_type="vln_progress_terminal_done",
            extra_executed_fields={
                "terminal_check": deepcopy(decision.terminal_check)
            },
        )
        return
    if (
        decision.navigation_mode.action_mode == "approach_to_stop"
        and decision.navigation_mode.approach_movement == "stay"
        and decision.action_call is None
    ):
        current_node_id = str(step.current_place_node_id)
        reason = str(decision.navigation_mode.reasoning_action)
        pending_stop = {
            "requested_at_step": int(step.place_step_index),
            "approach_completed_at_step": int(step.place_step_index),
            "origin_place_node_id": current_node_id,
            "movement": "stay",
            "stop_objective": str(decision.navigation_mode.stop_objective),
            "reasoning": reason,
            "rgb_history_obs_ids": [],
        }
        action_payload = {
            "action_type": "approach_to_stop",
            "args": {"movement": "stay"},
            "source": "visual_waypoint_policy",
            "reason": reason,
        }
        step.agent_action = deepcopy(action_payload)
        step.agent_decision = {
            "action": deepcopy(action_payload),
            "reason": reason,
        }
        step.results = []
        step.navigation_segment_timings = []
        step.executed_action = {
            "type": "vln_approach_to_stop_stay",
            "movement": "stay",
            "origin_place_node_id": current_node_id,
            "decision": deepcopy(decision_payload),
            "pending_terminal_check": deepcopy(pending_stop),
            "rgb_history_obs_ids": [],
            "path_xy": [],
        }
        state.pending_vln_terminal_check = pending_stop
        state.pending_stop_confirmation = None
        state.reuse_current_place_next_step = LocalmapPlaceReuseState(
            place_node_id=current_node_id,
            reason="approach_to_stop_stay",
        )
        state.last_executed_action = step.executed_action
        return
    if decision.action_call is None:
        failure_reason = str(decision.failure_reason or "visual_waypoint_no_action")
        step.agent_action = {
            "action_type": "visual_waypoint",
            "args": {
                "selected_angle_deg": (
                    None if decision.local_move_plan is None else decision.local_move_plan.selected_angle_deg
                ),
                "selected_obs_id": "",
                "target": "",
                "failure_reason": failure_reason,
            },
            "source": "visual_waypoint_policy",
            "reason": str(decision.navigation_mode.reasoning_action),
        }
        step.agent_decision = {
            "action": deepcopy(step.agent_action),
            "reason": str(decision.navigation_mode.reasoning_action),
        }
        state.finalize_reason = f"visual_waypoint_failed:{failure_reason}"
        return

    if decision.action_call.action == "vertical_transition":
        direction = str(decision.action_call.args.get("direction", "")).strip()
        vertical_args: dict[str, object] = {"direction": direction}
        if str(decision.planning_current_node_id).strip() != "":
            vertical_args["planning_node_id"] = str(
                decision.planning_current_node_id
            )
        if decision.backtrack_contexts != []:
            vertical_args["backtrack_contexts"] = [
                deepcopy(item) for item in decision.backtrack_contexts
            ]
        agent_action = AgentAction(
            action_type=AgentActionType.VERTICAL_TRANSITION,
            args=vertical_args,
            source="visual_waypoint_policy",
            reason=str(decision.navigation_mode.reasoning_action),
        )
        step.agent_action = agent_action.to_dict()
        step.agent_decision = act_decision(
            action=agent_action,
            reason=str(decision.navigation_mode.reasoning_action),
        ).to_dict()
        execute_vertical_transition_action(
            context=context,
            state=state,
            step=step,
            action=agent_action,
        )
        apply_agent_feedback(action=agent_action, state=state, step=step)
        return

    if decision.grounded_waypoint_target is not None:
        grounded_target = decision.grounded_waypoint_target
        frontier_id = str(grounded_target.consume_frontier_id)
        if frontier_id != "":
            raise ValueError("canonical VLN waypoints must not consume frontier ids")
        if decision.local_move_plan is None or decision.local_move_plan.selected_angle_deg is None:
            raise ValueError("grounded waypoint requires local_move_plan.selected_angle_deg")
        stop_approach = (
            str(decision.navigation_mode.action_mode)
            in NAVIGATION_STOP_APPROACH_ACTION_MODES
        )
        navigation_action_mode = str(decision.navigation_mode.action_mode)
        decision_type = (
            navigation_action_mode
            if stop_approach or navigation_action_mode == "backtrack"
            else "go_to_waypoint"
        )
        agent_action = AgentAction(
            action_type=AgentActionType.VISUAL_WAYPOINT,
            args={
                "selected_angle_deg": int(decision.local_move_plan.selected_angle_deg),
                "selected_obs_id": str(grounded_target.obs_id),
                "target": str(decision.local_move_plan.waypoint_target),
                "stop_approach": stop_approach,
                "decision_type": decision_type,
                "waypoint_policy": str(grounded_target.policy_name),
            },
            source="visual_waypoint_policy",
            reason=str(decision.local_move_plan.waypoint_target),
        )
        step.agent_action = agent_action.to_dict()
        step.agent_decision = act_decision(
            action=agent_action,
            reason=str(decision.navigation_mode.reasoning_action),
        ).to_dict()
        _execute_visual_action_call(
            state=state,
            step=step,
            action=agent_action,
            action_call=decision.action_call,
            decision_payload=decision_payload,
        )
        apply_agent_feedback(action=agent_action, state=state, step=step)
        return

    if decision.local_move_plan is None or decision.local_move_plan.selected_angle_deg is None:
        raise ValueError("visual waypoint action requires local_move_plan.selected_angle_deg")
    selected_obs_id = "" if decision.waypoint is None else str(decision.waypoint.obs_id)
    target = "" if decision.visual_action is None else str(decision.visual_action.target)
    agent_action = AgentAction(
        action_type=AgentActionType.VISUAL_WAYPOINT,
        args={
            "selected_angle_deg": int(decision.local_move_plan.selected_angle_deg),
            "selected_obs_id": selected_obs_id,
            "target": target,
            "stop_approach": (
                str(decision.navigation_mode.action_mode)
                in NAVIGATION_STOP_APPROACH_ACTION_MODES
            ),
            "decision_type": (
                str(decision.navigation_mode.action_mode)
                if str(decision.navigation_mode.action_mode)
                in {*NAVIGATION_STOP_APPROACH_ACTION_MODES, "backtrack"}
                else "go_to_waypoint"
            ),
        },
        source="visual_waypoint_policy",
        reason=str(decision.local_move_plan.waypoint_target),
    )
    step.agent_action = agent_action.to_dict()
    step.agent_decision = act_decision(
        action=agent_action,
        reason=str(decision.navigation_mode.reasoning_action),
    ).to_dict()
    _execute_visual_action_call(
        state=state,
        step=step,
        action=agent_action,
        action_call=decision.action_call,
        decision_payload=decision_payload,
    )
    apply_agent_feedback(action=agent_action, state=state, step=step)


def _execute_visual_action_call(
    *,
    state: "NavClawAgentState",
    step,
    action: AgentAction,
    action_call: ActionCall,
    decision_payload: dict[str, object],
) -> None:
    segment_timer_started = datetime.now().isoformat()
    result = state.action_executor.execute(action_call)
    step.results = [(action_call, result)]
    rgb_history_obs_ids = _rgb_history_obs_ids_from_result(result)
    path_xy = _path_xy_from_result(result)
    step.navigation_segment_timings = [
        {
            "segment_index": 0,
            "segment_type": str(action_call.action),
            "move_ok": bool(result.ok),
            "started_at": segment_timer_started,
            "timing": deepcopy(result.data.get("timing", {})),
            "rgb_history_obs_ids": rgb_history_obs_ids,
            "path_point_count": len(path_xy),
        }
    ]
    step.executed_action = {
        "type": "move_to_visual_waypoint",
        "route_completed": bool(result.ok),
        "origin_place_node_id": str(step.current_place_node_id),
        "decision": deepcopy(decision_payload),
        "move_call": action_call.to_dict(),
        "move_result": result.to_dict(),
        "rgb_history_obs_ids": rgb_history_obs_ids,
        "path_xy": path_xy,
    }
    state.last_executed_action = step.executed_action
    if bool(result.ok):
        stop_approach = bool(action.args.get("stop_approach", False))
        state.previous_place_node_id = str(step.current_place_node_id)
        decision_type = str(action.args.get("decision_type", "")).strip()
        navigation_mode = decision_payload.get("navigation_mode")
        navigation_action_mode = (
            str(navigation_mode.get("action_mode", "")).strip()
            if isinstance(navigation_mode, dict)
            else ""
        )
        backtrack_reference_node_id = ""
        backtrack_contexts = decision_payload.get("backtrack_contexts", [])
        if not isinstance(backtrack_contexts, list):
            backtrack_contexts = []
        if backtrack_contexts != []:
            latest_backtrack = backtrack_contexts[-1]
            if isinstance(latest_backtrack, dict):
                backtrack_reference_node_id = str(
                    latest_backtrack.get("anchor_node_id", "")
                ).strip()
            decision_type = "backtrack"
        elif navigation_action_mode == "backtrack":
            decision_type = "backtrack"
            backtrack_reference_node_id = str(
                navigation_mode.get("backtrack_anchor_node_id", "")
            ).strip()
        set_pending_node_move(
            state=state,
            step_id=int(step.place_step_index),
            from_node_id=str(step.current_place_node_id),
            reason=str(action.reason),
            move_mode=(
                decision_type if decision_type != "" else "visual_waypoint"
            ),
            backtrack_reference_node_id=backtrack_reference_node_id,
            backtrack_contexts=[
                deepcopy(item)
                for item in backtrack_contexts
                if isinstance(item, dict)
            ],
            rgb_history_obs_ids=rgb_history_obs_ids,
            path_xy=path_xy,
        )
        if stop_approach:
            local_move_plan = decision_payload.get("local_move_plan")
            visual_action = decision_payload.get("visual_action")
            waypoint = decision_payload.get("waypoint")
            pending_stop = {
                "requested_at_step": int(step.place_step_index),
                "approach_completed_at_step": int(step.place_step_index),
                "origin_place_node_id": str(step.current_place_node_id),
                "stop_objective": (
                    str(navigation_mode.get("stop_objective", ""))
                    if isinstance(navigation_mode, dict)
                    else ""
                ),
                "waypoint_target": (
                    str(local_move_plan.get("waypoint_target", ""))
                    if isinstance(local_move_plan, dict)
                    else str(action.reason)
                ),
                "reasoning": (
                    str(navigation_mode.get("reasoning_action", ""))
                    if isinstance(navigation_mode, dict)
                    else ""
                ),
                "selected_angle_deg": action.args.get("selected_angle_deg"),
                "selected_obs_id": action.args.get("selected_obs_id"),
                "target": action.args.get("target"),
                "visual_action_point_2d": (
                    deepcopy(visual_action.get("point_2d"))
                    if isinstance(visual_action, dict)
                    else None
                ),
                "visual_action_target": (
                    str(visual_action.get("target", ""))
                    if isinstance(visual_action, dict)
                    else ""
                ),
                "visual_action_reasoning": (
                    str(visual_action.get("reasoning", ""))
                    if isinstance(visual_action, dict)
                    else ""
                ),
                "waypoint_obs_id": (
                    str(waypoint.get("obs_id", ""))
                    if isinstance(waypoint, dict)
                    else ""
                ),
                "waypoint_point_2d": (
                    deepcopy(waypoint.get("point_2d"))
                    if isinstance(waypoint, dict)
                    else None
                ),
                "waypoint_point_pixel": (
                    deepcopy(waypoint.get("point_pixel"))
                    if isinstance(waypoint, dict)
                    else None
                ),
                "grounded_waypoint_target": (
                    str(waypoint.get("target", ""))
                    if isinstance(waypoint, dict)
                    else ""
                ),
                "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
            }
            if decision_type == "approach_to_stop":
                pending_stop["movement"] = "move"
                state.pending_vln_terminal_check = pending_stop
            else:
                state.pending_stop_confirmation = pending_stop
        return
    state.finalize_reason = "visual_waypoint_move_failed"


def _rgb_history_obs_ids_from_result(result) -> list[str]:
    return [str(obs_id) for obs_id in list(result.data.get("rgb_history_obs_ids", []))]


def _path_xy_from_result(result) -> list[list[float]]:
    raw_path = result.data.get("path_xy", [])
    if not isinstance(raw_path, list):
        return []
    return [
        [float(point[0]), float(point[1])]
        for point in raw_path
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]
