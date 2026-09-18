from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING

from navclaw.agent.node_moves import attach_backtrack_edge_knowledge
from navclaw.agent.visual_action_context import (
    VisualActionContext,
    VisualViewContext,
    build_visual_action_context_for_node,
)
from navclaw.agent.visual_grounding import VisualWaypoint
from navclaw.agent.visual_policy_decisions import (
    VisualActionPointDecision,
    VisualWaypointVerificationDecision,
)
from navclaw.agent.vertical_fss import prepare_vertical_fss_candidates, vertical_fss_visual_waypoint
from navclaw.agent.vertical_transition_policy import select_vertical_fss_waypoint
from navclaw.agent.vertical_transition_policy import (
    VerticalTransitionStepDecision,
    decide_vertical_transition_step,
)
from navclaw.graph.graph import Floor, Graph
from navclaw.mapping.place_graph_ops import _create_place_node
from navclaw.perception.panorama import PanoramaCaptureResult, PanoramaView, _capture_panorama
from navclaw.runtime.timing import WallStepTimer
from navclaw.schemas import ActionCall, ActionResult
from navclaw.types import LocalmapPlaceReuseState

if TYPE_CHECKING:
    from navclaw.agent.actions import AgentAction
    from navclaw.agent.state import NavClawAgentContext, NavClawAgentState, NavClawStepState


DEFAULT_VERTICAL_FLOOR_MATCH_THRESHOLD_M = 0.8
VERTICAL_STEP_REPLAN_MAX_ATTEMPTS = 4


class _NoopVerticalTransitionExploration:
    def observe_raw_observation(self, observation: object) -> None:
        return None


@dataclass(frozen=True)
class VerticalTransitionWaypointPlan:
    step_decision: VerticalTransitionStepDecision
    visual_action: VisualActionPointDecision
    waypoint_verification: VisualWaypointVerificationDecision
    waypoint: VisualWaypoint
    selected_view: VisualViewContext
    attempts: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "step_decision": self.step_decision.to_dict(),
            "visual_action": self.visual_action.to_dict(),
            "waypoint_verification": self.waypoint_verification.to_dict(),
            "waypoint": self.waypoint.to_dict(),
            "selected_view": self.selected_view.to_dict(),
            "attempts": [dict(item) for item in self.attempts],
        }


def execute_vertical_transition_action(
    *,
    context: "NavClawAgentContext",
    state: "NavClawAgentState",
    step: "NavClawStepState",
    action: "AgentAction",
) -> None:
    direction = str(action.args.get("direction", "")).strip()
    if direction not in {"up", "down"}:
        raise ValueError(f"vertical_transition requires direction up or down, got {direction!r}")
    objective = str(action.args.get("waypoint_target", "")).strip()
    subgoal_id = action.args.get("subgoal_id")
    subgoal_attempt = action.args.get("subgoal_attempt")
    completion_key = None
    if not objective:
        raise ValueError("NavProbe VerticalMove requires waypoint_target")
    memory = state.system.memory.task_progress
    if subgoal_id is None:
        if memory.items:
            raise ValueError("NavProbe VerticalMove requires a subgoal for a nonempty agenda")
    else:
        selected = next((item for item in memory.items if item.subgoal_id == subgoal_id), None)
        if selected is None or selected.status != "active" or selected.attempt != subgoal_attempt:
            raise ValueError("NavProbe VerticalMove requires the selected active subgoal attempt")
        completion_key = f"{subgoal_id}:{subgoal_attempt}"
    floor_match_threshold_m = float(
        action.args.get(
            "floor_match_threshold_m",
            getattr(
                context.args,
                "vertical_floor_match_threshold_m",
                DEFAULT_VERTICAL_FLOOR_MATCH_THRESHOLD_M,
            ),
        )
    )
    if floor_match_threshold_m <= 0.0:
        raise ValueError(
            f"vertical_floor_match_threshold_m must be positive, got {floor_match_threshold_m}"
        )
    before_node_id = str(step.current_place_node_id or state.current_place_node_id or "").strip()
    if before_node_id == "":
        raise ValueError("vertical_transition requires a current place node")
    before_floor_id = str(state.system.current_floor_id)
    before_floor_height = float(state.system.current_floor_height)
    planning_node_id = str(
        action.args.get("planning_node_id", before_node_id)
    ).strip() or before_node_id
    backtrack_contexts = [
        deepcopy(item)
        for item in list(action.args.get("backtrack_contexts", []))
        if isinstance(item, dict)
    ]
    state.ensure_floor_runtime_state(before_floor_id, context.global_bev_kwargs)
    before_exploration = state.global_exploration_for_floor(before_floor_id)
    state.action_executor.exploration = before_exploration
    vt_noop_exploration = _NoopVerticalTransitionExploration()

    step.policy_decision.setdefault("vertical_transition", {})
    step.policy_decision["vertical_transition"] = {
        "direction": direction,
        "algorithm": "detected_stair_fss",
        "subgoal_id": subgoal_id,
        "subgoal_attempt": subgoal_attempt,
        "waypoint_target": objective,
        "max_steps": 1,
        "floor_match_threshold_m": float(floor_match_threshold_m),
        "before_node_id": before_node_id,
        "before_floor_id": before_floor_id,
        "before_floor_height": before_floor_height,
        "planning_node_id": planning_node_id,
        "backtrack_contexts": [
            deepcopy(item) for item in backtrack_contexts
        ],
    }

    current_panorama = None
    initial_panorama_source = "current_step"
    if planning_node_id != before_node_id:
        planning_visual_context = build_visual_action_context_for_node(
            state=state,
            node_id=planning_node_id,
            context_evidence_text="",
        )
        current_panorama = _vertical_panorama_from_visual_context(
            planning_visual_context
        )
        initial_panorama_source = f"planning_node:{planning_node_id}"
    if current_panorama is None:
        current_panorama = _vertical_panorama_from_current_step(step)
    if current_panorama is None:
        current_panorama = _capture_vertical_panorama(
            context=context,
            state=state,
        )
        initial_panorama_source = "captured"
    step.policy_decision["vertical_transition"]["initial_panorama_source"] = (
        initial_panorama_source
    )
    attempts: list[dict[str, object]] = []
    executed_move_history: list[dict[str, object]] = []
    current_retry_feedback: dict[str, object] | None = None
    vt_rgb_history_obs_ids: list[str] = []
    vt_rgb_history_blocks: list[list[str]] = []
    vt_path_xy: list[tuple[float, float]] = []
    vt_observed_heights: list[float] = [float(before_floor_height)]
    results: list[tuple[ActionCall, ActionResult]] = []
    segment_timings: list[dict[str, object]] = []
    final_step_decision: VerticalTransitionStepDecision | None = None
    final_waypoint_verification: VisualWaypointVerificationDecision | None = None
    last_waypoint_plan: VerticalTransitionWaypointPlan | None = None
    failure_reason = "vertical_transition_exhausted"
    move_count = 0
    step_replan_count = 0

    while True:
        visual_context = _visual_context_from_panorama(
            current_node_id=(
                planning_node_id if move_count == 0 else before_node_id
            ),
            panorama=current_panorama,
        )
        current_height = _panorama_anchor_height(cache=state.cache, panorama=current_panorama)
        try:
            step_decision = decide_vertical_transition_step(
                client=state.llm_client,
                cache=state.cache,
                instruction=objective,
                visual_context=visual_context,
                initial_observation=(move_count == 0),
                movement_rgb_history_blocks=vt_rgb_history_blocks,
                current_retry_feedback=current_retry_feedback,
                planning_reference=planning_node_id != before_node_id and move_count == 0,
            )
        except ValueError as exc:
            failure_reason = f"step_planner_failed:{exc}"
            attempts.append(
                {
                    "transition_index": int(move_count),
                    "ok": False,
                    "failure_reason": failure_reason,
                    "panorama": current_panorama.to_dict(),
                    "current_height_m": float(current_height),
                    "height_delta_m": float(current_height) - float(before_floor_height),
                }
            )
            break
        final_step_decision = step_decision
        attempt_payload = {
            "transition_index": int(move_count),
            "panorama": current_panorama.to_dict(),
            "current_height_m": float(current_height),
            "height_delta_m": float(current_height) - float(before_floor_height),
            "step_decision": step_decision.to_dict(),
        }
        height_progress_m = _vertical_transition_directional_height_progress(
            direction=direction,
            observed_heights=vt_observed_heights,
        )
        attempt_payload["vertical_transition_height_progress_m"] = float(height_progress_m)
        # NavProbe returns to the executive after one grounded local move.
        # The fresh status records floor/endpoint evidence without completing
        # the agenda or forcing a full staircase traversal within this skill.
        local_move_finished = move_count == 1
        if step_decision.transition_status == "complete" or local_move_finished:
            if planning_node_id != before_node_id and move_count == 0:
                feedback = {
                    "type": "completion_rejected",
                    "requested_direction": str(direction),
                    "reason": "virtual_reference_is_not_physical_completion",
                    "planner_reason": str(step_decision.reason),
                    "height_progress_m": float(height_progress_m),
                }
                step_replan_count += 1
                feedback["transition_index"] = int(move_count)
                feedback["step_replan_attempt_index"] = int(step_replan_count)
                current_retry_feedback = feedback
                attempt_payload["transition_completion_forced_continue_reason"] = (
                    "virtual_reference_is_not_physical_completion"
                )
                attempt_payload["replanned_by_vt_step_planner"] = True
                attempt_payload["step_replan_feedback"] = feedback
                attempts.append(attempt_payload)
                if step_replan_count >= VERTICAL_STEP_REPLAN_MAX_ATTEMPTS:
                    failure_reason = (
                        "vt_step_replan_exhausted:"
                        "completion_rejected_virtual_reference"
                    )
                    break
                continue
            attempts.append(attempt_payload)
            try:
                _finish_vertical_transition_action(
                    context=context,
                    state=state,
                    step=step,
                    results=results,
                    segment_timings=segment_timings,
                    attempts=attempts,
                    executed_move_history=executed_move_history,
                    before_node_id=before_node_id,
                    before_floor_id=before_floor_id,
                    before_floor_height=before_floor_height,
                    direction=direction,
                    floor_match_threshold_m=floor_match_threshold_m,
                    final_panorama=current_panorama,
                    rgb_history_obs_ids=vt_rgb_history_obs_ids,
                    path_xy=vt_path_xy,
                    step_decision=step_decision,
                    waypoint_plan=last_waypoint_plan,
                    planning_node_id=planning_node_id,
                    backtrack_contexts=backtrack_contexts,
                    destination_floor_reached=step_decision.destination_floor_reached,
                    subgoal_id=subgoal_id,
                    subgoal_attempt=subgoal_attempt,
                )
                step.executed_action.update({
                    "algorithm": "detected_stair_fss", "subgoal_id": subgoal_id,
                    "subgoal_attempt": subgoal_attempt, "waypoint_target": objective,
                    "objective_completed": step_decision.transition_status == "complete",
                })
                evidence = {
                    key: deepcopy(step.executed_action.get(key))
                    for key in ("subgoal_id", "subgoal_attempt", "waypoint_target", "direction",
                                "before_node_id", "after_node_id", "before_floor_id",
                                "after_floor_id", "edge_id", "objective_completed")
                }
                if completion_key is not None and step_decision.transition_status == "complete":
                    state.completed_stair_items[completion_key] = evidence
                results[-1][1].data.update(evidence)
            except ValueError as exc:
                failure_reason = f"floor_resolution_failed:{exc}"
                attempt_payload["floor_resolution_failure"] = str(exc)
                break
            return
        if step_decision.transition_status == "fail":
            failure_reason = f"step_planner_failed:{step_decision.reason}"
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break

        if move_count >= 1:
            failure_reason = "vertical_transition_exhausted"
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break

        try:
            waypoint_plan = (_plan_navprobe_vertical_waypoint)(
                state=state,
                direction=direction,
                visual_context=visual_context,
                step_decision=step_decision,
                current_height_m=current_height,
                objective=objective,
                task_context=_navprobe_vertical_task_context(step, action),
            )
        except ValueError as exc:
            failure_reason = f"waypoint_plan_failed:{exc}"
            attempt_payload["ok"] = False
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break
        final_waypoint_verification = waypoint_plan.waypoint_verification
        last_waypoint_plan = waypoint_plan

        move_path = list(waypoint_plan.waypoint.path_xy)
        if move_count == 0:
            if planning_node_id != before_node_id:
                physical_pose = context.env.get_obs().pose
                prefix = before_exploration.map.compute_astar_path(
                    start_xy=(float(physical_pose.x), float(physical_pose.y)),
                    goal_xy=move_path[0],
                )
                if prefix is None or len(prefix) == 0:
                    failure_reason = "no_physical_path_to_backtrack_anchor"
                    attempt_payload["failure_reason"] = failure_reason
                    attempts.append(attempt_payload)
                    break
                move_path = [tuple(map(float, point)) for point in prefix] + move_path[1:]
            if subgoal_id is not None:
                state.system.memory.task_progress.mark_subgoal_started(subgoal_id, before_node_id)
        move_call = ActionCall(
            action="move_along_path",
            args={
                "path_xy": [[float(x), float(y)] for x, y in move_path],
                "final_yaw": float(waypoint_plan.waypoint.goal_yaw),
                "final_z": float(waypoint_plan.waypoint.raw_world_z),
            },
        )
        segment_timer = WallStepTimer(env_step_start=_current_episode_step(context))
        move_result = _execute_vt_move(
            state=state,
            move_call=move_call,
            vt_exploration=vt_noop_exploration,
        )
        segment_timing = segment_timer.finish(_current_episode_step(context))
        results.append((move_call, move_result))
        segment_timings.append(
            {
                "segment_index": int(move_count),
                "segment_type": "vertical_transition_va_move",
                "direction": direction,
                "move_ok": bool(move_result.ok),
                "timing": segment_timing,
                "rgb_history_obs_ids": [
                    str(obs_id)
                    for obs_id in list(move_result.data.get("rgb_history_obs_ids", []))
                ],
                "path_point_count": len(list(move_result.data.get("path_xy", []))),
            }
        )
        attempt_payload["waypoint_plan"] = waypoint_plan.to_dict()
        attempt_payload["move_call"] = move_call.to_dict()
        attempt_payload["move_result"] = move_result.to_dict()
        attempt_payload["timing"] = segment_timing
        attempt_payload["ok"] = bool(move_result.ok)
        attempts.append(attempt_payload)
        if not bool(move_result.ok):
            failure_reason = "move_failed"
            executed_move_history.append(
                _vertical_transition_va_history_entry(
                    transition_index=move_count,
                    waypoint_plan=waypoint_plan,
                    move_result=move_result,
                    failure_reason=failure_reason,
                )
            )
            break
        movement_rgb_obs_ids = [
            str(obs_id)
            for obs_id in list(move_result.data.get("rgb_history_obs_ids", []))
        ]
        vt_rgb_history_blocks.append(movement_rgb_obs_ids)
        _extend_unique_obs_ids(vt_rgb_history_obs_ids, movement_rgb_obs_ids)
        _extend_path_xy(vt_path_xy, move_result.data.get("path_xy"))
        current_panorama = _capture_vertical_panorama(
            context=context,
            state=state,
        )
        post_move_height = _panorama_anchor_height(cache=state.cache, panorama=current_panorama)
        vt_observed_heights.append(float(post_move_height))
        attempt_payload["post_move_panorama"] = current_panorama.to_dict()
        attempt_payload["post_move_height_m"] = float(post_move_height)
        attempt_payload["post_move_height_delta_m"] = (
            float(post_move_height) - float(before_floor_height)
        )
        executed_move_history.append(
            _vertical_transition_va_history_entry(
                transition_index=move_count,
                waypoint_plan=waypoint_plan,
                move_result=move_result,
                post_move_height_m=post_move_height,
                post_move_height_delta_m=float(post_move_height) - float(before_floor_height),
            )
        )
        height_progress_m = _vertical_transition_directional_height_progress(
            direction=direction,
            observed_heights=vt_observed_heights,
        )
        attempt_payload["vertical_transition_height_progress_m"] = float(height_progress_m)
        current_retry_feedback = None
        move_count += 1
        step_replan_count = 0

    failure_call = ActionCall(action="vertical_transition", args={"direction": direction})
    failure_result = ActionResult(
        ok=False,
        data={
            "direction": direction,
            "before_node_id": before_node_id,
            "before_floor_id": before_floor_id,
            "before_floor_height": before_floor_height,
            "attempt_count": int(len(attempts)),
            "attempts": attempts,
            "executed_move_history": [dict(item) for item in executed_move_history],
            "va_result_history": [dict(item) for item in executed_move_history],
            "current_retry_feedback": (
                None if current_retry_feedback is None else dict(current_retry_feedback)
            ),
            "step_decision": None if final_step_decision is None else final_step_decision.to_dict(),
            "verification": (
                None
                if final_waypoint_verification is None
                else final_waypoint_verification.to_dict()
            ),
            "failure_reason": failure_reason,
        },
        message=f"Vertical transition {direction} failed: {failure_reason}",
    )
    results.append((failure_call, failure_result))
    step.results = results
    step.navigation_segment_timings = segment_timings
    step.executed_action = {
        "type": "vertical_transition",
        "route_completed": False,
        "algorithm": "detected_stair_fss",
        "subgoal_id": subgoal_id,
        "subgoal_attempt": subgoal_attempt,
        "waypoint_target": objective,
        "direction": direction,
        "before_node_id": before_node_id,
        "planning_node_id": planning_node_id,
        "before_floor_id": before_floor_id,
        "backtrack_contexts": [
            deepcopy(item) for item in backtrack_contexts
        ],
        "attempts": attempts,
        "executed_move_history": [dict(item) for item in executed_move_history],
        "va_result_history": [dict(item) for item in executed_move_history],
        "current_retry_feedback": (
            None if current_retry_feedback is None else dict(current_retry_feedback)
        ),
        "step_decision": None if final_step_decision is None else final_step_decision.to_dict(),
        "verification": (
            None
            if final_waypoint_verification is None
            else final_waypoint_verification.to_dict()
        ),
        "failure_reason": failure_reason,
    }
    state.last_executed_action = step.executed_action
    state.finalize_reason = f"vertical_transition_failed:{failure_reason}"


def _finish_vertical_transition_action(
    *,
    context: "NavClawAgentContext",
    state: "NavClawAgentState",
    step: "NavClawStepState",
    results: list[tuple[ActionCall, ActionResult]],
    segment_timings: list[dict[str, object]],
    attempts: list[dict[str, object]],
    executed_move_history: list[dict[str, object]],
    before_node_id: str,
    before_floor_id: str,
    before_floor_height: float,
    direction: str,
    floor_match_threshold_m: float,
    final_panorama: PanoramaCaptureResult,
    rgb_history_obs_ids: list[str],
    path_xy: list[tuple[float, float]],
    step_decision: VerticalTransitionStepDecision,
    waypoint_plan: VerticalTransitionWaypointPlan | None,
    planning_node_id: str,
    backtrack_contexts: list[dict[str, object]],
    destination_floor_reached: bool,
    subgoal_id: str | None = None,
    subgoal_attempt: int | None = None,
) -> None:
    completion = _complete_vertical_transition(
        graph=state.graph,
        cache=state.cache,
        before_node_id=before_node_id,
        before_floor_id=before_floor_id,
        before_floor_height=before_floor_height,
        direction=direction,
        floor_match_threshold_m=floor_match_threshold_m,
        final_panorama=final_panorama,
        rgb_history_obs_ids=rgb_history_obs_ids,
        path_xy=path_xy,
        destination_floor_reached=destination_floor_reached,
    )
    state.system.set_current_floor(
        str(completion["after_floor_id"]),
        height=float(completion["after_floor_height"]),
    )
    state.ensure_floor_runtime_state(
        str(completion["after_floor_id"]),
        context.global_bev_kwargs,
    )
    after_exploration = state.global_exploration_for_floor(str(completion["after_floor_id"]))
    state.action_executor.exploration = after_exploration
    state.current_place_node_id = str(completion["after_node_id"])
    state.previous_place_node_id = before_node_id
    state.current_obs_id = str(completion["current_obs_id_after"])
    state.reuse_current_place_next_step = LocalmapPlaceReuseState(
        place_node_id=str(completion["after_node_id"]),
        reason="vertical_transition_completed",
        recovery=deepcopy(completion),
    )
    step.current_place_node_id = str(completion["after_node_id"])
    step.current_obs_id_after = str(completion["current_obs_id_after"])
    _record_vertical_transition_node_move(
        state=state,
        step_id=int(step.place_step_index),
        before_node_id=before_node_id,
        after_node_id=str(completion["after_node_id"]),
        direction=direction,
        rgb_history_obs_ids=rgb_history_obs_ids,
        planning_node_id=planning_node_id,
        backtrack_contexts=backtrack_contexts,
        subgoal_id=subgoal_id, subgoal_attempt=subgoal_attempt,
    )
    if backtrack_contexts != []:
        edge = next(
            (
                item
                for item in state.graph.iter_edges()
                if str(item.id) == str(completion["edge_id"])
            ),
            None,
        )
        if edge is None:
            raise ValueError(
                "vertical transition edge missing after completion: "
                f"{completion['edge_id']}"
            )
        attach_backtrack_edge_knowledge(
            state=state,
            edge=edge,
            update_node_id=str(completion["after_node_id"]),
            backtrack_contexts=backtrack_contexts,
            fallback_trigger_node_id=before_node_id,
            fallback_reference_node_id=planning_node_id,
            from_node_id=before_node_id,
            to_node_id=str(completion["after_node_id"]),
        )

    verification_payload = (
        None if waypoint_plan is None else waypoint_plan.waypoint_verification.to_dict()
    )
    waypoint_plan_payload = None if waypoint_plan is None else waypoint_plan.to_dict()
    completion_call = ActionCall(
        action="vertical_transition",
        args={"direction": direction},
    )
    completion_result = ActionResult(
        ok=True,
        data={
            **completion,
            "planning_node_id": planning_node_id,
            "backtrack_contexts": [
                deepcopy(item) for item in backtrack_contexts
            ],
            "obs_id": str(completion["current_obs_id_after"]),
            "attempt_count": int(len(attempts)),
            "step_decision": step_decision.to_dict(),
            "verification": verification_payload,
            "waypoint_plan": waypoint_plan_payload,
            "executed_move_history": [dict(item) for item in executed_move_history],
            "va_result_history": [dict(item) for item in executed_move_history],
        },
        message=f"Executed local VerticalMove {direction}; endpoint assessment: {step_decision.transition_status}.",
    )
    results.append((completion_call, completion_result))
    step.results = results
    step.navigation_segment_timings = segment_timings
    step.executed_action = {
        "type": "vertical_transition",
        "route_completed": True,
        "direction": direction,
        "before_node_id": before_node_id,
        "planning_node_id": planning_node_id,
        "before_floor_id": before_floor_id,
        "after_node_id": str(completion["after_node_id"]),
        "after_floor_id": str(completion["after_floor_id"]),
        "edge_id": str(completion["edge_id"]),
        "backtrack_contexts": [
            deepcopy(item) for item in backtrack_contexts
        ],
        "attempts": attempts,
        "executed_move_history": [dict(item) for item in executed_move_history],
        "va_result_history": [dict(item) for item in executed_move_history],
        "step_decision": step_decision.to_dict(),
        "verification": verification_payload,
        "waypoint_plan": waypoint_plan_payload,
    }
    state.last_executed_action = step.executed_action


def _navprobe_vertical_task_context(step, action):
    if str(action.args.get("task_context", "")).strip():
        return str(action.args["task_context"])
    retrieval = getattr(step, "episodic_retrieval", None) or {}
    assessment = retrieval.get("final_progress_update", {}).get("task_state_assessment", "")
    conclusions = [
        {key: record.get(key) for key in ("query", "items", "conclusion")}
        for record in retrieval.get("rounds", []) if record.get("conclusion")
    ]
    return (
        f"Executive assessment: {assessment}\n"
        f"Retrieval conclusions and references: {json.dumps(conclusions, ensure_ascii=False)}"
    )


def _plan_navprobe_vertical_waypoint(
    *, state, direction, visual_context, step_decision, current_height_m,
    objective, task_context,
):
    del current_height_m
    prepared = prepare_vertical_fss_candidates(
        cache=state.cache, visual_context=visual_context,
        exploration=state.global_exploration_for_floor(state.system.current_floor_id),
        direction=direction, detector=state.detector,
        map_floor_height=float(state.system.current_floor_height),
        frontier_only=state.waypoint_policy_name == "frontier",
    )
    if not prepared.detected_stair_regions:
        raise ValueError("no detected stair region in the current observation")
    if not prepared.candidates:
        raise ValueError("no visible height-connected stair FSS candidates")
    memory = state.system.memory.task_progress
    text = f"Original goal: {memory.original_goal}\n{memory.format_for_prompt()}\n{task_context}"
    selected_label, reason = select_vertical_fss_waypoint(
        client=state.llm_client, prepared=prepared, objective=objective,
        local_target=step_decision.waypoint_target, task_context=text,
    )
    view, candidate, projection = prepared.selected(candidate_label=selected_label)
    waypoint = vertical_fss_visual_waypoint(
        selected_view=view, world_candidate=candidate, projected_candidate=projection,
    )
    return VerticalTransitionWaypointPlan(
        step_decision=step_decision,
        visual_action=VisualActionPointDecision(
            point_2d=projection.point_2d, target=step_decision.waypoint_target,
            reasoning=reason,
        ),
        waypoint_verification=VisualWaypointVerificationDecision(
            verdict="execute", critique="Selected visible, height-connected FSS label: " + reason,
        ),
        waypoint=waypoint, selected_view=view,
        attempts=[{"algorithm": "detected_stair_fss", "candidate_label": selected_label, "candidate_set": prepared.to_dict()}],
    )


def _vertical_transition_va_history_entry(
    *,
    transition_index: int,
    waypoint_plan: VerticalTransitionWaypointPlan,
    move_result: ActionResult,
    post_move_height_m: float | None = None,
    post_move_height_delta_m: float | None = None,
    failure_reason: str = "",
) -> dict[str, object]:
    waypoint = waypoint_plan.waypoint
    entry: dict[str, object] = {
        "transition_index": int(transition_index),
        "selected_view": {
            "angle_deg": int(waypoint_plan.selected_view.angle_deg),
            "obs_id": str(waypoint_plan.selected_view.obs_id),
        },
        "step_decision": waypoint_plan.step_decision.to_dict(),
        "waypoint_attempts": [
            {"algorithm": item["algorithm"], "candidate_label": item["candidate_label"]}
            for item in waypoint_plan.attempts
        ],
        "final_visual_action": waypoint_plan.visual_action.to_dict(),
        "waypoint_verification": waypoint_plan.waypoint_verification.to_dict(),
        "grounded_waypoint": {
            "goal_xy": [float(waypoint.goal_xy[0]), float(waypoint.goal_xy[1])],
            "goal_yaw": float(waypoint.goal_yaw),
            "path_xy": [[float(x), float(y)] for x, y in waypoint.path_xy],
            "depth_m": float(waypoint.depth_m),
            "raw_world_z": float(waypoint.raw_world_z),
        },
        "move_result": {
            "ok": bool(move_result.ok),
            "message": str(move_result.message),
            "path_point_count": len(list(move_result.data.get("path_xy", []))),
        },
    }
    if post_move_height_m is not None:
        entry["post_move_height_m"] = float(post_move_height_m)
    if post_move_height_delta_m is not None:
        entry["post_move_height_delta_m"] = float(post_move_height_delta_m)
    if str(failure_reason).strip() != "":
        entry["failure_reason"] = str(failure_reason)
    return entry


def _vertical_transition_directional_height_progress(
    *,
    direction: str,
    observed_heights: list[float],
) -> float:
    if observed_heights == []:
        return 0.0
    start_height = float(observed_heights[0])
    if direction == "down":
        return max(0.0, start_height - min(float(height) for height in observed_heights))
    return max(0.0, max(float(height) for height in observed_heights) - start_height)


def _panorama_anchor_height(*, cache, panorama: PanoramaCaptureResult) -> float:
    anchor_obs = cache.get_observation(str(panorama.anchor_obs_id)).observation
    return float(anchor_obs.pose.z)


def _vertical_panorama_from_current_step(
    step: "NavClawStepState",
) -> PanoramaCaptureResult | None:
    raw_views = list(getattr(step, "panorama_views", []))
    if raw_views == []:
        return None
    views: list[PanoramaView] = []
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            return None
        try:
            view = PanoramaView(
                angle_deg=int(raw_view["angle_deg"]),
                obs_id=str(raw_view["obs_id"]),
                rgb_id=str(raw_view["rgb_id"]),
                depth_id=str(raw_view["depth_id"]),
                pose=dict(raw_view.get("pose", {})),
                yaw_debug=dict(raw_view.get("yaw_debug", {})),
            )
        except (KeyError, TypeError, ValueError):
            return None
        views.append(view)
    angle_to_obs_id = {
        int(angle): str(obs_id)
        for angle, obs_id in dict(getattr(step, "angle_to_obs_id", {})).items()
    }
    if angle_to_obs_id == {}:
        angle_to_obs_id = {int(view.angle_deg): str(view.obs_id) for view in views}
    anchor_obs_id = str(getattr(step, "anchor_obs_id", "") or "")
    if anchor_obs_id == "":
        anchor_obs_id = str(angle_to_obs_id.get(0, views[0].obs_id))
    current_obs_id = str(getattr(step, "current_obs_id", "") or anchor_obs_id)
    return PanoramaCaptureResult(
        views=views,
        angle_to_obs_id=angle_to_obs_id,
        anchor_obs_id=anchor_obs_id,
        current_obs_id=current_obs_id,
        angle_debug=dict(getattr(step, "panorama_angle_debug", {})),
    )


def _vertical_panorama_from_visual_context(
    visual_context: VisualActionContext,
) -> PanoramaCaptureResult:
    views = [
        PanoramaView(
            angle_deg=int(view.angle_deg),
            obs_id=str(view.obs_id),
            rgb_id=str(view.rgb_id),
            depth_id=str(view.depth_id),
            pose=dict(view.pose),
            yaw_debug={},
        )
        for view in visual_context.views
    ]
    if views == []:
        raise ValueError("vertical transition planning node has no panorama views")
    angle_to_obs_id = {
        int(view.angle_deg): str(view.obs_id) for view in views
    }
    anchor_obs_id = str(angle_to_obs_id.get(0, views[0].obs_id))
    return PanoramaCaptureResult(
        views=views,
        angle_to_obs_id=angle_to_obs_id,
        anchor_obs_id=anchor_obs_id,
        current_obs_id=anchor_obs_id,
        angle_debug={},
    )


def _capture_vertical_panorama(
    *,
    context: "NavClawAgentContext",
    state: "NavClawAgentState",
) -> PanoramaCaptureResult:
    return _capture_panorama(
        env=context.env,
        cache=state.cache,
        global_exploration=None,
        landmark_controller=None,
        panorama_config=context.panorama_config,
    )


def _visual_context_from_panorama(
    *,
    current_node_id: str,
    panorama: PanoramaCaptureResult,
) -> VisualActionContext:
    views = [
        VisualViewContext(
            angle_deg=int(view.angle_deg),
            obs_id=str(view.obs_id),
            rgb_id=str(view.rgb_id),
            depth_id=str(view.depth_id),
            pose=dict(view.pose),
        )
        for view in panorama.views
    ]
    return VisualActionContext(
        current_node_id=str(current_node_id),
        views=views,
    )


def _execute_vt_move(
    *,
    state: "NavClawAgentState",
    move_call: ActionCall,
    vt_exploration: _NoopVerticalTransitionExploration,
) -> ActionResult:
    previous_exploration = state.action_executor.exploration
    previous_handler = state.action_executor.step_observation_handler
    previous_enabled_getter = state.action_executor.step_observation_enabled_getter
    try:
        state.action_executor.exploration = vt_exploration
        state.action_executor.step_observation_handler = None
        state.action_executor.step_observation_enabled_getter = None
        return state.action_executor.execute(move_call)
    finally:
        state.action_executor.exploration = previous_exploration
        state.action_executor.step_observation_handler = previous_handler
        state.action_executor.step_observation_enabled_getter = previous_enabled_getter


def _extend_unique_obs_ids(target: list[str], raw_obs_ids: object) -> None:
    if not isinstance(raw_obs_ids, list):
        return
    for obs_id in raw_obs_ids:
        obs_id_text = str(obs_id).strip()
        if obs_id_text == "":
            continue
        if obs_id_text not in target:
            target.append(obs_id_text)


def _extend_path_xy(target: list[tuple[float, float]], raw_path_xy: object) -> None:
    if not isinstance(raw_path_xy, list):
        return
    for point in raw_path_xy:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        normalized = (float(point[0]), float(point[1]))
        if target != [] and target[-1] == normalized:
            continue
        target.append(normalized)


def _record_vertical_transition_node_move(
    *,
    state: "NavClawAgentState",
    step_id: int,
    before_node_id: str,
    after_node_id: str,
    direction: str,
    rgb_history_obs_ids: list[str],
    planning_node_id: str,
    backtrack_contexts: list[dict[str, object]],
    subgoal_id: str | None = None,
    subgoal_attempt: int | None = None,
) -> None:
    if str(before_node_id).strip() == "" or str(after_node_id).strip() == "":
        return
    if str(before_node_id) == str(after_node_id):
        return
    record = {
        "step_id": int(step_id),
        "from_node": str(before_node_id),
        "to_node": str(after_node_id),
        "move_mode": f"vertical_transition_{direction}",
        "reason": f"vertical_transition_{direction}",
        "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
    }
    if subgoal_id is not None:
        record.update(subgoal_id=subgoal_id, subgoal_attempt=subgoal_attempt)
    if backtrack_contexts != []:
        record["backtrack_reference_node_id"] = str(planning_node_id)
        record["backtrack_contexts"] = [
            deepcopy(item) for item in backtrack_contexts
        ]
    history = list(getattr(state, "node_move_history", []))
    history.append(record)
    state.node_move_history = history


def _complete_vertical_transition(
    *,
    graph: Graph,
    cache,
    before_node_id: str,
    before_floor_id: str,
    before_floor_height: float,
    direction: str,
    floor_match_threshold_m: float,
    final_panorama: PanoramaCaptureResult,
    rgb_history_obs_ids: list[str],
    path_xy: list[tuple[float, float]],
    destination_floor_reached: bool,
) -> dict[str, object]:
    anchor_obs_id = str(final_panorama.anchor_obs_id)
    anchor_observation = cache.get_observation(anchor_obs_id).observation
    after_height = float(anchor_observation.pose.z)
    if destination_floor_reached:
        after_floor, created_floor = _resolve_vertical_transition_floor(
            graph=graph,
            before_floor_id=before_floor_id,
            after_height=after_height,
            floor_match_threshold_m=float(floor_match_threshold_m),
        )
    else:
        after_floor, created_floor = graph.floors[before_floor_id], False
    after_node_id = _create_place_node(
        graph=graph,
        cache=cache,
        anchor_obs_id=anchor_obs_id,
        obs_ids=final_panorama.obs_ids,
        floor_id=str(after_floor.id),
    )
    relation = "stairs_up" if direction == "up" else "stairs_down"
    edge = graph.add_edge(
        src_id=str(before_node_id),
        dst_id=str(after_node_id),
        relation=relation,
        rgb_history_obs_ids=[str(obs_id) for obs_id in rgb_history_obs_ids],
        path_xy=[(float(x), float(y)) for x, y in path_xy],
    )
    return {
        "before_node_id": str(before_node_id),
        "before_floor_id": str(before_floor_id),
        "before_floor_height": float(before_floor_height),
        "after_node_id": str(after_node_id),
        "after_floor_id": str(after_floor.id),
        "after_floor_height": float(after_floor.height),
        "created_floor": bool(created_floor),
        "destination_floor_reached": bool(destination_floor_reached),
        "edge_id": str(edge.id),
        "edge_relation": str(edge.relation),
        "current_obs_id_after": anchor_obs_id,
        "final_panorama_obs_ids": final_panorama.obs_ids,
        "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
        "path_xy": [[float(x), float(y)] for x, y in path_xy],
        "register_frontiers_on_reuse": True,
    }


def _resolve_vertical_transition_floor(
    *,
    graph: Graph,
    before_floor_id: str,
    after_height: float,
    floor_match_threshold_m: float,
) -> tuple[Floor, bool]:
    threshold = float(floor_match_threshold_m)
    matches: list[tuple[float, str, Floor]] = []
    for floor_id, floor in graph.floors.items():
        if str(floor_id) == str(before_floor_id):
            continue
        floor_height = float(floor.height)
        distance = abs(floor_height - float(after_height))
        if distance <= threshold:
            matches.append((float(distance), str(floor.id), floor))
    if matches != []:
        matches.sort(key=lambda item: (item[0], item[1]))
        matched_floor = matches[0][2]
        matched_floor.height = float(after_height)
        return matched_floor, False
    return graph.add_floor(height=float(after_height)), True


def _current_episode_step(context: "NavClawAgentContext") -> int:
    info = context.env.current_episode_info()
    return int(info.get("pointnav_step_total", 0))
