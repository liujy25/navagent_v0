from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from navclaw.agent.node_moves import attach_backtrack_edge_knowledge
from navclaw.agent.visual_action_context import (
    VisualActionContext,
    VisualViewContext,
    build_visual_action_context_for_node,
)
from navclaw.agent.visual_grounding import VisualWaypoint, ground_visual_waypoint
from navclaw.agent.visual_policy import (
    decide_visual_action_point,
    verify_vertical_transition_waypoint,
)
from navclaw.agent.visual_policy_decisions import (
    VisualActionPointDecision,
    VisualWaypointVerificationDecision,
)
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
    from navclaw.runtime.cache import RuntimeCache


DEFAULT_MAX_VERTICAL_TRANSITION_STEPS = 8
DEFAULT_VERTICAL_FLOOR_MATCH_THRESHOLD_M = 0.8
VERTICAL_COMPLETION_MIN_PROGRESS_M = 0.3
VERTICAL_WAYPOINT_VERIFICATION_MAX_ATTEMPTS = 4
VERTICAL_WAYPOINT_GEOMETRY_DIRECTION_TOLERANCE_M = 0.15
VERTICAL_WAYPOINT_MAX_LOCAL_XY_DISTANCE_M = 2.0
VERTICAL_STEP_REPLAN_MAX_ATTEMPTS = 4


class _NoopVerticalTransitionExploration:
    def observe_raw_observation(self, observation: object) -> None:
        return None


class VerticalTransitionStepReplan(ValueError):
    def __init__(self, message: str, *, feedback: dict[str, object]) -> None:
        super().__init__(message)
        self.feedback = feedback


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
    max_steps = int(
        action.args.get(
            "max_steps",
            getattr(context.args, "max_vertical_transition_steps", DEFAULT_MAX_VERTICAL_TRANSITION_STEPS),
        )
    )
    if max_steps <= 0:
        raise ValueError(f"max_vertical_transition_steps must be positive, got {max_steps}")
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
        "max_steps": int(max_steps),
        "floor_match_threshold_m": float(floor_match_threshold_m),
        "completion_min_progress_m": float(VERTICAL_COMPLETION_MIN_PROGRESS_M),
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
                instruction=_vertical_transition_instruction(direction),
                visual_context=visual_context,
                initial_observation=(move_count == 0),
                movement_rgb_history_blocks=vt_rgb_history_blocks,
                current_retry_feedback=current_retry_feedback,
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
        attempt_payload["vertical_transition_min_completion_progress_m"] = float(
            VERTICAL_COMPLETION_MIN_PROGRESS_M
        )
        if step_decision.transition_status == "complete":
            if height_progress_m < float(VERTICAL_COMPLETION_MIN_PROGRESS_M):
                feedback = {
                    "type": "completion_rejected",
                    "requested_direction": str(direction),
                    "reason": "height_progress_below_min_completion_progress",
                    "thought": str(step_decision.thought),
                    "height_progress_m": float(height_progress_m),
                    "min_completion_progress_m": float(VERTICAL_COMPLETION_MIN_PROGRESS_M),
                }
                step_replan_count += 1
                feedback["transition_index"] = int(move_count)
                feedback["step_replan_attempt_index"] = int(step_replan_count)
                current_retry_feedback = feedback
                attempt_payload["transition_completion_forced_continue_reason"] = (
                    "height_progress_below_min_completion_progress"
                )
                attempt_payload["replanned_by_vt_step_planner"] = True
                attempt_payload["step_replan_feedback"] = feedback
                attempts.append(attempt_payload)
                if step_replan_count >= VERTICAL_STEP_REPLAN_MAX_ATTEMPTS:
                    failure_reason = (
                        "vt_step_replan_exhausted:"
                        "completion_rejected_height_progress_below_min_completion_progress"
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
                )
            except ValueError as exc:
                failure_reason = f"floor_resolution_failed:{exc}"
                attempt_payload["floor_resolution_failure"] = str(exc)
                break
            return
        if step_decision.transition_status == "fail":
            failure_reason = f"step_planner_failed:{step_decision.thought}"
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break

        if move_count >= max_steps:
            failure_reason = "vertical_transition_exhausted"
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break

        try:
            waypoint_plan = _plan_vertical_transition_waypoint(
                state=state,
                direction=direction,
                visual_context=visual_context,
                step_decision=step_decision,
                current_height_m=current_height,
            )
        except VerticalTransitionStepReplan as exc:
            step_replan_count += 1
            feedback = dict(exc.feedback)
            feedback["rejected_options"] = _merge_vertical_transition_rejected_options(
                current_retry_feedback,
                feedback,
            )
            feedback["transition_index"] = int(move_count)
            feedback["step_replan_attempt_index"] = int(step_replan_count)
            current_retry_feedback = feedback
            attempt_payload["replanned_by_vt_step_planner"] = True
            attempt_payload["step_replan_feedback"] = feedback
            attempts.append(attempt_payload)
            if step_replan_count >= VERTICAL_STEP_REPLAN_MAX_ATTEMPTS:
                failure_reason = f"vt_step_replan_exhausted:{exc}"
                break
            continue
        except ValueError as exc:
            failure_reason = f"waypoint_plan_failed:{exc}"
            attempt_payload["ok"] = False
            attempt_payload["failure_reason"] = failure_reason
            attempts.append(attempt_payload)
            break
        final_waypoint_verification = waypoint_plan.waypoint_verification
        last_waypoint_plan = waypoint_plan

        move_call = ActionCall(
            action="move_along_path",
            args={
                "path_xy": [[float(x), float(y)] for x, y in waypoint_plan.waypoint.path_xy],
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
        attempt_payload["vertical_transition_min_completion_progress_m"] = float(
            VERTICAL_COMPLETION_MIN_PROGRESS_M
        )
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
        message=(
            f"Completed vertical transition {direction} from "
            f"{before_floor_id} to {completion['after_floor_id']}."
        ),
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


def _plan_vertical_transition_waypoint(
    *,
    state: "NavClawAgentState",
    direction: str,
    visual_context: VisualActionContext,
    step_decision: VerticalTransitionStepDecision,
    current_height_m: float,
) -> VerticalTransitionWaypointPlan:
    goal_text = _vertical_transition_goal_text(direction)
    if step_decision.selected_angle_deg is None:
        raise ValueError("vertical_transition_step_missing_angle")
    if step_decision.waypoint_target == "":
        raise ValueError("vertical_transition_step_missing_waypoint_target")
    selected_view = visual_context.view_for_angle(step_decision.selected_angle_deg)

    revision_feedback = ""
    rejected_point_2d: tuple[float, float] | None = None
    attempts: list[dict[str, object]] = []
    for attempt_index in range(VERTICAL_WAYPOINT_VERIFICATION_MAX_ATTEMPTS):
        visual_action = decide_visual_action_point(
            client=state.llm_client,
            cache=state.cache,
            goal_text=goal_text,
            selected_view=selected_view,
            waypoint_target=str(step_decision.waypoint_target),
            revision_feedback=revision_feedback,
            rejected_point_2d=rejected_point_2d,
            include_graph_context=visual_context.graph_context_visible,
        )
        attempt_payload: dict[str, object] = {
            "attempt_index": int(attempt_index),
            "visual_action": visual_action.to_dict(),
        }
        if visual_action.failure_reason != "":
            attempt_payload["failure_reason"] = str(visual_action.failure_reason)
            attempts.append(attempt_payload)
            raise ValueError(str(visual_action.failure_reason))
        if visual_action.point_2d is None:
            attempt_payload["failure_reason"] = "visual_action_missing_point_2d"
            attempts.append(attempt_payload)
            raise ValueError("visual_action_missing_point_2d")

        grounded_waypoint: VisualWaypoint | None = None
        geometry_warning = ""
        try:
            grounded_waypoint = ground_visual_waypoint(
                cache=state.cache,
                exploration=state.global_exploration_for_floor(str(state.system.current_floor_id)),
                obs_id=selected_view.obs_id,
                angle_deg=selected_view.angle_deg,
                point_2d=visual_action.point_2d,
                waypoint_target=str(step_decision.waypoint_target),
                target=str(visual_action.target),
                use_navigation_snap=False,
            )
            attempt_payload["grounded_waypoint_preview"] = grounded_waypoint.to_dict()
            geometry_warning = _vertical_waypoint_geometry_warning(
                direction=direction,
                current_height_m=current_height_m,
                waypoint=grounded_waypoint,
            )
            if geometry_warning != "":
                attempt_payload["geometry_warning"] = geometry_warning
        except ValueError as exc:
            attempt_payload["grounding_failure"] = str(exc)

        if grounded_waypoint is not None:
            local_distance_m = _vertical_waypoint_local_xy_distance_m(
                cache=state.cache,
                waypoint=grounded_waypoint,
            )
            attempt_payload["local_xy_distance_m"] = float(local_distance_m)
            if local_distance_m > float(VERTICAL_WAYPOINT_MAX_LOCAL_XY_DISTANCE_M):
                attempt_payload["failure_reason"] = "vertical_waypoint_too_far"
                attempts.append(attempt_payload)
                revision_feedback = (
                    "The selected point is beyond one local stair-transition move. "
                    "Select a nearer reachable point on the same route, using the "
                    "first stable floor immediately beyond the final tread when it "
                    "is visible."
                )
                rejected_point_2d = visual_action.point_2d
                if (
                    attempt_index + 1
                    >= VERTICAL_WAYPOINT_VERIFICATION_MAX_ATTEMPTS
                ):
                    raise VerticalTransitionStepReplan(
                        "vertical_waypoint_too_far",
                        feedback={
                            "type": "previous_waypoint_rejected",
                            "requested_direction": str(direction),
                            "rejected_options": [
                                {
                                    "angle_deg": int(selected_view.angle_deg),
                                    "waypoint_target": str(
                                        step_decision.waypoint_target
                                    ),
                                }
                            ],
                            "feedback": revision_feedback,
                            "selected_waypoint_geometry": {
                                "local_xy_distance_m": float(local_distance_m),
                                "max_local_xy_distance_m": float(
                                    VERTICAL_WAYPOINT_MAX_LOCAL_XY_DISTANCE_M
                                ),
                            },
                        },
                    )
                continue

        verification = verify_vertical_transition_waypoint(
            client=state.llm_client,
            cache=state.cache,
            instruction=goal_text,
            selected_view=selected_view,
            waypoint_target=str(step_decision.waypoint_target),
            visual_action=visual_action,
            geometry_warning=geometry_warning,
            allow_revise_verdict=attempt_index + 1 < VERTICAL_WAYPOINT_VERIFICATION_MAX_ATTEMPTS,
        )
        attempt_payload["verification"] = verification.to_dict()
        attempts.append(attempt_payload)
        if verification.verdict == "execute":
            waypoint = grounded_waypoint
            if waypoint is None:
                waypoint = ground_visual_waypoint(
                    cache=state.cache,
                    exploration=state.global_exploration_for_floor(str(state.system.current_floor_id)),
                    obs_id=selected_view.obs_id,
                    angle_deg=selected_view.angle_deg,
                    point_2d=visual_action.point_2d,
                    waypoint_target=str(step_decision.waypoint_target),
                    target=str(visual_action.target),
                    use_navigation_snap=False,
                )
            return VerticalTransitionWaypointPlan(
                step_decision=step_decision,
                visual_action=visual_action,
                waypoint_verification=verification,
                waypoint=waypoint,
                selected_view=selected_view,
                attempts=attempts,
            )
        if verification.verdict == "fallback_navigation":
            raise VerticalTransitionStepReplan(
                "waypoint_verifier_fallback_navigation",
                feedback=_vertical_transition_step_replan_feedback(
                    direction=direction,
                    current_height_m=current_height_m,
                    selected_view=selected_view,
                    step_decision=step_decision,
                    visual_action=visual_action,
                    verification=verification,
                    waypoint=grounded_waypoint,
                    geometry_warning=geometry_warning,
                ),
            )
        if verification.verdict != "revise_same_view":
            raise ValueError(f"waypoint_verifier_{verification.verdict}:{verification.critique}")
        revision_feedback = str(verification.critique)
        rejected_point_2d = visual_action.point_2d
    raise ValueError("vertical_waypoint_verification_exhausted")


def _vertical_waypoint_local_xy_distance_m(
    *,
    cache: "RuntimeCache",
    waypoint: VisualWaypoint,
) -> float:
    observation = cache.get_observation(str(waypoint.obs_id)).observation
    return float(
        math.hypot(
            float(waypoint.raw_world_xy[0]) - float(observation.pose.x),
            float(waypoint.raw_world_xy[1]) - float(observation.pose.y),
        )
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
        "va_attempts": _compact_va_attempts(waypoint_plan.attempts),
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


def _vertical_transition_step_replan_feedback(
    *,
    direction: str,
    current_height_m: float,
    selected_view: VisualViewContext,
    step_decision: VerticalTransitionStepDecision,
    visual_action: VisualActionPointDecision,
    verification: VisualWaypointVerificationDecision,
    waypoint: VisualWaypoint | None,
    geometry_warning: str,
) -> dict[str, object]:
    feedback: dict[str, object] = {
        "type": "previous_waypoint_rejected",
        "rejected_options": [
            {
                "angle_deg": int(selected_view.angle_deg),
                "waypoint_target": str(step_decision.waypoint_target),
            }
        ],
        "requested_direction": str(direction),
        "previous_waypoint": {
            "angle_deg": int(selected_view.angle_deg),
            "obs_id": str(selected_view.obs_id),
            "waypoint_target": str(step_decision.waypoint_target),
            "selected_point_target": str(visual_action.target),
            "point_2d": (
                None
                if visual_action.point_2d is None
                else [float(visual_action.point_2d[0]), float(visual_action.point_2d[1])]
            ),
        },
        "validation": {
            "verdict": str(verification.verdict),
            "critique": str(verification.critique),
        },
        "feedback": str(verification.critique),
    }
    if str(geometry_warning).strip() != "":
        feedback["geometry_warning"] = str(geometry_warning)
    if waypoint is not None:
        waypoint_height = float(waypoint.raw_world_z)
        height_delta = waypoint_height - float(current_height_m)
        feedback["selected_waypoint_geometry"] = {
            "raw_world_z": waypoint_height,
            "current_height_m": float(current_height_m),
            "height_delta_m": float(height_delta),
        }
    return feedback


def _merge_vertical_transition_rejected_options(
    previous_feedback: dict[str, object] | None,
    current_feedback: dict[str, object],
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for feedback in (previous_feedback, current_feedback):
        if not isinstance(feedback, dict):
            continue
        for raw_option in list(feedback.get("rejected_options", [])):
            if not isinstance(raw_option, dict) or raw_option.get("angle_deg") is None:
                continue
            target = str(raw_option.get("waypoint_target", "")).strip()
            if target == "":
                continue
            option = (int(raw_option["angle_deg"]), target)
            if option in seen:
                continue
            seen.add(option)
            options.append(
                {
                    "angle_deg": int(option[0]),
                    "waypoint_target": str(option[1]),
                }
            )
    return options


def _compact_va_attempts(attempts: list[dict[str, object]]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for attempt in attempts:
        item: dict[str, object] = {
            "attempt_index": int(attempt.get("attempt_index", 0)),
        }
        visual_action = attempt.get("visual_action")
        if isinstance(visual_action, dict):
            item["visual_action"] = {
                "point_2d": visual_action.get("point_2d"),
                "target": str(visual_action.get("target", "")),
                "failure_reason": str(visual_action.get("failure_reason", "")),
            }
        verification = attempt.get("verification")
        if isinstance(verification, dict):
            item["verification"] = {
                "verdict": str(verification.get("verdict", "")),
                "critique": str(verification.get("critique", "")),
                "transition_complete_after_execution": bool(
                    verification.get("transition_complete_after_execution", False)
                ),
            }
        geometry_warning = str(attempt.get("geometry_warning", "")).strip()
        if geometry_warning != "":
            item["geometry_warning"] = geometry_warning
        failure_reason = str(attempt.get("failure_reason", "")).strip()
        if failure_reason != "":
            item["failure_reason"] = failure_reason
        compact.append(item)
    return compact


def _vertical_waypoint_geometry_warning(
    *,
    direction: str,
    current_height_m: float,
    waypoint: VisualWaypoint,
) -> str:
    waypoint_height = float(waypoint.raw_world_z)
    height_delta = waypoint_height - float(current_height_m)
    tolerance = float(VERTICAL_WAYPOINT_GEOMETRY_DIRECTION_TOLERANCE_M)
    if direction == "down" and height_delta > tolerance:
        return (
            "Geometry estimates the selected waypoint is higher than the current position "
            f"(waypoint_z={waypoint_height:.3f}m, current_z={float(current_height_m):.3f}m, "
            f"delta={height_delta:+.3f}m). The requested vertical transition is Go down stairs, "
            "so the waypoint may be on an upward or opposite-direction path. "
            "Reconsider the selected direction or local waypoint region for the requested downward transition."
        )
    if direction == "up" and height_delta < -tolerance:
        return (
            "Geometry estimates the selected waypoint is lower than the current position "
            f"(waypoint_z={waypoint_height:.3f}m, current_z={float(current_height_m):.3f}m, "
            f"delta={height_delta:+.3f}m). The requested vertical transition is Go upstairs, "
            "so the waypoint may be on a downward or opposite-direction path. "
            "Reconsider the selected direction or local waypoint region for the requested upward transition."
        )
    return ""


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
) -> dict[str, object]:
    anchor_obs_id = str(final_panorama.anchor_obs_id)
    anchor_observation = cache.get_observation(anchor_obs_id).observation
    after_height = float(anchor_observation.pose.z)
    after_floor, created_floor = _resolve_vertical_transition_floor(
        graph=graph,
        before_floor_id=before_floor_id,
        after_height=after_height,
        floor_match_threshold_m=float(floor_match_threshold_m),
    )
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


def _vertical_transition_goal_text(direction: str) -> str:
    return _vertical_transition_instruction(direction)


def _vertical_transition_instruction(direction: str) -> str:
    if direction == "up":
        return "Go upstairs."
    return "Go downstairs."


def _current_episode_step(context: "NavClawAgentContext") -> int:
    info = context.env.current_episode_info()
    return int(info.get("pointnav_step_total", 0))
