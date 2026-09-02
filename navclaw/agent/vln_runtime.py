from __future__ import annotations

from navclaw.agent.node_moves import complete_pending_node_move
from navclaw.agent.state import NavClawAgentContext, NavClawAgentState, NavClawStepState
from navclaw.mapping.frontier_update import update_localmap_place_frontiers
from navclaw.mapping.place_step import prepare_localmap_place_step
from navclaw.runtime.timing import StepTimingRecorder


def robot_action_count(context: NavClawAgentContext) -> int:
    info = context.env.current_episode_info()
    return int(info.get("pointnav_step_total", info.get("action_step_total", 0)))


def begin_vln_step(
    *,
    state: NavClawAgentState,
    context: NavClawAgentContext,
    place_step_index: int,
) -> NavClawStepState:
    reuse = state.reuse_current_place_next_step
    state.reuse_current_place_next_step = None
    return NavClawStepState(
        place_step_index=int(place_step_index),
        timing=StepTimingRecorder(env_step_start=robot_action_count(context)),
        place_reuse_state=reuse,
    )


def refresh_vln_state(
    *,
    context: NavClawAgentContext,
    state: NavClawAgentState,
    step: NavClawStepState,
) -> None:
    floor_id = str(state.system.current_floor_id)
    state.ensure_floor_runtime_state(floor_id, context.global_bev_kwargs)
    global_exploration = state.global_exploration_for_floor(floor_id)
    frontier_filter = state.frontier_filter_exploration_for_floor(floor_id)
    state.action_executor.exploration = global_exploration

    place_step = prepare_localmap_place_step(
        env=context.env,
        cache=state.cache,
        graph=state.graph,
        global_exploration=global_exploration,
        landmark_controller=state.landmark_controller,
        local_explorations_by_node_id=state.local_explorations_by_node_id,
        bev_map_kwargs=context.global_bev_kwargs,
        panorama_config=context.panorama_config,
        current_place_node_id=state.current_place_node_id,
        current_floor_id=floor_id,
        previous_place_node_id=state.previous_place_node_id,
        place_reuse_state=step.place_reuse_state,
        timing=step.timing,
        current_step_count=lambda: robot_action_count(context),
    )
    step.place_step = place_step
    state.current_place_node_id = str(place_step.current_place_node_id)
    step.current_place_node_id = str(place_step.current_place_node_id)
    step.anchor_obs_id = str(place_step.anchor_obs_id)
    state.current_obs_id = str(place_step.current_obs_id)
    step.current_obs_id = str(place_step.current_obs_id)
    step.panorama_obs_ids = list(place_step.panorama_obs_ids)
    step.angle_to_obs_id = dict(place_step.angle_to_obs_id)
    step.panorama_views = [view.to_dict() for view in place_step.panorama_views]
    step.panorama_angle_debug = dict(place_step.panorama_angle_debug)
    step.reused_place_obs_ids = list(place_step.reused_place_obs_ids)
    step.current_projection_obs_ids = list(place_step.current_projection_obs_ids)
    step.local_exploration = place_step.local_exploration
    step.reachable_candidates = list(place_step.reachable_candidates)

    complete_pending_node_move(state=state, step=step)
    if step.local_exploration is None:
        raise ValueError("place refresh did not produce a local exploration map")
    frontier_filter.merge_raw_layers_from_local_exploration(
        step.local_exploration,
        obs_ids=step.current_projection_obs_ids,
    )
    step.frontier_update = update_localmap_place_frontiers(
        graph=state.graph,
        cache=state.cache,
        global_exploration=frontier_filter,
        local_exploration=step.local_exploration,
        place_reuse_state=step.place_reuse_state,
        frontier_records=state.frontier_records_for_floor(floor_id),
        overlay_records=state.overlay_records_for_floor(floor_id),
        local_explorations_by_node_id=state.local_explorations_by_node_id,
        current_place_node_id=str(step.current_place_node_id),
        current_projection_obs_ids=step.current_projection_obs_ids,
        current_projection_angle_to_obs_id=step.angle_to_obs_id,
        reachable_candidates=step.reachable_candidates,
        candidate_dedup_radius_m=float(context.args.candidate_dedup_radius),
    )
