from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from navclaw.agent.action_execution import LocalmapActionExecutor
from navclaw.agent.state import NavClawAgentContext, NavClawAgentState
from navclaw.agent.visual_navigation import (
    ensure_current_node_summary_for_visual_policy,
)
from navclaw.agent.visual_policy import ensure_task_progress_memory
from navclaw.agent.visual_step_execution import (
    run_pending_stop_confirmation,
    run_visual_waypoint_decision,
)
from navclaw.agent.vln_runtime import begin_vln_step, refresh_vln_state, robot_action_count
from navclaw.config.vln_runtime import VlnRuntimeConfig
from navclaw.env.interface import EnvInterface
from navclaw.graph.graph import Graph
from navclaw.llm.client import LLMClient
from navclaw.mapping.exploration.bev_map import Map1Map2BEVMap
from navclaw.mapping.exploration.manager import ExplorationManager
from navclaw.perception.detectors.yoloworld_local import YOLOWorldLocalDetector
from navclaw.perception.goal import vln_instruction_goal_spec
from navclaw.perception.landmark_buffer import LANDMARK_BUFFER_MERGE_DISTANCE_M
from navclaw.perception.landmark_detection import (
    LANDMARK_DETECTOR_BOX_THRESHOLD,
    LANDMARK_GEOMETRY_BBOX_CENTER,
    LandmarkDetectionController,
)
from navclaw.runtime.cache import RuntimeCache
from navclaw.runtime.panorama_config import PanoramaConfig, robot_panorama_config
from navclaw.runtime.run_trace import RunTrace
from navclaw.system_state import GoalState, SystemState


@dataclass(frozen=True)
class VlnEpisodeResult:
    instruction: str
    termination_reason: str
    declared_done: bool
    place_steps: int
    robot_actions: int
    final_pose: dict[str, float] | None
    llm_usage: dict[str, object]
    action_trace: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "termination_reason": self.termination_reason,
            "declared_done": self.declared_done,
            "place_steps": self.place_steps,
            "robot_actions": self.robot_actions,
            "final_pose": self.final_pose,
            "llm_usage": self.llm_usage,
            "action_trace": self.action_trace,
        }


def _initial_floor_height(reset_payload: dict[str, Any]) -> float:
    pose = reset_payload.get("pose")
    return float(pose.get("z", 0.0)) if isinstance(pose, dict) else 0.0


def _landmark_categories(client: LLMClient, instruction: str) -> list[str]:
    response = client._create_visual_json_completion(
        call_name="vln_landmark_detection_list",
        system_prompt=(
            "Choose visual landmark categories for a VLN navigation instruction. "
            "Return JSON only."
        ),
        user_prompt=f"""
Instruction:
{instruction}

Return concrete visible object or fixture categories useful for following the
instruction. Categories must have stable image bounding boxes. Do not include
rooms, floors, stairs, passages, openings, routes, or directions.

Return JSON only:
{{"landmark_categories": ["chair", "table"]}}
""".strip(),
        max_new_tokens=1024,
        token_field="max_completion_tokens",
        retry_count=2,
        client_kind="la",
    )
    categories: list[str] = []
    seen: set[str] = set()
    raw_categories = response.get("landmark_categories", [])
    for item in raw_categories if isinstance(raw_categories, list) else []:
        category = str(item).strip().lower().replace("_", " ")
        if category and category not in seen:
            seen.add(category)
            categories.append(category)
    return categories


def _configure_landmarks(
    controller: LandmarkDetectionController,
    categories: list[str],
) -> None:
    if not categories:
        controller.enabled = False
        return
    controller.configure_landmark_class_map(
        {
            category: {
                "class_name": category,
                "threshold": float(LANDMARK_DETECTOR_BOX_THRESHOLD),
                "merge_distance_m": float(LANDMARK_BUFFER_MERGE_DISTANCE_M),
                "geometry": LANDMARK_GEOMETRY_BBOX_CENTER,
            }
            for category in categories
        },
        reset_buffer=True,
        confirmation_min_observations=1,
        auto_confirm_without_llm=True,
    )
    controller.llm_client = None
    controller.enabled = True


def _build_state(
    *,
    env: EnvInterface,
    instruction: str,
    initial_floor_height: float,
    llm_client: LLMClient,
    detector: YOLOWorldLocalDetector,
    config: VlnRuntimeConfig,
) -> NavClawAgentState:
    graph = Graph()
    graph.set_floor_height("floor_0", initial_floor_height)
    cache = RuntimeCache()
    exploration = ExplorationManager(
        bev_map=Map1Map2BEVMap(**config.global_bev_kwargs())
    )
    action_executor = LocalmapActionExecutor(
        env=env,
        graph=graph,
        cache=cache,
        detector=detector,
        exploration=exploration,
    )
    landmark_controller = LandmarkDetectionController(
        detector=detector,
        cache_provider=lambda: cache,
        history_len_provider=lambda: len(graph.iter_nodes()),
        llm_client=None,
    )
    landmark_controller.enabled = False

    def post_execute(_call: object, _result: object) -> None:
        landmark_controller.handle_post_action_execute(_call, _result)

    def store_movement_observation(observation: object, source: str) -> dict[str, object]:
        record = cache.store_observation(observation)
        return {
            "obs_id": record.id,
            "pose": observation.pose.to_dict(),
            "source": str(source),
        }

    action_executor.post_execute_handler = post_execute
    action_executor.step_observation_handler = store_movement_observation
    action_executor.step_observation_enabled_getter = lambda: False
    system = SystemState(
        goal=GoalState(target=instruction),
        graph=graph,
        landmark_controller=landmark_controller,
        cache=cache,
        current_floor_id="floor_0",
        current_floor_height=initial_floor_height,
    )
    return NavClawAgentState(
        system=system,
        run_trace=RunTrace(graph=graph),
        llm_client=llm_client,
        detector=detector,
        action_executor=action_executor,
        max_retrieve_rounds=config.max_retrieve_rounds,
        global_explorations_by_floor={"floor_0": exploration},
        frontier_records_by_floor={"floor_0": {}},
        overlay_records_by_floor={"floor_0": {}},
    )


def _usage_brief(client: LLMClient) -> dict[str, object]:
    usage = client.get_usage_summary()
    return {
        "model": usage.get("model"),
        "request_count": int(usage.get("request_count", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def run_vln_episode(
    *,
    env: EnvInterface,
    instruction: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    detector_model_path: str = "weights/yolov8x-world.pt",
    detector_device: str = "cuda",
    config: VlnRuntimeConfig | None = None,
    panorama_config: PanoramaConfig | None = None,
) -> dict[str, object]:
    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError("instruction must be non-empty")
    runtime = VlnRuntimeConfig() if config is None else config
    panorama = robot_panorama_config() if panorama_config is None else panorama_config
    env.health() if hasattr(env, "health") else None
    reset_payload = env.reset(instruction)
    client = LLMClient(model=model, api_key=api_key, base_url=base_url)
    client.reset_usage()
    detector = YOLOWorldLocalDetector(
        model_path=detector_model_path,
        device=detector_device,
        conf_threshold=LANDMARK_DETECTOR_BOX_THRESHOLD,
    )
    state = _build_state(
        env=env,
        instruction=instruction,
        initial_floor_height=_initial_floor_height(reset_payload),
        llm_client=client,
        detector=detector,
        config=runtime,
    )
    goal_spec = vln_instruction_goal_spec(instruction)
    context = NavClawAgentContext(
        args=runtime,
        env=env,
        panorama_config=panorama,
        global_bev_kwargs=runtime.global_bev_kwargs(),
        reset_payload=reset_payload,
        run_goal=instruction,
        goal_spec=goal_spec,
    )
    initialization = ensure_task_progress_memory(
        client=client,
        task_progress=state.system.memory.task_progress,
        goal_text=instruction,
        task_type="vln_instruction",
        cache=state.cache,
        visual_context=None,
    )
    state.pending_vln_task_progress_initialization = dict(initialization)
    _configure_landmarks(
        state.landmark_controller,
        _landmark_categories(client, instruction),
    )

    action_trace: list[dict[str, object]] = []
    completed_steps = 0
    try:
        for place_step_index in range(runtime.max_place_steps):
            step = begin_vln_step(
                state=state,
                context=context,
                place_step_index=place_step_index,
            )
            refresh_vln_state(context=context, state=state, step=step)
            ensure_current_node_summary_for_visual_policy(state=state, step=step)
            if not run_pending_stop_confirmation(context=context, state=state, step=step):
                run_visual_waypoint_decision(
                    context=context,
                    state=state,
                    step=step,
                    context_evidence_text="",
                )
            completed_steps += 1
            action_trace.append(
                {
                    "step": int(place_step_index),
                    "node_id": state.current_place_node_id,
                    "action": dict(step.agent_action),
                    "termination_reason": (
                        state.finalize_reason
                        if state.finalize_reason != "terminated"
                        else None
                    ),
                }
            )
            if state.system.is_done:
                break
        else:
            state.finalize_reason = "max_place_steps"
    except BaseException:
        try:
            env.stop()
        finally:
            env.finalize_run(reason="exception")
        raise

    if state.stop_result is None:
        state.stop_result = env.stop()
    env.finalize_run(reason=state.finalize_reason)
    pose_payload = state.stop_result.get("pose") if isinstance(state.stop_result, dict) else None
    final_pose = (
        {
            key: float(pose_payload[key])
            for key in ("x", "y", "z", "yaw")
            if key in pose_payload
        }
        if isinstance(pose_payload, dict)
        else None
    )
    return VlnEpisodeResult(
        instruction=instruction,
        termination_reason=state.finalize_reason,
        declared_done=state.finalize_reason == "done",
        place_steps=completed_steps,
        robot_actions=robot_action_count(context),
        final_pose=final_pose,
        llm_usage=_usage_brief(client),
        action_trace=action_trace,
    ).to_dict()
