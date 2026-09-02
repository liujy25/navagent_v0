from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from navclaw.perception.detectors.interface import DetectorInterface
    from navclaw.env.interface import EnvInterface
    from navclaw.mapping.exploration.bev_map import FrontierCandidate
    from navclaw.mapping.exploration.manager import ExplorationManager
    from navclaw.graph.graph import Graph
    from navclaw.llm.client import LLMClient
    from navclaw.agent.action_execution import LocalmapActionExecutor
    from navclaw.mapping.frontier_update import LocalmapFrontierUpdateResult
    from navclaw.perception.goal import GoalSpec
    from navclaw.perception.landmark_detection import LandmarkDetectionController
    from navclaw.mapping.place_step import LocalmapPlaceStepResult
    from navclaw.schemas import ActionCall, ActionResult
    from navclaw.types import (
        LocalmapFrontierRecord,
        LocalmapNavigationPlan,
        LocalmapOverlayRecord,
        LocalmapPlaceReuseState,
        LocalmapRouteTarget,
    )
    from navclaw.system_state import SystemState
    from navclaw.runtime.panorama_config import PanoramaConfig
    from navclaw.runtime.run_trace import RunTrace
    from navclaw.runtime.timing import StepTimingRecorder


@dataclass
class NavClawAgentContext:
    args: Any
    env: EnvInterface
    panorama_config: PanoramaConfig
    global_bev_kwargs: dict[str, Any]
    reset_payload: dict[str, Any]
    run_goal: str
    goal_spec: GoalSpec | None


@dataclass
class NavClawAgentState:
    system: SystemState
    run_trace: RunTrace
    llm_client: LLMClient
    detector: DetectorInterface
    action_executor: LocalmapActionExecutor
    max_retrieve_rounds: int = 8
    global_explorations_by_floor: dict[str, ExplorationManager] = field(default_factory=dict)
    frontier_filter_explorations_by_floor: dict[str, ExplorationManager] = field(default_factory=dict)
    frontier_records_by_floor: dict[str, dict[str, LocalmapFrontierRecord]] = field(default_factory=dict)
    overlay_records_by_floor: dict[str, dict[str, LocalmapOverlayRecord]] = field(default_factory=dict)
    local_explorations_by_node_id: dict[str, ExplorationManager] = field(default_factory=dict)
    _finalize_reason: str = "terminated"
    previous_place_node_id: str | None = None
    current_obs_id: str | None = None
    last_executed_action: dict[str, object] | None = None
    stop_result: dict[str, Any] | None = None
    episode_result: dict[str, Any] = field(default_factory=dict)
    reuse_current_place_next_step: LocalmapPlaceReuseState | None = None
    current_decision_observation: dict[str, object] | None = None
    observation_resolvers: dict[str, dict[str, object]] = field(default_factory=dict)
    next_agent_observation_index: int = 0
    pending_node_move: dict[str, object] | None = None
    node_move_history: list[dict[str, object]] = field(default_factory=list)
    last_waypoint_selection_context: dict[str, object] | None = None
    waypoint_selection_context_by_node_id: dict[str, dict[str, object]] = field(default_factory=dict)
    pending_vln_terminal_check: dict[str, object] | None = None
    pending_vln_task_progress_initialization: dict[str, object] | None = None
    pending_stop_confirmation: dict[str, object] | None = None
    stop_confirmation_history: list[dict[str, object]] = field(default_factory=list)
    consecutive_need_more_evidence_count: int = 0

    def __post_init__(self) -> None:
        if int(self.max_retrieve_rounds) <= 0:
            raise ValueError("max_retrieve_rounds must be positive")

    @property
    def graph(self) -> Graph:
        return self.system.graph

    @property
    def cache(self):
        return self.system.cache

    @property
    def landmark_controller(self) -> LandmarkDetectionController:
        return self.system.landmark_controller

    @property
    def current_place_node_id(self) -> str | None:
        return self.system.current_node_id

    @current_place_node_id.setter
    def current_place_node_id(self, node_id: str | None) -> None:
        self.system.set_current_node_id(node_id)

    def ensure_floor_runtime_state(self, floor_id: str, bev_map_kwargs: dict[str, Any]) -> None:
        floor_id_str = str(floor_id).strip()
        if floor_id_str == "":
            raise ValueError("floor_id must be non-empty")
        self.frontier_records_by_floor.setdefault(floor_id_str, {})
        self.overlay_records_by_floor.setdefault(floor_id_str, {})
        if (
            floor_id_str not in self.global_explorations_by_floor
            or floor_id_str not in self.frontier_filter_explorations_by_floor
        ):
            from navclaw.mapping.exploration.bev_map import Map1Map2BEVMap
            from navclaw.mapping.exploration.manager import ExplorationManager

        if floor_id_str not in self.global_explorations_by_floor:
            self.global_explorations_by_floor[floor_id_str] = ExplorationManager(
                bev_map=Map1Map2BEVMap(**dict(bev_map_kwargs))
            )
        if floor_id_str not in self.frontier_filter_explorations_by_floor:
            self.frontier_filter_explorations_by_floor[floor_id_str] = ExplorationManager(
                bev_map=Map1Map2BEVMap(**dict(bev_map_kwargs))
            )

    def global_exploration_for_floor(self, floor_id: str) -> ExplorationManager:
        floor_id_str = str(floor_id)
        try:
            return self.global_explorations_by_floor[floor_id_str]
        except KeyError as exc:
            raise ValueError(f"missing global exploration for floor_id={floor_id_str!r}") from exc

    def frontier_filter_exploration_for_floor(self, floor_id: str) -> ExplorationManager:
        floor_id_str = str(floor_id)
        try:
            return self.frontier_filter_explorations_by_floor[floor_id_str]
        except KeyError as exc:
            raise ValueError(f"missing frontier filter exploration for floor_id={floor_id_str!r}") from exc

    def frontier_records_for_floor(self, floor_id: str) -> dict[str, LocalmapFrontierRecord]:
        floor_id_str = str(floor_id)
        try:
            return self.frontier_records_by_floor[floor_id_str]
        except KeyError as exc:
            raise ValueError(f"missing frontier records for floor_id={floor_id_str!r}") from exc

    def overlay_records_for_floor(self, floor_id: str) -> dict[str, LocalmapOverlayRecord]:
        floor_id_str = str(floor_id)
        try:
            return self.overlay_records_by_floor[floor_id_str]
        except KeyError as exc:
            raise ValueError(f"missing overlay records for floor_id={floor_id_str!r}") from exc

    def all_frontier_records(self) -> dict[str, LocalmapFrontierRecord]:
        merged: dict[str, LocalmapFrontierRecord] = {}
        for floor_id, records in self.frontier_records_by_floor.items():
            for frontier_id, record in records.items():
                merged[f"{floor_id}:{frontier_id}"] = record
        return merged

    def all_overlay_records(self) -> dict[str, LocalmapOverlayRecord]:
        merged: dict[str, LocalmapOverlayRecord] = {}
        for floor_id, records in self.overlay_records_by_floor.items():
            for overlay_id, record in records.items():
                merged[f"{floor_id}:{overlay_id}"] = record
        return merged

    @property
    def finalize_reason(self) -> str:
        return self._finalize_reason

    @finalize_reason.setter
    def finalize_reason(self, reason: str) -> None:
        reason_text = str(reason)
        self._finalize_reason = reason_text
        if reason_text != "terminated":
            self.system.mark_done(reason_text)


@dataclass
class NavClawStepState:
    place_step_index: int
    timing: StepTimingRecorder
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    policy_decision: dict[str, object] = field(default_factory=dict)
    visual_decision: dict[str, object] = field(default_factory=dict)
    stop_confirmation: dict[str, object] = field(default_factory=dict)
    node_summary: dict[str, object] = field(default_factory=dict)
    episodic_retrieval: dict[str, object] = field(default_factory=dict)
    agent_decision: dict[str, object] = field(default_factory=dict)
    agent_action: dict[str, object] = field(default_factory=dict)
    agent_feedback: dict[str, object] = field(default_factory=dict)
    agent_observations: list[dict[str, object]] = field(default_factory=list)
    llm_decision_context: dict[str, object] = field(default_factory=dict)
    refresh_state_summary: dict[str, object] = field(default_factory=dict)
    reasonnav_semantic_memory_update: dict[str, object] = field(default_factory=dict)
    place_reuse_state: LocalmapPlaceReuseState | None = None
    place_step: LocalmapPlaceStepResult | None = None
    current_place_node_id: str | None = None
    anchor_obs_id: str | None = None
    current_obs_id: str | None = None
    panorama_obs_ids: list[str] = field(default_factory=list)
    angle_to_obs_id: dict[int, str] = field(default_factory=dict)
    panorama_views: list[dict[str, object]] = field(default_factory=list)
    panorama_angle_debug: dict[str, object] = field(default_factory=dict)
    reused_place_obs_ids: list[str] = field(default_factory=list)
    current_projection_obs_ids: list[str] = field(default_factory=list)
    local_exploration: ExplorationManager | None = None
    reachable_candidates: list[FrontierCandidate] = field(default_factory=list)
    frontier_update: LocalmapFrontierUpdateResult | None = None
    detection_decision: dict[str, object] | None = None
    passive_goal_decision: dict[str, object] | None = None
    goal_recovery: dict[str, object] | None = None
    navigation_plan: LocalmapNavigationPlan | None = None
    selected_route_target: LocalmapRouteTarget | None = None
    executed_action: dict[str, object] | None = None
    results: list[tuple[ActionCall, ActionResult]] = field(default_factory=list)
    navigation_segment_timings: list[dict[str, object]] = field(default_factory=list)
    decision_payload: dict[str, object] = field(default_factory=dict)
    pre_action_payload: dict[str, object] | None = None
    current_obs_id_after: str | None = None
    created_frontier_ids: list[str] = field(default_factory=list)
