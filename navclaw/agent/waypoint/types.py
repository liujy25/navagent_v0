from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from navclaw.schemas import ActionCall

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState, NavClawStepState
    from navclaw.agent.visual_action_context import VisualActionContext
    from navclaw.agent.visual_navigation import VisualNavigationDecision
    from navclaw.agent.visual_policy_decisions import NavigationModeDecision
    from navclaw.agent.visual_policy_decisions import TaskProgressDecision
    from navclaw.memory.task_progress import TaskProgressUpdateResult
    from navclaw.visualization.action_mode_overlays import BevOverlayTransform


FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY = "frontier_skeleton_sample"
@dataclass(frozen=True)
class GroundedWaypointTarget:
    goal_xy: tuple[float, float]
    goal_yaw: float
    world_z: float | None
    policy_name: str
    source_type: str
    source_id: str = ""
    obs_id: str = ""
    angle_deg: int | None = None
    point_2d: tuple[float, float] | None = None
    raw_world_xy: tuple[float, float] | None = None
    consume_frontier_id: str = ""

    def to_action_call(self) -> ActionCall:
        args: dict[str, object] = {
            "x": float(self.goal_xy[0]),
            "y": float(self.goal_xy[1]),
            "yaw": float(self.goal_yaw),
        }
        if self.world_z is not None:
            args["z"] = float(self.world_z)
        return ActionCall(action="move", args=args)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "goal_yaw": float(self.goal_yaw),
            "world_z": None if self.world_z is None else float(self.world_z),
            "policy_name": str(self.policy_name),
            "source_type": str(self.source_type),
            "source_id": str(self.source_id),
            "obs_id": str(self.obs_id),
            "angle_deg": None if self.angle_deg is None else int(self.angle_deg),
            "point_2d": (
                None
                if self.point_2d is None
                else [float(self.point_2d[0]), float(self.point_2d[1])]
            ),
            "raw_world_xy": (
                None
                if self.raw_world_xy is None
                else [float(self.raw_world_xy[0]), float(self.raw_world_xy[1])]
            ),
            "consume_frontier_id": str(self.consume_frontier_id),
        }


@dataclass(frozen=True)
class WaypointPolicyResult:
    policy_name: str
    target: GroundedWaypointTarget | None
    reasoning: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_name": str(self.policy_name),
            "target": None if self.target is None else self.target.to_dict(),
            "reasoning": str(self.reasoning),
            "failure_reason": str(self.failure_reason),
        }


@dataclass(frozen=True)
class WaypointPolicyOutput:
    result: WaypointPolicyResult
    decision: VisualNavigationDecision


@dataclass(frozen=True)
class WaypointPlanningContext:
    state: NavClawAgentState
    step: NavClawStepState
    goal_text: str
    goal_kind: str
    visual_context: VisualActionContext
    task_progress: TaskProgressDecision
    task_progress_text: str
    task_progress_initialization: dict[str, object]
    navigation_mode: NavigationModeDecision
    task_progress_update_result: TaskProgressUpdateResult
    context_evidence_text: str
    execution_visual_context: VisualActionContext | None = None
    inherited_agent_context_content: list[dict[str, object]] = field(default_factory=list)
    active_progress_item: str = ""
    candidate_bev_landmark_markers: list[dict[str, object]] = field(default_factory=list)
    candidate_bev_base_image: object | None = None
    candidate_bev_transform: BevOverlayTransform | None = None
    candidate_bev_reference_node_marker: dict[str, object] | None = None
    candidate_bev_context_text: str = ""
