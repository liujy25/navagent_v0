from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from navclaw.memory.task_progress import TaskProgressMemory

if TYPE_CHECKING:
    from navclaw.graph.graph import Graph
    from navclaw.perception.landmark_detection import LandmarkDetectionController
    from navclaw.runtime.cache import RuntimeCache


@dataclass
class GoalState:
    target: str

    def __post_init__(self) -> None:
        target = str(self.target).strip()
        if target == "":
            raise ValueError("GoalState.target must be non-empty")
        self.target = target

    def to_dict(self) -> dict[str, object]:
        return {"target": str(self.target)}


@dataclass
class AgentStepRecord:
    step_index: int
    decision: dict[str, Any] = field(default_factory=dict)
    feedback_summary: dict[str, Any] = field(default_factory=dict)
    state_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": int(self.step_index),
            "decision": deepcopy(self.decision),
            "feedback_summary": deepcopy(self.feedback_summary),
            "state_summary": deepcopy(self.state_summary),
        }


@dataclass
class AgentHistory:
    steps: list[AgentStepRecord] = field(default_factory=list)

    def append_step(
        self,
        *,
        step_index: int,
        decision: dict[str, Any],
        feedback_summary: dict[str, Any],
        state_summary: dict[str, Any],
    ) -> AgentStepRecord:
        record = AgentStepRecord(
            step_index=int(step_index),
            decision=deepcopy(decision),
            feedback_summary=deepcopy(feedback_summary),
            state_summary=deepcopy(state_summary),
        )
        self.steps.append(record)
        return record

    def to_dict(self) -> dict[str, object]:
        return {"steps": [record.to_dict() for record in self.steps]}


@dataclass
class AgentMemory:
    summaries: list[str] = field(default_factory=list)
    task_progress: TaskProgressMemory = field(default_factory=TaskProgressMemory)

    def to_dict(self) -> dict[str, object]:
        return {
            "summaries": [str(summary) for summary in self.summaries],
            "task_progress": self.task_progress.to_dict(),
        }


@dataclass
class SystemState:
    goal: GoalState
    graph: Graph
    landmark_controller: LandmarkDetectionController
    cache: RuntimeCache
    current_node_id: str | None = None
    current_floor_id: str = "floor_0"
    current_floor_height: float = 0.0
    history: AgentHistory = field(default_factory=AgentHistory)
    memory: AgentMemory = field(default_factory=AgentMemory)
    is_done: bool = False
    done_reason: str | None = None

    def set_current_node_id(self, node_id: str | None) -> None:
        if node_id is None:
            self.current_node_id = None
            return
        node_id_str = str(node_id)
        if not self.graph.has_node(node_id_str):
            raise ValueError(f"current_node_id must exist in graph: {node_id_str!r}")
        self.current_node_id = node_id_str

    def set_current_floor(self, floor_id: str, height: float | None = None) -> None:
        floor_id_str = str(floor_id).strip()
        if floor_id_str == "":
            raise ValueError("current_floor_id must be non-empty")
        if floor_id_str not in self.graph.floors:
            self.graph.add_floor(
                floor_id=floor_id_str,
                height=float(0.0 if height is None else height),
            )
        floor = self.graph.get_floor(floor_id_str)
        if height is not None:
            floor.height = float(height)
        self.current_floor_id = floor_id_str
        self.current_floor_height = float(floor.height)

    def mark_done(self, reason: str) -> None:
        reason_text = str(reason).strip()
        if reason_text == "":
            raise ValueError("done reason must be non-empty")
        self.is_done = True
        self.done_reason = reason_text

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal.to_dict(),
            "current_node_id": self.current_node_id,
            "current_floor_id": self.current_floor_id,
            "current_floor_height": float(self.current_floor_height),
            "graph": self.graph.to_dict(),
            "landmark_buffer": self.landmark_controller.buffer.to_context(
                step_index=len(self.history.steps)
            ),
            "history": self.history.to_dict(),
            "memory": self.memory.to_dict(),
            "is_done": bool(self.is_done),
            "done_reason": self.done_reason,
        }
