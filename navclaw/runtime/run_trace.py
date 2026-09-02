from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from navclaw.graph.graph import Graph


@dataclass
class RunTraceEvent:
    step: int
    action: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "summary": self.summary,
            "data": self.data,
        }


class RunTrace:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.path: list[str] = []
        self.events: list[RunTraceEvent] = []

    def record_node(self, node_id: str) -> None:
        self.path.append(node_id)

    def add_event(self, event: RunTraceEvent) -> None:
        self.events.append(event)

    def forget_node(self, node_id: str) -> None:
        node_id_str = str(node_id)
        self.path = [item for item in self.path if str(item) != node_id_str]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": list(self.path),
            "events": [event.to_dict() for event in self.events],
        }

    def to_context_str(self) -> str:
        path_text = " -> ".join(self.path) if self.path else "empty"
        event_text = "\n".join(
            f"{event.step}: action={event.action}, summary={event.summary}"
            for event in self.events
        )
        if event_text == "":
            event_text = "empty"
        return f"Path:\n{path_text}\nEvents:\n{event_text}"
