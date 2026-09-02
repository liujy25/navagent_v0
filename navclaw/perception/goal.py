from __future__ import annotations

from dataclasses import dataclass, field

from navclaw.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION


@dataclass
class GoalSpec:
    """A language instruction in the canonical VLN task."""

    description: str
    goal_type: str = "description"
    goal_kind: str = GOAL_KIND_VLN_INSTRUCTION
    detector_prompt: str = ""
    confusable_categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        instruction = str(self.description).strip()
        if not instruction:
            raise ValueError("VLN instruction must be non-empty")
        self.description = instruction
        self.detector_prompt = instruction
        self.goal_type = "description"
        self.goal_kind = GOAL_KIND_VLN_INSTRUCTION
        self.confusable_categories = []

    def navigation_goal_text(self) -> str:
        return self.description

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_type": self.goal_type,
            "goal_kind": self.goal_kind,
            "description": self.description,
        }


def vln_instruction_goal_spec(instruction_text: str) -> GoalSpec:
    return GoalSpec(description=str(instruction_text))
