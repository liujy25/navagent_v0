from __future__ import annotations

from dataclasses import dataclass

from navclaw.agent.actions import AgentAction


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action is None:
            raise ValueError("agent decision requires action")

    def to_dict(self) -> dict[str, object]:
        return {
            "action": None if self.action is None else self.action.to_dict(),
            "reason": str(self.reason),
        }


def act_decision(
    *,
    action: AgentAction,
    reason: str = "",
) -> AgentDecision:
    return AgentDecision(
        action=action,
        reason=str(reason),
    )
