from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionCall:
    action: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": self.args}


@dataclass
class ActionResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    subactions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "message": self.message,
            "subactions": self.subactions,
        }


@dataclass
class ActionDecision:
    thought: str
    call: ActionCall | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionDecision":
        call = None

        if "call" in data and data["call"] not in (None, {}):
            if "action" not in data["call"]:
                raise ValueError(f'ActionDecision.call must contain "action": {data["call"]}')
            call = ActionCall(
                action=data["call"]["action"],
                args=data["call"].get("args", {}),
            )
        elif "action" in data:
            call = ActionCall(action=data["action"], args=data.get("args", {}))
        elif data.get("actions"):
            first = data["actions"][0]
            call = ActionCall(action=first["action"], args=first.get("args", {}))

        return cls(
            thought=data.get("thought", ""),
            call=call,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "thought": self.thought,
            "call": self.call.to_dict() if self.call is not None else None,
        }
        if self.debug != {}:
            payload["debug"] = self.debug
        return payload
