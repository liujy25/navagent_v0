from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from navclaw.agent.waypoint.types import WaypointPlanningContext
from navclaw.agent.waypoint.types import WaypointPolicyOutput


class WaypointPolicy(Protocol):
    name: str

    def plan(self, context: WaypointPlanningContext) -> WaypointPolicyOutput:
        ...


@dataclass(frozen=True)
class FunctionWaypointPolicy:
    name: str
    handler: Callable[[WaypointPlanningContext], WaypointPolicyOutput]

    def plan(self, context: WaypointPlanningContext) -> WaypointPolicyOutput:
        return self.handler(context)


class WaypointPolicyRegistry:
    def __init__(self, policies: list[WaypointPolicy]) -> None:
        self._policies = {str(policy.name): policy for policy in policies}
        if len(self._policies) != len(policies):
            raise ValueError("waypoint policy names must be unique")

    def get(self, name: str) -> WaypointPolicy:
        policy_name = str(name).strip()
        try:
            return self._policies[policy_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown waypoint policy {policy_name!r}; available={sorted(self._policies)}"
            ) from exc
