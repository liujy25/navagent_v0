from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VlnRuntimeConfig:
    """Fixed mainline settings plus robot-specific safety bounds."""

    max_place_steps: int = 40
    max_retrieve_rounds: int = 8
    candidate_prune_radius: float = 1.0
    candidate_dedup_radius: float = 0.8
    max_vertical_transition_steps: int = 8
    vertical_floor_match_threshold_m: float = 0.8
    bev_min_height_m: float = 0.3
    bev_max_height_m: float = 1.4
    robot_radius_m: float = 0.4

    def __post_init__(self) -> None:
        if self.max_place_steps <= 0:
            raise ValueError("max_place_steps must be positive")
        if self.max_retrieve_rounds <= 0:
            raise ValueError("max_retrieve_rounds must be positive")
        if self.max_vertical_transition_steps <= 0:
            raise ValueError("max_vertical_transition_steps must be positive")
        if self.robot_radius_m <= 0.0:
            raise ValueError("robot_radius_m must be positive")

    def global_bev_kwargs(self) -> dict[str, float]:
        return {
            "min_height": float(self.bev_min_height_m),
            "max_height": float(self.bev_max_height_m),
            "agent_radius": float(self.robot_radius_m),
        }
