from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class TimingStageRecord:
    name: str
    elapsed_seconds: float
    env_step_before: int
    env_step_after: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def env_step_delta(self) -> int:
        return int(self.env_step_after) - int(self.env_step_before)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "elapsed_seconds": float(self.elapsed_seconds),
            "env_step_before": int(self.env_step_before),
            "env_step_after": int(self.env_step_after),
            "env_step_delta": int(self.env_step_delta),
        }
        if self.details != {}:
            payload["details"] = dict(self.details)
        return payload


class StepTimingRecorder:
    def __init__(self, env_step_start: int) -> None:
        self.env_step_start = int(env_step_start)
        self._step_wall_start = time.perf_counter()
        self._active_name: str | None = None
        self._active_wall_start: float | None = None
        self._active_env_step_start: int | None = None
        self._stages: list[TimingStageRecord] = []

    def start(self, name: str, env_step: int) -> None:
        if self._active_name is not None:
            raise ValueError(f"Timing stage already active: {self._active_name}")
        self._active_name = str(name)
        self._active_wall_start = time.perf_counter()
        self._active_env_step_start = int(env_step)

    def stop(self, name: str, env_step: int, details: dict[str, Any] | None = None) -> TimingStageRecord:
        if self._active_name is None:
            raise ValueError(f"Timing stage was not started: {name}")
        if self._active_name != str(name):
            raise ValueError(f"Stopping timing stage {name}, but active stage is {self._active_name}")
        if self._active_wall_start is None or self._active_env_step_start is None:
            raise ValueError(f"Timing stage has incomplete active state: {name}")
        record = TimingStageRecord(
            name=str(name),
            elapsed_seconds=time.perf_counter() - float(self._active_wall_start),
            env_step_before=int(self._active_env_step_start),
            env_step_after=int(env_step),
            details={} if details is None else dict(details),
        )
        self._stages.append(record)
        self._active_name = None
        self._active_wall_start = None
        self._active_env_step_start = None
        return record

    def to_dict(self, env_step_end: int) -> dict[str, object]:
        if self._active_name is not None:
            raise ValueError(f"Timing stage still active: {self._active_name}")
        env_step_end_int = int(env_step_end)
        return {
            "step_total": {
                "elapsed_seconds": float(time.perf_counter() - self._step_wall_start),
                "env_step_before": int(self.env_step_start),
                "env_step_after": env_step_end_int,
                "env_step_delta": env_step_end_int - int(self.env_step_start),
            },
            "stages": [stage.to_dict() for stage in self._stages],
        }


class WallStepTimer:
    def __init__(self, env_step_start: int) -> None:
        self.env_step_start = int(env_step_start)
        self.wall_start = time.perf_counter()

    def finish(self, env_step_end: int) -> dict[str, object]:
        env_step_end_int = int(env_step_end)
        return {
            "elapsed_seconds": float(time.perf_counter() - self.wall_start),
            "env_step_before": int(self.env_step_start),
            "env_step_after": env_step_end_int,
            "env_step_delta": env_step_end_int - int(self.env_step_start),
        }
