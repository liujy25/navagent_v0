from __future__ import annotations

from typing import Any

import requests

from navclaw.env.interface import EnvInterface, Pose, RawObservation
from navclaw.env.rpc_protocol import decode_array, decode_depth_meters, decode_rgb_image


def _raise_for_status_with_body(response: requests.Response) -> None:
    if response.ok:
        return
    body = response.text.strip()
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            body = str(payload["error"])
    except ValueError:
        pass
    message = f"{response.status_code} Server Error for url: {response.url}"
    raise requests.HTTPError(f"{message}: {body}" if body else message, response=response)


class RobotRPCEnv(EnvInterface):
    """HTTP adapter for the RGB-D/odometry and navigation service on the robot."""

    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._supports_path_move: bool | None = None
        self._last_move_intermediate_observations: list[RawObservation] = []

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        _raise_for_status_with_body(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"{path} response must be a JSON object")
        return payload

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        _raise_for_status_with_body(response)
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError(f"{path} response must be a JSON object")
        return result

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def reset(self, instruction: str) -> dict[str, Any]:
        self._last_move_intermediate_observations = []
        return self._post("/reset", {"goal": str(instruction)})

    def current_episode_info(self) -> dict[str, Any]:
        return self._get("/current_episode")

    def finalize_run(self, reason: str) -> dict[str, Any]:
        return self._post("/finalize_run", {"reason": str(reason)})

    @staticmethod
    def _pose(payload: dict[str, Any]) -> Pose:
        return Pose(
            x=float(payload["x"]),
            y=float(payload["y"]),
            z=float(payload["z"]),
            yaw=float(payload["yaw"]),
        )

    @classmethod
    def _observation(cls, payload: dict[str, Any]) -> RawObservation:
        return RawObservation(
            pose=cls._pose(dict(payload["pose"])),
            rgb=decode_rgb_image(str(payload["rgb"])),
            depth=decode_depth_meters(dict(payload["depth"])).astype("float32"),
            intrinsics=decode_array(dict(payload["intrinsics"])).tolist(),
            T_cam_odom=decode_array(dict(payload["T_cam_odom"])).tolist(),
            T_odom_base=decode_array(dict(payload["T_odom_base"])).tolist(),
        )

    def get_obs(self) -> RawObservation:
        return self._observation(self._post("/get_obs", {}))

    def turn(self, direction: str) -> Pose:
        if direction not in {"left", "right"}:
            raise ValueError(f"Unsupported turn direction: {direction!r}")
        payload = self._post("/turn", {"direction": direction})
        return self._pose(dict(payload["pose"]))

    def supports_move_along_path(self) -> bool:
        if self._supports_path_move is None:
            self._supports_path_move = bool(
                self.health().get("supports_path_move", False)
            )
        return bool(self._supports_path_move)

    def move(
        self,
        x: float,
        y: float,
        yaw: float,
        z: float | None = None,
    ) -> Pose:
        payload: dict[str, Any] = {
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
        }
        if z is not None:
            payload["z"] = float(z)
        return self._move(payload)

    def move_along_path(
        self,
        path_xy: list[list[float]],
        final_yaw: float,
        final_z: float | None = None,
    ) -> Pose:
        if not path_xy:
            raise ValueError("move_along_path requires at least one path point")
        final_xy = path_xy[-1]
        payload: dict[str, Any] = {
            "x": float(final_xy[0]),
            "y": float(final_xy[1]),
            "yaw": float(final_yaw),
            "path_xy": [[float(point[0]), float(point[1])] for point in path_xy],
        }
        if final_z is not None:
            payload["z"] = float(final_z)
        return self._move(payload)

    def _move(self, request: dict[str, Any]) -> Pose:
        payload = self._post("/move", request)
        raw_observations = payload.get("intermediate_observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError("move response intermediate_observations must be a list")
        self._last_move_intermediate_observations = [
            self._observation(dict(item))
            for item in raw_observations
            if isinstance(item, dict)
        ]
        return self._pose(dict(payload["pose"]))

    def pop_last_move_intermediate_observations(self) -> list[RawObservation]:
        observations = self._last_move_intermediate_observations
        self._last_move_intermediate_observations = []
        return observations

    def pop_last_turn_observation(self) -> RawObservation | None:
        return None

    def stop(self) -> dict[str, Any]:
        return self._post("/stop", {})
