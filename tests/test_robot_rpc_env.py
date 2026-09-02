from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from navclaw.env.robot_rpc_env import RobotRPCEnv
from navclaw.env.rpc_protocol import encode_array, encode_depth_meters, encode_rgb_image


def _observation_payload() -> dict[str, object]:
    return {
        "pose": {"x": 1.0, "y": 2.0, "z": 0.1, "yaw": 30.0},
        "rgb": encode_rgb_image(np.zeros((4, 5, 3), dtype=np.uint8)),
        "depth": encode_depth_meters(np.ones((4, 5), dtype=np.float32)),
        "intrinsics": encode_array(np.eye(3, dtype=np.float32)),
        "T_cam_odom": encode_array(np.eye(4, dtype=np.float32)),
        "T_odom_base": encode_array(np.eye(4, dtype=np.float32)),
    }


def test_robot_rpc_observation_and_move_history(monkeypatch) -> None:
    def fake_post(url, json, timeout):
        del timeout
        if url.endswith("/get_obs"):
            payload = _observation_payload()
        elif url.endswith("/move"):
            payload = {
                "pose": _observation_payload()["pose"],
                "intermediate_observations": [_observation_payload()],
            }
            assert json["x"] == 2.0
        else:
            raise AssertionError(url)
        return SimpleNamespace(
            ok=True,
            json=lambda: payload,
            text="",
            status_code=200,
            url=url,
        )

    monkeypatch.setattr("navclaw.env.robot_rpc_env.requests.post", fake_post)
    env = RobotRPCEnv("http://robot:1877")
    observation = env.get_obs()
    assert observation.rgb.shape == (4, 5, 3)
    assert observation.depth.shape == (4, 5)
    pose = env.move(2.0, 3.0, 45.0)
    assert pose.yaw == 30.0
    assert len(env.pop_last_move_intermediate_observations()) == 1
    assert env.pop_last_move_intermediate_observations() == []
