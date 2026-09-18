from __future__ import annotations

from types import SimpleNamespace

import navprobe.llm.client as client_module
from navprobe.agent.actions import AgentActionType
from navprobe.config.vln_runtime import VlnRuntimeConfig
from navprobe.perception.goal import vln_instruction_goal_spec
from navprobe.runtime.panorama_config import robot_panorama_config
from navprobe.runners.robot_vln_runner import build_parser


def test_cli_exposes_only_robot_vln_controls() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {
        "robot_url",
        "instruction",
        "model",
        "base_url",
        "api_key",
        "yolow_model",
        "detector_device",
        "max_place_steps",
        "output",
    } <= destinations
    assert not {
        "ablation",
        "dataset_root",
        "split",
        "record_dir",
        "results_dir",
        "plan_mode",
        "waypoint_policy",
        "waypoint_context",
    } & destinations


def test_runtime_and_goal_are_fixed_to_canonical_vln() -> None:
    config = VlnRuntimeConfig()
    assert config.max_retrieve_rounds == 6
    goal = vln_instruction_goal_spec("Walk past the table and stop by the door.")
    assert goal.goal_kind == "vln_instruction"
    assert goal.navigation_goal_text().startswith("Walk past")
    assert {item.value for item in AgentActionType} == {
        "visual_waypoint",
        "vertical_transition",
        "finalize",
    }


def test_robot_panorama_uses_front_left_back_right_views() -> None:
    panorama = robot_panorama_config()
    assert panorama.observation_count == 4
    assert panorama.turns_per_observation == 1
    assert panorama.turn_direction == "left"
    assert panorama.observation_spacing_degrees == 90.0


def test_one_model_is_used_for_every_completion(monkeypatch) -> None:
    requests: list[dict[str, object]] = []
    clients: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            )
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=usage,
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    client = client_module.LLMClient(
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
    )
    for method in (
        client.generate_task_progress_memory,
        client.decide_vln_task_progress_step,
        client.decide_vln_navigation_step,
        client.manage_knowledge,
        client.summarize_node,
        client.decide_vertical_transition_step,
    ):
        assert method("system", "user") == {"ok": True}
    for call_name in ("vln_waypoint_planner", "vertical_fss_waypoint_planner", "vln_landmark_detection_list"):
        assert client._create_visual_json_completion(
            call_name=call_name,
            system_prompt="system",
            user_prompt=[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}],
            max_new_tokens=32,
            token_field="max_completion_tokens",
            retry_count=1,
        ) == {"ok": True}
    assert len(clients) == 1
    assert clients[0]["base_url"] == "https://provider.example/v1"
    assert clients[0]["api_key"] == "test-key"
    assert len(requests) == 9
    assert {request["model"] for request in requests} == {"test-model"}
    assert client.get_usage_summary()["total_tokens"] == 108
