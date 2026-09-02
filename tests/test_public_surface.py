from __future__ import annotations

from types import SimpleNamespace

import navclaw.llm.client as client_module
from navclaw.agent.actions import AgentActionType
from navclaw.config.vln_runtime import VlnRuntimeConfig
from navclaw.perception.goal import vln_instruction_goal_spec
from navclaw.runners.robot_vln_runner import build_parser


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
    assert config.max_retrieve_rounds == 8
    goal = vln_instruction_goal_spec("Walk past the table and stop by the door.")
    assert goal.goal_kind == "vln_instruction"
    assert goal.navigation_goal_text().startswith("Walk past")
    assert {item.value for item in AgentActionType} == {
        "visual_waypoint",
        "vertical_transition",
        "finalize",
    }


def test_one_model_is_used_for_every_completion(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

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
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    client = client_module.LLMClient(
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
    )
    parsed = client._create_visual_json_completion(
        call_name="test",
        system_prompt="system",
        user_prompt="user",
        max_new_tokens=32,
        token_field="max_completion_tokens",
        retry_count=1,
        client_kind="va",
    )
    assert parsed == {"ok": True}
    assert requests[0]["model"] == "test-model"
    assert client.get_usage_summary()["total_tokens"] == 12
