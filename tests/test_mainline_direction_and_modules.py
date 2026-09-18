from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from navclaw.agent.episodic_retrieval import _add_node_rgb_panel
from navclaw.agent.episodic_retrieval import RetrievalWorkspace
from navclaw.agent.node_summary import summarize_current_node
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import ordered_panorama_angles
from navclaw.agent.visual_action_context import VisualActionContext
from navclaw.agent.visual_action_context import VisualViewContext
from navclaw.agent.visual_policy_prompt_images import (
    image_content_for_current_panorama_views,
)
from navclaw.llm.client import LLMClient


def _views() -> list[VisualViewContext]:
    return [
        VisualViewContext(
            angle_deg=angle,
            obs_id=f"obs_{angle}",
            rgb_id=f"rgb_{angle}",
            depth_id=f"depth_{angle}",
        )
        for angle in (270, 0, 90, 180)
    ]


def _cache() -> object:
    return SimpleNamespace(
        get_observation=lambda _obs_id: SimpleNamespace(
            observation=SimpleNamespace(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8)
            )
        )
    )


class MainlineDirectionAndModuleTests(unittest.TestCase):
    def test_model_panorama_order_and_separate_images(self) -> None:
        with patch(
            "navclaw.agent.visual_policy_prompt_images.image_content_for_camera_array",
            return_value={"type": "image_url"},
        ):
            content = image_content_for_current_panorama_views(
                cache=_cache(),
                views=_views(),
                include_visited_nodes=False,
            )

        self.assertEqual(
            ordered_panorama_angles([270, 90, 0, 180]),
            [0, 180, 90, 270],
        )
        self.assertEqual(
            [direction_for_angle(angle) for angle in (0, 180, 90, 270)],
            ["front", "back", "left", "right"],
        )
        self.assertEqual(
            [item["text"] for item in content if item["type"] == "text"],
            [
                "Current front view:",
                "Current back view:",
                "Current left view:",
                "Current right view:",
            ],
        )
        self.assertEqual(
            [item["type"] for item in content],
            ["text", "image_url"] * 4,
        )

    def test_node_summary_and_retrieval_use_directional_images(self) -> None:
        class SummaryClient:
            def __init__(self) -> None:
                self.content: list[dict[str, object]] = []

            def summarize_node(self, _system_prompt, content):
                self.content = list(content)
                return {
                    "node_summary": "A corridor junction.",
                    "direction_summaries": {
                        "front": "open corridor",
                        "back": "wall",
                        "left": "doorway",
                        "right": "side corridor",
                    },
                }

        client = SummaryClient()
        with patch(
            "navclaw.agent.visual_policy_prompt_images.image_content_for_camera_array",
            return_value={"type": "image_url"},
        ):
            summary = summarize_current_node(
                client=client,
                cache=_cache(),
                visual_context=VisualActionContext(
                    current_node_id="n1",
                    views=_views(),
                ),
            )
            workspace = RetrievalWorkspace(entries=[], max_retrieve_rounds=1)
            _add_node_rgb_panel(
                state=SimpleNamespace(cache=_cache()),
                workspace=workspace,
                key="n0:rgb",
                node_id="n0",
                obs_ids=["obs_0", "obs_90", "obs_180", "obs_270"],
            )

        self.assertEqual(
            list(summary.to_dict()["direction_summaries"]),
            ["front", "back", "left", "right"],
        )
        summary_prompt = str(client.content[0]["text"])
        self.assertIn("Distinguish visibility from room membership", summary_prompt)
        self.assertIn(
            "determine the current room from spatial boundaries",
            summary_prompt,
        )
        self.assertNotIn("entry/exit evidence", summary_prompt)
        retrieved = workspace.materialized_context_content()
        self.assertEqual(
            [
                item["text"]
                for item in retrieved
                if str(item.get("text", "")).startswith(
                    "Retrieved node n0 panorama view:"
                )
            ],
            [
                "Retrieved node n0 panorama view: front.",
                "Retrieved node n0 panorama view: back.",
                "Retrieved node n0 panorama view: left.",
                "Retrieved node n0 panorama view: right.",
            ],
        )
        self.assertEqual(
            sum(item.get("type") == "image_url" for item in retrieved),
            4,
        )




    def test_client_uses_distinct_module_call_names(self) -> None:
        client = object.__new__(LLMClient)
        calls: list[str] = []
        client._call = lambda call_name, *_args, **_kwargs: calls.append(call_name)

        client.decide_vln_task_progress_step("system", "evidence")
        client.decide_vln_navigation_step("system", "progress")

        self.assertEqual(
            calls,
            [
                "vln_task_progress_updater",
                "vln_progress_conditioned_navigation_planner",
            ],
        )



if __name__ == "__main__":
    unittest.main()
