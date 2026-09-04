from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from navclaw.agent.episodic_retrieval import _add_node_rgb_panel
from navclaw.agent.episodic_retrieval import MemoryIndexEntry
from navclaw.agent.episodic_retrieval import RetrievalWorkspace
from navclaw.agent.episodic_retrieval import RetrieveItem
from navclaw.agent.episodic_retrieval import RetrieveRequest
from navclaw.agent.episodic_retrieval import RetrieveRound
from navclaw.agent.node_summary import summarize_current_node
from navclaw.agent.vertical_transition_policy import (
    _normalize_vertical_transition_step,
)
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import ordered_panorama_angles
from navclaw.agent.visual_action_context import VisualActionContext
from navclaw.agent.visual_action_context import VisualViewContext
from navclaw.agent.visual_navigation import _vln_waypoint_inherited_agent_context
from navclaw.agent.visual_navigation import run_episodic_retrieval_loop
from navclaw.agent.visual_policy import decide_vln_navigation_step
from navclaw.agent.visual_policy import decide_vln_task_progress_step
from navclaw.agent.visual_policy import normalize_vln_navigation_step
from navclaw.agent.visual_policy import normalize_vln_task_progress_step
from navclaw.agent.visual_policy_decisions import NavigationModeDecision
from navclaw.agent.visual_policy_decisions import TaskProgressDecision
from navclaw.agent.visual_policy_prompt_images import (
    image_content_for_current_panorama_views,
)
from navclaw.agent.vln_waypoint_policy import _selected_sampled_candidate
from navclaw.agent.vln_waypoint_policy import _VlnViewCandidateSet
from navclaw.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate
from navclaw.llm.client import LLMClient
from navclaw.graph.graph import Graph
from navclaw.memory.task_progress import TaskProgressItem
from navclaw.memory.task_progress import TaskProgressMemory
from navclaw.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION


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

    def test_progress_and_navigation_have_independent_prompts(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, list[dict[str, object]]]] = []

            def decide_vln_task_progress_step(self, system_prompt, content):
                self.calls.append(("progress", system_prompt, list(content)))
                return {
                    "tool_calls": [
                        {
                            "name": "update_progress",
                            "arguments": {"progress_updates": []},
                        }
                    ]
                }

            def decide_vln_navigation_step(self, system_prompt, content):
                self.calls.append(("navigation", system_prompt, list(content)))
                return {
                    "go_to_waypoint": {
                        "direction": "front",
                        "reason": "It advances the current progress item.",
                    }
                }

        client = Client()
        visual_context = VisualActionContext(current_node_id="n1", views=_views())
        with patch(
            "navclaw.agent.visual_policy_prompt_images.image_content_for_camera_array",
            return_value={"type": "image_url"},
        ):
            progress = decide_vln_task_progress_step(
                client=client,
                cache=_cache(),
                visual_context=visual_context,
                task_progress=TaskProgressMemory(),
                memory_index_text="",
                retrieval_workspace_content=[],
                retrieve_max_rounds=0,
                retrieve_completed_rounds=0,
                retrieve_fields_by_ref={},
                allow_retrieve=False,
                allow_update_progress=True,
                require_retrieval_conclusion=False,
            )
            decide_vln_navigation_step(
                client=client,
                cache=_cache(),
                visual_context=visual_context,
                task_progress=TaskProgressMemory(),
                latest_task_progress=progress,
                retrieval_workspace_content=[],
            )

        progress_call, navigation_call = client.calls
        progress_text = "\n".join(
            str(item.get("text", "")) for item in progress_call[2]
        )
        navigation_text = "\n".join(
            str(item.get("text", "")) for item in navigation_call[2]
        )
        self.assertIn("Task Progress Updater", progress_call[1])
        self.assertIn(
            "Progress-Conditioned Navigation Planner",
            navigation_call[1],
        )
        self.assertNotIn("### `retrieve`", navigation_text)
        self.assertNotIn("### `update_progress`", navigation_text)
        self.assertIn("### `update_progress`", progress_text)
        self.assertNotIn("Navigation task:", progress_text)
        self.assertNotIn("Navigation task:", navigation_text)

    def test_module_normalizers_reject_cross_module_outputs(self) -> None:
        progress = TaskProgressDecision(progress_analysis="", progress_reasoning="")
        with self.assertRaises(ValueError):
            normalize_vln_task_progress_step(
                {
                    "go_to_waypoint": {
                        "objective": "Enter the corridor.",
                        "reason": "It advances the route.",
                    }
                },
                retrieve_fields_by_ref={},
                allow_retrieve=False,
                allow_update_progress=True,
                require_retrieval_conclusion=False,
            )
        with self.assertRaises(ValueError):
            normalize_vln_navigation_step(
                {
                    "progress_condition_updates": [],
                    "update_progress": {"progress_updates": []},
                },
                latest_task_progress=progress,
            )

    def test_retrieval_and_progress_precede_navigation_in_runtime_loop(self) -> None:
        graph = Graph()
        node = graph.add_node(
            position=(0.0, 0.0, 0.0),
            yaw=0.0,
            obs_id="obs_0",
        )
        state = SimpleNamespace(
            max_retrieve_rounds=8,
            graph=graph,
            cache=SimpleNamespace(),
            llm_client=SimpleNamespace(),
            landmark_controller=SimpleNamespace(),
            pending_vln_terminal_check=None,
            pending_vln_task_progress_initialization=None,
            system=SimpleNamespace(
                current_floor_id="floor_0",
                memory=SimpleNamespace(
                    task_progress=TaskProgressMemory(
                        items=[TaskProgressItem(content="Leave the room.")]
                    )
                ),
            ),
        )
        visual_context = VisualActionContext(
            current_node_id=str(node.id),
            views=[
                VisualViewContext(
                    angle_deg=0,
                    obs_id="obs_0",
                    rgb_id="rgb_0",
                    depth_id="depth_0",
                )
            ],
        )
        progress = TaskProgressDecision(
            progress_analysis="The route item remains active.",
            progress_reasoning="No completion evidence yet.",
            retrieval_conclusion="The earlier view confirms the route landmark.",
        )
        navigation = NavigationModeDecision(
            action_mode="go_to_waypoint",
            stop_objective="",
            progress_analysis="",
            progress_reasoning="",
            reasoning_action="Explore the visible corridor.",
            action_reason="It advances the active route item.",
        )

        retrieve = RetrieveRequest(
            query="What was visible at the earlier node?",
            items=(RetrieveItem(ref="n_history", fields=("rgb",)),),
        )

        def fake_execute_retrieve_request(*, workspace, request, **_kwargs):
            workspace.rounds.append(
                RetrieveRound(
                    round_index=0,
                    request=request,
                    source_obs_ids=(),
                )
            )
            workspace.text_evidence["raw"] = "RAW_RETRIEVAL_EVIDENCE"

        with (
            patch(
                "navclaw.agent.visual_navigation.build_memory_index",
                return_value=[
                    MemoryIndexEntry(
                        ref="n_history",
                        kind="node",
                        floor_id="floor_0",
                        available_fields={"rgb": 1},
                    )
                ],
            ),
            patch(
                "navclaw.agent.visual_navigation.execute_retrieve_request",
                side_effect=fake_execute_retrieve_request,
            ),
            patch(
                "navclaw.agent.visual_navigation.manage_retrieved_knowledge",
                return_value={"status": "ok"},
            ) as manage_knowledge,
            patch(
                "navclaw.agent.visual_navigation.decide_vln_task_progress_step",
                side_effect=[retrieve, progress],
            ) as update_progress,
            patch(
                "navclaw.agent.visual_navigation.decide_vln_navigation_step",
                return_value=navigation,
            ) as plan_navigation,
        ):
            result = run_episodic_retrieval_loop(
                state=state,
                step=SimpleNamespace(place_step_index=1, episodic_retrieval={}),
                goal=SimpleNamespace(goal_kind=GOAL_KIND_VLN_INSTRUCTION),
                goal_text="Leave the room.",
                visual_context=visual_context,
                vln_landmark_context=None,
            )

        self.assertEqual(update_progress.call_count, 2)
        self.assertEqual(plan_navigation.call_count, 1)
        self.assertEqual(manage_knowledge.call_count, 1)
        self.assertIs(
            plan_navigation.call_args.kwargs["latest_task_progress"],
            progress,
        )
        self.assertEqual(result.navigation_mode.action_mode, "go_to_waypoint")
        planner_context = plan_navigation.call_args.kwargs[
            "retrieval_workspace_content"
        ]
        self.assertNotIn("RAW_RETRIEVAL_EVIDENCE", str(planner_context))
        self.assertIn("The earlier view confirms", str(planner_context))

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

    def test_context_and_direction_response_compatibility(self) -> None:
        progress = TaskProgressDecision(progress_analysis="", progress_reasoning="")
        navigation = NavigationModeDecision(
            action_mode="go_to_waypoint",
            stop_objective="",
            progress_analysis="",
            progress_reasoning="",
            reasoning_action="Enter the corridor.",
            direction="front",
        )
        content = _vln_waypoint_inherited_agent_context(
            context_loop=None,
            task_progress=progress,
            navigation_mode=navigation,
            active_progress_item="Leave the room.",
            task_progress_text="0. [active] Leave the room.",
        )
        prompt_text = "\n".join(str(item.get("text", "")) for item in content)
        self.assertNotIn("Navigation task:", prompt_text)
        self.assertIn(
            "Active item and its read-only grounding conditions:\nLeave the room.",
            prompt_text,
        )
        self.assertIn('"direction": "front"', prompt_text)
        self.assertLess(
            prompt_text.index("High-level navigation action:"),
            prompt_text.index("Navigation state:"),
        )

        for payload in (
            {
                "thought": "The stairs continue behind the robot.",
                "transition_status": "continue",
                "waypoint_target": "the next stair tread",
                "selected_direction": "back",
            },
            {
                "thought": "The stairs continue behind the robot.",
                "transition_status": "continue",
                "waypoint_target": "the next stair tread",
                "selected_angle_deg": 180,
            },
        ):
            self.assertEqual(
                _normalize_vertical_transition_step(payload).selected_angle_deg,
                180,
            )

        candidate = VlnSampledWaypointCandidate(
            label=7,
            goal_xy=(1.0, 0.0),
            path_xy=[(0.0, 0.0), (1.0, 0.0)],
            point_pixel=(4.0, 4.0),
            point_2d=(0.5, 0.5),
            euclidean_distance_m=1.0,
            path_length_m=1.0,
            projected_depth_m=1.0,
        )
        view = VisualViewContext(
            angle_deg=270,
            obs_id="obs_270",
            rgb_id="rgb_270",
            depth_id="depth_270",
        )
        view_sets = [
            _VlnViewCandidateSet(
                view=view,
                candidates=[candidate],
                rgb_overlay=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        ]
        for response in (
            {
                "reasoning": "The right corridor advances the route.",
                "selected_direction": "right",
                "candidate_label": 7,
            },
            {
                "reasoning": "The right corridor advances the route.",
                "selected_angle_deg": 270,
                "candidate_label": 7,
            },
        ):
            selected, selected_view, error = _selected_sampled_candidate(
                response=response,
                view_candidate_sets=view_sets,
            )
            self.assertIs(selected, candidate)
            self.assertIs(selected_view, view)
            self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()
