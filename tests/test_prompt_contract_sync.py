from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from navclaw.agent.entity_knowledge import manage_retrieved_knowledge
from navclaw.agent.episodic_retrieval import (
    RetrievalWorkspace,
    RetrieveItem,
    RetrieveRequest,
    RetrieveRound,
)
from navclaw.agent.vertical_transition import (
    VerticalTransitionStepReplan,
    _plan_vertical_transition_waypoint,
)
from navclaw.agent.vertical_transition_policy import VerticalTransitionStepDecision
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_navigation import _vln_backtrack_context_text
from navclaw.agent.visual_policy import (
    decide_vln_task_progress_step,
    decide_vertical_transition_visual_action_point,
    ensure_task_progress_memory,
    normalize_vln_navigation_step,
    normalize_vln_task_progress_step,
)
from navclaw.agent.visual_policy_decisions import (
    TaskProgressDecision,
    VisualActionPointDecision,
    normalize_vertical_transition_visual_action_point,
)
from navclaw.agent.vln_runner import _landmark_categories
from navclaw.agent.vln_waypoint_policy import (
    _VlnViewCandidateSet,
    _selected_sampled_candidate,
)
from navclaw.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate
from navclaw.graph.graph import Graph
from navclaw.memory.task_progress import TaskProgressItem, TaskProgressMemory
from navclaw.perception.landmark_detection import LANDMARK_DETECTOR_BOX_THRESHOLD


class PromptContractSyncTests(unittest.TestCase):
    def test_initial_progress_prompt_and_schema_are_strict(self) -> None:
        class Client:
            system_prompt = ""
            user_prompt = ""

            def generate_task_progress_memory(self, system_prompt, user_prompt):
                self.system_prompt = str(system_prompt)
                self.user_prompt = str(user_prompt)
                return {
                    "progress_items": [
                        {
                            "content": "Exit the current room.",
                            "status": "active",
                            "result": "",
                        }
                    ]
                }

        client = Client()
        memory = TaskProgressMemory()
        result = ensure_task_progress_memory(
            client=client,
            task_progress=memory,
            goal_text="Exit the current room.",
            task_type="vln_instruction",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )

        self.assertEqual(result["source"], "llm")
        self.assertIn("Initial Task Progress Generator", client.system_prompt)
        self.assertIn("Decomposition rules:", client.user_prompt)
        self.assertIn("Preserve instruction order", client.user_prompt)
        self.assertNotIn('"conditions"', client.user_prompt)

        invalid_memory = TaskProgressMemory()
        invalid = ensure_task_progress_memory(
            client=SimpleNamespace(
                generate_task_progress_memory=lambda _system, _user: {
                    "progress_items": [
                        {
                            "content": "Exit the current room.",
                            "status": "done",
                            "result": "Exited.",
                            "conditions": [],
                        }
                    ]
                }
            ),
            task_progress=invalid_memory,
            goal_text="Exit the current room.",
            task_type="vln_instruction",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )
        self.assertEqual(invalid["source"], "fallback")
        self.assertIn("must contain exactly", invalid["error"])

    def test_tpu_prompt_uses_route_context_and_transition_conditions(self) -> None:
        captured: dict[str, object] = {}

        def decide(system_prompt, content):
            captured["system_prompt"] = str(system_prompt)
            captured["content"] = list(content)
            return {
                "tool_calls": [
                    {
                        "name": "update_progress",
                        "arguments": {"progress_updates": []},
                    }
                ]
            }

        decide_vln_task_progress_step(
            client=SimpleNamespace(decide_vln_task_progress_step=decide),
            cache=SimpleNamespace(),
            visual_context=VisualActionContext(current_node_id="n1", views=[]),
            task_progress=TaskProgressMemory(
                items=[
                    TaskProgressItem(content="Complete the first route stage."),
                    TaskProgressItem(content="Continue to the next route target."),
                ]
            ),
            memory_index_text="",
            retrieval_workspace_content=[],
            retrieve_max_rounds=0,
            retrieve_completed_rounds=0,
            retrieve_fields_by_ref={},
            allow_retrieve=False,
            allow_update_progress=True,
            require_retrieval_conclusion=False,
        )

        prompt_text = "\n".join(
            str(item.get("text", ""))
            for item in captured["content"]
            if item.get("type") == "text"
        )
        self.assertIn("one stage of the ordered navigation route", prompt_text)
        self.assertIn("consistently with the surrounding route context", prompt_text)
        self.assertIn(
            "route relation connecting the parent item to an adjacent task-progress item",
            prompt_text,
        )
        self.assertIn("Use the preceding item to interpret", prompt_text)
        self.assertIn("Use the following item when it disambiguates", prompt_text)
        self.assertIn(
            "Do not require the following item itself to be completed",
            prompt_text,
        )
        self.assertIn("Add a transition condition only when", prompt_text)
        self.assertIn(
            "Do not add a transition condition when the parent item's completion",
            prompt_text,
        )

    def test_transition_condition_stays_with_parent_item(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(content="Complete the first route stage."),
                TaskProgressItem(content="Continue to the next route target."),
            ]
        )

        memory.apply_condition_updates(
            [
                {
                    "op": "add",
                    "item_index": 0,
                    "content": "The first route stage connects to the next route target.",
                    "status": "unconfirmed",
                }
            ],
            current_node_id="n1",
        )

        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].conditions[0].status, "unconfirmed")
        self.assertEqual(memory.items[1].status, "active")
        self.assertEqual(memory.items[1].conditions, [])

    def test_landmark_prompt_matches_bounded_place_identity_contract(self) -> None:
        captured: dict[str, object] = {}

        def complete(**kwargs):
            captured.update(kwargs)
            return {"landmark_categories": ["bed"]}

        categories = _landmark_categories(
            SimpleNamespace(_create_visual_json_completion=complete),
            "Enter the bedroom and wait near the entrance.",
        )

        self.assertEqual(categories, ["bed"])
        self.assertEqual(LANDMARK_DETECTOR_BOX_THRESHOLD, 0.5)
        system_prompt = str(captured["system_prompt"])
        user_prompt = str(captured["user_prompt"])
        self.assertIn("VLN Landmark Category Generator", system_prompt)
        self.assertIn("most diagnostic typical detectable objects", system_prompt)
        self.assertIn("exclusively as visual route evidence", system_prompt)
        self.assertIn("For each distinct place type", user_prompt)
        self.assertIn("at most one inferred category", user_prompt)
        self.assertNotIn("bedroom ->", user_prompt)

    def test_tpu_tool_calls_and_immutable_item_updates(self) -> None:
        decision = normalize_vln_task_progress_step(
            {
                "tool_calls": [
                    {
                        "name": "update_progress_conditions",
                        "arguments": {
                            "condition_updates": [
                                {
                                    "op": "add",
                                    "item_index": 0,
                                    "content": "The threshold was crossed.",
                                    "status": "confirmed",
                                    "reason": "The executed edge crosses it.",
                                }
                            ]
                        },
                    },
                    {
                        "name": "update_progress",
                        "arguments": {
                            "progress_updates": [
                                {
                                    "op": "update",
                                    "index": 0,
                                    "status": "done",
                                    "result": "Crossed the threshold.",
                                    "reason": "The movement evidence confirms it.",
                                }
                            ]
                        },
                    },
                ]
            },
            retrieve_fields_by_ref={},
            allow_retrieve=False,
            allow_update_progress=True,
            require_retrieval_conclusion=False,
        )

        self.assertEqual(decision.progress_updates[0]["reason"], "The movement evidence confirms it.")
        self.assertEqual(decision.progress_condition_updates[0]["status"], "confirmed")
        serialized = decision.to_dict()
        self.assertEqual(serialized["tool_calls"][0]["name"], "update_progress_conditions")
        self.assertEqual(serialized["tool_calls"][-1]["name"], "update_progress")

        with self.assertRaisesRegex(ValueError, "only op='update'"):
            normalize_vln_task_progress_step(
                {
                    "tool_calls": [
                        {
                            "name": "update_progress",
                            "arguments": {
                                "progress_updates": [
                                    {
                                        "op": "rewrite",
                                        "index": 0,
                                        "content": "Changed item.",
                                        "status": "active",
                                        "result": "",
                                        "reason": "Unsupported mutation.",
                                    }
                                ]
                            },
                        }
                    ]
                },
                retrieve_fields_by_ref={},
                allow_retrieve=False,
                allow_update_progress=True,
                require_retrieval_conclusion=False,
            )

    def test_terminal_check_serializes_external_reason_field(self) -> None:
        decision = normalize_vln_task_progress_step(
            {
                "tool_calls": [
                    {
                        "name": "update_progress",
                        "arguments": {
                            "progress_updates": [],
                            "terminal_check": {
                                "decision": "continue",
                                "reason": "The final relation is not established.",
                                "missing_constraints": ["Reach the entrance-side region."],
                            },
                        },
                    }
                ]
            },
            retrieve_fields_by_ref={},
            allow_retrieve=False,
            allow_update_progress=True,
            require_retrieval_conclusion=False,
            require_terminal_check=True,
        )

        terminal = decision.to_dict()["tool_calls"][-1]["arguments"]["terminal_check"]
        self.assertEqual(terminal["reason"], "The final relation is not established.")
        self.assertNotIn("reasoning", terminal)

    def test_ekm_receives_round_aligned_retrieval_trace(self) -> None:
        graph = Graph()
        node = graph.add_node(
            position=(0.0, 0.0, 0.0),
            yaw=0.0,
            obs_id="obs_0",
        )
        workspace = RetrievalWorkspace(entries=[], max_retrieve_rounds=1)
        workspace.rounds.append(
            RetrieveRound(
                round_index=0,
                request=RetrieveRequest(
                    query="Was the threshold crossed?",
                    items=(RetrieveItem(ref=node.id, fields=("rgb",)),),
                ),
                source_obs_ids=("obs_0",),
                conclusion="The stored view confirms the threshold crossing.",
            )
        )
        captured: dict[str, str] = {}

        def manage(system_prompt, user_prompt):
            captured["system_prompt"] = str(system_prompt)
            captured["user_prompt"] = str(user_prompt)
            return {"entity_knowledge_updates": []}

        manage_retrieved_knowledge(
            client=SimpleNamespace(manage_knowledge=manage),
            state=SimpleNamespace(
                graph=graph,
                landmark_controller=SimpleNamespace(buffer=SimpleNamespace()),
            ),
            workspace=workspace,
            current_node_id=node.id,
            progress_updates=[],
        )

        self.assertIn("Entity Knowledge Manager (EKM)", captured["system_prompt"])
        self.assertIn('"round": 1', captured["user_prompt"])
        self.assertIn('"request": [', captured["user_prompt"])
        self.assertIn('"conclusion": "The stored view confirms', captured["user_prompt"])
        self.assertNotIn('"round_index"', captured["user_prompt"])

    def test_pcnp_owns_horizontal_direction(self) -> None:
        progress = TaskProgressDecision(progress_analysis="", progress_reasoning="")
        decision = normalize_vln_navigation_step(
            {
                "approach_to_stop": {
                    "movement": "move",
                    "direction": "left",
                    "stop_objective": "Wait near the entrance.",
                    "reason": "The entrance-side floor is visible.",
                }
            },
            latest_task_progress=progress,
            system_owned_waypoint_objective=True,
        )
        self.assertEqual(decision.direction, "left")
        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            normalize_vln_navigation_step(
                {
                    "approach_to_stop": {
                        "movement": "move",
                        "stop_objective": "Wait near the entrance.",
                        "reason": "The entrance-side floor is visible.",
                    }
                },
                latest_task_progress=progress,
                system_owned_waypoint_objective=True,
            )

    def test_backtrack_is_labeled_as_a_planning_reference(self) -> None:
        text = _vln_backtrack_context_text(
            [
                {
                    "trigger_planning_node_id": "n2",
                    "anchor_node_id": "n0",
                    "physical_robot_node_id": "n2",
                    "objective": "Reassess the missed branch.",
                    "reason": "The route evidence conflicts.",
                }
            ]
        )
        self.assertIn("Backtrack 1:", text)
        self.assertIn("Trigger planning reference node: n2", text)
        self.assertIn("Planning reference node: n0", text)
        self.assertIn("Physical robot node: n2", text)

    def test_waypoint_parser_does_not_require_json_key_order(self) -> None:
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
        selected, selected_view, error = _selected_sampled_candidate(
            response={
                "selected_direction": "right",
                "candidate_label": 7,
                "reasoning": "The candidate advances the fixed route direction.",
            },
            view_candidate_sets=[
                _VlnViewCandidateSet(
                    view=view,
                    candidates=[candidate],
                    rgb_overlay=None,
                )
            ],
        )
        self.assertIs(selected, candidate)
        self.assertIs(selected_view, view)
        self.assertEqual(error, "")

    def test_vertical_grounder_uses_explicit_union_and_replans_on_failure(self) -> None:
        failure_payload = {
            "status": "failure",
            "reasoning": "The stair surface is occluded.",
            "point_2d": None,
            "target": "",
            "failure_reason": "No safe matching walking surface is visible.",
        }
        normalized = normalize_vertical_transition_visual_action_point(failure_payload)
        self.assertEqual(normalized.status, "failure")

        client = SimpleNamespace(
            decide_visual_action=lambda system_prompt, content: failure_payload
        )
        with patch(
            "navclaw.agent.visual_policy.image_content_for_view",
            return_value={"type": "image_url"},
        ):
            result = decide_vertical_transition_visual_action_point(
                client=client,
                cache=SimpleNamespace(),
                selected_view=VisualViewContext(
                    angle_deg=0,
                    obs_id="obs_0",
                    rgb_id="rgb_0",
                    depth_id="depth_0",
                ),
                waypoint_target="the next stair tread",
            )
        self.assertEqual(result.status, "failure")

        with patch(
            "navclaw.agent.vertical_transition.decide_vertical_transition_visual_action_point",
            return_value=VisualActionPointDecision(
                status="failure",
                reasoning="The stair surface is occluded.",
                point_2d=None,
                target="",
                failure_reason="No safe matching walking surface is visible.",
            ),
        ), patch("navclaw.agent.vertical_transition.ground_visual_waypoint") as project, patch(
            "navclaw.agent.vertical_transition.verify_vertical_transition_waypoint"
        ) as verify:
            with self.assertRaises(VerticalTransitionStepReplan):
                _plan_vertical_transition_waypoint(
                    state=SimpleNamespace(llm_client=object(), cache=SimpleNamespace()),
                    direction="up",
                    visual_context=VisualActionContext(
                        current_node_id="n0",
                        views=[
                            VisualViewContext(
                                angle_deg=0,
                                obs_id="obs_0",
                                rgb_id="rgb_0",
                                depth_id="depth_0",
                            )
                        ],
                    ),
                    step_decision=VerticalTransitionStepDecision(
                        thought="Continue upward.",
                        transition_status="continue",
                        waypoint_target="the next stair tread",
                        selected_angle_deg=0,
                    ),
                    current_height_m=0.0,
                )
            project.assert_not_called()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
