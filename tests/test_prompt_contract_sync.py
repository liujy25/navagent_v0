from __future__ import annotations

from types import SimpleNamespace
import unittest

from navclaw.agent.entity_knowledge import manage_retrieved_knowledge
from navclaw.agent.episodic_retrieval import (
    RetrievalWorkspace,
    RetrieveItem,
    RetrieveRequest,
    RetrieveRound,
)
from navclaw.agent.visual_action_context import VisualViewContext
from navclaw.agent.visual_navigation import _vln_backtrack_context_text
from navclaw.agent.vln_runner import _landmark_categories
from navclaw.agent.vln_waypoint_policy import (
    _VlnViewCandidateSet,
    _selected_sampled_candidate,
)
from navclaw.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate
from navclaw.graph.graph import Graph
from navclaw.perception.landmark_detection import LANDMARK_DETECTOR_BOX_THRESHOLD


class PromptContractSyncTests(unittest.TestCase):



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



if __name__ == "__main__":
    unittest.main()
