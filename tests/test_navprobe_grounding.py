import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from navclaw.agent import visual_navigation
from navclaw.agent.episodic_retrieval import RetrievalWorkspace
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_navigation import _vln_waypoint_inherited_agent_context
from navclaw.agent.visual_policy_decisions import NavigationModeDecision, TaskProgressDecision
from navclaw.agent.vln_waypoint_policy import _selected_sampled_candidate, _VlnViewCandidateSet, plan_vln_waypoint_loop
from navclaw.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate
from navclaw.agent.waypoint.types import GroundedWaypointTarget, WaypointPolicyOutput, WaypointPolicyResult
from navclaw.graph.graph import Graph
from navclaw.memory.task_progress import TaskProgressItem, TaskProgressMemory, TaskProgressUpdateResult
from navclaw.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION


def no_match_response():
    return {
        "reasoning": "All labeled endpoints remain outside the requested doorway.",
        "selected_direction": None, "candidate_label": None,
        "failure_reason": "No candidate reaches the doorway interior.",
    }


class NavProbeGroundingTests(unittest.TestCase):
    def test_grounder_receives_self_contained_skill_without_task_state_or_update_log(self):
        assessment = TaskProgressDecision(
            progress_analysis="EXECUTIVE_ASSESSMENT", progress_reasoning="",
            progress_updates=[{"op": "rewrite", "content": "UPDATE_LOG"}],
        )
        context = SimpleNamespace(
            task_progress_decision=assessment,
            workspace=SimpleNamespace(rounds=[object()], retrieval_log_text=lambda: "RETRIEVAL_LOG"),
            backtrack_contexts=[],
        )
        for mode, sid in (("go_to_waypoint", "sg2"), ("approach_to_stop", None)):
            with self.subTest(mode=mode):
                action = NavigationModeDecision(
                    action_mode=mode, stop_objective="Stand on the floor beside the sink.",
                    progress_analysis="", progress_reasoning="", reasoning_action="",
                    action_reason="Enter the doorway beside the shelving, which was misdetected as a refrigerator.",
                    action_objective="Reach the floor inside that doorway.",
                    direction="left", subgoal_id=sid, subgoal_attempt=2 if sid else None,
                    approach_movement="move" if sid is None else "",
                )
                blocks = _vln_waypoint_inherited_agent_context(
                    context_loop=context, task_progress=assessment, navigation_mode=action,
                    task_progress_text="FULL_AGENDA_AND_HISTORY", original_instruction="ORIGINAL_GOAL",
                    dynamic_agenda=True,
                )
                self.assertEqual(len(blocks), 1)
                text = blocks[0]["text"]
                payload = json.loads(text.split("\n", 1)[1])
                self.assertEqual(payload["mode"], mode)
                self.assertEqual(payload["reason"], action.action_reason)
                self.assertEqual(payload["objective"], action.action_objective)
                self.assertEqual(payload["direction"], "left")
                if sid:
                    self.assertEqual((payload["subgoal_id"], payload["subgoal_attempt"]), (sid, 2))
                else:
                    self.assertEqual(payload["movement"], "move")
                for sentinel in ("EXECUTIVE_ASSESSMENT", "UPDATE_LOG", "RETRIEVAL_LOG", "FULL_AGENDA_AND_HISTORY", "ORIGINAL_GOAL"):
                    self.assertNotIn(sentinel, text)
                self.assertEqual(len(context.workspace.rounds), 1)
                self.assertEqual(assessment.progress_updates[0]["content"], "UPDATE_LOG")

    def test_no_matching_label_returns_grounding_failure(self):
        selected, view, failure = _selected_sampled_candidate(
            response=no_match_response(),
            view_candidate_sets=[],
        )
        self.assertIsNone(selected)
        self.assertIsNone(view)
        self.assertEqual(failure, "No candidate reaches the doorway interior.")

    def run_waypoint_loop(self, responses):
        views = [VisualViewContext(angle_deg=angle, obs_id=f"obs_{angle}", rgb_id="rgb", depth_id="depth")
                 for angle in (0, 90)]
        candidates = [
            _VlnViewCandidateSet(view=view, candidates=[VlnSampledWaypointCandidate(
                label=label, goal_xy=(1.0, 0.0), path_xy=[(0.0, 0.0), (1.0, 0.0)],
                point_pixel=(4.0, 4.0), point_2d=(0.5, 0.5), euclidean_distance_m=1.0,
                path_length_m=1.0, projected_depth_m=1.0,
            )], rgb_overlay=np.zeros((8, 8, 3), dtype=np.uint8))
            for label, view in zip((4, 9), views)
        ]
        cache = SimpleNamespace(get_observation=lambda _obs_id: SimpleNamespace(
            observation=SimpleNamespace(T_odom_base=np.eye(4)),
        ))
        client = SimpleNamespace(_create_visual_json_completion=Mock(side_effect=responses))
        with patch("navclaw.agent.vln_waypoint_policy.draw_vln_sampled_waypoint_bev_overlay",
                   return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            result = plan_vln_waypoint_loop(
                client=client, cache=cache, goal_text="", floor_height_m=0.0,
                visual_context=VisualActionContext(current_node_id="n0", views=views),
                task_progress_analysis="", task_progress_text="", exploration=SimpleNamespace(),
                inherited_agent_context_content=[{"type": "text", "text": "Enter the doorway interior."}],
                candidate_set_builder=lambda **_kwargs: (candidates, []),
            )
        return result, client._create_visual_json_completion

    def test_explicit_no_match_returns_explanation_for_replanning_without_grounding_retry(self):
        for include_failure_reason in (True, False):
            with self.subTest(include_failure_reason=include_failure_reason):
                response = no_match_response()
                if not include_failure_reason:
                    response.pop("failure_reason")
                result, completion = self.run_waypoint_loop([response])
                completion.assert_called_once()
                self.assertEqual(result.failure_reason, "navigation_intent_has_no_matching_waypoint")
                self.assertIsNone(result.waypoint)
                self.assertIsNone(result.local_move_plan)
                self.assertEqual(result.attempt_records, [])
                self.assertEqual(len(result.navigation_replan_feedback), 1)
                feedback = result.navigation_replan_feedback[0]
                self.assertEqual(feedback["failure_summary"], response.get("failure_reason", response["reasoning"]))
                self.assertEqual(feedback["grounding_response"], response)
                self.assertEqual(feedback["candidate_labels_by_direction"], {"front": [4], "left": [9]})
                response["reasoning"] = "Changed after the response."
                self.assertNotEqual(feedback["grounding_response"]["reasoning"], response["reasoning"])

    def test_missing_label_is_validation_failure_before_explicit_no_match(self):
        malformed = {"reasoning": "The doorway has no matching candidate.", "selected_direction": None}
        result, completion = self.run_waypoint_loop([malformed, no_match_response()])
        self.assertEqual(completion.call_count, 2)
        self.assertIn("Validation feedback", str(completion.call_args.kwargs["user_prompt"]))
        self.assertEqual(result.failure_reason, "navigation_intent_has_no_matching_waypoint")
        self.assertEqual(len(result.navigation_replan_feedback), 1)
        self.assertEqual(result.navigation_replan_feedback[0]["grounding_response"], no_match_response())


class NavProbeGroundingReplanTests(unittest.TestCase):
    def setUp(self):
        self.memory = TaskProgressMemory()
        self.memory.initialize_agenda("Enter the kitchen, then stop by the sink.", [TaskProgressItem("Enter the kitchen.")])
        self.graph = Graph()
        node = self.graph.add_node(position=(0.0, 0.0, 0.0), yaw=0.0, obs_id="obs_0")
        self.context = VisualActionContext(current_node_id=str(node.id), views=[])
        self.state = SimpleNamespace(
            waypoint_policy_name="frontier_skeleton_sample",
            episodic_retrieval_enabled=True, max_retrieve_rounds=1, ablation_name="none",
            graph=self.graph, cache=SimpleNamespace(), llm_client=SimpleNamespace(), landmark_controller=SimpleNamespace(),
            system=SimpleNamespace(current_floor_id="floor_0", memory=SimpleNamespace(task_progress=self.memory)),
            action_executor=SimpleNamespace(execute=Mock()),
        )
        self.step = SimpleNamespace(place_step_index=1, episodic_retrieval={})
        self.assessment = TaskProgressDecision(progress_analysis="The kitchen entry remains unresolved.", progress_reasoning="")
        self.executive_result = visual_navigation.EpisodicRetrievalLoopResult(
            task_progress_initialization={}, task_progress_decision=self.assessment,
            task_progress_update_result=TaskProgressUpdateResult(), navigation_mode=self.navigation("front"),
            workspace=RetrievalWorkspace(entries=[], max_retrieve_rounds=1), progress_update_count=1,
            planning_visual_context=self.context,
        )
        self.executive = Mock(return_value=self.executive_result)
        self.semantic_policy = Mock(return_value=self.navigation("left"))
        self.grounder = Mock()
        for name, kwargs in (
            ("build_visual_action_context", {"return_value": self.context}),
            ("build_vln_landmark_context", {"return_value": None}),
            ("run_episodic_retrieval_loop", {"side_effect": self.executive}),
            ("decide_vln_navigation_step", {"side_effect": self.semantic_policy}),
            ("_vln_backtrack_node_ids", {"return_value": {"n_history"}}),
            ("_vln_waypoint_candidate_bev_base", {"return_value": None}),
            ("_waypoint_policy_registry", {"return_value": SimpleNamespace(get=lambda _name: SimpleNamespace(plan=self.grounder))}),
        ):
            patcher = patch.object(visual_navigation, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def navigation(self, direction):
        return NavigationModeDecision(
            action_mode="go_to_waypoint", stop_objective="", progress_analysis="", progress_reasoning="",
            reasoning_action="Enter the visible kitchen doorway.", action_reason="Enter the visible kitchen doorway.",
            direction=direction, subgoal_id="sg0",
        )

    def grounding_output(self, context, *, failure=""):
        target = None if failure else GroundedWaypointTarget(
            goal_xy=(1.0, 0.0), goal_yaw=0.0, world_z=0.0,
            policy_name="frontier_skeleton_sample", source_type="sampled_candidate",
        )
        feedback = [{"failure_reason": "navigation_intent_has_no_matching_waypoint", "failure_summary": failure}] if failure else []
        result = WaypointPolicyResult(
            policy_name="frontier_skeleton_sample", target=target,
            failure_reason="navigation_intent_has_no_matching_waypoint" if failure else "",
        )
        return WaypointPolicyOutput(result=result, decision=visual_navigation.VisualNavigationDecision(
            visual_context=context.visual_context, task_progress_initialization=context.task_progress_initialization,
            navigation_mode=context.navigation_mode, task_progress_update_result=context.task_progress_update_result,
            verify_memory_update_result=context.verify_memory_update_result, visual_action=None, waypoint=None,
            action_call=None if target is None else target.to_action_call(),
            failure_reason=result.failure_reason, navigation_replan_feedback=feedback,
        ))

    def run_navigation(self):
        before = self.memory.to_dict()
        result = visual_navigation.plan_visual_navigation_action(
            state=self.state, step=self.step, context_evidence_text="",
            goal=SimpleNamespace(goal_kind=GOAL_KIND_VLN_INSTRUCTION, description=self.memory.original_goal),
        )
        self.executive.assert_called_once()
        self.state.action_executor.execute.assert_not_called()
        self.assertEqual(self.memory.to_dict(), before)
        return result

    def test_no_match_replans_only_semantic_policy_and_grounds_its_revised_intent(self):
        explanation = no_match_response()["failure_reason"]
        def ground(context):
            return self.grounding_output(context, failure=explanation if self.grounder.call_count == 1 else "")
        self.grounder.side_effect = ground
        result = self.run_navigation()
        self.semantic_policy.assert_called_once()
        policy_context = self.semantic_policy.call_args.kwargs
        self.assertIn(explanation, policy_context["navigation_replan_feedback_text"])
        self.assertIs(policy_context["latest_task_progress"], self.assessment)
        self.assertIs(policy_context["task_progress"], self.memory)
        self.assertEqual(policy_context["allowed_backtrack_node_ids"], set())
        self.assertEqual([call.args[0].navigation_mode.direction for call in self.grounder.call_args_list], ["front", "left"])
        self.assertIn('"direction": "left"', str(self.grounder.call_args.args[0].inherited_agent_context_content))
        self.assertEqual(result.failure_reason, "")
        self.assertEqual(result.action_call.action, "move")
        self.assertEqual(result.navigation_mode.direction, "left")
        self.assertEqual(len(result.navigation_replan_feedback), 1)
        self.assertEqual(result.navigation_replan_feedback[0]["failure_summary"], explanation)

    def test_repeated_no_match_is_bounded_and_preserves_each_rejection_once(self):
        def ground(context):
            return self.grounding_output(context, failure=f"No connected doorway candidate on attempt {self.grounder.call_count}.")
        self.grounder.side_effect = ground
        result = self.run_navigation()
        limit = visual_navigation.NAVIGATION_REPLAN_MAX_ATTEMPTS
        self.assertEqual(self.grounder.call_count, limit)
        self.assertEqual(self.semantic_policy.call_count, limit - 1)
        self.assertEqual(result.failure_reason, "navigation_intent_has_no_matching_waypoint")
        self.assertIsNone(result.action_call)
        self.assertIsNone(result.grounded_waypoint_target)
        self.assertEqual([entry["failure_summary"] for entry in result.navigation_replan_feedback],
                         [f"No connected doorway candidate on attempt {index}." for index in range(1, limit + 1)])
        for index, call in enumerate(self.semantic_policy.call_args_list, start=1):
            for earlier in range(1, index + 1):
                self.assertIn(f"attempt {earlier}.", call.kwargs["navigation_replan_feedback_text"])


if __name__ == "__main__":
    unittest.main()
