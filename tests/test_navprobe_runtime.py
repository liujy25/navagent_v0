from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from navclaw.agent.actions import AgentAction, AgentActionType
from navclaw.agent.episodic_retrieval import MemoryIndexEntry
from navclaw.agent.visual_action_context import VisualActionContext
from navclaw.agent.visual_navigation import _bind_navigation_subgoal, run_episodic_retrieval_loop
from navclaw.agent.visual_policy_decisions import NavigationModeDecision
from navclaw.agent.visual_step_execution import _execute_visual_action_call, _mark_selected_subgoal_started
from navclaw.graph.graph import Graph
from navclaw.memory.task_progress import TaskProgressItem, TaskProgressMemory
from navclaw.perception.goal_identity import GOAL_KIND_OBJECT_CATEGORY, GOAL_KIND_VLN_INSTRUCTION
from navclaw.schemas import ActionCall, ActionResult


def executive_response(updates=(), conditions=(), *, retrieve=False, conclusion=None, terminal=None):
    calls = []
    if conditions:
        calls.append({"name": "update_predicates", "arguments": {"predicate_updates": list(conditions)}})
    if updates or terminal:
        arguments = {"agenda_updates": list(updates)}
        if terminal:
            arguments["terminal_check"] = terminal
        calls.append({"name": "update_task_state", "arguments": arguments})
    if retrieve:
        calls.append({"name": "retrieve", "arguments": {
            "query": "Did the earlier doorway lead into the kitchen?",
            "items": [{"ref": "e0", "fields": ["movement_rgb"]}],
        }})
    result = {"task_state_assessment": "Assess the original route and stopping requirements from available evidence.", "tool_calls": calls}
    if conclusion is not None:
        result["retrieval_conclusion"] = conclusion
    return result


def navigation(mode="go_to_waypoint", subgoal_id="sg0", **kwargs):
    return NavigationModeDecision(
        action_mode=mode, stop_objective="", progress_analysis="", progress_reasoning="",
        reasoning_action="Advance the supported objective.", subgoal_id=subgoal_id, **kwargs,
    )


class NavProbeRuntimeTests(unittest.TestCase):
    def setUp(self):
        graph = Graph()
        node = graph.add_node(position=(0.0, 0.0, 0.0), yaw=0.0, obs_id="obs_0")
        self.goal_text = "Enter the kitchen, then stop beside the table."
        self.memory = TaskProgressMemory()
        self.memory.initialize_agenda(self.goal_text, [TaskProgressItem("Enter the kitchen."), TaskProgressItem("Stop beside the table.")])
        self.client = SimpleNamespace(decide_vln_task_progress_step=Mock())
        self.state = SimpleNamespace(
            waypoint_policy_name="frontier_skeleton_sample",
            episodic_retrieval_enabled=True, max_retrieve_rounds=1,
            ablation_name="none", graph=graph, cache=SimpleNamespace(), llm_client=self.client,
            landmark_controller=SimpleNamespace(), pending_vln_terminal_check=None,
            pending_vln_task_progress_initialization=None,
            system=SimpleNamespace(current_floor_id="floor_0", memory=SimpleNamespace(task_progress=self.memory)),
        )
        self.context = VisualActionContext(current_node_id=str(node.id), views=[])
        self.step = SimpleNamespace(place_step_index=1, episodic_retrieval={})
        entries = [MemoryIndexEntry(ref="e0", kind="edge", floor_id="floor_0", available_fields={"movement_rgb": 1})]
        self.policy = Mock(return_value=navigation())
        self.materialize = Mock(side_effect=self.materialize_evidence)
        for target, kwargs in (
            ("navclaw.agent.visual_navigation.build_memory_index", {"return_value": entries}),
            ("navclaw.agent.visual_navigation.decide_vln_navigation_step", {"side_effect": self.policy}),
            ("navclaw.agent.episodic_retrieval._materialize_request", {"side_effect": self.materialize}),
            ("navclaw.agent.episodic_retrieval._render_shared_bev", {}),
            ("navclaw.agent.visual_navigation.manage_retrieved_knowledge", {"return_value": {}}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def materialize_evidence(self, **kwargs):
        kwargs["workspace"].text_evidence["e0.movement_rgb"] = "Historical doorway movement evidence."
        return ["obs_history"]

    def run_loop(self, goal_kind=GOAL_KIND_VLN_INSTRUCTION):
        return run_episodic_retrieval_loop(
            state=self.state, step=self.step, goal=SimpleNamespace(goal_kind=goal_kind),
            goal_text=self.goal_text, visual_context=self.context, vln_landmark_context=None,
        )

    def test_object_search_activates_empty_agenda_and_passes_original_goal_to_policy(self):
        self.goal_text = "Find a mug."
        self.memory = TaskProgressMemory()
        self.state.system.memory.task_progress = self.memory
        self.client.decide_vln_task_progress_step.return_value = executive_response([
            {"op": "add", "content": "Inspect the visible countertop.", "position": 0},
        ])
        self.run_loop(GOAL_KIND_OBJECT_CATEGORY)
        self.assertTrue(self.memory.agenda_initialized)
        self.assertEqual(self.memory.original_goal, self.goal_text)
        self.assertEqual(self.memory.items[0].subgoal_id, "sg0")
        kwargs = self.policy.call_args.kwargs
        self.assertEqual(kwargs["original_instruction"], self.goal_text)
        self.assertIn(self.goal_text, str(self.client.decide_vln_task_progress_step.call_args))

    def test_retrieval_sees_committed_agenda_then_reopens_resolved_objective(self):
        self.memory.mark_subgoal_started("sg0", self.context.current_node_id)
        self.client.decide_vln_task_progress_step.side_effect = [
            executive_response([
                {"op": "complete", "subgoal_id": "sg0", "result": "Initially accepted kitchen entry."},
            ], [{"op": "add", "subgoal_id": "sg0", "content": "The kitchen threshold was crossed.", "status": "confirmed"}], retrieve=True),
            executive_response([
                {"op": "reopen", "subgoal_id": "sg0", "position": 0},
            ], [{"op": "update", "subgoal_id": "sg0", "predicate_id": "pc0", "status": "unconfirmed"}], conclusion="The earlier doorway led to a bedroom, contradicting kitchen entry."),
        ]
        def materialize(**kwargs):
            self.assertEqual([item.subgoal_id for item in self.memory.items], ["sg1"])
            self.assertEqual(self.memory.history[0]["result"], "Initially accepted kitchen entry.")
            return self.materialize_evidence(**kwargs)
        self.materialize.side_effect = materialize
        result = self.run_loop()
        self.assertEqual(result.workspace.retrieve_count, 1)
        self.assertFalse(result.workspace.has_pending_evidence)
        self.assertEqual((self.memory.items[0].subgoal_id, self.memory.items[0].attempt), ("sg0", 2))
        self.assertEqual(self.memory.history[0]["attempt"], 1)
        self.assertEqual(self.memory.items[0].start_node_id, "")
        self.assertEqual(result.navigation_mode.subgoal_attempt, 2)
        self.assertEqual(self.memory.items[0].conditions[0].status, "unconfirmed")
        self.assertEqual(self.memory.history[0]["conditions"][0]["status"], "confirmed")

    def test_policy_binding_uses_selected_identity_and_attempt_without_starting_execution(self):
        self.memory.apply_agenda_updates([
            {"op": "abandon", "subgoal_id": "sg1", "result": "Earlier endpoint estimate was wrong."},
            {"op": "reopen", "subgoal_id": "sg1", "position": 0},
        ], [])
        bound = _bind_navigation_subgoal(navigation(subgoal_id="sg1"), self.memory)
        self.assertEqual((bound.subgoal_id, bound.subgoal_attempt), ("sg1", 2))
        self.assertEqual(bound.to_dict()["subgoal_id"], "sg1")
        self.assertEqual(bound.to_dict()["subgoal_attempt"], 2)
        self.assertTrue(all(not item.start_node_id for item in self.memory.items))
        with self.assertRaisesRegex(ValueError, "inactive subgoal"):
            _bind_navigation_subgoal(navigation(subgoal_id="missing"), self.memory)

    def test_physical_execution_starts_selected_attempt_and_records_stable_binding(self):
        selected = _bind_navigation_subgoal(navigation(subgoal_id="sg1"), self.memory)
        self.step.current_place_node_id = self.context.current_node_id
        self.step.agent_decision = {}
        action = AgentAction(AgentActionType.VISUAL_WAYPOINT, args={"selected_angle_deg": 0})
        call = ActionCall("move_along_path", {"path_xy": [[0.0, 0.0], [1.0, 0.0]]})
        def execute(action_call):
            self.assertIs(action_call, call)
            self.assertEqual(self.memory.items[1].start_node_id, self.context.current_node_id)
            self.assertEqual(self.memory.items[0].start_node_id, "")
            return ActionResult(ok=True, data={"path_xy": [[0.0, 0.0], [1.0, 0.0]]})
        self.state.action_executor = SimpleNamespace(execute=Mock(side_effect=execute))
        with patch("navclaw.agent.visual_step_execution.set_pending_node_move") as record_move:
            _execute_visual_action_call(
                state=self.state, step=self.step, action=action, action_call=call,
                decision_payload={"navigation_mode": selected.to_dict(), "planning_node_id": "virtual-anchor"},
            )
        self.assertEqual(self.step.executed_action["subgoal_id"], "sg1")
        self.assertEqual(self.step.executed_action["subgoal_attempt"], 1)
        self.assertEqual(action.args["subgoal_id"], "sg1")
        self.assertEqual(record_move.call_args.kwargs["subgoal_id"], "sg1")
        self.assertEqual(record_move.call_args.kwargs["subgoal_attempt"], 1)

    def test_stale_attempt_cannot_start_or_execute_after_objective_reopens(self):
        selected = _bind_navigation_subgoal(navigation(subgoal_id="sg0"), self.memory)
        self.memory.apply_agenda_updates([
            {"op": "abandon", "subgoal_id": "sg0", "result": "Earlier doorway interpretation was incorrect."},
            {"op": "reopen", "subgoal_id": "sg0", "position": 0},
        ], [])
        self.step.current_place_node_id = self.context.current_node_id
        self.step.agent_decision = {}
        self.state.action_executor = SimpleNamespace(execute=Mock())
        with self.assertRaisesRegex(ValueError, "stale subgoal attempt"):
            _mark_selected_subgoal_started(state=self.state, step=self.step, navigation_mode=selected)
        with self.assertRaisesRegex(ValueError, "stale subgoal attempt"):
            _execute_visual_action_call(
                state=self.state, step=self.step,
                action=AgentAction(AgentActionType.VISUAL_WAYPOINT, args={"selected_angle_deg": 0}),
                action_call=ActionCall("move_along_path", {}),
                decision_payload={"navigation_mode": selected.to_dict()},
            )
        self.state.action_executor.execute.assert_not_called()
        self.assertEqual(self.memory.items[0].start_node_id, "")
        current = _bind_navigation_subgoal(navigation(subgoal_id="sg0"), self.memory)
        _mark_selected_subgoal_started(state=self.state, step=self.step, navigation_mode=current)
        self.assertEqual(self.memory.items[0].start_node_id, self.context.current_node_id)

    def test_invalid_combined_response_retries_without_partial_mutation(self):
        original = self.memory.to_dict()
        invalid = executive_response([
            {"op": "add", "content": "Inspect a doorway.", "position": 0},
            {"op": "complete", "subgoal_id": "missing", "result": "Unsupported."},
        ], [{"op": "add", "subgoal_id": "sg0", "content": "Kitchen fixtures visible.", "status": "confirmed"}])
        def tpu(_system, content):
            self.assertEqual(self.memory.to_dict(), original)
            if self.client.decide_vln_task_progress_step.call_count == 1:
                return invalid
            self.assertIn("unknown active subgoal", str(content))
            return executive_response()
        self.client.decide_vln_task_progress_step.side_effect = tpu
        result = self.run_loop()
        self.assertEqual(self.client.decide_vln_task_progress_step.call_count, 2)
        self.assertEqual(self.memory.to_dict(), original)
        self.assertEqual(result.task_progress_update_result.applied_updates, [])

    def test_virtual_backtrack_archives_at_physical_node_and_does_not_start_new_work(self):
        anchor = self.state.graph.add_node(position=(1.0, 0.0, 0.0), yaw=0.0, obs_id="obs_anchor")
        self.memory.mark_subgoal_started("sg0", self.context.current_node_id)
        self.client.decide_vln_task_progress_step.side_effect = [
            executive_response(),
            executive_response([
                {"op": "abandon", "subgoal_id": "sg0", "result": "The earlier entry attempt used the wrong doorway."},
                {"op": "add", "content": "Return to the correct kitchen entrance.", "position": 0},
            ]),
        ]
        self.policy.side_effect = [
            navigation("backtrack", backtrack_anchor_node_id=str(anchor.id), backtrack_reason="Earlier doorway needs reconsideration.", backtrack_objective="Inspect the kitchen entrance."),
            navigation(subgoal_id="sg2"),
        ]
        with patch("navclaw.agent.visual_navigation.build_visual_action_context_for_node", return_value=VisualActionContext(current_node_id=str(anchor.id), views=[])), patch("navclaw.agent.visual_navigation.build_vln_landmark_context", return_value=None):
            result = self.run_loop()
        self.assertEqual(result.planning_visual_context.current_node_id, str(anchor.id))
        self.assertEqual(self.memory.history[0]["completion_node_id"], self.context.current_node_id)
        self.assertNotEqual(self.memory.history[0]["completion_node_id"], str(anchor.id))
        self.assertEqual(self.memory.items[0].start_node_id, "")
        self.assertEqual(self.step.episodic_retrieval["physical_current_node_id"], self.context.current_node_id)
        self.assertIsNone(self.state.pending_vln_terminal_check)
        self.assertFalse(self.step.episodic_retrieval["post_approach_terminal_check"])
        for call in self.client.decide_vln_task_progress_step.call_args_list:
            self.assertIn("NavProbe task executive", call.args[0])
            self.assertNotIn("Post-approach terminal check:", str(call.args[1]))

    def test_terminal_continue_accepts_empty_agenda_when_original_goal_is_unmet(self):
        self.memory.apply_agenda_updates([
            {"op": "abandon", "subgoal_id": "sg0", "result": "Current search objective is no longer useful."},
            {"op": "abandon", "subgoal_id": "sg1", "result": "Endpoint estimate contradicted."},
        ], [])
        self.state.pending_vln_terminal_check = {"movement": "stay", "stop_objective": "Stop beside the table."}
        self.client.decide_vln_task_progress_step.return_value = executive_response(terminal={
            "decision": "continue", "missing_constraints": ["The required table-side endpoint remains unconfirmed."],
        })
        self.policy.return_value = navigation("approach_to_stop", subgoal_id=None)
        result = self.run_loop()
        self.assertEqual(result.task_progress_decision.terminal_check_decision, "continue")
        self.assertEqual(self.memory.items, [])
        self.assertIsNone(self.state.pending_vln_terminal_check)
        self.assertEqual(self.policy.call_count, 1)

    def test_terminal_done_retries_until_remaining_objectives_are_resolved(self):
        self.state.pending_vln_terminal_check = {"movement": "stay", "stop_objective": "Stop beside the table."}
        done = {"decision": "done", "missing_constraints": []}
        self.client.decide_vln_task_progress_step.side_effect = [
            executive_response(terminal=done),
            executive_response([
                {"op": "complete", "subgoal_id": "sg0", "result": "Kitchen entry verified."},
                {"op": "complete", "subgoal_id": "sg1", "result": "Current endpoint beside the table verified."},
            ], terminal=done),
        ]
        result = self.run_loop()
        self.assertEqual(result.task_progress_decision.terminal_check_decision, "done")
        self.assertEqual(self.client.decide_vln_task_progress_step.call_count, 2)
        self.assertEqual(self.memory.items, [])
        self.assertEqual(len(self.memory.history), 2)
        self.assertEqual(self.policy.call_count, 0)
        self.assertIn(self.goal_text, str(self.client.decide_vln_task_progress_step.call_args))


if __name__ == "__main__":
    unittest.main()
