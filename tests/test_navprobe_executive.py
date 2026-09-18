from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from navprobe.agent.episodic_retrieval import RetrieveRequest
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.agent.visual_policy import decide_vln_navigation_step
from navprobe.agent.visual_policy import decide_vln_task_progress_step
from navprobe.agent.visual_policy import ensure_task_progress_memory
from navprobe.agent.visual_policy import normalize_vln_navigation_step
from navprobe.agent.visual_policy import normalize_vln_task_progress_step
from navprobe.agent.visual_policy_decisions import TaskProgressDecision
from navprobe.memory.task_progress import TaskProgressMemory
from navprobe.perception.goal_identity import GOAL_KIND_OBJECT_CATEGORY
from navprobe.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION


def agenda(goal="Enter the kitchen, then stop beside the table.", objectives=()):
    memory = TaskProgressMemory()
    memory.initialize_agenda(goal, [{"content": item} for item in objectives])
    return memory


def response(*, updates=(), conditions=(), query=None, conclusion=None, terminal=None):
    payload = {"task_state_assessment": "The observed evidence supports this agenda assessment.", "tool_calls": []}
    if conclusion is not None:
        payload["retrieval_conclusion"] = conclusion
    if conditions:
        payload["tool_calls"].append({"name": "update_predicates", "arguments": {"predicate_updates": list(conditions)}})
    if updates or terminal is not None:
        arguments = {"agenda_updates": list(updates)}
        if terminal is not None:
            arguments["terminal_check"] = terminal
        payload["tool_calls"].append({"name": "update_task_state", "arguments": arguments})
    if query:
        payload["tool_calls"].append({"name": "retrieve", "arguments": {
            "query": query, "items": [{"ref": "n0", "fields": ["rgb"]}],
        }})
    return payload


def assess(memory, answers, **overrides):
    client = SimpleNamespace(decide_vln_task_progress_step=Mock(side_effect=answers))
    kwargs = dict(
        client=client, cache=SimpleNamespace(), task_progress=memory,
        visual_context=VisualActionContext(current_node_id="n1", views=[]),
        memory_index_text="n0: kitchen doorway; available_fields=rgb",
        retrieval_workspace_content=[], retrieve_max_rounds=2, retrieve_completed_rounds=0,
        retrieve_fields_by_ref={"n0": ["rgb"]}, allow_retrieve=True,
        allow_update_progress=True, require_retrieval_conclusion=False,
        original_instruction=memory.original_goal,
    )
    kwargs.update(overrides)
    return decide_vln_task_progress_step(**kwargs), client


def prompt_text(mock):
    return "\n".join(str(item.get("text", "")) for item in mock.call_args.args[1])


class NavProbeExecutiveTests(unittest.TestCase):
    def test_vln_initialization_preserves_original_partial_stair_stop(self):
        goal = "Go upstairs and stop halfway up the staircase."
        client = SimpleNamespace(generate_task_progress_memory=Mock(return_value={
            "agenda": [{"content": goal, "status": "active", "result": ""}],
        }))
        memory = TaskProgressMemory()
        result = ensure_task_progress_memory(
            client=client, task_progress=memory, goal_text=goal,
            task_type=GOAL_KIND_VLN_INSTRUCTION,
        )
        self.assertEqual(result["source"], "llm")
        self.assertEqual(memory.original_goal, goal)
        self.assertEqual(memory.items[0].content, goal)
        self.assertEqual(memory.items[0].subgoal_id, "sg0")
        prompt = client.generate_task_progress_memory.call_args.args[1]
        self.assertIn("stopping partway up or down a staircase", prompt)
        self.assertNotIn("Stair-stop adjustment", prompt)

    def test_objectnav_starts_empty_and_does_not_reinitialize_exhausted_agenda(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                memory = TaskProgressMemory(task_constraints="Reach connected floor near the target." if wrapped else "")
                client = SimpleNamespace(generate_task_progress_memory=Mock())
                kwargs = dict(
                    client=client, task_progress=memory, goal_text="Find a chair.",
                    task_type=GOAL_KIND_VLN_INSTRUCTION if wrapped else GOAL_KIND_OBJECT_CATEGORY,
                )
                self.assertTrue(ensure_task_progress_memory(**kwargs)["initialized"])
                self.assertFalse(ensure_task_progress_memory(**kwargs)["initialized"])
                self.assertEqual(memory.items, [])
                self.assertEqual(memory.original_goal, "Find a chair.")
                client.generate_task_progress_memory.assert_not_called()

    def test_initialization_fallback_retains_exact_instruction(self):
        goal = "Enter the kitchen and stop halfway up the stairs."
        memory = TaskProgressMemory()
        result = ensure_task_progress_memory(
            client=SimpleNamespace(generate_task_progress_memory=Mock(side_effect=ValueError("invalid output"))),
            task_progress=memory, goal_text=goal, task_type=GOAL_KIND_VLN_INSTRUCTION,
        )
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(memory.items[0].content, goal)


    def test_invalid_nonterminal_transaction_retries_without_partial_mutation(self):
        memory = agenda(objectives=["Enter kitchen."])
        before = deepcopy(memory.to_dict())
        invalid = response(updates=[
            {"op": "rewrite", "subgoal_id": "sg0", "content": "Incorrect partial mutation."},
            {"op": "complete", "subgoal_id": "missing", "result": "Unsupported completion."},
        ])
        decision, client = assess(memory, [invalid, response()])
        self.assertEqual(client.decide_vln_task_progress_step.call_count, 2)
        self.assertEqual(memory.to_dict(), before)
        self.assertEqual(decision.progress_updates, [])
        self.assertIn("unknown active subgoal", prompt_text(client.decide_vln_task_progress_step))

    def test_invalid_condition_reference_is_validated_before_handoff(self):
        memory = agenda(objectives=["Inspect counter."])
        invalid = response(conditions=[{"op": "update", "subgoal_id": "sg0", "predicate_id": "absent", "status": "confirmed"}])
        _, client = assess(memory, [invalid, response()])
        self.assertEqual(client.decide_vln_task_progress_step.call_count, 2)
        self.assertEqual(memory.items[0].conditions, [])

    def test_retrieval_can_reopen_history_then_formulate_new_objective_at_budget_limit(self):
        memory = agenda(objectives=["Enter kitchen."])
        memory.apply_agenda_updates(
            [{"op": "complete", "subgoal_id": "sg0", "result": "Believed the room was kitchen."}],
            [{"op": "add", "subgoal_id": "sg0", "content": "Entered the kitchen.", "status": "confirmed"}], "n0",
        )
        first, _ = assess(memory, [response(query="Was n0 really the kitchen?")])
        self.assertIsInstance(first, RetrieveRequest)
        second, _ = assess(memory, [response(
            updates=[{"op": "reopen", "subgoal_id": "sg0", "position": 0}],
            conditions=[{"op": "update", "subgoal_id": "sg0", "predicate_id": "pc0", "status": "unconfirmed"}],
            query="Does the doorway at n0 lead toward kitchen fixtures?",
            conclusion="n0 shows dining furniture; the kitchen-entry judgment was unsupported.",
        )], require_retrieval_conclusion=True, retrieve_completed_rounds=1)
        memory.apply_agenda_updates(list(second.progress_updates), list(second.progress_condition_updates), "n1")
        final, client = assess(memory, [response(
            updates=[{"op": "add", "content": "Inspect the doorway beside the kitchen fixtures.", "position": 0}],
            conclusion="The n0 panorama shows kitchen fixtures beside the overlooked doorway.",
        )], require_retrieval_conclusion=True, retrieve_completed_rounds=2, allow_retrieve=False)
        self.assertEqual(final.progress_updates[0]["op"], "add")
        prompt = prompt_text(client.decide_vln_task_progress_step)
        self.assertIn("n0: kitchen doorway", prompt)
        self.assertIn("remaining rounds: 0", prompt)
        self.assertNotIn("### `retrieve`", prompt)
        self.assertEqual(memory.history[0]["result"], "Believed the room was kitchen.")
        self.assertEqual(memory.history[0]["conditions"][0]["status"], "confirmed")
        self.assertEqual(memory.items[0].conditions[0].status, "unconfirmed")
        self.assertEqual(memory.items[0].attempt, 2)

    def test_final_retrieval_batch_requires_conclusion_even_without_more_retrieval(self):
        memory = agenda(objectives=["Inspect doorway."])
        valid = response(conclusion="n0 leaves the region beyond the doorway unresolved.")
        _, client = assess(memory, [response(), valid],
            allow_retrieve=False, retrieve_completed_rounds=2, require_retrieval_conclusion=True)
        self.assertEqual(client.decide_vln_task_progress_step.call_count, 2)

    def test_negative_search_resolution_allows_continue_with_empty_agenda(self):
        memory = agenda("Find a chair.", ["Inspect the countertop region."])
        decision, _ = assess(memory, [response(
            updates=[{"op": "complete", "subgoal_id": "sg0", "result": "Inspected region; no chair found."}],
            terminal={"decision": "continue", "missing_constraints": ["No target has been reached."]},
        )], goal_kind=GOAL_KIND_OBJECT_CATEGORY, terminal_check_context={"movement": "stay"})
        memory.apply_agenda_updates(decision.progress_updates, decision.progress_condition_updates, "n1")
        self.assertEqual(memory.items, [])
        self.assertEqual(decision.terminal_check_decision, "continue")
        self.assertEqual(memory.history[0]["result"], "Inspected region; no chair found.")

    def test_done_requires_resolving_remaining_agenda_in_same_transaction(self):
        memory = agenda(objectives=["Stop beside table."])
        terminal = {"decision": "done", "missing_constraints": []}
        valid = response(updates=[{"op": "complete", "subgoal_id": "sg0", "result": "Reached table endpoint."}], terminal=terminal)
        decision, client = assess(memory, [response(terminal=terminal), valid],
            terminal_check_context={"movement": "stay", "stop_objective": "Stop beside table."})
        self.assertEqual(client.decide_vln_task_progress_step.call_count, 2)
        self.assertEqual(decision.terminal_check_decision, "done")

    def test_dynamic_protocol_rejects_positional_references(self):
        for invalid in (
            response(updates=[{"op": "update", "index": 0, "status": "done", "result": "Done."}]),
            response(conditions=[{"op": "add", "item_index": 0, "content": "Door seen.", "status": "confirmed"}]),
        ):
            with self.assertRaises(ValueError):
                normalize_vln_task_progress_step(invalid, retrieve_fields_by_ref={}, allow_retrieve=False,
                    allow_update_progress=True, require_retrieval_conclusion=False)

    def test_native_protocol_roundtrip_rejects_legacy_fields(self):
        canonical = response(updates=[{"op": "rewrite", "subgoal_id": "sg0", "content": "Inspect doorway."}])
        kwargs = dict(retrieve_fields_by_ref={}, allow_retrieve=False, allow_update_progress=True,
            require_retrieval_conclusion=False)
        decision = normalize_vln_task_progress_step(canonical, **kwargs)
        self.assertEqual(decision.progress_analysis, canonical["task_state_assessment"])
        self.assertEqual(decision.progress_updates, canonical["tool_calls"][0]["arguments"]["agenda_updates"])
        self.assertEqual(decision.to_dict(), canonical)
        legacy = {"progress_analysis": "The route stage is complete.", "tool_calls": [{
            "name": "update_progress", "arguments": {"progress_updates": [
                {"op": "update", "index": 0, "status": "done", "result": "Entered the kitchen."},
            ]},
        }]}
        with self.assertRaisesRegex(ValueError, "task_state_assessment"):
            normalize_vln_task_progress_step(legacy, **kwargs)
        mixed = response(conditions=[{"op": "update", "subgoal_id": "sg0", "condition_id": "pc0", "status": "confirmed"}])
        with self.assertRaisesRegex(ValueError, "predicate_id"):
            normalize_vln_task_progress_step(mixed, **kwargs)

    def test_terminal_retry_feedback_uses_mainline_schema_names(self):
        memory = agenda(objectives=["Stop beside table."])
        valid = response(terminal={"decision": "continue", "missing_constraints": ["Endpoint unresolved."]})
        _, client = assess(memory, [response(), valid], terminal_check_context={"movement": "stay"})
        prompt = prompt_text(client.decide_vln_task_progress_step)
        self.assertIn("update_task_state.arguments.terminal_check", prompt)
        for legacy in ("progress_analysis", "progress_updates", "update_progress", "TPU"):
            self.assertNotIn(legacy, prompt)

    def test_initializer_uses_agenda_by_default(self):
        memory = TaskProgressMemory()
        client = SimpleNamespace(generate_task_progress_memory=Mock(return_value={
            "agenda": [{"content": "Enter kitchen.", "status": "active", "result": ""}],
        }))
        result = ensure_task_progress_memory(client=client, task_progress=memory, goal_text="Enter kitchen.",
            task_type=GOAL_KIND_VLN_INSTRUCTION)
        self.assertEqual(result["source"], "llm")
        self.assertTrue(memory.agenda_initialized)
        self.assertIn('"agenda"', client.generate_task_progress_memory.call_args.args[1])

    def test_navigation_uses_active_stable_id_and_preserves_partial_stair_endpoint(self):
        memory = agenda("Go halfway up the stairs and stop.", ["Go halfway up the stairs and stop."])
        payload = {"vertical_transition": {
            "subgoal_id": "sg0", "vertical_direction": "up",
            "waypoint_target": "Go halfway up the stairs and stop.",
            "reason": "The required ascending staircase is visible and accessible.",
        }}
        client = SimpleNamespace(decide_vln_navigation_step=Mock(return_value=payload))
        decision = decide_vln_navigation_step(
            client=client, cache=SimpleNamespace(), task_progress=memory,
            visual_context=VisualActionContext(current_node_id="n1", views=[]),
            latest_task_progress=TaskProgressDecision(progress_analysis="Stairs identified.", progress_reasoning=""),
            retrieval_workspace_content=[], original_instruction=memory.original_goal,
        )
        self.assertEqual(decision.subgoal_id, "sg0")
        self.assertEqual(decision.waypoint_target, memory.items[0].content)
        self.assertIsNone(decision.task_item_index)
        prompt = prompt_text(client.decide_vln_navigation_step)
        self.assertIn("partial-stair stopping", prompt)
        self.assertIn("next decision step", prompt)
        self.assertIn("self-contained", prompt)
        self.assertIn("Semantic skill selection", prompt)
        self.assertIn("The waypoint grounder receives this selected action", prompt)
        for legacy in ("TPU", "PCNP", "task-progress", "Latest progress updates"):
            self.assertNotIn(legacy, prompt)

    def test_navigation_rejects_unknown_ids_and_allows_null_for_empty_agenda_actions(self):
        latest = TaskProgressDecision(progress_analysis="Assessed.", progress_reasoning="")
        kwargs = dict(latest_task_progress=latest, system_owned_waypoint_objective=True)
        with self.assertRaisesRegex(ValueError, "active subgoal_id"):
            normalize_vln_navigation_step({"go_to_waypoint": {"subgoal_id": "historical", "direction": "front", "reason": "Explore."}},
                active_subgoal_ids={"sg0"}, **kwargs)
        stop = {"approach_to_stop": {"subgoal_id": None, "movement": "stay", "stop_objective": "Stop beside table.", "reason": "Endpoint observed."}}
        with self.assertRaisesRegex(ValueError, "empty agenda"):
            normalize_vln_navigation_step(stop, active_subgoal_ids={"sg0"}, **kwargs)
        decision = normalize_vln_navigation_step(stop, active_subgoal_ids=set(), **kwargs)
        self.assertIsNone(decision.subgoal_id)
        self.assertEqual(decision.action_mode, "approach_to_stop")
        actions = [
            {"go_to_waypoint": {"subgoal_id": None, "direction": "front", "reason": "Explore the visible room for the original object goal."}},
            {"vertical_transition": {"subgoal_id": None, "vertical_direction": "up", "waypoint_target": "Ascend halfway and stop as instructed.", "reason": "The original task requires this visible staircase."}},
            {"backtrack": {"subgoal_id": None, "anchor_node_id": "n0", "objective": "Reassess the original route at the junction.", "reason": "The recorded junction has an unexplored route."}},
        ]
        for payload in actions:
            with self.subTest(action=next(iter(payload))):
                decision = normalize_vln_navigation_step(payload, active_subgoal_ids=set(), allowed_backtrack_node_ids={"n0"}, **kwargs)
                self.assertIsNone(decision.subgoal_id)
                with self.assertRaisesRegex(ValueError, "empty agenda"):
                    normalize_vln_navigation_step(payload, active_subgoal_ids={"sg0"}, allowed_backtrack_node_ids={"n0"}, **kwargs)

    def test_virtual_reference_prompt_distinguishes_observation_from_execution(self):
        memory = agenda(objectives=["Enter kitchen."])
        _, client = assess(memory, [response()], planning_reference_panorama=True,
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
            backtrack_context_text="Physical current node: n1. Planning reference: n0.")
        prompt = prompt_text(client.decide_vln_task_progress_step)
        self.assertIn("Physical current node: n1", prompt)
        self.assertIn("A planning-reference switch supplies no new execution evidence", prompt)
        self.assertIn("actual observations and executed trajectories", prompt)


if __name__ == "__main__":
    unittest.main()
