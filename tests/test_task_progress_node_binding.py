from __future__ import annotations

from types import SimpleNamespace
import unittest

from navclaw.agent.visual_action_context import VisualActionContext
from navclaw.agent.visual_navigation import _current_active_task_progress_item
from navclaw.agent.visual_policy import ensure_task_progress_memory
from navclaw.memory.task_progress import TaskProgressItem
from navclaw.memory.task_progress import TaskProgressCondition
from navclaw.memory.task_progress import TaskProgressMemory


def _agenda(items: list[TaskProgressItem]) -> TaskProgressMemory:
    memory = TaskProgressMemory()
    memory.initialize_agenda("Follow the route and stop at the endpoint.", items)
    return memory


class TaskProgressNodeBindingTest(unittest.TestCase):
    def test_initialization_defers_start_binding_until_objective_selection(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(content="Exit the room."),
                TaskProgressItem(content="Turn left."),
            ]
        )

        ensure_task_progress_memory(
            client=SimpleNamespace(),
            task_progress=memory,
            goal_text="Exit the room and turn left.",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )

        self.assertEqual(memory.items[0].start_node_id, "")
        self.assertEqual(memory.items[1].start_node_id, "")
        memory.mark_subgoal_started("sg1", "n0")
        self.assertEqual(memory.items[0].start_node_id, "")
        self.assertEqual(memory.items[1].start_node_id, "n0")

    def test_completion_archives_binding_without_starting_next_subgoal(self) -> None:
        memory = _agenda([
            TaskProgressItem("Exit the room.", start_node_id="n0"),
            TaskProgressItem("Turn left."),
        ])
        result = memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Exited through the doorway."},
        ], [], "n2")
        self.assertEqual(len(result.applied_updates), 1)
        self.assertEqual(memory.history[0]["start_node_id"], "n0")
        self.assertEqual(memory.history[0]["completion_node_id"], "n2")
        self.assertEqual(memory.items[0].subgoal_id, "sg1")
        self.assertEqual(memory.items[0].start_node_id, "")
        memory.mark_subgoal_started("sg1", "n3")
        self.assertEqual(memory.items[0].start_node_id, "n3")

    def test_added_subgoal_starts_only_after_explicit_selection(self) -> None:
        memory = _agenda([TaskProgressItem("Exit the room.", start_node_id="n0")])
        memory.apply_agenda_updates([
            {"op": "add", "content": "Continue through the doorway.", "position": 1},
        ], [], "n1")
        self.assertEqual(memory.items[1].start_node_id, "")
        memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Exited the room."},
        ], [], "n2")
        self.assertEqual(memory.items[0].start_node_id, "")
        memory.mark_subgoal_started("sg1", "n3")
        memory.mark_subgoal_started("sg1", "n4")
        self.assertEqual(memory.items[0].start_node_id, "n3")

    def test_multiple_completions_bind_history_without_starting_remaining_subgoal(self) -> None:
        memory = _agenda([
            TaskProgressItem("Exit the room.", start_node_id="n0"),
            TaskProgressItem("Turn left.", start_node_id="n1"),
            TaskProgressItem("Stop beyond the doorway."),
        ])
        memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Exited the room."},
            {"op": "complete", "subgoal_id": "sg1", "result": "Turned left."},
        ], [], "n2")
        self.assertEqual([record["completion_node_id"] for record in memory.history], ["n2", "n2"])
        self.assertEqual([record["start_node_id"] for record in memory.history], ["n0", "n1"])
        self.assertEqual(memory.items[0].subgoal_id, "sg2")
        self.assertEqual(memory.items[0].start_node_id, "")

    def test_completion_cannot_copy_node_annotation_into_content(self) -> None:
        memory = _agenda([TaskProgressItem("Exit the room.", start_node_id="n0")])
        before = memory.to_dict()
        with self.assertRaisesRegex(ValueError, "invalid agenda update fields"):
            memory.apply_agenda_updates([{
                "op": "complete", "subgoal_id": "sg0", "result": "Exited.",
                "content": "Exit the room.; nodes=start=n0",
            }], [], "n1")
        self.assertEqual(memory.to_dict(), before)

    def test_rewrite_preserves_existing_node_binding(self) -> None:
        memory = _agenda([TaskProgressItem("Exit.", start_node_id="n0")])
        memory.apply_agenda_updates([{
            "op": "rewrite", "subgoal_id": "sg0", "content": "Exit through the bedroom doorway.",
        }], [], "n1")
        self.assertEqual(memory.items[0].content, "Exit through the bedroom doorway.")
        self.assertEqual(memory.items[0].start_node_id, "n0")
        self.assertEqual(memory.items[0].completion_node_id, "")

    def test_unstarted_subgoal_defaults_to_empty_node_bindings(self) -> None:
        memory = TaskProgressMemory.from_dict(
            {
                "progress_items": [
                    {
                        "content": "Exit the room.",
                        "status": "active",
                        "result": "",
                    }
                ]
            }
        )

        self.assertEqual(memory.items[0].start_node_id, "")
        self.assertEqual(memory.items[0].completion_node_id, "")

    def test_prompt_can_hide_node_bindings_for_no_graph_ablation(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Exit the room.",
                    status="done",
                    result="Exited.",
                    start_node_id="n0",
                    completion_node_id="n2",
                )
            ]
        )

        memory.initialize_agenda("Exit the room.", memory.items)
        visible = memory.format_for_prompt(include_node_bindings=True)
        hidden = memory.format_for_prompt(include_node_bindings=False)

        self.assertIn("node span: n0 -> n2", visible)
        self.assertNotIn("n0", hidden)
        self.assertNotIn("n2", hidden)

    def test_vln_initialization_prompt_preserves_direction_commands_without_conditions(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.system_prompt = ""
                self.user_prompt = ""

            def generate_task_progress_memory(self, system_prompt, user_prompt):
                self.system_prompt = str(system_prompt)
                self.user_prompt = str(user_prompt)
                return {
                    "agenda": [
                        {
                            "content": "Turn around and turn right in the doorway after the set of red chairs.",
                            "status": "active",
                            "result": "",
                        },
                        {
                            "content": "Wait just outside this same doorway.",
                            "status": "active",
                            "result": "",
                        },
                    ]
                }

        client = _Client()
        memory = TaskProgressMemory()

        result = ensure_task_progress_memory(
            client=client,
            task_progress=memory,
            goal_text=(
                "Turn around and turn right in the doorway after the set of red chairs. "
                "Wait just outside this same doorway."
            ),
            task_type="vln_instruction",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )

        self.assertTrue(result["initialized"])
        self.assertNotIn("navigation decision points", client.user_prompt)
        self.assertNotIn("cue for the direction", client.user_prompt)
        self.assertNotIn("action-driving route constraint", client.user_prompt)
        self.assertNotIn("do not create pure orientation", client.user_prompt.lower())
        self.assertIn("Initial Task Executive", client.system_prompt)
        self.assertIn("instruction order", client.user_prompt)
        self.assertIn("defining landmarks and spatial relations", client.user_prompt)
        self.assertIn("Evidence predicates are initially empty", client.user_prompt)
        self.assertNotIn("Example:", client.user_prompt)
        self.assertEqual(client.user_prompt.count("red chairs"), 1)
        self.assertNotIn('"conditions"', client.user_prompt)
        self.assertEqual(len(memory.items), 2)
        self.assertTrue(memory.items[0].content.startswith("Turn around and turn right"))
        self.assertEqual(memory.items[1].content, "Wait just outside this same doorway.")
        self.assertEqual(memory.items[0].conditions, [])

    def test_vln_initialization_preserves_intermediate_stair_stops(self) -> None:
        for direction, endpoint in (("up", "upper"), ("down", "lower")):
            with self.subTest(direction=direction):
                captured = {}
                contents = [
                    f"Go {direction} the stairs and stop on the third step.",
                ]

                def generate(system_prompt, user_prompt):
                    captured["user_prompt"] = user_prompt
                    return {
                        "agenda": [
                            {"content": content, "status": "active", "result": ""}
                            for content in contents
                        ]
                    }

                memory = TaskProgressMemory()
                result = ensure_task_progress_memory(
                    client=SimpleNamespace(generate_task_progress_memory=generate),
                    task_progress=memory,
                    goal_text=f"Go {direction} the stairs and stop on the third step.",
                    task_type="vln_instruction",
                    visual_context=VisualActionContext(current_node_id="n0", views=[]),
                )

                prompt = captured["user_prompt"]
                self.assertIn("never replace a partial stair objective with a full-floor transition", prompt)
                self.assertIn("Preserve all stated stopping constraints", prompt)
                self.assertNotIn("Example:", prompt)
                self.assertEqual(result["source"], "llm")
                self.assertEqual([item.content for item in memory.items], contents)
                self.assertTrue(all(item.status == "active" for item in memory.items))
                self.assertTrue(all(item.conditions == [] for item in memory.items))

    def test_object_navigation_initialization_omits_stair_stop_adjustment(self) -> None:
        captured = {}

        def generate(system_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return {
                "agenda": [
                    {"content": "Reach the chair.", "status": "active", "result": ""}
                ]
            }

        ensure_task_progress_memory(
            client=SimpleNamespace(generate_task_progress_memory=generate),
            task_progress=TaskProgressMemory(),
            goal_text="Find a chair.",
            task_type="object_category",
        )

        self.assertEqual(captured, {})

    def test_vln_initialization_rejects_online_conditions_and_done_status(self) -> None:
        client = SimpleNamespace(
            generate_task_progress_memory=lambda _system, _user: {
                "agenda": [
                    {
                        "content": "Exit the room.",
                        "status": "done",
                        "result": "Exited.",
                        "conditions": [],
                    }
                ]
            }
        )
        memory = TaskProgressMemory()

        result = ensure_task_progress_memory(
            client=client,
            task_progress=memory,
            goal_text="Exit the room.",
            task_type="vln_instruction",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )

        self.assertEqual(result["source"], "fallback")
        self.assertIn("must contain exactly", result["error"])
        self.assertTrue(all(item.status == "active" for item in memory.items))
        self.assertTrue(all(item.conditions == [] for item in memory.items))

    def test_progress_agent_can_create_conditions_on_an_empty_item(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Veer left and walk through the kitchen.",
                    start_node_id="n0",
                )
            ]
        )

        memory.initialize_agenda("Follow the route.", memory.items)
        result = memory.apply_agenda_updates(
            [],
            [
                {
                    "subgoal_id": "sg0",
                    "op": "add",
                    "content": "Veer left.",
                    "status": "confirmed",
                },
                {
                    "subgoal_id": "sg0",
                    "op": "add",
                    "content": "Walk through the kitchen.",
                    "status": "unconfirmed",
                },
            ],
            current_node_id="n1",
        )

        self.assertEqual(len(result.applied_updates), 2)
        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(
            [item.predicate_id for item in memory.items[0].conditions],
            ["pc0", "pc1"],
        )
        self.assertTrue(
            all(item.update_node_id == "n1" for item in memory.items[0].conditions)
        )

    def test_partial_condition_update_keeps_item_active(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Veer left and walk through the kitchen.",
                    start_node_id="n0",
                    conditions=[
                        TaskProgressCondition("pc0", "n0", "Veer left."),
                        TaskProgressCondition(
                            "pc1",
                            "n0",
                            "Walk through the kitchen.",
                        ),
                    ],
                )
            ]
        )

        memory.initialize_agenda("Follow the route.", memory.items)
        result = memory.apply_agenda_updates(
            [],
            [
                {
                    "subgoal_id": "sg0",
                    "op": "update",
                    "predicate_id": "pc0",
                    "status": "confirmed",
                }
            ],
            current_node_id="n1",
        )

        self.assertEqual(len(result.applied_updates), 1)
        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].result, "")
        self.assertEqual(memory.items[0].completion_node_id, "")
        self.assertEqual(memory.items[0].conditions[0].status, "confirmed")
        self.assertEqual(memory.items[0].conditions[0].update_node_id, "n1")
        self.assertEqual(memory.items[0].conditions[1].status, "unconfirmed")

    def test_unchanged_condition_status_is_not_rewritten_at_a_new_node(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Exit the room.",
                    conditions=[
                        TaskProgressCondition("pc0", "n0", "Exit the room.")
                    ],
                )
            ]
        )

        memory.initialize_agenda("Follow the route.", memory.items)
        result = memory.apply_agenda_updates(
            [],
            [
                {
                    "subgoal_id": "sg0",
                    "op": "update",
                    "predicate_id": "pc0",
                    "status": "unconfirmed",
                }
            ],
            current_node_id="n1",
        )

        self.assertEqual(len(result.applied_updates), 1)
        self.assertEqual(result.skipped_updates, [])
        self.assertEqual(memory.items[0].conditions[0].update_node_id, "n0")

    def test_completion_archives_final_predicate_evidence_atomically(self) -> None:
        memory = _agenda([
            TaskProgressItem("Veer left and walk through the kitchen.", start_node_id="n0", conditions=[
                TaskProgressCondition("pc0", "n1", "Veer left.", "confirmed"),
                TaskProgressCondition("pc1", "n0", "Walk through the kitchen."),
            ]),
            TaskProgressItem("Take the left turn."),
        ])
        memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Veered left and crossed the kitchen."},
        ], [{"op": "update", "subgoal_id": "sg0", "predicate_id": "pc1", "status": "confirmed"}], "n2")
        self.assertEqual(memory.history[0]["completion_node_id"], "n2")
        self.assertEqual(memory.history[0]["conditions"][1]["update_node_id"], "n2")
        self.assertEqual(memory.history[0]["conditions"][1]["status"], "confirmed")
        self.assertEqual(memory.items[0].start_node_id, "")

    def test_transition_predicate_stays_with_parent_subgoal(self) -> None:
        memory = _agenda([
            TaskProgressItem("Complete the first route stage.", start_node_id="n0"),
            TaskProgressItem("Continue to the next route target."),
        ])
        memory.apply_agenda_updates([], [{
            "op": "add", "subgoal_id": "sg0", "content": "The first stage connects to the next target.",
            "status": "unconfirmed",
        }], "n1")
        self.assertEqual(memory.items[1].conditions, [])
        self.assertEqual(memory.items[1].start_node_id, "")
        memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Completed the first route stage."},
        ], [{"op": "update", "subgoal_id": "sg0", "predicate_id": "pc0", "status": "confirmed"}], "n2")
        self.assertEqual(memory.history[0]["conditions"][0]["status"], "confirmed")
        self.assertEqual(memory.items[0].subgoal_id, "sg1")
        self.assertEqual(memory.items[0].conditions, [])
        self.assertEqual(memory.items[0].result, "")
        self.assertEqual(memory.items[0].start_node_id, "")

    def test_all_confirmed_conditions_do_not_force_item_done(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Continue through the kitchen and exit it.",
                    conditions=[
                        TaskProgressCondition("pc0", "n1", "Entered the kitchen.")
                    ],
                )
            ]
        )

        memory.initialize_agenda("Follow the route.", memory.items)
        result = memory.apply_agenda_updates(
            [],
            [
                {
                    "subgoal_id": "sg0",
                    "op": "update",
                    "predicate_id": "pc0",
                    "status": "confirmed",
                }
            ],
            current_node_id="n1",
        )

        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].conditions[0].status, "confirmed")
        self.assertEqual(len(result.applied_updates), 1)

    def test_completion_does_not_require_predicate_entries(self) -> None:
        memory = _agenda([TaskProgressItem("Exit the kitchen.", start_node_id="n0")])
        result = memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Exited the kitchen."},
        ], [], "n2")
        self.assertEqual(len(result.applied_updates), 1)
        self.assertEqual(memory.items, [])
        self.assertEqual(memory.history[0]["conditions"], [])
        self.assertEqual(memory.history[0]["completion_node_id"], "n2")

    def test_reopen_revises_new_attempt_predicate_without_mutating_history(self) -> None:
        memory = _agenda([TaskProgressItem(
            "Pass through the kitchen.", status="done", result="Passed through the kitchen.",
            start_node_id="n0", completion_node_id="n2", conditions=[
                TaskProgressCondition("pc0", "n2", "Pass through the kitchen.", "confirmed"),
            ],
        )])
        before = memory.to_dict()["history"]
        memory.apply_agenda_updates([
            {"op": "reopen", "subgoal_id": "sg0", "position": 0},
        ], [{"op": "update", "subgoal_id": "sg0", "predicate_id": "pc0", "status": "unconfirmed"}], "n3")
        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].result, "")
        self.assertEqual(memory.items[0].start_node_id, "")
        self.assertEqual(memory.items[0].completion_node_id, "")
        self.assertEqual(memory.items[0].attempt, 2)
        self.assertEqual(memory.items[0].conditions[0].update_node_id, "n3")
        self.assertEqual(memory.items[0].conditions[0].status, "unconfirmed")
        self.assertEqual(memory.history, before)

    def test_reopen_can_remove_last_predicate_without_mutating_history(self) -> None:
        memory = _agenda([TaskProgressItem(
            "Exit the room.", status="done", result="Exited the room.",
            start_node_id="n0", completion_node_id="n1", conditions=[
                TaskProgressCondition("pc0", "n1", "Exit the room.", "confirmed"),
            ],
        )])
        before = memory.to_dict()["history"]
        memory.apply_agenda_updates([
            {"op": "reopen", "subgoal_id": "sg0", "position": 0},
        ], [{"op": "remove", "subgoal_id": "sg0", "predicate_id": "pc0"}], "n2")
        self.assertEqual(memory.items[0].conditions, [])
        self.assertIn("predicates: empty", memory.format_for_prompt(show_empty_progress_conditions=True))
        self.assertNotIn("predicates: empty", memory.format_for_prompt())
        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].result, "")
        self.assertEqual(memory.items[0].completion_node_id, "")
        self.assertEqual(memory.history, before)

    def test_progress_conditions_support_add_rewrite_remove_and_round_trip(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Cross the room.",
                    conditions=[
                        TaskProgressCondition("pc0", "n0", "Enter the room."),
                        TaskProgressCondition("pc1", "n0", "Reach the far side."),
                    ],
                )
            ]
        )

        memory.initialize_agenda("Follow the route.", memory.items)
        result = memory.apply_agenda_updates(
            [],
            [
                {
                    "subgoal_id": "sg0",
                    "op": "add",
                    "content": "Pass the table.",
                    "status": "unconfirmed",
                },
                {
                    "subgoal_id": "sg0",
                    "op": "rewrite",
                    "predicate_id": "pc1",
                    "content": "Reach the doorway on the far side.",
                    "status": "confirmed",
                },
                {
                    "subgoal_id": "sg0",
                    "op": "remove",
                    "predicate_id": "pc0",
                },
            ],
            current_node_id="n1",
        )

        self.assertEqual(len(result.applied_updates), 3)
        self.assertEqual([update["op"] for update in result.applied_updates], ["add", "rewrite", "remove"])
        self.assertEqual(
            [item["predicate_id"] for item in result.applied_updates[0]["conditions"]],
            ["pc0", "pc1", "pc2"],
        )
        self.assertEqual(
            result.applied_updates[-1]["conditions"],
            [item.to_dict() for item in memory.items[0].conditions],
        )
        self.assertEqual(
            [(item.predicate_id, item.content) for item in memory.items[0].conditions],
            [
                ("pc1", "Reach the doorway on the far side."),
                ("pc2", "Pass the table."),
            ],
        )
        self.assertEqual(memory.items[0].conditions[0].status, "confirmed")
        self.assertEqual(memory.items[0].conditions[0].update_node_id, "n1")
        restored = TaskProgressMemory.from_dict(memory.to_dict())
        self.assertEqual(restored.to_dict(), memory.to_dict())

    def test_initial_objectives_keep_empty_predicates(self) -> None:
        memory = TaskProgressMemory.from_dict(
            {
                "progress_items": [
                    {
                        "content": "Exit the room.",
                        "status": "active",
                        "result": "",
                        "start_node_id": "n0",
                    }
                ]
            }
        )

        ensure_task_progress_memory(
            client=SimpleNamespace(),
            task_progress=memory,
            goal_text="Exit the room.",
            task_type="vln_instruction",
            visual_context=VisualActionContext(current_node_id="n1", views=[]),
        )

        self.assertEqual(memory.items[0].conditions, [])

    def test_legacy_predicate_identifiers_are_not_converted(self) -> None:
        with self.assertRaises(ValueError):
            TaskProgressCondition.from_dict({
                "condition_id": "pc0", "content": "Cross the doorway.",
                "status": "confirmed", "update_node_id": "n0",
            })

    def test_only_canonical_progress_items_snapshot_key_is_read(self) -> None:
        for alias in ("items", "todos"):
            with self.subTest(alias=alias):
                memory = TaskProgressMemory.from_dict({alias: [{"content": "Obsolete fixed-list task."}]})
                self.assertEqual(memory.items, [])
        self.assertEqual(TaskProgressMemory().format_for_prompt(), "empty")
        with self.assertRaisesRegex(ValueError, "agenda must be initialized"):
            TaskProgressMemory(items=[TaskProgressItem("Uninitialized objective.")]).format_for_prompt()

    def test_candidate_source_metadata_survives_native_history_and_round_trip(self) -> None:
        candidate = {"candidate_id": "c7", "category": "chair", "source_view": {"node_id": "n0", "image_path": "view.jpg"}}
        memory = _agenda([TaskProgressItem("Inspect the chair.", kind="verify_candidate", candidate=candidate)])
        candidate["source_view"]["node_id"] = "external-change"
        self.assertEqual(memory.items[0].candidate["source_view"]["node_id"], "n0")
        memory.apply_agenda_updates([
            {"op": "complete", "subgoal_id": "sg0", "result": "Chair inspected."},
        ], [], "n1")
        restored = TaskProgressMemory.from_dict(memory.to_dict())
        restored.apply_agenda_updates([{"op": "reopen", "subgoal_id": "sg0", "position": 0}], [], "n2")
        self.assertEqual(restored.items[0].kind, "verify_candidate")
        self.assertEqual(restored.items[0].candidate["candidate_id"], "c7")
        restored.items[0].candidate["source_view"]["node_id"] = "new-attempt"
        self.assertEqual(restored.history[0]["candidate"]["source_view"]["node_id"], "n0")
        self.assertEqual(memory.history[0]["candidate"]["source_view"]["node_id"], "n0")

    def test_waypoint_objective_lists_only_unconfirmed_progress_conditions(self) -> None:
        memory = TaskProgressMemory(
            items=[
                TaskProgressItem(
                    content="Veer left and walk through the kitchen.",
                    conditions=[
                        TaskProgressCondition(
                            "pc0", "n1", "Veer left.", "confirmed"
                        ),
                        TaskProgressCondition(
                            "pc1", "n1", "Walk through the kitchen."
                        ),
                    ],
                )
            ]
        )

        objective = _current_active_task_progress_item(memory)

        self.assertIn("Walk through the kitchen.", objective)
        self.assertNotIn("Unconfirmed progress conditions:\n- Veer left.", objective)


if __name__ == "__main__":
    unittest.main()
