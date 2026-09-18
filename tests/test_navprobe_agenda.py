from copy import deepcopy
import unittest

from navprobe.memory.task_progress import TaskProgressItem, TaskProgressMemory


class NavProbeAgendaTests(unittest.TestCase):
    def setUp(self):
        self.memory = TaskProgressMemory()
        self.memory.initialize_agenda(
            "Enter the kitchen, then stop beside the table.",
            [TaskProgressItem("Enter the kitchen."), TaskProgressItem("Stop beside the table.")],
        )

    def apply(self, updates, conditions=None, node="n1"):
        return self.memory.apply_agenda_updates(updates, conditions or [], node)

    def test_editing_and_reordering_preserve_objective_identity_and_evidence(self):
        self.memory.mark_subgoal_started("sg0", "n0")
        self.apply(
            [{"op": "add", "content": "Inspect the doorway.", "position": 0}],
            [{"op": "add", "subgoal_id": "sg0", "content": "Kitchen fixtures identified.", "status": "confirmed"}],
        )
        self.apply([
            {"op": "rewrite", "subgoal_id": "sg0", "content": "Enter through the kitchen doorway."},
            {"op": "reorder", "subgoal_ids": ["sg1", "sg2", "sg0"]},
        ])
        item = self.memory.items[-1]
        self.assertEqual(item.subgoal_id, "sg0")
        self.assertEqual(item.start_node_id, "n0")
        self.assertEqual(item.conditions[0].status, "confirmed")
        self.assertEqual(self.memory.original_goal, "Enter the kitchen, then stop beside the table.")

    def test_resolution_archives_latest_predicates_and_spatial_references(self):
        self.memory.mark_subgoal_started("sg0", "n0")
        self.apply(
            [{"op": "complete", "subgoal_id": "sg0", "result": "Entered the kitchen."}],
            [{"op": "add", "subgoal_id": "sg0", "content": "Kitchen threshold crossed.", "status": "confirmed"}],
            node="n2",
        )
        record = self.memory.history[0]
        self.assertEqual(record["outcome"], "completed")
        self.assertEqual((record["start_node_id"], record["completion_node_id"]), ("n0", "n2"))
        self.assertEqual(record["conditions"][0]["update_node_id"], "n2")
        self.assertEqual([item.subgoal_id for item in self.memory.items], ["sg1"])

    def test_abandonment_and_reopening_preserve_prior_attempt(self):
        self.apply(
            [{"op": "abandon", "subgoal_id": "sg0", "result": "Doorway leads to the wrong room."}],
            [{"op": "add", "subgoal_id": "sg0", "content": "Kitchen entrance identified.", "status": "unconfirmed"}],
        )
        previous = deepcopy(self.memory.history[0])
        self.apply([{"op": "reopen", "subgoal_id": "sg0", "position": 0}])
        item = self.memory.items[0]
        self.assertEqual((item.subgoal_id, item.attempt, item.status), ("sg0", 2, "active"))
        self.assertEqual((item.result, item.start_node_id, item.completion_node_id), ("", "", ""))
        self.apply([], [{"op": "rewrite", "subgoal_id": "sg0", "predicate_id": "pc0", "content": "Another doorway is the kitchen entrance.", "status": "confirmed"}])
        self.assertEqual(self.memory.history[0], previous)
        self.memory.mark_subgoal_started("sg0", "n3")
        self.apply([{"op": "complete", "subgoal_id": "sg0", "result": "Entered via the correct doorway."}], node="n4")
        self.assertEqual([record["attempt"] for record in self.memory.history], [1, 2])
        self.assertEqual(self.memory.history[-1]["start_node_id"], "n3")

    def test_invalid_batch_leaves_conditions_agenda_history_and_id_counter_unchanged(self):
        before = self.memory.to_dict()
        with self.assertRaisesRegex(ValueError, "unknown active"):
            self.apply([
                {"op": "add", "content": "Inspect the room.", "position": 0},
                {"op": "complete", "subgoal_id": "missing", "result": "Unsupported."},
            ], [{"op": "add", "subgoal_id": "sg0", "content": "Doorway found.", "status": "confirmed"}])
        self.assertEqual(self.memory.to_dict(), before)
        invalid = [
            {"op": "reorder", "subgoal_ids": ["sg0", "sg0"]},
            {"op": "add", "content": "Inspect.", "position": True},
            {"op": "complete", "subgoal_id": "sg0", "result": ""},
            {"op": "reopen", "subgoal_id": "sg0", "position": 0},
            {"op": "rewrite", "subgoal_id": "sg0", "content": ""},
        ]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.apply([update])
            self.assertEqual(self.memory.to_dict(), before)

    def test_reopened_predicates_change_in_same_batch_without_changing_archived_evidence(self):
        self.apply(
            [{"op": "complete", "subgoal_id": "sg0", "result": "Kitchen entry initially accepted."}],
            [{"op": "add", "subgoal_id": "sg0", "content": "Kitchen threshold crossed.", "status": "confirmed"}],
        )
        archived = deepcopy(self.memory.history[0])
        self.apply([
            {"op": "reopen", "subgoal_id": "sg0", "position": 0},
            {"op": "complete", "subgoal_id": "sg1", "result": "Original table-side endpoint verified."},
        ], [
            {"op": "update", "subgoal_id": "sg0", "predicate_id": "pc0", "status": "unconfirmed"},
            {"op": "add", "subgoal_id": "sg1", "content": "Agent stands beside the required table.", "status": "confirmed"},
        ], node="n3")
        self.assertEqual(self.memory.items[0].conditions[0].status, "unconfirmed")
        self.assertEqual(self.memory.items[0].conditions[0].update_node_id, "n3")
        self.assertEqual(self.memory.items[0].attempt, 2)
        self.assertEqual(self.memory.history[0], archived)
        self.assertEqual(self.memory.history[1]["conditions"][0]["status"], "confirmed")

    def test_invalid_deferred_predicate_rolls_back_reopen_and_other_changes(self):
        self.apply([{"op": "abandon", "subgoal_id": "sg0", "result": "Earlier doorway was incorrect."}])
        before = self.memory.to_dict()
        with self.assertRaisesRegex(ValueError, "unknown progress condition"):
            self.apply([
                {"op": "add", "content": "Inspect the adjacent doorway.", "position": 0},
                {"op": "reopen", "subgoal_id": "sg0", "position": 0},
            ], [{"op": "update", "subgoal_id": "sg0", "predicate_id": "missing", "status": "unconfirmed"}])
        self.assertEqual(self.memory.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "unknown active subgoal"):
            self.apply([], [{"op": "add", "subgoal_id": "sg0", "content": "Another doorway observed.", "status": "confirmed"}])
        self.assertEqual(self.memory.to_dict(), before)

    def test_round_trip_keeps_history_and_stable_floor_transition_records(self):
        self.memory.completed_stair_items = {0: {"edge_id": "e0"}, "sg0:1": {"edge_id": "e1"}}
        self.apply([{"op": "complete", "subgoal_id": "sg0", "result": "Kitchen entered."}])
        restored = TaskProgressMemory.from_dict(self.memory.to_dict())
        self.assertEqual(restored.to_dict(), self.memory.to_dict())
        restored.apply_agenda_updates([{"op": "add", "content": "Inspect the table.", "position": 0}], [])
        self.assertEqual(restored.items[0].subgoal_id, "sg2")
        restored.history[0]["result"] = "Changed only on the copy."
        self.assertEqual(self.memory.history[0]["result"], "Kitchen entered.")

    def test_empty_object_search_agenda_stays_initialized_after_negative_search(self):
        memory = TaskProgressMemory()
        memory.initialize_agenda("Find a mug.", [])
        memory.initialize_agenda("Find a mug.", [])
        self.assertEqual(memory.items, [])
        memory.apply_agenda_updates([{"op": "add", "content": "Inspect the countertop.", "position": 0}], [])
        memory.apply_agenda_updates([{"op": "complete", "subgoal_id": "sg0", "result": "Countertop inspected; no mug found."}], [])
        restored = TaskProgressMemory.from_dict(memory.to_dict())
        restored.initialize_agenda("Find a mug.", [])
        self.assertEqual(restored.items, [])
        self.assertTrue(restored.agenda_initialized)
        text = restored.format_for_prompt()
        for expected in ("Original task goal:", "Find a mug.", "Active subgoal agenda", "Execution history:", "no mug found"):
            self.assertIn(expected, text)

    def test_virtual_reassessment_does_not_start_an_objective(self):
        self.apply([], node="virtual-anchor")
        self.assertTrue(all(not item.start_node_id for item in self.memory.items))
        self.memory.mark_subgoal_started("sg1", "physical-node")
        self.assertEqual(self.memory.items[1].start_node_id, "physical-node")
        self.assertEqual(self.memory.items[0].start_node_id, "")

    def test_initializing_existing_progress_archives_done_items_without_losing_nodes(self):
        memory = TaskProgressMemory()
        memory.initialize_agenda("Enter then stop.", [
            TaskProgressItem("Enter.", status="done", result="Entered.", start_node_id="n0", completion_node_id="n1"),
            TaskProgressItem("Stop."),
        ])
        self.assertEqual(memory.items[0].subgoal_id, "sg1")
        self.assertEqual(memory.history[0]["completion_node_id"], "n1")
        with self.assertRaisesRegex(ValueError, "immutable"):
            memory.initialize_agenda("Different task.", [])


if __name__ == "__main__":
    unittest.main()
