from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from navprobe.agent.vertical_fss import (
    _connected_support,
    _fuse_vertical_support_with_map,
    _detected_stair_mask,
    _sample_vertical_candidates,
    _temporary_stair_support,
    prepare_vertical_fss_candidates,
)
from navprobe.agent.vertical_transition import (
    _complete_vertical_transition,
    _record_vertical_transition_node_move,
    execute_vertical_transition_action,
)
from navprobe.agent.vertical_transition_policy import (
    VerticalTransitionStepDecision,
    _normalize_vertical_transition_step,
    decide_vertical_transition_step,
    select_vertical_fss_waypoint,
)
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.env.interface import Pose, RawObservation
from navprobe.memory.task_progress import TaskProgressMemory
from navprobe.perception.detectors.interface import Detection2D
from navprobe.runtime.cache import RuntimeCache
from navprobe.mapping.exploration.bev_map import GlobalBEVMap
from navprobe.schemas import ActionResult


class NavProbeVerticalFssTest(unittest.TestCase):
    @patch("navprobe.agent.vertical_fss._horizontal_support_mask")
    @patch("navprobe.agent.vertical_fss._backproject")
    @patch("navprobe.agent.vertical_fss._projected_views_for_waypoint")
    def test_detected_stairs_extend_only_temporary_traversability(self, projections, backproject, horizontal):
        cache = RuntimeCache()
        transform = np.eye(4)
        transform[2, 3] = -1.2
        observation = RawObservation(
            pose=Pose(x=0, y=0, z=0, yaw=0),
            rgb=np.zeros((10, 20, 3), dtype=np.uint8),
            depth=np.ones((10, 20)), intrinsics=np.eye(3), T_cam_odom=transform,
        )
        obs_id = cache.store_observation(observation).id
        view = VisualViewContext(angle_deg=0, obs_id=obs_id, rgb_id="r", depth_id="d")
        ys, xs = np.indices((10, 20))
        world = np.stack([0.025 + xs * 0.05, 0.025 + (ys - 5) * 0.05, (0.025 + xs * 0.05) * 0.3], axis=-1)
        backproject.return_value = (world, world)
        horizontal.return_value = np.ones((10, 20), dtype=bool)
        projections.return_value = [(1.0, view, (10.0, 5.0), 1.0)]
        detector = Mock()
        detector.detect.return_value = [Detection2D("stairs", [7, 0, 20, 10], 0.9)]
        map_obj = GlobalBEVMap(size=201, agent_radius=0.1, max_depth=10.0)
        map_obj.explored_area[:] = True
        map_obj.known_map[:] = True
        map_obj.obstacle_map[:, 109:] = True
        before = {key: getattr(map_obj, key).copy() for key in ("explored_area", "known_map", "obstacle_map")}
        kwargs = dict(cache=cache, visual_context=VisualActionContext(current_node_id="n0", views=[view]), exploration=SimpleNamespace(map=map_obj), direction="up", detector=detector, map_floor_height=0.0)
        stairs = prepare_vertical_fss_candidates(**kwargs)
        detector.detect.return_value = []
        no_stairs = prepare_vertical_fss_candidates(**kwargs)
        self.assertGreater(stairs.temporary_stair_cell_count, 0)
        self.assertGreater(stairs.height_connected_cell_count, no_stairs.height_connected_cell_count)
        self.assertTrue(any(candidate.height_delta_m > 0.12 for candidate in stairs.candidates))
        self.assertTrue(all(candidate.height_delta_m <= 0.12 for candidate in no_stairs.candidates))
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(map_obj, key), value)

    def test_detection_uses_rgb_detector_region(self):
        observation = SimpleNamespace(rgb=np.zeros((20, 30, 3), dtype=np.uint8))
        detector = Mock()
        detector.detect.return_value = [Detection2D("stairs", [-5, 7, 13, 40], 0.9)]
        mask, records = _detected_stair_mask(
            detector=detector, observation=observation, obs_id="o7"
        )
        self.assertEqual(int(mask.sum()), 13 * 13)
        self.assertFalse(mask[:7].any())
        self.assertEqual(records[0]["bbox"], [0, 7, 13, 20])
        self.assertEqual(records[0]["obs_id"], "o7")
        self.assertEqual(detector.detect.call_args.kwargs["class_name"], "stairs")

    def test_support_path_cannot_cut_a_blocked_diagonal_corner(self):
        support = np.zeros((5, 5), dtype=bool)
        support[2, 2] = support[3, 3] = True
        connected, _, _ = _connected_support(support, np.zeros_like(support, dtype=float), (2, 2), 0.0)
        self.assertTrue(connected[2, 2])
        self.assertFalse(connected[3, 3])

    def test_map_occupancy_is_retained_outside_detected_stairs(self):
        map_obj = GlobalBEVMap(size=40, agent_radius=0.1)
        map_obj.explored_area[:] = map_obj.known_map[:] = True
        map_obj.obstacle_map[:] = True
        support = np.ones((6, 12), dtype=bool)
        stairs = np.zeros_like(support)
        stairs[:, 4:7] = True
        heights = np.zeros(support.shape)
        heights[:, 4:7] = 0.18
        allowed, _, _ = _fuse_vertical_support_with_map(
            map_obj=map_obj, origin_xy=np.zeros(2), observed_support=support,
            observed_heights=heights, stair_cells=stairs, floor_height=0.0,
        )
        self.assertTrue(allowed[:, 4:7].all())
        self.assertFalse(allowed[:, :4].any())
        self.assertFalse(allowed[:, 7:].any())
        self.assertTrue(map_obj.obstacle_map.all())

    def test_paths_respect_robot_radius_through_narrow_gap(self):
        support = np.zeros((41, 61), dtype=bool)
        support[10:31, 5:56] = True
        support[10:31, 27:34] = False
        support[19:22, 27:34] = True
        heights = np.zeros(support.shape)
        point_connected, _, _ = _connected_support(support, heights, (15, 20), 0.0)
        radius_connected, _, _ = _connected_support(
            support, heights, (15, 20), 0.0,
            require_robot_seed=True, clearance_radius_m=0.15,
        )
        self.assertTrue(point_connected[20, 45])
        self.assertFalse(radius_connected[20, 45])
        self.assertTrue(radius_connected[20, 15])
        self.assertTrue(support[20, 30])

    def test_stair_override_connects_adjoining_landing_without_freeing_other_levels(self):
        support = np.ones((3, 10), dtype=bool)
        heights = np.tile([0.0, 0.0, 0.18, 0.36, 0.54, 0.72, 0.72, 0.72, 1.5, 1.5], (3, 1))
        stairs = np.zeros_like(support)
        stairs[:, 2:5] = True
        traversable = _temporary_stair_support(support=support, heights=heights, stair_cells=stairs, base_z=0.0)
        self.assertTrue(traversable[:, :8].all())
        self.assertFalse(traversable[:, 8:].any())

    def test_frontier_and_skeleton_anchors_have_connected_paths(self):
        support = np.zeros((101, 101), dtype=bool)
        support[42:59, 45:92] = True
        heights = np.broadcast_to(np.maximum(0, np.arange(101) - 55) * 0.02, support.shape).copy()
        connected, parents, seed = _connected_support(support, heights, (50, 50), 0.0)
        observed = np.ones_like(support)
        observed[:, 92:] = False
        candidates = _sample_vertical_candidates(
            connected=connected, heights=heights, parents=parents, seed=seed,
            robot_pixel=(50, 50), base_z=0.0, direction="up", observed=observed,
        )
        self.assertEqual({c["anchor_source"] for c in candidates}, {"frontier", "skeleton"})
        for candidate in candidates:
            self.assertEqual(candidate["path_pixels"][0], seed)
            self.assertTrue(all(connected[y, x] for x, y in candidate["path_pixels"]))

    def test_vertical_frontier_only_removes_skeletons_without_changing_frontier_paths(self):
        support = np.zeros((101, 101), dtype=bool)
        support[42:59, 45:92] = True
        heights = np.broadcast_to(np.maximum(0, np.arange(101) - 55) * 0.02, support.shape).copy()
        connected, parents, seed = _connected_support(support, heights, (50, 50), 0.0)
        observed = np.ones_like(support)
        observed[:, 92:] = False
        kwargs = dict(connected=connected, heights=heights, parents=parents, seed=seed,
                      robot_pixel=(50, 50), base_z=0.0, direction="up", observed=observed)
        fss = _sample_vertical_candidates(**kwargs)
        frontier = _sample_vertical_candidates(**kwargs, frontier_only=True)
        self.assertTrue(frontier)
        self.assertEqual({item["anchor_source"] for item in frontier}, {"frontier"})
        self.assertEqual(frontier, [item for item in fss if item["anchor_source"] == "frontier"])

    def test_vertical_grounder_forwards_frontier_only_policy(self):
        from navprobe.agent.vertical_transition import _plan_navprobe_vertical_waypoint
        state = SimpleNamespace(cache=Mock(), detector=Mock(), global_exploration_for_floor=Mock(),
                                system=SimpleNamespace(current_floor_id="f0", current_floor_height=0.0))
        with patch("navprobe.agent.vertical_transition.prepare_vertical_fss_candidates", side_effect=ValueError("captured")) as prepare:
            for policy in ("frontier", "frontier_skeleton_sample"):
                state.waypoint_policy_name = policy
                with self.assertRaisesRegex(ValueError, "captured"):
                    _plan_navprobe_vertical_waypoint(state=state, direction="up", visual_context=Mock(),
                                                    step_decision=Mock(), current_height_m=0.0,
                                                    objective="Reach the landing.", task_context="")
                self.assertEqual(prepare.call_args.kwargs["frontier_only"], policy == "frontier")

    @patch("navprobe.agent.vertical_transition_policy.image_content_for_vertical_transition_panorama_views", return_value=[])
    def test_partial_stair_endpoint_is_preserved_in_local_controller(self, _images):
        client = Mock()
        client.decide_vertical_transition_step.return_value = {
            "transition_status": "complete", "waypoint_target": "",
            "selected_direction": None, "destination_floor_reached": False,
            "reason": "The robot is halfway up the staircase as requested.",
        }
        decision = decide_vertical_transition_step(
            client=client, cache=Mock(),
            instruction="Climb halfway up the staircase and stop.",
            visual_context=VisualActionContext(current_node_id="n0", views=[]),
        )
        content = client.decide_vertical_transition_step.call_args.args[1]
        text = "\n".join(block.get("text", "") for block in content)
        self.assertIn("Climb halfway up the staircase and stop.", text)
        self.assertIn("A partial stair target is complete at its stated position", text)
        self.assertEqual(decision.transition_status, "complete")
        self.assertFalse(decision.destination_floor_reached)

    def test_dynamic_status_requires_explicit_floor_evidence(self):
        with self.assertRaisesRegex(ValueError, "destination_floor_reached"):
            _normalize_vertical_transition_step(
                {"transition_status": "complete", "waypoint_target": "", "selected_direction": None, "reason": "At target"},
            )

    @patch("navprobe.agent.vertical_transition._resolve_vertical_transition_floor")
    @patch("navprobe.agent.vertical_transition._create_place_node", return_value="n1")
    def test_partial_stair_completion_does_not_invent_a_floor(self, _create, resolve):
        cache = RuntimeCache()
        obs_id = cache.store_observation(RawObservation(pose=Pose(x=1.0, y=0.0, z=0.15, yaw=0))).id
        floor = SimpleNamespace(id="f0", height=0.0)
        graph = Mock(floors={"f0": floor})
        graph.add_edge.return_value = SimpleNamespace(id="e0", relation="stairs_up")
        result = _complete_vertical_transition(
            graph=graph, cache=cache, before_node_id="n0", before_floor_id="f0",
            before_floor_height=0.0, direction="up", floor_match_threshold_m=0.8,
            final_panorama=SimpleNamespace(anchor_obs_id=obs_id, obs_ids=[obs_id]),
            rgb_history_obs_ids=[obs_id], path_xy=[(0.0, 0.0), (1.0, 0.0)],
            destination_floor_reached=False,
        )
        resolve.assert_not_called()
        self.assertEqual(result["after_floor_id"], "f0")
        self.assertFalse(result["created_floor"])

    def test_completion_evidence_does_not_block_refined_active_objective(self):
        memory = TaskProgressMemory()
        memory.initialize_agenda("Go upstairs.", [{"content": "Go upstairs."}])
        memory.completed_stair_items["sg0:1"] = {"route_completed": True}
        state = SimpleNamespace(
            system=SimpleNamespace(memory=SimpleNamespace(task_progress=memory)),
            completed_stair_items=memory.completed_stair_items,
        )
        action = SimpleNamespace(args={"direction": "up", "waypoint_target": "Go upstairs.", "subgoal_id": "sg0", "subgoal_attempt": 1, "floor_match_threshold_m": 0})
        with self.assertRaisesRegex(ValueError, "vertical_floor_match_threshold_m"):
            execute_vertical_transition_action(context=Mock(), state=state, step=Mock(), action=action)
        memory.items[0].attempt = 2
        with self.assertRaisesRegex(ValueError, "active subgoal attempt"):
            execute_vertical_transition_action(context=Mock(), state=state, step=Mock(), action=action)

    def test_unbound_vertical_action_is_valid_only_for_empty_agenda(self):
        memory = TaskProgressMemory()
        memory.initialize_agenda("Find a chair.", [])
        state = SimpleNamespace(system=SimpleNamespace(memory=SimpleNamespace(task_progress=memory)))
        action = SimpleNamespace(args={"direction": "up", "waypoint_target": "Inspect the upstairs landing.", "subgoal_id": None, "floor_match_threshold_m": 0})
        with self.assertRaisesRegex(ValueError, "vertical_floor_match_threshold_m"):
            execute_vertical_transition_action(context=Mock(), state=state, step=Mock(), action=action)
        memory.apply_agenda_updates([{"op": "add", "content": "Inspect upstairs", "position": 0}], [])
        with self.assertRaisesRegex(ValueError, "nonempty agenda"):
            execute_vertical_transition_action(context=Mock(), state=state, step=Mock(), action=action)

    def test_canonical_skill_returns_after_one_move_with_subgoal_still_active(self):
        from contextlib import ExitStack
        memory = TaskProgressMemory()
        memory.initialize_agenda("Go upstairs and stop halfway.", [{"content": "Go upstairs and stop halfway."}])
        state = SimpleNamespace(
            system=SimpleNamespace(current_floor_id="f0", current_floor_height=0.0, memory=SimpleNamespace(task_progress=memory)),
            current_place_node_id="n0", completed_stair_items=memory.completed_stair_items,
            ensure_floor_runtime_state=Mock(), global_exploration_for_floor=Mock(return_value=Mock()),
            action_executor=SimpleNamespace(exploration=None), cache=Mock(), llm_client=Mock(),
        )
        step = SimpleNamespace(current_place_node_id="n0", policy_decision={}, executed_action={})
        context = SimpleNamespace(args=SimpleNamespace(), global_bev_kwargs={}, env=Mock())
        context.env.current_episode_info.return_value = {}
        action = SimpleNamespace(args={"direction": "up", "waypoint_target": "Stop halfway up.", "subgoal_id": "sg0", "subgoal_attempt": 1, "task_context": "Current executive assessment"})
        decision = VerticalTransitionStepDecision("continue", "Next tread before halfway", 0, "More ascent is required", False)
        plan = Mock()
        plan.waypoint.path_xy = [(0.0, 0.0), (0.5, 0.0)]
        plan.waypoint.goal_yaw = 0.0
        plan.waypoint.raw_world_z = 0.18
        plan.to_dict.return_value = {}
        result = ActionResult(ok=True, data={"path_xy": [[0, 0], [0.5, 0]], "rgb_history_obs_ids": ["o1"]}, message="Moved")
        with ExitStack() as stack:
            mocked = {}
            for name in ("_vertical_panorama_from_current_step", "_capture_vertical_panorama", "_visual_context_from_panorama", "_panorama_anchor_height", "_plan_navprobe_vertical_waypoint", "_execute_vt_move", "_vertical_transition_va_history_entry", "_finish_vertical_transition_action", "decide_vertical_transition_step"):
                mocked[name] = stack.enter_context(patch("navprobe.agent.vertical_transition." + name))
            mocked["_panorama_anchor_height"].return_value = 0.18
            mocked["_plan_navprobe_vertical_waypoint"].return_value = plan
            mocked["_execute_vt_move"].return_value = result
            mocked["_vertical_transition_va_history_entry"].return_value = {}
            mocked["decide_vertical_transition_step"].return_value = decision
            execute_vertical_transition_action(context=context, state=state, step=step, action=action)
            mocked["_execute_vt_move"].assert_called_once()
            self.assertEqual(mocked["decide_vertical_transition_step"].call_count, 2)
            self.assertFalse(mocked["_finish_vertical_transition_action"].call_args.kwargs["destination_floor_reached"])
        self.assertFalse(step.executed_action["objective_completed"])
        self.assertEqual(memory.items[0].status, "active")
        self.assertEqual(memory.items[0].start_node_id, "n0")
        self.assertEqual(memory.completed_stair_items, {})

    def test_vertical_move_log_retains_selected_fss_label(self):
        from navprobe.agent.vertical_transition import _vertical_transition_va_history_entry
        waypoint = SimpleNamespace(goal_xy=(1.0, 0.0), goal_yaw=0.0, path_xy=[(0.0, 0.0), (1.0, 0.0)], depth_m=1.0, raw_world_z=0.2)
        plan = SimpleNamespace(
            waypoint=waypoint, selected_view=SimpleNamespace(angle_deg=0, obs_id="obs1"),
            step_decision=Mock(), visual_action=Mock(), waypoint_verification=Mock(),
            attempts=[{"algorithm": "detected_stair_fss", "candidate_label": 7, "candidate_set": {}}],
        )
        entry = _vertical_transition_va_history_entry(
            transition_index=0, waypoint_plan=plan,
            move_result=ActionResult(ok=True, data={"path_xy": [[0, 0], [1, 0]]}, message="Moved"),
        )
        self.assertEqual(entry["waypoint_attempts"], [{"algorithm": "detected_stair_fss", "candidate_label": 7}])
        self.assertEqual(entry["selected_view"]["obs_id"], "obs1")

    def test_vertical_movement_history_keeps_subgoal_attempt(self):
        state = SimpleNamespace(node_move_history=[])
        _record_vertical_transition_node_move(
            state=state, step_id=7, before_node_id="n0", after_node_id="n1",
            direction="up", rgb_history_obs_ids=["o1"], planning_node_id="n0",
            backtrack_contexts=[], subgoal_id="sg3", subgoal_attempt=2,
        )
        self.assertEqual(state.node_move_history[0]["subgoal_id"], "sg3")
        self.assertEqual(state.node_move_history[0]["subgoal_attempt"], 2)

    @patch("navprobe.agent.vertical_transition_policy.image_content_for_array", return_value={"type": "image_url"})
    def test_fss_selector_keeps_shared_label_and_executive_context(self, _image):
        client = Mock()
        client._create_visual_json_completion.return_value = {"candidate_label": 3, "reason": "The third tread matches the requested stop."}
        view = VisualViewContext(angle_deg=0, obs_id="o0", rgb_id="r0", depth_id="d0")
        prepared = SimpleNamespace(
            candidates=[SimpleNamespace(label=3)],
            view_candidate_sets=[SimpleNamespace(view=view, candidates=[SimpleNamespace(label=3)], rgb_overlay=np.zeros((2, 2, 3)))],
            bev_overlay=np.zeros((2, 2, 3)),
        )
        label, _ = select_vertical_fss_waypoint(
            client=client, prepared=prepared, objective="Stop at the third tread.",
            local_target="The third tread", task_context="Retrieved conclusion: left flight is the required route.",
        )
        content = client._create_visual_json_completion.call_args.kwargs["user_prompt"]
        text = "\n".join(block.get("text", "") for block in content)
        self.assertEqual(label, 3)
        self.assertIn("left flight is the required route", text)
        self.assertIn("labels: [3]", text)
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 2)


if __name__ == "__main__":
    unittest.main()
