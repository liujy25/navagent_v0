from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch


from navprobe.agent.actions import AgentAction
from navprobe.agent.actions import AgentActionType
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.agent.visual_navigation import VisualNavigationDecision
from navprobe.agent.visual_policy_decisions import LocalMovePlanDecision
from navprobe.agent.visual_policy_decisions import NavigationModeDecision
from navprobe.agent.visual_step_execution import _execute_visual_action_call
from navprobe.agent.visual_step_execution import run_visual_waypoint_decision
from navprobe.agent.waypoint import GroundedWaypointTarget
from navprobe.memory.task_progress import TaskProgressMemory, TaskProgressUpdateResult
from navprobe.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION
from navprobe.schemas import ActionCall
from navprobe.schemas import ActionResult


def _stop_navigation_mode() -> NavigationModeDecision:
    return NavigationModeDecision(
        action_mode="approach_to_stop",
        approach_movement="move",
        stop_objective="Stop at the instruction endpoint.",
        progress_analysis="The route endpoint is locally reachable.",
        progress_reasoning="The terminal landmark is visible.",
        reasoning_action="Ground the final stop location.",
    )


def _backtrack_navigation_mode() -> NavigationModeDecision:
    return NavigationModeDecision(
        action_mode="backtrack",
        stop_objective="",
        progress_analysis="The current branch is exhausted.",
        progress_reasoning="Use the earlier junction as the replan reference.",
        reasoning_action="Try the alternate branch.",
        route_status="",
        action_objective="Try the alternate branch.",
        action_reason="The current branch is exhausted.",
        backtrack_reason="The current branch is exhausted.",
        backtrack_anchor_node_id="n1",
        backtrack_objective="Try the alternate branch.",
    )


def _step() -> SimpleNamespace:
    return SimpleNamespace(
        episodic_retrieval={},
        place_step_index=3,
        current_place_node_id="n2",
        visual_decision={},
        policy_decision={},
        agent_action={},
        agent_decision={},
        results=[],
    )


class StopFssExecutionTest(unittest.TestCase):
    @patch("navprobe.agent.visual_step_execution.plan_visual_navigation_action")
    def test_vln_stop_stay_reuses_current_node_without_movement(
        self,
        plan_visual_navigation_action,
    ) -> None:
        navigation_mode = NavigationModeDecision(
            action_mode="approach_to_stop",
            approach_movement="stay",
            stop_objective="Remain at the current instruction endpoint.",
            progress_analysis="The route endpoint is the current pose.",
            progress_reasoning="No further approach is needed.",
            reasoning_action="Keep the current stopping pose.",
            route_status="",
            action_objective="Remain at the current instruction endpoint.",
            action_reason="No further approach is needed.",
        )
        plan_visual_navigation_action.return_value = VisualNavigationDecision(
            visual_context=VisualActionContext(current_node_id="n2", views=[]),
            task_progress_initialization={},
            navigation_mode=navigation_mode,
            task_progress_update_result=TaskProgressUpdateResult(),
            verify_memory_update_result={},
            visual_action=None,
            waypoint=None,
            action_call=None,
        )
        state = SimpleNamespace(
            pending_vln_terminal_check=None,
            reuse_current_place_next_step=None,
            last_executed_action=None,
            finalize_reason="",
        )
        step = _step()

        run_visual_waypoint_decision(
            context=SimpleNamespace(
                goal_spec=SimpleNamespace(goal_kind=GOAL_KIND_VLN_INSTRUCTION)
            ),
            state=state,
            step=step,
            context_evidence_text="",
        )

        self.assertEqual(step.agent_action["args"]["movement"], "stay")
        self.assertEqual(step.results, [])
        self.assertEqual(step.navigation_segment_timings, [])
        self.assertEqual(step.executed_action["type"], "vln_approach_to_stop_stay")
        self.assertEqual(state.pending_vln_terminal_check["movement"], "stay")
        self.assertEqual(
            state.reuse_current_place_next_step.place_node_id,
            "n2",
        )
        self.assertEqual(state.finalize_reason, "")

    @patch("navprobe.agent.visual_step_execution.apply_agent_feedback")
    @patch("navprobe.agent.visual_step_execution._execute_visual_action_call")
    @patch("navprobe.agent.visual_step_execution.plan_visual_navigation_action")
    def test_backtracked_stay_executes_return_to_planning_node(
        self,
        plan_visual_navigation_action,
        execute_visual_action_call,
        _apply_agent_feedback,
    ) -> None:
        navigation_mode = NavigationModeDecision(
            action_mode="approach_to_stop",
            approach_movement="stay",
            stop_objective="Remain at the backtrack anchor.",
            progress_analysis="The anchor is the stopping region.",
            progress_reasoning="Return there before the terminal check.",
            reasoning_action="Return to n1.",
        )
        target = GroundedWaypointTarget(
            goal_xy=(1.0, 0.0),
            goal_yaw=0.0,
            world_z=0.0,
            policy_name="frontier_skeleton_sample",
            source_type="planning_node",
            source_id="n1",
            obs_id="obs_n1",
            angle_deg=0,
        )
        plan_visual_navigation_action.return_value = VisualNavigationDecision(
            visual_context=VisualActionContext(current_node_id="n1", views=[]),
            task_progress_initialization={},
            navigation_mode=navigation_mode,
            task_progress_update_result=TaskProgressUpdateResult(),
            verify_memory_update_result={},
            visual_action=None,
            waypoint=None,
            action_call=target.to_action_call(),
            local_move_plan=LocalMovePlanDecision(
                selected_angle_deg=0,
                waypoint_target="return to planning node n1",
                reasoning="Return to the selected stopping region.",
            ),
            grounded_waypoint_target=target,
            physical_current_node_id="n2",
            planning_current_node_id="n1",
            backtrack_contexts=[{"anchor_node_id": "n1"}],
        )

        run_visual_waypoint_decision(
            context=SimpleNamespace(goal_spec=None),
            state=SimpleNamespace(),
            step=_step(),
            context_evidence_text="",
        )

        self.assertEqual(execute_visual_action_call.call_count, 1)
        action = execute_visual_action_call.call_args.kwargs["action"]
        self.assertTrue(action.args["stop_approach"])
        self.assertEqual(action.args["decision_type"], "approach_to_stop")

    @patch("navprobe.agent.visual_step_execution.apply_agent_feedback")
    @patch("navprobe.agent.visual_step_execution.execute_vertical_transition_action")
    @patch("navprobe.agent.visual_step_execution.plan_visual_navigation_action")
    def test_vertical_transition_forwards_backtrack_planning_context(
        self,
        plan_visual_navigation_action,
        execute_vertical_transition_action,
        _apply_agent_feedback,
    ) -> None:
        navigation_mode = NavigationModeDecision(
            action_mode="vertical_transition",
            stop_objective="",
            progress_analysis="The stair branch begins at n1.",
            progress_reasoning="Use the anchor panorama.",
            reasoning_action="Go downstairs.",
            vertical_direction="down",
        )
        backtrack_contexts = [
            {
                "trigger_planning_node_id": "n2",
                "anchor_node_id": "n1",
                "physical_robot_node_id": "n2",
                "objective": "Use the stairs.",
                "reason": "The current branch is exhausted.",
            }
        ]
        plan_visual_navigation_action.return_value = VisualNavigationDecision(
            visual_context=VisualActionContext(current_node_id="n1", views=[]),
            task_progress_initialization={},
            navigation_mode=navigation_mode,
            task_progress_update_result=TaskProgressUpdateResult(),
            verify_memory_update_result={},
            visual_action=None,
            waypoint=None,
            action_call=ActionCall(
                action="vertical_transition",
                args={"direction": "down"},
            ),
            physical_current_node_id="n2",
            planning_current_node_id="n1",
            backtrack_contexts=backtrack_contexts,
        )

        run_visual_waypoint_decision(
            context=SimpleNamespace(goal_spec=None),
            state=SimpleNamespace(system=SimpleNamespace(memory=SimpleNamespace(task_progress=TaskProgressMemory()))),
            step=_step(),
            context_evidence_text="",
        )

        action = execute_vertical_transition_action.call_args.kwargs["action"]
        self.assertEqual(action.args["planning_node_id"], "n1")
        self.assertEqual(action.args["backtrack_contexts"], backtrack_contexts)

    @patch("navprobe.agent.visual_step_execution.set_pending_node_move")
    def test_backtrack_execution_preserves_reference_in_pending_move(
        self,
        set_pending_node_move,
    ) -> None:
        executor = SimpleNamespace(
            execute=lambda _action_call: ActionResult(
                ok=True,
                data={
                    "rgb_history_obs_ids": ["move_obs"],
                    "path_xy": [[0.0, 0.0], [1.0, 0.0]],
                },
            )
        )
        state = SimpleNamespace(
            action_executor=executor,
            last_executed_action=None,
            previous_place_node_id=None,
        )
        step = SimpleNamespace(
            place_step_index=3,
            current_place_node_id="n3",
            results=[],
            navigation_segment_timings=[],
            executed_action={},
        )
        action = AgentAction(
            action_type=AgentActionType.VISUAL_WAYPOINT,
            args={
                "selected_angle_deg": 0,
                "stop_approach": False,
                "decision_type": "go_to_waypoint",
            },
            source="visual_waypoint_policy",
            reason="sampled candidate label 2",
        )

        _execute_visual_action_call(
            state=state,
            step=step,
            action=action,
            action_call=ActionCall(action="goto", args={"x": 1.0, "y": 0.0}),
            decision_payload={
                "navigation_mode": _backtrack_navigation_mode().to_dict()
            },
        )

        self.assertEqual(
            set_pending_node_move.call_args.kwargs["move_mode"],
            "backtrack",
        )
        self.assertEqual(
            set_pending_node_move.call_args.kwargs[
                "backtrack_reference_node_id"
            ],
            "n1",
        )

    @patch("navprobe.agent.visual_step_execution.apply_agent_feedback")
    @patch("navprobe.agent.visual_step_execution._execute_visual_action_call")
    @patch("navprobe.agent.visual_step_execution.plan_visual_navigation_action")
    def test_grounded_stop_candidate_is_executed_as_stop_approach(
        self,
        plan_visual_navigation_action,
        execute_visual_action_call,
        _apply_agent_feedback,
    ) -> None:
        target = GroundedWaypointTarget(
            goal_xy=(1.0, 2.0),
            goal_yaw=0.0,
            world_z=0.0,
            policy_name="frontier_skeleton_sample",
            source_type="stop_sample",
            obs_id="obs_0",
            angle_deg=0,
        )
        decision = VisualNavigationDecision(
            visual_context=VisualActionContext(current_node_id="n2", views=[]),
            task_progress_initialization={},
            navigation_mode=_stop_navigation_mode(),
            task_progress_update_result=TaskProgressUpdateResult(),
            verify_memory_update_result={},
            visual_action=None,
            waypoint=None,
            action_call=target.to_action_call(),
            local_move_plan=LocalMovePlanDecision(
                selected_angle_deg=0,
                waypoint_target="sampled candidate label 4",
                reasoning="Candidate 4 is the local endpoint.",
            ),
            grounded_waypoint_target=target,
        )
        plan_visual_navigation_action.return_value = decision

        run_visual_waypoint_decision(
            context=SimpleNamespace(goal_spec=None),
            state=SimpleNamespace(),
            step=_step(),
            context_evidence_text="",
        )

        action = execute_visual_action_call.call_args.kwargs["action"]
        self.assertTrue(action.args["stop_approach"])
        self.assertEqual(action.args["decision_type"], "approach_to_stop")

    @patch("navprobe.agent.visual_step_execution.set_pending_node_move")
    def test_backtracked_vln_stop_approach_defers_to_terminal_progress_check(
        self,
        set_pending_node_move,
    ) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.calls: list[ActionCall] = []

            def execute(self, action_call: ActionCall) -> ActionResult:
                self.calls.append(action_call)
                if action_call.action == "done":
                    return ActionResult(
                        ok=True,
                        data={"stop_result": {"episode_success": True}},
                    )
                return ActionResult(
                    ok=True,
                    data={
                        "rgb_history_obs_ids": ["move_obs"],
                        "path_xy": [[0.0, 0.0], [1.0, 0.0]],
                    },
                )

        executor = _Executor()
        state = SimpleNamespace(
            action_executor=executor,
            last_executed_action=None,
            stop_result=None,
            previous_place_node_id=None,
            pending_vln_terminal_check=None,
            pending_stop_confirmation={"legacy": True},
            finalize_reason="",
        )
        step = SimpleNamespace(
            place_step_index=5,
            current_place_node_id="n5",
            results=[],
            navigation_segment_timings=[],
            executed_action={},
        )
        action = AgentAction(
            action_type=AgentActionType.VISUAL_WAYPOINT,
            args={
                "selected_angle_deg": 0,
                "selected_obs_id": "obs_0",
                "target": "candidate 6",
                "stop_approach": True,
                "decision_type": "approach_to_stop",
            },
            source="visual_waypoint_policy",
            reason="Reach the selected endpoint.",
        )

        _execute_visual_action_call(
            state=state,
            step=step,
            action=action,
            action_call=ActionCall(action="goto", args={"x": 1.0, "y": 0.0}),
            decision_payload={
                "navigation_mode": _stop_navigation_mode().to_dict(),
                "backtrack_contexts": [
                    {
                        "anchor_node_id": "n1",
                        "physical_robot_node_id": "n5",
                    }
                ],
            },
        )

        self.assertEqual([call.action for call in executor.calls], ["goto"])
        self.assertEqual(len(step.results), 1)
        self.assertEqual(len(step.navigation_segment_timings), 1)
        self.assertEqual(step.executed_action["type"], "move_to_visual_waypoint")
        self.assertIsNone(state.stop_result)
        self.assertEqual(state.finalize_reason, "")
        self.assertEqual(
            state.pending_vln_terminal_check["stop_objective"],
            "Stop at the instruction endpoint.",
        )
        self.assertIsNone(state.pending_stop_confirmation)
        self.assertEqual(
            set_pending_node_move.call_args.kwargs["move_mode"],
            "backtrack",
        )



    @patch("navprobe.agent.visual_step_execution.apply_agent_feedback")
    @patch("navprobe.agent.visual_step_execution.plan_visual_navigation_action")
    def test_progress_terminal_done_executes_without_navigation_action(
        self,
        plan_visual_navigation_action,
        _apply_agent_feedback,
    ) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.calls: list[ActionCall] = []

            def execute(self, action_call: ActionCall) -> ActionResult:
                self.calls.append(action_call)
                return ActionResult(
                    ok=True,
                    data={"stop_result": {"episode_success": True}},
                )

        plan_visual_navigation_action.return_value = VisualNavigationDecision(
            visual_context=VisualActionContext(current_node_id="n6", views=[]),
            task_progress_initialization={},
            navigation_mode=None,
            task_progress_update_result=TaskProgressUpdateResult(),
            verify_memory_update_result={},
            visual_action=None,
            waypoint=None,
            action_call=ActionCall(action="done", args={}),
            terminal_check={
                "decision": "done",
                "reason": "All route and endpoint constraints are complete.",
                "missing_constraints": [],
            },
        )
        executor = _Executor()
        state = SimpleNamespace(
            action_executor=executor,
            stop_result=None,
            last_executed_action=None,
            pending_vln_terminal_check=None,
            finalize_reason="",
        )
        step = _step()
        step.navigation_segment_timings = []
        step.executed_action = {}

        run_visual_waypoint_decision(
            context=SimpleNamespace(
                goal_spec=SimpleNamespace(goal_kind=GOAL_KIND_VLN_INSTRUCTION)
            ),
            state=state,
            step=step,
            context_evidence_text="",
        )

        self.assertEqual([call.action for call in executor.calls], ["done"])
        self.assertEqual(step.policy_decision["source"], "vln_progress_terminal_check")
        self.assertEqual(step.executed_action["type"], "vln_progress_terminal_done")
        self.assertEqual(state.finalize_reason, "done")

if __name__ == "__main__":
    unittest.main()
