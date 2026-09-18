# NavProbe robot VLN algorithm

This standalone implementation synchronizes the applicable NavProbe VLN algorithm
and prompt contracts through `navagent-agent-mvp` commit
`bfa2805b219bfe7e0e0f721ed9bb0a47f9049f61`.
The Python package is `navprobe`; robot script filenames remain unchanged.
The fixed configuration uses FSS, active entity-field retrieval, and entity
knowledge consolidation. Simulation runners and experiment controls are excluded.

## Task state and initialization

The Initial Task Executive decomposes the original instruction into an initial
`agenda`. Every objective starts active, with an empty result. The original
instruction remains authoritative throughout navigation, including route order,
spatial relations, and stopping partway up or down stairs. Initialization preserves
negation, ordinal references, before/after dependencies, and ambiguity for online
interpretation. Generated helper objectives do not add requirements to the original
task; historical node IDs alone do not prescribe exact stopping poses.

Task state consists of a mutable agenda, evidence predicates, and execution
history. Objectives have stable `subgoal_id` values and an `attempt` number.
The Executive can add, rewrite, reorder, complete, abandon, or reopen objectives.
Completion or abandonment archives the result, predicates, and physical node
span. Reopening preserves prior history and starts another attempt under the
same ID. Predicates are supporting evidence rather than a completion checklist.

Responses use `task_state_assessment` and ordered `tool_calls`:

1. `update_predicates` with `predicate_updates` and stable `predicate_id` references.
2. `update_task_state` with `agenda_updates` and, when requested, `terminal_check`.
3. `retrieve` with a query and entity-field requests.

A whole update batch is validated on a projected copy before committing. Invalid
references or operations leave agenda, history, predicates, and ID allocation
unchanged. Old fixed-index `update_progress` responses are rejected.

The landmark generator retains the robot release's minimal set of bounded,
detector-groundable object or fixture categories. YOLO-World remains the detector.
Place identity, passage relations, and route geometry are interpreted from visual
evidence; seeing an object through a doorway does not establish room membership.

## Place step and retrieval

Each place step acquires or reuses its physical observations, updates per-floor
RGB-D maps and graph entities, and obtains the current node summary. The Executive
receives the original instruction, agenda/history, current visual context, and
compact graph-organized memory index.

Retrieval can request node RGB/landmarks, edge RGB/trajectory/movement RGB, and
landmark RGB. Each batch returns to the same Executive for interpretation. Only
the latest unconcluded raw evidence remains in model context. Query-bound textual
conclusions, entity references, and cumulative per-floor BEV survive its release.
The last batch must still be interpreted when the budget is exhausted, and the
compact text index remains available.

The default budget is six retrieval rounds per place step. Virtual Backtrack
reference changes share that budget. Eligibility is stated separately from remaining
budget. Retrieve for a decision-relevant gap that history could resolve, batching
complementary records. Re-read only for a specific gap, lost detail, or contradiction.
The budget is a ceiling; stop retrieving when action or fresh observation is needed,
retaining uncertainty and interpreting any pending final batch.

A response without retrieval hands committed
task state, Executive assessment, conclusions, and spatial context to the skill
selector. Retrieved raw images and update transaction logs are not forwarded as
the selector's retrieval context. Text-only EKM consolidates retrieved conclusions
once for the final decision of that retrieval loop; its entity writes inform later
steps. In-memory observations needed for retrieval remain available even though
raw model requests and per-step images are not written to disk.

## Skills, grounding, and physical execution

The semantic selector reads the original instruction, current task state,
Executive assessment, retrieval conclusions, and current observations. It selects
a skill and active objective ID. Its reason must state the immediate route or
target, identifying evidence, and constraints needed for grounding.

The waypoint grounder receives this self-contained action, physical/planning
reference context, and labeled RGB/BEV FSS candidates. It does not receive the
whole instruction, agenda, Executive assessment, or retrieval transcript again.
It selects a candidate label and supporting view, or returns an explicit no-match
explanation for bounded semantic replanning.

- **GoToWaypoint:** ground a local movement using frontier and free-space skeleton
  proposals, RGB-D visibility, reachability, and robot clearance.
- **VerticalMove:** detect the observed stair region, temporarily make its support
  and adjoining landings traversable in a map crop, and select a connected FSS
  candidate respecting robot radius. One invocation executes a local movement,
  then returns to the Executive. Partial-stair endpoints are preserved. A local
  stair completion does not invent a new floor without supporting height evidence.
- **Backtrack:** switch to a stored planning reference and rerun the Executive and
  selector under the same budget. The switch itself does not move the robot or
  complete an objective. Grounded execution begins at the physical pose; graph
  edges preserve both physical movement and the planning reference. Returning to
  the physical view or visiting the anchor is not required merely because the
  reference changed. An anchor-grounded waypoint does not guarantee passage
  through the anchor; any necessary physical revisit needs task/path evidence.
- **Stop:** `approach_to_stop(move)` grounds a local approach. With a physical
  reference, `stay` retains the physical pose. With a historical reference, `stay`
  navigates to the planning node and records terminal `movement=move`; use it
  only when intentionally returning to a supported final stopping position.
  The next place step
  asks the Executive to verify the original task and terminal constraints.
  `continue` may formulate further objectives even with an empty agenda. `done`
  requires resolving remaining agenda entries and the original stopping conditions.

Physical execution checks `subgoal_id` and `subgoal_attempt`, starts the selected
objective at the actual node, and preserves this binding in movement history.
A stale attempt cannot execute after its objective has been reopened.

## Robot boundary

The existing `RobotRPCEnv`, ROS 2/Nav2 bridge, and compact episode runner remain.
Every model call uses the configured `--model`; LA/VA environment overrides are
not introduced. The entry remains `scripts/run_vln_robot.py --instruction ...`.

Robot panorama capture is front, left, back, right with one 90-degree left turn
between observations and a final turn returning to the original heading. Model
images are ordered front, back, left, right. Both the service constant and panorama
configuration remain 90 degrees. Depth is metric and yaw is in degrees.

Robot BEV defaults retain a 0.4 m radius and 0.3–1.4 m obstacle-height range.
The bridge must supply calibrated intrinsics, `T_cam_odom`, `T_odom_base`, and
movement observations. It executes Nav2 pose goals; actual staircase traversal
requires a robot controller capable of executing those motions. Algorithm-level
vertical grounding does not add that hardware capability.

The compact result retains termination reason, declared completion, action and
place-step counts, final pose, token totals, and action trace. Exceptions during
the decision loop stop the robot and finalize the run with reason `exception`.
