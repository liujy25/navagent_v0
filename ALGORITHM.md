# Canonical VLN algorithm

NavAgent v0 exposes one fixed robot mainline: modular planning, visual waypoint
context, Frontier Skeleton Sampling (FSS), active episodic retrieval, and entity
knowledge consolidation. Robot RPC, measured RGB-D/pose input, per-floor BEV
mapping, and YOLO-World detection remain the robot-specific execution surface.

## Initialization

The Initial Task Progress Generator receives the original navigation instruction
once and creates ordered route-level subtasks plus a final stopping requirement
when one is stated. Every item starts as `active` with an empty result. Item
content, order, and count are immutable after initialization; later partial facts
are stored as dynamic progress conditions.

The Landmark Category Generator selects a minimal set of bounded physical object
or fixture categories that can characterize scene content or place identity and
that YOLO-World can localize with stable boxes. Explicit eligible objects may be
used. For each distinct named place type, it may infer at most one strongly
typical, distinctive, detector-groundable object when that helps recognize the
place. Place types, passage and circulation structures, routes, spatial regions,
and directions remain visual navigation evidence rather than detector-facing
landmark categories.

After initialization, task-progress memory is the authoritative instruction
state. The raw instruction is not repeated to TPU, PCNP, or the inherited
Waypoint Planner context.

## Place step

At every place step the agent:

1. captures four RGB-D views physically in front, left, back, right order;
2. presents those views to models as separate front, back, left, right images;
3. updates local and cumulative per-floor BEV maps;
4. creates or reuses the current place node and completes any pending physical
   movement edge;
5. writes a concise Current Node Summary and current landmark evidence;
6. builds a text index over historical nodes, physical movement edges, and
   landmarks;
7. runs the Task Progress Updater (TPU) with bounded active retrieval, then runs
   the Progress-Conditioned Navigation Planner (PCNP).

TPU is the only module that retrieves raw historical fields. Its response has
one or two ordered `tool_calls`: an optional non-empty
`update_progress_conditions` call first, followed by exactly one final
`retrieve` or `update_progress` call. Item updates permit only `update` with
`index`, `status`, `result`, and `reason`; conditions may be added, have their
status updated, be rewritten, or be removed. A target being visible does not by
itself prove that an instructed movement or spatial relation was completed.

The memory index follows the current observation and retrieval workspace in the
prompt. Each retrieval query asks one focused progress-verification question and
loads only the needed available fields. Retrieved node panoramas, landmark
overlays, edge start-view/path overlays, edge BEV trajectories, chronological
movement RGB, and landmark crops return to TPU. The default maximum remains
eight retrieval rounds per place step. Invalid semantic responses are corrected
with at most two retries; retry feedback is appended last and does not consume a
retrieval round.

PCNP cannot retrieve or mutate task progress. It selects exactly one high-level
action using updated progress, current observation, retrieval conclusions, and
the cumulative BEV. Raw retrieved images are not forwarded. If retrieval
occurred, the text-only Entity Knowledge Manager (EKM) runs once after the PCNP
action or terminal decision is fixed. Each EKM trace round contains its
one-based round number, verification query, actual entity-field request, and the
matching TPU conclusion. EKM can update only compact reusable knowledge on the
retrieved graph entities.

## Navigation actions and grounding

PCNP selects one of:

- `go_to_waypoint`: continue the current route stage, returning a local panorama
  `direction` and evidence-based `reason`;
- `backtrack`: switch to a stored prior node as a planning reference, without
  physically moving the robot, then rerun TPU/retrieval/PCNP;
- `approach_to_stop`: perform one final local approach or retain the current pose;
  a moving approach includes `direction`, while `stay` omits it;
- `vertical_transition`: enter the bounded up/down stair-transition controller.

Backtrack context explicitly distinguishes the trigger planning-reference node,
new planning-reference node, and physical robot node. The stored anchor panorama
is historical planning-reference evidence. Later movement still executes from
the robot's physical pose, and the resulting graph edge records the actual
physical endpoints.

For horizontal movement, the Waypoint Planner receives the fixed PCNP action
first, followed by current progress, optional active item and conditions,
retrieval conclusions, progress updates, and backtrack context. It then receives
the waypoint reference, candidate BEV, and four directional RGB candidate
overlays. It grounds the fixed action to exactly one provided FSS world-space
candidate; it cannot change the route branch. White numbered circles are
selectable candidates, blue numbered circles are graph overlays, and landmark
boxes are evidence rather than waypoint surfaces. JSON field order has no
semantic meaning.

FSS samples connected free-space skeleton/frontier geometry, projects candidates
into directional RGB views, and supplies a BEV with the same labels. Ordinary
movement removes candidates near previously visited nodes; final approaches use
denser local sampling without that de-duplication.

## Vertical transitions

The Vertical Transition Step Planner owns completion of a fixed up/down action.
At each fresh panorama it returns `complete`, `continue`, or `fail`. A turn
landing is not completion; completion requires a stable destination-floor
walking surface beyond the traversed staircase. For `continue`, it chooses one
visible reachable region and view, prioritizing destination-floor surface,
continued stair progress, then stair entry.

The dedicated Vertical Transition Visual Action Grounder returns an explicit
success/failure union. Success contains one normalized point on the intended
walking surface. Failure contains `point_2d=null`, an empty target, and a
non-empty failure reason, and returns directly to the bounded Step Planner
replanning path before projection or verification. The verifier checks only
whether a successful point is a valid next local move for the fixed target and
direction; it does not decide transition completion.

## Termination

After every moving or stationary `approach_to_stop`, the next place step runs
the same TPU in terminal-check mode. `done` is valid only when all task items are
done and no terminal constraint is missing; otherwise `continue` returns to
PCNP. This TPU terminal check is the only VLN completion authority. There is no
separate stop-confirmation path and no backtrack-specific terminal path.

Waypoint, vertical-transition, robot-motion, and step-budget failures terminate
with an explicit reason. The implementation does not silently switch planner,
detector, memory mode, or retrieval budget.
