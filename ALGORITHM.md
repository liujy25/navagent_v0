# Canonical VLN algorithm

NavAgent v0 exposes one fixed mainline: modular planning, visual waypoint
context, Frontier Skeleton Sampling, active episodic retrieval, and knowledge
consolidation.

## Initialization

For each instruction, the agent creates ordered task-progress items before the
first navigation step. It also asks the multimodal model for concrete landmark
categories and configures YOLO-World to detect those categories. Room names,
route directions, openings, floors, and stairs are not treated as boxed object
categories.

## Place step

At every place step the agent:

1. captures a six-view RGB-D panorama;
2. updates the local and cumulative BEV maps;
3. creates or reuses the current place node and updates graph frontiers;
4. summarizes the current node and stores landmark evidence;
5. builds a text memory index over historical nodes, movement edges, and
   landmarks;
6. runs a bounded progress/retrieval/navigation loop.

The current panorama and current landmark detections are base observations.
Historical RGB, edge trajectory/movement RGB, landmark crops, and graph BEV are
loaded only through `RETRIEVE`. A retrieval request may ask for multiple refs
and fields; each follow-up round must target remaining evidence. The maximum is
eight rounds per place step.

After the model concludes retrieval, it updates task progress. If retrieval
occurred, the Knowledge Manager consolidates reusable entity knowledge back
into graph nodes, edges, or landmarks. This does not change task-progress
state.

## Navigation actions

The progress-navigation model selects exactly one of:

- `GO_TO_WAYPOINT`: ground the current active progress item to an FSS candidate;
- `BACKTRACK`: use a stored prior-node panorama as the planning reference while
  executing the resulting route from the robot's physical pose;
- `APPROACH_TO_STOP`: retain the pose or select a denser nearby FSS candidate,
  then reassess terminal completion from a fresh panorama;
- `VERTICAL_TRANSITION`: enter the bounded stair/floor transition controller.

FSS samples reachable skeleton/frontier geometry, projects candidates into the
directional RGB views, supplies RGB overlays plus a local BEV, and asks the
multimodal model to select one candidate. Candidates too close to visited place
nodes are removed for normal waypoint moves; stop approaches use denser local
sampling without node de-duplication.

## Termination

`done` is accepted only through the post-approach terminal check or the explicit
stop-confirmation module. A waypoint or navigation failure terminates with an
explicit failure reason. The implementation does not silently switch to a
different planner, detector, memory mode, or retrieval budget.
