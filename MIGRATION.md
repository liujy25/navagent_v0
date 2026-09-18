# Current synchronization: NavProbe naming and VA cleanup

Date: 2026-09-18. Upstream: `7e2699971dc854d8ed875a8c2cc8fe01f9518777`.
Robot baseline: `77f00c1`; package release: `0.3.0`.

## Robot experiment changes

- The Python package is now `navprobe`; imports, agent types (`NavProbeAgentState`,
  `NavProbeAgentContext`, `NavProbeStepState`), prompts, asset references, and test
  patches use the canonical names. No old-package alias is provided.
- Model configuration uses `NAVPROBE_MODEL` or `--model`. The old `NAVCLAW_MODEL`
  variable is no longer read. `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and their CLI
  overrides remain unchanged. All modules still use one multimodal client.
- Removed the ignored `client_kind` parameter and every caller argument, including
  FSS, stair FSS, and the robot landmark generator. Removed the unused
  `LLMClient.decide_visual_action` method, matching upstream VA API removal.
  The robot package already excluded VA clients, bbox/pixel methods and schemas,
  so those require no additional migration. Active FSS visual inputs remain.
- ROS node/thread names use `navprobe`; update custom ROS configurations that
  address the former names. HTTP endpoints, wire format, Nav2 actions, metric
  depth, and 90-degree panorama turns are unchanged.
- Renamed assets retain their original bytes. Explicit package-data configuration
  now includes the four PNG icons in wheels; installation checks found they were
  previously omitted by the package build rules.

The repository/distribution name `navagent-v0`, console command `navagent-vln`,
and scripts `run_vln_robot.py` / `serve_robot_rpc.py` remain the robot delivery
interface. The experiment stays independent of the sibling NavProbe checkout.
UniLaViRA single-floor dataset selection and simulation batch metadata changes
are outside the robot runtime and were not imported.

## Upgrade

Run in each environment using this repository (compute machine and robot bridge):

```bash
pip install -e .
export NAVPROBE_MODEL="your-multimodal-model"
python scripts/run_vln_robot.py --help
```

Update custom Python imports to `navprobe` and environment/ROS launch references
as described above. Existing `--model`, `--api-key`, `--base-url`, `--robot-url`,
and `--instruction` arguments keep their meaning.

## Validation

- All 109 tests pass, including the complete simulated robot RPC/FSS/stop cases.
- Model-routing coverage exercises nine active completion paths against one
  configured client and checks aggregate usage.
- Seven complete representative model requests match upstream `7e26999`
  byte-for-byte (including the updated NavProbe brand and removed route argument).
- Source and tests resolve the renamed package; syntax and undefined-name/unused
  import checks pass. Binary assets match the previous robot release.
- The wheel builds, includes all four icons, and its installed `navagent-vln`
  entry point loads the renamed robot runner. No physical robot, ROS process,
  detector weights, or online model calls were used in these checks.

The previous synchronization record below is historical; its old package names
and version statements describe release 0.2.0, not the current interface.

---

# NavProbe synchronization — 2026-09-18

Upstream: `navagent-agent-mvp`, commit
`f26d5a108876cdbe47023879cd5cf31ac92cd197`.
Robot baseline: `b10c398`; package release: `0.2.0`.

This is a source migration into an independent package, not a runtime dependency
on the sibling repository. Later NavProbe commits do not affect this release.
The package name `navclaw` and existing robot launch commands are retained.

## Changes

- Dynamic agenda with stable objective IDs, evidence predicates, execution
  attempts/history, and atomic updates replaces the fixed instruction list.
- The native Executive protocol uses `task_state_assessment`, `update_task_state`
  with `agenda_updates`, and `update_predicates` with `predicate_id` references.
  Executive and semantic selection keep the original instruction in context.
- Entity-field retrieval retains only the latest unconcluded raw evidence batch,
  requires final-batch interpretation, and hands conclusions to skill selection.
  EKM, arrival trajectory overlays, and retrieval trajectory rendering match the
  synchronized source. The default budget is six rounds shared across Backtrack
  references within a place step.
- FSS grounding consumes the selected skill's self-contained local intent.
  No-match evidence returns to bounded semantic replanning.
- Physical actions and movement history bind the selected objective ID and
  attempt. Virtual planning-reference switches do not establish execution.
- VerticalMove uses observed stair support and robot-clearance-aware FSS. Each
  local movement returns to Executive assessment, preserving partial-stair and
  landing objectives. The old multi-move transition limit is removed.
- Stop proposes local positioning first, then verifies original-task constraints
  on the next step. Exhausting the agenda alone does not declare completion.

## Retained robot boundary

- `RobotRPCEnv`, ROS 2/Nav2 service, and RGB-D/odometry wire format are retained.
- Both the robot service and panorama configuration retain 90-degree turns.
  The upstream service's 60-degree constant is deliberately not copied.
- The single-model client, YOLO-World detector, compact runner, and JSON output
  remain. Upstream LA/VA model routing, raw request logging, evaluation services,
  experiment ablations, and research retrieval controls are not imported.
- Robot radius and obstacle-height settings remain 0.4 m and 0.3–1.4 m.
- Runtime imports resolve entirely within this repository and its declared
  third-party dependencies. No upstream checkout, symlink, or `PYTHONPATH`
  reference is required.

The core modules retain upstream names to make future comparison practical.
`task_progress` and some LLM call names therefore remain internal identifiers;
they implement NavProbe's dynamic task state rather than the old TPU protocol.

## Validation

The old robot release passed 23 tests before migration. The migrated release
passes 109 tests, including retained robot-interface and model-routing checks,
ported NavProbe agenda/protocol/grounding tests, and four full runner cases.
Obsolete tests asserting immutable task lists and retired vertical point
grounding are replaced by native agenda and stair-FSS behavior tests.

The full runner cases use simulated HTTP responses and deterministic model
responses with actual RPC decoding, RGB-D map updates, panorama capture,
Executive/skill logic, FSS generation, and execution. They cover:

- staying at the endpoint followed by Executive completion;
- FSS-selected movement, movement-image memory, and subsequent completion;
- model failure followed by robot stop and exception finalization;
- stopping at the step limit without incorrectly declaring completion.

Seven representative complete model requests were compared against the frozen
upstream snapshot: initialization, Executive assessment, virtual reference,
last retrieval batch, terminal continuation, semantic selection, and waypoint
grounding. Their serialized request contents were byte-for-byte identical.

Additional checks: all 87 package Python modules parse, internal imports resolve,
undefined-name/unused-import checks pass, and the public robot CLI help loads.

To run the regression suite in an environment with the declared dependencies:

```bash
pip install -e '.[test]'
python -m pytest -q
ruff check navclaw tests --select F821,F822,F823,F401
python scripts/run_vln_robot.py --help
```

No physical robot, live model service, or detector weights were used for these
checks. Calibrated sensor input and actual navigation remain deployment checks;
stair traversal also requires an appropriate physical controller, since this
migration does not replace the Nav2 bridge with a stair controller.
