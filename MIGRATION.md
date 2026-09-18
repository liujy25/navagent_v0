# Current synchronization: prompt contracts and compact decision guidance

Date: 2026-09-18. Upstream: `bfa2805b219bfe7e0e0f721ed9bb0a47f9049f61`.
Robot baseline: `98f050a1bd5fb9ee774269e4c15b86037c44393b`.
The package version remains `0.3.0`; launch and robot interfaces are unchanged.

## Review and scope

The robot's pre-migration prompt text matched the pre-P0 NavProbe implementation.
Its differences were deliberate release boundaries: always-dynamic task state,
no research ablations, no raw model request logging, and no auxiliary stop verifier.
Those differences are preserved. Only `navprobe/agent/visual_policy.py` changes
in the runtime package; this is a selective source migration, not a file overwrite
or a dependency on the sibling repository.

The execution review verified virtual Backtrack in `visual_navigation.py`,
physical versus historical `stay` in that same planner, and post-movement
`movement=move` in `visual_step_execution.py`. The revised prompt descriptions
match these existing execution paths; the paths themselves are unchanged.

## Migrated changes

- **P0 contracts:** declare conditional `terminal_check` arguments and valid JSON
  examples; advertise the actual enabled tools and order; distinguish retrieval
  eligibility from remaining budget. Tool masks, parsers, and transaction behavior
  remain unchanged.
- **P0 semantics:** preserve original movement/stopping relations, negation,
  ordinals, and ambiguity. Helpers cannot strengthen the original task. A planning
  reference is evidence, not the robot's pose or new execution. Anchor-grounded
  waypoints execute from the physical pose without a mandatory anchor return.
  Historical `stay` explicitly retains its physical-return exception.
- **P1.1 retrieval:** ask decision-relevant questions, batch complementary evidence,
  preserve conclusions and uncertainty, and end retrieval when movement/new
  observations are needed. Every pending final batch still requires interpretation.
  Re-reading is allowed for a specific unresolved gap, detail, or contradiction.
- **P1.2 compact guidance:** replace prescribed Executive/Skill workflows with
  concise decision guidance while retaining tool contracts, role boundaries,
  structured state, and image meanings. Grounded hypotheses may guide investigation
  but do not establish task completion.

P1.3 was an upstream handoff audit, not another runtime change. The robot keeps
Executive assessment and retrieval conclusions in the Skill input, and passes
the self-contained selected local intent to the waypoint grounder. No full-agenda
packet or repeated retrieval transcript was added to grounding.

## Robot boundary retained

RobotRPCEnv, ROS 2/Nav2, RGB-D wire formats, four-view 90-degree panorama capture,
single-model routing, detector configuration, FSS clearance, retrieval budget,
and compact episode output are unchanged. No simulation runner, research logging,
model-routing split, or experiment control was imported.

The compact robot output does not contain raw per-module prompts/responses.
Detailed retrieval-cause analysis like the research log review still requires
separate instrumentation; it cannot be reconstructed from aggregate token totals.

## Validation and prompt usage

- The unmodified robot baseline passed all **109** tests; the migrated repository
  passes **117**, including simulated RPC/FSS/full-episode stop and failure cases.
- Added regression coverage checks the 16 Executive retrieval/pending-evidence/
  historical-reference/terminal combinations, parser-compatible schema examples,
  first-step tool eligibility, anchor action masks, physical/historical `stay`,
  original-goal authority, complementary retrieval, and uncertain terminal handoff
  with budget remaining. The historical `stay` execution test checks `movement=move`.
- All ten matched default text fixtures equal upstream's frozen P1.2 fixtures
  exactly (system and every user text block). This is text parity; image payloads
  were not part of this comparison. Upstream fixture SHA-256:
  `8692216b0651c48ca8870e29324b6416a6d8f9cfe61c896664e8c55e1584a1bc`.
- Undefined-name/unused-import checks and robot CLI help pass. No physical robot,
  ROS process, model service, simulator rollout, or detector weights were invoked.

See [PROMPT_USAGE_REPORT.md](docs/prompt-sync-20260918/PROMPT_USAGE_REPORT.md)
for all character/token deltas and [prompt_usage.json](docs/prompt-sync-20260918/prompt_usage.json)
for complete before/after text, flags, and source hashes. The standalone measurement
script uses this repository's constructors and has no sibling-checkout dependency.
Typical text token changes: regular Executive `2165 -> 1925` (-11.1%), physical
Skill `1250 -> 1207` (-3.4%), historical Skill `1390 -> 1466` (+5.5%). The last
increase reflects the repaired reference/execution contract. These local text
counts exclude images, API envelopes, generated tokens, and changing histories.

Navigation success and stronger-model compatibility remain unmeasured here.
New offline/robot logs are needed for behavioral analysis; paired evaluation
remains deferred. In an environment with the declared test dependencies, run:

```bash
python -m pytest -q
ruff check navprobe tests docs/prompt-sync-20260918/measure_prompt_usage.py --select F821,F822,F823,F401
python scripts/run_vln_robot.py --help
```

The records below describe earlier synchronizations and are retained as history.

---

# Previous synchronization: NavProbe naming and VA cleanup

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
