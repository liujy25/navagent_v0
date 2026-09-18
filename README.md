# NavAgent v0

NavAgent v0 is the standalone robot-facing release of the NavProbe visual
language navigation (VLN) agent. It contains one algorithm configuration:

- a Task Executive with a mutable agenda, stable objective IDs, and attempt history;
- active entity-field retrieval with up to six rounds per place step;
- entity knowledge consolidation after retrieval;
- virtual planning-reference backtracking and Executive-owned terminal checks;
- Frontier Skeleton Sampling (FSS) waypoint grounding from RGB-D;
- observation-grounded stair FSS and local vertical movements;
- YOLO-World landmark detection.

The applicable algorithm and prompt contracts are synchronized through NavProbe
commit `bfa2805` (2026-09-18), retaining this repository's robot interfaces. The original instruction remains
available to the Executive and semantic skill selector. The waypoint grounder
receives a self-contained selected skill and candidate evidence. Task updates
use the native agenda/predicate protocol and commit atomically; an empty agenda
does not establish task completion.

This repository does not import or require the sibling NavProbe repository.
Future changes there do not change this installed agent. See
[MIGRATION.md](MIGRATION.md) for the synchronization boundary and validation. The
[2026-09-18 prompt usage report](docs/prompt-sync-20260918/PROMPT_USAGE_REPORT.md)
records reproducible text character/token changes.

It intentionally excludes Habitat/HM3D/R2R evaluation runners, experiment
ablations, replay tools, batch launchers, web viewers, and raw prompt/response
logging.

## Installation

Python 3.9 or newer is required. Install the algorithm package with:

```bash
pip install -e .
```

When upgrading from 0.2.x, rerun this installation command in each environment
that runs the agent or robot bridge. The Python package is now `navprobe`, and
the model environment variable is `NAVPROBE_MODEL`. Update custom imports and
launch configurations accordingly; the old package and variable are not aliases.
The repository/distribution name `navagent-v0`, command `navagent-vln`, and the
two script filenames remain unchanged.

YOLO-World weights are not bundled. Put a compatible checkpoint at
`weights/yolov8x-world.pt`, or pass `--yolow-model`.

The optional ROS 2 RPC server additionally needs ROS 2, Nav2, `cv_bridge`,
`message_filters`, and `tf2_ros` in the robot environment.

## Run on a robot

Start the provided ROS 2/Nav2 bridge on the robot computer:

```bash
python scripts/serve_robot_rpc.py --host 0.0.0.0 --port 1877
```

Then run the agent on the compute machine:

```bash
export OPENAI_API_KEY="..."
export NAVPROBE_MODEL="gpt-5.2"

python scripts/run_vln_robot.py \
  --robot-url http://ROBOT_IP:1877 \
  --instruction "Leave the office, turn left, and stop by the red chair." \
  --model "$NAVPROBE_MODEL" \
  --output outputs/episode.json
```

For an OpenAI-compatible provider, also set `OPENAI_BASE_URL` or pass
`--base-url`. The `--model` value is used by every language and visual-language
module; no separate VA configuration is required or read.

By default the agent captures four RGB-D views facing front, left, back, and
right, separated by 90-degree left turns. The physical capture order is front,
left, back, right; prompts present the four images independently in the fixed
front, back, left, right order. The bridge and robot controller must return to
the original heading after the fourth turn.

## Robot API

`RobotRPCEnv` expects these endpoints:

- `GET /health`
- `POST /reset` with `{"goal": "..."}`
- `GET /current_episode`
- `POST /get_obs`
- `POST /turn` with `{"direction": "left|right"}`
- `POST /move` with world-frame `x`, `y`, `yaw`, and optional `z`
- `POST /stop`
- `POST /finalize_run`

Observations contain RGB, metric depth, camera intrinsics, `T_cam_odom`,
`T_odom_base`, and a world-frame pose. Yaw is expressed in degrees throughout
the provided bridge.

The optional `--output` writes one compact JSON result with termination reason,
declared completion, step/action counts, final pose, aggregate token usage, and
a compact action trace. No per-step images or raw LLM prompts are written.

See [ALGORITHM.md](ALGORITHM.md) for the fixed execution flow.

The minimal frontier-detection dependency retained from the original project is
distributed under the terms in
[`LICENSES/frontier_exploration-LICENSE`](LICENSES/frontier_exploration-LICENSE).
