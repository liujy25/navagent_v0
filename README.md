# NavAgent v0

NavAgent v0 is the robot-facing release of the canonical NavClaw visual
language navigation (VLN) agent. It contains one algorithm configuration:

- independent task-progress updating and progress-conditioned navigation;
- active episodic retrieval with up to eight rounds per place step;
- entity knowledge consolidation after retrieval;
- virtual planning-reference backtracking and TPU-owned terminal checks;
- Frontier Skeleton Sampling (FSS) waypoint grounding from RGB-D;
- multi-floor vertical-transition handling;
- YOLO-World landmark detection.

The prompt/runtime contracts match the current NavClaw mainline: strict initial
progress items, TPU ordered `tool_calls`, direction-bearing PCNP actions, fixed
action grounding by the Waypoint Planner, round-aligned EKM traces, and explicit
vertical-grounding success/failure responses. The raw instruction is used only
for initialization; later planning uses maintained progress state.

It intentionally excludes Habitat/HM3D/R2R evaluation runners, experiment
ablations, replay tools, batch launchers, web viewers, and raw prompt/response
logging.

## Installation

Python 3.9 or newer is required. Install the algorithm package with:

```bash
pip install -e .
```

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
export NAVCLAW_MODEL="gpt-5.2"

python scripts/run_vln_robot.py \
  --robot-url http://ROBOT_IP:1877 \
  --instruction "Leave the office, turn left, and stop by the red chair." \
  --model "$NAVCLAW_MODEL" \
  --output outputs/episode.json
```

For an OpenAI-compatible provider, also set `OPENAI_BASE_URL` or pass
`--base-url`. The `--model` value is used by every language and visual-language
module; there are no hidden VA/LA model overrides.

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
