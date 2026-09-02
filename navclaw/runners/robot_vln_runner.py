from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from navclaw.agent.vln_runner import run_vln_episode
from navclaw.config.vln_runtime import VlnRuntimeConfig
from navclaw.env.robot_rpc_env import RobotRPCEnv
from navclaw.perception.detectors.yoloworld_local import DEFAULT_YOLOWORLD_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical NavClaw VLN agent on a robot RPC service."
    )
    parser.add_argument("--robot-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("NAVCLAW_MODEL", ""),
        help="OpenAI-compatible multimodal model name (or set NAVCLAW_MODEL).",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--yolow-model", default=DEFAULT_YOLOWORLD_MODEL_PATH)
    parser.add_argument("--detector-device", default="cuda")
    parser.add_argument("--max-place-steps", type=int, default=40)
    parser.add_argument(
        "--output",
        help="Optional path for one compact episode-result JSON file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = str(args.model).strip()
    if not model:
        raise ValueError("--model or NAVCLAW_MODEL is required")
    env = RobotRPCEnv(base_url=str(args.robot_url))
    result = run_vln_episode(
        env=env,
        instruction=str(args.instruction),
        model=model,
        api_key=args.api_key,
        base_url=args.base_url,
        detector_model_path=str(args.yolow_model),
        detector_device=str(args.detector_device),
        config=VlnRuntimeConfig(
            max_place_steps=int(args.max_place_steps),
        ),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
