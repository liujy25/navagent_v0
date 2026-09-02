from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navclaw.env.robot_rpc_server import RobotRPCController
from navclaw.env.robot_rpc_server import build_parser
from navclaw.env.robot_rpc_server import create_app
from navclaw.env.robot_rpc_server import main

__all__ = ["RobotRPCController", "build_parser", "create_app", "main"]


if __name__ == "__main__":
    main()
