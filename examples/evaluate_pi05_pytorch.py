#!/usr/bin/env python3
"""Evaluate a LeRobot PyTorch Pi0.5 checkpoint on LIBERO-Dynamic."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import SCRIPTS, add_common_arguments, require_existing_path, run_example, validate_common_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_port=18125)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-tokenizer", type=Path, required=True)
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    args = parser.parse_args()
    validate_common_arguments(args)
    args.checkpoint = require_existing_path(args.checkpoint, "--checkpoint")
    args.local_tokenizer = require_existing_path(args.local_tokenizer, "--local-tokenizer")

    command = [
        str(args.server_python),
        str(SCRIPTS / "serve_lerobot_pi05_openpi_ws.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--local-tokenizer",
        str(args.local_tokenizer),
        "--device",
        "cuda",
        "--port",
        str(args.port),
        "--actions-per-chunk",
        str(args.actions_per_chunk),
    ]
    return run_example(args, server_command=command)


if __name__ == "__main__":
    raise SystemExit(main())
