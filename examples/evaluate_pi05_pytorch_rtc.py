#!/usr/bin/env python3
"""Evaluate LeRobot PyTorch Pi0.5 with the RTC-queue runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import SCRIPTS, add_common_arguments, require_existing_path, run_example, validate_common_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_port=18145)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-tokenizer", type=Path, required=True)
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--rtc-execution-horizon", type=int, default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
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
        "--enable-rtc",
    ]
    if args.rtc_execution_horizon is not None:
        command.extend(["--rtc-execution-horizon", str(args.rtc_execution_horizon)])
    if args.rtc_max_guidance_weight is not None:
        command.extend(["--rtc-max-guidance-weight", str(args.rtc_max_guidance_weight)])
    return run_example(
        args,
        server_command=command,
        evaluator_script=SCRIPTS / "run_pi05_rtc_eval.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
