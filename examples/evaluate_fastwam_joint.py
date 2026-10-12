#!/usr/bin/env python3
"""Evaluate a FastWAM-Joint checkpoint on LIBERO-Dynamic."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import SCRIPTS, add_common_arguments, require_existing_path, run_example, server_pythonpath, validate_common_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_port=18126)
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    args = parser.parse_args()
    validate_common_arguments(args)
    args.fastwam_root = require_existing_path(args.fastwam_root, "--fastwam-root")
    args.checkpoint = require_existing_path(args.checkpoint, "--checkpoint")
    args.dataset_stats = require_existing_path(args.dataset_stats, "--dataset-stats")
    args.wan_root = require_existing_path(args.wan_root, "--wan-root")

    command = [
        str(args.server_python),
        str(SCRIPTS / "serve_fastwam_joint_openpi_ws.py"),
        "--fastwam-root",
        str(args.fastwam_root),
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-stats",
        str(args.dataset_stats),
        "--device",
        "cuda",
        "--port",
        str(args.port),
        "--actions-per-chunk",
        str(args.actions_per_chunk),
        "--num-inference-steps",
        str(args.num_inference_steps),
    ]
    server_env = {
        "DIFFSYNTH_MODEL_BASE_PATH": str(args.wan_root),
        "PYTHONPATH": server_pythonpath(
            args,
            args.fastwam_root,
            args.fastwam_root / "src",
            args.fastwam_root / "experiments" / "libero",
            args.openpi_root / "packages" / "openpi-client" / "src",
        ),
    }
    return run_example(args, server_command=command, server_environment=server_env)


if __name__ == "__main__":
    raise SystemExit(main())
