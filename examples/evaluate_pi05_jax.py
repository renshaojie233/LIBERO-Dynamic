#!/usr/bin/env python3
"""Evaluate an OpenPI JAX Pi0.5 checkpoint on LIBERO-Dynamic."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_arguments, base_environment, require_existing_path, run_example, server_pythonpath, validate_common_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_port=18124)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-config", default="pi05_libero")
    args = parser.parse_args()
    validate_common_arguments(args)
    args.checkpoint = require_existing_path(args.checkpoint, "--checkpoint")

    env = base_environment(args)
    env["PYTHONPATH"] = server_pythonpath(
        args,
        args.openpi_root / "src",
        args.openpi_root / "packages" / "openpi-client" / "src",
    )
    command = [
        str(args.server_python),
        str(args.openpi_root / "scripts" / "serve_policy.py"),
        "--port",
        str(args.port),
        "policy:checkpoint",
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        str(args.checkpoint),
    ]
    return run_example(args, server_command=command, server_environment=env)


if __name__ == "__main__":
    raise SystemExit(main())
