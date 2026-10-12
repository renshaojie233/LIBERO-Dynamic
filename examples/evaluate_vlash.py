#!/usr/bin/env python3
"""Evaluate official VLASH Pi0.5 Async5 on LIBERO-Dynamic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import (
    SCRIPTS,
    add_common_arguments,
    base_environment,
    require_existing_path,
    run_example,
    server_pythonpath,
    validate_common_arguments,
)


OFFICIAL_ACTIONS_PER_CHUNK = 5
OFFICIAL_MAX_TRAINED_DELAY_STEPS = 4


def measured_latency_to_steps(seconds: float, control_frequency_hz: int) -> int:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("--measured-latency-seconds must be finite and non-negative")
    if control_frequency_hz <= 0:
        raise ValueError("--control-frequency-hz must be positive")
    return int(math.ceil(seconds * float(control_frequency_hz) - 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_port=18240)
    parser.add_argument(
        "--vlash-root",
        type=Path,
        required=True,
        help="Root of the official mit-han-lab/vlash checkout.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Hugging Face cache containing google/paligemma-3b-pt-224.",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable Hugging Face and transformers network access after assets are cached.",
    )
    parser.add_argument(
        "--measured-latency-seconds",
        type=float,
        required=True,
        help=(
            "Clean-GPU batch-one end-to-end latency. It is mapped once to "
            "ceil(latency * control_frequency_hz) delayed visual steps."
        ),
    )
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.set_defaults(
        replan_steps=OFFICIAL_ACTIONS_PER_CHUNK,
        calibrated_latency_seconds=0.0,
    )
    args = parser.parse_args()
    validate_common_arguments(args)

    args.vlash_root = require_existing_path(args.vlash_root, "--vlash-root")
    args.checkpoint = require_existing_path(args.checkpoint, "--checkpoint")
    if args.hf_home is not None:
        args.hf_home = require_existing_path(args.hf_home, "--hf-home")
    if not (args.vlash_root / "vlash").is_dir():
        raise FileNotFoundError(
            f"VLASH Python package was not found under {args.vlash_root / 'vlash'}"
        )
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"VLASH config.json was not found in {args.checkpoint}")
    if int(args.num_workers) != 1:
        raise ValueError(
            "evaluate_vlash.py starts one stateful policy server and therefore requires "
            "--num-workers 1; use one independent server per worker for a formal sweep"
        )
    if int(args.replan_steps) != OFFICIAL_ACTIONS_PER_CHUNK:
        raise ValueError("official VLASH Pi0.5 Async5 requires --replan-steps 5")
    if float(args.calibrated_latency_seconds) != 0.0:
        raise ValueError(
            "VLASH latency is represented by delayed visual observations; "
            "--calibrated-latency-seconds must remain 0 to avoid double counting"
        )

    delay_steps = measured_latency_to_steps(
        float(args.measured_latency_seconds),
        int(args.control_frequency_hz),
    )
    if delay_steps > OFFICIAL_MAX_TRAINED_DELAY_STEPS:
        raise ValueError(
            f"measured latency maps to {delay_steps} steps, outside the official "
            f"checkpoint training range 0-{OFFICIAL_MAX_TRAINED_DELAY_STEPS}"
        )

    env = base_environment(args)
    env["PYTHONPATH"] = server_pythonpath(args, args.vlash_root)
    if args.hf_home is not None:
        env["HF_HOME"] = str(args.hf_home)
    if bool(args.offline):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    server_command = [
        str(args.server_python),
        str(SCRIPTS / "serve_vlash_openpi_ws.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    evaluator_extra_args = [
        "--reviewed-old-physics",
        "--control-mode",
        "vlash_future_state",
        "--vlash-visual-delay-steps",
        str(delay_steps),
        "--vlash-measured-latency-seconds",
        str(args.measured_latency_seconds),
        "--control-freq",
        str(args.control_frequency_hz),
    ]
    print(
        "VLASH latency mapping: "
        f"{args.measured_latency_seconds:.9f}s -> {delay_steps} visual steps "
        f"({delay_steps / float(args.control_frequency_hz):.3f}s represented), "
        "with current robot state and no generic queue delay",
        flush=True,
    )
    return run_example(
        args,
        server_command=server_command,
        server_environment=env,
        evaluator_script=SCRIPTS / "run_realtime_eval.py",
        evaluator_extra_args=evaluator_extra_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
