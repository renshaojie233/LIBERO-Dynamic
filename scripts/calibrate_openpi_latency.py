#!/usr/bin/env python3
"""Measure single-request OpenPI policy latency on LIBERO observations."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from libero.libero.envs import OffScreenRenderEnv

from eval_dynamic_dummy_policy import action_dim, build_items, task_bddl, write_effective_config_files
from eval_dynamic_libero_style import (
    OpenPIWebsocketPolicy,
    preview_step_with_physical_dynamic,
    setup_preview_compatible_dynamic,
    update_preview_state_offsets,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), float(q)))


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(sample["latency_seconds"]) for sample in samples]
    return {
        "count": len(latencies),
        "mean_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "median_seconds": percentile(latencies, 50),
        "p90_seconds": percentile(latencies, 90),
        "p95_seconds": percentile(latencies, 95),
        "p99_seconds": percentile(latencies, 99),
        "min_seconds": float(np.min(latencies)) if latencies else 0.0,
        "max_seconds": float(np.max(latencies)) if latencies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", type=Path, default=repo_root / "libero" / "libero" / "dynamic_configs")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-items", type=int, default=12)
    parser.add_argument("--samples-per-item", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-env-steps", type=int, default=10)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", type=int, default=8124)
    parser.add_argument("--openpi-resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--rtc-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure steady-state RTC-conditioned requests instead of ordinary action-chunk requests.",
    )
    parser.add_argument(
        "--rtc-inference-delay-steps",
        type=int,
        default=None,
        help="Control-step delay supplied to RTC; required with --rtc-conditioning.",
    )
    args = parser.parse_args()

    if args.rtc_conditioning and args.rtc_inference_delay_steps is None:
        parser.error("--rtc-inference-delay-steps is required with --rtc-conditioning")
    if args.rtc_inference_delay_steps is not None and int(args.rtc_inference_delay_steps) < 0:
        parser.error("--rtc-inference-delay-steps must be non-negative")

    args.output_json = args.output_json.expanduser().resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.keep_static_support_contact = True
    args.static_support_clearance_m = 0.001
    args.platform_settle_steps = 40
    args.realign_after_platform_settle = True
    args.preview_compatible_settle_steps = 80
    config_dir = args.config_dir.expanduser().resolve()
    effective_dir = args.output_json.parent / f"{args.output_json.stem}_effective_configs"
    wrapper_disabled = args.output_json.parent / "_calibration_preview_wrapper_disabled.yaml"
    config_path_by_level = write_effective_config_files(config_dir, effective_dir, args.levels, args)
    items = build_items(
        config_dir,
        args.levels,
        args.traj_indices,
        keep_static_support_contact=True,
        static_support_clearance_m=0.001,
        config_path_by_level=config_path_by_level,
    )
    items = items[int(args.start_index) : int(args.start_index) + int(args.num_items)]
    if not items:
        raise RuntimeError("No calibration items selected.")

    os.environ.setdefault("LIBERO_DYNAMIC_AUTO_FINALIZE", "false")
    policy: OpenPIWebsocketPolicy | None = None
    samples: list[dict[str, Any]] = []
    warmup_done = 0
    previous_original_chunk: np.ndarray | None = None

    for item in items:
        # Formal old-physics evaluation uses the preview-compatible carrier
        # path. Disable env-wrapper config loading and construct the identical
        # physical scene explicitly so calibration observes the same images.
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(wrapper_disabled)
        os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
        os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = item["variant_id"]
        suite, task, bddl = task_bddl(item["suite"], item["task_id"])
        init_states = suite.get_task_init_states(item["task_id"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=int(args.camera_size),
            camera_widths=int(args.camera_size),
        )
        try:
            env.seed(int(args.seed))
            obs = env.reset()
            obs, preview_states, _ = setup_preview_compatible_dynamic(
                env,
                item,
                args,
                init_states[int(args.init_state_id)],
            )
            zero_action = np.zeros((action_dim(env),), dtype=np.float32)
            preview_elapsed = 0.0
            preview_dt = 1.0 / 20.0
            for _ in range(max(0, int(args.warmup_env_steps))):
                obs, _, _, _ = preview_step_with_physical_dynamic(
                    env,
                    zero_action,
                    preview_states,
                    preview_elapsed,
                    preview_dt,
                )
                preview_elapsed += preview_dt
                update_preview_state_offsets(preview_states, preview_elapsed)
            if policy is None:
                policy = OpenPIWebsocketPolicy(
                    host=str(args.openpi_host),
                    port=int(args.openpi_port),
                    resize_size=int(args.openpi_resize_size),
                    replan_steps=int(args.replan_steps),
                    action_dim=action_dim(env),
                )
                if args.rtc_conditioning and not policy.supports_rtc_conditioning:
                    raise RuntimeError(
                        "--rtc-conditioning requires a policy server reporting "
                        "rtc_conditioning_supported=true"
                    )
            policy.reset()
            for sample_index in range(int(args.samples_per_item)):
                started = time.perf_counter()
                if args.rtc_conditioning:
                    if previous_original_chunk is None:
                        previous_original_chunk = np.zeros(
                            (int(args.replan_steps), action_dim(env)),
                            dtype=np.float32,
                        )
                    result = policy.get_action_chunk_rtc(
                        obs,
                        getattr(task, "language", ""),
                        prev_chunk_left_over=previous_original_chunk,
                        inference_delay_steps=int(args.rtc_inference_delay_steps),
                    )
                    previous_original_chunk = np.asarray(
                        result.get("original_actions", result["actions"]),
                        dtype=np.float32,
                    )
                else:
                    policy.get_action_chunk(obs, getattr(task, "language", ""))
                elapsed = time.perf_counter() - started
                sample = {
                    "item_id": item["item_id"],
                    "sample_index": sample_index,
                    "latency_seconds": float(elapsed),
                    "warmup": warmup_done < int(args.warmup_requests),
                    "rtc_conditioning": bool(args.rtc_conditioning),
                    "rtc_inference_delay_steps": (
                        int(args.rtc_inference_delay_steps) if args.rtc_conditioning else None
                    ),
                }
                if sample["warmup"]:
                    warmup_done += 1
                else:
                    samples.append(sample)
                obs, _, _, _ = preview_step_with_physical_dynamic(
                    env,
                    zero_action,
                    preview_states,
                    preview_elapsed,
                    preview_dt,
                )
                preview_elapsed += preview_dt
                update_preview_state_offsets(preview_states, preview_elapsed)
        finally:
            env.close()

    output = {
        "openpi_host": str(args.openpi_host),
        "openpi_port": int(args.openpi_port),
        "openpi_resize_size": int(args.openpi_resize_size),
        "replan_steps": int(args.replan_steps),
        "rtc_conditioning": bool(args.rtc_conditioning),
        "rtc_inference_delay_steps": (
            int(args.rtc_inference_delay_steps) if args.rtc_conditioning else None
        ),
        "warmup_requests": int(args.warmup_requests),
        "summary": summarize(samples),
        "samples": samples,
    }
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)
    print(args.output_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
