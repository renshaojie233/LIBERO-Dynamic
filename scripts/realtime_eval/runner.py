from __future__ import annotations

import argparse
import json
import inspect
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import eval_dynamic_libero_style as legacy

from .control import CONTROL_MODE_CHOICES
from .runtime import RealtimeEpisode


if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            options = []
            for option in option_strings:
                options.append(option)
                if option.startswith("--"):
                    options.append("--no-" + option[2:])
            super().__init__(options, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction


def _patch_legacy_torch_load() -> None:
    try:
        import torch
    except Exception:
        return
    if "weights_only" in inspect.signature(torch.load).parameters:
        return
    original_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("weights_only", None)
        return original_load(*args, **kwargs)

    torch.load = compatible_load


_patch_legacy_torch_load()


def _assign_worker_egl_device(slot: int) -> None:
    raw = os.environ.get("LIBERO_EGL_DEVICE_IDS", "").strip()
    if not raw:
        return
    devices = [value.strip() for value in raw.replace(";", ",").split(",") if value.strip()]
    if not devices:
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = devices[int(slot) % len(devices)]


def _rate(count: int, total: int) -> float:
    return 0.0 if int(total) <= 0 else round(float(count) / float(total), 9)


def _finite_numbers(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _action_metric_aggregate(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    episode_metrics = [row.get("action_metrics", {}).get(metric, {}) for row in rows]
    episode_metrics = [value for value in episode_metrics if int(value.get("count", 0)) > 0]
    means = _finite_numbers([value.get("mean") for value in episode_metrics])
    p95s = _finite_numbers([value.get("p95") for value in episode_metrics])
    maxima = _finite_numbers([value.get("max") for value in episode_metrics])
    return {
        "definition": "||a_t||_2" if metric == "action_l2" else "||a_t - a_(t-1)||_2",
        "episodes": len(episode_metrics),
        "samples": sum(int(value.get("count", 0)) for value in episode_metrics),
        "mean_of_episode_means": None if not means else round(sum(means) / len(means), 9),
        "mean_of_episode_p95": None if not p95s else round(sum(p95s) / len(p95s), 9),
        "max": None if not maxima else round(max(maxima), 9),
    }


def _physical_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    flag_counts: dict[str, int] = {}
    for row in rows:
        row_targets = row.get("diagnostics", {}).get("targets", {})
        if isinstance(row_targets, dict):
            targets.extend(value for value in row_targets.values() if isinstance(value, dict))
        for flag in row.get("diagnostics", {}).get("flags", []):
            flag_type = str(flag.get("type", "unknown")) if isinstance(flag, dict) else "unknown"
            flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1

    def values(key: str) -> list[float]:
        return _finite_numbers([target.get(key) for target in targets])

    platform = values("max_platform_path_error_m")
    follow = values("max_follow_error_m")
    tilt = values("max_tilt_deg")
    bottom = values("min_relative_bottom_z_m")
    birdview = values("birdview_offscreen_ratio")
    agentview = values("agentview_offscreen_ratio")
    return {
        "target_count": len(targets),
        "max_platform_path_error_m": None if not platform else max(platform),
        "max_follow_error_m": None if not follow else max(follow),
        "max_tilt_deg": None if not tilt else max(tilt),
        "min_relative_bottom_z_m": None if not bottom else min(bottom),
        "max_birdview_offscreen_ratio": None if not birdview else max(birdview),
        "max_agentview_offscreen_ratio": None if not agentview else max(agentview),
        "diagnostic_flag_counts": dict(sorted(flag_counts.items())),
    }


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda row: row.get("item_id", ""))
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + ("\n" if ordered else ""),
        encoding="utf-8",
    )
    total = len(ordered)
    ok_status = sum(1 for row in ordered if row.get("status") == "ok")
    diagnostic_ok = sum(1 for row in ordered if row.get("diagnostics", {}).get("ok", False))
    success_count = sum(1 for row in ordered if row.get("success"))
    issues = [
        row
        for row in ordered
        if row.get("status") != "ok" or not row.get("diagnostics", {}).get("ok", False)
    ]
    action_runtime = [row.get("action_runtime", {}) for row in ordered]
    controller_summaries = [row.get("controller_summary", {}) for row in ordered]
    executed_actions = sum(int(value.get("executed_actions", 0)) for value in action_runtime)
    underflow_steps = sum(int(value.get("queue_underflow_steps", 0)) for value in action_runtime)
    summary = {
        "runtime_version": "realtime_v2",
        "metric_schema_version": "libero_dynamic_main_v1",
        "total": total,
        "ok_status": ok_status,
        "ok_status_rate": _rate(ok_status, total),
        "diagnostic_ok": diagnostic_ok,
        "diagnostic_ok_rate": _rate(diagnostic_ok, total),
        "success_count": success_count,
        "success_rate": _rate(success_count, total),
        "control_modes": sorted({str(row.get("control_mode")) for row in ordered}),
        "policies": sorted({str(row.get("policy")) for row in ordered}),
        "latency_sources": sorted({str(row.get("latency_source")) for row in ordered}),
        "sim_latency_decoupled_from_wallclock": sorted(
            {bool(row.get("sim_latency_decoupled_from_wallclock")) for row in ordered}
        ),
        "shared_policy_latency_contaminated_items": sum(
            1 for row in ordered if row.get("shared_policy_latency_contaminated")
        ),
        "runtime_metrics": {
            "executed_actions": executed_actions,
            "enqueued_chunks": sum(int(value.get("enqueued_chunks", 0)) for value in action_runtime),
            "queue_underflow_steps": underflow_steps,
            "queue_underflow_rate": _rate(underflow_steps, executed_actions),
            "policy_requests": sum(int(value.get("policy_requests", 0)) for value in controller_summaries),
            "max_policy_latency_s": max(
                [0.0]
                + _finite_numbers([value.get("policy_latency_max_s") for value in controller_summaries])
            ),
            "max_simulated_policy_latency_s": max(
                [0.0]
                + _finite_numbers([value.get("simulated_policy_latency_max_s") for value in controller_summaries])
            ),
            "sync_inference_latency_elapsed_s_total": round(
                sum(float(row.get("sync_inference_latency_elapsed_s", 0.0)) for row in ordered),
                9,
            ),
            "rtc_conditioning_requested_items": sum(
                int(bool(value.get("rtc_conditioning_requested"))) for value in controller_summaries
            ),
            "rtc_conditioning_supported_items": sum(
                int(bool(value.get("rtc_conditioning_supported"))) for value in controller_summaries
            ),
            "rtc_conditioning_used_items": sum(
                int(bool(value.get("rtc_conditioning_used"))) for value in controller_summaries
            ),
        },
        "action_metrics": {
            "action_l2": _action_metric_aggregate(ordered, "action_l2"),
            "action_delta_l2": _action_metric_aggregate(ordered, "action_delta_l2"),
        },
        "physical_diagnostics": _physical_diagnostic_summary(ordered),
        "issues": len(issues),
        "issue_items": [
            {
                "item_id": row.get("item_id"),
                "status": row.get("status"),
                "success": row.get("success"),
                "flags": row.get("diagnostics", {}).get("flags", []),
                "error": row.get("error"),
            }
            for row in issues
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_worker_partition(slot: int, items: list[dict[str, Any]], args_dict: dict[str, Any]) -> list[dict[str, Any]]:
    _assign_worker_egl_device(int(slot))
    args = argparse.Namespace(**args_dict)
    results: list[dict[str, Any]] = []
    shard_path = args.output_dir / f"worker_{int(slot):02d}_manifest.jsonl"
    for idx, item in enumerate(items, start=1):
        print(f"[worker {slot}] [{idx}/{len(items)}] {item['item_id']}", flush=True)
        result = RealtimeEpisode(item, args, worker_slot=int(slot)).run()
        print(
            f"[worker {slot}] {result['item_id']} status={result.get('status')} "
            f"success={result.get('success')} diag_ok={result.get('diagnostics', {}).get('ok')}",
            flush=True,
        )
        results.append(result)
        shard_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
            encoding="utf-8",
        )
    return results


def _split_round_robin(items: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    partitions = [[] for _ in range(int(num_workers))]
    for idx, item in enumerate(items):
        partitions[idx % int(num_workers)].append(item)
    return partitions


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run LIBERO-Dynamic realtime runtime v2.")
    parser.add_argument("--config-dir", type=Path, default=repo_root / "libero" / "libero" / "dynamic_configs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--item-ids-file",
        type=Path,
        default=None,
        help="Optional JSON list of exact item IDs; preserves the listed order and rejects missing IDs.",
    )
    parser.add_argument("--reviewed-old-physics", action="store_true")

    parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODE_CHOICES,
        default="sync_hold",
        help=(
            "sync_hold: block for each chunk, hold robot joints, and advance dynamic physics by the exact latency; "
            "async_queue/rtc_queue: prefetch action chunks through an external controller; "
            "vlash_future_state: official VLASH delayed-vision/current-state chunk protocol. "
            "hold_latency and naive_async_queue are backward-compatible aliases."
        ),
    )
    parser.add_argument(
        "--prefetch-threshold",
        type=int,
        default=5,
        help="Legacy fixed-step prefetch threshold; used only when --chunk-size-threshold is unset.",
    )
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=None,
        help=(
            "LeRobot official async/RTC trigger: request a chunk when "
            "queue_len / replan_steps <= threshold. Official default is 0.5."
        ),
    )
    parser.add_argument(
        "--aggregate-fn-name",
        choices=("weighted_average",),
        default="weighted_average",
        help="LeRobot async aggregate function implemented here: 0.3 * old + 0.7 * new.",
    )
    parser.add_argument(
        "--empty-action",
        choices=("hold", "zero"),
        default="zero",
        help=(
            "Action to execute when the policy queue is empty. zero is the real-robot-style hold/no-op for delta "
            "control; hold is the legacy repeat-last-action behavior."
        ),
    )
    parser.add_argument(
        "--freeze-robot-on-underflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the action queue underflows, restore robot/gripper joint state after the env step so the arm "
            "stays fixed while dynamic scene elements continue to advance. This is the strict real-robot sync hold."
        ),
    )

    parser.add_argument("--policy", choices=("zero", "random", "openpi_ws"), default="zero")
    parser.add_argument("--random-scale", type=float, default=0.02)
    parser.add_argument("--simulated-inference-seconds", type=float, default=0.0)
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", type=int, default=8124)
    parser.add_argument("--openpi-resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--vlash-visual-delay-steps",
        type=int,
        default=1,
        help=(
            "VLASH official LIBERO async delay in control steps. Only used by "
            "--control-mode vlash_future_state. Derive it as "
            "ceil(measured_latency_seconds * control_freq)."
        ),
    )
    parser.add_argument(
        "--vlash-measured-latency-seconds",
        type=float,
        default=None,
        help=(
            "Measured policy latency used to derive VLASH visual staleness: "
            "visual_delay_steps must equal ceil(latency * control_freq). This value is metadata and "
            "must not also be supplied as the generic simulated queue latency."
        ),
    )
    parser.add_argument("--per-worker-policy-endpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-server-count", type=int, default=0)
    parser.add_argument("--policy-port-stride", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--num-steps-wait", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--require-rtc-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the episode unless the connected policy server reports true RTC conditioning support.",
    )
    parser.add_argument("--latency-mode", choices=("disabled", "measured", "fixed"), default="disabled")
    parser.add_argument("--fixed-inference-seconds", type=float, default=None)
    parser.add_argument("--latency-source", choices=("measured_wallclock", "fixed", "empirical"), default="measured_wallclock")
    parser.add_argument("--calibrated-latency-seconds", type=float, default=None)
    parser.add_argument("--calibrated-first-inference-seconds", type=float, default=None)
    parser.add_argument("--latency-samples-json", type=Path, default=None)
    parser.add_argument("--latency-sample-seed", type=int, default=0)

    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-review-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-model-input-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-input-video-layout", choices=("agentview", "dual"), default="dual")
    parser.add_argument(
        "--model-input-video-size",
        type=int,
        default=128,
        help="Saved square size per policy view; does not change policy observations.",
    )
    parser.add_argument("--save-wrist-input-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-step-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=60.0,
        help="Saved simulation-time video rate only. Default: 60; use 100 for short timing diagnostics.",
    )
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--record-warmup", action="store_true")
    parser.add_argument("--record-latency-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-style", choices=("libero", "review"), default="review")

    parser.add_argument("--dynamic-phase-prime-steps", type=int, default=0)
    parser.add_argument("--fixture-passenger-phase-prime-steps", type=int, default=None)
    parser.add_argument("--pre-policy-dynamic-steps", type=int, default=0)
    parser.add_argument("--reset-platforms-after-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-static-support-contact", action="store_true")
    parser.add_argument("--static-support-clearance-m", type=float, default=0.0)
    parser.add_argument("--platform-settle-steps", type=int, default=20)
    parser.add_argument("--realign-after-platform-settle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-compatible-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-compatible-scope", choices=("fragile", "all"), default="fragile")
    parser.add_argument("--preview-compatible-settle-steps", type=int, default=80)

    parser.add_argument("--platform-error-threshold-m", type=float, default=0.01)
    parser.add_argument("--follow-error-threshold-m", type=float, default=0.08)
    parser.add_argument("--tilt-threshold-deg", type=float, default=35.0)
    parser.add_argument("--drop-threshold-m", type=float, default=0.04)
    parser.add_argument("--bottom-baseline-mode", choices=("first_sample", "reset"), default="first_sample")
    parser.add_argument("--offscreen-ratio-threshold", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if float(args.video_fps) <= 0.0:
        raise ValueError("--video-fps must be greater than 0")
    if int(args.camera_size) <= 0:
        raise ValueError("--camera-size must be greater than 0")
    if bool(args.save_model_input_videos) and int(args.model_input_video_size) <= 0:
        raise ValueError("--model-input-video-size must be greater than 0 when model-input video is enabled")
    if args.latency_source == "fixed" and args.calibrated_latency_seconds is None:
        if args.fixed_inference_seconds is not None:
            args.calibrated_latency_seconds = float(args.fixed_inference_seconds)
        else:
            raise ValueError("--calibrated-latency-seconds is required for --latency-source fixed")
    if args.latency_source == "empirical" and args.latency_samples_json is None:
        raise ValueError("--latency-samples-json is required for --latency-source empirical")
    if args.chunk_size_threshold is not None and not (0.0 <= float(args.chunk_size_threshold) <= 1.0):
        raise ValueError("--chunk-size-threshold must be between 0 and 1")
    if int(args.policy_server_count) < 0:
        raise ValueError("--policy-server-count must be >= 0")
    if bool(args.reviewed_old_physics):
        legacy.apply_reviewed_old_physics_profile(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    config_path_by_level = legacy.write_effective_config_files(config_dir, args.output_dir, args.levels, args)
    items = legacy.build_items(
        config_dir,
        args.levels,
        args.traj_indices,
        keep_static_support_contact=bool(args.keep_static_support_contact),
        static_support_clearance_m=float(args.static_support_clearance_m),
        config_path_by_level=config_path_by_level,
    )
    if args.item_ids_file is not None:
        requested = json.loads(args.item_ids_file.expanduser().read_text())
        if not isinstance(requested, list) or not requested or not all(isinstance(x, str) for x in requested):
            raise ValueError("--item-ids-file must contain a non-empty JSON list of strings")
        if len(requested) != len(set(requested)):
            raise ValueError("--item-ids-file contains duplicate item IDs")
        by_id = {item["item_id"]: item for item in items}
        missing = [item_id for item_id in requested if item_id not in by_id]
        if missing:
            raise ValueError(f"requested item IDs were not built: {missing}")
        items = [by_id[item_id] for item_id in requested]
    if args.start_index:
        items = items[int(args.start_index) :]
    if args.limit is not None:
        items = items[: int(args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    args_dict = vars(args).copy()
    results: list[dict[str, Any]] = []
    if int(args.num_workers) <= 1:
        results = run_worker_partition(0, items, args_dict)
    else:
        partitions = _split_round_robin(items, int(args.num_workers))
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(run_worker_partition, slot, partition, args_dict): slot
                for slot, partition in enumerate(partitions)
                if partition
            }
            for future in as_completed(futures):
                slot = futures[future]
                try:
                    results.extend(future.result())
                except Exception as exc:
                    print(f"[worker {slot}] fatal error: {exc!r}", flush=True)
    write_summary(args.output_dir, results)
    return 0
