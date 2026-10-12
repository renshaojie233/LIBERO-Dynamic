#!/usr/bin/env python3
"""Render and validate the policy-free LIBERO-Dynamic speed sweep.

This entry point orchestrates the repository's reviewed physical preview
renderer. Each shard runs in a fresh Python process so EGL contexts are
created after process start rather than inherited through ``fork``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "libero" / "libero" / "dynamic_configs"
PREVIEW_SCRIPT = REPO_ROOT / "scripts" / "render_dynamic_config_previews.py"
DEFAULT_SPEEDS = (0.03, 0.06, 0.09)
DEFAULT_EXPECTED_ITEMS = 600
QA_THRESHOLDS = {
    "max_platform_path_error_m": 0.01,
    "max_object_to_platform_error_m": 0.08,
    "max_tilt_deg": 35.0,
    "max_drop_m": 0.04,
    "min_visible_ratio": 0.75,
}


@dataclass(frozen=True)
class Shard:
    speed_mps: float
    speed_label: str
    start_index: int
    limit: int
    output_dir: Path
    log_path: Path

    @property
    def expected_indices(self) -> list[int]:
        return list(range(self.start_index, self.start_index + self.limit))

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def speed_label(speed_mps: float) -> str:
    hundredths_of_a_metre_per_second = int(round(float(speed_mps) * 100.0))
    return f"speed_{hundredths_of_a_metre_per_second:03d}"


def build_shards(
    output_dir: Path,
    speeds: Iterable[float],
    *,
    items_per_speed: int,
    shard_size: int,
) -> list[Shard]:
    if items_per_speed <= 0:
        raise ValueError("items_per_speed must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    shards = []
    for speed in speeds:
        label = speed_label(speed)
        for start in range(0, items_per_speed, shard_size):
            limit = min(shard_size, items_per_speed - start)
            end = start + limit - 1
            shard_dir = output_dir / label / "shards" / f"items_{start:03d}_{end:03d}"
            shards.append(
                Shard(
                    speed_mps=float(speed),
                    speed_label=label,
                    start_index=start,
                    limit=limit,
                    output_dir=shard_dir,
                    log_path=output_dir / "logs" / f"{label}__items_{start:03d}_{end:03d}.log",
                )
            )
    return shards


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_is_complete(shard: Shard) -> bool:
    if not shard.manifest_path.is_file():
        return False
    try:
        records = read_manifest(shard.manifest_path).get("records", [])
    except (OSError, ValueError):
        return False
    indices = sorted(int(record.get("index", -1)) for record in records)
    if indices != shard.expected_indices:
        return False
    for record in records:
        video = record.get("video")
        if record.get("status") != "ok" or record.get("error"):
            return False
        if not video or not Path(video).is_file() or Path(video).stat().st_size <= 0:
            return False
    return True


def preview_command(shard: Shard, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PREVIEW_SCRIPT),
        "--config-dir",
        str(CONFIG_DIR),
        "--output-dir",
        str(shard.output_dir),
        "--levels",
        "l1",
        "l2",
        "l3",
        "--cycles",
        str(args.cycles),
        "--speed-mps",
        str(shard.speed_mps),
        "--fps",
        str(args.fps),
        "--sim-dt",
        str(args.sim_dt),
        "--camera-size",
        str(args.camera_size),
        "--start-index",
        str(shard.start_index),
        "--limit",
        str(shard.limit),
        "--keep-static-support-contact",
        "--static-support-clearance-m",
        str(args.static_support_clearance_m),
    ]
    if not args.overlay_traces:
        command.append("--no-overlay-traces")
    return command


def run_shard(shard: Shard, args: argparse.Namespace, environment: dict[str, str]) -> tuple[Shard, int]:
    shard.output_dir.mkdir(parents=True, exist_ok=True)
    shard.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = preview_command(shard, args)
    with shard.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return shard, result.returncode


def qa_failure_reasons(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record.get("status") != "ok" or record.get("error"):
        reasons.append(f"status={record.get('status')} error={record.get('error')}")
    physics = record.get("preview_physics") or {}
    motion = physics.get("motion_diagnostics") or {}
    safety = physics.get("safety_diagnostics") or {}
    targets = set(record.get("target_names") or []) | set(motion) | set(safety)
    for target in sorted(targets):
        target_motion = motion.get(target) or {}
        target_safety = safety.get(target) or {}
        platform_error = float(target_motion.get("max_platform_path_error_m", float("inf")))
        follow_error = float(target_motion.get("max_object_to_platform_error_m", float("inf")))
        tilt = float(target_motion.get("max_tilt_deg", float("inf")))
        drop = float(target_motion.get("base_bottom_z", float("inf"))) - float(
            target_motion.get("min_bottom_z", float("-inf"))
        )
        collision_count = int(target_safety.get("collision_count", 10**9))
        if platform_error > QA_THRESHOLDS["max_platform_path_error_m"]:
            reasons.append(f"{target}:platform_path={platform_error}")
        if follow_error > QA_THRESHOLDS["max_object_to_platform_error_m"]:
            reasons.append(f"{target}:object_follow={follow_error}")
        if tilt > QA_THRESHOLDS["max_tilt_deg"]:
            reasons.append(f"{target}:tilt={tilt}")
        if drop > QA_THRESHOLDS["max_drop_m"]:
            reasons.append(f"{target}:drop={drop}")
        if collision_count != 0:
            reasons.append(f"{target}:collisions={collision_count}")
        for view in ("birdview", "agentview"):
            ratio = float(target_safety.get(f"{view}_visible_ratio", -1.0))
            if ratio < QA_THRESHOLDS["min_visible_ratio"]:
                reasons.append(f"{target}:{view}_visible={ratio}")
    video = record.get("video")
    if not video or not Path(video).is_file():
        reasons.append("video_missing")
    elif Path(video).stat().st_size <= 0:
        reasons.append("video_empty")
    return reasons


def materialize_final_videos(records: list[dict[str, Any]], output_dir: Path) -> None:
    final_dir = output_dir / "final_videos"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    for record in records:
        source = Path(record["video"])
        destination_dir = (
            final_dir / record["speed_label"] / record["level"] / record["suite"]
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / (
            f"{int(record['index']):03d}__task{int(record['task_id']):02d}"
            f"__{record['variant_name']}.mp4"
        )
        os.link(source, destination)
        record["source_video"] = str(source)
        record["final_video"] = str(destination)


def probe_video(video_path: str, ffprobe: str) -> str | None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        return result.stderr.strip() or f"ffprobe exited with {result.returncode}"
    try:
        stream = json.loads(result.stdout)["streams"][0]
        if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
            return "invalid_dimensions"
        if float(stream.get("duration", 0) or 0) <= 0:
            return "invalid_duration"
        if int(stream.get("nb_frames", 0) or 0) <= 0:
            return "invalid_frame_count"
    except (ValueError, KeyError, IndexError) as error:
        return f"invalid_ffprobe_output:{error}"
    return None


def consolidate(shards: list[Shard], output_dir: Path, *, probe_videos: bool, probe_workers: int) -> dict[str, Any]:
    records = []
    structural_errors = []
    seen = {}
    for shard in shards:
        if not shard.manifest_path.is_file():
            structural_errors.append(f"missing manifest: {shard.manifest_path}")
            continue
        for source_record in read_manifest(shard.manifest_path).get("records", []):
            record = dict(source_record)
            key = (shard.speed_label, int(record.get("index", -1)))
            if key in seen:
                structural_errors.append(f"duplicate record {key}: {seen[key]} and {shard.manifest_path}")
                continue
            seen[key] = str(shard.manifest_path)
            record["speed_mps"] = shard.speed_mps
            record["speed_label"] = shard.speed_label
            record["source_manifest"] = str(shard.manifest_path)
            reasons = qa_failure_reasons(record)
            record["qa_pass"] = not reasons
            record["qa_failure_reasons"] = reasons
            records.append(record)

    for speed in sorted({shard.speed_label for shard in shards}):
        expected = sorted(
            index
            for shard in shards
            if shard.speed_label == speed
            for index in shard.expected_indices
        )
        actual = sorted(record["index"] for record in records if record["speed_label"] == speed)
        if actual != expected:
            structural_errors.append(
                f"{speed}: expected {len(expected)} unique indices, found {len(actual)}"
            )

    records.sort(key=lambda record: (record["speed_label"], int(record["index"])))
    if not structural_errors:
        materialize_final_videos(records, output_dir)

    probe_failures = []
    ffprobe = shutil.which("ffprobe") if probe_videos else None
    if probe_videos and ffprobe and not structural_errors:
        with ThreadPoolExecutor(max_workers=max(1, probe_workers)) as executor:
            future_to_record = {
                executor.submit(probe_video, record["final_video"], ffprobe): record
                for record in records
            }
            for future in as_completed(future_to_record):
                error = future.result()
                if error:
                    record = future_to_record[future]
                    probe_failures.append(
                        {"video": record["final_video"], "error": error}
                    )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    qa_failures = [record for record in records if not record.get("qa_pass")]
    summary = {
        "records_total": len(records),
        "raw_ok": sum(record.get("status") == "ok" and not record.get("error") for record in records),
        "qa_pass": len(records) - len(qa_failures),
        "qa_warnings": len(qa_failures),
        "qa_warning_examples": [
            {
                "speed_label": record["speed_label"],
                "index": record["index"],
                "variant_id": record.get("variant_id"),
                "reasons": record["qa_failure_reasons"],
            }
            for record in qa_failures[:20]
        ],
        "qa_thresholds": QA_THRESHOLDS,
        "structural_errors": structural_errors,
        "video_probe": {
            "requested": probe_videos,
            "ffprobe_available": ffprobe is not None,
            "passed": len(records) - len(probe_failures) if ffprobe else None,
            "failed": len(probe_failures) if ffprobe else None,
            "failure_examples": probe_failures[:20],
        },
        "manifest": str(manifest_path),
        "final_videos": str(output_dir / "final_videos"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=list(DEFAULT_SPEEDS))
    parser.add_argument("--limit-per-speed", type=int, default=DEFAULT_EXPECTED_ITEMS)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="0", help="CUDA/EGL device index.")
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Saved review-video rate only. Default: 60; this does not change simulation stepping.",
    )
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--static-support-clearance-m", type=float, default=0.001)
    parser.add_argument("--overlay-traces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-workers", type=int, default=16)
    parser.add_argument("--fail-on-qa-warning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = build_shards(
        args.output_dir,
        args.speeds,
        items_per_speed=args.limit_per_speed,
        shard_size=args.shard_size,
    )
    pending = [shard for shard in shards if not (args.resume and shard_is_complete(shard))]
    print(
        json.dumps(
            {
                "speeds": args.speeds,
                "items_per_speed": args.limit_per_speed,
                "shards_total": len(shards),
                "shards_pending": len(pending),
                "workers": args.workers,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for shard in pending:
            print(" ".join(preview_command(shard, args)))
        return 0

    environment = os.environ.copy()
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LIBERO_CONFIG_PATH": str(REPO_ROOT / "libero" / "libero" / "configs"),
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )
    failed_shards = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_shard, shard, args, environment): shard
            for shard in pending
        }
        for future in as_completed(futures):
            shard, returncode = future.result()
            state = "ok" if returncode == 0 and shard_is_complete(shard) else "failed"
            print(f"[{state}] {shard.speed_label} {shard.start_index}:{shard.start_index + shard.limit}", flush=True)
            if state != "ok":
                failed_shards.append(shard)

    summary = consolidate(
        shards,
        args.output_dir,
        probe_videos=args.probe_videos,
        probe_workers=args.probe_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if failed_shards or summary["structural_errors"] or summary["video_probe"]["failed"]:
        return 1
    if args.fail_on_qa_warning and summary["qa_warnings"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
