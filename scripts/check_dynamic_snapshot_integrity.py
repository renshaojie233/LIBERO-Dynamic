#!/usr/bin/env python3
"""Validate LIBERO-dynamic latency snapshots and merged video metadata.

This catches the class of bugs where offline-rendered latency frames are built
from incomplete MuJoCo state and no longer match the live rollout scene.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_STATE_KEYS = {
    "qpos",
    "qvel",
    "states",
    "times",
    "model_body_pos",
    "model_body_quat",
    "model_geom_pos",
    "model_geom_quat",
    "model_site_pos",
    "model_site_quat",
    "model_cam_pos",
    "model_cam_quat",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_logs(output_dir: Path) -> list[Path]:
    log_dir = output_dir / "libero_dynamic_logs"
    if log_dir.exists():
        return sorted(p for p in log_dir.rglob("*.json") if not p.name.endswith(".lock"))
    return sorted(p for p in output_dir.rglob("libero_dynamic.json"))


def _check_npz(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing latency state npz: {path}"]
    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing = sorted(REQUIRED_STATE_KEYS - keys)
            if missing:
                errors.append(f"{path}: missing keys {missing}")
                return errors
            if data["states"].ndim != 2 or data["states"].shape[0] == 0:
                errors.append(f"{path}: invalid states shape {data['states'].shape}")
            if data["qpos"].shape[0] != data["states"].shape[0]:
                errors.append(f"{path}: qpos/state frame count mismatch")
            if data["qvel"].shape[0] != data["states"].shape[0]:
                errors.append(f"{path}: qvel/state frame count mismatch")
            if data["times"].shape[0] != data["states"].shape[0]:
                errors.append(f"{path}: times/state frame count mismatch")
            for key in sorted(k for k in keys if k.startswith("model_")):
                if data[key].size == 0:
                    errors.append(f"{path}: empty {key}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to read {path}: {exc}")
    return errors


def check_output_dir(output_dir: Path, *, allow_cleaned: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checked_npz = 0
    latency_events = 0
    cleaned_state_paths: list[str] = []

    logs = _iter_logs(output_dir)
    if not logs:
        errors.append(f"no dynamic logs found under {output_dir}")
    for log_path in logs:
        payload = _read_json(log_path)
        for episode in payload.get("episodes", []):
            for event in episode.get("events", []):
                state_path = event.get("latency_state_path")
                if not state_path:
                    continue
                latency_events += 1
                path = Path(state_path)
                if path.exists():
                    checked_npz += 1
                    errors.extend(_check_npz(path))
                elif allow_cleaned:
                    cleaned_state_paths.append(str(path))
                else:
                    errors.append(f"state npz missing; rerun before cleanup or pass --allow-cleaned: {path}")

    if cleaned_state_paths:
        warnings.append(
            "state npz files were already cleaned: "
            f"count={len(cleaned_state_paths)}, examples={cleaned_state_paths[:3]}"
        )

    for merge_path in output_dir.rglob("dynamic_video_merge_summary.json"):
        payload = _read_json(merge_path)
        for record in payload.get("merged_videos", []):
            missing = record.get("missing_latency_frames") or []
            if missing:
                errors.append(f"{merge_path}: missing latency frames for {record.get('enhanced_video')}: {missing}")
            if record.get("inserted_latency_frames", 0) <= 0:
                warnings.append(f"{merge_path}: no inserted latency frames for {record.get('enhanced_video')}")

    return {
        "output_dir": str(output_dir),
        "logs": [str(p) for p in logs],
        "latency_events_with_state": latency_events,
        "checked_npz": checked_npz,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(int(args.task_id))
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(args.camera_size),
        camera_widths=int(args.camera_size),
    )
    try:
        env.set_init_state(suite.get_task_init_states(int(args.task_id))[0])
        env.step(np.zeros(7))
        time.sleep(float(args.probe_latency_seconds))
        env.step(np.zeros(7))
        events = [
            e
            for e in env.dynamic_controller.applied_events
            if e.get("reason") == "latency_drift_continuous"
        ]
        if not events:
            return {"ok": False, "errors": ["probe produced no latency_drift_continuous event"]}
        state_path = Path(events[-1]["latency_state_path"])
        errors = _check_npz(state_path)
        render_error = None
        if not errors:
            try:
                scripts_dir = Path(__file__).resolve().parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from merge_dynamic_latency_videos import _iter_state_frames

                frames = list(
                    _iter_state_frames(
                        {
                            "type": "state_npz",
                            "path": str(state_path),
                            "suite": args.suite,
                            "task_id": int(args.task_id),
                        },
                        (int(args.camera_size), int(args.camera_size) * 2),
                        {},
                    )
                )
                if not frames:
                    render_error = "state renderer produced no frames"
            except Exception as exc:  # noqa: BLE001
                render_error = f"state renderer failed: {exc}"
        if render_error:
            errors.append(render_error)
        return {
            "ok": not errors,
            "errors": errors,
            "state_path": str(state_path),
            "latency_seconds": events[-1].get("latency_seconds"),
            "latency_source": events[-1].get("latency_source"),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-cleaned", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--probe-latency-seconds", type=float, default=0.12)
    args = parser.parse_args()

    if args.probe:
        result = run_probe(args)
    else:
        result = check_output_dir(args.output_dir.resolve(), allow_cleaned=args.allow_cleaned)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
