#!/usr/bin/env python3
"""Merge LIBERO-dynamic latency frames into rollout MP4s.

This is an optional post-processing tool: the core benchmark can run without
modifying eval code. During evaluation, LIBERO-dynamic writes latency frame PNGs
and event metadata to sidecar JSON. This script scans the eval output and the
sidecar logs, then writes enhanced MP4s with latency frames inserted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

VIDEO_RE = re.compile(r"episode=task(?P<task_id>\d+)_trial(?P<trial_id>\d+)")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_dynamic_logs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            out.append(path)
        elif path.is_dir():
            out.extend(p for p in path.rglob("*.json") if not p.name.endswith(".lock"))
    return sorted(set(out))


def _index_rollout_videos(eval_output_dir: Path) -> dict[tuple[str, int, int], Path]:
    videos: dict[tuple[str, int, int], Path] = {}
    for path in sorted(eval_output_dir.rglob("*.mp4")):
        if path.parent.name != "videos":
            continue
        if "--dynamic-enhanced" in path.name:
            continue
        if any(part in {"dynamic_real_time_videos", "dynamic_enhanced_videos", "dynamic_enhanced_videos_offset30", "model_input_observation_videos"} for part in path.parts):
            continue
        match = VIDEO_RE.search(path.name)
        if match is None:
            continue
        suite_name = path.parent.parent.name
        task_id = int(match.group("task_id"))
        trial_id = int(match.group("trial_id"))
        videos[(suite_name, task_id, trial_id)] = path
    return videos


def _resize_to(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if frame.shape[:2] == (target_h, target_w):
        return frame
    return np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR))


def _load_png(path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = np.asarray(imageio.imread(path))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return _resize_to(frame.astype(np.uint8), target_shape)


def _iter_video_frames(path: Path, target_shape: tuple[int, int]):
    if not path.exists():
        return
    reader = imageio.get_reader(path)
    try:
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            yield _resize_to(frame.astype(np.uint8), target_shape)
    finally:
        reader.close()


def _get_state_renderer(renderers: dict[tuple[str, int, tuple[int, int]], Any], suite_name: str, task_id: int, target_shape: tuple[int, int]):
    key = (suite_name, int(task_id), target_shape)
    if key in renderers:
        return renderers[key]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    target_h, target_w = target_shape
    camera_height = int(target_h)
    camera_width = int(target_w // 2) if target_w >= 2 else int(target_w)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(int(task_id))
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.reset()
    renderers[key] = env
    return env


def _restore_model_snapshot(inner: Any, model_snapshot: dict[str, np.ndarray] | None) -> None:
    if not model_snapshot:
        return
    model = inner.sim.model
    for key, value in model_snapshot.items():
        if not key.startswith("model_"):
            continue
        name = key[len("model_") :]
        if not hasattr(model, name):
            continue
        target = getattr(model, name)
        value = np.asarray(value)
        if target.shape != value.shape:
            continue
        target[:] = value


def _render_state_frame(
    env: Any,
    target_shape: tuple[int, int],
    *,
    state: np.ndarray | None = None,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    model_snapshot: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    inner = getattr(env, "env", env)
    _restore_model_snapshot(inner, model_snapshot)
    if state is not None:
        try:
            inner.sim.set_state_from_flattened(np.asarray(state, dtype=float))
        except Exception:
            # DynamicWrapper has compatibility logic for legacy init states whose
            # length differs after installing carrier joints. Use it if present.
            if hasattr(env, "set_state"):
                env.set_state(np.asarray(state, dtype=float))
            else:
                raise
    else:
        if qpos is None or qvel is None:
            raise ValueError("state renderer needs either flattened state or qpos/qvel")
        inner.sim.data.qpos[:] = qpos
        inner.sim.data.qvel[:] = qvel
    inner.sim.forward()
    inner._post_process()
    inner._update_observables(force=True)
    obs = inner._get_observations()
    images = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key in obs:
            images.append(np.asarray(obs[key], dtype=np.uint8)[::-1, ::-1])
    if not images:
        raise RuntimeError("state renderer produced no camera observations")
    frame = np.concatenate(images, axis=1)
    return _resize_to(frame.astype(np.uint8), target_shape)


def _iter_state_frames(ref: dict[str, Any], target_shape: tuple[int, int], renderers: dict[tuple[str, int, tuple[int, int]], Any]):
    state_path = Path(ref["path"])
    if not state_path.exists():
        return
    with np.load(state_path) as data:
        env = _get_state_renderer(renderers, str(ref["suite"]), int(ref["task_id"]), target_shape)
        model_snapshot = {
            key: np.asarray(data[key]).copy()
            for key in data.files
            if key.startswith("model_")
        }
        if "states" in data:
            for state in data["states"]:
                yield _render_state_frame(
                    env, target_shape, state=state, model_snapshot=model_snapshot
                )
        else:
            for qpos, qvel in zip(data["qpos"], data["qvel"]):
                yield _render_state_frame(
                    env, target_shape, qpos=qpos, qvel=qvel, model_snapshot=model_snapshot
                )


def _append_latency_ref(writer, ref: Any, target_shape: tuple[int, int], renderers: dict[tuple[str, int, tuple[int, int]], Any]) -> tuple[int, list[str]]:
    if isinstance(ref, dict) and ref.get("type") == "video":
        video_path = Path(ref["path"])
        if not video_path.exists():
            return 0, [str(video_path)]
        count = 0
        for latency_frame in _iter_video_frames(video_path, target_shape):
            writer.append_data(latency_frame)
            count += 1
        return count, []

    if isinstance(ref, dict) and ref.get("type") == "state_npz":
        state_path = Path(ref["path"])
        if not state_path.exists():
            return 0, [str(state_path)]
        count = 0
        for latency_frame in _iter_state_frames(ref, target_shape, renderers):
            writer.append_data(latency_frame)
            count += 1
        return count, []

    latency_path = Path(ref)
    latency_frame = _load_png(latency_path, target_shape)
    if latency_frame is None:
        return 0, [str(latency_path)]
    writer.append_data(latency_frame)
    return 1, []


def _expand_latency_frame_paths(
    event: dict[str, Any],
    frame_paths: list[Path],
    *,
    fps: float,
    preserve_duration: bool,
) -> list[Path]:
    if not preserve_duration or not frame_paths:
        return frame_paths
    latency_seconds = max(0.0, float(event.get("latency_seconds", 0.0) or 0.0))
    target_count = max(len(frame_paths), int(round(latency_seconds * float(fps))))
    if target_count <= len(frame_paths):
        return frame_paths

    frame_times = [float(x) for x in event.get("latency_frame_times", [])]
    if len(frame_times) != len(frame_paths):
        frame_times = [
            (i + 1) * latency_seconds / max(1, len(frame_paths))
            for i in range(len(frame_paths))
        ]

    expanded: list[Path] = []
    for i in range(target_count):
        t = (i + 1) / float(fps)
        # Use the latest sampled latency frame at or before display time.
        idx = 0
        for j, frame_time in enumerate(frame_times):
            if frame_time <= t + 1e-9:
                idx = j
            else:
                break
        expanded.append(frame_paths[min(idx, len(frame_paths) - 1)])
    return expanded


def _collect_episode_inserts(
    episode: dict[str, Any],
    *,
    num_steps_wait: int,
    min_latency_seconds: float,
    drop_prepend: bool,
    fps: float,
    preserve_latency_duration: bool,
) -> tuple[dict[int, list[Any]], list[Any], list[dict[str, Any]]]:
    inserts_after: dict[int, list[Any]] = defaultdict(list)
    prepend: list[Any] = []
    skipped: list[dict[str, Any]] = []

    for event in episode.get("events", []):
        video_path = event.get("latency_video_path")
        state_path = event.get("latency_state_path")
        frame_paths = [Path(p) for p in event.get("latency_frame_paths", [])]
        if state_path:
            frame_refs: list[Any] = [{
                "type": "state_npz",
                "path": str(state_path),
                "suite": str(episode.get("suite")),
                "task_id": int(episode.get("task_id")),
            }]
            num_frames = int(event.get("latency_state_frame_count", 0) or 0)
        elif video_path:
            frame_refs = [{"type": "video", "path": str(video_path)}]
            num_frames = int(event.get("latency_video_frame_count", 0) or 0)
        else:
            if not frame_paths:
                continue
            frame_paths = _expand_latency_frame_paths(
                event,
                frame_paths,
                fps=fps,
                preserve_duration=preserve_latency_duration,
            )
            frame_refs = frame_paths
            num_frames = len(frame_paths)

        if not frame_refs:
            continue
        latency_seconds = float(event.get("latency_seconds", 0.0) or 0.0)
        if latency_seconds < float(min_latency_seconds):
            skipped.append({
                "reason": "latency_below_merge_threshold",
                "step_idx": int(event.get("step_idx", -1)),
                "latency_seconds": latency_seconds,
                "num_frames": num_frames,
            })
            continue
        step_idx = int(event.get("insert_before_control_step", event.get("step_idx", -1)))
        frame_idx = step_idx - int(num_steps_wait)
        if frame_idx < 0:
            if drop_prepend:
                skipped.append({
                    "reason": "insert_before_first_video_frame",
                    "step_idx": step_idx,
                    "frame_idx": frame_idx,
                    "latency_seconds": latency_seconds,
                    "num_frames": num_frames,
                })
            else:
                prepend.extend(frame_refs)
            continue
        inserts_after[frame_idx].extend(frame_refs)

    return inserts_after, prepend, skipped

def _write_merged_video(
    *,
    original_video: Path,
    output_video: Path,
    inserts_after: dict[int, list[Any]],
    prepend: list[Any],
    fps: float,
) -> dict[str, Any]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(original_video)
    writer = imageio.get_writer(output_video, fps=fps)
    inserted_frames = 0
    missing_frames: list[str] = []
    original_frames = 0
    target_shape: tuple[int, int] | None = None
    renderers: dict[tuple[str, int, tuple[int, int]], Any] = {}

    try:
        for frame_idx, frame in enumerate(reader):
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frame = frame.astype(np.uint8)
            target_shape = frame.shape[:2]

            if frame_idx == 0:
                for latency_ref in prepend:
                    count, missing = _append_latency_ref(writer, latency_ref, target_shape, renderers)
                    inserted_frames += count
                    missing_frames.extend(missing)

            writer.append_data(frame)
            original_frames += 1

            for latency_ref in inserts_after.get(frame_idx, []):
                count, missing = _append_latency_ref(writer, latency_ref, target_shape, renderers)
                inserted_frames += count
                missing_frames.extend(missing)
    finally:
        reader.close()
        writer.close()
        for renderer in renderers.values():
            try:
                renderer.close()
            except Exception:
                pass

    unused_insert_steps = sorted(k for k in inserts_after if k >= original_frames)
    return {
        "original_video": str(original_video),
        "enhanced_video": str(output_video),
        "original_frames": int(original_frames),
        "inserted_latency_frames": int(inserted_frames),
        "missing_latency_frames": missing_frames,
        "unused_insert_steps": unused_insert_steps,
    }


def merge_all(
    *,
    eval_output_dir: Path,
    dynamic_log_paths: list[Path],
    output_dir: Path,
    num_steps_wait: int,
    fps: float,
    min_latency_seconds: float,
    drop_prepend: bool,
    preserve_latency_duration: bool,
) -> dict[str, Any]:
    videos = _index_rollout_videos(eval_output_dir)
    logs = _iter_dynamic_logs(dynamic_log_paths)
    results: list[dict[str, Any]] = []
    unmatched_episodes: list[dict[str, Any]] = []

    for log_path in logs:
        payload = _read_json(log_path)
        for episode in payload.get("episodes", []):
            suite_name = str(episode.get("suite"))
            task_id = int(episode.get("task_id"))
            episode_index = int(episode.get("episode_index"))
            key = (suite_name, task_id, episode_index)
            inserts_after, prepend, skipped = _collect_episode_inserts(
                episode,
                num_steps_wait=num_steps_wait,
                min_latency_seconds=min_latency_seconds,
                drop_prepend=drop_prepend,
                fps=fps,
                preserve_latency_duration=preserve_latency_duration,
            )
            requested_frames = len(prepend) + sum(len(v) for v in inserts_after.values())
            if requested_frames == 0:
                continue
            original_video = videos.get(key)
            if original_video is None:
                unmatched_episodes.append(
                    {
                        "dynamic_log": str(log_path),
                        "suite": suite_name,
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "requested_latency_frames": requested_frames,
                    }
                )
                continue

            rel_parent = original_video.parent.parent.name
            output_video = output_dir / rel_parent / "videos" / original_video.name.replace(
                ".mp4", "--dynamic-enhanced.mp4"
            )
            item = _write_merged_video(
                original_video=original_video,
                output_video=output_video,
                inserts_after=inserts_after,
                prepend=prepend,
                fps=fps,
            )
            item.update(
                {
                    "dynamic_log": str(log_path),
                    "suite": suite_name,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "requested_latency_frames": requested_frames,
                    "skipped_events": skipped,
                }
            )
            results.append(item)

    summary = {
        "eval_output_dir": str(eval_output_dir),
        "dynamic_logs": [str(p) for p in logs],
        "output_dir": str(output_dir),
        "num_steps_wait": int(num_steps_wait),
        "fps": float(fps),
        "min_latency_seconds": float(min_latency_seconds),
        "drop_prepend": bool(drop_prepend),
        "preserve_latency_duration": bool(preserve_latency_duration),
        "merged_videos": results,
        "unmatched_episodes": unmatched_episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dynamic_video_merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-dir", required=True, type=Path)
    parser.add_argument(
        "--dynamic-log",
        action="append",
        type=Path,
        default=None,
        help="Dynamic sidecar JSON file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--dynamic-log-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--min-latency-seconds",
        type=float,
        default=0.05,
        help="Only merge latency frames from events at or above this latency. Defaults to 0.05s to avoid visual flicker from near-zero wall-clock gaps.",
    )
    parser.add_argument(
        "--drop-prepend",
        dest="drop_prepend",
        action="store_true",
        default=True,
        help="Drop latency frames that would be inserted before the first rollout frame instead of prepending them. Enabled by default.",
    )
    parser.add_argument(
        "--keep-prepend",
        dest="drop_prepend",
        action="store_false",
        help="Keep latency frames that map before the first rollout frame by prepending them.",
    )
    parser.add_argument(
        "--preserve-latency-duration",
        dest="preserve_latency_duration",
        action="store_true",
        default=False,
        help="Repeat saved latency frames as needed so inserted frames occupy approximately latency_seconds at the output FPS. Disabled by default; prefer saving real latency frames at the desired FPS.",
    )
    parser.add_argument(
        "--no-preserve-latency-duration",
        dest="preserve_latency_duration",
        action="store_false",
        help="Insert each saved latency PNG exactly once. This is the default and preserves one output frame per real sampled latency frame.",
    )
    args = parser.parse_args()

    log_paths: list[Path] = []
    if args.dynamic_log:
        log_paths.extend(args.dynamic_log)
    if args.dynamic_log_dir is not None:
        log_paths.append(args.dynamic_log_dir)
    if not log_paths:
        default_log_dir = args.eval_output_dir / "libero_dynamic_logs"
        if default_log_dir.exists():
            log_paths.append(default_log_dir)
        else:
            log_paths.append(Path("./libero_dynamic_logs"))

    output_dir = args.output_dir or (args.eval_output_dir / "dynamic_enhanced_videos")
    summary = merge_all(
        eval_output_dir=args.eval_output_dir,
        dynamic_log_paths=log_paths,
        output_dir=output_dir,
        num_steps_wait=args.num_steps_wait,
        fps=args.fps,
        min_latency_seconds=args.min_latency_seconds,
        drop_prepend=args.drop_prepend,
        preserve_latency_duration=args.preserve_latency_duration,
    )
    print(json.dumps({
        "summary_path": summary["summary_path"],
        "num_merged_videos": len(summary["merged_videos"]),
        "num_unmatched_episodes": len(summary["unmatched_episodes"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
