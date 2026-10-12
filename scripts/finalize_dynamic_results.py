#!/usr/bin/env python3
"""Finalize LIBERO-dynamic outputs after an eval run.

This is safe to run repeatedly. It creates:
- real-time videos with latency frames merged in;
- a manifest that pairs model-observation videos with real-time videos;
- a merge summary JSON.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_dynamic_latency_videos import merge_all, _index_rollout_videos, _iter_dynamic_logs  # noqa: E402


@contextmanager
def _finalize_lock(eval_output_dir: Path):
    lock_path = eval_output_dir / ".libero_dynamic_finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _find_result_jsons(eval_output_dir: Path) -> list[Path]:
    return sorted(
        p for p in eval_output_dir.rglob('gpu*_task*_results.json')
        if not p.name.endswith('_dynamic_results.json')
    )


MODEL_INPUT_VIDEO_RE = re.compile(r"(?:episode=)?task(?P<task_id>\d+)_trial(?P<trial_id>\d+).*--model-input-observations")
RESULT_JSON_RE = re.compile(r"gpu(?P<gpu_id>\d+)_task(?P<task_id>\d+)_results\.json")


def _index_task_results(eval_output_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in _find_result_jsons(eval_output_dir):
        payload = _read_json(path)
        suite = str(payload.get('task_suite') or path.parent.name)
        match = RESULT_JSON_RE.search(path.name)
        task_id = int(payload.get('task_id', match.group('task_id') if match else -1))
        out[(suite, task_id)] = {
            'path': path,
            'payload': payload,
        }
    return out


def _episode_success_from_task_result(task_result: dict[str, Any] | None, episode_index: int) -> bool | None:
    if task_result is None:
        return None
    payload = task_result.get('payload', {})
    if int(episode_index) in [int(x) for x in payload.get('success_episodes', [])]:
        return True
    if int(episode_index) in [int(x) for x in payload.get('failure_episodes', [])]:
        return False
    return None


def _task_description_from_task_result(task_result: dict[str, Any] | None) -> str | None:
    if task_result is None:
        return None
    value = task_result.get('payload', {}).get('task_description')
    return str(value) if value is not None else None


def _gpu_id_from_task_result(task_result: dict[str, Any] | None) -> int | None:
    if task_result is None:
        return None
    value = task_result.get('payload', {}).get('gpu_id')
    return int(value) if value is not None else None


def _episode_status(dynamic_status: Any, success: bool | None) -> str:
    if success is True:
        return 'success'
    if success is False:
        return 'failure'
    return str(dynamic_status) if dynamic_status is not None else 'unknown'


def _index_model_input_observation_videos(eval_output_dir: Path) -> dict[tuple[str, int, int], Path]:
    videos: dict[tuple[str, int, int], Path] = {}
    root = eval_output_dir / 'model_input_observation_videos'
    if not root.exists():
        return videos
    for path in sorted(root.rglob('*.mp4')):
        if path.parent.name != 'videos':
            continue
        match = MODEL_INPUT_VIDEO_RE.search(path.name)
        if match is None:
            continue
        suite_name = path.parent.parent.name
        task_id = int(match.group('task_id'))
        trial_id = int(match.group('trial_id'))
        videos[(suite_name, task_id, trial_id)] = path
    return videos


def _enhanced_video_index(merge_summary: dict[str, Any]) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for item in merge_summary.get('merged_videos', []):
        key = (str(item.get('suite')), int(item.get('task_id')), int(item.get('episode_index')))
        out[key] = str(item.get('enhanced_video'))
    return out


def _latency_frame_roots(dynamic_log_paths: list[Path]) -> list[Path]:
    roots: set[Path] = set()
    for log_path in _iter_dynamic_logs(dynamic_log_paths):
        payload = _read_json(log_path)
        for ep in payload.get('episodes', []):
            for event in ep.get('events', []):
                for frame_path in event.get('latency_frame_paths', []):
                    path = Path(frame_path)
                    # .../<log_stem>_latency_frames/<episode_key>/step_xxxxxx/latency_0000.png
                    if len(path.parents) >= 3:
                        roots.add(path.parents[2])
                video_path = event.get('latency_video_path')
                if video_path:
                    path = Path(video_path)
                    # .../<log_stem>_latency_video_segments/<episode_key>/step_xxxxxx.mp4
                    if len(path.parents) >= 2:
                        roots.add(path.parents[1])
                state_path = event.get('latency_state_path')
                if state_path:
                    path = Path(state_path)
                    # .../<log_stem>_latency_state_segments/<episode_key>/step_xxxxxx.npz
                    if len(path.parents) >= 2:
                        roots.add(path.parents[1])
    return sorted(roots)


def _cleanup_latency_frames(dynamic_log_paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for root in _latency_frame_roots(dynamic_log_paths):
        if not root.exists() or not root.is_dir():
            continue
        shutil.rmtree(root)
        removed.append(str(root))
    return removed


def _build_manifest(*, eval_output_dir: Path, dynamic_log_paths: list[Path], merge_summary: dict[str, Any]) -> dict[str, Any]:
    observation_videos = _index_rollout_videos(eval_output_dir)
    model_input_videos = _index_model_input_observation_videos(eval_output_dir)
    enhanced_videos = _enhanced_video_index(merge_summary)
    task_results = _index_task_results(eval_output_dir)
    logs = _iter_dynamic_logs(dynamic_log_paths)
    episodes: list[dict[str, Any]] = []

    dynamic_episode_keys = set()
    for log_path in logs:
        payload = _read_json(log_path)
        for ep in payload.get('episodes', []):
            suite = str(ep.get('suite'))
            task_id = int(ep.get('task_id'))
            episode_index = int(ep.get('episode_index'))
            key = (suite, task_id, episode_index)
            dynamic_episode_keys.add(key)
            task_result = task_results.get((suite, task_id))
            success = _episode_success_from_task_result(task_result, episode_index)
            episodes.append({
                'task_suite': suite,
                'task_id': task_id,
                'task_description': _task_description_from_task_result(task_result),
                'episode_index': episode_index,
                'success': success,
                'status': _episode_status(ep.get('status'), success),
                'model_observation_video': str(model_input_videos[key]) if key in model_input_videos else None,
                'real_time_video': enhanced_videos.get(key),
                'dynamic_log': str(log_path),
                'num_dynamic_events': len(ep.get('events', [])),
                'num_latency_trace': len(ep.get('latency_trace', [])),
            })

    for key, video_path in observation_videos.items():
        if key in dynamic_episode_keys:
            continue
        suite, task_id, episode_index = key
        task_result = task_results.get((suite, task_id))
        success = _episode_success_from_task_result(task_result, episode_index)
        episodes.append({
            'task_suite': suite,
            'task_id': task_id,
            'task_description': _task_description_from_task_result(task_result),
            'episode_index': episode_index,
            'success': success,
            'status': _episode_status('static_or_no_dynamic_log', success),
            'model_observation_video': str(model_input_videos[key]) if key in model_input_videos else None,
            'real_time_video': None,
            'dynamic_log': None,
            'num_dynamic_events': 0,
            'num_latency_trace': 0,
        })

    episodes.sort(key=lambda x: (x['task_suite'], x['task_id'], x['episode_index']))
    tasks = []
    for (suite, task_id), item in sorted(task_results.items()):
        payload = item['payload']
        tasks.append({
            'task_suite': suite,
            'task_id': task_id,
            'task_description': payload.get('task_description'),
            'successes': payload.get('successes'),
            'total_episodes': payload.get('total_episodes'),
            'success_episodes': payload.get('success_episodes'),
            'failure_episodes': payload.get('failure_episodes'),
            'source_result_json': str(item['path']),
        })
    return {
        'benchmark': 'LIBERO-dynamic',
        'result_format': 'libero_compatible_dynamic_summary',
        'eval_output_dir': str(eval_output_dir),
        'result_jsons': [str(p) for p in _find_result_jsons(eval_output_dir)],
        'merge_summary_path': merge_summary.get('summary_path'),
        'num_tasks': len(tasks),
        'tasks': tasks,
        'num_episodes': len(episodes),
        'episodes': episodes,
    }


def _write_dynamic_task_result_jsons(*, eval_output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    task_results = _index_task_results(eval_output_dir)
    if not task_results:
        return []

    episodes_by_task: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in manifest.get('episodes', []):
        key = (str(episode.get('task_suite')), int(episode.get('task_id')))
        episodes_by_task.setdefault(key, []).append(episode)

    written: list[str] = []
    for key, item in sorted(task_results.items()):
        suite, task_id = key
        source_payload = dict(item['payload'])
        source_path = Path(item['path'])
        gpu_id = _gpu_id_from_task_result(item)
        if gpu_id is None:
            gpu_id = int(source_payload.get('gpu_id', 0))
        dynamic_payload = dict(source_payload)
        dynamic_payload.update({
            'benchmark': 'LIBERO-dynamic',
            'source_result_json': str(source_path),
            'dynamic_result_format': 'libero_compatible_task_result',
            'dynamic_video_fields': ['model_observation_video', 'real_time_video'],
            'dynamic_results_summary': str(eval_output_dir / 'dynamic_results_summary.json'),
            'dynamic_episodes': episodes_by_task.get((suite, task_id), []),
        })
        output_path = source_path.parent / f'gpu{gpu_id}_task{task_id}_dynamic_results.json'
        output_path.write_text(
            json.dumps(dynamic_payload, indent=4, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        written.append(str(output_path))
    return written


def finalize_dynamic_results(
    *,
    eval_output_dir: Path,
    dynamic_log_dir: Path | None = None,
    output_dir: Path | None = None,
    num_steps_wait: int = 30,
    fps: float = 24.0,
    min_latency_seconds: float = 0.05,
    cleanup_latency_frames: bool = True,
) -> dict[str, Any]:
    eval_output_dir = eval_output_dir.resolve()
    dynamic_log_paths = [dynamic_log_dir or (eval_output_dir / 'libero_dynamic_logs')]
    output_dir = output_dir or (eval_output_dir / 'dynamic_real_time_videos')

    with _finalize_lock(eval_output_dir):
        merge_summary = merge_all(
            eval_output_dir=eval_output_dir,
            dynamic_log_paths=dynamic_log_paths,
            output_dir=output_dir,
            num_steps_wait=num_steps_wait,
            fps=fps,
            min_latency_seconds=min_latency_seconds,
            drop_prepend=True,
            preserve_latency_duration=False,
        )
        manifest = _build_manifest(
            eval_output_dir=eval_output_dir,
            dynamic_log_paths=dynamic_log_paths,
            merge_summary=merge_summary,
        )
        removed_latency_frame_dirs = []
        if cleanup_latency_frames and len(merge_summary.get('unmatched_episodes', [])) == 0 and len(merge_summary.get('merged_videos', [])) > 0:
            removed_latency_frame_dirs = _cleanup_latency_frames(dynamic_log_paths)
            manifest['removed_latency_frame_dirs'] = removed_latency_frame_dirs
        elif cleanup_latency_frames:
            manifest['removed_latency_frame_dirs'] = []
            if len(merge_summary.get('unmatched_episodes', [])) > 0:
                manifest['cleanup_skipped_reason'] = 'unmatched_dynamic_episodes'
            else:
                manifest['cleanup_skipped_reason'] = 'no_merged_videos'

        manifest_path = eval_output_dir / 'dynamic_results_summary.json'
        dynamic_result_jsons = _write_dynamic_task_result_jsons(
            eval_output_dir=eval_output_dir,
            manifest=manifest,
        )
        manifest['dynamic_result_jsons'] = dynamic_result_jsons
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return {
            'eval_output_dir': str(eval_output_dir),
            'real_time_output_dir': str(output_dir),
            'merge_summary_path': merge_summary.get('summary_path'),
            'manifest_path': str(manifest_path),
            'dynamic_result_jsons': dynamic_result_jsons,
            'num_merged_videos': len(merge_summary.get('merged_videos', [])),
            'num_manifest_episodes': len(manifest.get('episodes', [])),
            'removed_latency_frame_dirs': removed_latency_frame_dirs,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-output-dir', required=True, type=Path)
    parser.add_argument('--dynamic-log-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--num-steps-wait', type=int, default=30)
    parser.add_argument('--fps', type=float, default=24.0)
    parser.add_argument('--min-latency-seconds', type=float, default=0.05)
    parser.add_argument('--cleanup-latency-frames', dest='cleanup_latency_frames', action='store_true', default=True)
    parser.add_argument('--keep-latency-frames', dest='cleanup_latency_frames', action='store_false')
    args = parser.parse_args()
    summary = finalize_dynamic_results(
        eval_output_dir=args.eval_output_dir,
        dynamic_log_dir=args.dynamic_log_dir,
        output_dir=args.output_dir,
        num_steps_wait=args.num_steps_wait,
        fps=args.fps,
        min_latency_seconds=args.min_latency_seconds,
        cleanup_latency_frames=args.cleanup_latency_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
