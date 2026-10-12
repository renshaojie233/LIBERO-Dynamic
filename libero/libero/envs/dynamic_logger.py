from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _locked_file(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class DynamicRunLogger:
    def __init__(self, output_path: str | Path, run_id: str | None = None):
        self.output_path = Path(output_path)
        self.lock_path = self.output_path.with_suffix(self.output_path.suffix + ".lock")
        self.run_id = run_id or self.output_path.stem

    def _update(self, update_fn) -> None:
        with _locked_file(self.lock_path):
            payload = _read_json(self.output_path)
            if not payload:
                payload = {
                    "benchmark": "LIBERO-dynamic",
                    "run_id": self.run_id,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "episodes": [],
                }
            payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            update_fn(payload)
            _atomic_write_json(self.output_path, payload)

    @staticmethod
    def _episode_key(suite_name: str, task_id: int, episode_index: int) -> str:
        return f"{suite_name}::task{int(task_id)}::episode{int(episode_index)}"

    def upsert_episode(
        self,
        *,
        suite_name: str,
        task_id: int,
        episode_index: int,
        planned: dict[str, Any],
        status: str = "running",
    ) -> None:
        episode_key = self._episode_key(suite_name, task_id, episode_index)

        def update(payload: dict[str, Any]) -> None:
            episodes = payload.setdefault("episodes", [])
            for ep in episodes:
                if ep.get("episode_key") == episode_key:
                    ep["planned"] = planned
                    ep["status"] = status
                    ep["events"] = []
                    ep["latency_trace"] = []
                    ep.pop("finished_at", None)
                    ep["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return

            episodes.append(
                {
                    "episode_key": episode_key,
                    "suite": suite_name,
                    "task_id": int(task_id),
                    "episode_index": int(episode_index),
                    "status": status,
                    "planned": planned,
                    "events": [],
                    "latency_trace": [],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        self._update(update)

    def append_event(
        self,
        *,
        suite_name: str,
        task_id: int,
        episode_index: int,
        event: dict[str, Any],
    ) -> None:
        episode_key = self._episode_key(suite_name, task_id, episode_index)

        def update(payload: dict[str, Any]) -> None:
            for ep in payload.setdefault("episodes", []):
                if ep.get("episode_key") == episode_key:
                    ep.setdefault("events", []).append(event)
                    ep["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return
            raise KeyError(f"Episode not found for dynamic event: {episode_key}")

        self._update(update)

    def append_latency(
        self,
        *,
        suite_name: str,
        task_id: int,
        episode_index: int,
        step_idx: int,
        latency_seconds: float,
    ) -> None:
        episode_key = self._episode_key(suite_name, task_id, episode_index)

        def update(payload: dict[str, Any]) -> None:
            for ep in payload.setdefault("episodes", []):
                if ep.get("episode_key") == episode_key:
                    ep.setdefault("latency_trace", []).append(
                        {
                            "step_idx": int(step_idx),
                            "latency_seconds": float(latency_seconds),
                        }
                    )
                    ep["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return
            raise KeyError(f"Episode not found for dynamic latency: {episode_key}")

        self._update(update)

    def finalize_episode(
        self,
        *,
        suite_name: str,
        task_id: int,
        episode_index: int,
        status: str,
    ) -> None:
        episode_key = self._episode_key(suite_name, task_id, episode_index)

        def update(payload: dict[str, Any]) -> None:
            for ep in payload.setdefault("episodes", []):
                if ep.get("episode_key") == episode_key:
                    ep["status"] = status
                    ep["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    ep["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return
            raise KeyError(f"Episode not found for finalize: {episode_key}")

        self._update(update)
