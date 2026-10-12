#!/usr/bin/env python3
"""Shared process management for the model-specific evaluation examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
CONFIGS = REPO_ROOT / "libero" / "libero" / "dynamic_configs"


def add_common_arguments(parser: argparse.ArgumentParser, *, default_port: int) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument(
        "--server-pythonpath",
        action="append",
        type=Path,
        default=[],
        help="Additional import path for the model-server environment; repeat as needed.",
    )
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="Physical CUDA/EGL device index.")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--level", choices=("l1", "l2", "l3"), default="l1")
    parser.add_argument("--trajectory", type=int, choices=range(5), default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument(
        "--item-ids-file",
        type=Path,
        default=None,
        help="Optional JSON list of exact evaluation item IDs for a stratified run.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a normal evaluation after task success; disable for fixed-length timing audits.",
    )
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--calibrated-latency-seconds", type=float, default=0.0)
    parser.add_argument(
        "--model-input-video-size",
        type=int,
        default=128,
        help="Saved square size per policy view; this does not change the model observation resolution.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=60.0,
        help="Saved simulation-time video rate. Default: 60; use 100 for short timing diagnostics.",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=600.0)


def pythonpath(*paths: Path) -> str:
    values = [str(path.expanduser().resolve()) for path in paths if path is not None]
    current = os.environ.get("PYTHONPATH")
    if current:
        values.append(current)
    return os.pathsep.join(values)


def base_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
            "LIBERO_CONFIG_PATH": str(REPO_ROOT / "libero" / "libero" / "configs"),
            "LIBERO_DYNAMIC_AUTO_FINALIZE": "false",
            "PYTHONPATH": pythonpath(
                REPO_ROOT,
                SCRIPTS,
                args.openpi_root / "packages" / "openpi-client" / "src",
            ),
        }
    )
    return env


def server_pythonpath(args: argparse.Namespace, *paths: Path) -> str:
    return pythonpath(REPO_ROOT, SCRIPTS, *paths, *args.server_pythonpath)


def port_is_open(port: int) -> bool:
    try:
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{int(port)}",
            open_timeout=0.25,
            close_timeout=0.25,
        ):
            return True
    except Exception:
        return False


def wait_for_server(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with status {process.returncode}")
        if port_is_open(port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"policy server did not open port {port} within {timeout:.0f}s")


def _require_executable(value: str, label: str) -> str:
    path = Path(value).expanduser()
    if path.parent != Path("."):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {value}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"{label} is not executable: {value}")
        return str(path)
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"{label} is not on PATH: {value}")
    return resolved


def validate_common_arguments(args: argparse.Namespace) -> None:
    args.eval_python = _require_executable(str(args.eval_python), "--eval-python")
    args.server_python = _require_executable(str(args.server_python), "--server-python")
    args.openpi_root = args.openpi_root.expanduser().resolve()
    openpi_client_src = args.openpi_root / "packages" / "openpi-client" / "src"
    if not openpi_client_src.exists():
        raise FileNotFoundError(
            "openpi client source directory was not found. Expected "
            f"{openpi_client_src}. Pass --openpi-root pointing at a cloned openpi repository."
        )
    for extra in args.server_pythonpath:
        if not extra.expanduser().exists():
            raise FileNotFoundError(f"--server-pythonpath does not exist: {extra}")
    if float(args.video_fps) <= 0.0:
        raise ValueError("--video-fps must be greater than 0")
    if int(args.model_input_video_size) <= 0:
        raise ValueError("--model-input-video-size must be greater than 0")
    if int(args.num_workers) <= 0:
        raise ValueError("--num-workers must be greater than 0")
    if args.item_ids_file is not None and not args.item_ids_file.expanduser().is_file():
        raise FileNotFoundError(f"--item-ids-file does not exist: {args.item_ids_file}")


def require_existing_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_example(
    args: argparse.Namespace,
    *,
    server_command: list[str],
    server_environment: dict[str, str] | None = None,
    evaluator_script: Path | None = None,
    evaluator_extra_args: list[str] | None = None,
) -> int:
    validate_common_arguments(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if port_is_open(args.port):
        raise RuntimeError(f"port {args.port} is already in use")

    eval_env = base_environment(args)
    server_env = eval_env.copy()
    if server_environment:
        server_env.update(server_environment)
    server_log_path = args.output_dir / "policy_server.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            server_command,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            wait_for_server(server, args.port, args.startup_timeout_seconds)
            evaluator_script = evaluator_script or (SCRIPTS / "run_pi05_sync_eval.py")
            levels = [args.level]
            trajectories = [int(args.trajectory)]
            if args.item_ids_file is not None:
                item_ids = json.loads(args.item_ids_file.expanduser().read_text())
                if not isinstance(item_ids, list) or not item_ids or not all(isinstance(x, str) for x in item_ids):
                    raise ValueError("--item-ids-file must contain a non-empty JSON list of strings")
                levels = sorted({item_id.split("__")[-2] for item_id in item_ids})
                trajectories = sorted({int(item_id.rsplit("traj", 1)[1]) for item_id in item_ids})

            command = [
                str(args.eval_python),
                str(evaluator_script),
                "--config-dir",
                str(CONFIGS),
                "--output-dir",
                str(args.output_dir / "evaluation"),
                "--levels",
                *levels,
                "--traj-indices",
                *(str(value) for value in trajectories),
                "--num-workers",
                str(args.num_workers),
                "--policy",
                "openpi_ws",
                "--openpi-host",
                "127.0.0.1",
                "--openpi-port",
                str(args.port),
                "--policy-server-count",
                "1",
                "--no-per-worker-policy-endpoint",
                "--latency-source",
                "fixed",
                "--calibrated-latency-seconds",
                str(args.calibrated_latency_seconds),
                "--replan-steps",
                str(args.replan_steps),
                "--max-steps",
                str(args.max_steps),
                "--stop-on-success" if bool(args.stop_on_success) else "--no-stop-on-success",
                "--save-videos",
                "--no-save-review-video",
                "--save-model-input-videos",
                "--model-input-video-layout",
                "dual",
                "--model-input-video-size",
                str(args.model_input_video_size),
                "--video-fps",
                str(args.video_fps),
                "--render-every",
                "1",
                "--record-latency-frames",
                "--camera-size",
                "256",
            ]
            if evaluator_extra_args:
                command.extend(evaluator_extra_args)
            if args.item_ids_file is None:
                command.extend(["--start-index", str(args.start_index), "--limit", str(args.task_count)])
            else:
                command.extend(["--item-ids-file", str(args.item_ids_file.expanduser().resolve())])
            print("Running:", " ".join(command), flush=True)
            return subprocess.run(command, env=eval_env, check=False).returncode
        finally:
            stop_process_group(server)
            print(f"Policy server log: {server_log_path}", flush=True)
