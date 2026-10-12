from __future__ import annotations

"""LeRobot official asynchronous action-queue loop.

Async means policy inference happens in a background thread while the simulator
continues to execute queued robot actions. The environment is the clock: every
iteration advances exactly one control step.

This is a compact reference for LeRobot's async client behavior:

* request a new chunk when ``queue_len / action_chunk_size <= threshold``;
* skip actions whose timesteps are already stale when the chunk arrives;
* replace the future queue with the incoming future actions;
* merge overlapping timesteps with ``0.3 * old + 0.7 * new``.
"""

import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np


def _copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if hasattr(value, "shape") else value
        for key, value in obs.items()
    }


def _chunk_actions(chunk: Any, action_dim: int) -> list[np.ndarray]:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    return [np.asarray(action[:action_dim], dtype=np.float32) for action in actions]


def _official_async_merge(
    queue: deque[np.ndarray],
    chunk: list[np.ndarray],
    *,
    executed_actions_since_submit: int,
) -> dict[str, int]:
    """LeRobot RobotClient-style future queue replacement.

    LeRobot associates chunk actions with future control timesteps. Actions
    older than the latest executed timestep are skipped. The incoming future
    queue replaces the old future queue; overlapping timesteps are aggregated
    with weighted_average = 0.3 * old + 0.7 * new.
    """

    dropped = min(max(0, int(executed_actions_since_submit)), len(chunk))
    usable = chunk[dropped:]
    overlap = min(len(queue), len(usable))
    merged: deque[np.ndarray] = deque()
    for idx in range(overlap):
        merged.append(np.asarray(0.3 * queue[idx] + 0.7 * usable[idx], dtype=np.float32))
    for action in usable[overlap:]:
        merged.append(np.asarray(action, dtype=np.float32))
    queue.clear()
    queue.extend(merged)
    return {"dropped_prefix": int(dropped), "overlap": int(overlap), "appended": int(max(0, len(usable) - overlap))}


def run_async_queue_loop(
    *,
    env: Any,
    policy: Any,
    initial_obs: dict[str, Any],
    language: str,
    zero_action: np.ndarray,
    max_steps: int = 600,
    control_freq: int = 20,
    chunk_size_threshold: float = 0.5,
    action_chunk_size: int = 10,
    action_dim: int | None = None,
    empty_action: str = "zero",
    stop_on_success: bool = True,
    fixed_latency_s: float | None = None,
) -> dict[str, Any]:
    """Run one episode with LeRobot official async queue semantics.

    If fixed_latency_s is None, action availability is based on measured policy
    wall-clock time. If fixed_latency_s is provided, that calibrated delay is
    used as simulator latency; a late policy result blocks wall-clock time as
    soon as the calibrated availability step is reached. The simulator never
    advances past that step while waiting, even when queued actions remain.
    """

    if empty_action not in {"hold", "zero"}:
        raise ValueError("empty_action must be 'hold' or 'zero'")

    obs = initial_obs
    action_dim = int(action_dim or len(zero_action))
    zero_action = np.asarray(zero_action[:action_dim], dtype=np.float32)
    last_action = np.array(zero_action, copy=True)
    queue: deque[np.ndarray] = deque()
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future | None = None
    future_submit_step = 0
    future_submit_executed_actions = 0
    future_available_step: int | None = None
    pending_chunk: list[np.ndarray] | None = None
    policy_requests = 0
    underflow_steps = 0
    wallclock_wait_s = 0.0
    inference_latencies: list[float] = []
    merge_stats: list[dict[str, int]] = []
    executed_queue_actions = 0
    success = False

    def submit(step_idx: int, current_obs: dict[str, Any]) -> None:
        nonlocal future, future_submit_step, future_submit_executed_actions, future_available_step, policy_requests
        if future is not None or pending_chunk is not None:
            return
        future_submit_step = int(step_idx)
        future_submit_executed_actions = int(executed_queue_actions)
        if fixed_latency_s is not None:
            delay_steps = int(math.ceil(max(0.0, float(fixed_latency_s)) * float(control_freq)))
            future_available_step = future_submit_step + delay_steps
        else:
            future_available_step = None
        obs_snapshot = _copy_obs(current_obs)

        def infer() -> tuple[Any, float]:
            started = time.perf_counter()
            chunk = policy.get_action_chunk(obs_snapshot, language)
            return chunk, time.perf_counter() - started

        future = executor.submit(infer)
        policy_requests += 1

    def collect(step_idx: int) -> None:
        nonlocal future, future_available_step, pending_chunk, wallclock_wait_s
        if future is None:
            return
        calibrated_due = fixed_latency_s is not None and future_available_step is not None and step_idx >= future_available_step
        should_wait_without_stepping = calibrated_due and not future.done()
        if not future.done() and not should_wait_without_stepping:
            return
        started_wait = time.perf_counter() if should_wait_without_stepping else None
        chunk, elapsed = future.result()
        if started_wait is not None:
            wallclock_wait_s += time.perf_counter() - started_wait
        inference_latencies.append(float(elapsed))
        if fixed_latency_s is None:
            delay_steps = int(math.ceil(max(0.0, float(elapsed)) * float(control_freq)))
            future_available_step = future_submit_step + delay_steps
        pending_chunk = _chunk_actions(chunk, action_dim)
        future = None

    def enqueue_if_available(step_idx: int) -> None:
        nonlocal pending_chunk, future_available_step
        if pending_chunk is None or future_available_step is None:
            return
        if step_idx < future_available_step:
            return
        merge_stats.append(
            _official_async_merge(
                queue,
                pending_chunk,
                executed_actions_since_submit=(
                    int(executed_queue_actions) - int(future_submit_executed_actions)
                ),
            )
        )
        pending_chunk = None
        future_available_step = None

    try:
        submit(0, obs)
        for step_idx in range(int(max_steps)):
            collect(step_idx)
            enqueue_if_available(step_idx)
            ready_for_next_chunk = float(len(queue)) / float(max(1, int(action_chunk_size))) <= float(
                chunk_size_threshold
            )
            if future is None and pending_chunk is None and ready_for_next_chunk:
                submit(step_idx, obs)

            if queue:
                action = queue.popleft()
                executed_queue_actions += 1
            else:
                underflow_steps += 1
                action = zero_action if empty_action == "zero" else last_action

            obs, _reward, done, _info = env.step(action)
            last_action = np.array(action, copy=True)
            success = bool(done) or bool(env.check_success())
            if success and stop_on_success:
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return {
        "success": bool(success),
        "policy_requests": policy_requests,
        "underflow_steps": underflow_steps,
        "remaining_queue": len(queue),
        "wallclock_wait_s": wallclock_wait_s,
        "inference_latencies": inference_latencies,
        "merge_stats": merge_stats,
        "chunk_size_threshold": float(chunk_size_threshold),
    }
