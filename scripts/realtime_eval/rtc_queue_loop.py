from __future__ import annotations

"""LeRobot official RTC action chunk loop.

RTC here means a realtime control loop with a policy action queue:

* the simulator advances one control step per iteration;
* a policy request returns a chunk of future robot actions;
* while the next chunk is inferred, the robot keeps executing queued actions;
* dynamic objects move every env.step(...);
* if the queue underflows, a configured fallback action is used and counted.

This is intentionally a small reference loop. It mirrors LeRobot ActionQueue
semantics: pass original ``prev_chunk_left_over`` and inference delay to the
RTC-enabled policy, then drop the delay prefix and replace the executable
future queue. Unlike async_queue_loop, RTC does not do client-side weighted
overlap.
"""

import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PendingChunk:
    submit_step: int
    available_step: int | None
    prev_chunk_left_over: np.ndarray
    future: Future


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


def _normalize_policy_result(result: Any, action_dim: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if isinstance(result, dict):
        processed = np.asarray(result["actions"], dtype=np.float32)
        original = np.asarray(result.get("original_actions", processed), dtype=np.float32)
    else:
        processed = np.asarray(result, dtype=np.float32)
        original = processed
    if processed.ndim == 1:
        processed = processed.reshape(1, -1)
    if original.ndim == 1:
        original = original.reshape(1, -1)
    processed_list = [np.asarray(action[:action_dim], dtype=np.float32) for action in processed]
    original_list = [np.asarray(action[:action_dim], dtype=np.float32) for action in original]
    if len(original_list) < len(processed_list):
        raise RuntimeError("RTC original_actions must be at least as long as actions")
    return processed_list, original_list[: len(processed_list)]


def replace_queue_after_delay(
    *,
    queue: deque[np.ndarray],
    original_queue: deque[np.ndarray],
    processed_chunk: list[np.ndarray],
    original_chunk: list[np.ndarray],
    real_delay_steps: int,
) -> int:
    """LeRobot ActionQueue.merge(..., cfg.enabled=True) behavior.

    The policy-side RTC module uses the original leftover actions. The robot
    executes processed actions. On merge, LeRobot drops the first real_delay
    actions from both arrays and replaces both queues.
    """

    late = min(max(0, int(real_delay_steps)), len(processed_chunk), len(original_chunk))
    usable = processed_chunk[late:]
    usable_original = original_chunk[late:]
    queue.clear()
    original_queue.clear()
    queue.extend(usable)
    original_queue.extend(usable_original)
    return len(usable)


def run_rtc_queue_loop(
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
    """Run one LeRobot RTC episode.

    Unlike the synchronous loop, this loop never advances extra simulator steps
    just because policy inference is slow. The world advances only at the
    control tick. The queue decides what robot action is available at that tick.
    """

    if empty_action not in {"hold", "zero"}:
        raise ValueError("empty_action must be 'hold' or 'zero'")

    obs = initial_obs
    action_dim = int(action_dim or len(zero_action))
    zero_action = np.asarray(zero_action[:action_dim], dtype=np.float32)
    last_action = np.array(zero_action, copy=True)
    queue: deque[np.ndarray] = deque()
    original_queue: deque[np.ndarray] = deque()
    executor = ThreadPoolExecutor(max_workers=1)
    pending: PendingChunk | None = None
    policy_requests = 0
    merged_actions = 0
    underflow_steps = 0
    replace_stats: list[dict[str, int]] = []
    inference_latencies: list[float] = []
    wallclock_wait_s = 0.0
    success = False

    def submit(step_idx: int, current_obs: dict[str, Any]) -> None:
        nonlocal pending, policy_requests
        if pending is not None:
            return
        available_step = None
        if fixed_latency_s is not None:
            delay_steps = int(math.ceil(max(0.0, float(fixed_latency_s)) * float(control_freq)))
            available_step = int(step_idx) + delay_steps
        else:
            delay_steps = 0
        obs_snapshot = _copy_obs(current_obs)
        prev_left_over = np.asarray(list(original_queue), dtype=np.float32)

        def infer() -> tuple[Any, float]:
            started = time.perf_counter()
            if hasattr(policy, "get_action_chunk_rtc"):
                chunk = policy.get_action_chunk_rtc(
                    obs_snapshot,
                    language,
                    prev_chunk_left_over=prev_left_over,
                    inference_delay_steps=int(delay_steps),
                )
            else:
                chunk = policy.get_action_chunk(obs_snapshot, language)
            return chunk, time.perf_counter() - started

        pending = PendingChunk(
            submit_step=int(step_idx),
            available_step=available_step,
            prev_chunk_left_over=prev_left_over,
            future=executor.submit(infer),
        )
        policy_requests += 1

    def collect(step_idx: int) -> None:
        nonlocal pending, merged_actions, wallclock_wait_s
        if pending is None:
            return
        calibrated_due = (
            fixed_latency_s is not None
            and pending.available_step is not None
            and int(step_idx) >= int(pending.available_step)
        )
        if not pending.future.done() and not calibrated_due:
            return
        blocked_started = time.perf_counter() if not pending.future.done() else None
        chunk, elapsed = pending.future.result()
        if blocked_started is not None:
            wallclock_wait_s += time.perf_counter() - blocked_started
        inference_latencies.append(float(elapsed))
        available_step = pending.available_step
        if available_step is None:
            delay_steps = int(math.ceil(max(0.0, float(elapsed)) * float(control_freq)))
            available_step = pending.submit_step + delay_steps
        else:
            delay_steps = max(0, int(available_step) - int(pending.submit_step))
        if int(step_idx) < int(available_step):
            return
        processed_chunk, original_chunk = _normalize_policy_result(chunk, action_dim)
        added = replace_queue_after_delay(
            queue=queue,
            original_queue=original_queue,
            processed_chunk=processed_chunk,
            original_chunk=original_chunk,
            real_delay_steps=int(delay_steps),
        )
        merged_actions += added
        replace_stats.append({"delay_steps": int(delay_steps), "appended": int(added)})
        pending = None

    try:
        submit(0, obs)
        for step_idx in range(int(max_steps)):
            collect(step_idx)
            ready_for_next_chunk = float(len(queue)) / float(max(1, int(action_chunk_size))) <= float(
                chunk_size_threshold
            )
            if pending is None and ready_for_next_chunk:
                submit(step_idx, obs)

            if queue:
                action = queue.popleft()
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
        "merged_actions": merged_actions,
        "underflow_steps": underflow_steps,
        "remaining_queue": len(queue),
        "inference_latencies": inference_latencies,
        "wallclock_wait_s": wallclock_wait_s,
        "replace_stats": replace_stats,
        "chunk_size_threshold": float(chunk_size_threshold),
    }
