from __future__ import annotations

"""Synchronous policy loop.

This file is intentionally small. It shows the core timing semantics without
LIBERO setup, video, diagnostics, or config plumbing.

Expected adapters:

* env.notify_inference_latency(seconds) -> None
* env.step(action) -> (obs, reward, done, info)
* env.check_success() -> bool
* policy.get_action_chunk(obs, language) -> array[chunk_len, action_dim]

Synchronous means policy inference blocks the control loop. The environment
advances the dynamic world by the exact latency in MuJoCo physics substeps when
the next real control action executes.
"""

import time
from typing import Any

import numpy as np


def _copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy numpy-like observation values before sending them to policy code."""
    return {
        key: np.array(value, copy=True) if hasattr(value, "shape") else value
        for key, value in obs.items()
    }


def _chunk_actions(chunk: Any, action_dim: int) -> list[np.ndarray]:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    return [np.asarray(action[:action_dim], dtype=np.float32) for action in actions]


def run_sync_loop(
    *,
    env: Any,
    policy: Any,
    initial_obs: dict[str, Any],
    language: str,
    zero_action: np.ndarray,
    max_steps: int = 600,
    control_freq: int = 20,
    action_dim: int | None = None,
    stop_on_success: bool = True,
    measured_latency: bool = True,
    fixed_latency_s: float | None = None,
) -> dict[str, Any]:
    """Run one episode with blocking policy inference.

    In this baseline, there is no overlap between inference and robot action
    execution. Latency is passed to the environment as continuous physics time;
    it is not rounded to a number of ``control_freq`` action slots.
    """

    obs = initial_obs
    action_dim = int(action_dim or len(zero_action))
    zero_action = np.asarray(zero_action[:action_dim], dtype=np.float32)
    step_idx = 0
    inference_latencies: list[float] = []
    simulated_latency_elapsed_s = 0.0
    policy_requests = 0
    success = False

    while step_idx < int(max_steps):
        policy_requests += 1
        started = time.perf_counter()
        chunk = policy.get_action_chunk(_copy_obs(obs), language)
        elapsed = time.perf_counter() - started
        inference_latencies.append(float(elapsed))

        if fixed_latency_s is not None:
            latency_s = float(fixed_latency_s)
        elif measured_latency:
            latency_s = float(elapsed)
        else:
            latency_s = 0.0

        notify_latency = getattr(env, "notify_inference_latency", None)
        if not callable(notify_latency):
            raise RuntimeError("sync env must provide notify_inference_latency(seconds)")
        notify_latency(max(0.0, latency_s))
        simulated_latency_elapsed_s += max(0.0, latency_s)

        for action in _chunk_actions(chunk, action_dim):
            if step_idx >= int(max_steps):
                break
            obs, _reward, done, _info = env.step(action)
            step_idx += 1
            success = bool(done) or bool(env.check_success())
            if success and stop_on_success:
                return {
                    "success": True,
                    "steps": step_idx,
                    "policy_requests": policy_requests,
                    "latency_hold_steps": 0,
                    "simulated_latency_elapsed_s": simulated_latency_elapsed_s,
                    "inference_latencies": inference_latencies,
                }

    return {
        "success": bool(success),
        "steps": step_idx,
        "policy_requests": policy_requests,
        "latency_hold_steps": 0,
        "simulated_latency_elapsed_s": simulated_latency_elapsed_s,
        "inference_latencies": inference_latencies,
    }
