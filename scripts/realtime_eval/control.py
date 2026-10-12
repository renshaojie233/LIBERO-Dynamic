from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CONTROL_MODE_CHOICES = (
    "sync_hold",
    "async_queue",
    "rtc_queue",
    "vlash_future_state",
    "hold_latency",
    "naive_async_queue",
)

MODE_ALIASES = {
    "hold_latency": "sync_hold",
}


@dataclass(frozen=True)
class ObservationPacket:
    episode_id: str
    step_idx: int
    sim_time_s: float
    language: str
    obs: dict[str, Any]
    queue_len: int
    last_action: np.ndarray


@dataclass(frozen=True)
class ActionChunkPacket:
    episode_id: str
    chunk_id: int
    obs_step_idx: int
    submit_step_idx: int
    available_step_idx: int
    policy_elapsed_s: float
    actions: np.ndarray
    controller: str
    original_actions: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step_idx: int
    sim_time_s: float
    action_source: str
    queue_len_before: int
    queue_len_after: int
    underflow: bool
    chunk_id: int | None
    success: bool
    diagnostics_ok: bool | None


class ActionRuntime:
    """The only action bridge into the simulated world."""

    def __init__(self, action_dim: int, empty_action: str = "zero") -> None:
        if empty_action not in {"hold", "zero"}:
            raise ValueError(f"Unsupported empty_action: {empty_action}")
        self.action_dim = int(action_dim)
        self.empty_action = str(empty_action)
        self.zero_action = np.zeros((self.action_dim,), dtype=np.float32)
        self.last_action = np.array(self.zero_action, copy=True)
        self._queue: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.underflow_steps = 0
        self.enqueued_chunks = 0
        self.executed_actions = 0
        self.executed_queue_actions = 0
        self.executed_action_history: list[np.ndarray] = []

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    def enqueue(self, packet: ActionChunkPacket) -> int:
        actions = np.asarray(packet.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        original_actions = self._packet_original_actions(packet, len(actions))
        added = 0
        for action, original_action in zip(actions, original_actions, strict=False):
            self._queue.append(
                (
                    np.asarray(action[: self.action_dim], dtype=np.float32),
                    np.asarray(original_action[: self.action_dim], dtype=np.float32),
                    int(packet.chunk_id),
                )
            )
            added += 1
        self.enqueued_chunks += 1
        return added

    def _packet_original_actions(self, packet: ActionChunkPacket, fallback_len: int | None = None) -> np.ndarray:
        if packet.original_actions is None:
            original_actions = np.asarray(packet.actions, dtype=np.float32)
        else:
            original_actions = np.asarray(packet.original_actions, dtype=np.float32)
        if original_actions.ndim == 1:
            original_actions = original_actions.reshape(1, -1)
        if fallback_len is not None and len(original_actions) < int(fallback_len):
            raise RuntimeError(
                f"Original action chunk has {len(original_actions)} actions, shorter than processed chunk {fallback_len}."
            )
        return original_actions[:, : self.action_dim]

    def snapshot_actions(self, *, original: bool = False) -> np.ndarray:
        if not self._queue:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        if original:
            return np.asarray([original_action for _action, original_action, _chunk_id in self._queue], dtype=np.float32)
        return np.asarray([action for action, _original_action, _chunk_id in self._queue], dtype=np.float32)

    def snapshot_executed_actions(self, limit: int | None = None) -> np.ndarray:
        history = self.executed_action_history
        if limit is not None:
            history = history[-max(0, int(limit)) :]
        if not history:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        return np.asarray(history, dtype=np.float32)

    def merge_weighted(self, packet: ActionChunkPacket) -> dict[str, int]:
        """LeRobot async queue update with weighted overlap.

        LeRobot async inference treats each action in a returned chunk as tied
        to a future control timestep. Actions whose timesteps have already been
        executed are skipped. The remaining incoming actions replace the future
        queue; overlapping timesteps are merged with the official
        ``weighted_average`` aggregate function: ``0.3 * old + 0.7 * new``.
        """
        actions = np.asarray(packet.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        actions = actions[:, : self.action_dim]
        original_actions = self._packet_original_actions(packet, len(actions))
        # LeRobot timestamps only actions that are actually dequeued and sent.
        # An underflow hold advances simulation time, but it does not make a
        # returned policy action stale.
        drop_prefix = int(max(0, packet.metadata.get("executed_queue_actions_since_submit", 0)))
        dropped = min(drop_prefix, len(actions))
        usable = actions[dropped:]
        usable_original = original_actions[dropped:]
        overlap = min(len(self._queue), len(usable))
        new_queue: list[tuple[np.ndarray, np.ndarray, int]] = []
        for idx in range(overlap):
            old_action, old_original_action, old_chunk_id = self._queue[idx]
            merged = 0.3 * old_action + 0.7 * usable[idx]
            new_queue.append(
                (
                    np.asarray(merged, dtype=np.float32),
                    np.asarray(usable_original[idx], dtype=np.float32)
                    if packet.original_actions is not None
                    else old_original_action,
                    old_chunk_id,
                )
            )
        appended = 0
        for action, original_action in zip(usable[overlap:], usable_original[overlap:], strict=False):
            new_queue.append(
                (
                    np.asarray(action, dtype=np.float32),
                    np.asarray(original_action, dtype=np.float32),
                    int(packet.chunk_id),
                )
            )
            appended += 1
        self._queue = new_queue
        self.enqueued_chunks += 1
        return {"overlap": int(overlap), "appended": int(appended), "dropped_prefix": int(dropped)}

    def replace_with_delay_aware_merge(self, packet: ActionChunkPacket, *, drop_prefix: int) -> dict[str, int]:
        """LeRobot RTC ActionQueue replacement semantics.

        Official RTC ActionQueue.merge(..., cfg.enabled=True) clamps the real
        delay, discards the corresponding prefix from the returned chunk, and
        replaces the executable queue. The policy-side smoothness comes from
        RTC conditioning, not from a client-side weighted overlap.
        """
        actions = np.asarray(packet.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        actions = actions[:, : self.action_dim]
        original_actions = self._packet_original_actions(packet, len(actions))
        dropped = min(max(0, int(drop_prefix)), len(actions))
        usable = actions[dropped:]
        usable_original = original_actions[dropped:]
        new_queue = [
            (
                np.asarray(action, dtype=np.float32),
                np.asarray(original_action, dtype=np.float32),
                int(packet.chunk_id),
            )
            for action, original_action in zip(usable, usable_original, strict=False)
        ]
        self._queue = new_queue
        self.enqueued_chunks += 1
        return {"overlap": 0, "appended": int(len(usable)), "dropped_prefix": int(dropped)}

    def next_action(self) -> tuple[np.ndarray, np.ndarray, str, bool, int | None, int, int]:
        before = len(self._queue)
        if self._queue:
            action, original_action, chunk_id = self._queue.pop(0)
            self.executed_queue_actions += 1
            source = "queue"
            underflow = False
        else:
            self.underflow_steps += 1
            underflow = True
            chunk_id = None
            if self.empty_action == "zero":
                action = np.array(self.zero_action, copy=True)
                original_action = np.array(self.zero_action, copy=True)
                source = "underflow_zero_hold"
            else:
                action = np.array(self.last_action, copy=True)
                original_action = np.array(self.last_action, copy=True)
                source = "underflow_repeat_last"
        self.last_action = np.array(action, copy=True)
        self.executed_action_history.append(np.array(action, copy=True))
        self.executed_actions += 1
        after = len(self._queue)
        return action, original_action, source, underflow, chunk_id, before, after


def _numeric_samples(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    samples: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            samples.append(float(item))
        elif isinstance(item, dict):
            for key in (
                "latency_seconds",
                "inference_seconds",
                "elapsed_seconds",
                "policy_elapsed_s",
                "mean_policy_inference_seconds",
            ):
                if isinstance(item.get(key), (int, float)):
                    samples.append(float(item[key]))
                    break
    return [sample for sample in samples if sample >= 0.0]


def load_latency_samples(path: Path) -> list[float]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(data, list):
        samples = _numeric_samples(data)
    elif isinstance(data, dict):
        samples = []
        for key in ("samples", "latencies", "latency_samples", "inference_samples", "records"):
            samples = _numeric_samples(data.get(key))
            if samples:
                break
    else:
        samples = []
    if not samples:
        raise ValueError(f"No numeric latency samples found in {path}")
    return samples


@dataclass
class LatencySampler:
    source: str = "measured_wallclock"
    fixed_seconds: float | None = None
    first_seconds: float | None = None
    samples: list[float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.source not in {"measured_wallclock", "fixed", "empirical"}:
            raise ValueError(f"Unsupported latency source: {self.source}")
        if self.source == "fixed" and (self.fixed_seconds is None or float(self.fixed_seconds) < 0.0):
            raise ValueError("--calibrated-latency-seconds must be provided for --latency-source fixed")
        if self.first_seconds is not None and float(self.first_seconds) < 0.0:
            raise ValueError("--calibrated-first-inference-seconds must be non-negative")
        if self.source == "empirical" and not self.samples:
            raise ValueError("--latency-samples-json must provide samples for --latency-source empirical")
        self.rng = np.random.default_rng(int(self.seed))
        self.sampled_values: list[float] = []

    @property
    def decoupled_from_wallclock(self) -> bool:
        return self.source in {"fixed", "empirical"}

    def sample(self, wallclock_seconds: float | None = None) -> float | None:
        if self.source == "measured_wallclock":
            if wallclock_seconds is None:
                return None
            value = float(wallclock_seconds)
        elif self.source == "fixed":
            value = float(
                self.first_seconds
                if self.first_seconds is not None and not self.sampled_values
                else (self.fixed_seconds or 0.0)
            )
        else:
            assert self.samples is not None
            value = float(self.samples[int(self.rng.integers(0, len(self.samples)))])
        self.sampled_values.append(value)
        return value

    def summary(self) -> dict[str, Any]:
        values = self.sampled_values
        configured = self.samples or []
        return {
            "latency_source": self.source,
            "sim_latency_decoupled_from_wallclock": self.decoupled_from_wallclock,
            "calibrated_latency_seconds": None if self.fixed_seconds is None else float(self.fixed_seconds),
            "calibrated_first_inference_seconds": (
                None if self.first_seconds is None else float(self.first_seconds)
            ),
            "latency_sample_count": len(configured),
            "simulated_policy_latency_mean_s": 0.0 if not values else round(float(np.mean(values)), 6),
            "simulated_policy_latency_max_s": 0.0 if not values else round(float(np.max(values)), 6),
            "simulated_policy_latency_p90_s": 0.0 if not values else round(float(np.percentile(values, 90)), 6),
        }


class PolicyController:
    """Action producer interface.

    A controller may call a policy, wait, prefetch, queue, or predict future
    actions. It must not call env.step() or mutate the simulated world.
    """

    name = "base"

    def reset(self) -> None:
        pass

    def tick(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def summary(self) -> dict[str, Any]:
        return {}


class ChunkQueueControllerBase(PolicyController):
    """Shared implementation for chunk-based control strategies.

    Strategy files override only should_start_request(). The rest of the
    behavior, including policy calls, simulated latency, queueing, summaries,
    and cleanup, stays identical across sync / async / RTC variants.
    """

    def __init__(
        self,
        *,
        policy: Any,
        language: str,
        control_freq: int,
        mode: str,
        prefetch_threshold: int,
        action_chunk_size: int,
        chunk_size_threshold: float | None,
        latency_sampler: LatencySampler,
        vlash_visual_delay_steps: int = 1,
        vlash_measured_latency_seconds: float | None = None,
    ) -> None:
        self.policy = policy
        self.language = str(language)
        self.control_freq = int(control_freq)
        self.requested_mode = str(mode)
        self.mode = MODE_ALIASES.get(str(mode), str(mode))
        self.prefetch_threshold = int(prefetch_threshold)
        self.action_chunk_size = int(max(1, action_chunk_size))
        self.chunk_size_threshold = None if chunk_size_threshold is None else float(chunk_size_threshold)
        self.latency_sampler = latency_sampler
        self.vlash_visual_delay_steps = int(vlash_visual_delay_steps)
        self.vlash_measured_latency_seconds = (
            None
            if vlash_measured_latency_seconds is None
            else float(vlash_measured_latency_seconds)
        )
        self.name = self.mode
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Future | None = None
        self.submit_step_idx = 0
        self.submit_executed_queue_actions = 0
        self.obs_step_idx = 0
        self.chunk_id = 0
        self.policy_latencies: list[float] = []
        self.prefetch_count = 0
        self.dropped_late_packets = 0
        self.pending_chunk: np.ndarray | None = None
        self.pending_original_chunk: np.ndarray | None = None
        self.pending_wall_latency_s: float | None = None
        self.pending_sim_latency_s: float | None = None
        self.pending_available_step_idx: int | None = None
        self.pending_leftover_actions: np.ndarray | None = None
        self.merge_stats: list[dict[str, int | str]] = []
        self.wallclock_wait_blocked_s = 0.0
        self.observation_history: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.future = None
        self.submit_step_idx = 0
        self.submit_executed_queue_actions = 0
        self.obs_step_idx = 0
        self.chunk_id = 0
        self.policy_latencies = []
        self.prefetch_count = 0
        self.dropped_late_packets = 0
        self.pending_chunk = None
        self.pending_original_chunk = None
        self.pending_wall_latency_s = None
        self.pending_sim_latency_s = None
        self.pending_available_step_idx = None
        self.pending_leftover_actions = None
        self.merge_stats = []
        self.wallclock_wait_blocked_s = 0.0
        self.observation_history = []

    def _submit(self, packet: ObservationPacket, actions: ActionRuntime) -> None:
        if self.future is not None or self.pending_chunk is not None:
            return
        self.submit_step_idx = int(packet.step_idx)
        self.submit_executed_queue_actions = int(actions.executed_queue_actions)
        self.obs_step_idx = int(packet.step_idx)
        self.pending_leftover_actions = actions.snapshot_actions(original=True)
        if self.latency_sampler.decoupled_from_wallclock:
            self.pending_sim_latency_s = float(self.latency_sampler.sample() or 0.0)
            delay_steps = int(np.ceil(max(0.0, self.pending_sim_latency_s) * float(self.control_freq)))
            self.pending_available_step_idx = self.submit_step_idx + max(0, delay_steps)
        else:
            self.pending_sim_latency_s = None
            self.pending_available_step_idx = None
        obs_snapshot = self.snapshot_request_observation(packet, actions)

        def infer() -> tuple[np.ndarray, np.ndarray | None, float]:
            started = time.perf_counter()
            if self.uses_rtc_conditioning() and bool(
                getattr(self.policy, "supports_rtc_conditioning", False)
            ) and hasattr(self.policy, "get_action_chunk_rtc"):
                result = self.policy.get_action_chunk_rtc(
                    obs_snapshot,
                    self.language,
                    prev_chunk_left_over=self.pending_leftover_actions,
                    inference_delay_steps=(
                        None
                        if self.pending_available_step_idx is None
                        else int(self.pending_available_step_idx) - int(self.submit_step_idx)
                    ),
                )
            elif hasattr(self.policy, "get_action_chunk"):
                result = self.policy.get_action_chunk(obs_snapshot, self.language)
            else:
                result = np.asarray([self.policy.get_action(obs_snapshot, self.language)], dtype=np.float32)
            processed, original = self._normalize_policy_result(result)
            return processed, original, time.perf_counter() - started

        self.future = self.executor.submit(infer)
        self.prefetch_count += 1

    def snapshot_request_observation(
        self,
        packet: ObservationPacket,
        actions: ActionRuntime,
    ) -> dict[str, Any]:
        obs_snapshot = {
            key: np.array(value, copy=True)
            for key, value in packet.obs.items()
            if key
            in {
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            }
        }
        required_obs = int(getattr(self.policy, "required_observation_history_frames", 1))
        if required_obs > 1:
            obs_snapshot["__observation_history"] = [
                {key: np.array(value, copy=True) for key, value in history_obs.items()}
                for history_obs in self.observation_history[-required_obs:]
            ]
        required_actions = int(getattr(self.policy, "required_action_history_steps", 0))
        if required_actions > 0:
            obs_snapshot["__executed_action_history"] = actions.snapshot_executed_actions(required_actions)
        return obs_snapshot

    def _normalize_policy_result(self, result: Any) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(result, dict):
            actions = np.asarray(result["actions"], dtype=np.float32)
            original_actions = result.get("original_actions")
            original = None if original_actions is None else np.asarray(original_actions, dtype=np.float32)
        else:
            actions = np.asarray(result, dtype=np.float32)
            original = None
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if original is not None and original.ndim == 1:
            original = original.reshape(1, -1)
        return actions, original

    def uses_rtc_conditioning(self) -> bool:
        return False

    def should_start_request(self, actions: ActionRuntime) -> bool:
        raise NotImplementedError

    def official_threshold_ready(self, actions: ActionRuntime) -> bool:
        configured_threshold = self.prefetch_threshold
        if self.chunk_size_threshold is not None:
            configured_threshold = int(np.ceil(float(self.action_chunk_size) * float(self.chunk_size_threshold)))
        return actions.queue_len <= configured_threshold

    def latency_budget_steps(self) -> int:
        """Number of queued actions needed to cover one policy request."""
        latency_s = 0.0
        if self.latency_sampler.source == "fixed":
            latency_s = float(self.latency_sampler.fixed_seconds or 0.0)
        elif self.latency_sampler.source == "empirical" and self.latency_sampler.samples:
            latency_s = float(np.percentile(self.latency_sampler.samples, 90))
        elif self.policy_latencies:
            latency_s = float(np.percentile(self.policy_latencies, 90))
        return int(np.ceil(max(0.0, latency_s) * float(self.control_freq)))

    def effective_prefetch_threshold_steps(self) -> int:
        configured_threshold = self.prefetch_threshold
        if self.chunk_size_threshold is not None:
            configured_threshold = int(np.ceil(float(self.action_chunk_size) * float(self.chunk_size_threshold)))
        return configured_threshold

    def _should_submit(self, actions: ActionRuntime) -> bool:
        if self.future is not None or self.pending_chunk is not None:
            return False
        return self.should_start_request(actions)

    def merge_ready_packet(self, actions: ActionRuntime, packet: ActionChunkPacket) -> dict[str, int | str]:
        added = actions.enqueue(packet)
        return {"strategy": "append", "overlap": 0, "appended": int(added), "dropped_prefix": 0}

    def _collect_future(
        self,
        packet: ObservationPacket,
        actions: ActionRuntime,
        *,
        force_wait: bool = False,
    ) -> None:
        if self.future is None:
            return
        should_block_for_calibrated_latency = (
            self.latency_sampler.decoupled_from_wallclock
            and self.pending_available_step_idx is not None
            and int(packet.step_idx) >= int(self.pending_available_step_idx)
            and not self.future.done()
        )
        should_block = bool(force_wait or should_block_for_calibrated_latency)
        if not self.future.done() and not should_block:
            return
        blocked_started = time.perf_counter() if should_block and not self.future.done() else None
        chunk, original_chunk, elapsed = self.future.result()
        if blocked_started is not None:
            self.wallclock_wait_blocked_s += max(0.0, time.perf_counter() - blocked_started)
        self.future = None
        self.pending_chunk = chunk
        self.pending_original_chunk = original_chunk
        self.pending_wall_latency_s = float(elapsed)
        self.policy_latencies.append(float(elapsed))
        if not self.latency_sampler.decoupled_from_wallclock:
            self.pending_sim_latency_s = float(self.latency_sampler.sample(float(elapsed)) or 0.0)
            delay_steps = int(np.ceil(max(0.0, self.pending_sim_latency_s) * float(self.control_freq)))
            self.pending_available_step_idx = int(self.submit_step_idx) + max(0, delay_steps)

    def _enqueue_if_available(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        if self.pending_chunk is None or self.pending_available_step_idx is None:
            return []
        if int(packet.step_idx) < int(self.pending_available_step_idx):
            return []
        chunk = self.pending_chunk
        original_chunk = self.pending_original_chunk
        wall_elapsed = float(self.pending_wall_latency_s or 0.0)
        sim_latency = float(self.pending_sim_latency_s or 0.0)
        delay_steps = int(max(0, int(self.pending_available_step_idx) - int(self.submit_step_idx)))
        executed_queue_actions_since_submit = int(
            max(0, int(actions.executed_queue_actions) - int(self.submit_executed_queue_actions))
        )
        self.chunk_id += 1
        ready = ActionChunkPacket(
            episode_id=packet.episode_id,
            chunk_id=self.chunk_id,
            obs_step_idx=int(self.obs_step_idx),
            submit_step_idx=int(self.submit_step_idx),
            available_step_idx=int(self.pending_available_step_idx),
            policy_elapsed_s=wall_elapsed,
            actions=chunk,
            controller=self.name,
            original_actions=original_chunk,
            metadata={
                "delay_steps": int(delay_steps),
                "executed_queue_actions_since_submit": int(executed_queue_actions_since_submit),
                "sim_latency_s": round(sim_latency, 6),
                "wallclock_policy_elapsed_s": round(wall_elapsed, 6),
                "latency_source": self.latency_sampler.source,
                "prev_chunk_left_over": 0
                if self.pending_leftover_actions is None
                else int(len(self.pending_leftover_actions)),
                "rtc_conditioning_supported": bool(getattr(self.policy, "supports_rtc_conditioning", False)),
                "rtc_conditioning_requested": bool(self.uses_rtc_conditioning()),
                "rtc_conditioning_used": bool(
                    self.uses_rtc_conditioning()
                    and getattr(self.policy, "supports_rtc_conditioning", False)
                    and hasattr(self.policy, "get_action_chunk_rtc")
                ),
            },
        )
        merge_stat = self.merge_ready_packet(actions, ready)
        ready.metadata.update({f"merge_{key}": value for key, value in merge_stat.items()})
        self.merge_stats.append({"chunk_id": int(self.chunk_id), **merge_stat})
        self.pending_chunk = None
        self.pending_original_chunk = None
        self.pending_wall_latency_s = None
        self.pending_sim_latency_s = None
        self.pending_available_step_idx = None
        self.pending_leftover_actions = None
        return [ready]

    def record_observation(self, packet: ObservationPacket) -> None:
        history_obs = {
            key: np.array(value, copy=True)
            for key, value in packet.obs.items()
            if key
            in {
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            }
        }
        self.observation_history.append(history_obs)
        required_obs = max(1, int(getattr(self.policy, "required_observation_history_frames", 1)))
        if len(self.observation_history) > required_obs:
            self.observation_history = self.observation_history[-required_obs:]

    def tick(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        self.record_observation(packet)
        ready_packets: list[ActionChunkPacket] = []
        self._collect_future(packet, actions)
        ready_packets.extend(self._enqueue_if_available(packet, actions))
        if self._should_submit(actions):
            self._submit(packet, actions)
            available_now = (
                self.latency_sampler.decoupled_from_wallclock
                and self.pending_available_step_idx is not None
                and int(self.pending_available_step_idx) <= int(packet.step_idx)
            )
            if available_now:
                self._collect_future(packet, actions, force_wait=True)
                ready_packets.extend(self._enqueue_if_available(packet, actions))
        return ready_packets

    def close(self) -> None:
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)

    def summary(self) -> dict[str, Any]:
        lat = self.policy_latencies
        rtc_requested = bool(self.uses_rtc_conditioning())
        rtc_supported = bool(getattr(self.policy, "supports_rtc_conditioning", False))
        return {
            "controller": self.name,
            "requested_controller": self.requested_mode,
            "controller_contract": (
                "Controller produces action chunks only and never mutates the simulated world. "
                "The runtime applies synchronous latency as continuous physics time; async and RTC "
                "queue availability remains indexed by control ticks."
            ),
            "policy_requests": int(self.prefetch_count),
            "policy_latency_mean_s": 0.0 if not lat else round(float(np.mean(lat)), 6),
            "policy_latency_max_s": 0.0 if not lat else round(float(np.max(lat)), 6),
            "policy_latency_p90_s": 0.0 if not lat else round(float(np.percentile(lat, 90)), 6),
            "wallclock_wait_blocked_s": round(float(self.wallclock_wait_blocked_s), 6),
            "dropped_late_packets": int(self.dropped_late_packets),
            "action_chunk_size": int(self.action_chunk_size),
            "chunk_size_threshold": self.chunk_size_threshold,
            "prefetch_threshold_steps": int(self.prefetch_threshold),
            "latency_budget_steps": int(self.latency_budget_steps()),
            "effective_prefetch_threshold_steps": int(self.effective_prefetch_threshold_steps()),
            "rtc_conditioning_requested": rtc_requested,
            "rtc_conditioning_supported": rtc_supported,
            "rtc_conditioning_used": bool(
                rtc_requested and rtc_supported and hasattr(self.policy, "get_action_chunk_rtc")
            ),
            "merge_stats_tail": self.merge_stats[-8:],
            **self.latency_sampler.summary(),
        }


class ReservedController(PolicyController):
    def __init__(self, name: str, message: str) -> None:
        self.name = str(name)
        self.message = str(message)

    def tick(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        del packet, actions
        raise NotImplementedError(self.message)


def build_policy_controller(
    *,
    policy: Any,
    language: str,
    control_freq: int,
    mode: str,
    prefetch_threshold: int,
    action_chunk_size: int,
    chunk_size_threshold: float | None,
    latency_sampler: LatencySampler,
    vlash_visual_delay_steps: int = 1,
    vlash_measured_latency_seconds: float | None = None,
) -> PolicyController:
    canonical_mode = MODE_ALIASES.get(str(mode), str(mode))
    if canonical_mode == "sync_hold":
        from .strategies.sync_hold import SyncHoldController

        return SyncHoldController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
    if canonical_mode == "async_queue":
        from .strategies.async_queue import AsyncQueueController

        return AsyncQueueController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
    if canonical_mode == "naive_async_queue":
        from .strategies.naive_async_queue import NaiveAsyncQueueController

        return NaiveAsyncQueueController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
    if canonical_mode == "rtc_queue":
        from .strategies.rtc_queue import RtcQueueController

        return RtcQueueController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
    if canonical_mode == "vlash_future_state":
        from .strategies.vlash_future_state import VLASHFutureStateController

        return VLASHFutureStateController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
            vlash_measured_latency_seconds=vlash_measured_latency_seconds,
        )
    raise ValueError(f"Unsupported control mode: {mode}")
