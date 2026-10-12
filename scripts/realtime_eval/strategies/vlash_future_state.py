from __future__ import annotations

from typing import Any

import numpy as np

from ..control import ActionChunkPacket, ActionRuntime, ChunkQueueControllerBase, ObservationPacket


class VLASHFutureStateController(ChunkQueueControllerBase):
    """VLASH's official LIBERO delayed-vision/current-state protocol.

    The released LIBERO evaluator emulates asynchronous inference by pairing
    images from ``d`` control steps ago with the robot state at the action
    chunk's actual start. The first chunk uses the current image because no
    delayed frame exists yet. Chunks are executed intact without LeRobot queue
    blending or prefix dropping.
    """

    _IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
    _STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

    def __init__(self, *args: Any, vlash_visual_delay_steps: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, vlash_visual_delay_steps=vlash_visual_delay_steps, **kwargs)
        self.pending_visual_obs_step_idx = 0
        self.pending_state_obs_step_idx = 0
        self.pending_bootstrap_no_delay = False
        self.vlash_latency_steps_from_measurement: int | None = None
        if self.vlash_visual_delay_steps < 0:
            raise ValueError("vlash_visual_delay_steps must be non-negative")
        if self.chunk_size_threshold is not None:
            raise ValueError("vlash_future_state does not use --chunk-size-threshold")
        if not self.latency_sampler.decoupled_from_wallclock:
            raise ValueError(
                "vlash_future_state requires fixed zero simulated latency; "
                "the official LIBERO delay is represented by delayed images, not a second queue delay"
            )
        if float(self.latency_sampler.fixed_seconds or 0.0) != 0.0:
            raise ValueError(
                "vlash_future_state requires --calibrated-latency-seconds 0; "
                "the measured latency is represented once as visual observation staleness"
            )
        if self.vlash_measured_latency_seconds is not None:
            if not np.isfinite(self.vlash_measured_latency_seconds) or self.vlash_measured_latency_seconds < 0.0:
                raise ValueError("vlash_measured_latency_seconds must be finite and non-negative")
            self.vlash_latency_steps_from_measurement = int(
                np.ceil(self.vlash_measured_latency_seconds * float(self.control_freq) - 1e-12)
            )
            if self.vlash_visual_delay_steps != self.vlash_latency_steps_from_measurement:
                raise ValueError(
                    "VLASH measured-latency mapping mismatch: "
                    f"ceil({self.vlash_measured_latency_seconds} * {self.control_freq})="
                    f"{self.vlash_latency_steps_from_measurement}, but --vlash-visual-delay-steps="
                    f"{self.vlash_visual_delay_steps}"
                )

    def reset(self) -> None:
        super().reset()
        self.pending_visual_obs_step_idx = 0
        self.pending_state_obs_step_idx = 0
        self.pending_bootstrap_no_delay = False

    def should_start_request(self, actions: ActionRuntime) -> bool:
        return actions.queue_len == 0

    def record_observation(self, packet: ObservationPacket) -> None:
        history_obs = {
            key: np.array(value, copy=True)
            for key, value in packet.obs.items()
            if key in self._IMAGE_KEYS + self._STATE_KEYS
        }
        self.observation_history.append(history_obs)
        keep = max(1, self.vlash_visual_delay_steps + 1)
        if len(self.observation_history) > keep:
            self.observation_history = self.observation_history[-keep:]

    def snapshot_request_observation(
        self,
        packet: ObservationPacket,
        actions: ActionRuntime,
    ) -> dict[str, Any]:
        del actions
        current = {
            key: np.array(packet.obs[key], copy=True)
            for key in self._IMAGE_KEYS + self._STATE_KEYS
        }
        if len(self.observation_history) > self.vlash_visual_delay_steps:
            delayed = self.observation_history[-(self.vlash_visual_delay_steps + 1)]
            for key in self._IMAGE_KEYS:
                current[key] = np.array(delayed[key], copy=True)
            self.pending_visual_obs_step_idx = int(packet.step_idx) - int(self.vlash_visual_delay_steps)
            self.pending_bootstrap_no_delay = False
        else:
            self.pending_visual_obs_step_idx = int(packet.step_idx)
            self.pending_bootstrap_no_delay = True
        self.pending_state_obs_step_idx = int(packet.step_idx)
        return current

    def _enqueue_if_available(
        self,
        packet: ObservationPacket,
        actions: ActionRuntime,
    ) -> list[ActionChunkPacket]:
        ready = super()._enqueue_if_available(packet, actions)
        for chunk in ready:
            chunk.metadata.update(
                {
                    "vlash_protocol": "official_libero_delayed_visual_current_state",
                    "vlash_visual_observation_step_idx": int(self.pending_visual_obs_step_idx),
                    "vlash_state_observation_step_idx": int(self.pending_state_obs_step_idx),
                    "vlash_effective_visual_delay_steps": int(
                        self.pending_state_obs_step_idx - self.pending_visual_obs_step_idx
                    ),
                    "vlash_bootstrap_no_delay": bool(self.pending_bootstrap_no_delay),
                    "vlash_measured_latency_seconds": self.vlash_measured_latency_seconds,
                    "vlash_latency_steps_from_measurement": self.vlash_latency_steps_from_measurement,
                    "vlash_latency_representation": "visual_observation_staleness_only",
                }
            )
        return ready

    def merge_ready_packet(
        self,
        actions: ActionRuntime,
        packet: ActionChunkPacket,
    ) -> dict[str, int | str]:
        if actions.queue_len != 0:
            raise RuntimeError("VLASH chunk became ready before the previous chunk was exhausted")
        added = actions.enqueue(packet)
        return {
            "strategy": "vlash_intact_chunk_replace",
            "overlap": 0,
            "appended": int(added),
            "dropped_prefix": 0,
        }

    def summary(self) -> dict[str, Any]:
        represented_latency_s = float(self.vlash_visual_delay_steps) / float(self.control_freq)
        quantization_error_s = (
            None
            if self.vlash_measured_latency_seconds is None
            else represented_latency_s - self.vlash_measured_latency_seconds
        )
        return {
            **super().summary(),
            "vlash_protocol": "official_libero_delayed_visual_current_state",
            "vlash_visual_delay_steps": int(self.vlash_visual_delay_steps),
            "vlash_measured_latency_seconds": self.vlash_measured_latency_seconds,
            "vlash_latency_steps_from_measurement": self.vlash_latency_steps_from_measurement,
            "vlash_represented_latency_seconds": represented_latency_s,
            "vlash_latency_quantization_error_seconds": quantization_error_s,
            "vlash_latency_mapping": "ceil(measured_latency_seconds * control_frequency_hz)",
            "vlash_latency_representation": "visual_observation_staleness_only",
            "vlash_state_at_chunk_start": "current_robot_state",
            "vlash_bootstrap_visual_delay_steps": 0,
            "vlash_chunk_merge": "intact_replace_no_weighting",
            "vlash_simulated_inference_queue_delay_steps": 0,
            "vlash_latency_double_counted": False,
        }
