from __future__ import annotations

from ..control import ActionChunkPacket, ActionRuntime, ChunkQueueControllerBase, ObservationPacket


class SyncHoldController(ChunkQueueControllerBase):
    """Synchronous baseline.

    Start a policy request only after the action queue is empty and wait for it
    before returning. The runtime advances the dynamic world by the exact
    inference duration using physics substeps while holding the robot. It does
    not turn inference latency into zero-action control ticks.
    """

    def should_start_request(self, actions: ActionRuntime) -> bool:
        return actions.queue_len <= 0

    def tick(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        self.record_observation(packet)
        if not self._should_submit(actions):
            return []

        self._submit(packet, actions)
        self._collect_future(packet, actions, force_wait=True)

        sim_latency = float(self.pending_sim_latency_s or 0.0)
        equivalent_control_steps = sim_latency * float(self.control_freq)

        # Synchronous inference completes before this control action executes.
        # The runtime applies sim_latency as continuous physics time, so queue
        # availability must not be delayed by a second, quantized mechanism.
        self.pending_available_step_idx = int(packet.step_idx)
        ready = self._enqueue_if_available(packet, actions)
        for item in ready:
            item.metadata.update(
                {
                    "delay_steps": 0,
                    "queue_delay_steps": 0,
                    "latency_equivalent_control_steps": round(equivalent_control_steps, 6),
                    "latency_application": "continuous_physics",
                    "physics_time_quantized_to_control_steps": False,
                }
            )
        return ready
