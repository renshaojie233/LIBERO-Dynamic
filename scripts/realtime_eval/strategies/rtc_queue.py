from __future__ import annotations

from ..control import ActionChunkPacket, ActionRuntime, ChunkQueueControllerBase


class RtcQueueController(ChunkQueueControllerBase):
    """LeRobot RTC action-queue control.

    The simulator keeps executing queued robot actions while the next chunk is
    inferred. At submission time we pass the official RTC conditioning inputs:
    the unconsumed original policy chunk and the measured/simulated inference
    delay. When the new chunk arrives, the first ``delay`` actions are skipped
    and the remaining future queue is replaced, matching LeRobot ActionQueue.

    If a policy exposes get_action_chunk_rtc(..., prev_chunk_left_over,
    inference_delay_steps), the controller passes official RTC conditioning
    arguments. Plain OpenPI websocket policies do not currently expose that
    method, so this controller still records the missing support in manifests.
    """

    def should_start_request(self, actions: ActionRuntime) -> bool:
        return self.official_threshold_ready(actions)

    def uses_rtc_conditioning(self) -> bool:
        return True

    def merge_ready_packet(self, actions: ActionRuntime, packet: ActionChunkPacket) -> dict[str, int | str]:
        delay_steps = int((packet.metadata or {}).get("delay_steps", 0))
        stats = actions.replace_with_delay_aware_merge(packet, drop_prefix=delay_steps)
        return {"strategy": "lerobot_rtc_action_queue_replace", **stats}
