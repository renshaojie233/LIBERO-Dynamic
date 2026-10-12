from __future__ import annotations

from ..control import ActionChunkPacket, ActionRuntime, ChunkQueueControllerBase


class AsyncQueueController(ChunkQueueControllerBase):
    """LeRobot official asynchronous action chunking.

    A new observation is sent when ``queue_len / action_chunk_size`` falls below
    the configured chunk-size threshold. Returned future actions replace the
    queue; overlapping timesteps use LeRobot's default weighted_average:
    ``0.3 * old + 0.7 * new``.
    """

    def should_start_request(self, actions: ActionRuntime) -> bool:
        return self.official_threshold_ready(actions)

    def merge_ready_packet(self, actions: ActionRuntime, packet: ActionChunkPacket) -> dict[str, int | str]:
        stats = actions.merge_weighted(packet)
        return {"strategy": "lerobot_async_weighted_average", **stats}
