from __future__ import annotations

from ..control import ActionRuntime, ChunkQueueControllerBase


class NaiveAsyncQueueController(ChunkQueueControllerBase):
    """Append-only async baseline.

    This is intentionally not the official async controller. It is kept for
    ablations and for interpreting older runs where new chunks were appended
    behind old queued actions.
    """

    def should_start_request(self, actions: ActionRuntime) -> bool:
        return actions.queue_len <= self.prefetch_threshold
