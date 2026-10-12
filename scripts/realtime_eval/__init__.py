"""Realtime-style LIBERO-Dynamic runtime v2.

The runtime keeps the simulated world advancing on a fixed simulation clock.
Policy controllers are external action producers; they never call env.step().
"""

