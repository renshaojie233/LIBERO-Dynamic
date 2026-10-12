#!/usr/bin/env python3
"""Run the formal realtime-v2 evaluator with the reviewed physics profile.

This is the benchmark-facing synchronous entrypoint. It uses continuous
MuJoCo-time inference holds, time-faithful video sampling, and the same
reviewed physical-carrier setup as the model-specific realtime examples.
"""

from __future__ import annotations

from run_pi05_sync_eval import main_entry


if __name__ == "__main__":
    raise SystemExit(main_entry())
