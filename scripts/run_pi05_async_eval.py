#!/usr/bin/env python3
"""Pi0.5 LeRobot-style async eval entrypoint.

The next chunk is requested shortly before the queue runs out. When it arrives,
overlapping future actions are merged with a weighted average, following the
standard async action-chunk pattern instead of append-only stale queuing.
"""

from __future__ import annotations

import sys

from realtime_eval.runner import main


FIXED_ARGS = {
    "--reviewed-old-physics": None,
    "--control-mode": "async_queue",
    "--prefetch-threshold": "2",
    "--chunk-size-threshold": "0.5",
    "--empty-action": "hold",
}


def main_entry() -> int:
    argv = list(sys.argv[1:])
    for flag, value in reversed(list(FIXED_ARGS.items())):
        if flag not in argv:
            if value is None:
                argv.insert(0, flag)
            else:
                argv[0:0] = [flag, value]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_entry())
