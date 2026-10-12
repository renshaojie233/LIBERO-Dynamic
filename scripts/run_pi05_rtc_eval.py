#!/usr/bin/env python3
"""Pi0.5 RTC-style realtime eval entrypoint.

This uses delay-aware chunk replacement. Already committed actions during the
inference delay are frozen; the returned chunk updates the remaining future
queue. If the policy exposes RTC conditioning, the controller also passes
prev_chunk_left_over and inference_delay_steps to it.
"""

from __future__ import annotations

import sys

from realtime_eval.runner import main


FIXED_ARGS = {
    "--reviewed-old-physics": None,
    "--control-mode": "rtc_queue",
    "--prefetch-threshold": "10",
    "--empty-action": "hold",
    "--require-rtc-conditioning": None,
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
