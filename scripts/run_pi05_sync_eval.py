#!/usr/bin/env python3
"""Synchronous realtime-v2 evaluation entrypoint.

Policy inference blocks action production. During its exact continuous
latency, MuJoCo physics and the dynamic scene advance while robot joints are
held. No extra 20 Hz zero action is inserted for inference latency.
"""

from __future__ import annotations

import sys

from realtime_eval.runner import main


FIXED_ARGS = {
    "--reviewed-old-physics": None,
    "--control-mode": "sync_hold",
    "--prefetch-threshold": "0",
    "--empty-action": "zero",
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
