#!/usr/bin/env python3
"""Run reviewed old-physics LIBERO-Dynamic with realtime runtime v2."""

from __future__ import annotations

import sys

from realtime_eval.runner import main


def main_entry() -> int:
    argv = list(sys.argv[1:])
    if "--reviewed-old-physics" not in argv:
        argv.insert(0, "--reviewed-old-physics")
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_entry())

