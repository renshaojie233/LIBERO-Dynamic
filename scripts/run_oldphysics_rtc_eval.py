#!/usr/bin/env python3
"""Run old-physics LIBERO-Dynamic eval with Pi0.5/OpenPI RTC action queueing.

RTC here means asynchronous action-chunk queue semantics:

* one OpenPI websocket inference request returns a chunk of robot actions;
* while the next chunk is being inferred, the simulator continues to step;
* dynamic objects keep moving every simulator step;
* the robot executes already queued actions during inference;
* if the queue underflows, the configured fallback action is executed and
  counted in the manifest;
* this entrypoint does not use the ordinary freeze-during-inference latency
  path.

The dynamic scene route is always the reviewed old preview-physics profile.
"""

from __future__ import annotations

import argparse

from eval_dynamic_libero_style import main


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def main_entry() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rtc-prefetch-threshold",
        type=int,
        default=5,
        help="Submit the next async chunk when queued actions are at or below this count.",
    )
    parser.add_argument(
        "--rtc-empty-action",
        choices=("hold", "zero"),
        default="hold",
        help="Fallback action if the queue underflows before the next chunk is ready.",
    )
    known, rest = parser.parse_known_args()
    forbidden = {"--policy", "--latency-mode"}
    if any(arg in forbidden or any(arg.startswith(f"{name}=") for name in forbidden) for arg in rest):
        raise SystemExit("run_oldphysics_rtc_eval.py fixes --policy=openpi_ws_rtc and --latency-mode=disabled")

    forwarded = [
        *rest,
        "--policy",
        "openpi_ws_rtc",
        "--latency-mode",
        "disabled",
    ]
    if not _has_arg(forwarded, "--rtc-prefetch-threshold"):
        forwarded.extend(["--rtc-prefetch-threshold", str(int(known.rtc_prefetch_threshold))])
    if not _has_arg(forwarded, "--rtc-empty-action"):
        forwarded.extend(["--rtc-empty-action", str(known.rtc_empty_action)])
    print(
        "[rtc] using OpenPI async action queue: policy=openpi_ws_rtc, "
        f"prefetch_threshold={known.rtc_prefetch_threshold}, empty_action={known.rtc_empty_action}",
        flush=True,
    )
    return main(forwarded, reviewed_old_physics=True)


if __name__ == "__main__":
    raise SystemExit(main_entry())
