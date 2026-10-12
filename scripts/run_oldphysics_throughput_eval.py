#!/usr/bin/env python3
"""Run old-physics LIBERO-Dynamic eval with calibrated fixed latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_dynamic_libero_style import main


def calibrated_latency_seconds(path: Path, statistic: str) -> float:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    key = f"{statistic}_seconds"
    if key not in summary:
        raise KeyError(f"{path} does not contain summary.{key}")
    value = float(summary[key])
    if value <= 0.0:
        raise ValueError(f"Invalid calibrated latency {value} from {path}")
    return value


def main_entry() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument(
        "--latency-stat",
        choices=("mean", "median", "p90", "p95", "p99"),
        default="p90",
        help="Calibration summary statistic to use for fixed simulation latency.",
    )
    known, rest = parser.parse_known_args()
    latency = calibrated_latency_seconds(known.calibration_json, known.latency_stat)
    forwarded = [
        *rest,
        "--latency-mode",
        "fixed",
        "--fixed-inference-seconds",
        f"{latency:.6f}",
    ]
    print(
        f"[throughput] using fixed latency {latency:.6f}s "
        f"from {known.calibration_json} ({known.latency_stat})",
        flush=True,
    )
    return main(forwarded, reviewed_old_physics=True)


if __name__ == "__main__":
    raise SystemExit(main_entry())
