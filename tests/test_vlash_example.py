from __future__ import annotations

from pathlib import Path
import sys

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from evaluate_vlash import measured_latency_to_steps  # noqa: E402


@pytest.mark.parametrize(
    ("seconds", "frequency_hz", "expected"),
    [
        (0.0, 20, 0),
        (0.05, 20, 1),
        (0.0500001, 20, 2),
        (0.10398730050292215, 20, 3),
    ],
)
def test_measured_latency_to_steps(seconds: float, frequency_hz: int, expected: int) -> None:
    assert measured_latency_to_steps(seconds, frequency_hz) == expected


@pytest.mark.parametrize("seconds", [-0.01, float("inf"), float("nan")])
def test_measured_latency_to_steps_rejects_invalid_latency(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        measured_latency_to_steps(seconds, 20)


def test_measured_latency_to_steps_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        measured_latency_to_steps(0.1, 0)
