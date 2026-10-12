"""Compatibility imports; new code should use realtime_eval.control."""

from .control import LatencySampler, load_latency_samples

__all__ = ["LatencySampler", "load_latency_samples"]
