#!/usr/bin/env python3
"""Fail-fast motion-family checks shared by generation and preview code.

The reviewed benchmark stores a few legacy duplicate motion blocks (for
example ``motion`` and ``latency``).  Those blocks are useful for compatibility
but historically made it possible for a later merge to silently replace a
linear L1 trajectory with an L2 polynomial spline, or vice versa.  This module
classifies every concrete variant before execution and rejects cross-family
mixing instead of silently accepting it.
"""

from __future__ import annotations

from typing import Any, Iterable


EXPECTED_FAMILY_BY_LEVEL = {
    "l1": "linear",
    "l2": "curvilinear",
    "l3": "zigzag",
}


def _level_name(level: str) -> str:
    value = str(level).strip().lower()
    if value not in EXPECTED_FAMILY_BY_LEVEL:
        raise ValueError(f"Unknown dynamic-motion level: {level!r}")
    return value


def _xyz_waypoints_are_collinear(points: Any, tolerance: float = 1e-7) -> bool:
    """Return whether 3-D waypoints lie on one line, without numpy."""
    if not isinstance(points, list):
        return False
    vectors: list[tuple[float, float, float]] = []
    origin: tuple[float, float, float] | None = None
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        current = (float(point[0]), float(point[1]), float(point[2]))
        if origin is None:
            origin = current
            continue
        vector = (
            current[0] - origin[0],
            current[1] - origin[1],
            current[2] - origin[2],
        )
        if sum(component * component for component in vector) > tolerance * tolerance:
            vectors.append(vector)
    if len(vectors) <= 1:
        return True
    reference = vectors[0]
    reference_norm = sum(component * component for component in reference) ** 0.5
    for vector in vectors[1:]:
        vector_norm = sum(component * component for component in vector) ** 0.5
        cross = (
            reference[1] * vector[2] - reference[2] * vector[1],
            reference[2] * vector[0] - reference[0] * vector[2],
            reference[0] * vector[1] - reference[1] * vector[0],
        )
        cross_norm = sum(component * component for component in cross) ** 0.5
        if cross_norm > tolerance * max(reference_norm * vector_norm, 1.0):
            return False
    return True


def motion_family(motion: dict[str, Any], *, level_hint: str | None = None) -> str:
    """Classify a concrete motion block as linear, curvilinear, or zigzag."""
    if not isinstance(motion, dict) or not motion:
        return "unknown"
    declared = " ".join(
        str(motion.get(key, "")).strip().lower()
        for key in ("type", "trajectory", "editor_motion_type")
    )
    if "polynomial_spline" in declared or "smoothstep" in declared:
        return "curvilinear"
    if "conveyor_pingpong" in declared or "conveyor_linear" in declared:
        return "linear"
    if "pingpong" in declared and "zigzag" not in declared:
        return "linear"
    if "zigzag" in declared:
        # L1 translation fixtures use the legacy type name
        # body_translation_zigzag_waypoints, but their reviewed geometry is a
        # single out-and-back straight segment.  Geometry disambiguates that
        # compatibility encoding from a true L3 multi-corner zigzag.
        if (
            str(level_hint or "").lower() == "l1"
            and _xyz_waypoints_are_collinear(motion.get("waypoints_xyz"))
        ):
            return "linear"
        return "zigzag"
    if motion.get("waypoints_xyz"):
        return "linear" if _xyz_waypoints_are_collinear(motion["waypoints_xyz"]) else "zigzag"
    return "unknown"


def iter_variant_motion_blocks(variant: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield every motion replica carried by one reviewed variant."""
    for key in ("motion", "latency"):
        value = variant.get(key)
        if isinstance(value, dict) and value:
            yield f"variant.{key}", value
    for index, target in enumerate(variant.get("target_objects") or []):
        if not isinstance(target, dict):
            continue
        for key in ("motion", "latency"):
            value = target.get(key)
            if isinstance(value, dict) and value:
                yield f"target_objects[{index}].{key}", value


def validate_variant_motion_family(
    level: str,
    variant: dict[str, Any],
    *,
    context: str,
) -> None:
    """Reject a variant whose motion replicas do not match its L1/L2/L3 family."""
    level_name = _level_name(level)
    expected = EXPECTED_FAMILY_BY_LEVEL[level_name]
    blocks = list(iter_variant_motion_blocks(variant))
    if not blocks:
        raise ValueError(f"{context}: no concrete motion block")
    errors = []
    for path, motion in blocks:
        actual = motion_family(motion, level_hint=level_name)
        if actual != expected:
            errors.append(
                f"{path}={actual} "
                f"(type={motion.get('type')!r}, trajectory={motion.get('trajectory')!r}, "
                f"editor_motion_type={motion.get('editor_motion_type')!r})"
            )
    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"{context}: expected {expected} motion for {level_name.upper()}, "
            f"but found cross-family configuration: {details}"
        )

