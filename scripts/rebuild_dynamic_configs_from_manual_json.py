#!/usr/bin/env python3
"""Rebuild LIBERO-Dynamic YAML configs from manual JSON without trajectory shrink.

The manual JSON is the source of truth for object placements and waypoint
geometry. This script preserves the current YAML structure and physics fields,
but rewrites target motion waypoints from the manual annotations and removes
automatic shrink metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def as_xyz_offsets(points_xy: list[list[float]]) -> list[list[float]]:
    if not points_xy:
        return [[0.0, 0.0, 0.0]]
    base = points_xy[0]
    return [[float(x) - float(base[0]), float(y) - float(base[1]), 0.0] for x, y in points_xy]


def clean_motion(motion: dict[str, Any], plan_target: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(motion or {})
    waypoints_xy = plan_target.get("waypoints_xy") or []
    out["waypoints_xyz"] = as_xyz_offsets(waypoints_xy)
    out["editor_motion_type"] = plan_target.get("motion_type", out.get("editor_motion_type", out.get("type")))
    motion_type = str(plan_target.get("motion_type") or out.get("type") or "zigzag_waypoints")
    if motion_type == "conveyor_pingpong":
        out["type"] = "zigzag_waypoints"
        out["trajectory"] = "zigzag_waypoints"
    else:
        out["type"] = motion_type
        out["trajectory"] = motion_type
    if plan_target.get("alpha_mps") is not None:
        out["alpha_mps"] = float(plan_target["alpha_mps"])
    out["loop"] = True
    out.setdefault("start_step", 30)
    out.setdefault("end_step", 250)
    # The final benchmark follows the annotated polyline at constant segment
    # speed and does not decelerate or pause at waypoint corners.
    out.pop("waypoint_accel_mps2", None)
    out.pop("waypoint_stop_seconds", None)
    out["planner"] = "manual_json_noshrink"
    out.pop("shrink_notes", None)
    out.pop("scale_xy", None)
    return out


def plan_offsets(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    offsets: dict[str, dict[str, Any]] = {}
    for name, obj in (plan.get("objects") or {}).items():
        xy = obj.get("xy")
        original_xy = obj.get("original_xy")
        if not xy or not original_xy:
            continue
        yaw = float(obj.get("yaw_rad", obj.get("original_yaw_rad", 0.0)) or 0.0)
        original_yaw = float(obj.get("original_yaw_rad", yaw) or 0.0)
        offsets[str(name)] = {
            "offset_xyz": [
                float(xy[0]) - float(original_xy[0]),
                float(xy[1]) - float(original_xy[1]),
                0.0,
            ],
            "yaw_delta_rad": yaw - original_yaw,
        }
    return offsets


def propagate_coupled_offsets(
    offsets: dict[str, dict[str, Any]],
    task_cfg: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return reviewed initial object offsets exactly as stored in JSON."""
    out = copy.deepcopy(offsets)
    return out


def tune_target_physics(target: dict[str, Any]) -> None:
    name = str(target.get("target_object_name") or "").lower()
    target["friction"] = [1200.0, 80.0, 20.0]
    target["solref"] = "0.0005 1"
    target["solimp"] = "0.995 0.999 0.0001"
    target["platform_tracking_gain"] = 8.0
    target["platform_tracking_max_m"] = 0.04
    target["platform_velocity_max_mps"] = 0.22
    target["contact_all_target_geoms"] = True
    # A tiny overlap guarantees the hidden conveyor establishes contact before
    # the table support is excluded.  Keeping it at 1 mm avoids visible lift.
    target["surface_epsilon_m"] = 0.0015
    target["margin"] = max(float(target.get("margin", 0.0001) or 0.0001), 0.001)
    target["collision_penetration_threshold_m"] = 0.0005
    target.setdefault("bottom_contact_margin_m", 0.0015)
    if any(k in name for k in ("basket", "milk", "orange_juice", "tomato_sauce", "chocolate_pudding", "cookies")):
        # These grocery-package / basket assets rely on LIBERO's normal settle
        # against the table support. Keep the original support contact and
        # place the hidden conveyor just above that support plane, otherwise
        # the platform can sit below the scene or lift the object too high.
        target["exclude_static_support_contact"] = False
        target["platform_use_support_plane_z"] = True
        target["surface_epsilon_m"] = 0.001
    for key in (
        "contact_proxy_enabled",
        "contact_proxy_size_x_m",
        "contact_proxy_size_y_m",
        "contact_proxy_size_z_m",
    ):
        target.pop(key, None)
    if any(k in name for k in ("butter", "cream_cheese")):
        target["stability_proxy_enabled"] = True
        target["stability_proxy_size"] = [0.055, 0.04, 0.001]
        target["stability_proxy_bottom_epsilon_m"] = 0.0
        # Thin box assets can miss the conveyor contact if the patch is flush,
        # but a larger lift becomes visible.  This value was validated on all
        # 75 butter / cream-cheese variants.
        target["surface_epsilon_m"] = 0.0008
    elif any(k in name for k in ("black_book", "akita_black_bowl", "basket", "plate", "cookies", "ramekin", "mug")):
        target["stability_proxy_enabled"] = True
        target["stability_proxy_size"] = "auto"
        target["stability_proxy_thickness_m"] = 0.0015
        target["stability_proxy_bottom_epsilon_m"] = 0.00015
    else:
        for key in (
            "stability_proxy_enabled",
            "stability_proxy_size",
            "stability_proxy_thickness_m",
            "stability_proxy_bottom_epsilon_m",
        ):
            target.pop(key, None)
    if "moka_pot" in name:
        target["stabilizer_rim_enabled"] = True
        target["stabilizer_rim_height_m"] = 0.018
        target["stabilizer_rim_thickness_m"] = 0.003
    else:
        target.pop("stabilizer_rim_enabled", None)
        target.pop("stabilizer_rim_height_m", None)
        target.pop("stabilizer_rim_thickness_m", None)


def strip_shrink_metadata(value: Any) -> Any:
    """Remove legacy trajectory shrink fields from any nested YAML subtree."""
    if isinstance(value, dict):
        return {
            key: strip_shrink_metadata(child)
            for key, child in value.items()
            if key not in {"shrink_notes", "scale_xy"}
        }
    if isinstance(value, list):
        return [strip_shrink_metadata(child) for child in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()

    rows = json.loads(args.source_json.read_text(encoding="utf-8")).get("rows", [])
    rows_by_variant = {str(row["variant_id"]): row for row in rows}

    args.output_config.mkdir(parents=True, exist_ok=True)
    summary = {"updated_variants": 0, "updated_targets": 0, "missing_rows": []}
    for level in ("l1", "l2", "l3"):
        cfg_path = args.input_config / f"libero_dynamic_manual_{level}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        carrier = cfg.setdefault("defaults", {}).setdefault("carrier", {})
        carrier["friction"] = [1200.0, 80.0, 20.0]
        carrier["solref"] = "0.0005 1"
        carrier["solimp"] = "0.995 0.999 0.0001"
        carrier["contact_all_target_geoms"] = True
        carrier["surface_epsilon_m"] = 0.0015
        carrier["margin"] = 0.001
        carrier["collision_penetration_threshold_m"] = 0.0005
        for suite_tasks in (cfg.get("tasks") or {}).values():
            for task_cfg in (suite_tasks or {}).values():
                for variant in task_cfg.get("activation_variants") or []:
                    variant_id = str(variant.get("variant_id") or "")
                    row = rows_by_variant.get(variant_id)
                    if row is None:
                        summary["missing_rows"].append(variant_id)
                        continue
                    plan = row.get("plan") or {}
                    plan_targets = {
                        str(t.get("target_object")): t
                        for t in plan.get("target_objects") or []
                        if t.get("target_object")
                    }
                    adjustments = variant.setdefault("initial_scene_adjustments", {})
                    adjustments["enabled"] = True
                    adjustments["settle_steps"] = max(int(adjustments.get("settle_steps", 500) or 500), 500)
                    adjustments["freeze_robot_during_settle"] = True
                    adjustments["object_offsets"] = propagate_coupled_offsets(plan_offsets(plan), task_cfg, variant)
                    for target in variant.get("target_objects") or []:
                        name = str(target.get("target_object_name") or target.get("target_object") or "")
                        plan_target = plan_targets.get(name)
                        if not plan_target:
                            continue
                        for key in ("target_kind", "dynamic_role", "target_body_name", "body_name"):
                            if plan_target.get(key) is not None:
                                target[key] = plan_target[key]
                        target["motion"] = clean_motion(target.get("motion") or {}, plan_target)
                        target["latency"] = copy.deepcopy(target["motion"])
                        tune_target_physics(target)
                        summary["updated_targets"] += 1
                    variant["maskfit_alignment"] = None
                    summary["updated_variants"] += 1
        cfg = strip_shrink_metadata(cfg)
        out_path = args.output_config / cfg_path.name
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (args.output_config / "rebuild_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
