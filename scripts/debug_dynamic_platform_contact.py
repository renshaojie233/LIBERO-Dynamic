#!/usr/bin/env python3
"""Print contact/platform diagnostics for one LIBERO-Dynamic eval item."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from libero.libero.envs import OffScreenRenderEnv
from scripts.eval_dynamic_dummy_policy import (
    action_dim,
    build_items,
    task_bddl,
    write_effective_config_files,
)


def geom_name(env: OffScreenRenderEnv, geom_id: int) -> str:
    try:
        return str(env.sim.model.geom_id2name(int(geom_id)))
    except Exception:
        return f"<geom:{geom_id}>"


def body_name(env: OffScreenRenderEnv, body_id: int) -> str:
    try:
        return str(env.sim.model.body_id2name(int(body_id)))
    except Exception:
        return f"<body:{body_id}>"


def contact_rows(env: OffScreenRenderEnv, *, target_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_geom_ids: dict[str, set[int]] = {}
    for name in target_names:
        ids: set[int] = set()
        for method_name in ("_target_all_contact_geom_names", "_target_contact_geom_names"):
            try:
                names = getattr(env, method_name)(name)
            except Exception:
                names = []
            for geom in names or []:
                try:
                    ids.add(int(env.sim.model.geom_name2id(str(geom))))
                except Exception:
                    pass
        target_geom_ids[name] = ids
    platform_geom_ids = set()
    for platform in (getattr(env, "dynamic_physical_platforms", {}) or {}).values():
        for key in ("geom_name", "stability_proxy_geom_name"):
            value = platform.get(key)
            if not value:
                continue
            try:
                platform_geom_ids.add(int(env.sim.model.geom_name2id(str(value))))
            except Exception:
                pass
        for value in platform.get("carrier_geom_names", []) or []:
            try:
                platform_geom_ids.add(int(env.sim.model.geom_name2id(str(value))))
            except Exception:
                pass
    support_bodies = set()
    try:
        for name in env._support_body_names():
            support_bodies.add(int(env.sim.model.body_name2id(str(name))))
    except Exception:
        pass
    for idx in range(int(env.sim.data.ncon)):
        c = env.sim.data.contact[idx]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(env.sim.model.geom_bodyid[g1])
        b2 = int(env.sim.model.geom_bodyid[g2])
        row = {
            "idx": idx,
            "geom1": geom_name(env, g1),
            "geom2": geom_name(env, g2),
            "body1": body_name(env, b1),
            "body2": body_name(env, b2),
            "dist": float(c.dist),
            "pos": [float(x) for x in c.pos],
            "platform": bool(g1 in platform_geom_ids or g2 in platform_geom_ids),
            "support": bool(b1 in support_bodies or b2 in support_bodies),
            "targets": [
                name
                for name, ids in target_geom_ids.items()
                if g1 in ids or g2 in ids
            ],
        }
        if row["platform"] or row["support"] or row["targets"]:
            rows.append(row)
    return rows


def target_report(env: OffScreenRenderEnv, name: str) -> dict[str, Any]:
    joint = env._object_joint_name(name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
    quat = qpos[3:7] if qpos.size >= 7 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    upright = float(1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
    upright = max(-1.0, min(1.0, upright))
    tilt = math.degrees(math.acos(upright))
    platform = (getattr(env, "dynamic_physical_platforms", {}) or {}).get(name)
    platform_pos = None
    platform_top_z = None
    if platform:
        joint_names = platform.get("joint_names") or {}
        try:
            platform_pos = [
                float(env.sim.data.get_joint_qpos(joint_names["x"])),
                float(env.sim.data.get_joint_qpos(joint_names["y"])),
                float(env.sim.data.get_joint_qpos(joint_names["z"])),
            ]
            platform_top_z = float(platform_pos[2]) + float(platform.get("size", [0, 0, 0])[2])
        except Exception:
            pass
    try:
        bottom_z_legacy = float(env._target_bottom_z_legacy(name))
    except Exception:
        bottoms = []
        try:
            contact_geoms_for_legacy = env._target_contact_geom_names(name)
        except Exception:
            contact_geoms_for_legacy = []
        for geom in contact_geoms_for_legacy or []:
            try:
                geom_id = int(env.sim.model.geom_name2id(str(geom)))
                bottoms.append(
                    float(env.sim.data.geom_xpos[geom_id][2])
                    - float(env.sim.model.geom_size[geom_id][2])
                )
            except Exception:
                pass
        bottom_z_legacy = float(min(bottoms)) if bottoms else float("nan")
    out = {
        "name": name,
        "joint": joint,
        "qpos": [float(x) for x in qpos[:7]],
        "tilt_deg": tilt,
        "bottom_z": float(env._target_bottom_z(name)),
        "bottom_z_legacy": bottom_z_legacy,
        "support_contacts": len(env._target_support_contacts(name)),
        "target_contact_geoms": env._target_contact_geom_names(name, 0.002),
        "all_contact_geoms": env._target_all_contact_geom_names(name),
        "platform": platform,
        "platform_pos": platform_pos,
        "platform_top_z": platform_top_z,
    }
    if platform_top_z is not None:
        out["bottom_minus_platform_top"] = float(out["bottom_z"]) - float(platform_top_z)
        out["legacy_bottom_minus_platform_top"] = float(out["bottom_z_legacy"]) - float(platform_top_z)
    return out


def snapshot(env: OffScreenRenderEnv, label: str, target_names: list[str]) -> dict[str, Any]:
    rows = contact_rows(env, target_names=target_names)
    return {
        "label": label,
        "ncon": int(env.sim.data.ncon),
        "targets": [target_report(env, name) for name in target_names],
        "interesting_contacts": rows[:80],
        "platform_contact_count": sum(1 for r in rows if r["platform"] and r["targets"]),
        "support_contact_count": sum(1 for r in rows if r["support"] and r["targets"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=242)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--pre-policy-dynamic-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--platform-settle-steps", type=int, default=40)
    parser.add_argument("--keep-static-support-contact", action="store_true")
    parser.add_argument("--static-support-clearance-m", type=float, default=0.001)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--control-freq", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective = write_effective_config_files(args.config_dir, args.output_dir, args.levels, args)
    items = build_items(
        args.config_dir,
        args.levels,
        args.traj_indices,
        keep_static_support_contact=bool(args.keep_static_support_contact),
        static_support_clearance_m=float(args.static_support_clearance_m),
        config_path_by_level=effective,
    )
    item = items[int(args.start_index)]
    os.environ["LIBERO_DYNAMIC_CONFIG"] = item["config_path"]
    os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
    os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = item["variant_id"]
    os.environ.setdefault("LIBERO_DYNAMIC_AUTO_FINALIZE", "false")

    suite, _task, bddl = task_bddl(item["suite"], item["task_id"])
    init_states = suite.get_task_init_states(item["task_id"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=int(args.camera_size),
        camera_widths=int(args.camera_size),
        control_freq=int(args.control_freq),
    )
    try:
        env.reset()
        env.set_init_state(init_states[int(args.init_state_id)])
        target_names = [
            str(t.get("target_object_name") or t.get("object_name"))
            for t in (getattr(env, "dynamic_runtime_target_entries", None) or item["task_cfg"].get("target_objects") or [])
            if "moka_pot" in str(t.get("target_object_name") or t.get("object_name"))
        ]
        if not target_names:
            target_names = [
                str(t.get("target_object_name") or t.get("object_name"))
                for t in item["task_cfg"].get("target_objects", [])
            ]
        zero = np.zeros((action_dim(env),), dtype=np.float32)
        reports = {
            "item": item,
            "runtime_target_entries": getattr(env, "dynamic_runtime_target_entries", None),
            "snapshots": [snapshot(env, "after_set_init_state", target_names)],
        }
        checkpoints = {args.num_steps_wait: "after_wait", args.num_steps_wait + args.pre_policy_dynamic_steps: "after_pre_policy"}
        checkpoints[int(args.max_steps)] = "after_max_steps"
        for step in range(1, int(args.max_steps) + 1):
            env.step(zero)
            if step in checkpoints:
                reports["snapshots"].append(snapshot(env, f"{checkpoints[step]}_{step}", target_names))
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    finally:
        env.close()


if __name__ == "__main__":
    main()
