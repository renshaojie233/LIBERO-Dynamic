#!/usr/bin/env python3
"""Template-based LIBERO-Dynamic demonstration generator.

The controller deliberately stays non-neural: it replays a verified successful
official static demonstration as the timing/contact template, while continuously
tracking the observed dynamic target displacement in the simulator and adding a
smooth end-effector correction through normal env.step(action) calls.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


REPO_ROOT = Path(os.environ.get("LIBERO_DYNAMIC_REPO_ROOT", str(Path.cwd()))).expanduser().resolve()
DATASET_ROOT = Path(
    os.environ.get(
        "LIBERO_DYNAMIC_DATASET_ROOT",
        str(REPO_ROOT / "datasets" / "libero_datasets"),
    )
).expanduser().resolve()
NO_DYNAMIC_CONFIG = Path(
    os.environ.get(
        "LIBERO_DYNAMIC_NO_DYNAMIC_CONFIG",
        str(REPO_ROOT / "outputs" / "_no_dynamic_disabled.yaml"),
    )
).expanduser().resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
scripts_dir = REPO_ROOT / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils import transform_utils as T  # noqa: E402

from eval_dynamic_dummy_policy import (  # noqa: E402
    build_items,
    init_diag_state,
    summarize_diagnostics,
    target_entries_for_preview,
    task_bddl,
    update_diagnostics,
    write_effective_config_files,
)

from eval_dynamic_libero_style import (  # noqa: E402
    PREVIEW_COMPATIBLE_CARRIER_TYPES,
    item_preview_carrier_type,
    should_use_preview_compatible_eval,
    setup_preview_compatible_dynamic,
    update_preview_compatible_diagnostics,
    update_preview_state_offsets,
    preview_step_with_physical_dynamic,
    preview_step_with_platform_hold,
)

try:
    from render_dynamic_config_previews import (  # noqa: E402
        motion_period_seconds as preview_motion_period_seconds,
        step_with_platform_hold as preview_step_with_platform_hold,
    )
except Exception:  # pragma: no cover
    preview_step_with_platform_hold = None
    preview_motion_period_seconds = None


@dataclass
class StaticDemo:
    demo_key: str
    demo_path: Path
    actions: np.ndarray
    states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


@dataclass
class StaticTrace:
    demo: StaticDemo
    eef_pos: np.ndarray
    target_pos: dict[str, np.ndarray]
    target_quat: dict[str, np.ndarray]
    target_phase: dict[str, dict[str, Any]]
    static_success_step: int


def ensure_no_dynamic_config() -> None:
    NO_DYNAMIC_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if not NO_DYNAMIC_CONFIG.exists():
        NO_DYNAMIC_CONFIG.write_text("enabled: false\nperturb_type: none\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = list(data_group.keys())
    return sorted(keys, key=lambda k: int(str(k).split("_")[-1]))


def dataset_path_for_task(suite_name: str, task: Any) -> Path:
    return DATASET_ROOT / suite_name / f"{task.name}_demo.hdf5"


def read_static_demo(path: Path, demo_key: str) -> StaticDemo:
    with h5py.File(path, "r") as f:
        g = f[f"data/{demo_key}"]
        actions = np.asarray(g["actions"][()], dtype=np.float32)
        states = np.asarray(g["states"][()], dtype=np.float64)
        rewards = np.asarray(g["rewards"][()], dtype=np.float32) if "rewards" in g else np.zeros(len(actions), dtype=np.float32)
        dones = np.asarray(g["dones"][()], dtype=np.bool_) if "dones" in g else np.zeros(len(actions), dtype=np.bool_)
    return StaticDemo(
        demo_key=str(demo_key),
        demo_path=path,
        actions=actions,
        states=states,
        rewards=rewards,
        dones=dones,
    )


def env_action_dim(env: OffScreenRenderEnv) -> int:
    dim = getattr(env.env, "action_dim", None)
    if dim is not None:
        return int(dim)
    return int(sum(int(getattr(robot, "action_dim", 0)) for robot in getattr(env.env, "robots", []) or []))


def eef_pos_from_obs(obs: dict[str, Any]) -> np.ndarray:
    for key in ("robot0_eef_pos", "ee_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=float).copy()
    raise KeyError(f"No eef position key in obs keys: {sorted(obs.keys())}")


def resize_frame_nearest(frame: np.ndarray, size: int | None = None) -> np.ndarray:
    if size is not None and int(size) > 0 and (frame.shape[0] != int(size) or frame.shape[1] != int(size)):
        y_idx = np.linspace(0, frame.shape[0] - 1, int(size)).astype(np.int64)
        x_idx = np.linspace(0, frame.shape[1] - 1, int(size)).astype(np.int64)
        frame = frame[y_idx][:, x_idx]
    return np.ascontiguousarray(frame)


def obs_rgb_array(obs: dict[str, Any], key: str, size: int | None = None) -> np.ndarray:
    frame = np.asarray(obs[key], dtype=np.uint8)[::-1, ::-1]
    return resize_frame_nearest(frame, size)


def obs_frame(obs: dict[str, Any], size: int | None = None, key: str = "agentview_image") -> np.ndarray:
    frame = np.asarray(obs[key], dtype=np.uint8)[::-1, ::-1]
    return resize_frame_nearest(frame, size)


def align_preview_targets_to_current_phase(env: OffScreenRenderEnv, states: list[dict[str, Any]]) -> None:
    """Initialize phase-offset episodes with object and carrier at the same phase."""
    for state in states:
        if state.get("is_joint"):
            continue
        joint = state.get("joint")
        object_base_xy = state.get("object_base_xy")
        object_to_platform_xy = state.get("object_to_platform_xy")
        offset = np.asarray(state.get("last_offset", np.zeros(3, dtype=float)), dtype=float)
        if not joint or object_base_xy is None:
            continue
        try:
            qpos = np.asarray(env.sim.data.get_joint_qpos(str(joint)).copy(), dtype=float)
        except Exception:
            continue
        if qpos.size < 2:
            continue
        desired_xy = np.asarray(object_base_xy, dtype=float)[:2] + offset[:2]
        qpos[:2] = desired_xy
        env.sim.data.set_joint_qpos(str(joint), qpos)
        try:
            qvel = np.asarray(env.sim.data.get_joint_qvel(str(joint)).copy(), dtype=float)
            qvel[:] = 0.0
            env.sim.data.set_joint_qvel(str(joint), qvel)
        except Exception:
            pass
        platform = state.get("platform") or {}
        joint_names = platform.get("joint_names") or {}
        if object_to_platform_xy is not None and joint_names:
            platform_xy = desired_xy - np.asarray(object_to_platform_xy, dtype=float)[:2]
            for axis, value in (("x", platform_xy[0]), ("y", platform_xy[1])):
                platform_joint = joint_names.get(axis)
                if platform_joint:
                    env.sim.data.set_joint_qpos(str(platform_joint), float(value))
                    try:
                        env.sim.data.set_joint_qvel(str(platform_joint), 0.0)
                    except Exception:
                        pass
    env.sim.forward()


def platform_current_pos(env: OffScreenRenderEnv, platform: dict[str, Any]) -> np.ndarray | None:
    joint_names = (platform or {}).get("joint_names") or {}
    if not joint_names:
        return None
    try:
        return np.asarray(
            [
                float(env.sim.data.get_joint_qpos(str(joint_names["x"]))),
                float(env.sim.data.get_joint_qpos(str(joint_names["y"]))),
                float(env.sim.data.get_joint_qpos(str(joint_names["z"]))),
            ],
            dtype=float,
        )
    except Exception:
        return None


def settle_preview_targets_at_current_phase(
    env: OffScreenRenderEnv,
    states: list[dict[str, Any]],
    *,
    steps: int,
) -> dict[str, Any] | None:
    if int(steps) <= 0 or preview_step_with_platform_hold is None:
        return None
    original_base_pos: list[tuple[dict[str, Any], Any]] = []
    for state in states:
        platform = state.get("platform") or {}
        current_pos = platform_current_pos(env, platform)
        original_base_pos.append((state, state.get("platform_base_pos")))
        if current_pos is not None:
            state["platform_base_pos"] = current_pos
    obs = None
    zero_action = np.zeros(env_action_dim(env), dtype=np.float32)
    try:
        for _ in range(int(steps)):
            obs, _reward, _done, _info = preview_step_with_platform_hold(env, zero_action, states)
    finally:
        for state, base_pos in original_base_pos:
            state["platform_base_pos"] = base_pos
        env.sim.forward()
    return obs


def reanchor_preview_targets_to_current_phase(env: OffScreenRenderEnv, states: list[dict[str, Any]]) -> None:
    for state in states:
        if state.get("is_joint"):
            continue
        joint = state.get("joint")
        offset = np.asarray(state.get("last_offset", np.zeros(3, dtype=float)), dtype=float)
        if not joint:
            continue
        try:
            qpos = np.asarray(env.sim.data.get_joint_qpos(str(joint)).copy(), dtype=float)
        except Exception:
            continue
        if qpos.size < 2:
            continue
        state["base_qpos"] = qpos.copy()
        state["object_base_xy"] = qpos[:2].copy() - offset[:2]
        state["object_base_z"] = float(qpos[2]) if qpos.size >= 3 else float(state.get("object_base_z", 0.0))
        platform = state.get("platform") or {}
        current_platform = platform_current_pos(env, platform)
        if current_platform is not None:
            platform_base = current_platform.copy()
            platform_base[:2] -= offset[:2]
            state["platform_base_pos"] = platform_base
            state["object_to_platform_xy"] = qpos[:2].copy() - current_platform[:2]
    env.sim.forward()


def official_obs_record(obs: dict[str, Any], size: int | None = None) -> dict[str, np.ndarray]:
    ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    ee_ori = np.asarray(T.quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)), dtype=np.float64)
    gripper_states = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    joint_states = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
    return {
        "agentview_rgb": obs_rgb_array(obs, "agentview_image", size=size),
        "eye_in_hand_rgb": obs_rgb_array(obs, "robot0_eye_in_hand_image", size=size),
        "ee_pos": ee_pos,
        "ee_ori": ee_ori,
        "ee_states": np.concatenate([ee_pos, ee_ori]).astype(np.float64),
        "gripper_states": gripper_states,
        "joint_states": joint_states,
        "robot_states": np.concatenate(
            [
                gripper_states,
                ee_pos,
                np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
            ]
        ).astype(np.float64),
    }


def _polyline_length(points: list[Any], *, loop: bool) -> float:
    pts = [np.asarray(point, dtype=float) for point in points if len(point) >= 3]
    if len(pts) <= 1:
        return 0.0
    if bool(loop) and float(np.linalg.norm(pts[-1] - pts[0])) > 1e-9:
        pts = [*pts, pts[0]]
    return float(sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])))


def motion_period_seconds(motion_cfg: dict[str, Any]) -> float:
    motion_cfg = motion_cfg or {}
    # Use exactly the same path sampling as the physical motion runtime.
    # In particular, polynomial splines must use their sampled curve length;
    # treating their control points as a straight polyline changes the period
    # and therefore produces the wrong phase offset.
    if preview_motion_period_seconds is not None:
        runtime_period = preview_motion_period_seconds(motion_cfg)
        if runtime_period is not None and float(runtime_period) > 1e-9:
            return float(runtime_period)
    speed = abs(float(motion_cfg.get("alpha_mps", 0.055) or 0.055))
    if speed <= 1e-9:
        return 0.0
    waypoints = motion_cfg.get("waypoints_xyz")
    if waypoints:
        return _polyline_length(waypoints, loop=bool(motion_cfg.get("loop", True))) / speed
    if "half_period_seconds" in motion_cfg:
        return 2.0 * abs(float(motion_cfg.get("half_period_seconds") or 0.0))
    amplitude = abs(float(motion_cfg.get("amplitude_m", motion_cfg.get("max_abs_offset_m", 0.06)) or 0.06))
    return 4.0 * amplitude / speed if amplitude > 1e-9 else 0.0


def item_longest_motion_period_seconds(item: dict[str, Any]) -> float:
    task_cfg = item.get("task_cfg", {}) or {}
    motions: list[dict[str, Any]] = []
    if isinstance(task_cfg.get("motion"), dict):
        motions.append(task_cfg["motion"])
    for target in task_cfg.get("target_objects") or []:
        if isinstance((target or {}).get("motion"), dict):
            motions.append(target["motion"])
    periods = [motion_period_seconds(motion) for motion in motions]
    return max([period for period in periods if period > 1e-9], default=0.0)


def set_dynamic_speed_mps(node: Any, speed_mps: float | None) -> Any:
    if speed_mps is None:
        return node
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "alpha_mps":
                out[key] = float(speed_mps)
            else:
                out[key] = set_dynamic_speed_mps(value, speed_mps)
        return out
    if isinstance(node, list):
        return [set_dynamic_speed_mps(value, speed_mps) for value in node]
    return node


RECEPTACLE_MOTION_KEYWORDS = ("basket", "bowl", "plate", "tray", "rack", "caddy")
STABLE_PLATFORM_CONTACT_KEYWORDS = (
    "akita_black_bowl",
    "bowl",
    "plate",
    "ramekin",
    "mug",
    "tray",
)


def is_receptacle_motion_target(target: dict[str, Any]) -> bool:
    name = str(target.get("target_object_name") or target.get("object_name") or "").lower()
    return any(keyword in name for keyword in RECEPTACLE_MOTION_KEYWORDS)


def freeze_receptacle_motion_targets(task_cfg: dict[str, Any]) -> dict[str, Any]:
    task_cfg = deepcopy_jsonable(task_cfg)
    targets = task_cfg.get("target_objects") or []
    if targets:
        task_cfg["target_objects"] = [
            target for target in targets if not is_receptacle_motion_target(target or {})
        ]
    if is_receptacle_motion_target(task_cfg):
        task_cfg.pop("motion", None)
        task_cfg["enabled"] = False
    return task_cfg


def target_needs_stable_platform_contact(target: dict[str, Any]) -> bool:
    name = str(target.get("target_object_name") or target.get("object_name") or "").lower()
    return any(keyword in name for keyword in STABLE_PLATFORM_CONTACT_KEYWORDS)


def apply_stable_platform_contact_to_target(target: dict[str, Any]) -> bool:
    if not target_needs_stable_platform_contact(target):
        return False
    target["contact_all_target_geoms"] = True
    target["exclude_static_support_contact"] = False
    target["surface_epsilon_m"] = max(float(target.get("surface_epsilon_m", 0.0) or 0.0), 0.001)
    target["bottom_contact_margin_m"] = max(float(target.get("bottom_contact_margin_m", 0.0015) or 0.0015), 0.01)
    target["stability_proxy_enabled"] = False
    for key in ("stability_proxy_size", "stability_proxy_thickness_m", "stability_proxy_bottom_epsilon_m"):
        target.pop(key, None)
    return True


def normalize_stable_platform_contact_in_task_cfg(task_cfg: dict[str, Any]) -> dict[str, Any]:
    task_cfg = deepcopy_jsonable(task_cfg)
    changed = False
    for target in task_cfg.get("target_objects") or []:
        changed = apply_stable_platform_contact_to_target(target) or changed
    if changed and task_cfg.get("target_objects"):
        task_cfg["motion"] = deepcopy_jsonable((task_cfg["target_objects"][0] or {}).get("motion", task_cfg.get("motion", {})))
    return task_cfg


def normalize_stable_platform_contact_in_config_files(path_by_level: dict[str, str]) -> None:
    for path_value in path_by_level.values():
        path = Path(path_value)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        changed = False
        for suite_tasks in (cfg.get("tasks") or {}).values():
            for task_cfg in (suite_tasks or {}).values():
                for variant in task_cfg.get("activation_variants") or []:
                    for target in variant.get("target_objects") or []:
                        changed = apply_stable_platform_contact_to_target(target) or changed
        if changed:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_item_augmentation(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    speed_mps = getattr(args, "dynamic_speed_mps", None)
    phase_ratio = float(getattr(args, "phase_offset_ratio", 0.0) or 0.0)
    suffix = str(getattr(args, "item_id_suffix", "") or "").strip("_")
    freeze_receptacle_motion = bool(getattr(args, "freeze_receptacle_motion", False))
    out: list[dict[str, Any]] = []
    for item in items:
        item = deepcopy_jsonable(item)
        base_item_id = str(item["item_id"])
        if not bool(getattr(args, "exact_benchmark_phase", False)):
            item["task_cfg"] = normalize_stable_platform_contact_in_task_cfg(item.get("task_cfg", {}) or {})
        if speed_mps is not None:
            item["task_cfg"] = set_dynamic_speed_mps(item.get("task_cfg", {}) or {}, float(speed_mps))
        if freeze_receptacle_motion:
            item["task_cfg"] = freeze_receptacle_motion_targets(item.get("task_cfg", {}) or {})
        period = item_longest_motion_period_seconds(item)
        item["base_item_id"] = base_item_id
        item["augmentation"] = {
            "item_id_suffix": suffix,
            "dynamic_speed_mps": float(speed_mps) if speed_mps is not None else None,
            "phase_offset_ratio": float(phase_ratio),
            "longest_period_seconds": float(period),
            "phase_offset_seconds": float(period * phase_ratio),
            "camera_size": int(getattr(args, "camera_size", 128)),
            "freeze_receptacle_motion": bool(freeze_receptacle_motion),
        }
        if suffix:
            item["item_id"] = f"{base_item_id}__{suffix}"
        out.append(item)
    return out


def deepcopy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def body_id_for_name(env: OffScreenRenderEnv, name: str, target_cfg: dict[str, Any] | None = None) -> int | None:
    target_cfg = target_cfg or {}
    body_name = target_cfg.get("target_body_name")
    if body_name:
        try:
            return int(env.sim.model.body_name2id(str(body_name)))
        except Exception:
            pass

    wrapped = env.env
    obj_body_id = getattr(wrapped, "obj_body_id", {}) or {}
    if name in obj_body_id:
        try:
            return int(obj_body_id[name])
        except Exception:
            pass

    if name in getattr(wrapped, "objects_dict", {}):
        obj = wrapped.objects_dict[name]
        for attr in ("root_body", "name"):
            candidate = getattr(obj, attr, None)
            if candidate:
                try:
                    return int(env.sim.model.body_name2id(str(candidate)))
                except Exception:
                    pass

    if name in getattr(wrapped, "fixtures_dict", {}):
        fixture = wrapped.fixtures_dict[name]
        for candidate in (getattr(fixture, "root_body", None), f"{name}_main", name):
            if candidate:
                try:
                    return int(env.sim.model.body_name2id(str(candidate)))
                except Exception:
                    pass
    return None


def pose_for_name(env: OffScreenRenderEnv, name: str, target_cfg: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    body_id = body_id_for_name(env, name, target_cfg)
    if body_id is not None:
        return (
            np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy(),
            np.asarray(env.sim.data.body_xquat[body_id], dtype=float).copy(),
        )
    for site_name in (name, f"{name}_default_site"):
        try:
            site_id = int(env.sim.model.site_name2id(str(site_name)))
            return (
                np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy(),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        except Exception:
            pass
    try:
        joint = target_cfg.get("target_joint_name") if target_cfg else None
        if joint is None and hasattr(env, "_object_joint_name"):
            joint = env._object_joint_name(name)
        qpos = np.asarray(env.sim.data.get_joint_qpos(str(joint)), dtype=float)
        if qpos.size >= 7:
            return qpos[:3].copy(), qpos[3:7].copy()
        if qpos.size >= 3:
            return qpos[:3].copy(), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    except Exception:
        return None
    return None


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    n = float(np.linalg.norm(q))
    if n <= 1e-9:
        return np.eye(3)
    q /= n
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_delta(static_quat: np.ndarray, dynamic_quat: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_mat(dynamic_quat) @ quat_wxyz_to_mat(static_quat).T


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def proximity_weight(distance: float, near: float, far: float) -> float:
    if far <= near:
        return 1.0 if distance <= near else 0.0
    return smoothstep((far - float(distance)) / (far - near))


def infer_target_phases(
    actions: np.ndarray,
    eef_pos: np.ndarray,
    target_pos: dict[str, np.ndarray],
    *,
    grasp_dist_m: float,
) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    gripper = np.asarray(actions[:, -1], dtype=float) if actions.size else np.zeros(0, dtype=float)
    close_mask = gripper > 0.2
    open_mask = gripper < -0.2
    for name, positions in target_pos.items():
        pos = np.asarray(positions, dtype=float)
        valid = np.all(np.isfinite(pos), axis=1)
        if len(pos) == 0 or not np.any(valid):
            phases[name] = {"picked_target": False}
            continue
        dist = np.full(len(pos), np.inf, dtype=float)
        dist[valid] = np.linalg.norm(np.asarray(eef_pos, dtype=float)[valid] - pos[valid], axis=1)
        pick_candidates = np.where(close_mask[: len(dist)] & (dist < float(grasp_dist_m)))[0]
        if len(pick_candidates) == 0:
            phases[name] = {"picked_target": False, "min_static_dist_m": round(float(np.nanmin(dist)), 5)}
            continue
        pick_step = int(pick_candidates[0])
        release_candidates = np.where(open_mask[: len(dist)] & (np.arange(len(dist)) > pick_step + 8))[0]
        release_step = int(release_candidates[0]) if len(release_candidates) else None
        phases[name] = {
            "picked_target": True,
            "pick_step": pick_step,
            "release_step": release_step,
            "min_static_dist_m": round(float(np.nanmin(dist)), 5),
        }
    return phases


def item_target_cfgs(item: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        entries = target_entries_for_preview(item["task_cfg"])
    except Exception:
        entries = item.get("task_cfg", {}).get("target_objects", [])
    out = []
    for entry in entries or []:
        name = entry.get("target_object_name") or entry.get("object_name")
        if not name:
            continue
        copied = dict(entry)
        copied["name"] = str(name)
        out.append(copied)
    if out:
        return out
    name = item.get("task_cfg", {}).get("target_object_name")
    if name:
        return [{"name": str(name), "target_object_name": str(name)}]
    return []


GOAL_RECEPTACLE_RE = re.compile(r"\((?:On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)")


def goal_receptacle_map_from_bddl(bddl_path: str | Path) -> dict[str, str]:
    try:
        text = Path(bddl_path).read_text(encoding="utf-8")
    except Exception:
        return {}
    goal_text = text.split("(:goal", 1)[1] if "(:goal" in text else text
    return {obj: receptacle for obj, receptacle in GOAL_RECEPTACLE_RE.findall(goal_text)}


def trace_cfgs_with_goal_receptacles(target_cfgs: list[dict[str, Any]], bddl_path: str | Path) -> list[dict[str, Any]]:
    out = [dict(cfg) for cfg in target_cfgs]
    seen = {cfg["name"] for cfg in out}
    goal_map = goal_receptacle_map_from_bddl(bddl_path)
    # Trace both the goal object and its destination. Dynamic YAML often names
    # only a supporting object, while the BDDL goal names the real pick target.
    for name in sorted(set(goal_map) | set(goal_map.values())):
        if name not in seen:
            out.append({"name": name})
            seen.add(name)
    return out


def choose_panel_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "libero_object_task00__l1__traj00",  # basket target
        "libero_object_task07__l2__traj02",  # tall carton into basket
        "libero_goal_task08__l1__traj01",  # bowl / plate
        "libero_spatial_task03__l2__traj03",  # simple spatial bowl / plate
        "libero_spatial_task04__l3__traj04",  # cabinet + plate fixture
        "libero_goal_task00__l2__traj00",  # drawer
        "libero_goal_task07__l2__traj02",  # stove
        "libero_10_task03__l2__traj01",  # stove + moka pot
        "libero_10_task03__l2__traj03",  # bowl drawer close
        "libero_10_task04__l3__traj04",  # two mugs / plates
        "libero_10_task08__l3__traj02",  # two moka pots
        "libero_10_task09__l2__traj00",  # microwave close
    }
    by_id = {row["item_id"]: row for row in items}
    panel = [by_id[item_id] for item_id in sorted(wanted) if item_id in by_id]
    if len(panel) < 8:
        panel.extend(items[: 12 - len(panel)])
    return panel


def make_env(bddl: Path, camera_size: int, control_freq: int) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
        control_freq=int(control_freq),
    )


def static_replay_success(
    suite_name: str,
    task_id: int,
    demo: StaticDemo,
    *,
    camera_size: int,
    control_freq: int,
    max_steps: int | None = None,
) -> tuple[bool, int]:
    os.environ["LIBERO_DYNAMIC_CONFIG"] = str(NO_DYNAMIC_CONFIG)
    os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"
    _suite, _task, bddl = task_bddl(suite_name, task_id)
    env = make_env(bddl, camera_size=128 if camera_size > 128 else camera_size, control_freq=control_freq)
    try:
        env.reset()
        env.set_init_state(demo.states[0])
        limit = len(demo.actions) if max_steps is None else min(len(demo.actions), int(max_steps))
        success = False
        step = 0
        for step, action in enumerate(demo.actions[:limit], start=1):
            _obs, _reward, done, _info = env.step(action)
            success = bool(done) or bool(env.check_success())
            if success:
                return True, step
        return bool(success), int(step)
    finally:
        env.close()


def find_static_demo(
    suite_name: str,
    task_id: int,
    *,
    search_limit: int,
    candidate_scan: int,
    camera_size: int,
    control_freq: int,
    cache_dir: Path,
    demo_key_override: str | None = None,
) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    demo_path = dataset_path_for_task(suite_name, task)
    if demo_key_override:
        demo = read_static_demo(demo_path, str(demo_key_override))
        skip_replay_validation = os.environ.get("LIBERO_EXACT_SKIP_STATIC_REPLAY_VALIDATION", "0") == "1"
        if skip_replay_validation:
            success = True
            success_step = int(len(demo.actions))
        else:
            success, success_step = static_replay_success(
                suite_name,
                task_id,
                demo,
                camera_size=camera_size,
                control_freq=control_freq,
            )
            if not success:
                raise RuntimeError(f"Static demo override {demo_key_override} did not replay successfully for {suite_name} task {task_id}")
        return {
            "suite": suite_name,
            "task_id": int(task_id),
            "task_name": task.name,
            "language": task.language,
            "demo_path": str(demo_path),
            "demo_key": str(demo_key_override),
            "static_success_step": int(success_step),
            "candidate_selection": {
                "mode": "explicit_demo_key_override",
                "demo_key": str(demo_key_override),
            },
            "attempts": [{"demo_key": str(demo_key_override), "success": True, "success_step": int(success_step), "length": int(len(demo.actions))}],
        }
    cache_path = cache_dir / f"{suite_name}_task{int(task_id):02d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        demo_path = Path(cached["demo_path"])
        if demo_path.exists():
            return cached

    attempts = []
    with h5py.File(demo_path, "r") as f:
        keys = sorted_demo_keys(f["data"])
    successes: list[dict[str, Any]] = []
    for offset, demo_key in enumerate(keys[: int(search_limit)]):
        demo = read_static_demo(demo_path, demo_key)
        try:
            success, success_step = static_replay_success(
                suite_name,
                task_id,
                demo,
                camera_size=camera_size,
                control_freq=control_freq,
            )
        except Exception as exc:
            attempts.append({"demo_key": demo_key, "success": False, "error": repr(exc)})
            continue
        attempts.append({"demo_key": demo_key, "success": bool(success), "success_step": int(success_step), "length": int(len(demo.actions))})
        if success:
            successes.append(
                {
                    "demo_key": demo_key,
                    "static_success_step": int(success_step),
                    "length": int(len(demo.actions)),
                    "scan_offset": int(offset),
                }
            )
            if offset + 1 >= int(candidate_scan):
                break
    if successes:
        best = sorted(successes, key=lambda row: (int(row["static_success_step"]), int(row["length"]), int(row["scan_offset"])))[0]
        result = {
            "suite": suite_name,
            "task_id": int(task_id),
            "task_name": task.name,
            "language": task.language,
            "demo_path": str(demo_path),
            "demo_key": best["demo_key"],
            "static_success_step": int(best["static_success_step"]),
            "candidate_selection": {
                "mode": "shortest_static_success_step_within_scan",
                "candidate_scan": int(candidate_scan),
                "successes": successes,
            },
            "attempts": attempts,
        }
        write_json(cache_path, result)
        return result
    raise RuntimeError(f"No successful static replay found for {suite_name} task {task_id}; attempts={attempts[:8]}")


def build_static_trace(
    suite_name: str,
    task_id: int,
    demo_info: dict[str, Any],
    target_cfgs: list[dict[str, Any]],
    *,
    camera_size: int,
    control_freq: int,
    trace_cache_dir: Path | None = None,
) -> StaticTrace:
    demo = read_static_demo(Path(demo_info["demo_path"]), str(demo_info["demo_key"]))
    target_names = sorted({cfg["name"] for cfg in target_cfgs})
    cache_path = None
    if trace_cache_dir is not None:
        safe_targets = "__".join(name.replace("/", "_") for name in target_names) or "no_targets"
        cache_path = trace_cache_dir / f"{suite_name}_task{int(task_id):02d}__{demo.demo_key}__{safe_targets}.npz"
        meta_path = cache_path.with_suffix(".json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(cache_path, allow_pickle=False) as npz:
                target_pos = {name: np.asarray(npz[f"target_pos__{i}"], dtype=np.float32) for i, name in enumerate(meta["target_names"])}
                target_quat = {name: np.asarray(npz[f"target_quat__{i}"], dtype=np.float32) for i, name in enumerate(meta["target_names"])}
                return StaticTrace(
                    demo=demo,
                    eef_pos=np.asarray(npz["eef_pos"], dtype=np.float32),
                    target_pos=target_pos,
                    target_quat=target_quat,
                    target_phase=meta["target_phase"],
                    static_success_step=int(demo_info["static_success_step"]),
                )
    os.environ["LIBERO_DYNAMIC_CONFIG"] = str(NO_DYNAMIC_CONFIG)
    os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"
    _suite, _task, bddl = task_bddl(suite_name, task_id)
    env = make_env(bddl, camera_size=128 if camera_size > 128 else camera_size, control_freq=control_freq)
    eef_positions: list[np.ndarray] = []
    target_pos: dict[str, list[np.ndarray]] = {name: [] for name in target_names}
    target_quat: dict[str, list[np.ndarray]] = {name: [] for name in target_names}
    try:
        obs = env.reset()
        obs = env.set_init_state(demo.states[0])
        for action in demo.actions:
            eef_positions.append(eef_pos_from_obs(obs))
            for cfg in target_cfgs:
                name = cfg["name"]
                pose = pose_for_name(env, name, cfg)
                if pose is None:
                    target_pos[name].append(np.full(3, np.nan, dtype=float))
                    target_quat[name].append(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float))
                else:
                    pos, quat = pose
                    target_pos[name].append(pos)
                    target_quat[name].append(quat)
            obs, _reward, _done, _info = env.step(action)
    finally:
        env.close()
    eef_arr = np.asarray(eef_positions, dtype=np.float32)
    target_pos_arr = {name: np.asarray(vals, dtype=np.float32) for name, vals in target_pos.items()}
    target_quat_arr = {name: np.asarray(vals, dtype=np.float32) for name, vals in target_quat.items()}
    target_phase = infer_target_phases(
        demo.actions,
        eef_arr,
        target_pos_arr,
        grasp_dist_m=0.14,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"eef_pos": eef_arr}
        for i, name in enumerate(target_names):
            save_kwargs[f"target_pos__{i}"] = target_pos_arr[name]
            save_kwargs[f"target_quat__{i}"] = target_quat_arr[name]
        np.savez_compressed(cache_path, **save_kwargs)
        write_json(
            cache_path.with_suffix(".json"),
            {
                "suite": suite_name,
                "task_id": int(task_id),
                "demo_key": demo.demo_key,
                "target_names": target_names,
                "target_phase": target_phase,
            },
        )
    return StaticTrace(
        demo=demo,
        eef_pos=eef_arr,
        target_pos=target_pos_arr,
        target_quat=target_quat_arr,
        target_phase=target_phase,
        static_success_step=int(demo_info["static_success_step"]),
    )


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def temporal_pick_compensation_weight(phase: dict[str, Any], step: int) -> float:
    """Track the picked object through contact, then fade after grasp."""
    pick_step = phase.get("pick_step")
    if pick_step is None or not bool(phase.get("picked_target", False)):
        return -1.0
    pick = int(pick_step)
    ramp_steps = max(1, int(os.environ.get("LIBERO_EXACT_TEMPORAL_RAMP_STEPS", "40")))
    full_pre_steps = max(0, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FULL_PRE_STEPS", "12")))
    full_post_steps = max(0, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FULL_POST_STEPS", "8")))
    fade_steps = max(1, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FADE_STEPS", "12")))
    ramp_start = pick - ramp_steps
    full_start = pick - full_pre_steps
    full_end = pick + full_post_steps
    current = int(step)
    if current < ramp_start:
        return 0.0
    if current < full_start:
        return _smoothstep01((current - ramp_start) / max(1, full_start - ramp_start))
    if current <= full_end:
        return 1.0
    if current < full_end + fade_steps:
        return 1.0 - _smoothstep01((current - full_end) / fade_steps)
    return 0.0


def temporal_fixture_contact_weight(trace: StaticTrace, name: str, step: int) -> float:
    """Time-anchor articulated-fixture compensation around manipulation contact."""
    phase = trace.target_phase.get(str(name), {}) or {}
    if bool(phase.get("picked_target", False)):
        return -1.0
    target_path = trace.target_pos.get(str(name))
    if target_path is None or len(target_path) == 0 or len(trace.eef_pos) == 0:
        return -1.0
    count = min(len(target_path), len(trace.eef_pos))
    distances = np.linalg.norm(
        np.asarray(trace.eef_pos[:count], dtype=float) - np.asarray(target_path[:count], dtype=float),
        axis=1,
    )
    if not np.any(np.isfinite(distances)):
        return -1.0
    contact = int(np.nanargmin(distances))
    ramp_steps = max(1, int(os.environ.get("LIBERO_EXACT_TEMPORAL_RAMP_STEPS", "40")))
    full_pre_steps = max(0, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FULL_PRE_STEPS", "12")))
    fade_steps = max(1, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FADE_STEPS", "12")))
    full_start = contact - full_pre_steps
    ramp_start = full_start - ramp_steps
    # Articulated fixtures must remain synchronized through the complete static
    # interaction, rather than fading immediately after first contact.
    full_end = max(contact + 24, int(trace.static_success_step))
    current = int(step)
    if current < ramp_start:
        return 0.0
    if current < full_start:
        return _smoothstep01((current - ramp_start) / max(1, full_start - ramp_start))
    if current <= full_end:
        return 1.0
    if current < full_end + fade_steps:
        return 1.0 - _smoothstep01((current - full_end) / fade_steps)
    return 0.0


def temporal_receptacle_compensation_weight(
    item: dict[str, Any],
    trace: StaticTrace,
    receptacle_name: str,
    step: int,
) -> float:
    """After grasp, smoothly hand compensation from object to destination."""
    goal_map = item.get("_goal_receptacle_map", {}) or {}
    fade_steps = max(1, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FADE_STEPS", "12")))
    full_post_steps = max(0, int(os.environ.get("LIBERO_EXACT_TEMPORAL_FULL_POST_STEPS", "8")))
    current = int(step)
    weights = []
    for object_name, destination in goal_map.items():
        if str(destination) != str(receptacle_name):
            continue
        phase = trace.target_phase.get(str(object_name), {}) or {}
        if not bool(phase.get("picked_target", False)) or phase.get("pick_step") is None:
            continue
        pick = int(phase["pick_step"])
        release_raw = phase.get("release_step")
        release = int(release_raw) if release_raw is not None else pick + fade_steps
        transition_start = pick + full_post_steps
        transition_end = transition_start + fade_steps
        full_end = release + full_post_steps
        if current < transition_start:
            weight = 0.0
        elif current < transition_end:
            weight = _smoothstep01((current - transition_start) / fade_steps)
        elif current <= full_end:
            weight = 1.0
        elif current < full_end + fade_steps:
            weight = 1.0 - _smoothstep01((current - full_end) / fade_steps)
        else:
            weight = 0.0
        weights.append(float(weight))
    return max(weights) if weights else -1.0


def compensation_for_step(
    env: OffScreenRenderEnv,
    item: dict[str, Any],
    trace: StaticTrace,
    target_cfgs: list[dict[str, Any]],
    t: int,
    *,
    near_m: float,
    far_m: float,
    max_comp_m: float,
    disable_picked_after_release: bool,
    release_disable_margin: int,
    pick_enable_pre_margin: int,
    horizontal_comp_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    idx = min(int(t), len(trace.eef_pos) - 1)
    static_eef = np.asarray(trace.eef_pos[idx], dtype=float)
    weighted = np.zeros(3, dtype=float)
    weight_sum = 0.0
    details = []
    candidates = []
    use_temporal_compensation = os.environ.get("LIBERO_EXACT_TEMPORAL_COMPENSATION", "0") == "1"
    for cfg in target_cfgs:
        name = cfg["name"]
        if name not in trace.target_pos:
            continue
        phase = trace.target_phase.get(name, {})
        pick_step = phase.get("pick_step")
        if (
            not use_temporal_compensation
            and bool(phase.get("picked_target", False))
            and pick_step is not None
            and int(t) < int(pick_step) - int(pick_enable_pre_margin)
        ):
            details.append(
                {
                    "target": name,
                    "weight": 0.0,
                    "disabled_before_pick_step": int(pick_step),
                }
            )
            continue
        release_step = phase.get("release_step")
        if (
            disable_picked_after_release
            and not use_temporal_compensation
            and bool(phase.get("picked_target", False))
            and release_step is not None
            and int(t) >= int(release_step) + int(release_disable_margin)
        ):
            details.append(
                {
                    "target": name,
                    "weight": 0.0,
                    "disabled_after_release_step": int(release_step),
                }
            )
            continue
        static_pos = np.asarray(trace.target_pos[name][idx], dtype=float)
        if not np.all(np.isfinite(static_pos)):
            continue
        dyn_pose = pose_for_name(env, name, cfg)
        if dyn_pose is None:
            continue
        dyn_pos, dyn_quat = dyn_pose
        static_quat = np.asarray(trace.target_quat.get(name, np.zeros((len(trace.eef_pos), 4)))[idx], dtype=float)
        if not np.all(np.isfinite(static_quat)) or np.linalg.norm(static_quat) < 1e-9:
            static_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        distance = float(np.linalg.norm(static_eef - static_pos))
        temporal_weight = temporal_pick_compensation_weight(phase, int(t)) if use_temporal_compensation else -1.0
        receptacle_weight = (
            temporal_receptacle_compensation_weight(item, trace, name, int(t))
            if use_temporal_compensation
            else -1.0
        )
        fixture_contact_weight = (
            temporal_fixture_contact_weight(trace, name, int(t))
            if use_temporal_compensation
            else -1.0
        )
        if temporal_weight >= 0.0:
            weight = float(temporal_weight)
            weight_mode = "temporal_pick_phase"
        elif receptacle_weight >= 0.0:
            weight = float(receptacle_weight)
            weight_mode = "temporal_place_phase"
        elif fixture_contact_weight >= 0.0:
            weight = float(fixture_contact_weight)
            weight_mode = "temporal_contact_phase"
        else:
            weight = proximity_weight(distance, near=near_m, far=far_m)
            weight_mode = "spatial_proximity"
        if weight <= 1e-5:
            continue
        try:
            r_delta = rotation_delta(static_quat, dyn_quat)
        except Exception:
            r_delta = np.eye(3)
        desired_from_target = np.asarray(dyn_pos, dtype=float) + r_delta @ (static_eef - static_pos)
        comp = desired_from_target - static_eef
        if horizontal_comp_only:
            comp[2] = 0.0
        comp_norm = float(np.linalg.norm(comp))
        if comp_norm > max_comp_m > 0:
            comp = comp * (max_comp_m / max(comp_norm, 1e-9))
        candidates.append((weight_mode, float(weight), float(distance), name, comp.copy()))
        details.append(
            {
                "target": name,
                "distance_m": round(distance, 5),
                "weight": round(float(weight), 5),
                "weight_mode": weight_mode,
                "comp_xyz": [round(float(x), 5) for x in comp],
            }
        )
    temporal_candidates = [row for row in candidates if row[0].startswith("temporal_")]
    selected_candidates = temporal_candidates if temporal_candidates else candidates
    for _mode, _weight, _distance, _name, _comp in selected_candidates:
        weighted += _weight * _comp
        weight_sum += _weight
    selection_mode = "temporal_contact_dominant" if temporal_candidates else "spatial_fallback"
    if weight_sum <= 1e-9:
        return np.zeros(3, dtype=float), {"weight_sum": 0.0, "selection_mode": selection_mode, "targets": details}
    comp = weighted / weight_sum
    comp_norm = float(np.linalg.norm(comp))
    if comp_norm > max_comp_m > 0:
        comp = comp * (max_comp_m / max(comp_norm, 1e-9))
    return comp, {"weight_sum": round(float(weight_sum), 5), "selection_mode": selection_mode, "targets": details}


def command_action(
    base_action: np.ndarray,
    current_eef: np.ndarray,
    desired_eef: np.ndarray,
    *,
    correction_gain: float,
    max_delta_m: float,
) -> np.ndarray:
    action = np.asarray(base_action, dtype=np.float32).copy()
    if action.size < 7:
        padded = np.zeros(7, dtype=np.float32)
        padded[: action.size] = action
        action = padded
    err = np.asarray(desired_eef, dtype=float) - np.asarray(current_eef, dtype=float)
    correction = float(correction_gain) * err[:3] / max(float(max_delta_m), 1e-6)
    action[:3] = np.clip(action[:3] + correction.astype(np.float32), -1.0, 1.0)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def active_grasp_guard_target(
    env: OffScreenRenderEnv,
    obs: dict[str, Any],
    trace: StaticTrace,
    target_cfgs: list[dict[str, Any]],
    template_idx: int,
    base_action: np.ndarray,
    hold_counts: dict[str, int],
    *,
    window: int,
    pre_margin: int,
    ready_dist_m: float,
    max_hold_steps: int,
) -> dict[str, Any] | None:
    if len(base_action) == 0 or float(base_action[-1]) <= 0.2:
        return None
    current = eef_pos_from_obs(obs)
    cfg_by_name = {cfg["name"]: cfg for cfg in target_cfgs}
    for name, phase in trace.target_phase.items():
        if not bool(phase.get("picked_target", False)):
            continue
        pick_step = phase.get("pick_step")
        release_step = phase.get("release_step")
        if pick_step is None:
            continue
        if int(template_idx) < int(pick_step) - int(pre_margin):
            continue
        if int(template_idx) > int(pick_step) + int(window):
            continue
        if release_step is not None and int(template_idx) >= int(release_step):
            continue
        if int(hold_counts.get(name, 0)) >= int(max_hold_steps):
            continue
        pose = pose_for_name(env, name, cfg_by_name.get(name, {"name": name}))
        if pose is None:
            continue
        dyn_pos, _dyn_quat = pose
        dist = float(np.linalg.norm(current - np.asarray(dyn_pos, dtype=float)))
        if dist > float(ready_dist_m):
            return {
                "target": name,
                "distance_m": dist,
                "dynamic_target_pos": [float(x) for x in np.asarray(dyn_pos, dtype=float)],
                "pick_step": int(pick_step),
                "template_idx": int(template_idx),
            }
    return None


SCRIPTED_RECEPTACLE_KEYWORDS = ("basket", "bowl", "plate", "tray", "rack")


def choose_receptacle_target(trace: StaticTrace) -> str | None:
    non_picked = [
        (name, phase)
        for name, phase in trace.target_phase.items()
        if not bool(phase.get("picked_target", False))
    ]
    if not non_picked:
        return None
    for keyword in SCRIPTED_RECEPTACLE_KEYWORDS:
        for name, _phase in non_picked:
            if keyword in name:
                return name
    return None


def scripted_pick_place_available(trace: StaticTrace, allowed_names: set[str] | None = None) -> bool:
    picked = [
        name
        for name, phase in trace.target_phase.items()
        if bool(phase.get("picked_target", False)) and (allowed_names is None or name in allowed_names)
    ]
    return bool(picked) and choose_receptacle_target(trace) is not None


def picked_target_names(trace: StaticTrace, allowed_names: set[str] | None = None) -> list[str]:
    return [
        name
        for name, phase in trace.target_phase.items()
        if bool(phase.get("picked_target", False)) and (allowed_names is None or name in allowed_names)
    ]


def language_receptacle_name(language: str) -> str | None:
    text = str(language).lower()
    if "stove" in text:
        return "flat_stove_1"
    return None


def needs_scripted_static_prefix(language: str) -> bool:
    text = str(language).lower()
    return "turn on the stove" in text or ("drawer" in text and "cabinet" in text)


def needs_scripted_static_suffix(language: str) -> bool:
    text = str(language).lower()
    return "close" in text and ("drawer" in text or "cabinet" in text)


def servo_action(
    current: np.ndarray,
    desired: np.ndarray,
    *,
    gripper: float,
    gain: float,
    max_delta_m: float,
    ori_action: np.ndarray | None = None,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    err = np.asarray(desired, dtype=float) - np.asarray(current, dtype=float)
    action[:3] = np.clip(float(gain) * err[:3] / max(float(max_delta_m), 1e-6), -1.0, 1.0)
    if ori_action is not None:
        action[3:6] = np.asarray(ori_action, dtype=np.float32)[:3]
    action[-1] = float(gripper)
    return action


def platform_xyz_for_state(env: OffScreenRenderEnv, platform: dict[str, Any]) -> np.ndarray | None:
    joint_names = platform.get("joint_names") or {}
    vals = []
    for axis in ("x", "y", "z"):
        joint_name = joint_names.get(axis) if isinstance(joint_names, dict) else None
        if not joint_name:
            return None
        try:
            vals.append(float(env.sim.data.get_joint_qpos(str(joint_name))))
        except Exception:
            return None
    return np.asarray(vals, dtype=float)


def quat_angle_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    if a.shape[0] != 4 or b.shape[0] != 4:
        return 0.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    dot = float(np.clip(abs(np.dot(a / an, b / bn)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def capture_motion_evidence_sample(env: OffScreenRenderEnv, target_cfgs: list[dict[str, Any]]) -> dict[str, Any]:
    sample: dict[str, Any] = {"platforms": {}, "targets": {}}
    for state in getattr(env, "dynamic_multi_target_states", []) or []:
        name = str(state.get("name") or "")
        platform = state.get("platform") or {}
        xyz = platform_xyz_for_state(env, platform)
        if name and xyz is not None and np.all(np.isfinite(xyz)):
            sample["platforms"][name] = [float(x) for x in xyz[:3]]
    for cfg in target_cfgs:
        name = str(cfg.get("name") or cfg.get("target_object_name") or "")
        if not name:
            continue
        pose = pose_for_name(env, name, cfg)
        if pose is None:
            continue
        pos, quat = pose
        sample["targets"][name] = {
            "pos": [float(x) for x in np.asarray(pos, dtype=float)[:3]],
            "quat": [float(x) for x in np.asarray(quat, dtype=float)[:4]],
        }
    return sample


def summarize_motion_evidence(
    samples: list[dict[str, Any]],
    *,
    carrier_type: str,
    task_cfg: dict[str, Any],
    min_platform_motion_m: float = 0.005,
    min_fixture_motion_m: float = 0.005,
    min_fixture_yaw_rad: float = 0.03,
) -> dict[str, Any]:
    planned_dynamic = bool(task_cfg.get("enabled", False)) and str(task_cfg.get("perturb_type", "none")) == "dynamic_motion"
    platform_series: dict[str, list[np.ndarray]] = {}
    target_pos_series: dict[str, list[np.ndarray]] = {}
    target_quat_series: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        for name, xyz in (sample.get("platforms") or {}).items():
            arr = np.asarray(xyz, dtype=float)
            if arr.shape[0] >= 3 and np.all(np.isfinite(arr[:3])):
                platform_series.setdefault(str(name), []).append(arr[:3])
        for name, pose in (sample.get("targets") or {}).items():
            pos = np.asarray((pose or {}).get("pos", []), dtype=float)
            quat = np.asarray((pose or {}).get("quat", []), dtype=float)
            if pos.shape[0] >= 3 and np.all(np.isfinite(pos[:3])):
                target_pos_series.setdefault(str(name), []).append(pos[:3])
            if quat.shape[0] >= 4 and np.all(np.isfinite(quat[:4])):
                target_quat_series.setdefault(str(name), []).append(quat[:4])

    platforms = {}
    for name, vals in sorted(platform_series.items()):
        arr = np.asarray(vals, dtype=float)
        xyz_range = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        platforms[name] = {
            "samples": int(len(vals)),
            "range_xyz_m": [round(float(x), 6) for x in xyz_range],
            "range_xy_m": round(float(np.linalg.norm(xyz_range[:2])), 6),
        }

    targets = {}
    for name, vals in sorted(target_pos_series.items()):
        arr = np.asarray(vals, dtype=float)
        xyz_range = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        quat_vals = target_quat_series.get(name, [])
        quat_range = 0.0
        if quat_vals:
            q0 = np.asarray(quat_vals[0], dtype=float)
            quat_range = max(quat_angle_rad(q0, np.asarray(q, dtype=float)) for q in quat_vals)
        targets[name] = {
            "samples": int(len(vals)),
            "range_xyz_m": [round(float(x), 6) for x in xyz_range],
            "range_xy_m": round(float(np.linalg.norm(xyz_range[:2])), 6),
            "quat_angle_range_rad": round(float(quat_range), 6),
        }

    reasons = []
    carrier = str(carrier_type or "none")
    if planned_dynamic and carrier in {"physical_conveyor_patch", "physical_platform"}:
        if not platforms:
            reasons.append("no_platform_motion_samples")
        for name, entry in platforms.items():
            if float(entry.get("range_xy_m", 0.0)) < float(min_platform_motion_m):
                reasons.append(f"platform_motion_too_small:{name}")
    elif planned_dynamic:
        max_target_xy = max((float(entry.get("range_xy_m", 0.0)) for entry in targets.values()), default=0.0)
        max_target_yaw = max((float(entry.get("quat_angle_range_rad", 0.0)) for entry in targets.values()), default=0.0)
        if max_target_xy < float(min_fixture_motion_m) and max_target_yaw < float(min_fixture_yaw_rad):
            reasons.append("fixture_or_target_motion_too_small")

    return {
        "planned_dynamic": bool(planned_dynamic),
        "carrier_type": carrier,
        "sample_count": int(len(samples)),
        "platforms": platforms,
        "targets": targets,
        "ok": not reasons,
        "failure_reasons": reasons,
        "thresholds": {
            "min_platform_motion_m": float(min_platform_motion_m),
            "min_fixture_motion_m": float(min_fixture_motion_m),
            "min_fixture_yaw_rad": float(min_fixture_yaw_rad),
        },
    }


def strict_success(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if row.get("status") != "ok":
        reasons.append("status_not_ok")
    if not row.get("raw_success", False):
        reasons.append("raw_success_false")
    if int(row.get("assist_steps", 0)) != 0:
        reasons.append("assist_steps_nonzero")
    diagnostics = row.get("diagnostics", {}) or {}
    if not diagnostics.get("ok", False):
        flag_types = [str(flag.get("type")) for flag in diagnostics.get("flags", [])]
        reasons.append("diagnostics_flags:" + ",".join(flag_types))
    if row.get("used_forbidden_state_edit", False):
        reasons.append("forbidden_state_edit")
    motion_evidence = row.get("motion_evidence")
    dynamic_cfg = row.get("dynamic_config") or {}
    expects_dynamic = bool(dynamic_cfg.get("carrier_type") is not None)
    if motion_evidence is None and expects_dynamic:
        reasons.append("missing_motion_evidence")
    elif isinstance(motion_evidence, dict) and bool(motion_evidence.get("planned_dynamic", False)) and not bool(motion_evidence.get("ok", False)):
        reasons.extend(str(reason) for reason in motion_evidence.get("failure_reasons", []) or ["dynamic_motion_evidence_failed"])
    return len(reasons) == 0, reasons


def attempt_summary(row: dict[str, Any]) -> dict[str, Any]:
    controller = row.get("controller") or {}
    return {
        "attempt_label": row.get("attempt_label") or controller.get("attempt_label"),
        "status": row.get("status"),
        "raw_success": row.get("raw_success"),
        "strict_physical_success": row.get("strict_physical_success"),
        "strict_failure_reasons": row.get("strict_failure_reasons"),
        "error": row.get("error"),
        "diagnostic_flags": (row.get("diagnostics") or {}).get("flags", []),
        "controller": controller,
    }


def task_language_for_item(item: dict[str, Any]) -> str:
    try:
        _suite, task, _bddl = task_bddl(item["suite"], int(item["task_id"]))
        return str(getattr(task, "language", ""))
    except Exception:
        return ""


def fallback_retry_specs(item: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    language = task_language_for_item(item).lower()
    preview_supported = item_preview_carrier_type(item) in PREVIEW_COMPATIBLE_CARRIER_TYPES
    route_label = "preview_compatible" if preview_supported else "env_wrapper"
    bottom_drawer_goal = "bottom drawer" in language and "cabinet" in language
    bottom_drawer_object_place_specs = [
        (
            f"{route_label}/scripted_fast_contact_margin100",
            {
                "force_preview_compatible_eval": bool(preview_supported),
                "controller_mode": "scripted_pick_place",
                "scripted_object_servo_place": True,
                "scripted_force_static_prefix": True,
                "scripted_prefix_pre_pick_margin": 100,
                "scripted_hover_z_m": 0.025,
                "scripted_hover_tol_m": 0.20,
                "scripted_object_hover_z_m": 0.06,
                "scripted_grasp_extra_z_m": -0.025,
                "scripted_servo_gain": 2.2,
                "scripted_move_steps": 60,
                "scripted_gripper_steps": 45,
                "scripted_contact_retry": True,
                "scripted_contact_retry_attempts": 9,
                "scripted_contact_retry_move_steps": 20,
                "scripted_contact_retry_gripper_steps": 30,
                "scripted_release_steps": 70,
                "scripted_lift_z_m": 0.18,
                "scripted_place_tol_m": 0.018,
                "scripted_drawer_close_push": True,
                "overwrite": True,
            },
        ),
        (
            f"{route_label}/scripted_object_place_margin50",
            {
                "force_preview_compatible_eval": bool(preview_supported),
                "controller_mode": "scripted_pick_place",
                "scripted_object_servo_place": True,
                "scripted_force_static_prefix": True,
                "scripted_prefix_pre_pick_margin": 50,
                "scripted_gripper_steps": 24,
                "overwrite": True,
            },
        ),
        (
            f"{route_label}/scripted_object_place_margin30",
            {
                "force_preview_compatible_eval": bool(preview_supported),
                "controller_mode": "scripted_pick_place",
                "scripted_object_servo_place": True,
                "scripted_gripper_steps": 24,
                "scripted_force_static_prefix": True,
                "scripted_prefix_pre_pick_margin": 30,
                "overwrite": True,
            },
        ),
        (
            f"{route_label}/scripted_object_place_drawer_push_margin20",
            {
                "force_preview_compatible_eval": bool(preview_supported),
                "controller_mode": "scripted_pick_place",
                "static_demo_key_override": "demo_1",
                "scripted_object_servo_place": True,
                "scripted_receptacle_center_bias": 1.0,
                "scripted_force_static_prefix": True,
                "scripted_prefix_pre_pick_margin": 20,
                "scripted_object_hover_z_m": 0.10,
                "scripted_object_place_extra_z_m": 0.02,
                "scripted_place_extra_z_m": 0.02,
                "scripted_grasp_extra_z_m": -0.005,
                "scripted_gripper_steps": 35,
                "scripted_release_steps": 70,
                "scripted_lift_z_m": 0.18,
                "scripted_move_steps": 100,
                "scripted_place_tol_m": 0.018,
                "scripted_drawer_close_push": True,
                "overwrite": True,
            },
        ),
    ]
    specs: list[tuple[str, dict[str, Any]]] = [
        (
            f"{route_label}/{args.fallback_controller_mode}",
            {
                "force_preview_compatible_eval": bool(preview_supported),
                "controller_mode": str(args.fallback_controller_mode),
                "overwrite": True,
            },
        )
    ]
    if "both moka pots" in language and "stove" in language:
        specs[:0] = [
            (
                f"{route_label}/scripted_moka_deferred_straight_y_push",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_8",
                    "scripted_grasp_extra_x_m": -0.02,
                    "scripted_moka_stove_push": True,
                    "scripted_moka_stove_push_pot1": True,
                    "scripted_moka_stove_defer_push": True,
                    "scripted_moka_stove_push_steps": 120,
                    "scripted_moka_stove_push_passes": 2,
                    "scripted_moka_stove_straight_y_push": True,
                    "scripted_moka_stove_straight_y_target_offset_m": 0.02,
                    "scripted_moka_stove_push_past_center_m": 0.16,
                    "scripted_moka_stove_push_pot1_target_x_m": 0.06,
                    "scripted_moka_stove_push_target_x_m": -0.10,
                    "overwrite": True,
                },
            ),
            (
                f"{route_label}/scripted_moka_deferred_radial_push",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_8",
                    "scripted_grasp_extra_x_m": -0.02,
                    "scripted_moka_stove_push": True,
                    "scripted_moka_stove_push_pot1": False,
                    "scripted_moka_stove_defer_push": True,
                    "scripted_moka_stove_push_steps": 120,
                    "scripted_moka_stove_push_passes": 1,
                    "scripted_moka_stove_straight_y_push": False,
                    "scripted_moka_stove_straight_y_target_offset_m": 0.0,
                    "scripted_moka_stove_push_past_center_m": 0.16,
                    "scripted_moka_stove_push_target_x_m": -0.10,
                    "overwrite": True,
                },
            ),
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        ]
    if "yellow and white mug" in language and "microwave" in language and "close" in language:
        specs.insert(
            0,
            (
                "template_demo8",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_8",
                    "overwrite": True,
                },
            ),
        )
    if "white mug" in language and "left plate" in language and "yellow and white mug" in language and "right plate" in language:
        specs.insert(
            1 if "milk" in language else 0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "bottom drawer" in language and "cabinet" in language and "close" in language:
        specs[:0] = (
            bottom_drawer_object_place_specs
            + [
                (
                    "template_demo1",
                    {
                        "force_preview_compatible_eval": False,
                        "controller_mode": "template",
                        "static_demo_key_override": "demo_1",
                        "overwrite": True,
                    },
                )
            ]
        )
    if "book" in language and "back compartment" in language and "caddy" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "white mug" in language and "plate" in language and "chocolate pudding" in language and "right of the plate" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "top drawer" in language and "bowl" in language and "inside" in language:
        specs.insert(
            0,
            (
                f"{route_label}/scripted_top_drawer_bowl_prefix0_objservo",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_6",
                    "scripted_force_static_prefix": True,
                    "scripted_prefix_pre_pick_margin": 0,
                    "scripted_object_servo_place": True,
                    "scripted_receptacle_center_bias": 1.0,
                    "scripted_object_hover_z_m": 0.10,
                    "scripted_object_place_extra_z_m": 0.0,
                    "scripted_place_extra_z_m": 0.0,
                    "scripted_grasp_extra_z_m": -0.005,
                    "scripted_gripper_steps": 35,
                    "scripted_release_steps": 80,
                    "scripted_lift_z_m": 0.18,
                    "scripted_move_steps": 90,
                    "scripted_place_tol_m": 0.018,
                    "overwrite": True,
                },
            ),
        )
        specs.insert(
            1,
            (
                "template_demo9",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_9",
                    "overwrite": True,
                },
            ),
        )
    if "push" in language and "plate" in language and "front of the stove" in language:
        specs.insert(
            0,
            (
                f"{route_label}/scripted_plate_front_push",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_7",
                    "scripted_plate_front_push": True,
                    "scripted_plate_push_setup_steps": 55,
                    "scripted_plate_push_steps": 130,
                    "scripted_plate_push_behind_m": 0.075,
                    "scripted_plate_push_past_m": 0.035,
                    "scripted_plate_push_height_m": 0.018,
                    "overwrite": True,
                },
            ),
        )
        specs.insert(
            1,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "cream cheese" in language and "bowl" in language:
        specs.insert(
            0,
            (
                f"{route_label}/scripted_cream_cheese_bowl_objservo",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_7",
                    "scripted_ignore_goal_receptacle_targets": True,
                    "scripted_object_servo_place": True,
                    "scripted_receptacle_center_bias": 1.0,
                    "scripted_object_hover_z_m": 0.10,
                    "scripted_object_place_extra_z_m": 0.025,
                    "scripted_place_extra_z_m": 0.025,
                    "scripted_grasp_extra_x_m": -0.026,
                    "scripted_grasp_extra_y_m": 0.017,
                    "scripted_grasp_extra_z_m": -0.05,
                    "scripted_gripper_steps": 60,
                    "scripted_release_steps": 85,
                    "scripted_lift_z_m": 0.18,
                    "scripted_move_steps": 120,
                    "scripted_place_tol_m": 0.018,
                    "overwrite": True,
                },
            ),
        )
        specs.insert(
            1,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "cream cheese" in language and "basket" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "wine bottle" in language and "rack" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
        specs.insert(
            0,
            (
                f"{route_label}/scripted_rack_object_servo",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_6",
                    "scripted_object_servo_place": True,
                    "scripted_receptacle_center_bias": 0.0,
                    "scripted_object_hover_z_m": 0.10,
                    "scripted_object_place_extra_z_m": 0.0,
                    "scripted_place_extra_z_m": 0.0,
                    "scripted_grasp_extra_z_m": -0.005,
                    "scripted_gripper_steps": 30,
                    "scripted_release_steps": 1,
                    "scripted_lift_z_m": 0.18,
                    "scripted_move_steps": 90,
                    "scripted_place_tol_m": 0.018,
                    "overwrite": True,
                },
            ),
        )
        if int(item.get("traj_index", -1)) == 3:
            specs.insert(
                1,
                (
                    "template_demo2",
                    {
                        "force_preview_compatible_eval": False,
                        "controller_mode": "template",
                        "static_demo_key_override": "demo_2",
                        "overwrite": True,
                    },
                ),
            )
    if "wine bottle" in language and "cabinet" in language and "top" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "butter" in language and "basket" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
        if int(item.get("traj_index", -1)) == 3:
            specs.insert(
                0,
                (
                    "template_demo7",
                    {
                        "force_preview_compatible_eval": False,
                        "controller_mode": "template",
                        "static_demo_key_override": "demo_7",
                        "overwrite": True,
                    },
                ),
            )
        if "cream cheese" not in language and int(item.get("traj_index", -1)) == 4:
            specs.insert(
                0,
                (
                    "template_demo6",
                    {
                        "force_preview_compatible_eval": False,
                        "controller_mode": "template",
                        "static_demo_key_override": "demo_6",
                        "overwrite": True,
                    },
                ),
            )
    if ("milk" in language and "basket" in language) or ("orange juice" in language and "basket" in language):
        if "milk" in language:
            specs.insert(
                0,
                (
                    f"{route_label}/scripted_milk_basket_planner_demo4",
                    {
                        "force_preview_compatible_eval": bool(preview_supported),
                        "controller_mode": "scripted_pick_place",
                        "static_demo_key_override": "demo_4",
                        "scripted_object_servo_place": True,
                        "scripted_receptacle_center_bias": 1.0,
                        "scripted_hover_z_m": 0.12,
                        "scripted_object_hover_z_m": 0.14,
                        "scripted_object_place_extra_z_m": 0.055,
                        "scripted_place_extra_z_m": 0.055,
                        "scripted_grasp_extra_z_m": -0.005,
                        "scripted_gripper_steps": 35,
                        "scripted_release_steps": 1,
                        "scripted_lift_z_m": 0.22,
                        "scripted_move_steps": 120,
                        "scripted_place_tol_m": 0.022,
                        "overwrite": True,
                    },
                ),
            )
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "alphabet soup" in language and "tomato sauce" in language and "basket" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "ramekin" in language and "plate" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "plate" in language and ("table center" in language or "cookie box" in language):
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "next to the plate" in language and "on the plate" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if language.strip() == "put the bowl on the plate":
        specs.insert(
            0,
            (
                "template_demo1",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_1",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "top drawer" in language and "plate" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "wooden cabinet" in language and "top drawer" not in language and "plate" in language:
        specs.insert(
            0,
            (
                "template_demo0",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_0",
                    "overwrite": True,
                },
            ),
        )
    if "black bowl" in language and "stove" in language and "plate" in language:
        specs.insert(
            0,
            (
                "template_demo1",
                {
                    "force_preview_compatible_eval": False,
                    "controller_mode": "template",
                    "static_demo_key_override": "demo_1",
                    "overwrite": True,
                },
            ),
        )
    if not bool(args.fallback_cascade):
        return specs

    specs.extend(
        [
            (
                f"{route_label}/template_grasp_guard",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "template",
                    "grasp_guard": True,
                    "overwrite": True,
                },
            ),
            (
                f"{route_label}/template_grasp_guard_gain12",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "template",
                    "grasp_guard": True,
                    "correction_gain": 1.2,
                    "comp_smoothing_alpha": 0.55,
                    "max_comp_m": 0.35,
                    "overwrite": True,
                },
            ),
        ]
    )

    if "stove" in language and not ("push" in language and "plate" in language and "front of the stove" in language):
        specs.append(
            (
                f"{route_label}/scripted_stove_xminus02",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "scripted_grasp_extra_x_m": -0.02,
                    "overwrite": True,
                },
            )
        )
    if "left plate" in language and "right plate" in language:
        specs.append(
            (
                f"{route_label}/auto_demo5",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "auto",
                    "static_demo_key_override": "demo_5",
                    "overwrite": True,
                },
            )
        )
        specs.append(
            (
                f"{route_label}/scripted_plate_center_object_servo",
                {
                    "force_preview_compatible_eval": bool(preview_supported),
                    "controller_mode": "scripted_pick_place",
                    "static_demo_key_override": "demo_5",
                    "scripted_object_servo_place": True,
                    "scripted_receptacle_center_bias": 1.0,
                    "scripted_object_place_extra_z_m": 0.02,
                    "scripted_place_extra_z_m": 0.02,
                    "scripted_grasp_extra_z_m": -0.005,
                    "scripted_gripper_steps": 30,
                    "scripted_release_steps": 70,
                    "scripted_lift_z_m": 0.18,
                    "scripted_move_steps": 70,
                    "scripted_place_tol_m": 0.018,
                    "overwrite": True,
                },
            )
        )
    if "alphabet soup" in language and "cream cheese" in language and "basket" in language:
        specs.extend(
            [
                (
                    f"{route_label}/auto_demo8",
                    {
                        "force_preview_compatible_eval": bool(preview_supported),
                        "controller_mode": "auto",
                        "static_demo_key_override": "demo_8",
                        "overwrite": True,
                    },
                ),
                (
                    f"{route_label}/auto_demo3",
                    {
                        "force_preview_compatible_eval": bool(preview_supported),
                        "controller_mode": "auto",
                        "static_demo_key_override": "demo_3",
                        "overwrite": True,
                    },
                ),
            ]
        )
    if bottom_drawer_goal and not any("/scripted_object_place" in label for label, _updates in specs):
        specs.extend(bottom_drawer_object_place_specs)
    return specs


def run_dynamic_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    args = argparse.Namespace(**vars(args))
    item_id = item["item_id"]
    item_dir = args.output_dir / "episodes" / item["level"] / item["suite"] / item_id
    manifest_path = item_dir / "manifest.json"
    episode_npz = item_dir / "episode.npz"
    video_path = item_dir / "agentview_rgb.mp4"
    eye_in_hand_video_path = item_dir / "eye_in_hand_rgb.mp4"
    if not args.overwrite and manifest_path.exists() and episode_npz.exists():
        row = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not bool(args.rerun_failed) or strict_success(row)[0]:
            row["resume_skipped"] = True
            return row

    started = time.time()
    item_dir.mkdir(parents=True, exist_ok=True)
    carrier_type = item_preview_carrier_type(item)
    use_preview_compatible_eval = bool(args.force_preview_compatible_eval) or should_use_preview_compatible_eval(item, args)
    if use_preview_compatible_eval:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(NO_DYNAMIC_CONFIG)
    else:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(item["config_path"])
    os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
    os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = str(item["variant_id"])
    os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"
    os.environ["LIBERO_DYNAMIC_EVAL_OUTPUT_DIR"] = str(args.output_dir)

    suite, task, bddl = task_bddl(item["suite"], item["task_id"])
    item["_goal_receptacle_map"] = goal_receptacle_map_from_bddl(bddl)
    explicit_static_demo_key_override = getattr(args, "static_demo_key_override", None)
    benchmark_init_states = suite.get_task_init_states(int(item["task_id"]))
    benchmark_init_state_id = int(getattr(args, "benchmark_init_state_id", 0))
    if benchmark_init_state_id < 0 or benchmark_init_state_id >= len(benchmark_init_states):
        raise IndexError(f"benchmark init-state id {benchmark_init_state_id} out of range {len(benchmark_init_states)}")
    benchmark_init_state = np.asarray(benchmark_init_states[benchmark_init_state_id], dtype=np.float64)
    task_language = getattr(task, "language", "")
    language_lower = task_language.lower()
    attempt_label = str(getattr(args, "attempt_label", "primary"))
    if (
        not bool(args.disable_primary_overrides)
        and attempt_label == "primary"
        and use_preview_compatible_eval
        and str(args.controller_mode) == "auto"
        and "bottom drawer" in language_lower
        and "cabinet" in language_lower
    ):
        args.attempt_label = "primary/scripted_bottom_drawer_object_place"
        args.controller_mode = "scripted_pick_place"
        args.scripted_object_servo_place = True
        args.scripted_force_static_prefix = True
        args.scripted_prefix_pre_pick_margin = 50
        args.scripted_gripper_steps = 24
    if (
        not bool(args.disable_primary_overrides)
        and use_preview_compatible_eval
        and str(args.controller_mode) == "scripted_pick_place"
        and "bottom drawer" in language_lower
        and "cabinet" in language_lower
    ):
        args.scripted_force_static_prefix = True
    if (
        not bool(args.disable_primary_overrides)
        and attempt_label == "primary"
        and use_preview_compatible_eval
        and str(args.controller_mode) == "template"
    ):
        if "book" in language_lower and "back compartment" in language_lower and "caddy" in language_lower:
            args.attempt_label = "primary/scripted_caddy_back_object_servo"
            args.controller_mode = "scripted_pick_place"
            args.static_demo_key_override = "demo_4"
            args.scripted_object_servo_place = True
            args.scripted_receptacle_center_bias = 1.0
            args.scripted_object_hover_z_m = 0.10
            args.scripted_object_place_extra_z_m = 0.0
            args.scripted_place_extra_z_m = 0.0
            args.scripted_grasp_extra_z_m = -0.005
            args.scripted_gripper_steps = 35
            args.scripted_release_steps = 1
            args.scripted_lift_z_m = 0.18
            args.scripted_move_steps = 100
            args.scripted_place_tol_m = 0.018
        elif "milk" in language_lower and "basket" in language_lower:
            args.attempt_label = "primary/scripted_milk_basket_planner_demo4"
            args.controller_mode = "scripted_pick_place"
            args.static_demo_key_override = "demo_4"
            args.scripted_object_servo_place = True
            args.scripted_receptacle_center_bias = 1.0
            args.scripted_hover_z_m = 0.12
            args.scripted_object_hover_z_m = 0.14
            args.scripted_object_place_extra_z_m = 0.055
            args.scripted_place_extra_z_m = 0.055
            args.scripted_grasp_extra_z_m = -0.005
            args.scripted_gripper_steps = 35
            args.scripted_release_steps = 1
            args.scripted_lift_z_m = 0.22
            args.scripted_move_steps = 120
            args.scripted_place_tol_m = 0.022
        elif "butter" in language_lower and "basket" in language_lower and "cream cheese" not in language_lower and int(item.get("traj_index", -1)) == 3:
            args.attempt_label = "primary/template_demo7"
            args.controller_mode = "template"
            args.static_demo_key_override = "demo_7"
        elif "butter" in language_lower and "basket" in language_lower and "cream cheese" not in language_lower and int(item.get("traj_index", -1)) == 4:
            args.attempt_label = "primary/template_demo6"
            args.controller_mode = "template"
            args.static_demo_key_override = "demo_6"
        elif "cream cheese" in language_lower and "butter" in language_lower and "basket" in language_lower:
            args.attempt_label = "primary/scripted_basket_object_servo"
            args.controller_mode = "scripted_pick_place"
            args.static_demo_key_override = "demo_4"
            args.scripted_object_servo_place = True
            args.scripted_receptacle_center_bias = 1.0
            args.scripted_object_hover_z_m = 0.10
            args.scripted_object_place_extra_z_m = 0.025
            args.scripted_place_extra_z_m = 0.025
            args.scripted_grasp_extra_z_m = -0.005
            args.scripted_gripper_steps = 30
            args.scripted_release_steps = 80
            args.scripted_lift_z_m = 0.18
            args.scripted_move_steps = 80
            args.scripted_place_tol_m = 0.02
        elif "alphabet soup" in language_lower and "cream cheese" in language_lower and "basket" in language_lower:
            args.attempt_label = "primary/auto_demo8"
            args.controller_mode = "auto"
            args.static_demo_key_override = "demo_8"
        elif "white mug" in language_lower and "left plate" in language_lower and "yellow and white mug" in language_lower and "right plate" in language_lower:
            args.attempt_label = "primary/scripted_plate_center_object_servo"
            args.controller_mode = "scripted_pick_place"
            args.static_demo_key_override = "demo_5"
            args.scripted_object_servo_place = True
            args.scripted_receptacle_center_bias = 1.0
            args.scripted_object_place_extra_z_m = 0.02
            args.scripted_place_extra_z_m = 0.02
            args.scripted_grasp_extra_z_m = -0.005
            args.scripted_gripper_steps = 30
            args.scripted_release_steps = 70
            args.scripted_lift_z_m = 0.18
            args.scripted_move_steps = 70
            args.scripted_place_tol_m = 0.018
        elif "wine bottle" in language_lower and "rack" in language_lower:
            if int(item.get("traj_index", -1)) == 3:
                args.attempt_label = "primary/template_demo2"
                args.controller_mode = "template"
                args.static_demo_key_override = "demo_2"
            else:
                args.attempt_label = "primary/scripted_rack_object_servo"
                args.controller_mode = "scripted_pick_place"
                args.static_demo_key_override = "demo_6"
                args.scripted_object_servo_place = True
                args.scripted_receptacle_center_bias = 0.0
                args.scripted_object_hover_z_m = 0.10
                args.scripted_object_place_extra_z_m = 0.0
                args.scripted_place_extra_z_m = 0.0
                args.scripted_grasp_extra_z_m = -0.005
                args.scripted_gripper_steps = 30
                args.scripted_release_steps = 1
                args.scripted_lift_z_m = 0.18
                args.scripted_move_steps = 90
                args.scripted_place_tol_m = 0.018
        elif "both moka pots" in language_lower and "stove" in language_lower:
            moka_traj_index = int(item.get("traj_index", -1))
            args.attempt_label = "primary/scripted_moka_deferred_radial_push"
            args.controller_mode = "scripted_pick_place"
            args.static_demo_key_override = "demo_8"
            args.scripted_grasp_extra_x_m = -0.02
            args.scripted_moka_stove_push = True
            args.scripted_moka_stove_push_pot1 = False
            args.scripted_moka_stove_defer_push = True
            args.scripted_moka_stove_push_steps = 120
            args.scripted_moka_stove_push_passes = 1
            args.scripted_moka_stove_straight_y_push = False
            args.scripted_moka_stove_straight_y_target_offset_m = 0.0
            args.scripted_moka_stove_push_past_center_m = 0.16
            args.scripted_moka_stove_push_target_x_m = -0.10
            if moka_traj_index == 2:
                args.attempt_label = "primary/scripted_moka_pot1first_prefix_place_push"
                args.scripted_grasp_extra_x_m = 0.0
                args.scripted_grasp_extra_y_m = 0.0
                args.scripted_grasp_extra_z_m = 0.0
                args.scripted_moka_stove_pot1_first = True
                args.scripted_force_static_prefix = True
                args.scripted_prefix_pre_pick_margin = 20
                args.scripted_lift_z_m = 0.16
                args.scripted_move_steps = 70
                args.scripted_gripper_steps = 35
                args.scripted_release_steps = 55
                args.scripted_receptacle_center_bias = 0.0
                args.scripted_place_extra_z_m = 0.035
                args.scripted_moka_stove_pot2_place_extra_x_m = 0.16
                args.scripted_moka_stove_pot2_place_extra_y_m = 0.09
                args.scripted_moka_stove_push_pot1 = True
                args.scripted_moka_stove_defer_push = True
                args.scripted_moka_stove_push_passes = 2
                args.scripted_moka_stove_straight_y_push = True
                args.scripted_moka_stove_straight_y_target_offset_m = 0.02
                args.scripted_moka_stove_push_pot1_target_x_m = 0.06
                args.scripted_moka_stove_push_target_x_m = -0.10
            elif moka_traj_index == 3:
                args.attempt_label = "primary/scripted_moka_demo2_object_servo_pot1first"
                args.static_demo_key_override = "demo_2"
                args.scripted_grasp_extra_x_m = 0.0
                args.scripted_grasp_extra_y_m = 0.0
                args.scripted_grasp_extra_z_m = 0.0
                args.scripted_moka_stove_push = False
                args.scripted_moka_stove_push_pot1 = False
                args.scripted_moka_stove_pot1_first = True
                args.scripted_force_static_prefix = True
                args.scripted_prefix_pre_pick_margin = 20
                args.scripted_lift_z_m = 0.16
                args.scripted_move_steps = 90
                args.scripted_gripper_steps = 45
                args.scripted_release_steps = 70
                args.scripted_receptacle_center_bias = 1.0
                args.scripted_place_extra_z_m = 0.0
                args.scripted_object_servo_place = True
                args.scripted_moka_stove_object_servo_place = True
                args.scripted_object_hover_z_m = 0.10
                args.scripted_object_place_extra_z_m = 0.0
            elif moka_traj_index == 4:
                args.attempt_label = "primary/scripted_moka_pot1first_prefix_straight_y_push"
                args.scripted_grasp_extra_x_m = 0.0
                args.scripted_grasp_extra_y_m = 0.0
                args.scripted_grasp_extra_z_m = 0.0
                args.scripted_moka_stove_pot1_first = True
                args.scripted_force_static_prefix = True
                args.scripted_prefix_pre_pick_margin = 20
                args.scripted_lift_z_m = 0.16
                args.scripted_move_steps = 70
                args.scripted_gripper_steps = 35
                args.scripted_release_steps = 55
                args.scripted_receptacle_center_bias = 0.0
                args.scripted_place_extra_z_m = 0.035
                args.scripted_moka_stove_push_pot1 = True
                args.scripted_moka_stove_defer_push = True
                args.scripted_moka_stove_push_passes = 2
                args.scripted_moka_stove_straight_y_push = True
                args.scripted_moka_stove_straight_y_target_offset_m = 0.02
                args.scripted_moka_stove_push_pot1_target_x_m = 0.06
                args.scripted_moka_stove_push_target_x_m = -0.10
            elif moka_traj_index != 0:
                args.attempt_label = "primary/scripted_moka_deferred_straight_y_push"
                args.scripted_moka_stove_push_pot1 = True
                args.scripted_moka_stove_push_passes = 2
                args.scripted_moka_stove_straight_y_push = True
                args.scripted_moka_stove_straight_y_target_offset_m = 0.02
                args.scripted_moka_stove_push_pot1_target_x_m = 0.06
    # Dynamic configs may list a supporting object (for example a cookie box)
    # without listing the BDDL goal object resting on it.  Include both sides of
    # the BDDL goal relation so compensation and scripted pick/place select the
    # actual task object instead of the moving support.
    target_cfgs = trace_cfgs_with_goal_receptacles(item_target_cfgs(item), bddl)
    dynamic_target_names = {cfg["name"] for cfg in target_cfgs}
    trace_target_cfgs = target_cfgs
    goal_receptacle_by_object = goal_receptacle_map_from_bddl(bddl)
    if explicit_static_demo_key_override:
        args.static_demo_key_override = explicit_static_demo_key_override
    if bool(getattr(args, "exact_benchmark_phase", False)):
        os.environ["LIBERO_EXACT_SKIP_STATIC_REPLAY_VALIDATION"] = "1"
        os.environ["LIBERO_EXACT_TEMPORAL_COMPENSATION"] = "1" if bool(args.temporal_compensation) else "0"
        os.environ["LIBERO_EXACT_TEMPORAL_RAMP_STEPS"] = str(int(args.temporal_comp_ramp_steps))
        os.environ["LIBERO_EXACT_TEMPORAL_FULL_PRE_STEPS"] = str(int(args.temporal_comp_full_pre_steps))
        os.environ["LIBERO_EXACT_TEMPORAL_FULL_POST_STEPS"] = str(int(args.temporal_comp_full_post_steps))
        os.environ["LIBERO_EXACT_TEMPORAL_FADE_STEPS"] = str(int(args.temporal_comp_fade_steps))
    demo_info = find_static_demo(
        item["suite"],
        int(item["task_id"]),
        search_limit=int(args.static_demo_search),
        candidate_scan=int(args.static_demo_candidate_scan),
        camera_size=int(args.camera_size),
        control_freq=int(args.control_freq),
        cache_dir=args.output_dir / "static_demo_cache",
        demo_key_override=args.static_demo_key_override,
    )
    trace = build_static_trace(
        item["suite"],
        int(item["task_id"]),
        demo_info,
        trace_target_cfgs,
        camera_size=int(args.camera_size),
        control_freq=int(args.control_freq),
        trace_cache_dir=args.output_dir / "static_trace_cache",
    )
    if use_preview_compatible_eval:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(NO_DYNAMIC_CONFIG)
    else:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(item["config_path"])
    os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
    os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = str(item["variant_id"])
    os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"

    env = None
    agentview_writer = None
    eye_in_hand_writer = None
    status = "ok"
    error = None
    error_traceback = None
    diagnostics = None
    raw_success = False
    step_count = 0
    template_idx = 0
    posthoc_success_state_index: int | None = None
    grasp_hold_counts: dict[str, int] = {}
    grasp_hold_events: list[dict[str, Any]] = []
    preview_states: list[dict[str, Any]] = []
    preview_elapsed = 0.0
    preview_dt = 1.0 / float(args.control_freq)
    phase_prime_seconds = 0.0
    phase_prime_steps = 0
    comp_filtered = np.zeros(3, dtype=float)
    max_comp_norm = 0.0
    comp_debug_samples: list[dict[str, Any]] = []
    action_records: list[np.ndarray] = []
    state_records: list[np.ndarray] = []
    reward_records: list[float] = []
    done_records: list[bool] = []
    obs_records: dict[str, list[np.ndarray]] = {
        "agentview_rgb": [],
        "eye_in_hand_rgb": [],
        "ee_pos": [],
        "ee_ori": [],
        "ee_states": [],
        "gripper_states": [],
        "joint_states": [],
        "robot_states": [],
    }
    init_state_record: np.ndarray | None = None
    model_file_record: str | None = None
    motion_samples: list[dict[str, Any]] = []
    final_target_poses: dict[str, dict[str, list[float]]] = {}
    grasp_guard_prev_pos: dict[str, np.ndarray] = {}
    controller_name = "static_template_dynamic_target_pose_compensation"
    exact_benchmark_phase = bool(getattr(args, "exact_benchmark_phase", False))
    selected_init_state = benchmark_init_state if exact_benchmark_phase else trace.demo.states[0]
    try:
        if exact_benchmark_phase:
            random.seed(0)
            np.random.seed(0)
        env = make_env(bddl, camera_size=int(args.camera_size), control_freq=int(args.control_freq))
        if exact_benchmark_phase:
            env.seed(0)
        env.reset()
        if use_preview_compatible_eval:
            obs, preview_states, _preview_settle_done = setup_preview_compatible_dynamic(
                env,
                item,
                args,
                selected_init_state,
            )
        else:
            obs = env.set_init_state(selected_init_state)
        if exact_benchmark_phase:
            augmentation = item.get("augmentation", {}) or {}
            benchmark_warmup_steps = int(getattr(args, "benchmark_num_steps_wait", 10))
            if benchmark_warmup_steps != 10:
                raise ValueError(f"formal benchmark requires exactly 10 warmup steps, got {benchmark_warmup_steps}")
            zero_action = np.zeros(env_action_dim(env), dtype=np.float32)
            for _ in range(benchmark_warmup_steps):
                if use_preview_compatible_eval:
                    obs, _reward, _done, _info = preview_step_with_platform_hold(
                        env,
                        zero_action,
                        preview_states,
                    )
                else:
                    obs, _reward, _done, _info = env.step(zero_action)
            phase_prime_steps = int(getattr(args, "benchmark_pre_policy_dynamic_steps", 40))
            if phase_prime_steps != 40:
                raise ValueError(f"formal benchmark requires exactly 40 pre-policy steps, got {phase_prime_steps}")
            phase_prime_seconds = float(phase_prime_steps) * preview_dt
            for _ in range(phase_prime_steps):
                if use_preview_compatible_eval:
                    obs, _reward, _done, _info = preview_step_with_physical_dynamic(
                        env,
                        zero_action,
                        preview_states,
                        preview_elapsed,
                        preview_dt,
                    )
                    preview_elapsed += preview_dt
                    update_preview_state_offsets(preview_states, preview_elapsed)
                    env.dynamic_multi_target_states = preview_states
                else:
                    obs, _reward, _done, _info = env.step(zero_action)
            phase_prime_applied_seconds = float(phase_prime_steps) * preview_dt
            try:
                obs = env.env._get_observations()
            except Exception:
                pass
        else:
            if use_preview_compatible_eval:
                augmentation = item.get("augmentation", {}) or {}
                phase_prime_seconds = float(augmentation.get("phase_offset_seconds", 0.0) or 0.0)
                if phase_prime_seconds > 1e-9:
                    zero_action = np.zeros(env_action_dim(env), dtype=np.float32)
                    phase_prime_steps = int(round(phase_prime_seconds / max(preview_dt, 1e-9)))
                    for _ in range(max(0, phase_prime_steps)):
                        obs, _reward, _done, _info = preview_step_with_physical_dynamic(
                            env,
                            zero_action,
                            preview_states,
                            preview_elapsed,
                            preview_dt,
                        )
                        preview_elapsed += preview_dt
                        update_preview_state_offsets(preview_states, preview_elapsed)
                        env.dynamic_multi_target_states = preview_states
                    reanchor_preview_targets_to_current_phase(env, preview_states)
                    try:
                        obs = env.env._get_observations()
                    except Exception:
                        pass
        init_state_record = np.asarray(env.get_sim_state(), dtype=np.float64)
        if exact_benchmark_phase:
            exact_snapshot_dir = args.output_dir / "exact_initial_snapshots"
            exact_snapshot_dir.mkdir(parents=True, exist_ok=True)
            exact_snapshot_stem = str(item["item_id"]).replace("/", "__")
            exact_snapshot_path = exact_snapshot_dir / f"{exact_snapshot_stem}.npz"
            exact_snapshot_arrays = {
                "sim_state": np.asarray(init_state_record, dtype=np.float64),
                "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
                "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).copy(),
                "agentview_image_raw": np.asarray(obs.get("agentview_image")),
                "eye_in_hand_image_raw": np.asarray(obs.get("robot0_eye_in_hand_image")),
            }
            np.savez_compressed(exact_snapshot_path, **exact_snapshot_arrays)
            exact_preview_states = []
            for state in preview_states:
                exact_preview_states.append({
                    "name": str(state.get("name", "")),
                    "elapsed_offset": float(state.get("elapsed_offset", 0.0) or 0.0),
                    "last_offset": np.asarray(state.get("last_offset", []), dtype=float).tolist(),
                    "carrier_type": str(state.get("carrier_type", "")),
                })
            write_json(
                exact_snapshot_dir / f"{exact_snapshot_stem}.json",
                {
                    "format": "libero_dynamic_exact_initial_snapshot_v1",
                    "item_id": str(item["item_id"]),
                    "suite": str(item["suite"]),
                    "task_id": int(item["task_id"]),
                    "level": str(item["level"]),
                    "traj_index": int(item["traj_index"]),
                    "benchmark_init_state_id": int(benchmark_init_state_id),
                    "control_freq_hz": int(args.control_freq),
                    "warmup_steps": int(benchmark_warmup_steps),
                    "dynamic_eval_route": "preview_compatible" if use_preview_compatible_eval else "env_wrapper",
                    "pre_policy_dynamic_steps": int(phase_prime_steps),
                    "pre_policy_dynamic_seconds": float(phase_prime_seconds),
                    "preview_elapsed_seconds": float(preview_elapsed),
                    "static_demo_key_override": str(args.static_demo_key_override),
                    "config_path": str(item["config_path"]),
                    "npz_path": str(exact_snapshot_path),
                    "preview_states": exact_preview_states,
                },
            )
        model_file_record = str(env.sim.model.get_xml())
        diagnostics = init_diag_state(env, args)
        if bool(args.save_videos):
            if imageio is None:
                raise RuntimeError("imageio is required when --save-videos is enabled")
            writer_kwargs = {
                "fps": float(args.video_fps),
                "macro_block_size": None,
                "quality": int(args.video_quality),
            }
            agentview_writer = imageio.get_writer(video_path, **writer_kwargs)
            eye_in_hand_writer = imageio.get_writer(eye_in_hand_video_path, **writer_kwargs)

        def step_once(action: np.ndarray, *, env_step: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            nonlocal obs, preview_elapsed, raw_success, step_count
            # Scripted controllers contain several bounded sub-routines; their
            # individual limits must never add up beyond the episode budget.
            # Once --max-steps is reached, return the current observation
            # without advancing physics. Remaining local loops then unwind
            # immediately and the item is recorded as an ordinary failure.
            if step_count >= int(args.max_steps):
                return obs, 0.0, False, {"max_steps_reached": True}
            motion_samples.append(capture_motion_evidence_sample(env, target_cfgs))
            if use_preview_compatible_eval:
                obs, reward, done, info = preview_step_with_physical_dynamic(
                    env,
                    action,
                    preview_states,
                    preview_elapsed,
                    preview_dt,
                )
                preview_elapsed += preview_dt
                update_preview_state_offsets(preview_states, preview_elapsed)
                env.dynamic_multi_target_states = preview_states
            else:
                obs, reward, done, info = env.step(action)
            action_records.append(np.asarray(action, dtype=np.float32))
            state_records.append(np.asarray(env.get_sim_state(), dtype=np.float64))
            reward_records.append(float(reward))
            for key, value in official_obs_record(obs, int(args.camera_size)).items():
                obs_records[key].append(value)
            raw_success = bool(done) or bool(env.check_success())
            done_records.append(bool(raw_success))
            if use_preview_compatible_eval:
                update_preview_compatible_diagnostics(env, diagnostics, preview_states, preview_elapsed)
            else:
                update_diagnostics(env, diagnostics)
            if agentview_writer is not None and (env_step % max(1, int(args.render_every)) == 0):
                frame_size = int(args.model_input_video_size) if args.model_input_video_size else None
                agentview_writer.append_data(obs_frame(obs, frame_size, key="agentview_image"))
                eye_in_hand_writer.append_data(obs_frame(obs, frame_size, key="robot0_eye_in_hand_image"))
            step_count = int(env_step) + 1
            return obs, reward, done, info

        default_receptacle_name = choose_receptacle_target(trace) or language_receptacle_name(task_language)
        picked_dynamic_names = picked_target_names(trace, dynamic_target_names)
        use_scripted = (
            str(args.controller_mode) == "scripted_pick_place"
            or (
                str(args.controller_mode) == "auto"
                and use_preview_compatible_eval
                and bool(picked_dynamic_names)
                and (
                    default_receptacle_name is not None
                    or any(name in goal_receptacle_by_object for name in picked_dynamic_names)
                )
            )
        )
        use_plate_front_push = (
            bool(args.scripted_plate_front_push)
            and "push" in language_lower
            and "plate" in language_lower
            and "front of the stove" in language_lower
        )

        if use_plate_front_push:
            controller_name = "scripted_plate_front_push"
            cfg_by_name = {cfg["name"]: cfg for cfg in trace_target_cfgs}
            scripted_events: list[dict[str, Any]] = []

            def pose_pos(name: str) -> np.ndarray | None:
                pose = pose_for_name(env, name, cfg_by_name.get(name, {"name": name}))
                return None if pose is None else np.asarray(pose[0], dtype=float)

            def scripted_ori_action(static_idx: int | None = None) -> np.ndarray | None:
                if not bool(args.scripted_orientation_template):
                    return None
                if trace.demo.actions.shape[1] < 6:
                    return None
                idx = int(step_count if static_idx is None else static_idx)
                idx = int(np.clip(idx, 0, len(trace.demo.actions) - 1))
                return np.asarray(trace.demo.actions[idx, 3:6], dtype=np.float32)

            def move_to(target_fn, *, gripper: float, max_steps: int, tol_m: float, label: str) -> bool:
                nonlocal step_count
                reached = False
                for _ in range(int(max_steps)):
                    target = target_fn()
                    if target is None:
                        break
                    current = eef_pos_from_obs(obs)
                    dist = float(np.linalg.norm(np.asarray(target, dtype=float) - current))
                    action = servo_action(
                        current,
                        np.asarray(target, dtype=float),
                        gripper=float(gripper),
                        gain=float(args.scripted_servo_gain),
                        max_delta_m=float(args.osc_pos_delta_m),
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                    if len(scripted_events) < int(args.max_logged_scripted_events):
                        scripted_events.append({"label": label, "step": int(step_count), "dist_m": round(dist, 5)})
                    if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                        return True
                    if dist <= float(tol_m):
                        reached = True
                        break
                return reached

            def plate_push_pose(kind: str) -> np.ndarray | None:
                plate = pose_pos("plate_1")
                region = pose_pos("main_table_stove_front_region")
                if plate is None or region is None:
                    return None
                target_xy = np.asarray(region[:2], dtype=float) + np.asarray(
                    [
                        float(args.scripted_plate_push_target_x_offset_m),
                        float(args.scripted_plate_push_target_y_offset_m),
                    ],
                    dtype=float,
                )
                plate_xy = np.asarray(plate[:2], dtype=float)
                direction = target_xy - plate_xy
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    direction = np.asarray([0.0, 1.0], dtype=float)
                else:
                    direction = direction / norm
                if kind == "end":
                    xy = target_xy + direction * float(args.scripted_plate_push_past_m)
                else:
                    xy = plate_xy - direction * float(args.scripted_plate_push_behind_m)
                z = float(plate[2]) + float(args.scripted_plate_push_height_m)
                if kind == "hover":
                    z += float(args.scripted_hover_z_m)
                return np.asarray([float(xy[0]), float(xy[1]), z], dtype=float)

            def plate_side_push_pose(kind: str) -> np.ndarray | None:
                plate = pose_pos("plate_1")
                region = pose_pos("main_table_stove_front_region")
                if plate is None or region is None:
                    return None
                direction_x = -1.0 if float(plate[0]) > float(region[0]) else 1.0
                target_x = float(region[0]) + float(args.scripted_plate_push_target_x_offset_m)
                target_y = float(plate[1])
                if kind == "end":
                    x = target_x + direction_x * float(args.scripted_plate_push_past_m)
                else:
                    x = float(plate[0]) - direction_x * float(args.scripted_plate_push_behind_m)
                z = float(plate[2]) + float(args.scripted_plate_push_height_m)
                if kind == "hover":
                    z += float(args.scripted_hover_z_m)
                return np.asarray([x, target_y, z], dtype=float)

            move_to(
                lambda: plate_push_pose("hover"),
                gripper=-1.0,
                max_steps=int(args.scripted_plate_push_setup_steps),
                tol_m=float(args.scripted_hover_tol_m),
                label="plate_front_push:hover",
            )
            if not raw_success:
                move_to(
                    lambda: plate_push_pose("start"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_setup_steps),
                    tol_m=float(args.scripted_hover_tol_m),
                    label="plate_front_push:start",
                )
            if not raw_success:
                move_to(
                    lambda: plate_push_pose("end"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_steps),
                    tol_m=float(args.scripted_place_tol_m),
                    label="plate_front_push:push",
                )
            if not raw_success:
                move_to(
                    lambda: plate_side_push_pose("hover"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_setup_steps),
                    tol_m=float(args.scripted_hover_tol_m),
                    label="plate_front_push:side_hover",
                )
            if not raw_success:
                move_to(
                    lambda: plate_side_push_pose("start"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_setup_steps),
                    tol_m=float(args.scripted_hover_tol_m),
                    label="plate_front_push:side_start",
                )
            if not raw_success:
                move_to(
                    lambda: plate_side_push_pose("end"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_steps),
                    tol_m=float(args.scripted_place_tol_m),
                    label="plate_front_push:side_push",
                )
            if not raw_success:
                move_to(
                    lambda: plate_push_pose("hover"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_setup_steps),
                    tol_m=float(args.scripted_hover_tol_m),
                    label="plate_front_push:final_hover",
                )
            if not raw_success:
                move_to(
                    lambda: plate_push_pose("start"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_setup_steps),
                    tol_m=float(args.scripted_hover_tol_m),
                    label="plate_front_push:final_start",
                )
            if not raw_success:
                move_to(
                    lambda: plate_push_pose("end"),
                    gripper=-1.0,
                    max_steps=int(args.scripted_plate_push_steps),
                    tol_m=float(args.scripted_place_tol_m),
                    label="plate_front_push:final_push",
                )
            if not raw_success:
                retreat_target = eef_pos_from_obs(obs) + np.asarray([0.0, 0.0, float(args.scripted_lift_z_m)], dtype=float)
                move_to(
                    lambda target=retreat_target: target,
                    gripper=-1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_hover_tol_m,
                    label="plate_front_push:retreat",
                )
            comp_debug_samples.append(
                {
                    "scripted_plate_front_push": True,
                    "events": scripted_events,
                    "target": "main_table_stove_front_region",
                }
            )
        elif use_scripted:
            controller_name = "scripted_relative_pick_place"
            cfg_by_name = {cfg["name"]: cfg for cfg in trace_target_cfgs}
            if bool(args.scripted_goal_object_only):
                picked_names = sorted(
                    name
                    for name in goal_receptacle_by_object
                    if name in dynamic_target_names and name in trace.target_pos
                )
            else:
                picked_names = sorted(
                    picked_dynamic_names,
                    key=lambda name: int(trace.target_phase[name].get("pick_step", 10**9)),
                )
            if bool(args.scripted_ignore_goal_receptacle_targets):
                goal_receptacles = {str(name) for name in goal_receptacle_by_object.values() if name}
                picked_names = [name for name in picked_names if name not in goal_receptacles]
            if bool(args.scripted_moka_stove_pot1_first) and {"moka_pot_1", "moka_pot_2"}.issubset(set(picked_names)):
                picked_names = sorted(picked_names, key=lambda obj_name: 0 if obj_name == "moka_pot_1" else 1)
            scripted_events: list[dict[str, Any]] = []
            scripted_release_source_steps: list[int] = []
            scripted_suffix_anchors: list[tuple[str | None, int]] = []
            deferred_moka_stove_pushes: list[tuple[str, str]] = []
            repick_info_by_name: dict[str, tuple[np.ndarray, np.ndarray, str | None]] = {}

            def receptacle_for_object(obj_name: str) -> str | None:
                if bool(args.scripted_ignore_goal_receptacle_targets) and default_receptacle_name is not None:
                    return default_receptacle_name
                return goal_receptacle_by_object.get(obj_name) or default_receptacle_name

            def pose_pos(name: str) -> np.ndarray | None:
                pose = pose_for_name(env, name, cfg_by_name.get(name, {"name": name}))
                return None if pose is None else np.asarray(pose[0], dtype=float)

            def transformed_relative_target(
                obj_name: str,
                rel: np.ndarray,
                static_quat: np.ndarray | None,
                *,
                offset_z: float = 0.0,
            ) -> np.ndarray | None:
                pose = pose_for_name(env, obj_name, cfg_by_name.get(obj_name, {"name": obj_name}))
                if pose is None:
                    return None
                dyn_pos, dyn_quat = pose
                rel_vec = np.asarray(rel, dtype=float)
                if static_quat is not None and np.all(np.isfinite(static_quat)) and np.linalg.norm(static_quat) > 1e-9:
                    try:
                        rel_vec = rotation_delta(np.asarray(static_quat, dtype=float), np.asarray(dyn_quat, dtype=float)) @ rel_vec
                    except Exception:
                        pass
                return np.asarray(dyn_pos, dtype=float) + rel_vec + np.asarray([0.0, 0.0, float(offset_z)], dtype=float)

            def target_contacted(obj_name: str) -> bool:
                target_diag = ((diagnostics or {}).get("targets") or {}).get(obj_name) or {}
                return bool(target_diag.get("robot_contacted", False))

            def scripted_ori_action(static_idx: int | None = None) -> np.ndarray | None:
                if not bool(args.scripted_orientation_template):
                    return None
                if trace.demo.actions.shape[1] < 6:
                    return None
                idx = int(step_count if static_idx is None else static_idx)
                idx = int(np.clip(idx, 0, len(trace.demo.actions) - 1))
                return np.asarray(trace.demo.actions[idx, 3:6], dtype=np.float32)

            scripted_prefix_steps = 0
            initial_wait_steps = max(0, int(args.scripted_initial_wait_steps))
            if initial_wait_steps:
                wait_action = np.zeros(env_action_dim(env), dtype=np.float32)
                for _ in range(initial_wait_steps):
                    step_once(wait_action, env_step=step_count)
            if (
                bool(args.scripted_static_prefix)
                and (needs_scripted_static_prefix(task_language) or bool(args.scripted_force_static_prefix))
                and picked_names
            ):
                first_pick_step = min(int(trace.target_phase[name].get("pick_step", 0)) for name in picked_names)
                prefix_limit = min(
                    len(trace.demo.actions),
                    max(0, int(first_pick_step) - int(args.scripted_prefix_pre_pick_margin)),
                )
                for prefix_idx in range(prefix_limit):
                    action = np.asarray(trace.demo.actions[prefix_idx], dtype=np.float32).copy()[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                    scripted_prefix_steps += 1
                    if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                        break

            def move_to(target_fn, *, gripper: float, max_steps: int, tol_m: float, label: str) -> bool:
                nonlocal step_count
                reached = False
                for _ in range(int(max_steps)):
                    target = target_fn()
                    if target is None:
                        break
                    current = eef_pos_from_obs(obs)
                    dist = float(np.linalg.norm(np.asarray(target, dtype=float) - current))
                    action = servo_action(
                        current,
                        np.asarray(target, dtype=float),
                        gripper=float(gripper),
                        gain=float(args.scripted_servo_gain),
                        max_delta_m=float(args.osc_pos_delta_m),
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                    if len(scripted_events) < int(args.max_logged_scripted_events):
                        scripted_events.append({"label": label, "step": int(step_count), "dist_m": round(dist, 5)})
                    if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                        return True
                    if dist <= float(tol_m):
                        reached = True
                        break
                return reached

            def move_object_to(
                obj_name: str,
                target_fn,
                *,
                gripper: float,
                max_steps: int,
                tol_m: float,
                label: str,
            ) -> bool:
                nonlocal step_count
                reached = False
                for _ in range(int(max_steps)):
                    target = target_fn()
                    pose = pose_for_name(env, obj_name, cfg_by_name.get(obj_name, {"name": obj_name}))
                    if target is None or pose is None:
                        break
                    current_obj = np.asarray(pose[0], dtype=float)
                    current_eef = eef_pos_from_obs(obs)
                    dist = float(np.linalg.norm(np.asarray(target, dtype=float) - current_obj))
                    desired_eef = current_eef + (np.asarray(target, dtype=float) - current_obj)
                    action = servo_action(
                        current_eef,
                        desired_eef,
                        gripper=float(gripper),
                        gain=float(args.scripted_servo_gain),
                        max_delta_m=float(args.osc_pos_delta_m),
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                    if len(scripted_events) < int(args.max_logged_scripted_events):
                        scripted_events.append({"label": label, "step": int(step_count), "dist_m": round(dist, 5)})
                    if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                        return True
                    if dist <= float(tol_m):
                        reached = True
                        break
                return reached

            for name in picked_names:
                if raw_success:
                    break
                phase = trace.target_phase[name]
                pick_step = int(phase.get("pick_step", 0))
                release_value = phase.get("release_step")
                if release_value is None:
                    release_step = min(len(trace.eef_pos) - 1, pick_step + 60)
                else:
                    release_step = int(release_value)
                scripted_release_source_steps.append(release_step)
                receptacle_name = receptacle_for_object(name)
                scripted_suffix_anchors.append((receptacle_name, release_step))
                static_pick_target = np.asarray(trace.target_pos[name][min(pick_step, len(trace.eef_pos) - 1)], dtype=float)
                if not np.all(np.isfinite(static_pick_target)):
                    continue
                static_pick_quat = np.asarray(
                    trace.target_quat.get(name, np.zeros((len(trace.eef_pos), 4)))[min(pick_step, len(trace.eef_pos) - 1)],
                    dtype=float,
                )
                rel_pick = np.asarray(trace.eef_pos[min(pick_step, len(trace.eef_pos) - 1)], dtype=float) - static_pick_target
                rel_pick = rel_pick + np.asarray(
                    [
                        float(args.scripted_grasp_extra_x_m),
                        float(args.scripted_grasp_extra_y_m),
                        float(args.scripted_grasp_extra_z_m),
                    ],
                    dtype=float,
                )
                if name in {"moka_pot_1", "moka_pot_2"}:
                    if name == "moka_pot_1":
                        rel_pick = rel_pick + np.asarray(
                            [
                                float(args.scripted_moka_stove_pot1_grasp_extra_x_m),
                                float(args.scripted_moka_stove_pot1_grasp_extra_y_m),
                                float(args.scripted_moka_stove_pot1_grasp_extra_z_m),
                            ],
                            dtype=float,
                        )
                    else:
                        rel_pick = rel_pick + np.asarray(
                            [
                                float(args.scripted_moka_stove_pot2_grasp_extra_x_m),
                                float(args.scripted_moka_stove_pot2_grasp_extra_y_m),
                                float(args.scripted_moka_stove_pot2_grasp_extra_z_m),
                            ],
                            dtype=float,
                        )
                if receptacle_name and receptacle_name in trace.target_pos:
                    static_rec = np.asarray(trace.target_pos[receptacle_name][min(release_step, len(trace.eef_pos) - 1)], dtype=float)
                    if np.all(np.isfinite(static_rec)):
                        rel_place = np.asarray(trace.eef_pos[min(release_step, len(trace.eef_pos) - 1)], dtype=float) - static_rec
                        static_obj_release = np.asarray(
                            trace.target_pos[name][min(release_step, len(trace.eef_pos) - 1)],
                            dtype=float,
                        )
                        rel_obj_place = static_obj_release - static_rec
                    else:
                        rel_place = np.asarray([0.0, 0.0, 0.16], dtype=float)
                        rel_obj_place = np.asarray([0.0, 0.0, 0.04], dtype=float)
                else:
                    rel_place = np.asarray([0.0, 0.0, 0.16], dtype=float)
                    rel_obj_place = np.asarray([0.0, 0.0, 0.04], dtype=float)
                if receptacle_name is not None:
                    center_bias = float(np.clip(args.scripted_receptacle_center_bias, 0.0, 1.0))
                    rel_place[:2] *= 1.0 - center_bias
                    rel_obj_place[:2] *= 1.0 - center_bias

                def grasp_target(
                    offset_z: float = 0.0,
                    obj_name: str = name,
                    rel: np.ndarray = rel_pick,
                    static_quat: np.ndarray = static_pick_quat,
                ):
                    return transformed_relative_target(obj_name, rel, static_quat, offset_z=float(offset_z))

                repick_info_by_name[name] = (rel_pick.copy(), static_pick_quat.copy(), receptacle_name)

                # Optional collision-avoidance waypoint used only by the
                # exact-phase orchestrator.  It offsets the normal hover pose
                # before returning through one or more evenly spaced
                # transition waypoints; object state and benchmark dynamics
                # are never edited.
                pregrasp_detour = np.asarray(
                    [
                        float(args.scripted_pregrasp_detour_x_m),
                        float(args.scripted_pregrasp_detour_y_m),
                        float(args.scripted_pregrasp_detour_z_m),
                    ],
                    dtype=float,
                )

                def pregrasp_detour_target(scale: float = 1.0) -> np.ndarray | None:
                    target = grasp_target(float(args.scripted_hover_z_m))
                    if target is None:
                        return None
                    return target + pregrasp_detour * float(scale)

                if float(np.linalg.norm(pregrasp_detour)) > 1e-9:
                    move_to(
                        pregrasp_detour_target,
                        gripper=-1.0,
                        max_steps=args.scripted_move_steps,
                        tol_m=args.scripted_hover_tol_m,
                        label=f"{name}:pregrasp_detour",
                    )
                    transition_stages = max(0, int(args.scripted_pregrasp_detour_transition_stages))
                    for transition_idx in range(transition_stages, 0, -1):
                        transition_scale = float(transition_idx) / float(transition_stages + 1)
                        move_to(
                            lambda scale=transition_scale: pregrasp_detour_target(scale),
                            gripper=-1.0,
                            max_steps=max(20, int(args.scripted_move_steps) // 2),
                            tol_m=args.scripted_hover_tol_m,
                            label=f"{name}:pregrasp_transition_{transition_idx}",
                        )
                move_to(lambda: grasp_target(float(args.scripted_hover_z_m)), gripper=-1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_hover_tol_m, label=f"{name}:hover_pick")
                move_to(lambda: grasp_target(0.0), gripper=-1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_grasp_tol_m, label=f"{name}:grasp_pose")
                for _ in range(int(args.scripted_gripper_steps)):
                    target = grasp_target(0.0)
                    if target is None:
                        break
                    current = eef_pos_from_obs(obs)
                    action = servo_action(
                        current,
                        target,
                        gripper=1.0,
                        gain=args.scripted_servo_gain,
                        max_delta_m=args.osc_pos_delta_m,
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                if bool(args.scripted_contact_retry) and not target_contacted(name):
                    retry_radius = float(args.scripted_contact_retry_offset_m)
                    retry_z = float(args.scripted_contact_retry_z_m)
                    retry_offsets = [
                        (0.0, 0.0, retry_z),
                        (retry_radius, 0.0, retry_z),
                        (-retry_radius, 0.0, retry_z),
                        (0.0, retry_radius, retry_z),
                        (0.0, -retry_radius, retry_z),
                        (retry_radius, retry_radius, retry_z),
                        (retry_radius, -retry_radius, retry_z),
                        (-retry_radius, retry_radius, retry_z),
                        (-retry_radius, -retry_radius, retry_z),
                    ]
                    retry_offsets = retry_offsets[: max(1, int(args.scripted_contact_retry_attempts))]
                    for retry_idx, (offset_x, offset_y, offset_z) in enumerate(retry_offsets, start=1):
                        if target_contacted(name):
                            break
                        retry_rel = rel_pick + np.asarray([offset_x, offset_y, offset_z], dtype=float)

                        def retry_grasp_target(
                            hover_z: float = 0.0,
                            obj_name: str = name,
                            rel: np.ndarray = retry_rel,
                            static_quat: np.ndarray = static_pick_quat,
                        ):
                            return transformed_relative_target(obj_name, rel, static_quat, offset_z=float(hover_z))

                        if len(scripted_events) < int(args.max_logged_scripted_events):
                            scripted_events.append(
                                {
                                    "label": f"{name}:contact_retry_{retry_idx}",
                                    "step": int(step_count),
                                    "offset_xyz": [round(float(offset_x), 5), round(float(offset_y), 5), round(float(offset_z), 5)],
                                }
                            )
                        move_to(
                            lambda: retry_grasp_target(float(args.scripted_hover_z_m)),
                            gripper=-1.0,
                            max_steps=max(1, int(args.scripted_contact_retry_move_steps)),
                            tol_m=float(args.scripted_hover_tol_m),
                            label=f"{name}:contact_retry_hover",
                        )
                        move_to(
                            lambda: retry_grasp_target(0.0),
                            gripper=-1.0,
                            max_steps=max(1, int(args.scripted_contact_retry_move_steps)),
                            tol_m=float(args.scripted_grasp_tol_m),
                            label=f"{name}:contact_retry_grasp",
                        )
                        for _ in range(int(args.scripted_contact_retry_gripper_steps)):
                            target = retry_grasp_target(0.0)
                            if target is None:
                                break
                            current = eef_pos_from_obs(obs)
                            action = servo_action(
                                current,
                                target,
                                gripper=1.0,
                                gain=args.scripted_servo_gain,
                                max_delta_m=args.osc_pos_delta_m,
                                ori_action=scripted_ori_action(),
                            )[: env_action_dim(env)]
                            step_once(action, env_step=step_count)
                lift_target = eef_pos_from_obs(obs) + np.asarray([0.0, 0.0, float(args.scripted_lift_z_m)], dtype=float)
                move_to(lambda target=lift_target: target, gripper=1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_hover_tol_m, label=f"{name}:lift")

                def place_target(offset_z: float = 0.0, rel: np.ndarray = rel_place):
                    if receptacle_name is None:
                        return None
                    pos = pose_pos(receptacle_name)
                    extra_x = float(args.scripted_place_extra_x_m)
                    extra_y = float(args.scripted_place_extra_y_m)
                    extra_z = 0.0
                    if bool(args.scripted_moka_stove_place_adjust) and name == "moka_pot_2":
                        extra_y += float(args.scripted_moka_stove_place_adjust_y_m)
                    if name == "moka_pot_1":
                        extra_x += float(args.scripted_moka_stove_pot1_place_extra_x_m)
                        extra_y += float(args.scripted_moka_stove_pot1_place_extra_y_m)
                        extra_z += float(args.scripted_moka_stove_pot1_place_extra_z_m)
                    elif name == "moka_pot_2":
                        extra_x += float(args.scripted_moka_stove_pot2_place_extra_x_m)
                        extra_y += float(args.scripted_moka_stove_pot2_place_extra_y_m)
                        extra_z += float(args.scripted_moka_stove_pot2_place_extra_z_m)
                    return (
                        None
                        if pos is None
                        else pos
                        + rel
                        + np.asarray(
                            [
                                extra_x,
                                extra_y,
                                float(offset_z) + extra_z,
                            ],
                            dtype=float,
                        )
                    )

                def object_place_target(offset_z: float = 0.0, rel: np.ndarray = rel_obj_place):
                    if receptacle_name is None:
                        return None
                    pos = pose_pos(receptacle_name)
                    if bool(args.scripted_moka_stove_object_servo_place) and name in {"moka_pot_1", "moka_pot_2"}:
                        if name == "moka_pot_1":
                            rel = np.asarray(
                                [
                                    float(args.scripted_moka_stove_pot1_object_target_x_m),
                                    float(args.scripted_moka_stove_pot1_object_target_y_m),
                                    float(args.scripted_moka_stove_pot1_object_target_z_m),
                                ],
                                dtype=float,
                            )
                        else:
                            rel = np.asarray(
                                [
                                    float(args.scripted_moka_stove_pot2_object_target_x_m),
                                    float(args.scripted_moka_stove_pot2_object_target_y_m),
                                    float(args.scripted_moka_stove_pot2_object_target_z_m),
                                ],
                                dtype=float,
                            )
                        return None if pos is None else pos + rel + np.asarray([0.0, 0.0, float(offset_z)], dtype=float)
                    return (
                        None
                        if pos is None
                        else pos
                        + rel
                        + np.asarray(
                            [
                                float(args.scripted_object_place_extra_x_m),
                                float(args.scripted_object_place_extra_y_m),
                                float(offset_z),
                            ],
                            dtype=float,
                        )
                    )

                move_to(lambda: place_target(float(args.scripted_hover_z_m)), gripper=1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_hover_tol_m, label=f"{name}:hover_place")
                if bool(args.scripted_drop_above_receptacle):
                    move_to(lambda: place_target(float(args.scripted_hover_z_m)), gripper=-1.0, max_steps=1, tol_m=0.0, label=f"{name}:drop_above_receptacle")
                elif bool(args.scripted_object_servo_place):
                    move_object_to(
                        name,
                        lambda: object_place_target(float(args.scripted_object_hover_z_m)),
                        gripper=1.0,
                        max_steps=args.scripted_move_steps,
                        tol_m=args.scripted_hover_tol_m,
                        label=f"{name}:object_hover_place",
                    )
                    move_object_to(
                        name,
                        lambda: object_place_target(float(args.scripted_object_place_extra_z_m)),
                        gripper=1.0,
                        max_steps=args.scripted_move_steps,
                        tol_m=args.scripted_place_tol_m,
                        label=f"{name}:object_place_pose",
                    )
                else:
                    move_to(lambda: place_target(float(args.scripted_place_extra_z_m)), gripper=1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_place_tol_m, label=f"{name}:place_pose")
                release_steps = 0 if bool(args.scripted_drop_above_receptacle) else int(args.scripted_release_steps)
                for _ in range(release_steps):
                    target = place_target(float(args.scripted_place_extra_z_m))
                    if target is None:
                        break
                    current = eef_pos_from_obs(obs)
                    action = servo_action(
                        current,
                        target,
                        gripper=-1.0,
                        gain=args.scripted_servo_gain,
                        max_delta_m=args.osc_pos_delta_m,
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                should_moka_stove_push = (
                    bool(args.scripted_moka_stove_push)
                    and receptacle_name is not None
                    and (name == "moka_pot_2" or (bool(args.scripted_moka_stove_push_pot1) and name == "moka_pot_1"))
                )
                if should_moka_stove_push:
                    if bool(args.scripted_moka_stove_defer_push):
                        deferred_moka_stove_pushes.append((name, receptacle_name))
                    else:
                        def push_pose(*, side_offset_m: float, height_m: float, past_center_m: float = 0.0):
                            obj_pos = pose_pos(name)
                            rec_pos = pose_pos(receptacle_name)
                            if obj_pos is None or rec_pos is None:
                                return None
                            rec_target = np.asarray(rec_pos, dtype=float) + np.asarray(
                                [
                                    float(args.scripted_moka_stove_push_pot1_target_x_m)
                                    if name == "moka_pot_1"
                                    else float(args.scripted_moka_stove_push_target_x_m),
                                    float(args.scripted_moka_stove_push_target_y_m),
                                    0.0,
                                ],
                                dtype=float,
                            )
                            if bool(args.scripted_moka_stove_push_direct_final) and past_center_m:
                                base = rec_target.copy()
                                base[2] = float(obj_pos[2]) + float(height_m)
                                return base
                            direction = np.asarray(obj_pos, dtype=float) - rec_target
                            direction[2] = 0.0
                            norm = float(np.linalg.norm(direction))
                            if norm < 1e-6:
                                direction = np.asarray([0.0, 1.0, 0.0], dtype=float)
                            else:
                                direction = direction / norm
                            base = np.asarray(obj_pos, dtype=float) + direction * float(side_offset_m)
                            if past_center_m:
                                base = rec_target - direction * float(past_center_m)
                            base[2] = float(obj_pos[2]) + float(height_m)
                            return base

                        move_to(
                            lambda: push_pose(
                                side_offset_m=float(args.scripted_moka_stove_push_hover_side_offset_m),
                                height_m=float(args.scripted_moka_stove_push_hover_height_m),
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_push_setup_steps),
                            tol_m=float(args.scripted_hover_tol_m),
                            label=f"{name}:stove_push_hover",
                        )
                        move_to(
                            lambda: push_pose(
                                side_offset_m=float(args.scripted_moka_stove_push_side_offset_m),
                                height_m=float(args.scripted_moka_stove_push_low_height_m),
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_push_setup_steps),
                            tol_m=float(args.scripted_hover_tol_m),
                            label=f"{name}:stove_push_start",
                        )
                        move_to(
                            lambda: push_pose(
                                side_offset_m=0.0,
                                height_m=float(args.scripted_moka_stove_push_low_height_m),
                                past_center_m=float(args.scripted_moka_stove_push_past_center_m),
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_push_steps),
                            tol_m=float(args.scripted_place_tol_m),
                            label=f"{name}:stove_push",
                        )
                retreat_target = eef_pos_from_obs(obs) + np.asarray([0.0, 0.0, float(args.scripted_lift_z_m)], dtype=float)
                move_to(lambda target=retreat_target: target, gripper=-1.0, max_steps=args.scripted_move_steps, tol_m=args.scripted_hover_tol_m, label=f"{name}:retreat")

            if deferred_moka_stove_pushes and not raw_success:
                ordered_deferred_pushes = sorted(
                    deferred_moka_stove_pushes,
                    key=lambda entry: 0 if entry[0] == "moka_pot_1" else 1,
                )
                repeated_deferred_pushes = ordered_deferred_pushes * max(1, int(args.scripted_moka_stove_push_passes))
                for push_name, push_receptacle in repeated_deferred_pushes:
                    def deferred_push_pose(*, side_offset_m: float, height_m: float, past_center_m: float = 0.0):
                        obj_pos = pose_pos(push_name)
                        rec_pos = pose_pos(push_receptacle)
                        if obj_pos is None or rec_pos is None:
                            return None
                        if bool(args.scripted_moka_stove_straight_y_push):
                            base = np.asarray(obj_pos, dtype=float).copy()
                            base[1] = (
                                float(obj_pos[1]) + float(side_offset_m)
                                if not past_center_m
                                else float(rec_pos[1]) + float(args.scripted_moka_stove_straight_y_target_offset_m)
                            )
                            base[2] = float(obj_pos[2]) + float(height_m)
                            return base
                        if bool(args.scripted_moka_stove_straight_x_push):
                            target_x = (
                                float(args.scripted_moka_stove_push_pot1_target_x_m)
                                if push_name == "moka_pot_1"
                                else float(args.scripted_moka_stove_push_target_x_m)
                            )
                            base = np.asarray(obj_pos, dtype=float).copy()
                            base[0] = float(obj_pos[0]) + float(side_offset_m) if not past_center_m else float(rec_pos[0]) + target_x
                            base[2] = float(obj_pos[2]) + float(height_m)
                            return base
                        rec_target = np.asarray(rec_pos, dtype=float) + np.asarray(
                            [
                                float(args.scripted_moka_stove_push_pot1_target_x_m)
                                if push_name == "moka_pot_1"
                                else float(args.scripted_moka_stove_push_target_x_m),
                                float(args.scripted_moka_stove_push_target_y_m),
                                0.0,
                            ],
                            dtype=float,
                        )
                        if bool(args.scripted_moka_stove_push_direct_final) and past_center_m:
                            base = rec_target.copy()
                            base[2] = float(obj_pos[2]) + float(height_m)
                            return base
                        direction = np.asarray(obj_pos, dtype=float) - rec_target
                        direction[2] = 0.0
                        norm = float(np.linalg.norm(direction))
                        if norm < 1e-6:
                            direction = np.asarray([0.0, 1.0, 0.0], dtype=float)
                        else:
                            direction = direction / norm
                        base = np.asarray(obj_pos, dtype=float) + direction * float(side_offset_m)
                        if past_center_m:
                            base = rec_target - direction * float(past_center_m)
                        base[2] = float(obj_pos[2]) + float(height_m)
                        return base

                    move_to(
                        lambda: deferred_push_pose(
                            side_offset_m=float(args.scripted_moka_stove_push_hover_side_offset_m),
                            height_m=float(args.scripted_moka_stove_push_hover_height_m),
                        ),
                        gripper=float(args.scripted_moka_stove_push_gripper),
                        max_steps=int(args.scripted_moka_stove_push_setup_steps),
                        tol_m=float(args.scripted_hover_tol_m),
                        label=f"{push_name}:deferred_stove_push_hover",
                    )
                    move_to(
                        lambda: deferred_push_pose(
                            side_offset_m=float(args.scripted_moka_stove_push_side_offset_m),
                            height_m=float(args.scripted_moka_stove_push_low_height_m),
                        ),
                        gripper=float(args.scripted_moka_stove_push_gripper),
                        max_steps=int(args.scripted_moka_stove_push_setup_steps),
                        tol_m=float(args.scripted_hover_tol_m),
                        label=f"{push_name}:deferred_stove_push_start",
                    )
                    move_to(
                        lambda: deferred_push_pose(
                            side_offset_m=0.0,
                            height_m=float(args.scripted_moka_stove_push_low_height_m),
                            past_center_m=float(args.scripted_moka_stove_push_past_center_m),
                        ),
                        gripper=float(args.scripted_moka_stove_push_gripper),
                        max_steps=int(args.scripted_moka_stove_push_steps),
                        tol_m=float(args.scripted_place_tol_m),
                        label=f"{push_name}:deferred_stove_push",
                    )
                if bool(args.scripted_moka_stove_after_radial_straight_y_push) and not raw_success:
                    after_ordered_pushes = ordered_deferred_pushes
                    if bool(args.scripted_moka_stove_after_radial_straight_y_pot2_only):
                        after_ordered_pushes = [entry for entry in after_ordered_pushes if entry[0] == "moka_pot_2"]
                    after_pushes = after_ordered_pushes * max(1, int(args.scripted_moka_stove_after_radial_straight_y_passes))
                    for push_name, push_receptacle in after_pushes:
                        def after_straight_y_pose(*, side_offset_m: float, height_m: float, final_y: bool = False):
                            obj_pos = pose_pos(push_name)
                            rec_pos = pose_pos(push_receptacle)
                            if obj_pos is None or rec_pos is None:
                                return None
                            base = np.asarray(obj_pos, dtype=float).copy()
                            base[1] = (
                                float(rec_pos[1]) + float(args.scripted_moka_stove_after_radial_straight_y_target_offset_m)
                                if final_y
                                else float(obj_pos[1]) + float(side_offset_m)
                            )
                            base[2] = float(obj_pos[2]) + float(height_m)
                            return base

                        move_to(
                            lambda: after_straight_y_pose(
                                side_offset_m=float(args.scripted_moka_stove_push_hover_side_offset_m),
                                height_m=float(args.scripted_moka_stove_push_hover_height_m),
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_push_setup_steps),
                            tol_m=float(args.scripted_hover_tol_m),
                            label=f"{push_name}:after_radial_straight_y_hover",
                        )
                        move_to(
                            lambda: after_straight_y_pose(
                                side_offset_m=float(args.scripted_moka_stove_push_side_offset_m),
                                height_m=float(args.scripted_moka_stove_push_low_height_m),
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_push_setup_steps),
                            tol_m=float(args.scripted_hover_tol_m),
                            label=f"{push_name}:after_radial_straight_y_start",
                        )
                        move_to(
                            lambda: after_straight_y_pose(
                                side_offset_m=0.0,
                                height_m=float(args.scripted_moka_stove_push_low_height_m),
                                final_y=True,
                            ),
                            gripper=float(args.scripted_moka_stove_push_gripper),
                            max_steps=int(args.scripted_moka_stove_after_radial_straight_y_steps),
                            tol_m=float(args.scripted_place_tol_m),
                            label=f"{push_name}:after_radial_straight_y_push",
                        )

            if bool(args.scripted_moka_stove_repick_pot2_object_servo) and not raw_success and "moka_pot_2" in repick_info_by_name:
                repick_name = "moka_pot_2"
                rel_pick, static_pick_quat, receptacle_name = repick_info_by_name[repick_name]

                def repick_grasp_target(offset_z: float = 0.0):
                    return transformed_relative_target(repick_name, rel_pick, static_pick_quat, offset_z=float(offset_z))

                move_to(
                    lambda: repick_grasp_target(float(args.scripted_hover_z_m)),
                    gripper=-1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_hover_tol_m,
                    label=f"{repick_name}:repick_hover",
                )
                move_to(
                    lambda: repick_grasp_target(0.0),
                    gripper=-1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_grasp_tol_m,
                    label=f"{repick_name}:repick_grasp_pose",
                )
                for _ in range(int(args.scripted_gripper_steps)):
                    target = repick_grasp_target(0.0)
                    if target is None:
                        break
                    current = eef_pos_from_obs(obs)
                    action = servo_action(
                        current,
                        target,
                        gripper=1.0,
                        gain=args.scripted_servo_gain,
                        max_delta_m=args.osc_pos_delta_m,
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                lift_target = eef_pos_from_obs(obs) + np.asarray([0.0, 0.0, float(args.scripted_lift_z_m)], dtype=float)
                move_to(
                    lambda target=lift_target: target,
                    gripper=1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_hover_tol_m,
                    label=f"{repick_name}:repick_lift",
                )

                def repick_object_target(offset_z: float = 0.0):
                    if receptacle_name is None:
                        return None
                    pos = pose_pos(receptacle_name)
                    rel = np.asarray(
                        [
                            float(args.scripted_moka_stove_pot2_object_target_x_m),
                            float(args.scripted_moka_stove_pot2_object_target_y_m),
                            float(args.scripted_moka_stove_pot2_object_target_z_m),
                        ],
                        dtype=float,
                    )
                    return None if pos is None else pos + rel + np.asarray([0.0, 0.0, float(offset_z)], dtype=float)

                move_object_to(
                    repick_name,
                    lambda: repick_object_target(float(args.scripted_object_hover_z_m)),
                    gripper=1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_hover_tol_m,
                    label=f"{repick_name}:repick_object_hover_place",
                )
                move_object_to(
                    repick_name,
                    lambda: repick_object_target(float(args.scripted_object_place_extra_z_m)),
                    gripper=1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_place_tol_m,
                    label=f"{repick_name}:repick_object_place",
                )
                for _ in range(int(args.scripted_release_steps)):
                    current = eef_pos_from_obs(obs)
                    action = servo_action(
                        current,
                        current,
                        gripper=-1.0,
                        gain=args.scripted_servo_gain,
                        max_delta_m=args.osc_pos_delta_m,
                        ori_action=scripted_ori_action(),
                    )[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                retreat_target = eef_pos_from_obs(obs) + np.asarray([0.0, 0.0, float(args.scripted_lift_z_m)], dtype=float)
                move_to(
                    lambda target=retreat_target: target,
                    gripper=-1.0,
                    max_steps=args.scripted_move_steps,
                    tol_m=args.scripted_hover_tol_m,
                    label=f"{repick_name}:repick_retreat",
                )

            scripted_suffix_steps = 0
            if (
                bool(args.scripted_static_suffix)
                and needs_scripted_static_suffix(task_language)
                and scripted_release_source_steps
                and not raw_success
            ):
                suffix_start = int(max(0, min(scripted_release_source_steps) + int(args.scripted_suffix_start_offset)))
                suffix_start = int(min(suffix_start, len(trace.demo.actions)))
                if bool(args.scripted_align_static_suffix) and scripted_suffix_anchors:
                    anchor_receptacle, _anchor_step = sorted(scripted_suffix_anchors, key=lambda x: int(x[1]))[0]
                    if anchor_receptacle and anchor_receptacle in trace.target_pos and suffix_start < len(trace.eef_pos):
                        static_rec = np.asarray(trace.target_pos[anchor_receptacle][suffix_start], dtype=float)
                        if np.all(np.isfinite(static_rec)):
                            static_rel = np.asarray(trace.eef_pos[suffix_start], dtype=float) - static_rec

                            def suffix_anchor_target(receptacle: str = anchor_receptacle, rel: np.ndarray = static_rel):
                                pos = pose_pos(receptacle)
                                return None if pos is None else pos + rel

                            move_to(
                                suffix_anchor_target,
                                gripper=-1.0,
                                max_steps=args.scripted_suffix_align_steps,
                                tol_m=args.scripted_hover_tol_m,
                                label="static_suffix:align",
                            )
                for suffix_idx in range(suffix_start, len(trace.demo.actions)):
                    action = np.asarray(trace.demo.actions[suffix_idx], dtype=np.float32).copy()[: env_action_dim(env)]
                    step_once(action, env_step=step_count)
                    scripted_suffix_steps += 1
                    if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                        break
                if (
                    bool(args.scripted_drawer_close_push)
                    and not raw_success
                    and scripted_suffix_anchors
                    and len(trace.eef_pos) > 0
                ):
                    anchor_receptacle, _anchor_step = sorted(scripted_suffix_anchors, key=lambda x: int(x[1]))[0]
                    if anchor_receptacle and anchor_receptacle in trace.target_pos:
                        final_idx = len(trace.eef_pos) - 1
                        static_closed_rec = np.asarray(trace.target_pos[anchor_receptacle][final_idx], dtype=float)
                        if np.all(np.isfinite(static_closed_rec)):
                            static_final_rel = np.asarray(trace.eef_pos[final_idx], dtype=float) - static_closed_rec

                            def drawer_push_start(receptacle: str = anchor_receptacle, rel: np.ndarray = static_final_rel):
                                pos = pose_pos(receptacle)
                                return None if pos is None else pos + rel

                            move_to(
                                drawer_push_start,
                                gripper=-1.0,
                                max_steps=args.scripted_suffix_align_steps,
                                tol_m=args.scripted_hover_tol_m,
                                label="drawer_close:align",
                            )
                            closed_y = float(static_closed_rec[1])
                            for _ in range(int(args.scripted_drawer_close_steps)):
                                rec_pos = pose_pos(anchor_receptacle)
                                if rec_pos is None:
                                    break
                                current = eef_pos_from_obs(obs)
                                if float(rec_pos[1]) >= closed_y - float(args.scripted_drawer_close_tol_m):
                                    break
                                desired = current.copy()
                                desired[0] = float(drawer_push_start()[0]) if drawer_push_start() is not None else float(current[0])
                                desired[1] = float(current[1]) + float(args.scripted_drawer_close_step_m)
                                desired[2] = float(drawer_push_start()[2]) if drawer_push_start() is not None else float(current[2])
                                action = servo_action(
                                    current,
                                    desired,
                                    gripper=-1.0,
                                    gain=float(args.scripted_servo_gain),
                                    max_delta_m=float(args.osc_pos_delta_m),
                                    ori_action=scripted_ori_action(final_idx),
                                )[: env_action_dim(env)]
                                step_once(action, env_step=step_count)
                                if len(scripted_events) < int(args.max_logged_scripted_events):
                                    scripted_events.append(
                                        {
                                            "label": "drawer_close:push",
                                            "step": int(step_count),
                                            "dist_m": round(float(closed_y - float(rec_pos[1])), 5),
                                        }
                                    )
                                if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                                    break

            comp_debug_samples.append(
                {
                    "scripted_pick_place": True,
                    "picked_targets": picked_names,
                    "receptacle_by_object": {name: receptacle_for_object(name) for name in picked_names},
                    "static_prefix_steps": int(scripted_prefix_steps),
                    "static_suffix_steps": int(scripted_suffix_steps),
                    "events": scripted_events,
                }
            )
        else:
            total_steps = min(int(args.max_steps), len(trace.demo.actions) + int(args.extra_steps))
            zero = np.zeros(env_action_dim(env), dtype=np.float32)
            last_base = trace.demo.actions[-1].copy()
            for t in range(total_steps):
                if template_idx < len(trace.demo.actions):
                    base = trace.demo.actions[template_idx].copy()
                else:
                    base = zero.copy()
                    base[-1] = last_base[-1]
                idx = min(template_idx, len(trace.eef_pos) - 1)
                comp_raw, comp_info = compensation_for_step(
                    env,
                    item,
                    trace,
                    target_cfgs,
                    template_idx,
                    near_m=float(args.comp_near_m),
                    far_m=float(args.comp_far_m),
                    max_comp_m=float(args.max_comp_m),
                    disable_picked_after_release=bool(args.disable_picked_target_after_release),
                    release_disable_margin=int(args.release_disable_margin),
                    pick_enable_pre_margin=int(args.pick_enable_pre_margin),
                    horizontal_comp_only=bool(args.horizontal_comp_only),
                )
                alpha = float(args.comp_smoothing_alpha)
                comp_filtered = (1.0 - alpha) * comp_filtered + alpha * comp_raw
                comp_norm = float(np.linalg.norm(comp_filtered))
                max_comp_norm = max(max_comp_norm, comp_norm)
                desired = np.asarray(trace.eef_pos[idx], dtype=float) + comp_filtered
                place_extra = np.asarray(
                    [
                        float(args.template_place_extra_x_m),
                        float(args.template_place_extra_y_m),
                        float(args.template_place_extra_z_m),
                    ],
                    dtype=float,
                )
                place_extra_start = None
                if np.linalg.norm(place_extra) > 0.0:
                    release_steps = [
                        int(phase["release_step"])
                        for phase in trace.target_phase.values()
                        if phase.get("release_step") is not None
                    ]
                    if int(args.template_place_extra_start_step) >= 0:
                        place_extra_start = int(args.template_place_extra_start_step)
                    elif release_steps:
                        place_extra_start = max(0, min(release_steps) - int(args.template_place_extra_pre_release_steps))
                    else:
                        place_extra_start = max(0, len(trace.demo.actions) - int(args.template_place_extra_pre_release_steps))
                    if int(template_idx) >= int(place_extra_start):
                        desired = desired + place_extra
                current = eef_pos_from_obs(obs)
                hold_info = None
                if bool(args.grasp_guard):
                    hold_info = active_grasp_guard_target(
                        env,
                        obs,
                        trace,
                        target_cfgs,
                        template_idx,
                        base,
                        grasp_hold_counts,
                        window=int(args.grasp_guard_window),
                        pre_margin=int(args.grasp_guard_pre_margin),
                        ready_dist_m=float(args.grasp_ready_dist_m),
                        max_hold_steps=int(args.max_grasp_hold_steps),
                    )
                    if hold_info is not None:
                        target_name = hold_info["target"]
                        grasp_hold_counts[target_name] = int(grasp_hold_counts.get(target_name, 0)) + 1
                        base[-1] = -1.0
                        phase = trace.target_phase.get(target_name, {})
                        pick_step = int(phase.get("pick_step", idx))
                        if target_name in trace.target_pos and pick_step < len(trace.eef_pos):
                            static_target_at_pick = np.asarray(trace.target_pos[target_name][pick_step], dtype=float)
                            if np.all(np.isfinite(static_target_at_pick)):
                                static_rel = np.asarray(trace.eef_pos[pick_step], dtype=float) - static_target_at_pick
                                dynamic_target_pos = np.asarray(hold_info["dynamic_target_pos"], dtype=float)
                                prev_pos = grasp_guard_prev_pos.get(target_name)
                                grasp_guard_prev_pos[target_name] = dynamic_target_pos.copy()
                                if prev_pos is not None:
                                    dynamic_target_pos = dynamic_target_pos + float(args.grasp_guard_lead_steps) * (dynamic_target_pos - prev_pos)
                                desired = dynamic_target_pos + static_rel
                                if place_extra_start is not None and int(template_idx) >= int(place_extra_start):
                                    desired = desired + place_extra
                        if len(grasp_hold_events) < int(args.max_logged_grasp_hold_events):
                            grasp_hold_events.append(
                                {
                                    **hold_info,
                                    "env_step": int(t),
                                    "hold_count": int(grasp_hold_counts[target_name]),
                                }
                            )
                if int(args.template_place_hold_gripper_steps) > 0:
                    release_steps = [
                        int(phase["release_step"])
                        for phase in trace.target_phase.values()
                        if phase.get("release_step") is not None
                    ]
                    if release_steps:
                        release_step = min(release_steps)
                        if int(release_step) <= int(template_idx) < int(release_step) + int(args.template_place_hold_gripper_steps):
                            base[-1] = 1.0
                action = command_action(
                    base,
                    current,
                    desired,
                    correction_gain=float(args.correction_gain),
                    max_delta_m=float(args.osc_pos_delta_m),
                )[: env_action_dim(env)]

                step_once(action, env_step=step_count)
                if len(comp_debug_samples) < int(args.comp_debug_samples) and (t % max(1, int(args.comp_debug_stride)) == 0):
                    comp_debug_samples.append(
                        {
                            "step": int(t),
                            "template_idx": int(template_idx),
                            "comp_filtered_xyz": [round(float(x), 5) for x in comp_filtered],
                            "current_eef": [round(float(x), 5) for x in current],
                            "desired_eef": [round(float(x), 5) for x in desired],
                            "grasp_hold": hold_info,
                            "comp": comp_info,
                        }
                    )
                if hold_info is None and template_idx < len(trace.demo.actions):
                    template_idx += 1
                if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                    break
            if bool(args.template_push_object_to_receptacle) and not raw_success:
                push_events: list[dict[str, Any]] = []

                def push_pose(kind: str) -> np.ndarray | None:
                    obj_pose = pose_for_name(env, str(args.template_push_object_name), {"name": str(args.template_push_object_name)})
                    rec_pose = pose_for_name(env, str(args.template_push_receptacle_name), {"name": str(args.template_push_receptacle_name)})
                    if obj_pose is None or rec_pose is None:
                        return None
                    obj_pos = np.asarray(obj_pose[0], dtype=float)
                    rec_pos = np.asarray(rec_pose[0], dtype=float)
                    direction = rec_pos[:2] - obj_pos[:2]
                    norm = float(np.linalg.norm(direction))
                    if norm < 1e-6:
                        direction = np.asarray([0.0, 1.0], dtype=float)
                    else:
                        direction = direction / norm
                    if kind == "end":
                        xy = rec_pos[:2] + direction * float(args.template_push_past_center_m)
                    else:
                        xy = obj_pos[:2] - direction * float(args.template_push_behind_m)
                    z = float(obj_pos[2]) + float(args.template_push_height_m)
                    if kind == "hover":
                        z += float(args.template_push_hover_z_m)
                    return np.asarray([float(xy[0]), float(xy[1]), z], dtype=float)

                def push_move(kind: str, steps: int, tol_m: float) -> None:
                    nonlocal raw_success
                    for _ in range(max(1, int(steps))):
                        target = push_pose(kind)
                        if target is None:
                            break
                        current = eef_pos_from_obs(obs)
                        dist = float(np.linalg.norm(np.asarray(target, dtype=float) - current))
                        action = servo_action(
                            current,
                            np.asarray(target, dtype=float),
                            gripper=-1.0,
                            gain=float(args.template_push_servo_gain),
                            max_delta_m=float(args.osc_pos_delta_m),
                        )[: env_action_dim(env)]
                        step_once(action, env_step=step_count)
                        if len(push_events) < int(args.max_logged_scripted_events):
                            push_events.append({"label": f"template_push:{kind}", "step": int(step_count), "dist_m": round(dist, 5)})
                        if raw_success and bool(args.stop_on_success) and step_count >= int(args.min_steps_before_stop):
                            break
                        if dist <= float(tol_m):
                            break

                push_move("hover", int(args.template_push_setup_steps), float(args.scripted_hover_tol_m))
                if not raw_success:
                    push_move("start", int(args.template_push_setup_steps), float(args.scripted_hover_tol_m))
                if not raw_success:
                    push_move("end", int(args.template_push_steps), float(args.scripted_place_tol_m))
                comp_debug_samples.append(
                    {
                        "template_push_object_to_receptacle": True,
                        "object": str(args.template_push_object_name),
                        "receptacle": str(args.template_push_receptacle_name),
                        "events": push_events,
                    }
                )
        if env is not None:
            try:
                env.sim.forward()
                raw_success = bool(raw_success) or bool(env.check_success())
                if raw_success and done_records:
                    done_records[-1] = True
            except Exception:
                pass
            if not raw_success and state_records and action_records:
                checker_env = None
                old_dynamic_config = os.environ.get("LIBERO_DYNAMIC_CONFIG")
                try:
                    os.environ.pop("LIBERO_DYNAMIC_CONFIG", None)
                    checker_env = make_env(bddl, camera_size=int(args.camera_size), control_freq=int(args.control_freq))
                    checker_env.reset()
                    for state_idx in range(len(state_records) - 1, 0, -1):
                        try:
                            checker_env.set_init_state(np.asarray(state_records[state_idx], dtype=np.float64))
                            checker_env.sim.forward()
                            if not bool(checker_env.check_success()):
                                continue
                        except Exception:
                            continue
                        posthoc_success_state_index = int(state_idx)
                        raw_success = True
                        action_records = action_records[:state_idx]
                        state_records = state_records[:state_idx]
                        reward_records = reward_records[:state_idx]
                        done_records = done_records[:state_idx]
                        for record_key in list(obs_records.keys()):
                            obs_records[record_key] = obs_records[record_key][:state_idx]
                        if done_records:
                            done_records[-1] = True
                        step_count = int(len(action_records))
                        break
                finally:
                    if checker_env is not None:
                        checker_env.close()
                    if old_dynamic_config is None:
                        os.environ.pop("LIBERO_DYNAMIC_CONFIG", None)
                    else:
                        os.environ["LIBERO_DYNAMIC_CONFIG"] = old_dynamic_config
            motion_samples.append(capture_motion_evidence_sample(env, target_cfgs))
        diagnostic_summary = summarize_diagnostics(diagnostics, args)
        for cfg in target_cfgs:
            name = str(cfg.get("name", ""))
            pose = pose_for_name(env, name, cfg)
            if name and pose is not None:
                final_target_poses[name] = {
                    "position": np.asarray(pose[0], dtype=float).round(6).tolist(),
                    "quaternion": np.asarray(pose[1], dtype=float).round(6).tolist(),
                }
    except Exception as exc:
        status = "error"
        error = repr(exc)
        error_traceback = traceback.format_exc()
        diagnostic_summary = {"targets": {}, "flags": [{"type": "exception", "value": error}], "ok": False}
    finally:
        if agentview_writer is not None:
            agentview_writer.close()
        if eye_in_hand_writer is not None:
            eye_in_hand_writer.close()
        if env is not None:
            env.close()

    if action_records and state_records:
        episode_arrays: dict[str, Any] = {
            "actions": np.asarray(action_records, dtype=np.float32),
            "states": np.asarray(state_records, dtype=np.float64),
            "rewards": np.asarray(reward_records, dtype=np.float32),
            "dones": np.asarray(done_records, dtype=np.bool_),
            "item_id": np.asarray(item_id),
        }
        if init_state_record is not None:
            episode_arrays["init_state"] = np.asarray(init_state_record, dtype=np.float64)
        if model_file_record is not None:
            episode_arrays["model_file"] = np.asarray(model_file_record)
        for record_key, values in obs_records.items():
            if len(values) == len(action_records):
                dtype = np.uint8 if record_key.endswith("_rgb") else np.float64
                episode_arrays[record_key] = np.asarray(values, dtype=dtype)
        np.savez_compressed(
            episode_npz,
            **episode_arrays,
        )
    motion_evidence = summarize_motion_evidence(
        motion_samples,
        carrier_type=carrier_type,
        task_cfg=item.get("task_cfg", {}) or {},
    )
    strict, strict_reasons = strict_success(
        {
            "status": status,
            "raw_success": bool(raw_success),
            "assist_steps": 0,
            "diagnostics": diagnostic_summary,
            "used_forbidden_state_edit": False,
            "dynamic_config": {"carrier_type": carrier_type},
            "motion_evidence": motion_evidence,
        }
    )
    row = {
        **{k: item[k] for k in ("item_id", "suite", "task_id", "level", "traj_index", "variant_name", "variant_id")},
        "status": status,
        "error": error,
        "error_traceback": error_traceback,
        "language": getattr(task, "language", ""),
        "generation_script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "attempt_label": str(getattr(args, "attempt_label", "primary")),
        "seed": int(args.seed),
        "init_state": {
            "source": "official_static_demo_hdf5_state0",
            "demo_path": str(trace.demo.demo_path),
            "demo_key": trace.demo.demo_key,
            "state_index": 0,
        },
        "dynamic_config": {
            "level": item["level"],
            "traj_index": int(item["traj_index"]),
            "variant_id": item["variant_id"],
            "effective_config_path": str(item["config_path"]),
            "target_names": [cfg["name"] for cfg in target_cfgs],
            "carrier_type": carrier_type,
            "augmentation": item.get("augmentation", {}),
            "base_item_id": item.get("base_item_id", item.get("item_id")),
        },
        "raw_success": bool(raw_success),
        "strict_physical_success": bool(strict),
        "strict_failure_reasons": strict_reasons,
        "assist_steps": 0,
        "used_forbidden_state_edit": False,
        "diagnostics": diagnostic_summary,
        "motion_evidence": motion_evidence,
        "final_target_poses": final_target_poses,
        "steps": int(step_count),
        "posthoc_success_state_index": posthoc_success_state_index,
        "template_steps_used": int(template_idx),
        "static_template": {
            "demo_path": str(trace.demo.demo_path),
            "demo_key": trace.demo.demo_key,
            "static_success_step": int(trace.static_success_step),
            "template_length": int(len(trace.demo.actions)),
        },
        "controller": {
            "attempt_label": str(getattr(args, "attempt_label", "primary")),
            "name": controller_name,
            "dynamic_eval_route": "preview_compatible" if use_preview_compatible_eval else "env_wrapper",
            "correction_gain": float(args.correction_gain),
            "comp_near_m": float(args.comp_near_m),
            "comp_far_m": float(args.comp_far_m),
            "max_comp_m": float(args.max_comp_m),
            "horizontal_comp_only": bool(args.horizontal_comp_only),
            "comp_smoothing_alpha": float(args.comp_smoothing_alpha),
            "max_comp_norm_m": round(float(max_comp_norm), 6),
            "target_phase": trace.target_phase,
            "grasp_guard": {
                "enabled": bool(args.grasp_guard),
                "hold_counts": grasp_hold_counts,
                "events": grasp_hold_events,
                "ready_dist_m": float(args.grasp_ready_dist_m),
                "window": int(args.grasp_guard_window),
            },
            "debug_samples": comp_debug_samples,
            "phase_prime": {
                "seconds": round(float(phase_prime_seconds), 6),
                "steps": int(phase_prime_steps),
            },
        },
        "episode_npz": str(episode_npz) if episode_npz.exists() else None,
        "packed_hdf5_demo_key": None,
        "video_path": str(video_path) if video_path.exists() else None,
        "video_paths": {
            "agentview_rgb": str(video_path) if video_path.exists() else None,
            "eye_in_hand_rgb": str(eye_in_hand_video_path) if eye_in_hand_video_path.exists() else None,
        },
        "review": {
            "review_html": str(args.output_dir / "review" / "index.html"),
            "manual_review_status": "pending_review_panel",
            "model_input_views": ["agentview_rgb", "eye_in_hand_rgb"],
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "resume_skipped": False,
    }
    write_json(manifest_path, row)
    return row


def rewrite_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: r.get("item_id", ""))
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "total": len(rows),
        "status_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "raw_success": sum(1 for row in rows if row.get("raw_success")),
        "strict_physical_success": sum(1 for row in rows if row.get("strict_physical_success")),
        "diagnostics_ok": sum(1 for row in rows if (row.get("diagnostics") or {}).get("ok")),
    }
    write_json(output_dir / "summary.json", summary)


def pack_hdf5(output_dir: Path, rows: list[dict[str, Any]], hdf5_name: str) -> tuple[Path, list[dict[str, Any]]]:
    strict_rows = [row for row in sorted(rows, key=lambda r: r.get("item_id", "")) if row.get("strict_physical_success")]
    hdf5_path = output_dir / hdf5_name
    if hdf5_path.exists():
        hdf5_path.unlink()
    with h5py.File(hdf5_path, "w") as f:
        data = f.create_group("data")
        first_row = strict_rows[0] if strict_rows else (rows[0] if rows else None)
        if first_row is not None:
            try:
                _suite, task, bddl = task_bddl(str(first_row["suite"]), int(first_row["task_id"]))
                official_path = dataset_path_for_task(str(first_row["suite"]), task)
                if official_path.exists():
                    with h5py.File(official_path, "r") as src:
                        for attr_name in (
                            "bddl_file_name",
                            "env_args",
                            "env_name",
                            "macros_image_convention",
                            "problem_info",
                            "tag",
                        ):
                            if attr_name in src["data"].attrs:
                                data.attrs[attr_name] = src["data"].attrs[attr_name]
                    data.attrs["source_official_demo_hdf5"] = str(official_path)
                else:
                    data.attrs["bddl_file_name"] = str(bddl)
            except Exception as exc:
                data.attrs["official_attr_copy_error"] = repr(exc)
        data.attrs["pack_type"] = "mixed_task_libero_dynamic_demonstrations"
        data.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data.attrs["source_manifest"] = str(output_dir / "manifest.jsonl")
        if "tag" not in data.attrs:
            data.attrs["tag"] = "libero-v1"
        if "macros_image_convention" not in data.attrs:
            data.attrs["macros_image_convention"] = "opengl"
        packed_idx = 0
        total_samples = 0
        for row in strict_rows:
            npz_path = Path(row["episode_npz"])
            if not npz_path.exists():
                row["strict_physical_success"] = False
                row.setdefault("strict_failure_reasons", []).append("missing_episode_npz_at_pack")
                continue
            with np.load(npz_path, allow_pickle=False) as npz:
                required_keys = [
                    "actions",
                    "states",
                    "rewards",
                    "dones",
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "ee_pos",
                    "ee_ori",
                    "ee_states",
                    "gripper_states",
                    "joint_states",
                    "robot_states",
                ]
                missing_keys = [key for key in required_keys if key not in npz]
                if missing_keys:
                    row["strict_physical_success"] = False
                    row.setdefault("strict_failure_reasons", []).append(
                        "missing_official_obs_keys_at_pack:" + ",".join(missing_keys)
                    )
                    continue
                actions = np.asarray(npz["actions"], dtype=np.float64)
                num_samples = int(actions.shape[0])
                length_mismatches = [
                    key
                    for key in required_keys
                    if int(np.asarray(npz[key]).shape[0]) != num_samples
                ]
                if length_mismatches:
                    row["strict_physical_success"] = False
                    row.setdefault("strict_failure_reasons", []).append(
                        "official_obs_length_mismatch_at_pack:" + ",".join(length_mismatches)
                    )
                    continue
                demo_key = f"demo_{packed_idx}"
                row["packed_hdf5_demo_key"] = demo_key
                g = data.create_group(demo_key)
                obs_grp = g.create_group("obs")
                obs_grp.create_dataset("agentview_rgb", data=np.asarray(npz["agentview_rgb"], dtype=np.uint8), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.asarray(npz["eye_in_hand_rgb"], dtype=np.uint8), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("ee_pos", data=np.asarray(npz["ee_pos"], dtype=np.float64), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("ee_ori", data=np.asarray(npz["ee_ori"], dtype=np.float64), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("ee_states", data=np.asarray(npz["ee_states"], dtype=np.float64), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("gripper_states", data=np.asarray(npz["gripper_states"], dtype=np.float64), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("joint_states", data=np.asarray(npz["joint_states"], dtype=np.float64), compression="gzip", compression_opts=4)
                g.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)
                g.create_dataset("states", data=np.asarray(npz["states"], dtype=np.float64), compression="gzip", compression_opts=4)
                g.create_dataset("rewards", data=np.asarray(npz["rewards"], dtype=np.float32), compression="gzip", compression_opts=4)
                g.create_dataset("dones", data=np.asarray(npz["dones"], dtype=np.uint8), compression="gzip", compression_opts=4)
                g.create_dataset("robot_states", data=np.asarray(npz["robot_states"], dtype=np.float64), compression="gzip", compression_opts=4)
                g.attrs["num_samples"] = num_samples
                if "init_state" in npz:
                    g.attrs["init_state"] = np.asarray(npz["init_state"], dtype=np.float64)
                if "model_file" in npz:
                    g.attrs["model_file"] = str(np.asarray(npz["model_file"]).item())
                g.attrs["item_id"] = row["item_id"]
                g.attrs["suite"] = row["suite"]
                g.attrs["task_id"] = int(row["task_id"])
                g.attrs["level"] = row["level"]
                g.attrs["traj_index"] = int(row["traj_index"])
                g.attrs["metadata_json"] = json.dumps(row, ensure_ascii=False)
                packed_idx += 1
                total_samples += num_samples
        data.attrs["num_demos"] = int(packed_idx)
        data.attrs["total"] = int(total_samples)
    return hdf5_path, rows


def write_audit(output_dir: Path, rows: list[dict[str, Any]], hdf5_path: Path | None) -> dict[str, Any]:
    failures = Counter()
    for row in rows:
        if not row.get("strict_physical_success"):
            for reason in row.get("strict_failure_reasons") or ["unknown"]:
                failures[str(reason)] += 1
    audit = {
        "total_items": len(rows),
        "raw_success": sum(1 for row in rows if row.get("raw_success")),
        "strict_physical_success": sum(1 for row in rows if row.get("strict_physical_success")),
        "status_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "diagnostics_ok": sum(1 for row in rows if (row.get("diagnostics") or {}).get("ok")),
        "assist_steps_total": sum(int(row.get("assist_steps", 0)) for row in rows),
        "failure_reasons": dict(failures),
        "manual_review_status": Counter((row.get("review") or {}).get("manual_review_status", "unknown") for row in rows),
        "packed_hdf5": str(hdf5_path) if hdf5_path is not None else None,
        "strict_item_ids": [row["item_id"] for row in rows if row.get("strict_physical_success")],
        "failed_items": [
            {
                "item_id": row.get("item_id"),
                "raw_success": row.get("raw_success"),
                "strict_physical_success": row.get("strict_physical_success"),
                "reasons": row.get("strict_failure_reasons"),
                "diagnostic_flags": (row.get("diagnostics") or {}).get("flags", []),
                "error": row.get("error"),
            }
            for row in rows
            if not row.get("strict_physical_success")
        ],
    }
    write_json(output_dir / "strict_audit.json", audit)
    return audit


def write_review_html(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.get("item_id", ""))
    cards = []
    for row in rows:
        video_paths = row.get("video_paths") if isinstance(row.get("video_paths"), dict) else {}
        agentview_video = video_paths.get("agentview_rgb") or row.get("video_path")
        eye_in_hand_video = video_paths.get("eye_in_hand_rgb")
        rel_agentview = os.path.relpath(agentview_video, review_dir) if agentview_video else ""
        rel_eye_in_hand = os.path.relpath(eye_in_hand_video, review_dir) if eye_in_hand_video else ""
        video_html = f"""
  <div class="videos">
    <figure>
      <figcaption>agentview_rgb</figcaption>
      {f'<video src="{html.escape(rel_agentview)}" controls preload="metadata"></video>' if rel_agentview else '<div class="missing">no agentview video</div>'}
    </figure>
    <figure>
      <figcaption>eye_in_hand_rgb</figcaption>
      {f'<video src="{html.escape(rel_eye_in_hand)}" controls preload="metadata"></video>' if rel_eye_in_hand else '<div class="missing">no eye_in_hand video</div>'}
    </figure>
  </div>
"""
        flags = ", ".join(str(flag.get("type")) for flag in (row.get("diagnostics") or {}).get("flags", []))
        cards.append(
            f"""
<article class="card" data-suite="{html.escape(str(row.get('suite')))}" data-level="{html.escape(str(row.get('level')))}" data-raw="{str(bool(row.get('raw_success'))).lower()}" data-strict="{str(bool(row.get('strict_physical_success'))).lower()}">
  <div class="meta">
    <strong>{html.escape(str(row.get('item_id')))}</strong>
    <span>{html.escape(str(row.get('language', '')))}</span>
    <span>raw={row.get('raw_success')} strict={row.get('strict_physical_success')} steps={row.get('steps')}</span>
    <span>{html.escape(flags or 'diagnostics ok')}</span>
  </div>
  {video_html}
</article>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO-Dynamic Review</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f6f7f9; color: #15171a; }}
header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d8dde5; padding: 12px 18px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
h1 {{ font-size: 18px; margin: 0 16px 0 0; }}
select, input {{ height: 32px; border: 1px solid #b8c0cc; border-radius: 6px; padding: 0 8px; background: white; }}
main {{ padding: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }}
.card {{ background: #fff; border: 1px solid #dce1e8; border-radius: 8px; overflow: hidden; }}
.meta {{ display: grid; gap: 4px; padding: 10px 12px; font-size: 12px; }}
.meta strong {{ font-size: 13px; }}
.videos {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #dce1e8; }}
figure {{ margin: 0; background: #fff; }}
figcaption {{ padding: 6px 8px; font-size: 11px; color: #3b4350; }}
video {{ display: block; width: 100%; background: #111; aspect-ratio: 1 / 1; }}
.missing {{ aspect-ratio: 1 / 1; display: grid; place-items: center; color: #777; font-size: 12px; background: #f0f2f5; }}
@media (max-width: 640px) {{
  main {{ grid-template-columns: 1fr; }}
  .videos {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>LIBERO-Dynamic Review</h1>
  <input id="q" placeholder="filter item/task">
  <select id="suite"><option value="">all suites</option><option>libero_10</option><option>libero_goal</option><option>libero_object</option><option>libero_spatial</option></select>
  <select id="level"><option value="">all levels</option><option>l1</option><option>l2</option><option>l3</option></select>
  <select id="strict"><option value="">strict any</option><option value="true">strict true</option><option value="false">strict false</option></select>
  <select id="raw"><option value="">raw any</option><option value="true">raw true</option><option value="false">raw false</option></select>
</header>
<main>
{''.join(cards)}
</main>
<script>
const controls = ["q", "suite", "level", "strict", "raw"].map(id => document.getElementById(id));
function applyFilters() {{
  const q = document.getElementById("q").value.toLowerCase();
  const suite = document.getElementById("suite").value;
  const level = document.getElementById("level").value;
  const strict = document.getElementById("strict").value;
  const raw = document.getElementById("raw").value;
  for (const card of document.querySelectorAll(".card")) {{
    const text = card.innerText.toLowerCase();
    const ok = (!q || text.includes(q)) && (!suite || card.dataset.suite === suite) && (!level || card.dataset.level === level) && (!strict || card.dataset.strict === strict) && (!raw || card.dataset.raw === raw);
    card.style.display = ok ? "" : "none";
  }}
}}
controls.forEach(c => c.addEventListener("input", applyFilters));
</script>
</body>
</html>
"""
    path = review_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "libero" / "libero" / "dynamic_configs")
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--item-ids", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-quality", type=int, default=10)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--model-input-video-size", type=int, default=0)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-speed-mps", type=float, default=None)
    parser.add_argument("--phase-offset-ratio", type=float, default=0.0)
    parser.add_argument("--exact-benchmark-phase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--benchmark-init-state-id", type=int, default=0)
    parser.add_argument("--benchmark-num-steps-wait", type=int, default=10)
    parser.add_argument("--temporal-compensation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-comp-ramp-steps", type=int, default=40)
    parser.add_argument("--temporal-comp-full-pre-steps", type=int, default=12)
    parser.add_argument("--temporal-comp-full-post-steps", type=int, default=8)
    parser.add_argument("--temporal-comp-fade-steps", type=int, default=12)
    parser.add_argument("--benchmark-pre-policy-dynamic-steps", type=int, default=40)
    parser.add_argument("--scripted-drop-above-receptacle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--item-id-suffix", default="")
    parser.add_argument("--freeze-receptacle-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-demo-search", type=int, default=50)
    parser.add_argument("--static-demo-candidate-scan", type=int, default=8)
    parser.add_argument("--static-demo-key-override", default=None)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--extra-steps", type=int, default=140)
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-steps-before-stop", type=int, default=20)
    parser.add_argument("--correction-gain", type=float, default=0.72)
    parser.add_argument("--osc-pos-delta-m", type=float, default=0.05)
    parser.add_argument("--comp-near-m", type=float, default=0.11)
    parser.add_argument("--comp-far-m", type=float, default=0.42)
    parser.add_argument("--max-comp-m", type=float, default=0.22)
    parser.add_argument("--horizontal-comp-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comp-smoothing-alpha", type=float, default=0.32)
    parser.add_argument("--template-place-extra-x-m", type=float, default=0.0)
    parser.add_argument("--template-place-extra-y-m", type=float, default=0.0)
    parser.add_argument("--template-place-extra-z-m", type=float, default=0.0)
    parser.add_argument("--template-place-extra-start-step", type=int, default=-1)
    parser.add_argument("--template-place-extra-pre-release-steps", type=int, default=20)
    parser.add_argument("--template-place-hold-gripper-steps", type=int, default=0)
    parser.add_argument("--template-push-object-to-receptacle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--template-push-object-name", default="akita_black_bowl_1")
    parser.add_argument("--template-push-receptacle-name", default="plate_1")
    parser.add_argument("--template-push-behind-m", type=float, default=0.045)
    parser.add_argument("--template-push-past-center-m", type=float, default=0.01)
    parser.add_argument("--template-push-height-m", type=float, default=-0.005)
    parser.add_argument("--template-push-hover-z-m", type=float, default=0.08)
    parser.add_argument("--template-push-setup-steps", type=int, default=45)
    parser.add_argument("--template-push-steps", type=int, default=80)
    parser.add_argument("--template-push-servo-gain", type=float, default=1.4)
    parser.add_argument("--comp-debug-samples", type=int, default=12)
    parser.add_argument("--comp-debug-stride", type=int, default=20)
    parser.add_argument("--disable-picked-target-after-release", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-disable-margin", type=int, default=8)
    parser.add_argument("--pick-enable-pre-margin", type=int, default=28)
    parser.add_argument("--grasp-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grasp-ready-dist-m", type=float, default=0.055)
    parser.add_argument("--grasp-guard-window", type=int, default=22)
    parser.add_argument("--grasp-guard-pre-margin", type=int, default=2)
    parser.add_argument("--max-grasp-hold-steps", type=int, default=35)
    parser.add_argument("--grasp-guard-lead-steps", type=float, default=0.0)
    parser.add_argument("--max-logged-grasp-hold-events", type=int, default=20)
    parser.add_argument("--controller-mode", choices=("auto", "template", "scripted_pick_place"), default="template")
    parser.add_argument("--disable-primary-overrides", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-servo-gain", type=float, default=1.5)
    parser.add_argument(
        "--scripted-goal-object-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use BDDL goal objects rather than support objects inferred from the static trace.",
    )
    parser.add_argument("--scripted-hover-z-m", type=float, default=0.10)
    parser.add_argument("--scripted-pregrasp-detour-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-pregrasp-detour-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-pregrasp-detour-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-pregrasp-detour-transition-stages", type=int, default=0)
    parser.add_argument("--scripted-lift-z-m", type=float, default=0.14)
    parser.add_argument("--scripted-move-steps", type=int, default=50)
    parser.add_argument("--scripted-gripper-steps", type=int, default=18)
    parser.add_argument("--scripted-contact-retry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-contact-retry-attempts", type=int, default=5)
    parser.add_argument("--scripted-contact-retry-offset-m", type=float, default=0.035)
    parser.add_argument("--scripted-contact-retry-z-m", type=float, default=-0.02)
    parser.add_argument("--scripted-contact-retry-move-steps", type=int, default=45)
    parser.add_argument("--scripted-contact-retry-gripper-steps", type=int, default=25)
    parser.add_argument("--scripted-release-steps", type=int, default=35)
    parser.add_argument("--scripted-hover-tol-m", type=float, default=0.035)
    parser.add_argument("--scripted-grasp-tol-m", type=float, default=0.035)
    parser.add_argument("--scripted-place-tol-m", type=float, default=0.022)
    parser.add_argument("--scripted-place-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-place-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-place-extra-z-m", type=float, default=0.035)
    parser.add_argument("--scripted-moka-stove-place-adjust", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-place-adjust-y-m", type=float, default=-0.16)
    parser.add_argument("--scripted-moka-stove-object-servo-place", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-pot1-object-target-x-m", type=float, default=0.004)
    parser.add_argument("--scripted-moka-stove-pot1-object-target-y-m", type=float, default=0.052)
    parser.add_argument("--scripted-moka-stove-pot1-object-target-z-m", type=float, default=0.099)
    parser.add_argument("--scripted-moka-stove-pot2-object-target-x-m", type=float, default=-0.036)
    parser.add_argument("--scripted-moka-stove-pot2-object-target-y-m", type=float, default=-0.033)
    parser.add_argument("--scripted-moka-stove-pot2-object-target-z-m", type=float, default=0.094)
    parser.add_argument("--scripted-moka-stove-pot1-grasp-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot1-grasp-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot1-grasp-extra-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-grasp-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-grasp-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-grasp-extra-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot1-place-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot1-place-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot1-place-extra-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-place-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-place-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-pot2-place-extra-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-repick-pot2-object-servo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-push-pot1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-pot1-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-defer-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-push-setup-steps", type=int, default=40)
    parser.add_argument("--scripted-moka-stove-push-steps", type=int, default=70)
    parser.add_argument("--scripted-moka-stove-push-passes", type=int, default=1)
    parser.add_argument("--scripted-moka-stove-push-hover-side-offset-m", type=float, default=0.12)
    parser.add_argument("--scripted-moka-stove-push-side-offset-m", type=float, default=0.06)
    parser.add_argument("--scripted-moka-stove-push-hover-height-m", type=float, default=0.12)
    parser.add_argument("--scripted-moka-stove-push-low-height-m", type=float, default=0.025)
    parser.add_argument("--scripted-moka-stove-push-direct-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-push-gripper", type=float, default=-1.0)
    parser.add_argument("--scripted-moka-stove-after-radial-straight-y-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-after-radial-straight-y-pot2-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-after-radial-straight-y-target-offset-m", type=float, default=-0.033)
    parser.add_argument("--scripted-moka-stove-after-radial-straight-y-steps", type=int, default=120)
    parser.add_argument("--scripted-moka-stove-after-radial-straight-y-passes", type=int, default=1)
    parser.add_argument("--scripted-moka-stove-straight-y-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-straight-x-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-moka-stove-straight-y-target-offset-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-push-past-center-m", type=float, default=0.16)
    parser.add_argument("--scripted-moka-stove-push-pot1-target-x-m", type=float, default=0.06)
    parser.add_argument("--scripted-moka-stove-push-target-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-moka-stove-push-target-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-object-servo-place", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-object-hover-z-m", type=float, default=0.08)
    parser.add_argument("--scripted-object-place-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-object-place-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-object-place-extra-z-m", type=float, default=0.0)
    parser.add_argument("--scripted-grasp-extra-x-m", type=float, default=0.0)
    parser.add_argument("--scripted-grasp-extra-y-m", type=float, default=0.0)
    parser.add_argument("--scripted-grasp-extra-z-m", type=float, default=-0.015)
    parser.add_argument("--scripted-orientation-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-receptacle-center-bias", type=float, default=0.5)
    parser.add_argument("--scripted-ignore-goal-receptacle-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-static-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-force-static-prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-prefix-pre-pick-margin", type=int, default=20)
    parser.add_argument("--scripted-initial-wait-steps", type=int, default=0)
    parser.add_argument("--scripted-static-suffix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-suffix-start-offset", type=int, default=0)
    parser.add_argument("--scripted-align-static-suffix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-suffix-align-steps", type=int, default=40)
    parser.add_argument("--scripted-drawer-close-push", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-drawer-close-steps", type=int, default=80)
    parser.add_argument("--scripted-drawer-close-step-m", type=float, default=0.018)
    parser.add_argument("--scripted-drawer-close-tol-m", type=float, default=0.012)
    parser.add_argument("--scripted-plate-front-push", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scripted-plate-push-setup-steps", type=int, default=55)
    parser.add_argument("--scripted-plate-push-steps", type=int, default=130)
    parser.add_argument("--scripted-plate-push-behind-m", type=float, default=0.075)
    parser.add_argument("--scripted-plate-push-past-m", type=float, default=0.035)
    parser.add_argument("--scripted-plate-push-height-m", type=float, default=0.018)
    parser.add_argument("--scripted-plate-push-target-x-offset-m", type=float, default=0.0)
    parser.add_argument("--scripted-plate-push-target-y-offset-m", type=float, default=0.0)
    parser.add_argument("--max-logged-scripted-events", type=int, default=80)
    parser.add_argument("--keep-static-support-contact", action="store_true", default=True)
    parser.add_argument("--static-support-clearance-m", type=float, default=0.001)
    parser.add_argument("--preview-compatible-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-preview-compatible-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preview-compatible-scope", choices=("fragile", "all"), default="all")
    parser.add_argument("--preview-compatible-settle-steps", type=int, default=80)
    parser.add_argument("--platform-settle-steps", type=int, default=20)
    parser.add_argument("--realign-after-platform-settle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-alignment-settle-steps", type=int, default=20)
    parser.add_argument("--platform-error-threshold-m", type=float, default=0.01)
    parser.add_argument("--follow-error-threshold-m", type=float, default=0.08)
    parser.add_argument("--tilt-threshold-deg", type=float, default=35.0)
    parser.add_argument("--drop-threshold-m", type=float, default=0.04)
    parser.add_argument("--bottom-baseline-mode", choices=("first_sample", "reset"), default="first_sample")
    parser.add_argument("--offscreen-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--pack-hdf5-name", default="final_600.hdf5")
    parser.add_argument("--fallback-preview-compatible", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fallback-controller-mode", choices=("auto", "template", "scripted_pick_place"), default="auto")
    parser.add_argument("--fallback-cascade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isolate-items", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--worker-no-finalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rerun-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pack-hdf5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pack-existing-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_no_dynamic_config()
    np.random.seed(int(args.seed))

    command_file = args.output_dir / "run_command.txt"
    command_file.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), args.output_dir / "generate_dynamic_demos.py")

    if bool(args.pack_existing_only):
        rows = list({row["item_id"]: row for row in load_jsonl(args.output_dir / "manifest.jsonl")}.values())
        hdf5_path = None
        if rows and bool(args.pack_hdf5):
            hdf5_path, rows = pack_hdf5(args.output_dir, rows, args.pack_hdf5_name)
            rewrite_manifest(args.output_dir, rows)
        audit = write_audit(args.output_dir, rows, hdf5_path)
        review_path = write_review_html(args.output_dir, rows)
        print(json.dumps({"audit": audit, "review": str(review_path)}, ensure_ascii=False, indent=2), flush=True)
        return 0 if rows and audit.get("strict_physical_success", 0) == len(rows) else 2

    config_path_by_level = write_effective_config_files(args.config_dir.resolve(), args.output_dir, args.levels, args)
    if not bool(getattr(args, "exact_benchmark_phase", False)):
        normalize_stable_platform_contact_in_config_files(config_path_by_level)
    effective_config_dir = args.output_dir / "effective_dynamic_configs"
    items = build_items(
        effective_config_dir.resolve(),
        args.levels,
        args.traj_indices,
        keep_static_support_contact=bool(args.keep_static_support_contact),
        static_support_clearance_m=float(args.static_support_clearance_m),
        config_path_by_level=config_path_by_level,
    )
    if args.item_ids:
        wanted = set(args.item_ids)
        items = [item for item in items if item["item_id"] in wanted]
    elif args.panel:
        items = choose_panel_items(items)
    else:
        if args.start_index:
            items = items[int(args.start_index) :]
        if args.limit is not None:
            items = items[: int(args.limit)]
    items = apply_item_augmentation(items, args)
    write_json(args.output_dir / "items.json", items)

    rows_by_id = {row["item_id"]: row for row in load_jsonl(args.output_dir / "manifest.jsonl")}
    if bool(args.isolate_items) and len(items) > 1:
        for idx, item in enumerate(items, start=1):
            existing = rows_by_id.get(item["item_id"])
            if existing is not None and not bool(args.overwrite):
                existing_npz_value = existing.get("episode_npz")
                existing_video_value = existing.get("video_path")
                existing_video_paths = existing.get("video_paths") if isinstance(existing.get("video_paths"), dict) else {}
                existing_npz = Path(existing_npz_value) if existing_npz_value else None
                existing_video = Path(existing_video_value) if bool(args.save_videos) and existing_video_value else None
                existing_eye_video_value = existing_video_paths.get("eye_in_hand_rgb")
                existing_eye_video = Path(existing_eye_video_value) if bool(args.save_videos) and existing_eye_video_value else None
                existing_has_official_obs = False
                if existing_npz is not None and existing_npz.exists():
                    try:
                        with np.load(existing_npz, allow_pickle=False) as existing_npz_file:
                            existing_has_official_obs = all(
                                key in existing_npz_file
                                for key in ("agentview_rgb", "eye_in_hand_rgb", "gripper_states", "joint_states", "robot_states")
                            )
                    except Exception:
                        existing_has_official_obs = False
                files_available = (
                    existing_npz is not None
                    and existing_npz.exists()
                    and existing_has_official_obs
                    and (not bool(args.save_videos) or (existing_video is not None and existing_video.exists()))
                    and (not bool(args.save_videos) or (existing_eye_video is not None and existing_eye_video.exists()))
                )
                existing_strict, existing_reasons = strict_success(existing)
                should_reuse = files_available and (not bool(args.rerun_failed) or existing_strict)
                if should_reuse:
                    existing["resume_skipped"] = True
                    existing["strict_physical_success"] = bool(existing_strict)
                    existing["strict_failure_reasons"] = existing_reasons
                    rows_by_id[existing["item_id"]] = existing
                    rewrite_manifest(args.output_dir, list(rows_by_id.values()))
                    print(
                        f"[{idx}/{len(items)}] {item['item_id']} (isolated, resume skipped) "
                        f"raw={existing.get('raw_success')} strict={existing.get('strict_physical_success')}",
                        flush=True,
                    )
                    continue
            print(f"[{idx}/{len(items)}] {item['item_id']} (isolated)", flush=True)
            child_output_dir = args.output_dir / "_isolated_workers" / item["level"] / item["suite"] / item["item_id"]
            if child_output_dir.exists():
                shutil.rmtree(child_output_dir)
            child_cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--output-dir",
                str(child_output_dir),
                "--item-ids",
                str(item.get("base_item_id") or item["item_id"]),
                "--worker-no-finalize",
            ]
            completed = subprocess.run(child_cmd, cwd=str(REPO_ROOT))
            child_item_dir = child_output_dir / "episodes" / item["level"] / item["suite"] / item["item_id"]
            manifest_path = child_item_dir / "manifest.json"
            if manifest_path.exists():
                row = json.loads(manifest_path.read_text(encoding="utf-8"))
                parent_item_dir = args.output_dir / "episodes" / item["level"] / item["suite"] / item["item_id"]
                if parent_item_dir.exists():
                    shutil.rmtree(parent_item_dir)
                parent_item_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(child_item_dir, parent_item_dir)

                def remap_child_output_paths(value: Any) -> Any:
                    if isinstance(value, str):
                        return value.replace(str(child_output_dir), str(args.output_dir))
                    if isinstance(value, list):
                        return [remap_child_output_paths(v) for v in value]
                    if isinstance(value, dict):
                        return {k: remap_child_output_paths(v) for k, v in value.items()}
                    return value

                row = remap_child_output_paths(row)
                row["episode_npz"] = str(parent_item_dir / "episode.npz")
                row["video_path"] = str(parent_item_dir / "agentview_rgb.mp4")
                row["video_paths"] = {
                    "agentview_rgb": str(parent_item_dir / "agentview_rgb.mp4"),
                    "eye_in_hand_rgb": str(parent_item_dir / "eye_in_hand_rgb.mp4"),
                }
                if isinstance(row.get("review"), dict):
                    row["review"]["review_html"] = str(args.output_dir / "review" / "index.html")
                strict, reasons = strict_success(row)
                row["strict_physical_success"] = bool(strict)
                row["strict_failure_reasons"] = reasons
                write_json(parent_item_dir / "manifest.json", row)
            else:
                row = {
                    **{k: item[k] for k in ("item_id", "suite", "task_id", "level", "traj_index", "variant_name", "variant_id")},
                    "status": "error",
                    "error": f"isolated child exited {completed.returncode} without manifest",
                    "raw_success": False,
                    "strict_physical_success": False,
                    "strict_failure_reasons": ["missing_manifest"],
                    "assist_steps": 0,
                    "used_forbidden_state_edit": False,
                }
            shutil.rmtree(child_output_dir, ignore_errors=True)
            rows_by_id[row["item_id"]] = row
            rewrite_manifest(args.output_dir, list(rows_by_id.values()))
            print(
                f"  isolated_exit={completed.returncode} status={row.get('status')} raw={row.get('raw_success')} "
                f"strict={row.get('strict_physical_success')} steps={row.get('steps')} "
                f"flags={(row.get('diagnostics') or {}).get('flags', [])}",
                flush=True,
            )
        write_json(args.output_dir / "items.json", items)
        rows = list(rows_by_id.values())
        hdf5_path = None
        if rows and bool(args.pack_hdf5):
            hdf5_path, rows = pack_hdf5(args.output_dir, rows, args.pack_hdf5_name)
            rewrite_manifest(args.output_dir, rows)
        audit = write_audit(args.output_dir, rows, hdf5_path)
        review_path = write_review_html(args.output_dir, rows)
        print(json.dumps({"audit": audit, "review": str(review_path)}, ensure_ascii=False, indent=2), flush=True)
        return 0 if audit.get("strict_physical_success", 0) == len(rows) else 2

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item['item_id']}", flush=True)
        args.attempt_label = "primary"
        row = run_dynamic_item(item, args)
        strict, reasons = strict_success(row)
        row["strict_physical_success"] = bool(strict)
        row["strict_failure_reasons"] = reasons
        if (
            bool(args.fallback_preview_compatible)
            and not bool(strict)
            and not bool(args.force_preview_compatible_eval)
        ):
            primary_attempt = {
                **attempt_summary(row),
            }
            fallback_attempts = [primary_attempt]
            for label, updates in fallback_retry_specs(item, args):
                print(f"  previous attempt failed; retrying {label}", flush=True)
                fallback_args = argparse.Namespace(**vars(args))
                for key, value in updates.items():
                    setattr(fallback_args, key, value)
                fallback_args.attempt_label = label
                row = run_dynamic_item(item, fallback_args)
                strict, reasons = strict_success(row)
                row["strict_physical_success"] = bool(strict)
                row["strict_failure_reasons"] = reasons
                row["fallback_attempts"] = list(fallback_attempts)
                if strict:
                    break
                fallback_attempts.append(attempt_summary(row))
        rows_by_id[row["item_id"]] = row
        rewrite_manifest(args.output_dir, list(rows_by_id.values()))
        print(
            f"  status={row.get('status')} raw={row.get('raw_success')} strict={row.get('strict_physical_success')} "
            f"steps={row.get('steps')} flags={(row.get('diagnostics') or {}).get('flags', [])}",
            flush=True,
        )

    if bool(args.worker_no_finalize):
        worker_rows = [rows_by_id[item["item_id"]] for item in items if item["item_id"] in rows_by_id]
        return 0 if worker_rows and all(strict_success(row)[0] for row in worker_rows) else 2

    rows = list(rows_by_id.values())
    hdf5_path = None
    if rows and bool(args.pack_hdf5):
        hdf5_path, rows = pack_hdf5(args.output_dir, rows, args.pack_hdf5_name)
        rewrite_manifest(args.output_dir, rows)
    audit = write_audit(args.output_dir, rows, hdf5_path)
    review_path = write_review_html(args.output_dir, rows)
    print(json.dumps({"audit": audit, "review": str(review_path)}, ensure_ascii=False, indent=2), flush=True)
    return 0 if audit.get("strict_physical_success", 0) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
