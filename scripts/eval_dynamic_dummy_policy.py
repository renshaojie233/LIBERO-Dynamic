#!/usr/bin/env python3
"""Run a policy-shaped LIBERO-Dynamic evaluation with a dummy action model.

This is the smallest formal evaluation harness for the benchmark itself:

1. create the normal LIBERO environment;
2. reset to a LIBERO init state;
3. read an observation;
4. ask a policy object for an action chunk;
5. notify inference latency (zero / measured / fixed, depending on CLI);
6. execute the chunk through env.step(action).

The dummy policy returns zero actions, so the robot should remain still.  The
dynamic scene still moves through the same env hooks used by real model eval.
Optional diagnostics and videos are meant for benchmark QA, not for final paper
metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - video writing dependency is optional for --no-save-videos.
    imageio = None

try:
    from render_dynamic_config_previews import (
        apply_static_support_contact_override,
        deep_merge,
        motion_period_seconds,
        project_world_to_camera,
        render_frame,
        target_entries_for_preview,
    )
except Exception:  # pragma: no cover - keeps import errors readable in user envs.
    apply_static_support_contact_override = None
    deep_merge = None
    motion_period_seconds = None
    project_world_to_camera = None
    render_frame = None
    target_entries_for_preview = None


@dataclass
class DummyChunkPolicy:
    action_dim: int
    chunk_size: int

    def predict_action_chunk(self, obs: dict[str, Any], language: str) -> np.ndarray:
        del obs, language
        return np.zeros((int(self.chunk_size), int(self.action_dim)), dtype=np.float32)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def action_dim(env: OffScreenRenderEnv) -> int:
    dim = getattr(env.env, "action_dim", None)
    if dim is not None:
        return int(dim)
    return int(sum(int(getattr(robot, "action_dim", 0)) for robot in getattr(env.env, "robots", []) or []))


def dynamic_config_path(config_dir: Path, level: str) -> Path:
    for name in (f"libero_dynamic_{level}.yaml", f"libero_dynamic_manual_{level}.yaml"):
        candidate = config_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find dynamic config for level={level} under {config_dir}")


def merge_dicts(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    if deep_merge is not None:
        return deep_merge(defaults, overrides)
    merged = dict(defaults or {})
    for key, value in (overrides or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def concrete_item_cfg(task_cfg: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Build the same concrete per-variant config used by preview rendering.

    The reviewed configs are defaults -> task -> activation_variant.  Formal
    eval must not pass the raw variant as task_cfg, because that drops task
    level fields such as carrier and latency.  This mirrors preview_items().
    """
    item_cfg = deepcopy(task_cfg)
    for key in (
        "initial_scene_adjustments",
        "manual_review",
        "logging",
        "maskfit_alignment",
        "target_object_name",
        "target_joint_name",
        "target_kind",
        "target_body_name",
        "dynamic_role",
        "latency",
    ):
        if key in variant:
            item_cfg[key] = deepcopy(variant[key])
    item_cfg["target_objects"] = deepcopy(variant.get("target_objects") or [])
    if item_cfg["target_objects"]:
        item_cfg["motion"] = deepcopy((item_cfg["target_objects"][0] or {}).get("motion", {}))
    else:
        item_cfg["motion"] = deepcopy(variant.get("motion") or item_cfg.get("motion") or {})
    item_cfg["selected_variant_name"] = variant.get("variant_name")
    item_cfg["selected_variant_id"] = variant.get("variant_id")
    return item_cfg


def apply_static_support_override_to_targets(
    targets: list[dict[str, Any]],
    *,
    keep_static_support_contact: bool,
    static_support_clearance_m: float,
) -> list[dict[str, Any]]:
    if apply_static_support_contact_override is not None:
        targets = apply_static_support_contact_override(
            targets,
            keep_static_support_contact=keep_static_support_contact,
            static_support_clearance_m=static_support_clearance_m,
        )
    if not keep_static_support_contact:
        return targets
    out = deepcopy(targets)
    for target in out:
        target["exclude_static_support_contact"] = False
        if static_support_clearance_m > 0.0:
            target["surface_epsilon_m"] = max(
                float(target.get("surface_epsilon_m", 0.0) or 0.0),
                float(static_support_clearance_m),
            )
        target_name = str(target.get("target_object_name") or target.get("object_name") or "").lower()
        if "basket" in target_name:
            target.setdefault("stability_proxy_enabled", True)
            target.setdefault("stability_proxy_size", "auto")
            target.setdefault("stability_proxy_thickness_m", 0.0015)
            target.setdefault("stability_proxy_bottom_epsilon_m", 0.00015)
    return out


def write_effective_config_files(config_dir: Path, output_dir: Path, levels: list[str], args: argparse.Namespace) -> dict[str, str]:
    """Write the exact configs the env will consume for this eval run."""
    effective_dir = output_dir / "effective_dynamic_configs"
    effective_dir.mkdir(parents=True, exist_ok=True)
    path_by_level: dict[str, str] = {}

    carrier_overrides = {}
    if hasattr(args, "platform_settle_steps"):
        carrier_overrides["platform_settle_steps"] = int(args.platform_settle_steps)
    if hasattr(args, "realign_after_platform_settle"):
        carrier_overrides["realign_after_platform_settle"] = bool(args.realign_after_platform_settle)

    for level in levels:
        src = dynamic_config_path(config_dir, level)
        cfg = load_yaml(src)
        if carrier_overrides:
            cfg.setdefault("defaults", {}).setdefault("carrier", {}).update(carrier_overrides)
        for suite_tasks in (cfg.get("tasks") or {}).values():
            for task_cfg in (suite_tasks or {}).values():
                if carrier_overrides:
                    task_cfg.setdefault("carrier", {}).update(carrier_overrides)
                for variant in task_cfg.get("activation_variants") or []:
                    targets = deepcopy(variant.get("target_objects") or [])
                    if targets:
                        variant["target_objects"] = apply_static_support_override_to_targets(
                            targets,
                            keep_static_support_contact=bool(args.keep_static_support_contact),
                            static_support_clearance_m=float(args.static_support_clearance_m),
                        )
        dst = effective_dir / src.name
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        path_by_level[str(level)] = str(dst)
    return path_by_level


def build_items(
    config_dir: Path,
    levels: list[str],
    traj_indices: list[int],
    *,
    keep_static_support_contact: bool = False,
    static_support_clearance_m: float = 0.0,
    config_path_by_level: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for level in levels:
        path = dynamic_config_path(config_dir, level)
        cfg = load_yaml(path)
        task_defaults = cfg.get("defaults") or {}
        for suite_name, suite_tasks in (cfg.get("tasks") or {}).items():
            for task_key, raw_task_cfg in (suite_tasks or {}).items():
                task_id = int(task_key)
                task_cfg = merge_dicts(task_defaults, raw_task_cfg or {})
                if not task_cfg.get("enabled", False):
                    continue
                variants = task_cfg.get("activation_variants") or []
                for traj_index in traj_indices:
                    if traj_index < 0 or traj_index >= len(variants):
                        continue
                    variant = variants[traj_index]
                    item_cfg = concrete_item_cfg(task_cfg, variant)
                    item_cfg["target_objects"] = apply_static_support_override_to_targets(
                        deepcopy(item_cfg.get("target_objects") or []),
                        keep_static_support_contact=bool(keep_static_support_contact),
                        static_support_clearance_m=float(static_support_clearance_m),
                    )
                    items.append(
                        {
                            "item_id": f"{suite_name}_task{task_id:02d}__{level}__traj{traj_index:02d}",
                            "suite": str(suite_name),
                            "task_id": int(task_id),
                            "level": str(level),
                            "traj_index": int(traj_index),
                            "variant_name": str(variant.get("variant_name", f"traj{traj_index:02d}")),
                            "variant_id": str(variant.get("variant_id", "")),
                            "config_path": str((config_path_by_level or {}).get(str(level), str(path))),
                            "task_cfg": item_cfg,
                        }
                    )
    return sorted(items, key=lambda x: (x["level"], x["suite"], x["task_id"], x["traj_index"]))


def task_bddl(suite_name: str, task_id: int) -> tuple[Any, Any, Path]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return suite, task, bddl


def get_joint_qpos(env: OffScreenRenderEnv, joint: str) -> np.ndarray | None:
    try:
        qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
    except Exception:
        return None
    return qpos if qpos.ndim > 0 else np.asarray([float(qpos)], dtype=float)


def platform_pos(env: OffScreenRenderEnv, platform: dict[str, Any]) -> np.ndarray | None:
    if not platform:
        return None
    support_body_name = str((platform or {}).get("support_body_name") or "")
    if support_body_name:
        try:
            body_id = int(env.sim.model.body_name2id(support_body_name))
            return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
        except Exception:
            return None
    joint_names = platform.get("joint_names") or {}
    try:
        return np.asarray(
            [
                float(env.sim.data.get_joint_qpos(joint_names["x"])),
                float(env.sim.data.get_joint_qpos(joint_names["y"])),
                float(env.sim.data.get_joint_qpos(joint_names["z"])),
            ],
            dtype=float,
        )
    except Exception:
        return None


def target_bottom_z(env: OffScreenRenderEnv, name: str) -> float | None:
    if not hasattr(env, "_target_contact_geom_names"):
        return None
    bottoms = []
    for geom_name in env._target_contact_geom_names(name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
            bottoms.append(float(env.sim.data.geom_xpos[geom_id][2]) - float(env.sim.model.geom_size[geom_id][2]))
        except Exception:
            continue
    return min(bottoms) if bottoms else None


def target_tilt_deg(qpos: np.ndarray, baseline_qpos: np.ndarray | None) -> float:
    if baseline_qpos is None or qpos is None or qpos.size < 7 or baseline_qpos.size < 7:
        return 0.0
    q0 = np.asarray(baseline_qpos[3:7], dtype=float)
    q1 = np.asarray(qpos[3:7], dtype=float)
    q0 /= max(1e-9, float(np.linalg.norm(q0)))
    q1 /= max(1e-9, float(np.linalg.norm(q1)))
    dot = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def _target_contact_geom_ids(env: OffScreenRenderEnv, name: str) -> set[int]:
    geom_names = []
    if hasattr(env, "_target_all_contact_geom_names"):
        try:
            geom_names = list(env._target_all_contact_geom_names(name))
        except Exception:
            geom_names = []
    if not geom_names and hasattr(env, "_target_contact_geom_names"):
        try:
            geom_names = list(env._target_contact_geom_names(name))
        except Exception:
            geom_names = []
    geom_ids = set()
    for geom_name in geom_names:
        try:
            geom_ids.add(int(env.sim.model.geom_name2id(str(geom_name))))
        except Exception:
            continue
    return geom_ids


def _is_robot_geom(env: OffScreenRenderEnv, geom_id: int) -> bool:
    if hasattr(env, "_is_robot_geom"):
        try:
            return bool(env._is_robot_geom(int(geom_id)))
        except Exception:
            pass
    try:
        geom_name = env.sim.model.geom_id2name(int(geom_id)) or ""
        body_id = int(env.sim.model.geom_bodyid[int(geom_id)])
        body_name = env.sim.model.body_id2name(body_id) or ""
    except Exception:
        return False
    return geom_name.startswith(("robot0_", "gripper0_", "mount0_")) or body_name.startswith(
        ("robot0_", "gripper0_", "mount0_")
    )


def robot_contacting_target(env: OffScreenRenderEnv, name: str) -> bool:
    target_geom_ids = _target_contact_geom_ids(env, name)
    if not target_geom_ids:
        return False
    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in target_geom_ids and _is_robot_geom(env, geom2):
            return True
        if geom2 in target_geom_ids and _is_robot_geom(env, geom1):
            return True
    return False


def collect_baselines(env: OffScreenRenderEnv, *, bottom_baseline_mode: str) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for state in getattr(env, "dynamic_multi_target_states", []) or []:
        qpos = get_joint_qpos(env, state["joint"])
        ppos = platform_pos(env, state["platform"])
        object_xy = qpos[:2].copy() if qpos is not None and qpos.size >= 2 else None
        baselines[state["name"]] = {
            "qpos": None if qpos is None else qpos.copy(),
            "platform_pos": None if ppos is None else ppos.copy(),
            "object_to_platform_xy": None
            if object_xy is None or ppos is None
            else (object_xy - ppos[:2]).copy(),
            "bottom_z": target_bottom_z(env, state["name"]) if bottom_baseline_mode == "reset" else None,
            "initial_bottom_z": target_bottom_z(env, state["name"]),
            "bottom_baseline_mode": bottom_baseline_mode,
            "bottom_baseline_source": "reset" if bottom_baseline_mode == "reset" else None,
        }
    return baselines


def init_diag_state(env: OffScreenRenderEnv, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    baselines = collect_baselines(env, bottom_baseline_mode=str(args.bottom_baseline_mode))
    return {
        name: {
            "baseline": baseline,
            "samples": 0,
            "max_platform_path_error_m": 0.0,
            "max_follow_error_m": 0.0,
            "max_tilt_deg": 0.0,
            "min_relative_bottom_z_m": 0.0,
            "birdview_offscreen": 0,
            "agentview_offscreen": 0,
            "robot_contacted": False,
            "robot_contact_sample": None,
            "samples_after_robot_contact": 0,
        }
        for name, baseline in baselines.items()
    }


def update_diagnostics(env: OffScreenRenderEnv, diag: dict[str, dict[str, Any]]) -> None:
    for state in getattr(env, "dynamic_multi_target_states", []) or []:
        name = state["name"]
        if name not in diag:
            continue
        entry = diag[name]
        baseline = entry["baseline"]
        entry["samples"] += 1

        ppos = platform_pos(env, state["platform"])
        if ppos is not None and state.get("platform_base_pos") is not None:
            expected = np.asarray(state["platform_base_pos"], dtype=float) + np.asarray(state.get("last_offset", [0, 0, 0]), dtype=float)
            entry["max_platform_path_error_m"] = max(
                float(entry["max_platform_path_error_m"]),
                float(np.linalg.norm(ppos[:2] - expected[:2])),
            )

        if entry.get("robot_contacted") or robot_contacting_target(env, name):
            if not entry.get("robot_contacted"):
                entry["robot_contacted"] = True
                entry["robot_contact_sample"] = int(entry["samples"])
            entry["samples_after_robot_contact"] = int(entry.get("samples_after_robot_contact", 0)) + 1
            continue

        qpos = get_joint_qpos(env, state["joint"])
        if qpos is not None and qpos.size >= 2 and ppos is not None and baseline.get("object_to_platform_xy") is not None:
            follow_error = np.linalg.norm((qpos[:2] - ppos[:2]) - np.asarray(baseline["object_to_platform_xy"], dtype=float))
            entry["max_follow_error_m"] = max(float(entry["max_follow_error_m"]), float(follow_error))
            entry["max_tilt_deg"] = max(float(entry["max_tilt_deg"]), target_tilt_deg(qpos, baseline.get("qpos")))
            if project_world_to_camera is not None and qpos.size >= 3:
                for camera_name, key in (("birdview", "birdview_offscreen"), ("agentview", "agentview_offscreen")):
                    if project_world_to_camera(env, qpos[:3], camera_name, 256) is None:
                        entry[key] += 1

        bottom = target_bottom_z(env, name)
        if bottom is not None and baseline.get("bottom_z") is None:
            baseline["bottom_z"] = float(bottom)
            baseline["bottom_baseline_source"] = "first_sample"
        if bottom is not None and baseline.get("bottom_z") is not None:
            rel = float(bottom) - float(baseline["bottom_z"])
            entry["min_relative_bottom_z_m"] = min(float(entry["min_relative_bottom_z_m"]), rel)


def summarize_diagnostics(diag: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    targets = {}
    flags = []
    for name, entry in diag.items():
        samples = max(1, int(entry.get("samples", 0)))
        target_summary = {
            "samples": int(entry.get("samples", 0)),
            "max_platform_path_error_m": round(float(entry["max_platform_path_error_m"]), 6),
            "max_follow_error_m": round(float(entry["max_follow_error_m"]), 6),
            "max_tilt_deg": round(float(entry["max_tilt_deg"]), 3),
            "min_relative_bottom_z_m": round(float(entry["min_relative_bottom_z_m"]), 6),
            "bottom_baseline_source": str((entry.get("baseline") or {}).get("bottom_baseline_source") or "none"),
            "birdview_offscreen_ratio": round(float(entry["birdview_offscreen"]) / samples, 4),
            "agentview_offscreen_ratio": round(float(entry["agentview_offscreen"]) / samples, 4),
            "robot_contacted": bool(entry.get("robot_contacted", False)),
            "robot_contact_sample": entry.get("robot_contact_sample"),
            "samples_after_robot_contact": int(entry.get("samples_after_robot_contact", 0)),
        }
        if target_summary["max_platform_path_error_m"] > args.platform_error_threshold_m:
            flags.append({"target": name, "type": "platform_path", "value": target_summary["max_platform_path_error_m"]})
        if target_summary["max_follow_error_m"] > args.follow_error_threshold_m:
            flags.append({"target": name, "type": "object_follow", "value": target_summary["max_follow_error_m"]})
        if target_summary["max_tilt_deg"] > args.tilt_threshold_deg:
            flags.append({"target": name, "type": "tilt", "value": target_summary["max_tilt_deg"]})
        if target_summary["min_relative_bottom_z_m"] < -abs(args.drop_threshold_m):
            flags.append({"target": name, "type": "relative_drop", "value": target_summary["min_relative_bottom_z_m"]})
        if target_summary["birdview_offscreen_ratio"] > args.offscreen_ratio_threshold:
            flags.append({"target": name, "type": "birdview_offscreen", "value": target_summary["birdview_offscreen_ratio"]})
        targets[name] = target_summary
    return {"targets": targets, "flags": flags, "ok": len(flags) == 0}


def duration_from_env(env: OffScreenRenderEnv, cycles: float, fallback_seconds: float) -> float:
    periods = []
    if motion_period_seconds is not None:
        for state in getattr(env, "dynamic_multi_target_states", []) or []:
            period = motion_period_seconds(state.get("motion") or {})
            if period is not None and period > 1e-6:
                periods.append(float(period))
    if periods:
        return float(cycles) * max(periods)
    return float(fallback_seconds)


def run_item(item: dict[str, Any], args_dict: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    os.environ["LIBERO_DYNAMIC_CONFIG"] = item["config_path"]
    os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
    os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = item["variant_id"]
    os.environ.setdefault("LIBERO_DYNAMIC_AUTO_FINALIZE", "false")

    suite, task, bddl = task_bddl(item["suite"], item["task_id"])
    init_states = suite.get_task_init_states(item["task_id"])
    video_dir = args.output_dir / item["level"] / item["suite"] / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{item['item_id']}.mp4"
    manifest_path = args.output_dir / item["level"] / item["suite"] / "manifests" / f"{item['item_id']}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    env = None
    writer = None
    started = time.time()
    replan_count = 0
    step_count = 0
    success = False
    error = None
    diagnostics = None
    latency_samples: list[float] = []
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=int(args.camera_size),
            camera_widths=int(args.camera_size),
            control_freq=int(args.control_freq),
        )
        obs = env.reset()
        obs = env.set_init_state(init_states[int(args.init_state_id)])
        policy = DummyChunkPolicy(action_dim(env), int(args.chunk_size))
        diagnostics = init_diag_state(env, args)
        duration_seconds = duration_from_env(env, float(args.cycles), float(args.duration_seconds))
        max_steps = int(math.ceil(duration_seconds * int(args.control_freq)))
        if args.max_steps is not None:
            max_steps = min(max_steps, int(args.max_steps))
        render_every = max(1, int(round(int(args.control_freq) / float(args.video_fps))))
        if bool(args.save_videos):
            if imageio is None:
                raise RuntimeError("imageio is required when --save-videos is enabled")
            writer = imageio.get_writer(video_path, fps=float(args.video_fps), macro_block_size=None)

        while step_count < max_steps:
            infer_started = time.perf_counter()
            chunk = policy.predict_action_chunk(obs, task.language)
            infer_elapsed = time.perf_counter() - infer_started
            latency_samples.append(float(infer_elapsed))
            replan_count += 1
            if hasattr(env, "notify_inference_latency"):
                if args.latency_mode == "measured":
                    env.notify_inference_latency(infer_elapsed)
                elif args.latency_mode == "fixed":
                    env.notify_inference_latency(None)
                else:
                    env.notify_inference_latency(0.0)
            for action in chunk:
                if step_count >= max_steps:
                    break
                obs, _reward, done, _info = env.step(action)
                success = bool(done) or bool(env.check_success())
                update_diagnostics(env, diagnostics)
                if writer is not None and (step_count % render_every == 0):
                    if render_frame is not None:
                        writer.append_data(render_frame(env, obs, int(args.camera_size), None, overlay_traces=False))
                    else:
                        frame = np.asarray(obs["agentview_image"])[::-1, ::-1]
                        writer.append_data(frame)
                step_count += 1
                if bool(args.stop_on_success) and success:
                    break
            if bool(args.stop_on_success) and success:
                break
        diagnostic_summary = summarize_diagnostics(diagnostics, args)
        status = "ok"
    except Exception as exc:
        status = "error"
        error = repr(exc)
        diagnostic_summary = {"targets": {}, "flags": [{"type": "exception", "value": error}], "ok": False}
        duration_seconds = 0.0
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()

    result = {
        **{k: item[k] for k in ("item_id", "suite", "task_id", "level", "traj_index", "variant_name", "variant_id")},
        "status": status,
        "error": error,
        "language": getattr(task, "language", ""),
        "init_state_id": int(args.init_state_id),
        "control_freq": int(args.control_freq),
        "chunk_size": int(args.chunk_size),
        "latency_mode": str(args.latency_mode),
        "keep_static_support_contact": bool(args.keep_static_support_contact),
        "static_support_clearance_m": float(args.static_support_clearance_m),
        "effective_config_path": str(item["config_path"]),
        "inference_latency_seconds": 0.0 if not latency_samples else round(float(np.mean(latency_samples)), 6),
        "inference_latency_max_seconds": 0.0 if not latency_samples else round(float(np.max(latency_samples)), 6),
        "policy": "DummyChunkPolicy(zero_actions)",
        "success": bool(success),
        "steps": int(step_count),
        "replans": int(replan_count),
        "duration_seconds": round(float(duration_seconds), 4),
        "elapsed_seconds": round(time.time() - started, 3),
        "target_names": [
            target.get("target_object_name")
            for target in (
                target_entries_for_preview(item["task_cfg"])
                if target_entries_for_preview is not None
                else item["task_cfg"].get("target_objects", [])
            )
        ],
        "diagnostics": diagnostic_summary,
        "video": str(video_path) if bool(args.save_videos) and status == "ok" else None,
        "manifest": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    issue_rows = [row for row in results if row.get("status") != "ok" or not row.get("diagnostics", {}).get("ok", False)]
    summary = {
        "total": len(results),
        "ok_status": sum(1 for row in results if row.get("status") == "ok"),
        "diagnostic_ok": sum(1 for row in results if row.get("diagnostics", {}).get("ok", False)),
        "issues": len(issue_rows),
        "issue_items": [
            {
                "item_id": row.get("item_id"),
                "status": row.get("status"),
                "flags": row.get("diagnostics", {}).get("flags", []),
                "error": row.get("error"),
            }
            for row in issue_rows
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", type=Path, default=repo_root / "libero" / "libero" / "dynamic_configs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--latency-mode",
        choices=["zero", "measured", "fixed"],
        default="zero",
        help=(
            "zero: notify 0s latency; measured: time policy.predict_action_chunk "
            "and notify that value; fixed: notify None so the env uses configured "
            "fixed latency level at each replan."
        ),
    )
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument(
        "--keep-static-support-contact",
        action="store_true",
        help="Keep target-vs-table/cabinet/floor support contacts enabled while the conveyor platform drives targets.",
    )
    parser.add_argument(
        "--static-support-clearance-m",
        type=float,
        default=0.0,
        help="Tiny platform surface lift used with --keep-static-support-contact, e.g. 0.001 for the reviewed table-contact run.",
    )
    parser.add_argument("--platform-error-threshold-m", type=float, default=0.01)
    parser.add_argument("--follow-error-threshold-m", type=float, default=0.08)
    parser.add_argument("--tilt-threshold-deg", type=float, default=35.0)
    parser.add_argument("--drop-threshold-m", type=float, default=0.04)
    parser.add_argument(
        "--bottom-baseline-mode",
        choices=("first_sample", "reset"),
        default="first_sample",
        help="Use the first rollout diagnostic sample or the raw reset state as the relative-drop baseline.",
    )
    parser.add_argument("--offscreen-ratio-threshold", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir = args.output_dir.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    config_path_by_level = write_effective_config_files(config_dir, args.output_dir, args.levels, args)
    items = build_items(
        config_dir,
        args.levels,
        args.traj_indices,
        keep_static_support_contact=bool(args.keep_static_support_contact),
        static_support_clearance_m=float(args.static_support_clearance_m),
        config_path_by_level=config_path_by_level,
    )
    if args.start_index:
        items = items[int(args.start_index) :]
    if args.limit is not None:
        items = items[: int(args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    args_dict = vars(args).copy()
    results: list[dict[str, Any]] = []
    if int(args.num_workers) <= 1:
        for idx, item in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] {item['item_id']}", flush=True)
            results.append(run_item(item, args_dict))
            write_summary(args.output_dir, results)
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            futures = {executor.submit(run_item, item, args_dict): item for item in items}
            for idx, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**item, "status": "error", "error": repr(exc), "diagnostics": {"ok": False, "flags": [{"type": "worker_exception", "value": repr(exc)}]}}
                print(f"[{idx}/{len(items)}] {result['item_id']} status={result.get('status')} diag_ok={result.get('diagnostics', {}).get('ok')}", flush=True)
                results.append(result)
                write_summary(args.output_dir, sorted(results, key=lambda row: row.get("item_id", "")))
    write_summary(args.output_dir, sorted(results, key=lambda row: row.get("item_id", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
