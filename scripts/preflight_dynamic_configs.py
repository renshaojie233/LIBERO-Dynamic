#!/usr/bin/env python3
"""Preflight LIBERO-dynamic task configs before running model evaluation.

The checker intentionally runs without a policy: it loads each configured task,
sets one init state, sends zero actions, and verifies that the dynamic target
stays inside configured bounds and does not make unexpected contacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.dynamic_perturbation import (
    load_dynamic_config,
    resolve_dynamic_config_for_task,
)


DEFAULT_ALLOWED_CONTACT_KEYWORDS = (
    "table",
    "floor",
    "ground",
    "carrier",
    "conveyor",
    "workspace",
    "region",
)


def _task_entries(config: dict[str, Any], suite_filter: str | None, task_filter: int | None):
    tasks = config.get("tasks", {}) or {}
    for suite_name, suite_tasks in tasks.items():
        if suite_filter is not None and suite_name != suite_filter:
            continue
        if not isinstance(suite_tasks, dict):
            continue
        for task_id_raw, entry in suite_tasks.items():
            task_id = int(task_id_raw)
            if task_filter is not None and task_id != task_filter:
                continue
            if isinstance(entry, dict) and bool(entry.get("enabled", True)):
                yield suite_name, task_id


def _zero_action(env) -> np.ndarray:
    action_dim = getattr(env.env, "action_dim", None)
    if action_dim is None:
        robots = getattr(env.env, "robots", []) or []
        action_dim = sum(int(getattr(robot, "action_dim", 0)) for robot in robots)
    if not action_dim:
        raise RuntimeError("Unable to infer env action_dim for zero-action preflight")
    return np.zeros(int(action_dim), dtype=np.float32)


def _target_body_names(env, target_name: str) -> set[str]:
    names = {target_name}
    for body_id in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(body_id)
        if body_name and (body_name == target_name or body_name.startswith(target_name)):
            names.add(body_name)
    return names


def _geom_owner_body(env, geom_id: int) -> str:
    body_id = int(env.sim.model.geom_bodyid[geom_id])
    return env.sim.model.body_id2name(body_id) or ""


def _contact_allowed(name: str, target_name: str, allowed_keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    if not name or name == target_name or name.startswith(target_name):
        return True
    return any(keyword in lowered for keyword in allowed_keywords)


def _unexpected_target_contacts(env, target_name: str, allowed_keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    target_bodies = _target_body_names(env, target_name)
    bad = []
    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = env.sim.model.geom_id2name(contact.geom1) or ""
        geom2 = env.sim.model.geom_id2name(contact.geom2) or ""
        body1 = _geom_owner_body(env, contact.geom1)
        body2 = _geom_owner_body(env, contact.geom2)
        side1_is_target = body1 in target_bodies or geom1.startswith(target_name)
        side2_is_target = body2 in target_bodies or geom2.startswith(target_name)
        if not (side1_is_target or side2_is_target):
            continue
        other_body = body2 if side1_is_target else body1
        other_geom = geom2 if side1_is_target else geom1
        if _contact_allowed(other_body, target_name, allowed_keywords) or _contact_allowed(other_geom, target_name, allowed_keywords):
            continue
        bad.append(
            {
                "contact_index": int(idx),
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "other_body": other_body,
                "other_geom": other_geom,
            }
        )
    return bad


def _safe_workspace(config: dict[str, Any], initial_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    safety = config.get("safety", {}) or {}
    motion = config.get("motion", {}) or {}
    workspace = safety.get("workspace_xyz") or motion.get("workspace_xyz")
    if workspace and "min" in workspace and "max" in workspace:
        return np.asarray(workspace["min"], dtype=float), np.asarray(workspace["max"], dtype=float), "configured"

    margin = float(safety.get("default_margin_m", 0.03))
    amplitude = float(abs(motion.get("amplitude_m", motion.get("max_abs_offset_m", 0.06))))
    direction = np.zeros(3, dtype=float)
    axis = str(motion.get("axis", "y")).lower()
    axis_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
    direction[axis_map.get(axis, 1)] = 1.0
    span = np.asarray(safety.get("default_span_xyz", [0.08, 0.08, 0.04]), dtype=float)
    span = np.maximum(span, np.abs(direction) * (amplitude + margin))
    return initial_pos - span, initial_pos + span, "inferred_from_initial_pose"


def _in_bounds(pos: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(np.all(pos >= low - 1e-9) and np.all(pos <= high + 1e-9))


def run_preflight_for_task(
    *,
    suite_name: str,
    task_id: int,
    concrete_config: dict[str, Any],
    steps: int,
    init_state_id: int,
    camera_size: int,
    allowed_contact_keywords: tuple[str, ...],
) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    if init_state_id >= len(init_states):
        raise IndexError(f"init_state_id={init_state_id} out of range for {suite_name} task {task_id}: {len(init_states)} states")

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    try:
        env.reset()
        env.set_init_state(init_states[init_state_id])
        if env.dynamic_controller is None:
            return {
                "suite": suite_name,
                "task_id": int(task_id),
                "language": task.language,
                "status": "static_or_disabled",
                "ok": True,
            }

        target_name = env.dynamic_controller.target.object_name
        motion_cfg = concrete_config.get("motion", {}) or {}
        safety_cfg = concrete_config.get("safety", {}) or {}
        check_from_step = int(safety_cfg.get("preflight_check_from_step", motion_cfg.get("start_step", 0)))
        positions = []
        checked_positions = []
        bounds_violations = []
        contact_violations = []
        events = []
        zero_action = _zero_action(env)
        low = high = None
        workspace_source = "not_initialized"
        done = False

        for step_idx in range(int(steps)):
            _obs, _reward, done, info = env.step(zero_action)
            pose = env.dynamic_controller.get_body_pose()
            pos = np.asarray(pose["pos"], dtype=float)
            positions.append(pos.tolist())
            if "dynamic_perturbation_events" in info:
                events.extend(info["dynamic_perturbation_events"])

            if step_idx < check_from_step:
                if done:
                    break
                continue

            if low is None or high is None:
                low, high, workspace_source = _safe_workspace(concrete_config, pos)
            checked_positions.append(pos.tolist())
            if not _in_bounds(pos, low, high):
                bounds_violations.append(
                    {
                        "step_idx": int(step_idx),
                        "pos": pos.tolist(),
                        "workspace_min": low.tolist(),
                        "workspace_max": high.tolist(),
                    }
                )
            bad_contacts = _unexpected_target_contacts(env, target_name, allowed_contact_keywords)
            if bad_contacts:
                contact_violations.append({"step_idx": int(step_idx), "contacts": bad_contacts[:8]})
            if done:
                break

        if low is None or high is None:
            current_pos = np.asarray(env.dynamic_controller.get_body_pose()["pos"], dtype=float)
            low, high, workspace_source = _safe_workspace(concrete_config, current_pos)
        pos_arr = np.asarray(positions or [env.dynamic_controller.get_body_pose()["pos"]], dtype=float)
        checked_pos_arr = np.asarray(checked_positions or positions or [env.dynamic_controller.get_body_pose()["pos"]], dtype=float)
        ok = not bounds_violations and not contact_violations
        return {
            "suite": suite_name,
            "task_id": int(task_id),
            "language": task.language,
            "target_object_name": target_name,
            "target_joint_name": env.dynamic_controller.target.joint_name,
            "ok": bool(ok),
            "status": "pass" if ok else "fail",
            "num_steps": int(len(positions) - 1),
            "workspace_source": workspace_source,
            "workspace_min": low.tolist(),
            "workspace_max": high.tolist(),
            "check_from_step": int(check_from_step),
            "position_min": pos_arr.min(axis=0).tolist(),
            "position_max": pos_arr.max(axis=0).tolist(),
            "checked_position_min": checked_pos_arr.min(axis=0).tolist(),
            "checked_position_max": checked_pos_arr.max(axis=0).tolist(),
            "num_dynamic_events": int(len(events)),
            "bounds_violations": bounds_violations[:20],
            "contact_violations": contact_violations[:20],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("LIBERO_DYNAMIC_CONFIG", "libero_dynamic"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allowed-contact-keyword", action="append", default=[])
    args = parser.parse_args()

    os.environ["LIBERO_DYNAMIC_CONFIG"] = str(args.config)
    config = load_dynamic_config(args.config)
    allowed_keywords = tuple(DEFAULT_ALLOWED_CONTACT_KEYWORDS + tuple(args.allowed_contact_keyword))
    results = []
    for suite_name, task_id in _task_entries(config, args.suite, args.task_id):
        concrete = resolve_dynamic_config_for_task(config, suite_name, task_id)
        try:
            result = run_preflight_for_task(
                suite_name=suite_name,
                task_id=task_id,
                concrete_config=concrete,
                steps=args.steps,
                init_state_id=args.init_state_id,
                camera_size=args.camera_size,
                allowed_contact_keywords=allowed_keywords,
            )
        except Exception as exc:
            result = {
                "suite": suite_name,
                "task_id": int(task_id),
                "ok": False,
                "status": "error",
                "error": repr(exc),
            }
        results.append(result)
        marker = "PASS" if result.get("ok") else "FAIL"
        print(f"[{marker}] {suite_name} task{task_id}: {result.get('status')} target={result.get('target_object_name')} events={result.get('num_dynamic_events')}")
        if result.get("bounds_violations"):
            print(f"  bounds violations: {len(result['bounds_violations'])}")
        if result.get("contact_violations"):
            print(f"  contact violations: {len(result['contact_violations'])}")
        if result.get("error"):
            print(f"  error: {result['error']}")

    summary = {
        "config": str(args.config),
        "steps": int(args.steps),
        "init_state_id": int(args.init_state_id),
        "num_tasks": len(results),
        "num_pass": sum(1 for r in results if r.get("ok")),
        "num_fail": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0 if summary["num_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
