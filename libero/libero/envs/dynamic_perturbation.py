from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import os

import numpy as np


_MISSING = object()


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _lookup_task_entry(tasks: dict[str, Any], suite_name: str, task_id: int) -> dict[str, Any] | None:
    suite_tasks = tasks.get(suite_name, {}) or {}
    if not isinstance(suite_tasks, dict):
        return None
    entry = suite_tasks.get(task_id, suite_tasks.get(str(task_id), None))
    return entry if isinstance(entry, dict) else None


def _variant_selector_from_env() -> tuple[str | None, int | None, str | None]:
    """Return variant selection requested by environment variables.

    This keeps the eval loop unchanged: users can run the same LIBERO eval
    script multiple times while selecting a LIBERO-Dynamic level / trajectory
    through environment variables.
    """
    variant_id = os.environ.get("LIBERO_DYNAMIC_VARIANT_ID")
    variant_name = os.environ.get("LIBERO_DYNAMIC_VARIANT")
    traj_raw = os.environ.get("LIBERO_DYNAMIC_TRAJ_INDEX")
    traj_index = None
    if traj_raw is not None and str(traj_raw).strip() != "":
        traj_index = int(traj_raw)
        variant_name = variant_name or f"traj{traj_index:02d}"
    return variant_id, traj_index, variant_name


def _select_activation_variant(task_entry: dict[str, Any]) -> dict[str, Any] | None:
    variants = task_entry.get("activation_variants") or []
    if not variants:
        return None
    variant_id, traj_index, variant_name = _variant_selector_from_env()

    if variant_id:
        for variant in variants:
            if str(variant.get("variant_id")) == str(variant_id):
                return variant
        raise KeyError(f"LIBERO_DYNAMIC_VARIANT_ID not found in task config: {variant_id}")

    if variant_name:
        for variant in variants:
            if str(variant.get("variant_name")) == str(variant_name):
                return variant
        raise KeyError(f"LIBERO_DYNAMIC_VARIANT not found in task config: {variant_name}")

    if traj_index is not None:
        if traj_index < 0 or traj_index >= len(variants):
            raise IndexError(
                f"LIBERO_DYNAMIC_TRAJ_INDEX={traj_index} out of range for {len(variants)} variants"
            )
        return variants[traj_index]

    default_name = str(task_entry.get("default_variant", "traj00"))
    for variant in variants:
        if str(variant.get("variant_name")) == default_name:
            return variant
    return variants[0]


def _apply_activation_variant(task_entry: dict[str, Any], variant: dict[str, Any] | None) -> dict[str, Any]:
    if variant is None:
        return dict(task_entry)
    concrete = dict(task_entry)
    for key in (
        "target_object_name",
        "target_joint_name",
        "target_kind",
        "target_body_name",
        "dynamic_role",
        "motion",
        "latency",
        "target_objects",
        "initial_scene_adjustments",
        "manual_review",
        "logging",
        "safety",
        "maskfit_alignment",
    ):
        if key in variant:
            concrete[key] = variant[key]
    concrete["selected_variant_name"] = variant.get("variant_name")
    concrete["selected_variant_id"] = variant.get("variant_id")
    return concrete


def resolve_dynamic_config_for_task(
    config: dict[str, Any],
    suite_name: str,
    task_id: int | None,
) -> dict[str, Any]:
    """Return the concrete dynamic config for one LIBERO task.

    Final benchmark configs use a compact schema:

    defaults:
      perturb_type: dynamic_motion
      motion: ...
      latency: ...
    tasks:
      libero_spatial:
        0:
          target_object_name: ...
          motion: ...

    This helper keeps the env code simple by expanding that into the legacy
    concrete shape with top-level motion / latency / carrier / targets.
    """
    if task_id is None or "tasks" not in config:
        return config

    task_entry = _lookup_task_entry(config.get("tasks", {}) or {}, suite_name, int(task_id))
    if task_entry is None:
        fallback_enabled = bool(config.get("default_enabled", False))
        return {
            "enabled": fallback_enabled,
            "perturb_type": "none",
            "config_path": config.get("config_path"),
            "unconfigured_task": {"suite": suite_name, "task_id": int(task_id)},
        }

    task_entry = _apply_activation_variant(task_entry, _select_activation_variant(task_entry))
    defaults = config.get("defaults", {}) or {}
    concrete = _deep_merge_dict(defaults, task_entry)
    concrete["enabled"] = bool(task_entry.get("enabled", defaults.get("enabled", config.get("enabled", True))))
    concrete.setdefault("benchmark_status", config.get("benchmark_status", "benchmark_config"))
    concrete.setdefault("config_path", config.get("config_path"))
    concrete["config_suite_name"] = suite_name
    concrete["config_task_id"] = int(task_id)
    if task_entry.get("selected_variant_name") is not None:
        concrete["selected_variant_name"] = task_entry.get("selected_variant_name")
    if task_entry.get("selected_variant_id") is not None:
        concrete["selected_variant_id"] = task_entry.get("selected_variant_id")

    target_object_name = concrete.pop("target_object_name", None)
    target_joint_name = concrete.pop("target_joint_name", None)
    target_cfg = concrete.pop("target", None)
    if target_cfg is not None:
        target_object_name = target_cfg.get("target_object_name", target_cfg.get("object_name", target_object_name))
        target_joint_name = target_cfg.get("target_joint_name", target_cfg.get("joint_name", target_joint_name))
    if target_object_name is None:
        raise KeyError(f"Dynamic task config missing target object for suite={suite_name} task_id={task_id}")
    concrete["targets"] = {
        suite_name: {
            int(task_id): {
                "target_object_name": target_object_name,
                **({"target_joint_name": target_joint_name} if target_joint_name is not None else {}),
                **({"target_body_name": concrete.get("target_body_name")} if concrete.get("target_body_name") is not None else {}),
                **({"target_kind": concrete.get("target_kind")} if concrete.get("target_kind") is not None else {}),
            }
        }
    }
    if concrete.get("target_objects"):
        concrete["multi_target_objects"] = concrete["target_objects"]
    return concrete


@dataclass
class DynamicTarget:
    object_name: str
    joint_name: str
    target_body_name: str | None = None
    target_kind: str | None = None

    @property
    def uses_model_body_pose(self) -> bool:
        return self.target_body_name is not None


def _resolve_dynamic_config_path(config_path: str | Path | None = None) -> Path:
    package_root = Path(__file__).resolve().parents[1]
    package_config_dir = package_root / "dynamic_configs"

    if config_path is None:
        config_path = os.environ.get("LIBERO_DYNAMIC_CONFIG")
    if config_path is None or str(config_path).strip() == "":
        return package_root / "dynamic_config.yaml"

    candidate = Path(config_path)
    if candidate.exists() or candidate.is_absolute() or candidate.parent != Path("."):
        return candidate

    name = str(config_path)
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return package_config_dir / name


def load_dynamic_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_path = _resolve_dynamic_config_path(config_path)
    if not config_path.exists():
        return {"enabled": False, "perturb_type": "none"}

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("config_path", str(config_path))
    return config


def infer_suite_and_task_id_from_bddl(bddl_file_name: str | Path) -> tuple[str, int | None]:
    from libero.libero import benchmark

    bddl_path = Path(bddl_file_name)
    suite_name = bddl_path.parent.name
    bddl_name = bddl_path.name

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        return suite_name, None

    suite = benchmark_dict[suite_name]()
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        if task.bddl_file == bddl_name:
            return suite_name, task_id

    return suite_name, None


class DynamicPerturbationController:
    def __init__(
        self,
        env,
        suite_name: str,
        task_id: int,
        config: dict[str, Any] | None = None,
    ):
        self.env = env
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.config = config if config is not None else load_dynamic_config()
        self.target = self._resolve_target()
        self.applied_events: list[dict[str, Any]] = []
        self.cumulative_shift = [0.0, 0.0, 0.0]

    def _resolve_target(self) -> DynamicTarget:
        targets = self.config.get("targets", {})
        suite_targets = targets.get(self.suite_name, {})
        task_targets = suite_targets.get(self.task_id, suite_targets.get(str(self.task_id), None))
        if not task_targets:
            raise KeyError(
                f"No dynamic target configured for suite={self.suite_name} task_id={self.task_id}"
            )

        object_name = task_targets["target_object_name"]
        joint_name = task_targets.get("target_joint_name")
        target_body_name = task_targets.get("target_body_name") or self.config.get("target_body_name")
        target_kind = task_targets.get("target_kind") or self.config.get("target_kind")
        is_object = object_name in getattr(self.env, "objects_dict", {})
        is_fixture = object_name in getattr(self.env, "fixtures_dict", {})

        if not is_object and not is_fixture:
            raise KeyError(f"Target object not in env objects or fixtures: {object_name}")

        if is_fixture and target_body_name is None:
            fixture = self.env.fixtures_dict[object_name]
            target_body_name = getattr(fixture, "root_body", None) or f"{object_name}_main"

        if joint_name is None:
            container = self.env.objects_dict[object_name] if is_object else self.env.fixtures_dict[object_name]
            joints = getattr(container, "joints", None) or []
            if len(joints) == 0:
                raise ValueError(f"Configured target object has no joints: {object_name}")
            joint_name = joints[0]

        joint_names = {
            self.env.sim.model.joint_id2name(i)
            for i in range(self.env.sim.model.njnt)
        }
        if joint_name not in joint_names:
            raise KeyError(f"Target joint not in sim joints: {joint_name}")

        if target_body_name is not None:
            try:
                self.env.sim.model.body_name2id(str(target_body_name))
            except Exception as exc:
                raise KeyError(f"Target body not in sim bodies: {target_body_name}") from exc

        return DynamicTarget(
            object_name=object_name,
            joint_name=joint_name,
            target_body_name=str(target_body_name) if target_body_name is not None else None,
            target_kind=str(target_kind) if target_kind is not None else None,
        )

    def get_body_pose(self, object_name: str | None = None) -> dict[str, list[float]]:
        object_name = object_name or self.target.object_name
        if object_name == self.target.object_name and self.target.target_body_name is not None:
            body_id = self.env.sim.model.body_name2id(self.target.target_body_name)
        else:
            body_id = self.env.obj_body_id[object_name]
        return {
            "pos": self.env.sim.data.body_xpos[body_id].copy().tolist(),
            "quat": self.env.sim.data.body_xquat[body_id].copy().tolist(),
        }

    def get_joint_qpos(self) -> list[float]:
        return self.env.sim.data.get_joint_qpos(self.target.joint_name).copy().tolist()

    def shift_target(
        self,
        delta_xyz,
        *,
        reason: str,
        step_idx: int | None = None,
        record_event: bool = True,
    ) -> dict[str, Any]:
        before_pose = self.get_body_pose()
        before_qpos = self.env.sim.data.get_joint_qpos(self.target.joint_name).copy()
        before_body_pos = None
        after_body_pos = None
        if self.target.uses_model_body_pose:
            body_id = self.env.sim.model.body_name2id(self.target.target_body_name)
            before_body_pos = self.env.sim.model.body_pos[body_id].copy()
            self.env.sim.model.body_pos[body_id] = before_body_pos + np.asarray(delta_xyz, dtype=float)[:3]
            after_body_pos = self.env.sim.model.body_pos[body_id].copy()
            after_qpos = before_qpos.copy()
        else:
            new_qpos = before_qpos.copy()
            new_qpos[0] += float(delta_xyz[0])
            new_qpos[1] += float(delta_xyz[1])
            new_qpos[2] += float(delta_xyz[2])
            self.env.sim.data.set_joint_qpos(self.target.joint_name, new_qpos)
            after_qpos = new_qpos.copy()
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)

        after_pose = self.get_body_pose()

        self.cumulative_shift = [
            self.cumulative_shift[i] + float(delta_xyz[i]) for i in range(3)
        ]

        event = {
            "reason": reason,
            "step_idx": step_idx,
            "target_object_name": self.target.object_name,
            "target_joint_name": self.target.joint_name,
            "delta_xyz": [float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2])],
            "before_pose": before_pose,
            "after_pose": after_pose,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": after_qpos.tolist(),
            "cumulative_shift": list(self.cumulative_shift),
        }
        if before_body_pos is not None:
            event["target_body_name"] = self.target.target_body_name
            event["before_body_pos"] = before_body_pos.tolist()
            event["after_body_pos"] = after_body_pos.tolist()
        if record_event:
            self.applied_events.append(event)
        return event

    def yaw_target(
        self,
        delta_yaw_rad,
        *,
        reason: str,
        step_idx: int | None = None,
        record_event: bool = True,
    ) -> dict[str, Any]:
        if not self.target.uses_model_body_pose:
            return self.shift_target([0.0, 0.0, 0.0], reason=reason, step_idx=step_idx, record_event=record_event)
        before_pose = self.get_body_pose()
        before_qpos = self.env.sim.data.get_joint_qpos(self.target.joint_name).copy()
        body_id = self.env.sim.model.body_name2id(self.target.target_body_name)
        before_quat = self.env.sim.model.body_quat[body_id].copy()
        half = 0.5 * float(delta_yaw_rad)
        delta = np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)
        w1, x1, y1, z1 = delta
        w2, x2, y2, z2 = before_quat
        after_quat = np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
        after_quat /= max(1e-9, float(np.linalg.norm(after_quat)))
        self.env.sim.model.body_quat[body_id] = after_quat
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        after_pose = self.get_body_pose()
        event = {
            "reason": reason,
            "step_idx": step_idx,
            "target_object_name": self.target.object_name,
            "target_joint_name": self.target.joint_name,
            "target_body_name": self.target.target_body_name,
            "delta_yaw_rad": float(delta_yaw_rad),
            "before_pose": before_pose,
            "after_pose": after_pose,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": before_qpos.tolist(),
            "before_body_quat": before_quat.tolist(),
            "after_body_quat": after_quat.tolist(),
        }
        if record_event:
            self.applied_events.append(event)
        return event
