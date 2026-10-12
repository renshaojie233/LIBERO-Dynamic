#!/usr/bin/env python3
"""Run LIBERO-Dynamic with the native LIBERO step-by-step eval loop.

This smoke harness intentionally mirrors libero/lifelong/metric.py:

1. create OffScreenRenderEnv from the original BDDL task;
2. env.reset();
3. env.set_init_state(fixed_init_state);
4. run a small zero-action physics warmup;
5. every eval step asks policy.get_action(obs, language);
6. optionally notifies env.notify_inference_latency(...) with the measured
   model-inference time;
7. calls env.step(action);
8. success is read from done / env.check_success().
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from libero.libero.envs import OffScreenRenderEnv

from eval_dynamic_dummy_policy import (
    action_dim,
    build_items,
    get_joint_qpos,
    init_diag_state,
    render_frame,
    robot_contacting_target,
    summarize_diagnostics,
    target_bottom_z as diag_target_bottom_z,
    target_entries_for_preview,
    target_tilt_deg,
    task_bddl,
    update_diagnostics,
    write_effective_config_files,
)

try:
    from render_dynamic_config_previews import (
        align_target_bottom_to_support_plane as preview_align_target_bottom_to_support_plane,
        advance_physical_dynamic_time as preview_advance_physical_dynamic_time,
        append_fixture_supported_passenger_targets as preview_append_fixture_supported_passenger_targets,
        apply_carrier_override as preview_apply_carrier_override,
        apply_carrier_settings_to_targets as preview_apply_carrier_settings_to_targets,
        apply_initial_adjustments as preview_apply_initial_adjustments,
        apply_static_support_contact_override as preview_apply_static_support_contact_override,
        carrier_type_for_task as preview_carrier_type_for_task,
        has_support_contact as preview_has_support_contact,
        install_preview_platform_bodies,
        is_articulated_target as preview_is_articulated_target,
        movable_offset as preview_movable_offset,
        object_joint as preview_object_joint,
        prepare_target_states as preview_prepare_target_states,
        realign_targets_to_intended_xy as preview_realign_targets_to_intended_xy,
        record_motion_baselines as preview_record_motion_baselines,
        refresh_target_state_anchors as preview_refresh_target_state_anchors,
        set_platform_state as preview_set_platform_state,
        step_with_physical_dynamic as preview_step_with_physical_dynamic,
        step_with_platform_hold as preview_step_with_platform_hold,
        target_bottom_z as preview_target_bottom_z,
    )
except Exception:  # pragma: no cover - only needed by --preview-compatible-eval.
    preview_align_target_bottom_to_support_plane = None
    preview_advance_physical_dynamic_time = None
    preview_append_fixture_supported_passenger_targets = None
    preview_apply_carrier_override = None
    preview_apply_carrier_settings_to_targets = None
    preview_apply_initial_adjustments = None
    preview_apply_static_support_contact_override = None
    preview_carrier_type_for_task = None
    preview_has_support_contact = None
    install_preview_platform_bodies = None
    preview_is_articulated_target = None
    preview_movable_offset = None
    preview_object_joint = None
    preview_prepare_target_states = None
    preview_realign_targets_to_intended_xy = None
    preview_record_motion_baselines = None
    preview_refresh_target_state_anchors = None
    preview_set_platform_state = None
    preview_step_with_physical_dynamic = None
    preview_step_with_platform_hold = None
    preview_target_bottom_z = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from openpi_client import image_tools as openpi_image_tools
    from openpi_client import websocket_client_policy as openpi_websocket_client_policy
except Exception:  # pragma: no cover
    openpi_image_tools = None
    openpi_websocket_client_policy = None


@dataclass
class LiberoStylePolicy:
    action_dim: int
    mode: str
    random_scale: float
    seed: int
    simulated_inference_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def reset(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def get_action(self, obs: dict[str, Any], language: str) -> np.ndarray:
        del obs, language
        if self.simulated_inference_seconds > 0.0:
            time.sleep(float(self.simulated_inference_seconds))
        if self.mode == "zero":
            return np.zeros((int(self.action_dim),), dtype=np.float32)
        if self.mode == "random":
            action = self.rng.normal(0.0, float(self.random_scale), size=int(self.action_dim))
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        raise ValueError(f"Unsupported policy mode: {self.mode}")


class OpenPIWebsocketPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int,
        replan_steps: int,
        action_dim: int,
    ) -> None:
        if openpi_image_tools is None or openpi_websocket_client_policy is None:
            raise RuntimeError(
                "openpi_client is required for --policy openpi_ws. "
                "Install the openpi client package or add "
                "/path/to/openpi/packages/openpi-client/src to PYTHONPATH."
            )
        self.host = str(host)
        self.port = int(port)
        self.resize_size = int(resize_size)
        self.replan_steps = int(replan_steps)
        self.action_dim = int(action_dim)
        self.client = openpi_websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
        self.server_metadata = self.client.get_server_metadata()
        self.supports_rtc_conditioning = bool(
            self.server_metadata.get("rtc_conditioning_supported", False)
        )
        self.required_observation_history_frames = int(
            self.server_metadata.get("required_observation_history_frames", 1)
        )
        self.required_action_history_steps = int(
            self.server_metadata.get("required_action_history_steps", 0)
        )

    def reset(self) -> None:
        self.client.reset()

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        rotated = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
        resized = openpi_image_tools.resize_with_pad(rotated, self.resize_size, self.resize_size)
        return openpi_image_tools.convert_to_uint8(resized)

    def _single_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation/image": self._preprocess_image(obs["agentview_image"]),
            "observation/wrist_image": self._preprocess_image(obs["robot0_eye_in_hand_image"]),
            "observation/state": np.concatenate(
                (
                    np.asarray(obs["robot0_eef_pos"]),
                    _quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy()),
                    np.asarray(obs["robot0_gripper_qpos"]),
                )
            ),
        }

    def _observation(self, obs: dict[str, Any], language: str) -> dict[str, Any]:
        request = self._single_observation(obs)
        request["prompt"] = str(language)
        history = obs.get("__observation_history")
        if history:
            encoded = [self._single_observation(one_obs) for one_obs in history]
            request["observation/history/image"] = np.stack(
                [one["observation/image"] for one in encoded], axis=0
            )
            request["observation/history/wrist_image"] = np.stack(
                [one["observation/wrist_image"] for one in encoded], axis=0
            )
            request["observation/history/state"] = np.stack(
                [one["observation/state"] for one in encoded], axis=0
            )
        action_history = obs.get("__executed_action_history")
        if action_history is not None:
            request["observation/action_history"] = np.asarray(action_history, dtype=np.float32)
        return request

    def get_action_chunk(self, obs: dict[str, Any], language: str) -> np.ndarray:
        result = self.client.infer(self._observation(obs, language))
        return self._actions_from_result(result)

    def get_action_chunk_rtc(
        self,
        obs: dict[str, Any],
        language: str,
        *,
        prev_chunk_left_over: np.ndarray | None,
        inference_delay_steps: int,
    ) -> dict[str, np.ndarray]:
        request = self._observation(obs, language)
        request["prev_chunk_left_over"] = (
            np.asarray(prev_chunk_left_over, dtype=np.float32)
            if prev_chunk_left_over is not None
            else np.zeros((0, self.action_dim), dtype=np.float32)
        )
        request["inference_delay_steps"] = int(inference_delay_steps)
        result = self.client.infer(request)
        actions = self._actions_from_result(result)
        original_actions = result.get("original_actions")
        if original_actions is None:
            return {"actions": actions}
        original = np.asarray(original_actions, dtype=np.float32)
        if original.ndim == 1:
            original = original.reshape(1, -1)
        if len(original) < self.replan_steps:
            raise RuntimeError(
                f"OpenPI returned {len(original)} original actions, shorter than --replan-steps {self.replan_steps}."
            )
        return {
            "actions": actions,
            "original_actions": original[: self.replan_steps, : self.action_dim],
        }

    def _actions_from_result(self, result: dict[str, Any]) -> np.ndarray:
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if len(actions) < self.replan_steps:
            raise RuntimeError(
                f"OpenPI returned {len(actions)} actions, shorter than --replan-steps {self.replan_steps}."
            )
        return actions[: self.replan_steps, : self.action_dim]


def snapshot_openpi_obs(obs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    return {key: np.array(obs[key], copy=True) for key in keys}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def obs_frame(obs: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(obs["agentview_image"])[::-1, ::-1]
    return np.ascontiguousarray(frame)


def resize_video_frame(frame: np.ndarray, size: int) -> np.ndarray:
    if int(size) <= 0:
        return np.ascontiguousarray(frame)
    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    if height == int(size) and width == int(size):
        return np.ascontiguousarray(frame)
    y_idx = np.linspace(0, height - 1, int(size)).astype(np.int64)
    x_idx = np.linspace(0, width - 1, int(size)).astype(np.int64)
    return np.ascontiguousarray(frame[y_idx][:, x_idx])


def model_input_frame(obs: dict[str, Any], layout: str, size: int) -> np.ndarray:
    agent = resize_video_frame(np.asarray(obs["agentview_image"])[::-1, ::-1], int(size))
    if str(layout) != "dual":
        return agent
    wrist = resize_video_frame(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1], int(size))
    return np.ascontiguousarray(np.concatenate([agent, wrist], axis=1))


def park_physical_platforms_below_scene(env: Any) -> None:
    for state in getattr(env, "dynamic_multi_target_states", []) or []:
        target_cfg = state.get("target_cfg") or {}
        if str(target_cfg.get("dynamic_role", "")) == "fixture_supported_passenger_conveyor":
            continue
        platform = state.get("platform") or {}
        joint_names = platform.get("joint_names") or {}
        if not joint_names:
            continue
        for axis, value in (("x", 0.0), ("y", 0.0), ("z", -1.0)):
            joint_name = joint_names.get(axis)
            if joint_name:
                env.sim.data.set_joint_qpos(joint_name, float(value))
                try:
                    env.sim.data.set_joint_qvel(joint_name, 0.0)
                except Exception:
                    pass
        state["platform_base_pos"] = None
        state["last_offset"] = np.zeros(3, dtype=float)
        state["last_velocity"] = np.zeros(3, dtype=float)
        state["cumulative_shift"] = np.zeros(3, dtype=float)
    env.sim.forward()


PREVIEW_COMPATIBLE_TARGET_KEYWORDS = (
    "moka_pot",
    "mug",
    "wine_bottle",
)

PREVIEW_COMPATIBLE_CARRIER_TYPES = {
    "physical_conveyor_patch",
    "physical_platform",
    "freejoint_platform",
}

RTC_QUEUE_LOGIC = (
    "Asynchronous action-chunk queue: one OpenPI request produces a chunk of "
    "robot actions; while that request is in flight, the simulator keeps "
    "stepping dynamic objects and executes already queued robot actions. If the "
    "queue is empty before the next chunk becomes available, rtc_empty_action is "
    "executed and rtc_underflow_steps is incremented. This mode does not use the "
    "ordinary freeze-during-inference latency path."
)


def apply_reviewed_old_physics_profile(args: argparse.Namespace) -> None:
    """Use the reviewed preview-physics route for every dynamic target."""
    args.preview_compatible_eval = True
    args.preview_compatible_scope = "all"
    args.keep_static_support_contact = True
    args.static_support_clearance_m = 0.001
    args.platform_settle_steps = 40
    args.preview_compatible_settle_steps = 80
    args.num_steps_wait = 10
    args.pre_policy_dynamic_steps = 40
    args.record_latency_frames = True


def preview_compatible_eval_available() -> bool:
    return all(
        func is not None
        for func in (
            preview_align_target_bottom_to_support_plane,
            preview_append_fixture_supported_passenger_targets,
            preview_apply_carrier_override,
            preview_apply_carrier_settings_to_targets,
            preview_apply_initial_adjustments,
            preview_apply_static_support_contact_override,
            preview_carrier_type_for_task,
            preview_has_support_contact,
            install_preview_platform_bodies,
            preview_is_articulated_target,
            preview_movable_offset,
            preview_object_joint,
            preview_prepare_target_states,
            preview_realign_targets_to_intended_xy,
            preview_record_motion_baselines,
            preview_refresh_target_state_anchors,
            preview_set_platform_state,
            preview_step_with_physical_dynamic,
            preview_step_with_platform_hold,
            preview_target_bottom_z,
        )
    )


def item_target_names(item: dict[str, Any]) -> list[str]:
    if target_entries_for_preview is None:
        targets = item["task_cfg"].get("target_objects", [])
    else:
        targets = target_entries_for_preview(item["task_cfg"])
    names = []
    for target in targets:
        name = target.get("target_object_name") or target.get("object_name")
        if name:
            names.append(str(name))
    return names


def item_preview_carrier_type(item: dict[str, Any]) -> str | None:
    if preview_apply_carrier_override is None or preview_carrier_type_for_task is None:
        return None
    try:
        task_cfg = preview_apply_carrier_override(item["task_cfg"], None)
        return str(preview_carrier_type_for_task(task_cfg))
    except Exception:
        return None


def should_use_preview_compatible_eval(item: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(args.preview_compatible_eval):
        return False
    if item_preview_carrier_type(item) not in PREVIEW_COMPATIBLE_CARRIER_TYPES:
        return False
    if str(getattr(args, "preview_compatible_scope", "fragile")) == "all":
        return True
    names = item_target_names(item)
    return any(keyword in name for name in names for keyword in PREVIEW_COMPATIBLE_TARGET_KEYWORDS)


def update_preview_state_offsets(states: list[dict[str, Any]], elapsed: float) -> None:
    if preview_movable_offset is None:
        return
    for state in states:
        state["last_offset"] = preview_movable_offset(state.get("motion", {}) or {}, float(elapsed))


def update_preview_compatible_diagnostics(
    env: Any,
    diag: dict[str, dict[str, Any]],
    states: list[dict[str, Any]],
    elapsed: float,
) -> None:
    if preview_movable_offset is None:
        return
    for state in states:
        name = state["name"]
        if name not in diag:
            continue
        entry = diag[name]
        baseline = entry["baseline"]
        entry["samples"] += 1

        if entry.get("robot_contacted") or robot_contacting_target(env, name):
            if not entry.get("robot_contacted"):
                entry["robot_contacted"] = True
                entry["robot_contact_sample"] = int(entry["samples"])
            entry["samples_after_robot_contact"] = int(entry.get("samples_after_robot_contact", 0)) + 1
            continue

        qpos = get_joint_qpos(env, state["joint"])
        object_base_xy = state.get("object_base_xy")
        if qpos is not None and qpos.size >= 2 and object_base_xy is not None:
            desired_xy = np.asarray(object_base_xy, dtype=float)[:2] + preview_movable_offset(
                state.get("motion", {}) or {},
                float(elapsed),
            )[:2]
            follow_error = float(np.linalg.norm(qpos[:2] - desired_xy))
            entry["max_follow_error_m"] = max(float(entry["max_follow_error_m"]), follow_error)
            entry["max_tilt_deg"] = max(float(entry["max_tilt_deg"]), target_tilt_deg(qpos, baseline.get("qpos")))

        bottom = diag_target_bottom_z(env, name)
        if bottom is not None and baseline.get("bottom_z") is None:
            baseline["bottom_z"] = float(bottom)
            baseline["bottom_baseline_source"] = "first_sample"
        if bottom is not None and baseline.get("bottom_z") is not None:
            rel = float(bottom) - float(baseline["bottom_z"])
            entry["min_relative_bottom_z_m"] = min(float(entry["min_relative_bottom_z_m"]), rel)


def setup_preview_compatible_dynamic(
    env: Any,
    item: dict[str, Any],
    args: argparse.Namespace,
    init_state: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    if not preview_compatible_eval_available():
        raise RuntimeError("render_dynamic_config_previews helpers are required for preview-compatible eval")
    if target_entries_for_preview is None:
        raise RuntimeError("target_entries_for_preview is required for preview-compatible eval")

    task_cfg = preview_apply_carrier_override(item["task_cfg"], None)
    carrier_type = preview_carrier_type_for_task(task_cfg)
    targets = target_entries_for_preview(task_cfg)
    targets = preview_apply_carrier_settings_to_targets(targets, task_cfg.get("carrier", {}) or {})
    use_physical_platforms = carrier_type in PREVIEW_COMPATIBLE_CARRIER_TYPES
    if not use_physical_platforms:
        raise RuntimeError(f"preview-compatible eval only supports physical carriers, got {carrier_type!r}")

    obs = env.set_init_state(init_state)
    preview_apply_initial_adjustments(env, task_cfg)
    action = np.zeros((action_dim(env),), dtype=np.float32)

    targets = preview_apply_static_support_contact_override(
        targets,
        keep_static_support_contact=bool(args.keep_static_support_contact),
        static_support_clearance_m=float(args.static_support_clearance_m),
    )
    for target in targets:
        name = str(target.get("target_object_name") or target.get("object_name"))
        try:
            joint_name = str(target.get("target_joint_name") or preview_object_joint(env, name))
            base_qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name).copy(), dtype=float)
        except Exception:
            base_qpos = None
        if preview_is_articulated_target(target, base_qpos):
            continue
        preview_align_target_bottom_to_support_plane(env, name)

    platforms: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    settle_on_platform = bool(
        (task_cfg.get("carrier", {}) or {}).get("settle_on_platform", False)
        or task_cfg.get("settle_on_platform", False)
    )
    if settle_on_platform:
        platforms = install_preview_platform_bodies(env, targets)
        states = preview_prepare_target_states(env, targets, platforms, carrier_type=carrier_type)
        for _ in range(max(0, int(args.preview_compatible_settle_steps))):
            obs, _, _, _ = preview_step_with_platform_hold(env, action, states)
    else:
        for _ in range(max(0, int(args.preview_compatible_settle_steps))):
            obs, _, _, _ = env.step(action)

    env.sim.forward()
    if not settle_on_platform:
        targets = preview_append_fixture_supported_passenger_targets(env, targets)
        targets = preview_apply_carrier_settings_to_targets(targets, task_cfg.get("carrier", {}) or {})
        targets = preview_apply_static_support_contact_override(
            targets,
            keep_static_support_contact=bool(args.keep_static_support_contact),
            static_support_clearance_m=float(args.static_support_clearance_m),
        )

    for target in targets:
        name = str(target.get("target_object_name") or target.get("object_name"))
        try:
            joint_name = str(target.get("target_joint_name") or preview_object_joint(env, name))
            base_qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name).copy(), dtype=float)
        except Exception:
            base_qpos = None
        if preview_is_articulated_target(target, base_qpos):
            continue
        active, contacts = preview_has_support_contact(env, name)
        if active and contacts:
            target["preinstall_support_contact_z"] = float(
                np.median([contact.get("pos", [0.0, 0.0, 0.9])[2] for contact in contacts])
            )
        try:
            formal_qpos = np.asarray(env.sim.data.get_joint_qpos(preview_object_joint(env, name)).copy(), dtype=float)
            if formal_qpos.size >= 3:
                target["formal_start_z"] = float(formal_qpos[2])
            formal_bottom = preview_target_bottom_z(env, name)
            if formal_bottom is not None:
                target["formal_start_bottom_z"] = float(formal_bottom)
        except Exception:
            pass

    if not platforms:
        platforms = install_preview_platform_bodies(env, targets)
    try:
        obs = env.env._get_observations()
    except Exception:
        pass
    if not states:
        states = preview_prepare_target_states(env, targets, platforms, carrier_type=carrier_type)
    for _ in range(max(0, int(args.platform_settle_steps))):
        obs, _, _, _ = preview_step_with_platform_hold(env, action, states)
    if bool(args.realign_after_platform_settle) or use_physical_platforms:
        preview_realign_targets_to_intended_xy(env, states)
        preview_refresh_target_state_anchors(env, states)
        for state in states:
            platform = state.get("platform")
            base_pos = state.get("platform_base_pos")
            if platform is not None and base_pos is not None:
                preview_set_platform_state(env, platform, np.asarray(base_pos, dtype=float), np.zeros(3, dtype=float))
        env.sim.forward()
    preview_refresh_target_state_anchors(env, states)
    preview_record_motion_baselines(env, states)
    update_preview_state_offsets(states, 0.0)
    env.dynamic_multi_target_states = states
    return obs, states, int(args.preview_compatible_settle_steps)


def build_policy(args: argparse.Namespace, env: Any, item: dict[str, Any]) -> Any:
    if args.policy in {"openpi_ws", "openpi_ws_rtc"}:
        return OpenPIWebsocketPolicy(
            host=str(args.openpi_host),
            port=int(args.openpi_port),
            resize_size=int(args.openpi_resize_size),
            replan_steps=int(args.replan_steps),
            action_dim=action_dim(env),
        )
    return LiberoStylePolicy(
        action_dim=action_dim(env),
        mode=str(args.policy),
        random_scale=float(args.random_scale),
        seed=int(args.seed or 0) + int(item["task_id"]) * 100 + int(item["traj_index"]),
        simulated_inference_seconds=float(args.simulated_inference_seconds),
    )


def run_item(item: dict[str, Any], args_dict: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    use_preview_compatible_eval = should_use_preview_compatible_eval(item, args)
    if use_preview_compatible_eval:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = str(args.output_dir / "_preview_compatible_eval_wrapper_disabled.yaml")
    else:
        os.environ["LIBERO_DYNAMIC_CONFIG"] = item["config_path"]
    os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(item["traj_index"])
    os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = item["variant_id"]
    os.environ.setdefault("LIBERO_DYNAMIC_AUTO_FINALIZE", "false")

    suite, task, bddl = task_bddl(item["suite"], item["task_id"])
    init_states = suite.get_task_init_states(item["task_id"])
    video_dir = args.output_dir / item["level"] / item["suite"] / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{item['item_id']}--libero-style-{args.policy}.mp4"
    model_input_video_dir = args.output_dir / item["level"] / item["suite"] / "model_input_videos"
    model_input_video_path = (
        model_input_video_dir
        / f"{item['item_id']}--modelinput-{args.model_input_video_layout}-{args.policy}.mp4"
    )
    manifest_path = args.output_dir / item["level"] / item["suite"] / "manifests" / f"{item['item_id']}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    env = None
    writer = None
    model_input_writer = None
    started = time.time()
    step_count = 0
    warmup_steps_done = 0
    phase_prime_steps_used = 0
    pre_policy_dynamic_steps_done = 0
    success = False
    error = None
    error_traceback = None
    diagnostics = None
    inference_latency_samples: list[float] = []
    env_latency_samples: list[float] = []
    replan_count = 0
    rtc_underflow_steps = 0
    rtc_prefetch_count = 0
    rtc_executor: ThreadPoolExecutor | None = None
    rtc_future: Future | None = None
    rtc_submit_step = 0
    rtc_future_elapsed: float | None = None
    rtc_available_step: int | None = None
    preview_states: list[dict[str, Any]] = []
    preview_elapsed = 0.0
    preview_dt = 1.0 / float(args.control_freq)
    preview_compatible_settle_steps_done = 0
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=int(args.camera_size),
            camera_widths=int(args.camera_size),
            control_freq=int(args.control_freq),
        )
        if args.seed is not None:
            env.seed(int(args.seed))
        obs = env.reset()
        if use_preview_compatible_eval:
            obs, preview_states, preview_compatible_settle_steps_done = setup_preview_compatible_dynamic(
                env,
                item,
                args,
                init_states[int(args.init_state_id)],
            )
        else:
            obs = env.set_init_state(init_states[int(args.init_state_id)])
        policy = build_policy(args, env, item)
        policy.reset()
        diagnostics = init_diag_state(env, args)

        if bool(args.save_videos):
            if imageio is None:
                raise RuntimeError("imageio is required when --save-videos is enabled")
            if bool(args.save_review_video):
                writer = imageio.get_writer(video_path, fps=float(args.video_fps), macro_block_size=None)
            if bool(args.save_model_input_videos):
                model_input_video_dir.mkdir(parents=True, exist_ok=True)
                model_input_writer = imageio.get_writer(
                    model_input_video_path,
                    fps=float(args.video_fps),
                    macro_block_size=None,
                )

        zero_action = np.zeros((action_dim(env),), dtype=np.float32)

        def step_env(action: np.ndarray, *, hold_preview_platforms: bool = False) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            nonlocal preview_elapsed
            if use_preview_compatible_eval:
                if hold_preview_platforms:
                    step_obs, reward, done, info = preview_step_with_platform_hold(env, action, preview_states)
                else:
                    step_obs, reward, done, info = preview_step_with_physical_dynamic(
                        env,
                        action,
                        preview_states,
                        preview_elapsed,
                        preview_dt,
                    )
                    preview_elapsed += preview_dt
                    update_preview_state_offsets(preview_states, preview_elapsed)
                    env.dynamic_multi_target_states = preview_states
                return step_obs, reward, done, info
            return env.step(action)

        def advance_preview_latency(latency_seconds: float) -> None:
            nonlocal obs, success
            if not use_preview_compatible_eval or str(args.latency_mode) == "disabled":
                return
            delay_steps = int(math.ceil(max(0.0, float(latency_seconds)) * float(args.control_freq)))
            if delay_steps <= 0:
                return
            for _ in range(delay_steps):
                obs, _reward, done, info = step_env(zero_action)
                env_latency_samples.append(float((info or {}).get("dynamic_latency_seconds", 0.0) or 0.0))
                success = bool(done) or bool(env.check_success())
                if diagnostics is not None:
                    update_preview_compatible_diagnostics(env, diagnostics, preview_states, preview_elapsed)
                if (writer is not None or model_input_writer is not None) and bool(args.record_latency_frames):
                    if args.video_style == "review" and render_frame is not None:
                        if writer is not None:
                            writer.append_data(render_frame(env, obs, int(args.camera_size), None, overlay_traces=False))
                    else:
                        if writer is not None:
                            writer.append_data(obs_frame(obs))
                    if model_input_writer is not None:
                        model_input_writer.append_data(
                            model_input_frame(obs, args.model_input_video_layout, int(args.model_input_video_size))
                        )
                if success and bool(args.stop_on_success):
                    break

        if (
            int(args.num_steps_wait) > 0
            and bool(args.reset_platforms_after_warmup)
            and getattr(env, "dynamic_multi_target_states", None)
            and not use_preview_compatible_eval
        ):
            park_physical_platforms_below_scene(env)
        for _ in range(int(args.num_steps_wait)):
            obs, _reward, done, _info = step_env(zero_action, hold_preview_platforms=use_preview_compatible_eval)
            warmup_steps_done += 1
            success = bool(done) or bool(env.check_success())
            if (writer is not None or model_input_writer is not None) and bool(args.record_warmup):
                if args.video_style == "review" and render_frame is not None:
                    if writer is not None:
                        writer.append_data(render_frame(env, obs, int(args.camera_size), None, overlay_traces=False))
                else:
                    if writer is not None:
                        writer.append_data(obs_frame(obs))
                if model_input_writer is not None:
                    model_input_writer.append_data(
                        model_input_frame(obs, args.model_input_video_layout, int(args.model_input_video_size))
                    )
            if success and bool(args.stop_on_success):
                break
        if (
            warmup_steps_done > 0
            and bool(args.reset_platforms_after_warmup)
            and hasattr(env, "append_fixture_supported_passenger_targets_from_contacts")
            and not use_preview_compatible_eval
        ):
            fixture_passengers_added = env.append_fixture_supported_passenger_targets_from_contacts()
            obs = env.env._get_observations() if hasattr(env, "env") else obs
        else:
            fixture_passengers_added = []
        if (
            warmup_steps_done > 0
            and bool(args.reset_platforms_after_warmup)
            and getattr(env, "dynamic_multi_target_states", None)
            and hasattr(env, "_reset_dynamic_physical_platform_poses")
            and not use_preview_compatible_eval
        ):
            env._reset_dynamic_physical_platform_poses()
            obs = env.env._get_observations() if hasattr(env, "env") else obs
        phase_prime_steps_used = int(args.dynamic_phase_prime_steps)
        if fixture_passengers_added and args.fixture_passenger_phase_prime_steps is not None:
            phase_prime_steps_used = int(args.fixture_passenger_phase_prime_steps)
        if (
            int(phase_prime_steps_used) > 0
            and getattr(env, "dynamic_multi_target_states", None)
            and hasattr(env, "prime_dynamic_physical_platform_phase")
            and not use_preview_compatible_eval
        ):
            env.prime_dynamic_physical_platform_phase(int(phase_prime_steps_used))
            obs = env.env._get_observations() if hasattr(env, "env") else obs
        if int(args.pre_policy_dynamic_steps) > 0 and getattr(env, "dynamic_multi_target_states", None):
            for _ in range(int(args.pre_policy_dynamic_steps)):
                obs, _reward, done, _info = step_env(zero_action)
                pre_policy_dynamic_steps_done += 1
                success = bool(done) or bool(env.check_success())
                if (writer is not None or model_input_writer is not None) and bool(args.record_warmup):
                    if args.video_style == "review" and render_frame is not None:
                        if writer is not None:
                            writer.append_data(render_frame(env, obs, int(args.camera_size), None, overlay_traces=False))
                    else:
                        if writer is not None:
                            writer.append_data(obs_frame(obs))
                    if model_input_writer is not None:
                        model_input_writer.append_data(
                            model_input_frame(obs, args.model_input_video_layout, int(args.model_input_video_size))
                        )
                if success and bool(args.stop_on_success):
                    break
        diagnostics = init_diag_state(env, args)

        action_plan: collections.deque[np.ndarray] = collections.deque()
        last_action = np.array(zero_action, copy=True)
        rtc_mode = str(args.policy) == "openpi_ws_rtc"
        rtc_prefetch_threshold = max(0, int(args.rtc_prefetch_threshold))

        def submit_rtc_request(current_obs: dict[str, Any]) -> None:
            nonlocal rtc_future, rtc_submit_step, rtc_future_elapsed, rtc_available_step, rtc_prefetch_count
            if rtc_future is not None:
                return
            rtc_submit_step = int(step_count)
            rtc_future_elapsed = None
            rtc_available_step = None
            obs_snapshot = snapshot_openpi_obs(current_obs)

            def infer_chunk() -> tuple[np.ndarray, float]:
                infer_started = time.perf_counter()
                chunk = policy.get_action_chunk(obs_snapshot, task.language)
                return np.asarray(chunk, dtype=np.float32), time.perf_counter() - infer_started

            rtc_future = rtc_executor.submit(infer_chunk)  # type: ignore[union-attr]
            rtc_prefetch_count += 1

        def consume_rtc_if_ready() -> None:
            nonlocal rtc_future, rtc_future_elapsed, rtc_available_step, replan_count
            if rtc_future is None or not rtc_future.done():
                return
            if rtc_future_elapsed is None:
                action_chunk, infer_elapsed = rtc_future.result()
                rtc_future._libero_action_chunk = action_chunk  # type: ignore[attr-defined]
                rtc_future_elapsed = float(infer_elapsed)
                inference_latency_samples.append(float(infer_elapsed))
                delay_steps = int(math.ceil(max(0.0, float(infer_elapsed)) * float(args.control_freq)))
                rtc_available_step = int(rtc_submit_step) + max(0, delay_steps)
            if rtc_available_step is not None and int(step_count) >= int(rtc_available_step):
                action_chunk = getattr(rtc_future, "_libero_action_chunk")
                action_plan.extend(np.asarray(action, dtype=np.float32) for action in action_chunk)
                replan_count += 1
                rtc_future = None
                rtc_future_elapsed = None
                rtc_available_step = None

        if rtc_mode:
            rtc_executor = ThreadPoolExecutor(max_workers=1)
            submit_rtc_request(obs)

        while step_count < int(args.max_steps) and not (success and bool(args.stop_on_success)):
            if rtc_mode:
                consume_rtc_if_ready()
                if (
                    rtc_future is None
                    and len(action_plan) <= rtc_prefetch_threshold
                    and not (success and bool(args.stop_on_success))
                ):
                    submit_rtc_request(obs)
            elif not action_plan:
                infer_started = time.perf_counter()
                if args.policy == "openpi_ws":
                    action_chunk = policy.get_action_chunk(obs, task.language)
                else:
                    action_chunk = np.asarray([policy.get_action(obs, task.language)], dtype=np.float32)
                infer_elapsed = time.perf_counter() - infer_started
                inference_latency_samples.append(float(infer_elapsed))
                replan_count += 1
                if hasattr(env, "notify_inference_latency"):
                    if args.latency_mode == "measured":
                        env.notify_inference_latency(float(infer_elapsed))
                    elif args.latency_mode == "fixed":
                        fixed_latency = getattr(args, "fixed_inference_seconds", None)
                        env.notify_inference_latency(float(fixed_latency) if fixed_latency is not None else None)
                if use_preview_compatible_eval:
                    if str(args.latency_mode) == "measured":
                        advance_preview_latency(float(infer_elapsed))
                    elif str(args.latency_mode) == "fixed":
                        fixed_latency = getattr(args, "fixed_inference_seconds", None)
                        if fixed_latency is not None:
                            advance_preview_latency(float(fixed_latency))
                action_plan.extend(np.asarray(action, dtype=np.float32) for action in action_chunk)
            if action_plan:
                action = np.asarray(action_plan.popleft(), dtype=np.float32)[: action_dim(env)]
                last_action = np.array(action, copy=True)
            elif rtc_mode:
                rtc_underflow_steps += 1
                if str(args.rtc_empty_action) == "zero":
                    action = np.array(zero_action, copy=True)
                else:
                    action = np.array(last_action, copy=True)
            else:
                action = np.asarray(action_plan.popleft(), dtype=np.float32)[: action_dim(env)]
            obs, _reward, done, info = step_env(action)
            env_latency_samples.append(float((info or {}).get("dynamic_latency_seconds", 0.0) or 0.0))
            success = bool(done) or bool(env.check_success())
            if use_preview_compatible_eval:
                update_preview_compatible_diagnostics(env, diagnostics, preview_states, preview_elapsed)
            else:
                update_diagnostics(env, diagnostics)
            if (writer is not None or model_input_writer is not None) and (
                step_count % max(1, int(args.render_every)) == 0
            ):
                if args.video_style == "review" and render_frame is not None:
                    if writer is not None:
                        writer.append_data(render_frame(env, obs, int(args.camera_size), None, overlay_traces=False))
                else:
                    if writer is not None:
                        writer.append_data(obs_frame(obs))
                if model_input_writer is not None:
                    model_input_writer.append_data(
                        model_input_frame(obs, args.model_input_video_layout, int(args.model_input_video_size))
                    )
            step_count += 1

        diagnostic_summary = summarize_diagnostics(diagnostics, args)
        status = "ok"
    except Exception as exc:
        status = "error"
        error = repr(exc)
        error_traceback = traceback.format_exc()
        diagnostic_summary = {"targets": {}, "flags": [{"type": "exception", "value": error}], "ok": False}
    finally:
        if writer is not None:
            writer.close()
        if model_input_writer is not None:
            model_input_writer.close()
        if rtc_executor is not None:
            rtc_executor.shutdown(wait=False, cancel_futures=True)
        if env is not None:
            env.close()

    result = {
        **{k: item[k] for k in ("item_id", "suite", "task_id", "level", "traj_index", "variant_name", "variant_id")},
        "status": status,
        "error": error,
        "error_traceback": error_traceback,
        "language": getattr(task, "language", ""),
        "eval_loop": "libero_step_by_step",
        "dynamic_eval_route": "preview_compatible" if use_preview_compatible_eval else "env_wrapper",
        "preview_compatible_settle_steps": int(args.preview_compatible_settle_steps),
        "preview_compatible_settle_steps_done": int(preview_compatible_settle_steps_done),
        "policy": str(args.policy),
        "openpi_host": str(args.openpi_host) if args.policy in {"openpi_ws", "openpi_ws_rtc"} else None,
        "openpi_port": int(args.openpi_port) if args.policy in {"openpi_ws", "openpi_ws_rtc"} else None,
        "openpi_resize_size": int(args.openpi_resize_size) if args.policy in {"openpi_ws", "openpi_ws_rtc"} else None,
        "replan_steps_requested": int(args.replan_steps),
        "rtc_enabled": str(args.policy) == "openpi_ws_rtc",
        "rtc_logic": RTC_QUEUE_LOGIC if str(args.policy) == "openpi_ws_rtc" else None,
        "rtc_prefetch_threshold": int(args.rtc_prefetch_threshold),
        "rtc_empty_action": str(args.rtc_empty_action),
        "rtc_prefetch_count": int(rtc_prefetch_count),
        "rtc_underflow_steps": int(rtc_underflow_steps),
        "random_scale": float(args.random_scale),
        "latency_mode": str(args.latency_mode),
        "fixed_inference_seconds": None
        if getattr(args, "fixed_inference_seconds", None) is None
        else float(args.fixed_inference_seconds),
        "simulated_inference_seconds": float(args.simulated_inference_seconds),
        "mean_policy_inference_seconds": 0.0
        if not inference_latency_samples
        else round(float(np.mean(inference_latency_samples)), 6),
        "max_policy_inference_seconds": 0.0
        if not inference_latency_samples
        else round(float(np.max(inference_latency_samples)), 6),
        "mean_env_latency_seconds": 0.0 if not env_latency_samples else round(float(np.mean(env_latency_samples)), 6),
        "max_env_latency_seconds": 0.0 if not env_latency_samples else round(float(np.max(env_latency_samples)), 6),
        "replans": int(replan_count),
        "init_state_id": int(args.init_state_id),
        "num_steps_wait": int(args.num_steps_wait),
        "warmup_steps_done": int(warmup_steps_done),
        "pre_policy_dynamic_steps": int(args.pre_policy_dynamic_steps),
        "pre_policy_dynamic_steps_done": int(pre_policy_dynamic_steps_done),
        "dynamic_phase_prime_steps": int(phase_prime_steps_used),
        "dynamic_phase_prime_steps_requested": int(args.dynamic_phase_prime_steps),
        "fixture_passenger_phase_prime_steps": (
            None
            if args.fixture_passenger_phase_prime_steps is None
            else int(args.fixture_passenger_phase_prime_steps)
        ),
        "steps": int(step_count),
        "max_steps": int(args.max_steps),
        "control_freq": int(args.control_freq),
        "success": bool(success),
        "keep_static_support_contact": bool(args.keep_static_support_contact),
        "static_support_clearance_m": float(args.static_support_clearance_m),
        "effective_config_path": str(item["config_path"]),
        "target_names": [
            target.get("target_object_name")
            for target in (
                target_entries_for_preview(item["task_cfg"])
                if target_entries_for_preview is not None
                else item["task_cfg"].get("target_objects", [])
            )
        ],
        "diagnostics": diagnostic_summary,
        "video": str(video_path) if bool(args.save_videos) and bool(args.save_review_video) and status == "ok" else None,
        "model_input_video": (
            str(model_input_video_path)
            if bool(args.save_videos) and bool(args.save_model_input_videos) and status == "ok"
            else None
        ),
        "model_input_video_layout": str(args.model_input_video_layout),
        "model_input_video_size": int(args.model_input_video_size),
        "manifest": str(manifest_path),
        "elapsed_seconds": round(time.time() - started, 3),
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
        "success_count": sum(1 for row in results if row.get("success")),
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


def main(argv: list[str] | None = None, *, reviewed_old_physics: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", type=Path, default=repo_root / "libero" / "libero" / "dynamic_configs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--traj-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--policy", choices=("zero", "random", "openpi_ws", "openpi_ws_rtc"), default="zero")
    parser.add_argument("--random-scale", type=float, default=0.02)
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", type=int, default=8001)
    parser.add_argument("--openpi-resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--rtc-prefetch-threshold",
        type=int,
        default=5,
        help="For --policy openpi_ws_rtc, submit the next async chunk when queued actions are at or below this count.",
    )
    parser.add_argument(
        "--rtc-empty-action",
        choices=("hold", "zero"),
        default="hold",
        help="For --policy openpi_ws_rtc, command to use if the action queue underflows before the next chunk is ready.",
    )
    parser.add_argument(
        "--simulated-inference-seconds",
        type=float,
        default=0.0,
        help="Optional sleep inside policy.get_action to emulate model inference time for latency testing.",
    )
    parser.add_argument(
        "--latency-mode",
        choices=("disabled", "measured", "fixed"),
        default="disabled",
        help=(
            "disabled: do not notify latency; measured: notify measured policy.get_action time; "
            "fixed: use --fixed-inference-seconds when provided, otherwise notify None so env uses configured fixed latency level."
        ),
    )
    parser.add_argument(
        "--fixed-inference-seconds",
        type=float,
        default=None,
        help=(
            "For --latency-mode fixed, advance dynamic physics by this calibrated policy latency "
            "without sleeping or using multi-worker queueing time."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--num-steps-wait", type=int, default=5, help="Native LIBERO zero-action physics warmup before policy rollout.")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-review-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --save-videos is enabled, save the third-person/review video.",
    )
    parser.add_argument(
        "--save-model-input-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save the LIBERO agentview frames used as the model/policy image input.",
    )
    parser.add_argument(
        "--model-input-video-layout",
        choices=("agentview", "dual"),
        default="agentview",
        help="agentview: save only the policy agentview; dual: save agentview and wrist side by side.",
    )
    parser.add_argument(
        "--model-input-video-size",
        type=int,
        default=0,
        help="Optional square output size for each model-input view before video encoding. 0 keeps camera size.",
    )
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--record-warmup", action="store_true")
    parser.add_argument(
        "--record-latency-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For preview-compatible measured latency, include zero-action latency frames in saved videos.",
    )
    parser.add_argument(
        "--dynamic-phase-prime-steps",
        type=int,
        default=0,
        help=(
            "After warmup/platform reset, prime dynamic platform phase to this control step "
            "without pre-drifting objects, so policy starts with nonzero conveyor velocity."
        ),
    )
    parser.add_argument(
        "--fixture-passenger-phase-prime-steps",
        type=int,
        default=None,
        help=(
            "Override dynamic phase priming for items where fixture-supported passenger "
            "objects are discovered after warmup. Use 0 to preserve the reviewed 8850 "
            "cabinet/stove timing while keeping phase priming for ordinary items."
        ),
    )
    parser.add_argument(
        "--pre-policy-dynamic-steps",
        type=int,
        default=0,
        help=(
            "After platform reset and optional phase priming, step the full environment "
            "with zero robot action before querying the policy. This lets moving objects "
            "reach an in-motion physical state without desynchronizing fixture passengers."
        ),
    )
    parser.add_argument(
        "--reset-platforms-after-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the LIBERO settling warmup, re-anchor physical conveyor platforms to the settled objects.",
    )
    parser.add_argument("--video-style", choices=("libero", "review"), default="libero")
    parser.add_argument("--keep-static-support-contact", action="store_true")
    parser.add_argument("--static-support-clearance-m", type=float, default=0.0)
    parser.add_argument("--platform-settle-steps", type=int, default=20)
    parser.add_argument("--realign-after-platform-settle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preview-compatible-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the reviewed preview-platform setup for fragile tall carried objects "
            "such as moka pots and mugs, while leaving ordinary objects on the env wrapper route."
        ),
    )
    parser.add_argument(
        "--preview-compatible-scope",
        choices=("fragile", "all"),
        default="fragile",
        help="fragile: route only selected fragile object types; all: route every dynamic item through preview-compatible physics.",
    )
    parser.add_argument(
        "--preview-compatible-settle-steps",
        type=int,
        default=80,
        help="Zero-action settle steps before installing preview-compatible dynamic platforms.",
    )
    parser.add_argument("--platform-error-threshold-m", type=float, default=0.01)
    parser.add_argument("--follow-error-threshold-m", type=float, default=0.08)
    parser.add_argument("--tilt-threshold-deg", type=float, default=35.0)
    parser.add_argument("--drop-threshold-m", type=float, default=0.04)
    parser.add_argument("--bottom-baseline-mode", choices=("first_sample", "reset"), default="first_sample")
    parser.add_argument("--offscreen-ratio-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)
    if reviewed_old_physics:
        apply_reviewed_old_physics_profile(args)

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
            result = run_item(item, args_dict)
            print(
                f"  status={result.get('status')} success={result.get('success')} "
                f"diag_ok={result.get('diagnostics', {}).get('ok')}",
                flush=True,
            )
            results.append(result)
            write_summary(args.output_dir, results)
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as pool:
            futures = {pool.submit(run_item, item, args_dict): item for item in items}
            for idx, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **item,
                        "status": "error",
                        "error": repr(exc),
                        "diagnostics": {"ok": False, "flags": [{"type": "worker_exception", "value": repr(exc)}]},
                    }
                print(
                    f"[{idx}/{len(items)}] {result['item_id']} status={result.get('status')} "
                    f"success={result.get('success')} diag_ok={result.get('diagnostics', {}).get('ok')}",
                    flush=True,
                )
                results.append(result)
                write_summary(args.output_dir, sorted(results, key=lambda row: row.get("item_id", "")))

    write_summary(args.output_dir, sorted(results, key=lambda row: row.get("item_id", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
