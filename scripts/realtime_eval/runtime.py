from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import eval_dynamic_libero_style as legacy
from libero.libero.envs import OffScreenRenderEnv

from .control import (
    ActionRuntime,
    LatencySampler,
    ObservationPacket,
    StepRecord,
    build_policy_controller,
    load_latency_samples,
)

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


RUNTIME_V2_WORLD_CONTRACT = (
    "Control actions execute at control_freq. In sync_hold, inference latency is "
    "applied as continuous MuJoCo physics time while robot joints are held; it is "
    "not rounded to control ticks. Async and RTC action queues remain indexed by "
    "control ticks."
)


def _rotated_frame(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def _safe_obs(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    if obs is not None:
        return obs
    if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
        return env.env._get_observations()
    return env._get_observations()


def _round_scalar(value: Any, digits: int = 6) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return round(out, int(digits))


def _jsonable_array(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        return _round_scalar(arr.item(), digits)
    return [_round_scalar(item, digits) for item in arr.reshape(-1)]


def _metric_summary(values: list[float]) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": round(float(np.mean(array)), 9),
        "p50": round(float(np.percentile(array, 50)), 9),
        "p95": round(float(np.percentile(array, 95)), 9),
        "max": round(float(np.max(array)), 9),
    }


def _env_method(env: Any, name: str) -> Any | None:
    for candidate in (env, getattr(env, "env", None)):
        if candidate is None:
            continue
        method = getattr(candidate, name, None)
        if callable(method):
            return method
    return None


def _env_mapping(env: Any, name: str) -> dict[str, Any]:
    for candidate in (env, getattr(env, "env", None)):
        mapping = getattr(candidate, name, None)
        if isinstance(mapping, dict):
            return mapping
    return {}


def _target_entries(item: dict[str, Any], env: Any) -> list[dict[str, Any]]:
    sources: list[Any] = []
    dynamic_entries = _env_method(env, "_dynamic_target_entries")
    if dynamic_entries is not None:
        try:
            sources.append(dynamic_entries())
        except Exception:
            pass
    runtime_entries = getattr(env, "dynamic_runtime_target_entries", None)
    if runtime_entries:
        sources.append(runtime_entries)
    if legacy.target_entries_for_preview is not None:
        try:
            sources.append(legacy.target_entries_for_preview(item["task_cfg"]))
        except Exception:
            pass
    sources.append(item.get("task_cfg", {}).get("target_objects") or [])

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entries in sources:
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("target_object_name") or entry.get("object_name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(dict(entry))
    return out


def _body_pose(env: Any, body_name: str | None) -> dict[str, Any]:
    if not body_name:
        return {}
    try:
        body_id = int(env.sim.model.body_name2id(str(body_name)))
        return {
            "body_name": str(body_name),
            "body_pos": _jsonable_array(env.sim.data.body_xpos[body_id]),
            "body_quat": _jsonable_array(env.sim.data.body_xquat[body_id]),
        }
    except Exception:
        return {}


def _object_pose(env: Any, name: str, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = entry or {}
    pose: dict[str, Any] = {"name": str(name)}
    joint_name = str(entry.get("target_joint_name") or entry.get("joint_name") or "")
    if not joint_name:
        object_joint_name = _env_method(env, "_object_joint_name")
        if object_joint_name is not None:
            try:
                joint_name = str(object_joint_name(str(name)))
            except Exception:
                joint_name = ""
    if not joint_name:
        joint_name = str(name)
    try:
        pose["joint_name"] = joint_name
        pose["qpos"] = _jsonable_array(env.sim.data.get_joint_qpos(joint_name))
    except Exception:
        pose["joint_name"] = joint_name
        pose["qpos"] = None

    body_candidates: list[str] = []
    for key in ("target_body_name", "body_name"):
        if entry.get(key):
            body_candidates.append(str(entry[key]))
    object_joint_body_name = _env_method(env, "_object_joint_body_name")
    if object_joint_body_name is not None:
        try:
            candidate = object_joint_body_name(str(name))
            if candidate:
                body_candidates.append(str(candidate))
        except Exception:
            pass
    target_body_names = _env_method(env, "_target_body_names")
    if target_body_names is not None:
        try:
            body_candidates.extend(str(candidate) for candidate in target_body_names(str(name)) or [])
        except Exception:
            pass
    body_candidates.extend([str(name), f"{name}_main"])
    for body_name in body_candidates:
        body_pose = _body_pose(env, body_name)
        if body_pose:
            pose.update(body_pose)
            break
    return pose


def _body_geom_ids(env: Any, body_name: str | None) -> set[int]:
    if not body_name:
        return set()
    try:
        body_id = int(env.sim.model.body_name2id(str(body_name)))
    except Exception:
        return set()
    out: set[int] = set()
    try:
        for geom_id, geom_body_id in enumerate(np.asarray(env.sim.model.geom_bodyid).reshape(-1)):
            if int(geom_body_id) == body_id:
                out.add(int(geom_id))
    except Exception:
        pass
    return out


def _target_geom_ids(env: Any, name: str, entry: dict[str, Any] | None = None) -> set[int]:
    entry = entry or {}
    geom_names: list[str] = []
    for method_name in ("_target_all_contact_geom_names", "_target_contact_geom_names"):
        method = _env_method(env, method_name)
        if method is None:
            continue
        try:
            geom_names.extend(str(geom_name) for geom_name in method(str(name)) or [])
        except Exception:
            pass
    objects_dict = _env_mapping(env, "objects_dict")
    obj = objects_dict.get(str(name))
    if obj is not None:
        geom_names.extend(str(geom_name) for geom_name in (getattr(obj, "contact_geoms", None) or []))
    geom_ids: set[int] = set()
    for geom_name in sorted(set(geom_names)):
        try:
            geom_ids.add(int(env.sim.model.geom_name2id(str(geom_name))))
        except Exception:
            pass
    for key in ("target_body_name", "body_name"):
        geom_ids.update(_body_geom_ids(env, entry.get(key)))
    target_body_names = _env_method(env, "_target_body_names")
    if target_body_names is not None:
        try:
            for body_name in target_body_names(str(name)) or []:
                geom_ids.update(_body_geom_ids(env, str(body_name)))
        except Exception:
            pass
    return geom_ids


def _geom_name(env: Any, geom_id: int) -> str:
    try:
        return str(env.sim.model.geom_id2name(int(geom_id)) or "")
    except Exception:
        return ""


def _body_name_for_geom(env: Any, geom_id: int) -> str:
    try:
        body_id = int(env.sim.model.geom_bodyid[int(geom_id)])
        return str(env.sim.model.body_id2name(body_id) or "")
    except Exception:
        return ""


def _is_robot_geom(env: Any, geom_id: int) -> bool:
    method = _env_method(env, "_is_robot_geom")
    if method is not None:
        try:
            return bool(method(int(geom_id)))
        except Exception:
            pass
    geom_name = _geom_name(env, int(geom_id))
    body_name = _body_name_for_geom(env, int(geom_id))
    return geom_name.startswith(("robot0_", "gripper0_", "mount0_")) or body_name.startswith(
        ("robot0_", "gripper0_", "mount0_")
    )


def _target_geom_id_map(env: Any, target_entries: list[dict[str, Any]]) -> dict[str, set[int]]:
    return {
        str(entry.get("target_object_name") or entry.get("object_name")): _target_geom_ids(
            env,
            str(entry.get("target_object_name") or entry.get("object_name")),
            entry,
        )
        for entry in target_entries
        if entry.get("target_object_name") or entry.get("object_name")
    }


def _contact_summary(env: Any, target_entries: list[dict[str, Any]], target_ids: dict[str, set[int]] | None = None) -> dict[str, Any]:
    try:
        target_ids = _target_geom_id_map(env, target_entries) if target_ids is None else target_ids
        ncon = int(env.sim.data.ncon)
        robot_target_names: set[str] = set()
        target_contact_names: set[str] = set()
        robot_contact_count = 0
        robot_target_contact_count = 0
        robot_other_contact_count = 0
        target_contact_count = 0
        sample_pairs: list[dict[str, Any]] = []
        for idx in range(ncon):
            contact = env.sim.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_targets = [name for name, ids in target_ids.items() if geom1 in ids]
            geom2_targets = [name for name, ids in target_ids.items() if geom2 in ids]
            pair_targets = sorted(set(geom1_targets + geom2_targets))
            geom1_robot = _is_robot_geom(env, geom1)
            geom2_robot = _is_robot_geom(env, geom2)
            robot_pair = bool(geom1_robot or geom2_robot)
            if robot_pair:
                robot_contact_count += 1
            if pair_targets:
                target_contact_count += 1
                target_contact_names.update(pair_targets)
            if robot_pair and pair_targets:
                robot_target_contact_count += 1
                robot_target_names.update(pair_targets)
            elif robot_pair:
                robot_other_contact_count += 1
            if len(sample_pairs) < 24:
                sample_pairs.append(
                    {
                        "geom1": _geom_name(env, geom1),
                        "geom2": _geom_name(env, geom2),
                        "body1": _body_name_for_geom(env, geom1),
                        "body2": _body_name_for_geom(env, geom2),
                        "target_names": pair_targets,
                        "robot_pair": robot_pair,
                        "distance": _round_scalar(contact.dist),
                        "pos": _jsonable_array(contact.pos),
                    }
                )
        return {
            "ncon": ncon,
            "target_geom_counts": {name: len(ids) for name, ids in target_ids.items()},
            "robot_contact_count": int(robot_contact_count),
            "robot_target_contact": bool(robot_target_contact_count > 0),
            "robot_target_contact_count": int(robot_target_contact_count),
            "robot_target_names": sorted(robot_target_names),
            "robot_other_contact_count": int(robot_other_contact_count),
            "target_contact_count": int(target_contact_count),
            "target_contact_names": sorted(target_contact_names),
            "sample_pairs": sample_pairs,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def _trace_record_to_json(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    return dict(getattr(record, "__dict__", {}))


def _capture_robot_hold_state(env: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
    state: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        for joint_id in range(int(env.sim.model.njnt)):
            joint_name = str(env.sim.model.joint_id2name(joint_id) or "")
            if not joint_name.startswith(("robot0_", "gripper0_")):
                continue
            state.append(
                (
                    joint_name,
                    np.asarray(env.sim.data.get_joint_qpos(joint_name)).copy(),
                    np.asarray(env.sim.data.get_joint_qvel(joint_name)).copy(),
                )
            )
    except Exception:
        return []
    return state


def _restore_robot_hold_state(
    env: Any,
    hold_state: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    zero_velocity: bool = True,
) -> None:
    if not hold_state:
        return
    for joint_name, qpos, qvel in hold_state:
        try:
            env.sim.data.set_joint_qpos(joint_name, qpos)
            env.sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel) if zero_velocity else qvel)
        except Exception:
            continue
    try:
        env.sim.forward()
    except Exception:
        pass


def _step_with_robot_hold(
    step_fn: Any,
    env: Any,
    action: np.ndarray,
    hold_state: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
    if not hold_state:
        return step_fn(env, action)
    sim = env.sim
    original_step = sim.step

    def _held_sim_step(*args: Any, **kwargs: Any) -> Any:
        _restore_robot_hold_state(env, hold_state, zero_velocity=True)
        result = original_step(*args, **kwargs)
        _restore_robot_hold_state(env, hold_state, zero_velocity=True)
        return result

    sim.step = _held_sim_step
    try:
        _restore_robot_hold_state(env, hold_state, zero_velocity=True)
        obs, reward, done, info = step_fn(env, action)
        _restore_robot_hold_state(env, hold_state, zero_velocity=True)
        obs = _safe_obs(env, None)
        return obs, reward, done, info
    finally:
        sim.step = original_step


class RealtimeEpisode:
    def __init__(self, item: dict[str, Any], args: argparse.Namespace, worker_slot: int = 0) -> None:
        self.item = item
        self.args = args
        self.worker_slot = int(worker_slot)
        self.preview_elapsed = 0.0
        self.preview_dt = 1.0 / float(args.control_freq)
        self.preview_states: list[dict[str, Any]] = []
        self.use_preview = legacy.should_use_preview_compatible_eval(item, args)

    def _configure_env_selection(self) -> None:
        if self.use_preview:
            disabled = self.args.output_dir / "_preview_compatible_eval_wrapper_disabled.yaml"
            os.environ["LIBERO_DYNAMIC_CONFIG"] = str(disabled)
        else:
            os.environ["LIBERO_DYNAMIC_CONFIG"] = str(self.item["config_path"])
        os.environ["LIBERO_DYNAMIC_TRAJ_INDEX"] = str(self.item["traj_index"])
        os.environ["LIBERO_DYNAMIC_VARIANT_ID"] = str(self.item["variant_id"])
        os.environ.setdefault("LIBERO_DYNAMIC_AUTO_FINALIZE", "false")

    def _step_env(
        self,
        env: Any,
        action: np.ndarray,
        *,
        hold_preview_platforms: bool = False,
        on_substep: Any | None = None,
    ) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
        if self.use_preview:
            if hold_preview_platforms:
                return legacy.preview_step_with_platform_hold(env, action, self.preview_states)
            obs, reward, done, info = legacy.preview_step_with_physical_dynamic(
                env,
                action,
                self.preview_states,
                self.preview_elapsed,
                self.preview_dt,
                on_substep=on_substep,
            )
            self.preview_elapsed += self.preview_dt
            legacy.update_preview_state_offsets(self.preview_states, self.preview_elapsed)
            env.dynamic_multi_target_states = self.preview_states
            return obs, reward, done, info
        if on_substep is None:
            return env.step(action)

        sim = env.sim
        original_step = sim.step
        sim_dt = float(getattr(sim.model.opt, "timestep", 0.0) or 0.0)
        advanced_s = 0.0

        def _step_with_recording(*args: Any, **kwargs: Any) -> Any:
            nonlocal advanced_s
            result = original_step(*args, **kwargs)
            advanced_s += sim_dt
            on_substep(advanced_s)
            return result

        sim.step = _step_with_recording
        try:
            return env.step(action)
        finally:
            sim.step = original_step

    def _apply_sync_latency(
        self,
        env: Any,
        duration: float,
        *,
        on_substep: Any | None = None,
    ) -> float:
        duration = max(0.0, float(duration))
        if self.use_preview:
            if duration <= 0.0:
                return 0.0
            if legacy.preview_advance_physical_dynamic_time is None:
                raise RuntimeError("preview continuous-physics latency support is unavailable")
            hold_state = _capture_robot_hold_state(env)
            sim = env.sim
            original_step = sim.step

            def _held_sim_step(*args: Any, **kwargs: Any) -> Any:
                _restore_robot_hold_state(env, hold_state, zero_velocity=True)
                result = original_step(*args, **kwargs)
                _restore_robot_hold_state(env, hold_state, zero_velocity=True)
                return result

            sim.step = _held_sim_step
            try:
                _restore_robot_hold_state(env, hold_state, zero_velocity=True)
                advanced = legacy.preview_advance_physical_dynamic_time(
                    env,
                    self.preview_states,
                    self.preview_elapsed,
                    duration,
                    on_substep=on_substep,
                )
                _restore_robot_hold_state(env, hold_state, zero_velocity=True)
            finally:
                sim.step = original_step
            self.preview_elapsed += float(advanced)
            legacy.update_preview_state_offsets(self.preview_states, self.preview_elapsed)
            env.dynamic_multi_target_states = self.preview_states
            return float(advanced)

        notify_latency = _env_method(env, "notify_inference_latency")
        if notify_latency is None:
            if duration > 0.0:
                raise RuntimeError("env-wrapper route cannot apply continuous inference latency")
            return 0.0
        notify_latency(duration)
        return duration

    def _setup_world(self, env: Any, init_state: np.ndarray) -> tuple[dict[str, Any], int, int, int, list[dict[str, Any]]]:
        obs = env.reset()
        preview_settle_done = 0
        warmup_done = 0
        pre_policy_done = 0
        fixture_passengers_added: list[dict[str, Any]] = []
        if self.use_preview:
            obs, self.preview_states, preview_settle_done = legacy.setup_preview_compatible_dynamic(
                env, self.item, self.args, init_state
            )
        else:
            obs = env.set_init_state(init_state)

        zero_action = np.zeros((legacy.action_dim(env),), dtype=np.float32)
        if (
            int(self.args.num_steps_wait) > 0
            and bool(self.args.reset_platforms_after_warmup)
            and getattr(env, "dynamic_multi_target_states", None)
            and not self.use_preview
        ):
            legacy.park_physical_platforms_below_scene(env)
        for _ in range(int(self.args.num_steps_wait)):
            obs, _reward, done, _info = self._step_env(
                env,
                zero_action,
                hold_preview_platforms=self.use_preview,
            )
            warmup_done += 1
            if (bool(done) or bool(env.check_success())) and bool(self.args.stop_on_success):
                break

        if (
            warmup_done > 0
            and bool(self.args.reset_platforms_after_warmup)
            and hasattr(env, "append_fixture_supported_passenger_targets_from_contacts")
            and not self.use_preview
        ):
            fixture_passengers_added = env.append_fixture_supported_passenger_targets_from_contacts()
            obs = _safe_obs(env, obs)
        if (
            warmup_done > 0
            and bool(self.args.reset_platforms_after_warmup)
            and getattr(env, "dynamic_multi_target_states", None)
            and hasattr(env, "_reset_dynamic_physical_platform_poses")
            and not self.use_preview
        ):
            env._reset_dynamic_physical_platform_poses()
            obs = _safe_obs(env, obs)

        phase_prime = int(self.args.dynamic_phase_prime_steps)
        if fixture_passengers_added and self.args.fixture_passenger_phase_prime_steps is not None:
            phase_prime = int(self.args.fixture_passenger_phase_prime_steps)
        if (
            phase_prime > 0
            and getattr(env, "dynamic_multi_target_states", None)
            and hasattr(env, "prime_dynamic_physical_platform_phase")
            and not self.use_preview
        ):
            env.prime_dynamic_physical_platform_phase(int(phase_prime))
            obs = _safe_obs(env, obs)

        for _ in range(int(self.args.pre_policy_dynamic_steps)):
            obs, _reward, done, _info = self._step_env(env, zero_action)
            pre_policy_done += 1
            if (bool(done) or bool(env.check_success())) and bool(self.args.stop_on_success):
                break
        return obs, preview_settle_done, warmup_done, pre_policy_done, fixture_passengers_added

    def _build_controller(self, policy: Any, language: str) -> Any:
        return build_policy_controller(
            policy=policy,
            language=language,
            control_freq=int(self.args.control_freq),
            mode=str(self.args.control_mode),
            prefetch_threshold=int(self.args.prefetch_threshold),
            action_chunk_size=int(self.args.replan_steps),
            chunk_size_threshold=self.args.chunk_size_threshold,
            latency_sampler=self._build_latency_sampler(),
            vlash_visual_delay_steps=int(self.args.vlash_visual_delay_steps),
            vlash_measured_latency_seconds=self.args.vlash_measured_latency_seconds,
        )

    def _build_latency_sampler(self) -> LatencySampler:
        samples = None
        if str(self.args.latency_source) == "empirical":
            samples = load_latency_samples(Path(self.args.latency_samples_json))
        item_seed = sum(ord(ch) for ch in str(self.item.get("item_id", "")))
        return LatencySampler(
            source=str(self.args.latency_source),
            fixed_seconds=self.args.calibrated_latency_seconds,
            first_seconds=self.args.calibrated_first_inference_seconds,
            samples=samples,
            seed=int(self.args.latency_sample_seed) + self.worker_slot * 1_000_003 + item_seed,
        )

    def _policy_server_index(self) -> int | None:
        if str(self.args.policy) != "openpi_ws":
            return None
        if not bool(self.args.per_worker_policy_endpoint):
            return 0
        server_count = int(getattr(self.args, "policy_server_count", 0))
        if server_count > 0:
            return int(self.worker_slot) % server_count
        return int(self.worker_slot)

    def _openpi_port(self) -> int | None:
        index = self._policy_server_index()
        if index is None:
            return None
        return int(self.args.openpi_port) + int(index) * int(self.args.policy_port_stride)

    def _shared_policy_wallclock_endpoint(self) -> bool:
        if str(self.args.policy) != "openpi_ws":
            return False
        if not bool(self.args.per_worker_policy_endpoint):
            return True
        server_count = int(getattr(self.args, "policy_server_count", 0))
        return server_count > 0 and int(self.args.num_workers) > server_count

    def run(self) -> dict[str, Any]:
        args = self.args
        item = self.item
        started = time.time()
        self._configure_env_selection()
        suite, task, bddl = legacy.task_bddl(item["suite"], item["task_id"])
        init_states = suite.get_task_init_states(item["task_id"])

        base_dir = args.output_dir / item["level"] / item["suite"]
        video_dir = base_dir / "videos"
        model_input_video_dir = base_dir / "model_input_videos"
        wrist_video_dir = base_dir / "wrist_input_videos"
        manifest_path = base_dir / "manifests" / f"{item['item_id']}.json"
        trace_path = base_dir / "traces" / f"{item['item_id']}--runtime-v2.jsonl"
        for path in (video_dir, model_input_video_dir, wrist_video_dir, manifest_path.parent, trace_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{item['item_id']}--runtime-v2-{args.control_mode}-{args.policy}.mp4"
        model_input_video_path = (
            model_input_video_dir
            / f"{item['item_id']}--modelinput-{args.model_input_video_layout}-runtime-v2-{args.control_mode}-{args.policy}.mp4"
        )
        wrist_video_path = wrist_video_dir / f"{item['item_id']}--wrist-runtime-v2-{args.control_mode}-{args.policy}.mp4"

        env = None
        writer = None
        model_input_writer = None
        wrist_writer = None
        controller = None
        policy = None
        actions = None
        step_records: list[dict[str, Any]] = []
        ready_chunks: list[dict[str, Any]] = []
        diagnostics = None
        success = False
        status = "ok"
        error = None
        error_traceback = None
        setup_counts: dict[str, Any] = {}
        step_count = 0
        rollout_sim_time_s = 0.0
        sync_latency_elapsed_s = 0.0
        next_video_sample_s = 1.0 / float(args.video_fps)
        video_frame_count = 0
        physics_timestep_s: float | None = None
        prev_action: np.ndarray | None = None
        prev_chunk_id: int | None = None
        action_l2_samples: list[float] = []
        action_delta_l2_samples: list[float] = []
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=int(args.camera_size),
                camera_widths=int(args.camera_size),
                control_freq=int(args.control_freq),
            )
            physics_timestep_s = float(getattr(env.sim.model.opt, "timestep", 0.0))
            if (
                bool(args.record_latency_frames)
                and bool(args.save_videos)
                and physics_timestep_s > 0.0
                and float(args.video_fps) > (1.0 / physics_timestep_s) + 1e-9
            ):
                raise ValueError(
                    f"--video-fps={args.video_fps:g} exceeds the MuJoCo physics rate "
                    f"of {1.0 / physics_timestep_s:g} Hz"
                )
            if args.seed is not None:
                env.seed(int(args.seed) + self.worker_slot)
            obs, preview_settle, warmup_done, pre_policy_done, fixture_passengers = self._setup_world(
                env,
                init_states[int(args.init_state_id)],
            )
            setup_counts = {
                "preview_compatible_settle_steps_done": int(preview_settle),
                "warmup_steps_done": int(warmup_done),
                "pre_policy_dynamic_steps_done": int(pre_policy_done),
                "fixture_passengers_added_count": len(fixture_passengers),
                "fixture_passengers_added": [repr(entry) for entry in fixture_passengers[:20]],
            }
            diagnostics = legacy.init_diag_state(env, args)
            policy_args = argparse.Namespace(**vars(args))
            openpi_port = self._openpi_port()
            if str(policy_args.policy) == "openpi_ws" and openpi_port is not None:
                policy_args.openpi_port = int(openpi_port)
            policy = legacy.build_policy(policy_args, env, item)
            if bool(args.require_rtc_conditioning) and not bool(
                getattr(policy, "supports_rtc_conditioning", False)
            ):
                raise RuntimeError(
                    "RTC evaluation requires a policy server reporting rtc_conditioning_supported=true"
                )
            controller = self._build_controller(policy, task.language)
            controller.reset()
            actions = ActionRuntime(legacy.action_dim(env), empty_action=str(args.empty_action))
            trace_target_entries = _target_entries(item, env)
            trace_target_geom_ids = _target_geom_id_map(env, trace_target_entries)

            if bool(args.save_videos):
                if imageio is None:
                    raise RuntimeError("imageio is required when --save-videos is enabled")
                if bool(args.save_review_video):
                    writer = imageio.get_writer(video_path, fps=float(args.video_fps), macro_block_size=None)
                if bool(args.save_model_input_videos):
                    model_input_writer = imageio.get_writer(
                        model_input_video_path,
                        fps=float(args.video_fps),
                        macro_block_size=None,
                    )
                    if bool(args.save_wrist_input_videos):
                        wrist_writer = imageio.get_writer(wrist_video_path, fps=float(args.video_fps), macro_block_size=None)

            def _record_time_faithful_frame(absolute_sim_time_s: float) -> None:
                nonlocal next_video_sample_s, video_frame_count
                if not bool(args.record_latency_frames):
                    return
                if writer is None and model_input_writer is None and wrist_writer is None:
                    return
                if absolute_sim_time_s + 1e-9 < next_video_sample_s:
                    return
                agent_frame = np.ascontiguousarray(
                    np.asarray(
                        env.sim.render(
                            int(args.camera_size),
                            int(args.camera_size),
                            camera_name="agentview",
                        )
                    )[::-1]
                )
                wrist_frame = np.ascontiguousarray(
                    np.asarray(
                        env.sim.render(
                            int(args.camera_size),
                            int(args.camera_size),
                            camera_name="robot0_eye_in_hand",
                        )
                    )[::-1]
                )
                if writer is not None:
                    if args.video_style == "review" and legacy.render_frame is not None:
                        writer.append_data(
                            legacy.render_frame(
                                env,
                                None,
                                int(args.camera_size),
                                None,
                                overlay_traces=False,
                            )
                        )
                    else:
                        writer.append_data(agent_frame)
                if model_input_writer is not None:
                    agent_input = legacy.resize_video_frame(agent_frame, int(args.model_input_video_size))
                    if str(args.model_input_video_layout) == "dual":
                        wrist_input = legacy.resize_video_frame(wrist_frame, int(args.model_input_video_size))
                        model_input_writer.append_data(
                            np.ascontiguousarray(np.concatenate([agent_input, wrist_input], axis=1))
                        )
                    else:
                        model_input_writer.append_data(agent_input)
                if wrist_writer is not None:
                    wrist_writer.append_data(wrist_frame)
                video_frame_count += 1
                next_video_sample_s += 1.0 / float(args.video_fps)

            while step_count < int(args.max_steps) and not (success and bool(args.stop_on_success)):
                packet = ObservationPacket(
                    episode_id=str(item["item_id"]),
                    step_idx=int(step_count),
                    sim_time_s=float(rollout_sim_time_s),
                    language=str(task.language),
                    obs=obs,
                    queue_len=actions.queue_len,
                    last_action=np.array(actions.last_action, copy=True),
                )
                ready_now = controller.tick(packet, actions)
                sync_latency_before_action_s = 0.0
                for ready in ready_now:
                    ready_chunks.append(
                        {
                            "chunk_id": ready.chunk_id,
                            "obs_step_idx": ready.obs_step_idx,
                            "submit_step_idx": ready.submit_step_idx,
                            "available_step_idx": ready.available_step_idx,
                            "policy_elapsed_s": round(float(ready.policy_elapsed_s), 6),
                            "controller": ready.controller,
                            "metadata": ready.metadata,
                        }
                    )
                    if str(ready.controller) == "sync_hold":
                        sync_latency_before_action_s += float(ready.metadata.get("sim_latency_s", 0.0))
                if ready_now and str(args.control_mode) in {"sync_hold", "hold_latency"}:
                    def _record_latency_substep(advanced_s: float) -> None:
                        _record_time_faithful_frame(float(rollout_sim_time_s) + float(advanced_s))

                    applied_latency = self._apply_sync_latency(
                        env,
                        sync_latency_before_action_s,
                        on_substep=_record_latency_substep,
                    )
                    sync_latency_elapsed_s += float(applied_latency)
                    rollout_sim_time_s += float(applied_latency)
                action, original_action, source, underflow, chunk_id, q_before, q_after = actions.next_action()
                action_start_sim_time_s = float(rollout_sim_time_s)
                env_step_start_sim_time_s = action_start_sim_time_s
                if not self.use_preview:
                    # The env-wrapper consumes notified sync latency inside the
                    # following env.step call, before executing the control action.
                    env_step_start_sim_time_s -= float(sync_latency_before_action_s)

                def _record_action_substep(advanced_s: float) -> None:
                    _record_time_faithful_frame(env_step_start_sim_time_s + float(advanced_s))

                robot_hold_state = (
                    _capture_robot_hold_state(env)
                    if bool(underflow) and bool(args.freeze_robot_on_underflow)
                    else []
                )
                if robot_hold_state:
                    def _step_env_with_recording(step_env: Any, step_action: np.ndarray) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
                        return self._step_env(step_env, step_action, on_substep=_record_action_substep)

                    obs, _reward, done, info = _step_with_robot_hold(
                        _step_env_with_recording,
                        env,
                        action,
                        robot_hold_state,
                    )
                else:
                    obs, _reward, done, info = self._step_env(
                        env,
                        action,
                        on_substep=_record_action_substep,
                    )
                del info
                rollout_sim_time_s += self.preview_dt
                success = bool(done) or bool(env.check_success())
                if self.use_preview:
                    legacy.update_preview_compatible_diagnostics(
                        env,
                        diagnostics,
                        self.preview_states,
                        self.preview_elapsed,
                    )
                else:
                    legacy.update_diagnostics(env, diagnostics)
                diag_summary = legacy.summarize_diagnostics(diagnostics, args)
                diag_ok = bool(diag_summary.get("ok", False))
                action_l2 = float(np.linalg.norm(np.asarray(action, dtype=float)))
                action_delta_l2 = (
                    0.0
                    if prev_action is None
                    else float(
                        np.linalg.norm(
                            np.asarray(action, dtype=float) - np.asarray(prev_action, dtype=float)
                        )
                    )
                )
                action_l2_samples.append(action_l2)
                action_delta_l2_samples.append(action_delta_l2)
                if bool(args.save_step_trace):
                    target_poses = {
                        str(entry.get("target_object_name") or entry.get("object_name")): _object_pose(
                            env,
                            str(entry.get("target_object_name") or entry.get("object_name")),
                            entry,
                        )
                        for entry in trace_target_entries
                        if entry.get("target_object_name") or entry.get("object_name")
                    }
                    step_records.append(
                        {
                            "step_idx": int(step_count),
                            "sim_time_s": round(float(rollout_sim_time_s), 6),
                            "sync_latency_before_action_s": round(float(sync_latency_before_action_s), 6),
                            "action_source": source,
                            "queue_len_before": int(q_before),
                            "queue_len_after": int(q_after),
                            "underflow": bool(underflow),
                            "freeze_robot_on_underflow": bool(
                                underflow and bool(args.freeze_robot_on_underflow) and robot_hold_state
                            ),
                            "freeze_robot_hold_joint_count": len(robot_hold_state),
                            "chunk_id": chunk_id,
                            "replan_boundary": bool(chunk_id is not None and chunk_id != prev_chunk_id),
                            "success": bool(success),
                            "diagnostics_ok": diag_ok,
                            "executed_action": _jsonable_array(action),
                            "original_action": _jsonable_array(original_action),
                            "action_l2": _round_scalar(action_l2),
                            "action_delta_l2": _round_scalar(action_delta_l2),
                            "eef_pos": _jsonable_array(obs.get("robot0_eef_pos")),
                            "eef_quat": _jsonable_array(obs.get("robot0_eef_quat")),
                            "gripper_qpos": _jsonable_array(obs.get("robot0_gripper_qpos")),
                            "target_object_poses": target_poses,
                            "contact_summary": _contact_summary(
                                env,
                                trace_target_entries,
                                trace_target_geom_ids,
                            ),
                        }
                    )
                prev_action = np.array(action, copy=True)
                prev_chunk_id = chunk_id
                if not bool(args.record_latency_frames) and (
                    writer is not None or model_input_writer is not None or wrist_writer is not None
                ) and (
                    step_count % max(1, int(args.render_every)) == 0
                ):
                    if writer is not None:
                        if args.video_style == "review" and legacy.render_frame is not None:
                            writer.append_data(
                                legacy.render_frame(env, obs, int(args.camera_size), None, overlay_traces=False)
                            )
                        else:
                            writer.append_data(legacy.obs_frame(obs))
                    if model_input_writer is not None:
                        model_input_writer.append_data(
                            legacy.model_input_frame(
                                obs,
                                str(args.model_input_video_layout),
                                int(args.model_input_video_size),
                            )
                        )
                    if wrist_writer is not None:
                        wrist_writer.append_data(_rotated_frame(obs["robot0_eye_in_hand_image"]))
                step_count += 1
            diagnostic_summary = legacy.summarize_diagnostics(diagnostics, args)
        except Exception as exc:
            status = "error"
            error = repr(exc)
            error_traceback = traceback.format_exc()
            diagnostic_summary = {"targets": {}, "flags": [{"type": "exception", "value": error}], "ok": False}
        finally:
            for maybe_writer in (writer, model_input_writer, wrist_writer):
                if maybe_writer is not None:
                    maybe_writer.close()
            if controller is not None:
                controller.close()
            if env is not None:
                env.close()

        if bool(args.save_step_trace) and step_records:
            trace_path.write_text(
                "\n".join(json.dumps(_trace_record_to_json(record), ensure_ascii=False) for record in step_records) + "\n",
                encoding="utf-8",
            )

        controller_summary = {} if controller is None else controller.summary()
        result = {
            **{k: item[k] for k in ("item_id", "suite", "task_id", "level", "traj_index", "variant_name", "variant_id")},
            "runtime_version": "realtime_v2",
            "world_contract": RUNTIME_V2_WORLD_CONTRACT,
            "clock_mode": "sim_time",
            "worker_slot": int(self.worker_slot),
            "status": status,
            "error": error,
            "error_traceback": error_traceback,
            "language": getattr(task, "language", ""),
            "dynamic_eval_route": "preview_compatible" if self.use_preview else "env_wrapper",
            "control_mode": str(args.control_mode),
            "policy": str(args.policy),
            "policy_server_metadata": (
                getattr(policy, "server_metadata", None) if policy is not None else None
            ),
            "openpi_host": str(args.openpi_host) if args.policy == "openpi_ws" else None,
            "openpi_port": self._openpi_port(),
            "policy_server_index": self._policy_server_index(),
            "policy_server_count": int(args.policy_server_count) if args.policy == "openpi_ws" else None,
            "per_worker_policy_endpoint": bool(args.per_worker_policy_endpoint),
            "policy_wallclock_shared_endpoint": self._shared_policy_wallclock_endpoint(),
            "latency_source": str(args.latency_source),
            "calibrated_latency_seconds": args.calibrated_latency_seconds,
            "vlash_measured_latency_seconds": args.vlash_measured_latency_seconds,
            "vlash_visual_delay_steps": int(args.vlash_visual_delay_steps),
            "sim_latency_decoupled_from_wallclock": bool(str(args.latency_source) in {"fixed", "empirical"}),
            "shared_policy_latency_contaminated": bool(
                self._shared_policy_wallclock_endpoint()
                and str(args.latency_source) == "measured_wallclock"
            ),
            "replan_steps_requested": int(args.replan_steps),
            "prefetch_threshold": int(args.prefetch_threshold),
            "empty_action": str(args.empty_action),
            "freeze_robot_on_underflow": bool(args.freeze_robot_on_underflow),
            "action_runtime": {
                "queue_underflow_steps": 0 if actions is None else int(actions.underflow_steps),
                "enqueued_chunks": 0 if actions is None else int(actions.enqueued_chunks),
                "executed_actions": 0 if actions is None else int(actions.executed_actions),
            },
            "action_metrics": {
                "definition": {
                    "action_l2": "||a_t||_2",
                    "action_delta_l2": "||a_t - a_(t-1)||_2",
                },
                "action_l2": _metric_summary(action_l2_samples),
                "action_delta_l2": _metric_summary(action_delta_l2_samples),
            },
            "controller_summary": controller_summary,
            "ready_chunks": ready_chunks[-50:],
            "steps": int(step_count),
            "max_steps": int(args.max_steps),
            "control_freq": int(args.control_freq),
            "physics_timestep_s": None if physics_timestep_s is None else round(physics_timestep_s, 9),
            "policy_rollout_sim_time_s": round(float(rollout_sim_time_s), 6),
            "sync_inference_latency_elapsed_s": round(float(sync_latency_elapsed_s), 6),
            "video_fps": float(args.video_fps),
            "video_sampling_mode": "simulation_time" if bool(args.record_latency_frames) else "control_step",
            "video_frame_count": int(video_frame_count) if bool(args.record_latency_frames) else None,
            "video_duration_s": (
                round(float(video_frame_count) / float(args.video_fps), 6)
                if bool(args.record_latency_frames)
                else None
            ),
            "success": bool(success),
            "diagnostics": diagnostic_summary,
            "setup": setup_counts,
            "effective_config_path": str(item["config_path"]),
            "target_names": [
                target.get("target_object_name")
                for target in (
                    legacy.target_entries_for_preview(item["task_cfg"])
                    if legacy.target_entries_for_preview is not None
                    else item["task_cfg"].get("target_objects", [])
                )
            ],
            "video": str(video_path)
            if bool(args.save_videos) and bool(args.save_review_video) and status == "ok"
            else None,
            "model_input_video": str(model_input_video_path)
            if bool(args.save_videos) and bool(args.save_model_input_videos) and status == "ok"
            else None,
            "model_input_video_layout": str(args.model_input_video_layout),
            "model_input_video_size": int(args.model_input_video_size),
            "model_input_video_width": int(args.model_input_video_size)
            * (2 if str(args.model_input_video_layout) == "dual" else 1),
            "model_input_video_height": int(args.model_input_video_size),
            "wrist_input_video": str(wrist_video_path)
            if bool(args.save_videos)
            and bool(args.save_model_input_videos)
            and bool(args.save_wrist_input_videos)
            and status == "ok"
            else None,
            "step_trace": str(trace_path) if bool(args.save_step_trace) and step_records else None,
            "manifest": str(manifest_path),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
