import atexit
import os
import sys
import threading
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

from libero.libero.envs.dynamic_perturbation import (
    DynamicPerturbationController,
    infer_suite_and_task_id_from_bddl,
    load_dynamic_config,
    resolve_dynamic_config_for_task,
)
from libero.libero.envs.dynamic_logger import DynamicRunLogger


_DYNAMIC_FINALIZER_REGISTRY = set()
_DYNAMIC_FINALIZER_REGISTRY_LOCK = threading.Lock()


def _env_flag_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _repo_root_from_env_wrapper():
    return Path(__file__).resolve().parents[3]


def _load_dynamic_results_finalizer():
    scripts_dir = _repo_root_from_env_wrapper() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from finalize_dynamic_results import finalize_dynamic_results

    return finalize_dynamic_results


def _run_dynamic_results_finalizer_once(
    *,
    eval_output_dir,
    dynamic_log_dir,
    output_dir,
    num_steps_wait,
    fps,
    min_latency_seconds,
    cleanup_latency_frames=True,
):
    finalize_dynamic_results = _load_dynamic_results_finalizer()
    return finalize_dynamic_results(
        eval_output_dir=Path(eval_output_dir),
        dynamic_log_dir=Path(dynamic_log_dir),
        output_dir=Path(output_dir),
        num_steps_wait=int(num_steps_wait),
        fps=float(fps),
        min_latency_seconds=float(min_latency_seconds),
        cleanup_latency_frames=bool(cleanup_latency_frames),
    )


def _register_dynamic_atexit_finalizer(finalizer_kwargs):
    key = (
        str(Path(finalizer_kwargs["eval_output_dir"]).resolve()),
        str(Path(finalizer_kwargs["dynamic_log_dir"]).resolve()),
        str(Path(finalizer_kwargs["output_dir"]).resolve()),
    )
    with _DYNAMIC_FINALIZER_REGISTRY_LOCK:
        if key in _DYNAMIC_FINALIZER_REGISTRY:
            return
        _DYNAMIC_FINALIZER_REGISTRY.add(key)

    def _finalize_at_exit():
        try:
            _run_dynamic_results_finalizer_once(**finalizer_kwargs)
        except Exception as exc:
            print(f"[LIBERO-dynamic warning] auto finalization at exit failed: {exc}")

    atexit.register(_finalize_at_exit)


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )
        self.dynamic_legacy_joint_layout = self._capture_dynamic_legacy_joint_layout()

        raw_dynamic_config = load_dynamic_config()
        self.dynamic_carrier = None
        self.dynamic_physical_platforms = {}
        self.dynamic_multi_target_states = []
        self.dynamic_controller = None
        self.dynamic_step_idx = 0
        self.dynamic_control_dt_seconds = 1.0 / float(control_freq)
        self.dynamic_motion_elapsed_seconds = 0.0
        self.dynamic_motion_position_m = 0.0
        self.dynamic_motion_direction_sign = 1.0
        self.dynamic_motion_offset_xyz = np.zeros(3, dtype=float)
        self.dynamic_fixture_body_state = None
        self.dynamic_surface_velocity_step_mps = None
        self.dynamic_surface_velocity_step_hook_count = 0
        self.pending_inference_latency_seconds = None
        self.pending_inference_latency_notified = False
        self.dynamic_suite_name, self.dynamic_task_id = infer_suite_and_task_id_from_bddl(
            bddl_file_name
        )
        self.dynamic_config = resolve_dynamic_config_for_task(
            raw_dynamic_config,
            self.dynamic_suite_name,
            self.dynamic_task_id,
        )

        if bool(self.dynamic_config.get("enabled", False)):
            if self.dynamic_task_id is None:
                raise ValueError(
                    f"Failed to infer dynamic task id from bddl_file_name={bddl_file_name}"
                )
            self._install_dynamic_carrier_if_needed()
            self.dynamic_controller = DynamicPerturbationController(
                self.env,
                suite_name=self.dynamic_suite_name,
                task_id=self.dynamic_task_id,
                config=self.dynamic_config,
            )
            self._install_dynamic_physical_platforms_if_needed()
            self._initialize_dynamic_fixture_body_state()

        self.dynamic_episode_index = -1
        self.dynamic_episode_active = False
        self.dynamic_episode_finalized = True
        self.dynamic_logger = None
        self.dynamic_finalize_threads = []
        self.dynamic_finalizer_kwargs = None
        self.dynamic_model_input_video_writer = None
        self.dynamic_model_input_video_path = None
        self.dynamic_model_input_frame_count = 0
        logging_cfg = self.dynamic_config.get("logging", {})
        if bool(logging_cfg.get("enabled", False)):
            output_path = logging_cfg.get("output_path")
            run_id = logging_cfg.get("run_id") or self.dynamic_config.get("config_path", "dynamic_run")
            log_dir_env = os.environ.get("LIBERO_DYNAMIC_LOG_DIR")
            eval_output_env = os.environ.get("LIBERO_DYNAMIC_EVAL_OUTPUT_DIR")
            if log_dir_env:
                default_log_dir = Path(log_dir_env)
            elif eval_output_env:
                default_log_dir = Path(eval_output_env) / "libero_dynamic_logs"
            else:
                default_log_dir = Path("./libero_dynamic_logs")
            if output_path is None:
                output_path = default_log_dir / f"{Path(str(run_id)).stem}.json"
            else:
                output_path = Path(output_path)
                if not output_path.is_absolute():
                    output_path = default_log_dir / output_path
            self.dynamic_logger = DynamicRunLogger(
                output_path,
                run_id=run_id,
            )

        self._setup_dynamic_auto_finalize()

    def _capture_dynamic_legacy_joint_layout(self):
        layout = []
        model = self.env.sim.model
        for joint_id in range(int(model.njnt)):
            name = model.joint_id2name(joint_id)
            if not name:
                continue
            qpos_start = int(model.jnt_qposadr[joint_id])
            qpos_end = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < int(model.njnt) else int(model.nq)
            dof_start = int(model.jnt_dofadr[joint_id])
            dof_end = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < int(model.njnt) else int(model.nv)
            layout.append(
                {
                    "name": str(name),
                    "qpos_start": qpos_start,
                    "qpos_end": qpos_end,
                    "dof_start": dof_start,
                    "dof_end": dof_end,
                }
            )
        return layout

    def _restore_legacy_state_by_joint_name(self, state):
        state = np.asarray(state, dtype=float)
        layout = getattr(self, "dynamic_legacy_joint_layout", []) or []
        if not layout:
            return False
        legacy_nq = max(int(item["qpos_end"]) for item in layout)
        legacy_nv = max(int(item["dof_end"]) for item in layout)
        if len(state) != 1 + legacy_nq + legacy_nv:
            return False

        legacy_qpos = state[1 : 1 + legacy_nq]
        legacy_qvel = state[1 + legacy_nq : 1 + legacy_nq + legacy_nv]
        self.env.sim.data.time = float(state[0])
        self.env.sim.data.qpos[:] = 0.0
        self.env.sim.data.qvel[:] = 0.0
        for item in layout:
            name = item["name"]
            try:
                current_joint_id = int(self.env.sim.model.joint_name2id(name))
            except Exception:
                continue
            current_qpos_start = int(self.env.sim.model.jnt_qposadr[current_joint_id])
            current_qpos_end = (
                int(self.env.sim.model.jnt_qposadr[current_joint_id + 1])
                if current_joint_id + 1 < int(self.env.sim.model.njnt)
                else int(self.env.sim.model.nq)
            )
            current_dof_start = int(self.env.sim.model.jnt_dofadr[current_joint_id])
            current_dof_end = (
                int(self.env.sim.model.jnt_dofadr[current_joint_id + 1])
                if current_joint_id + 1 < int(self.env.sim.model.njnt)
                else int(self.env.sim.model.nv)
            )
            qpos_values = legacy_qpos[int(item["qpos_start"]) : int(item["qpos_end"])]
            qvel_values = legacy_qvel[int(item["dof_start"]) : int(item["dof_end"])]
            qpos_width = min(len(qpos_values), current_qpos_end - current_qpos_start)
            qvel_width = min(len(qvel_values), current_dof_end - current_dof_start)
            self.env.sim.data.qpos[current_qpos_start : current_qpos_start + qpos_width] = qpos_values[:qpos_width]
            self.env.sim.data.qvel[current_dof_start : current_dof_start + qvel_width] = qvel_values[:qvel_width]
        self.env.sim.forward()
        return True



    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def _build_dynamic_plan(self):
        plan = {
            "enabled": bool(self.dynamic_config.get("enabled", False)),
            "perturb_type": self.dynamic_config.get("perturb_type", "none"),
            "suite_name": self.dynamic_suite_name,
            "task_id": self.dynamic_task_id,
        }
        perturb_type = plan["perturb_type"]
        if perturb_type == "mid_episode_shift":
            plan["mid_episode_shift"] = self.dynamic_config.get("mid_episode_shift", {})
        elif perturb_type == "latency_drift":
            plan["latency_drift"] = self.dynamic_config.get("latency_drift", {})
        elif perturb_type == "dynamic_motion":
            plan["dynamic_role"] = self.dynamic_config.get("dynamic_role", "unspecified")
            plan["motion"] = self.dynamic_config.get("motion", {})

        latency_cfg = self.dynamic_config.get("latency", {})
        if bool(latency_cfg.get("enabled", False)):
            plan["latency"] = latency_cfg
        if self.dynamic_controller is not None:
            plan["target_object_name"] = self.dynamic_controller.target.object_name
            plan["target_joint_name"] = self.dynamic_controller.target.joint_name
        return plan

    def _model_input_video_config(self):
        return self.dynamic_config.get("model_input_video", {}) or {}

    def _model_input_video_enabled(self):
        cfg = self._model_input_video_config()
        if not bool(cfg.get("enabled", True)):
            return False
        return self._model_input_eval_output_dir() is not None

    def _model_input_eval_output_dir(self):
        if self.dynamic_finalizer_kwargs is not None:
            return Path(self.dynamic_finalizer_kwargs["eval_output_dir"])
        eval_output = os.environ.get("LIBERO_DYNAMIC_EVAL_OUTPUT_DIR")
        if eval_output:
            return Path(eval_output).expanduser().resolve()
        log_dir = os.environ.get("LIBERO_DYNAMIC_LOG_DIR")
        if log_dir:
            log_path = Path(log_dir).expanduser().resolve()
            if log_path.name == "libero_dynamic_logs":
                return log_path.parent
        return None

    def _model_input_video_fps(self):
        cfg = self._model_input_video_config()
        return float(cfg.get("fps", 2.0))

    def _model_input_video_dir(self):
        eval_output_dir = self._model_input_eval_output_dir()
        if eval_output_dir is None:
            return None
        cfg = self._model_input_video_config()
        output_dir_cfg = cfg.get("output_dir")
        if output_dir_cfg:
            output_dir = Path(output_dir_cfg).expanduser()
            if not output_dir.is_absolute():
                output_dir = eval_output_dir / output_dir
        else:
            output_dir = eval_output_dir / str(cfg.get("output_dir_name", "model_input_observation_videos"))
        return output_dir / str(self.dynamic_suite_name) / "videos"

    def _processed_dynamic_task_description(self):
        description = str(getattr(self, "language_instruction", "dynamic_task"))
        return (
            description.lower()
            .replace(" ", "_")
            .replace("\n", "_")
            .replace(".", "_")[:50]
        )

    def _model_input_video_file(self):
        video_dir = self._model_input_video_dir()
        if video_dir is None:
            return None
        video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        task_description = self._processed_dynamic_task_description()
        return video_dir / (
            f"{timestamp}--episode=task{int(self.dynamic_task_id)}_trial{int(self.dynamic_episode_index)}"
            f"--task={task_description}--model-input-observations.mp4"
        )

    def _close_model_input_video_writer(self):
        writer = getattr(self, "dynamic_model_input_video_writer", None)
        if writer is not None:
            writer.close()
        self.dynamic_model_input_video_writer = None

    def _save_model_input_observation_frame(self, *, latency_seconds):
        if not self._model_input_video_enabled() or not self.dynamic_episode_active:
            return None
        obs = self.env._get_observations()
        frame = self._obs_to_dynamic_video_frame(obs)
        stitched = self._stitch_dynamic_video_frame(frame) if frame is not None else None
        if stitched is None:
            return None
        if self.dynamic_model_input_video_writer is None:
            video_path = self._model_input_video_file()
            if video_path is None:
                return None
            self.dynamic_model_input_video_path = str(video_path)
            self.dynamic_model_input_video_writer = imageio.get_writer(
                video_path,
                fps=self._model_input_video_fps(),
            )
        self.dynamic_model_input_video_writer.append_data(stitched)
        self.dynamic_model_input_frame_count += 1
        return {
            "model_input_video": self.dynamic_model_input_video_path,
            "model_input_frame_index": int(self.dynamic_model_input_frame_count - 1),
            "latency_seconds": float(latency_seconds),
        }


    def _log_dynamic_event(self, event):
        if self.dynamic_logger is None or not self.dynamic_episode_active:
            return
        self.dynamic_logger.append_event(
            suite_name=self.dynamic_suite_name,
            task_id=self.dynamic_task_id,
            episode_index=self.dynamic_episode_index,
            event=event,
        )

    def _log_dynamic_latency(self, latency_seconds):
        if self.dynamic_logger is None or not self.dynamic_episode_active:
            return
        self.dynamic_logger.append_latency(
            suite_name=self.dynamic_suite_name,
            task_id=self.dynamic_task_id,
            episode_index=self.dynamic_episode_index,
            step_idx=self.dynamic_step_idx,
            latency_seconds=latency_seconds,
        )

    def _finalize_dynamic_episode(self, status):
        if (
            self.dynamic_logger is None
            or not self.dynamic_episode_active
            or self.dynamic_episode_finalized
        ):
            return
        finalized_episode_index = self.dynamic_episode_index
        self.dynamic_logger.finalize_episode(
            suite_name=self.dynamic_suite_name,
            task_id=self.dynamic_task_id,
            episode_index=finalized_episode_index,
            status=status,
        )
        self._close_model_input_video_writer()
        self.dynamic_episode_finalized = True
        self._schedule_dynamic_episode_finalize(finalized_episode_index)

    def _finalize_config(self):
        return self.dynamic_config.get("finalize", {}) or {}

    def _resolve_dynamic_eval_output_dir(self, finalize_cfg):
        explicit = finalize_cfg.get("eval_output_dir") or os.environ.get("LIBERO_DYNAMIC_EVAL_OUTPUT_DIR")
        if explicit:
            return Path(explicit).expanduser().resolve()
        if self.dynamic_logger is None:
            return None

        log_output_path = self.dynamic_logger.output_path.expanduser()
        if not os.environ.get("LIBERO_DYNAMIC_LOG_DIR") and not log_output_path.is_absolute():
            return None
        log_dir = log_output_path.parent.resolve()
        if log_dir.name == "libero_dynamic_logs":
            return log_dir.parent
        return None

    def _setup_dynamic_auto_finalize(self):
        if self.dynamic_logger is None:
            return
        finalize_cfg = self._finalize_config()
        if not bool(finalize_cfg.get("enabled", False)):
            return
        if not _env_flag_enabled("LIBERO_DYNAMIC_AUTO_FINALIZE", default=True):
            return

        eval_output_dir = self._resolve_dynamic_eval_output_dir(finalize_cfg)
        if eval_output_dir is None:
            print(
                "[LIBERO-dynamic warning] auto finalization enabled but eval output "
                "dir is unknown. Set LIBERO_DYNAMIC_EVAL_OUTPUT_DIR or "
                "LIBERO_DYNAMIC_LOG_DIR=<eval_output>/libero_dynamic_logs."
            )
            return

        output_dir_cfg = finalize_cfg.get("output_dir")
        if output_dir_cfg:
            output_dir = Path(output_dir_cfg).expanduser()
            if not output_dir.is_absolute():
                output_dir = eval_output_dir / output_dir
        else:
            output_dir = eval_output_dir / str(finalize_cfg.get("output_dir_name", "dynamic_real_time_videos"))

        self.dynamic_finalizer_kwargs = {
            "eval_output_dir": eval_output_dir,
            "dynamic_log_dir": self.dynamic_logger.output_path.parent.expanduser().resolve(),
            "output_dir": output_dir.expanduser().resolve(),
            "num_steps_wait": int(finalize_cfg.get("num_steps_wait", 30)),
            "fps": float(finalize_cfg.get("fps", finalize_cfg.get("video_fps", 24.0))),
            "min_latency_seconds": float(finalize_cfg.get("min_latency_seconds", 0.05)),
            "cleanup_latency_frames": bool(finalize_cfg.get("cleanup_latency_frames", True)),
        }
        _register_dynamic_atexit_finalizer(self.dynamic_finalizer_kwargs)

    def _rollout_video_glob(self, episode_index):
        if self.dynamic_finalizer_kwargs is None:
            return None
        eval_output_dir = Path(self.dynamic_finalizer_kwargs["eval_output_dir"])
        return (
            eval_output_dir
            / str(self.dynamic_suite_name)
            / "videos"
            / f"*episode=task{int(self.dynamic_task_id)}_trial{int(episode_index)}*.mp4"
        )

    def _wait_for_rollout_video(self, episode_index):
        finalize_cfg = self._finalize_config()
        timeout = float(finalize_cfg.get("wait_timeout_seconds", 600.0))
        poll = float(finalize_cfg.get("poll_interval_seconds", 0.5))
        poll = max(0.05, poll)
        pattern = self._rollout_video_glob(episode_index)
        if pattern is None:
            return None

        deadline = time.time() + max(0.0, timeout)
        last_path = None
        last_size = -1
        while time.time() <= deadline:
            candidates = sorted(pattern.parent.glob(pattern.name), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            if candidates:
                path = candidates[-1]
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                if path == last_path and size > 0 and size == last_size:
                    return path
                last_path = path
                last_size = size
            time.sleep(poll)
        return None

    def _schedule_dynamic_episode_finalize(self, episode_index):
        if self.dynamic_finalizer_kwargs is None:
            return

        def _worker():
            try:
                video_path = self._wait_for_rollout_video(episode_index)
                if video_path is None:
                    print(
                        "[LIBERO-dynamic warning] rollout video not found before "
                        f"auto-finalize timeout: suite={self.dynamic_suite_name} "
                        f"task={self.dynamic_task_id} episode={episode_index}"
                    )
                    return
                summary = _run_dynamic_results_finalizer_once(**self.dynamic_finalizer_kwargs)
                print(
                    "[LIBERO-dynamic] finalized episode outputs: "
                    f"suite={self.dynamic_suite_name} task={self.dynamic_task_id} "
                    f"episode={episode_index} real_time_videos={summary.get('num_merged_videos')}"
                )
            except Exception as exc:
                print(f"[LIBERO-dynamic warning] episode auto finalization failed: {exc}")

        thread = threading.Thread(
            target=_worker,
            name=f"libero_dynamic_finalize_task{self.dynamic_task_id}_episode{episode_index}",
            daemon=True,
        )
        self.dynamic_finalize_threads.append(thread)
        thread.start()


    def _motion_level_alpha_mps(self, level):
        presets = {
            "L0": 0.0,
            "L1": 0.03,
            "L2": 0.06,
            "L3": 0.10,
        }
        return float(presets.get(str(level).upper(), presets["L1"]))

    def _motion_direction_xyz(self, motion_cfg):
        if "direction_xyz" in motion_cfg:
            direction = np.asarray(motion_cfg["direction_xyz"], dtype=float)
        else:
            axis = str(motion_cfg.get("axis", "x")).lower()
            axis_to_direction = {
                "x": [1.0, 0.0, 0.0],
                "-x": [-1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "-y": [0.0, -1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
                "-z": [0.0, 0.0, -1.0],
            }
            direction = np.asarray(axis_to_direction.get(axis, axis_to_direction["x"]), dtype=float)

        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return np.zeros(3, dtype=float)
        return direction / norm

    def _reset_dynamic_motion_state(self):
        motion_cfg = self.dynamic_config.get("motion", {}) or {}
        self.dynamic_motion_elapsed_seconds = 0.0
        self.dynamic_motion_position_m = float(motion_cfg.get("initial_offset_m", 0.0))
        initial_direction = str(motion_cfg.get("initial_direction", "positive")).lower()
        self.dynamic_motion_direction_sign = -1.0 if initial_direction in {"negative", "-", "-1", "backward"} else 1.0
        initial_offset_xyz = motion_cfg.get("initial_offset_xyz", None)
        if initial_offset_xyz is None:
            self.dynamic_motion_offset_xyz = self._motion_direction_xyz(motion_cfg) * self.dynamic_motion_position_m
        else:
            self.dynamic_motion_offset_xyz = np.asarray(initial_offset_xyz, dtype=float)

    def _is_pingpong_motion_type(self, motion_type):
        return str(motion_type).lower() in {"conveyor_pingpong", "pingpong", "triangle_wave"}

    def _is_bounded_motion_type(self, motion_type):
        motion_type = str(motion_type).lower()
        return motion_type.startswith(("body_yaw_", "body_translation_")) or motion_type in {
            "conveyor_pingpong",
            "pingpong",
            "triangle_wave",
            "sinusoidal_pingpong",
            "sine_pingpong",
            "polynomial_spline",
            "smoothstep_pingpong",
            "zigzag_waypoints",
            "stochastic_segmented",
        }

    def _resolved_motion_type(self, motion_cfg, default=""):
        motion_type = str(motion_cfg.get("type", "")).lower()
        trajectory = str(motion_cfg.get("trajectory", "")).lower()
        if motion_type.startswith(("body_yaw_", "body_translation_")):
            return motion_type
        return trajectory or motion_type or str(default).lower()

    def _initialize_dynamic_fixture_body_state(self):
        role = str(self.dynamic_config.get("dynamic_role", "")).lower()
        motion_type = str((self.dynamic_config.get("motion", {}) or {}).get("type", "")).lower()
        if role not in {"fixture_body_yaw_dynamic", "fixture_body_translation_dynamic"} and not motion_type.startswith(
            ("body_yaw_", "body_translation_")
        ):
            self.dynamic_fixture_body_state = None
            return

        target_name = str(self.dynamic_config.get("target_object_name", ""))
        body_name = self.dynamic_config.get("target_body_name")
        body_id = None
        if body_name:
            try:
                body_id = int(self.env.sim.model.body_name2id(str(body_name)))
            except Exception:
                body_id = None
        if body_id is None:
            body_id = self._fixture_body_id(target_name)
        if body_id is None:
            raise KeyError(f"Could not find fixture body for dynamic target: {target_name}")

        self.dynamic_fixture_body_state = {
            "target_name": target_name,
            "body_id": int(body_id),
            "body_name": str(self.env.sim.model.body_id2name(int(body_id))),
            "canonical_pos": np.asarray(self.env.sim.model.body_pos[int(body_id)], dtype=float).copy(),
            "canonical_quat": np.asarray(self.env.sim.model.body_quat[int(body_id)], dtype=float).copy(),
            "base_pos": np.asarray(self.env.sim.model.body_pos[int(body_id)], dtype=float).copy(),
            "base_quat": np.asarray(self.env.sim.model.body_quat[int(body_id)], dtype=float).copy(),
            "elapsed": 0.0,
            "last_offset_xyz": np.zeros(3, dtype=float),
            "last_yaw_rad": 0.0,
        }

    def _restore_dynamic_fixture_body_canonical_pose(self):
        state = self.dynamic_fixture_body_state
        if state is None:
            return
        body_id = int(state["body_id"])
        self.env.sim.model.body_pos[body_id] = np.asarray(state["canonical_pos"], dtype=float)
        self.env.sim.model.body_quat[body_id] = np.asarray(state["canonical_quat"], dtype=float)
        self.env.sim.forward()

    def _start_dynamic_fixture_body_episode(self):
        state = self.dynamic_fixture_body_state
        if state is None:
            return
        body_id = int(state["body_id"])
        state["base_pos"] = np.asarray(self.env.sim.model.body_pos[body_id], dtype=float).copy()
        state["base_quat"] = np.asarray(self.env.sim.model.body_quat[body_id], dtype=float).copy()
        state["elapsed"] = 0.0
        state["last_offset_xyz"] = np.zeros(3, dtype=float)
        state["last_yaw_rad"] = 0.0

    def _fixture_body_translation_offset(self, motion_cfg, elapsed_seconds):
        normalized = dict(motion_cfg)
        motion_type = self._resolved_motion_type(motion_cfg)
        normalized["type"] = motion_type.removeprefix("body_translation_")
        normalized["trajectory"] = normalized["type"]
        return np.asarray(self._motion_waypoint_offset(normalized, elapsed_seconds), dtype=float)

    def _fixture_body_yaw_offset(self, motion_cfg, elapsed_seconds):
        waypoints = motion_cfg.get("waypoints_yaw_rad")
        speed = max(
            1e-9,
            abs(float(motion_cfg.get("alpha_yaw_rad_per_s", motion_cfg.get("alpha_qpos_per_s", 0.20)))),
        )
        if waypoints:
            points = [[float(value), 0.0, 0.0] for value in waypoints]
            loop = bool(motion_cfg.get("loop", True))
            motion_type = str(motion_cfg.get("type", "")).lower()
            if motion_type == "body_yaw_polynomial_spline":
                points = self._curve_waypoint_samples(
                    points,
                    loop,
                    int(motion_cfg.get("curve_samples_per_segment", 16) or 16),
                )
            return float(self._polyline_motion_offset(points, elapsed_seconds, speed, loop)[0])

        amplitude = float(
            motion_cfg.get(
                "amplitude_yaw_rad",
                motion_cfg.get("amplitude_qpos", np.deg2rad(15.0)),
            )
        )
        period = max(1.0, 4.0 * abs(amplitude) / speed)
        return float(amplitude * np.sin(2.0 * np.pi * float(elapsed_seconds) / period))

    def _advance_dynamic_fixture_body(self, motion_cfg, delta_seconds, *, reason):
        state = self.dynamic_fixture_body_state
        if state is None:
            raise RuntimeError("Fixture-body motion requested without initialized fixture-body state")
        motion_type = self._resolved_motion_type(motion_cfg)
        old_elapsed = float(state["elapsed"])
        new_elapsed = old_elapsed + max(0.0, float(delta_seconds))
        body_id = int(state["body_id"])
        before_pos = np.asarray(self.env.sim.model.body_pos[body_id], dtype=float).copy()
        before_quat = np.asarray(self.env.sim.model.body_quat[body_id], dtype=float).copy()

        offset_xyz = np.zeros(3, dtype=float)
        yaw_rad = 0.0
        if motion_type.startswith("body_translation_"):
            offset_xyz = self._fixture_body_translation_offset(motion_cfg, new_elapsed)
            self.env.sim.model.body_pos[body_id] = np.asarray(state["base_pos"], dtype=float) + offset_xyz
            self.env.sim.model.body_quat[body_id] = np.asarray(state["base_quat"], dtype=float)
        elif motion_type.startswith("body_yaw_"):
            yaw_rad = self._fixture_body_yaw_offset(motion_cfg, new_elapsed)
            self.env.sim.model.body_pos[body_id] = np.asarray(state["base_pos"], dtype=float)
            self.env.sim.model.body_quat[body_id] = self._quat_multiply_wxyz(
                self._yaw_quat_wxyz(yaw_rad),
                np.asarray(state["base_quat"], dtype=float),
            )
        else:
            raise ValueError(f"Unsupported fixture-body dynamic_motion type: {motion_type}")

        self.env.sim.forward()
        after_pos = np.asarray(self.env.sim.model.body_pos[body_id], dtype=float).copy()
        after_quat = np.asarray(self.env.sim.model.body_quat[body_id], dtype=float).copy()
        state["elapsed"] = new_elapsed
        state["last_offset_xyz"] = offset_xyz
        state["last_yaw_rad"] = yaw_rad
        if self.dynamic_controller is not None:
            self.dynamic_controller.cumulative_shift = [float(x) for x in offset_xyz]
        return {
            "reason": reason,
            "mode": "fixture_body_pose",
            "step_idx": self.dynamic_step_idx,
            "target_object_name": state["target_name"],
            "target_body_name": state["body_name"],
            "motion_type": motion_type,
            "delta_seconds": float(delta_seconds),
            "elapsed_seconds": new_elapsed,
            "delta_xyz": [float(x) for x in (after_pos - before_pos)],
            "offset_xyz": [float(x) for x in offset_xyz],
            "yaw_offset_rad": float(yaw_rad),
            "before_body_pos": before_pos.tolist(),
            "after_body_pos": after_pos.tolist(),
            "before_body_quat": before_quat.tolist(),
            "after_body_quat": after_quat.tolist(),
        }

    def _motion_speed_and_amplitude(self, motion_cfg):
        speed = abs(float(motion_cfg.get("alpha_mps", self._motion_level_alpha_mps(motion_cfg.get("level", "L1")))))
        amplitude = abs(float(motion_cfg.get("amplitude_m", motion_cfg.get("max_abs_offset_m", 0.06))))
        return speed, amplitude

    def _advance_pingpong_motion_delta(self, motion_cfg, delta_seconds):
        speed = abs(float(motion_cfg.get("alpha_mps", self._motion_level_alpha_mps(motion_cfg.get("level", "L1")))))
        amplitude = abs(float(motion_cfg.get("amplitude_m", motion_cfg.get("max_abs_offset_m", 0.06))))
        if delta_seconds <= 0.0 or speed <= 0.0 or amplitude <= 0.0:
            return np.zeros(3, dtype=float)

        old_position = float(np.clip(self.dynamic_motion_position_m, -amplitude, amplitude))
        position = old_position
        direction_sign = 1.0 if self.dynamic_motion_direction_sign >= 0.0 else -1.0
        remaining = speed * float(delta_seconds)

        # Reflect at +/- amplitude so the target keeps moving without drifting
        # out of the safe camera / collision range.
        while remaining > 1e-12:
            boundary = amplitude if direction_sign >= 0.0 else -amplitude
            distance_to_boundary = abs(boundary - position)
            if remaining <= distance_to_boundary + 1e-12:
                position = position + direction_sign * remaining
                remaining = 0.0
            else:
                position = boundary
                remaining -= distance_to_boundary
                direction_sign *= -1.0

        self.dynamic_motion_elapsed_seconds += float(delta_seconds)
        self.dynamic_motion_position_m = float(position)
        self.dynamic_motion_direction_sign = float(direction_sign)
        delta_position = float(position - old_position)
        return self._motion_direction_xyz(motion_cfg) * delta_position

    def _smoothstep(self, u):
        u = np.clip(float(u), 0.0, 1.0)
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    def _offset_for_sinusoidal_motion(self, motion_cfg, elapsed_seconds):
        speed, amplitude = self._motion_speed_and_amplitude(motion_cfg)
        if speed <= 0.0 or amplitude <= 0.0:
            return np.zeros(3, dtype=float)
        omega = float(motion_cfg.get("omega", motion_cfg.get("omega_rad_s", speed / amplitude)))
        phase = float(motion_cfg.get("phase", 0.0))
        scalar = amplitude * np.sin(omega * float(elapsed_seconds) + phase)
        return self._motion_direction_xyz(motion_cfg) * scalar

    def _offset_for_polynomial_spline_motion(self, motion_cfg, elapsed_seconds):
        speed, amplitude = self._motion_speed_and_amplitude(motion_cfg)
        if speed <= 0.0 or amplitude <= 0.0:
            return np.zeros(3, dtype=float)
        half_period = float(motion_cfg.get("half_period_seconds", max(1e-6, 2.0 * amplitude / speed)))
        phase = (float(elapsed_seconds) / half_period) % 2.0
        if phase <= 1.0:
            scalar = -amplitude + 2.0 * amplitude * self._smoothstep(phase)
        else:
            scalar = amplitude - 2.0 * amplitude * self._smoothstep(phase - 1.0)
        return self._motion_direction_xyz(motion_cfg) * scalar

    def _zigzag_waypoints_xyz(self, motion_cfg):
        if "waypoints_xyz" in motion_cfg:
            return [np.asarray(point, dtype=float) for point in motion_cfg["waypoints_xyz"]]
        _, amplitude = self._motion_speed_and_amplitude(motion_cfg)
        main = self._motion_direction_xyz(motion_cfg)
        axis = str(motion_cfg.get("axis", "y")).lower().replace("-", "")
        ortho_axis = str(motion_cfg.get("zigzag_orthogonal_axis", "x" if axis != "x" else "y")).lower()
        ortho = self._motion_direction_xyz({"axis": ortho_axis})
        lateral = float(motion_cfg.get("zigzag_lateral_m", 0.35 * amplitude))
        return [
            np.zeros(3, dtype=float),
            main * amplitude,
            main * amplitude + ortho * lateral,
            -main * amplitude + ortho * lateral,
            -main * amplitude - ortho * lateral,
            np.zeros(3, dtype=float),
        ]

    def _offset_for_zigzag_waypoints_motion(self, motion_cfg, elapsed_seconds):
        speed, amplitude = self._motion_speed_and_amplitude(motion_cfg)
        waypoints = self._zigzag_waypoints_xyz(motion_cfg)
        if speed <= 0.0 or len(waypoints) < 2:
            return waypoints[0] if waypoints else np.zeros(3, dtype=float)
        loop = bool(motion_cfg.get("loop", True))
        durations = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            durations.append(max(1e-6, float(np.linalg.norm(end - start)) / speed))
        total = sum(durations)
        t = float(elapsed_seconds)
        if loop:
            t = t % total
        else:
            t = min(t, total)
        accum = 0.0
        for idx, duration in enumerate(durations):
            if t <= accum + duration or idx == len(durations) - 1:
                u = (t - accum) / duration
                return waypoints[idx] * (1.0 - u) + waypoints[idx + 1] * u
            accum += duration
        return waypoints[-1]

    def _advance_parametric_motion_delta(self, motion_cfg, delta_seconds):
        if delta_seconds <= 0.0:
            return np.zeros(3, dtype=float)
        motion_type = str(motion_cfg.get("type", motion_cfg.get("trajectory", "conveyor_pingpong"))).lower()
        old_offset = np.asarray(getattr(self, "dynamic_motion_offset_xyz", np.zeros(3)), dtype=float)
        new_elapsed = float(self.dynamic_motion_elapsed_seconds) + float(delta_seconds)
        if motion_type in {"sinusoidal_pingpong", "sine_pingpong"}:
            new_offset = self._offset_for_sinusoidal_motion(motion_cfg, new_elapsed)
        elif motion_type in {"polynomial_spline", "smoothstep_pingpong"}:
            new_offset = self._offset_for_polynomial_spline_motion(motion_cfg, new_elapsed)
        elif motion_type in {"zigzag_waypoints", "stochastic_segmented"}:
            # The first implementation treats stochastic_segmented as a seeded
            # waypoint family. Later configs can pass explicit sampled waypoints
            # for exact reproducibility.
            new_offset = self._offset_for_zigzag_waypoints_motion(motion_cfg, new_elapsed)
        else:
            raise ValueError(f"Unsupported parametric dynamic_motion type: {motion_type}")
        self.dynamic_motion_elapsed_seconds = new_elapsed
        self.dynamic_motion_offset_xyz = np.asarray(new_offset, dtype=float)
        return self.dynamic_motion_offset_xyz - old_offset

    def _carrier_config(self):
        return self.dynamic_config.get("carrier", {}) or {}

    def _dynamic_target_entries(self):
        entries = self.dynamic_config.get("multi_target_objects") or self.dynamic_config.get("target_objects") or []
        if entries:
            return [dict(entry) for entry in entries]
        if self.dynamic_controller is None:
            return []
        return [
            {
                "target_object_name": self.dynamic_controller.target.object_name,
                "target_joint_name": self.dynamic_controller.target.joint_name,
                "motion": dict(self.dynamic_config.get("motion", {}) or {}),
            }
        ]

    def _object_joint_name(self, object_name):
        if object_name in getattr(self.env, "objects_dict", {}):
            joints = getattr(self.env.objects_dict[object_name], "joints", None) or []
            if joints:
                return str(joints[0])
        if object_name in getattr(self.env, "fixtures_dict", {}):
            joints = getattr(self.env.fixtures_dict[object_name], "joints", None) or []
            if joints:
                return str(joints[0])
        return str(object_name)

    def _target_contact_geom_names(self, object_name, bottom_margin_m=0.002):
        if object_name not in getattr(self.env, "objects_dict", {}):
            return []
        obj = self.env.objects_dict[object_name]
        candidates = []
        for geom_name in list(getattr(obj, "contact_geoms", []) or []):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            if int(self.env.sim.model.geom_contype[geom_id]) == 0:
                continue
            geom_z = float(self.env.sim.data.geom_xpos[geom_id][2])
            geom_size_z = float(self.env.sim.model.geom_size[geom_id][2])
            xy_area = float(self.env.sim.model.geom_size[geom_id][0] * self.env.sim.model.geom_size[geom_id][1])
            candidates.append((geom_z - geom_size_z, xy_area, str(geom_name)))
        if not candidates:
            return []
        min_bottom = min(bottom for bottom, _, _ in candidates)
        bottom_candidates = [
            (area, name)
            for bottom, area, name in candidates
            if bottom <= min_bottom + min(float(bottom_margin_m), 0.002)
        ]
        return sorted({name for _, name in (bottom_candidates or [(0.0, min(candidates)[2])])})

    def _target_all_contact_geom_names(self, object_name):
        if object_name not in getattr(self.env, "objects_dict", {}):
            return []
        out = []
        for geom_name in list(getattr(self.env.objects_dict[object_name], "contact_geoms", []) or []):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            if int(self.env.sim.model.geom_contype[geom_id]) != 0:
                out.append(str(geom_name))
        return sorted(set(out))

    def _target_body_names(self, object_name):
        names = []
        for geom_name in self._target_all_contact_geom_names(object_name):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
                body_id = int(self.env.sim.model.geom_bodyid[geom_id])
                body_name = self.env.sim.model.body_id2name(body_id)
            except Exception:
                continue
            if body_name:
                names.append(str(body_name))
        return sorted(set(names))

    def _object_joint_body_name(self, object_name):
        try:
            joint_id = int(self.env.sim.model.joint_name2id(self._object_joint_name(object_name)))
            body_id = int(self.env.sim.model.jnt_bodyid[joint_id])
            return self.env.sim.model.body_id2name(body_id)
        except Exception:
            return None

    def _world_to_body_local(self, body_name, world_xyz):
        body_id = int(self.env.sim.model.body_name2id(body_name))
        body_pos = np.asarray(self.env.sim.data.body_xpos[body_id], dtype=float)
        body_rot = np.asarray(self.env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        return body_rot.T @ (np.asarray(world_xyz, dtype=float) - body_pos)

    def _support_body_names(self):
        keywords = ("table", "counter", "floor", "arena", "living_room_table")
        out = []
        for body_id in range(int(self.env.sim.model.nbody)):
            body_name = self.env.sim.model.body_id2name(body_id)
            if body_name and any(keyword in body_name for keyword in keywords):
                out.append(str(body_name))
        return sorted(set(out))

    def _target_platform_xy_center(self, object_name):
        points = []
        for geom_name in self._target_contact_geom_names(object_name):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            points.append(np.asarray(self.env.sim.data.geom_xpos[geom_id][:2], dtype=float))
        if points:
            return np.median(np.asarray(points, dtype=float), axis=0)
        qpos = np.asarray(self.env.sim.data.get_joint_qpos(self._object_joint_name(object_name)), dtype=float)
        return qpos[:2].copy()

    def _target_bottom_z(self, object_name):
        bottoms = []
        for geom_name in self._target_contact_geom_names(object_name):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            bottoms.append(float(self.env.sim.data.geom_xpos[geom_id][2]) - float(self.env.sim.model.geom_size[geom_id][2]))
        if not bottoms:
            qpos = np.asarray(self.env.sim.data.get_joint_qpos(self._object_joint_name(object_name)), dtype=float)
            return float(qpos[2]) if qpos.size >= 3 else 0.9
        return float(min(bottoms))

    def _estimate_platform_size(self, object_name):
        if "moka_pot" in str(object_name):
            return [0.06, 0.06, 0.00025]
        center_xy = self._target_platform_xy_center(object_name)
        radii = []
        for geom_name in self._target_contact_geom_names(object_name):
            try:
                geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            geom_pos = np.asarray(self.env.sim.data.geom_xpos[geom_id], dtype=float)
            rbound = float(self.env.sim.model.geom_rbound[geom_id])
            radii.append(np.abs(geom_pos[:2] - center_xy) + rbound)
        if not radii:
            return [0.06, 0.06, 0.00025]
        half_xy = np.max(np.asarray(radii, dtype=float), axis=0) + 0.035
        if "bowl" in str(object_name).lower():
            half_xy += 0.025
        half_xy = np.clip(half_xy, 0.06, 0.20)
        return [float(half_xy[0]), float(half_xy[1]), 0.00025]

    def _stability_proxy_half_size(self, target, object_name):
        if target.get("stability_proxy_size") not in (None, "auto"):
            values = [float(x) for x in target["stability_proxy_size"]]
            if len(values) == 2:
                values.append(float(target.get("stability_proxy_thickness_m", 0.0015)))
            return values[:3]
        if target.get("stability_proxy_size_xy_m") is not None:
            xy = [float(x) for x in target["stability_proxy_size_xy_m"]]
        else:
            platform_size = self._estimate_platform_size(object_name)
            xy = [
                float(np.clip(platform_size[0] * 0.75, 0.035, 0.075)),
                float(np.clip(platform_size[1] * 0.75, 0.035, 0.075)),
            ]
        return [xy[0], xy[1], float(target.get("stability_proxy_thickness_m", 0.0015))]

    def _add_stability_proxy_geom(self, root, target, object_name, proxy_idx):
        if not bool(target.get("stability_proxy_enabled", False)):
            return None
        body_name = self._object_joint_body_name(object_name)
        if not body_name:
            body_names = self._target_body_names(object_name)
            body_name = body_names[0] if body_names else None
        if not body_name:
            return None
        body_xml = root.find(f".//body[@name='{body_name}']")
        if body_xml is None:
            return None
        # Mesh/capsule assets can report a misleading geometric bottom.  When
        # the carrier measured a support contact plane, use that plane so the
        # invisible stability proxy is physically attached to the conveyor.
        support_z = target.get("preinstall_support_contact_z")
        bottom_z = float(support_z) if support_z is not None else self._target_bottom_z(object_name)
        if bottom_z is None:
            return None
        half_size = self._stability_proxy_half_size(target, object_name)
        proxy_name = f"libero_dynamic_stability_proxy_{proxy_idx}_{object_name}"
        if root.find(f".//geom[@name='{proxy_name}']") is not None:
            return proxy_name
        proxy_xy = self._target_platform_xy_center(object_name)
        world_center = np.asarray(
            [
                float(proxy_xy[0]),
                float(proxy_xy[1]),
                float(bottom_z + half_size[2] + float(target.get("stability_proxy_bottom_epsilon_m", 0.00015))),
            ],
            dtype=float,
        )
        local_center = self._world_to_body_local(body_name, world_center)
        ET.SubElement(
            body_xml,
            "geom",
            name=proxy_name,
            type="box",
            pos=" ".join(str(float(x)) for x in local_center),
            size=" ".join(str(float(x)) for x in half_size),
            rgba="1 0.8 0.1 0",
            contype="0",
            conaffinity="0",
            density="1",
        )
        return proxy_name

    def _smoothstep_scalar(self, u):
        u = float(np.clip(float(u), 0.0, 1.0))
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    def _motion_segments(self, waypoints, loop):
        pts = [np.asarray(point, dtype=float) for point in waypoints if len(point) >= 3]
        if len(pts) <= 1:
            return []
        if bool(loop) and float(np.linalg.norm(pts[-1] - pts[0])) > 1e-9:
            pts = [*pts, pts[0]]
        return [(pts[i], pts[i + 1], float(np.linalg.norm(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)]

    def _segment_accel_duration(self, length, speed, accel):
        length = float(length)
        speed = max(float(speed), 1e-9)
        accel = 0.0 if accel is None else float(accel)
        if length <= 1e-9:
            return 0.0
        if accel <= 1e-9:
            return length / speed
        accel_dist = speed * speed / accel
        if length <= accel_dist:
            return 2.0 * np.sqrt(length / accel)
        return 2.0 * speed / accel + (length - accel_dist) / speed

    def _segment_accel_distance(self, t, length, speed, accel):
        t = max(0.0, float(t))
        length = float(length)
        speed = max(float(speed), 1e-9)
        accel = 0.0 if accel is None else float(accel)
        if length <= 1e-9:
            return 0.0
        if accel <= 1e-9:
            return min(length, speed * t)
        accel_dist = speed * speed / accel
        if length <= accel_dist:
            peak_time = np.sqrt(length / accel)
            total = 2.0 * peak_time
            if t <= peak_time:
                return min(length, 0.5 * accel * t * t)
            dt = min(t, total) - peak_time
            return min(length, 0.5 * length + accel * peak_time * dt - 0.5 * accel * dt * dt)
        accel_time = speed / accel
        cruise_dist = length - accel_dist
        cruise_time = cruise_dist / speed
        total = 2.0 * accel_time + cruise_time
        if t <= accel_time:
            return min(length, 0.5 * accel * t * t)
        if t <= accel_time + cruise_time:
            return min(length, 0.5 * accel_dist + speed * (t - accel_time))
        dt = min(t, total) - accel_time - cruise_time
        return min(length, 0.5 * accel_dist + cruise_dist + speed * dt - 0.5 * accel * dt * dt)

    def _polyline_motion_duration(self, waypoints, speed, loop, stop_seconds=0.0, accel_mps2=None):
        segments = [seg for seg in self._motion_segments(waypoints, loop) if seg[2] > 1e-9]
        return float(
            sum(
                self._segment_accel_duration(length, speed, accel_mps2)
                + max(0.0, float(stop_seconds))
                for _a, _b, length in segments
            )
        )

    def _polyline_motion_offset(
        self,
        waypoints,
        elapsed_seconds,
        speed,
        loop,
        stop_seconds=0.0,
        smooth_stop=False,
        accel_mps2=None,
    ):
        pts = [np.asarray(point, dtype=float) for point in waypoints if len(point) >= 3]
        if not pts:
            return np.zeros(3, dtype=float)
        if len(pts) == 1:
            return pts[0].copy()
        segments = self._motion_segments(waypoints, loop)
        total = self._polyline_motion_duration(waypoints, speed, loop, stop_seconds, accel_mps2)
        if total <= 1e-9:
            return pts[0].copy()
        t = max(0.0, float(elapsed_seconds))
        if bool(loop):
            t = t % total
        else:
            t = min(t, total)
        for a, b, length in segments:
            if length <= 1e-9:
                continue
            move_duration = self._segment_accel_duration(length, speed, accel_mps2)
            if t <= move_duration:
                if accel_mps2 is not None and float(accel_mps2) > 1e-9:
                    ratio = self._segment_accel_distance(t, length, speed, accel_mps2) / max(length, 1e-9)
                else:
                    ratio = t / max(move_duration, 1e-9)
                    if smooth_stop:
                        ratio = self._smoothstep_scalar(ratio)
                return a + ratio * (b - a)
            t -= move_duration
            dwell = max(0.0, float(stop_seconds))
            if t <= dwell:
                return b.copy()
            t -= dwell
        return segments[-1][1].copy() if segments else pts[-1].copy()

    def _catmull_rom_point(self, p0, p1, p2, p3, u):
        u = float(np.clip(float(u), 0.0, 1.0))
        u2 = u * u
        u3 = u2 * u
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )

    def _curve_waypoint_samples(self, waypoints, loop, samples_per_segment=16):
        pts = [np.asarray(point, dtype=float) for point in waypoints if len(point) >= 3]
        if len(pts) <= 2:
            return [point.tolist() for point in pts]
        if bool(loop) and float(np.linalg.norm(pts[-1] - pts[0])) <= 1e-9:
            pts = pts[:-1]
        n = len(pts)
        if n <= 2:
            return [point.tolist() for point in pts]
        samples = []
        segment_count = n if bool(loop) else n - 1
        samples_per_segment = max(4, int(samples_per_segment))
        for i in range(segment_count):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p0 = pts[(i - 1) % n] if bool(loop) or i > 0 else pts[i]
            p3 = pts[(i + 2) % n] if bool(loop) or i + 2 < n else pts[i + 1]
            for j in range(samples_per_segment):
                samples.append(self._catmull_rom_point(p0, p1, p2, p3, j / float(samples_per_segment)).tolist())
        if bool(loop):
            samples.append(samples[0])
        else:
            samples.append(pts[-1].tolist())
        return samples

    def _motion_waypoint_offset(self, motion_cfg, elapsed_seconds):
        waypoints = motion_cfg.get("waypoints_xyz")
        if not waypoints:
            old_elapsed = self.dynamic_motion_elapsed_seconds
            old_offset = self.dynamic_motion_offset_xyz.copy()
            self.dynamic_motion_elapsed_seconds = float(elapsed_seconds)
            self.dynamic_motion_offset_xyz = old_offset.copy()
            try:
                return np.asarray(self._offset_for_zigzag_waypoints_motion(motion_cfg, elapsed_seconds), dtype=float)
            finally:
                self.dynamic_motion_elapsed_seconds = old_elapsed
                self.dynamic_motion_offset_xyz = old_offset
        motion_type = str(motion_cfg.get("type", motion_cfg.get("trajectory", "zigzag_waypoints"))).lower()
        speed = max(
            1e-9,
            abs(float(motion_cfg.get("alpha_mps", self._motion_level_alpha_mps(motion_cfg.get("level", "L1"))))),
        )
        loop = bool(motion_cfg.get("loop", True))
        if motion_type in {"polynomial_spline", "smoothstep_pingpong"}:
            sampled = self._curve_waypoint_samples(
                waypoints,
                loop,
                int(motion_cfg.get("curve_samples_per_segment", 16) or 16),
            )
            return self._polyline_motion_offset(sampled, elapsed_seconds, speed, loop)
        accel = (
            float(motion_cfg.get("waypoint_accel_mps2"))
            if motion_cfg.get("waypoint_accel_mps2", None) is not None
            else None
        )
        return self._polyline_motion_offset(
            waypoints,
            elapsed_seconds,
            speed,
            loop,
            stop_seconds=float(motion_cfg.get("waypoint_stop_seconds", 0.0) or 0.0),
            smooth_stop=bool(motion_cfg.get("waypoint_smooth_stop", False)),
            accel_mps2=accel,
        )

    def _set_dynamic_platform_state(self, platform, xyz, velocity_xyz):
        joint_names = platform["joint_names"]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.env.sim.data.set_joint_qpos(joint_names[axis], float(xyz[axis_idx]))
            self.env.sim.data.set_joint_qvel(joint_names[axis], float(velocity_xyz[axis_idx]))
        self.env.sim.forward()

    def _fixture_body_id(self, name):
        candidates = [f"{name}_main", str(name)]
        try:
            if name in getattr(self.env, "fixtures_dict", {}):
                fixture = self.env.fixtures_dict[name]
                root_body = getattr(fixture, "root_body", None)
                if root_body:
                    candidates.insert(0, str(root_body))
        except Exception:
            pass
        for candidate in candidates:
            try:
                return int(self.env.sim.model.body_name2id(str(candidate)))
            except Exception:
                pass
        for body_idx in range(int(self.env.sim.model.nbody)):
            body_name = self.env.sim.model.body_id2name(body_idx)
            if body_name and str(body_name).startswith(f"{name}_"):
                return int(body_idx)
        return None

    def _yaw_quat_delta(self, qpos, yaw_delta):
        qpos = np.asarray(qpos, dtype=float).copy()
        if qpos.size < 7 or abs(float(yaw_delta)) < 1e-12:
            return qpos
        half = 0.5 * float(yaw_delta)
        delta = np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)
        w1, x1, y1, z1 = delta
        w2, x2, y2, z2 = qpos[3:7]
        qpos[3:7] = [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
        qpos[3:7] /= max(1e-9, float(np.linalg.norm(qpos[3:7])))
        return qpos

    def _yaw_quat_wxyz(self, yaw_delta):
        half = 0.5 * float(yaw_delta)
        return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

    def _quat_multiply_wxyz(self, q1, q2):
        w1, x1, y1, z1 = np.asarray(q1, dtype=float)[:4]
        w2, x2, y2, z2 = np.asarray(q2, dtype=float)[:4]
        quat = np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
        quat /= max(1e-9, float(np.linalg.norm(quat)))
        return quat

    def _apply_dynamic_initial_scene_adjustments(self):
        adjustments = self.dynamic_config.get("initial_scene_adjustments", {}) or {}
        if not bool(adjustments.get("enabled", False)):
            return []

        events = []
        for name, item in (adjustments.get("object_offsets") or {}).items():
            offset = np.asarray(item.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=float)
            yaw_delta = float(item.get("yaw_delta_rad", 0.0) or 0.0)
            applied = False
            try:
                joint = self._object_joint_name(str(name))
                qpos = np.asarray(self.env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
                if qpos.ndim > 0 and qpos.size >= 3:
                    before = qpos.copy()
                    qpos[: min(3, qpos.size)] += offset[: min(3, qpos.size)]
                    qpos = self._yaw_quat_delta(qpos, yaw_delta)
                    self.env.sim.data.set_joint_qpos(joint, qpos)
                    events.append(
                        {
                            "reason": "initial_scene_adjustment",
                            "target_name": str(name),
                            "mode": "joint_qpos",
                            "joint_name": joint,
                            "offset_xyz": [float(x) for x in offset[:3]],
                            "yaw_delta_rad": float(yaw_delta),
                            "before_qpos": before.tolist(),
                            "after_qpos": qpos.tolist(),
                        }
                    )
                    applied = True
            except Exception:
                applied = False
            if applied:
                continue

            body_id = self._fixture_body_id(str(name))
            if body_id is None:
                raise KeyError(f"initial_scene_adjustments target not found: {name}")
            before_pos = np.asarray(self.env.sim.model.body_pos[body_id], dtype=float).copy()
            before_quat = np.asarray(self.env.sim.model.body_quat[body_id], dtype=float).copy()
            self.env.sim.model.body_pos[body_id] = before_pos + offset[:3]
            if abs(yaw_delta) > 1e-12:
                self.env.sim.model.body_quat[body_id] = self._quat_multiply_wxyz(
                    self._yaw_quat_wxyz(yaw_delta),
                    before_quat,
                )
            events.append(
                {
                    "reason": "initial_scene_adjustment",
                    "target_name": str(name),
                    "mode": "fixture_body",
                    "body_name": self.env.sim.model.body_id2name(body_id),
                    "offset_xyz": [float(x) for x in offset[:3]],
                    "yaw_delta_rad": float(yaw_delta),
                    "before_body_pos": before_pos.tolist(),
                    "after_body_pos": np.asarray(self.env.sim.model.body_pos[body_id], dtype=float).tolist(),
                }
            )

        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        return events

    def _install_dynamic_physical_platforms_if_needed(self):
        carrier_cfg = self._carrier_config()
        if str(carrier_cfg.get("type", "")).lower() not in {"physical_conveyor_patch", "physical_platform"}:
            return
        targets = self._dynamic_target_entries()
        if not targets:
            return

        root = ET.fromstring(self.env.model.get_xml())
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Cannot install dynamic physical platforms: MJCF has no worldbody")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")

        platforms = {}
        for idx, target in enumerate(targets):
            name = str(target.get("target_object_name") or target.get("object_name"))
            if name not in getattr(self.env, "objects_dict", {}):
                continue
            body_name = f"libero_dynamic_platform_{idx}_{name}"
            geom_name = f"{body_name}_geom"
            if root.find(f".//body[@name='{body_name}']") is not None:
                continue
            pair_geom_names = (
                self._target_all_contact_geom_names(name)
                if bool(target.get("contact_all_target_geoms", carrier_cfg.get("contact_all_target_geoms", True)))
                else self._target_contact_geom_names(name, float(target.get("bottom_contact_margin_m", 0.002)))
            )
            if not pair_geom_names:
                continue
            size = target.get("size", carrier_cfg.get("size", "auto"))
            if size in (None, "auto"):
                size = self._estimate_platform_size(name)
            size = [float(x) for x in size]
            friction = target.get("friction", carrier_cfg.get("friction", [1200.0, 80.0, 20.0]))
            friction_text = " ".join(str(float(x)) for x in friction)
            solref = str(target.get("solref", carrier_cfg.get("solref", "0.0005 1")))
            solimp = str(target.get("solimp", carrier_cfg.get("solimp", "0.995 0.999 0.0001")))
            margin = str(float(target.get("margin", carrier_cfg.get("margin", 0.001))))
            condim = str(int(target.get("condim", carrier_cfg.get("condim", 6))))
            joint_names = {"x": f"{body_name}_x", "y": f"{body_name}_y", "z": f"{body_name}_z"}
            body = ET.SubElement(worldbody, "body", name=body_name, pos="0 0 0")
            ET.SubElement(body, "joint", name=joint_names["x"], type="slide", axis="1 0 0", limited="false", damping="0")
            ET.SubElement(body, "joint", name=joint_names["y"], type="slide", axis="0 1 0", limited="false", damping="0")
            ET.SubElement(body, "joint", name=joint_names["z"], type="slide", axis="0 0 1", limited="false", damping="0")
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="box",
                size=" ".join(str(float(x)) for x in size),
                rgba="0.1 0.55 1.0 0.0",
                friction=friction_text,
                solref=solref,
                solimp=solimp,
                margin=margin,
                condim=condim,
                contype="0",
                conaffinity="0",
                density="10000",
            )
            proxy_geom_name = self._add_stability_proxy_geom(root, target, name, idx)
            platform_pair_geom_names = list(pair_geom_names)
            if proxy_geom_name:
                platform_pair_geom_names = sorted(set(platform_pair_geom_names + [proxy_geom_name]))
            for target_geom_name in platform_pair_geom_names:
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=geom_name,
                    geom2=target_geom_name,
                    condim=condim,
                    friction=friction_text,
                    solref=solref,
                    solimp=solimp,
                    margin=margin,
                )
            if bool(target.get("exclude_static_support_contact", carrier_cfg.get("exclude_static_support_contact", True))):
                for target_body in self._target_body_names(name):
                    for support_body in self._support_body_names():
                        ET.SubElement(contact, "exclude", body1=target_body, body2=support_body)
            platforms[name] = {
                "body_name": body_name,
                "geom_name": geom_name,
                "joint_names": joint_names,
                "size": size,
                "stability_proxy_geom_name": proxy_geom_name,
                "surface_epsilon_m": float(target.get("surface_epsilon_m", carrier_cfg.get("surface_epsilon_m", 0.001))),
            }

        if not platforms:
            return
        self.env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
        self.dynamic_physical_platforms = platforms
        self.dynamic_multi_target_states = []
        for target in targets:
            name = str(target.get("target_object_name") or target.get("object_name"))
            if name not in platforms:
                continue
            self.dynamic_multi_target_states.append({
                "name": name,
                "joint": str(target.get("target_joint_name") or self._object_joint_name(name)),
                "motion": dict(target.get("motion") or self.dynamic_config.get("motion", {}) or {}),
                "platform": platforms[name],
                "elapsed": 0.0,
                "last_offset": np.zeros(3, dtype=float),
                "last_velocity": np.zeros(3, dtype=float),
                "cumulative_shift": np.zeros(3, dtype=float),
                "platform_base_pos": None,
            })

    def _carrier_enabled(self):
        carrier_cfg = self._carrier_config()
        return bool(carrier_cfg.get("enabled", False)) and self.dynamic_carrier is not None

    def _use_carrier_for_target_motion(self):
        return self._carrier_enabled() and str(
            self.dynamic_config.get("dynamic_role", "")
        ) == "target_object_dynamic"

    def _carrier_contact_state(self, target_z):
        carrier_cfg = self._carrier_config()
        carrier_pos = self._get_dynamic_carrier_qpos()
        if carrier_pos is not None and len(carrier_pos) >= 3:
            default_surface_z = float(carrier_pos[2])
        else:
            default_surface_z = 0.895
        surface_z = float(carrier_cfg.get("surface_z", default_surface_z))
        release_margin = float(carrier_cfg.get("release_height_margin_m", 0.04))
        release_z = float(carrier_cfg.get("release_z", surface_z + release_margin))
        active = float(target_z) <= release_z
        skip_reason = None
        if not active:
            skip_reason = "target_above_carrier_release_height"
        return active, {
            "surface_z": surface_z,
            "release_height_margin_m": release_margin,
            "release_z": release_z,
            "carrier_active": bool(active),
            **({"skip_reason": skip_reason} if skip_reason is not None else {}),
        }

    def _apply_surface_velocity_to_target(self, delta_xyz, delta_seconds):
        dt = max(float(delta_seconds), 1e-9)
        requested_delta = np.asarray(delta_xyz, dtype=float)
        requested_velocity = requested_delta / dt
        return self._apply_surface_velocity_mps_to_target(requested_velocity)

    def _apply_surface_velocity_mps_to_target(self, velocity_mps):
        if self.dynamic_controller is None:
            return {}
        before_pose = self.dynamic_controller.get_body_pose()
        target_z = float(before_pose["pos"][2])
        active, carrier_state = self._carrier_contact_state(target_z)
        requested_velocity = np.asarray(velocity_mps, dtype=float)
        if not active:
            return {
                **carrier_state,
                "surface_velocity_mps": [0.0, 0.0, 0.0],
                "requested_surface_velocity_mps": [float(x) for x in requested_velocity[:3]],
            }
        joint_name = self.dynamic_controller.target.joint_name
        before_qvel = self.env.sim.data.get_joint_qvel(joint_name).copy()
        after_qvel = before_qvel.copy()
        after_qvel[:3] = requested_velocity[:3]
        self.env.sim.data.set_joint_qvel(joint_name, after_qvel)
        return {
            **carrier_state,
            "surface_velocity_mps": [float(x) for x in requested_velocity[:3]],
            "before_qvel": before_qvel.tolist(),
            "after_qvel": after_qvel.tolist(),
        }

    def _step_env_with_surface_velocity(self, action, velocity_mps):
        if velocity_mps is None:
            return self.env.step(action)
        sim = self.env.sim
        original_step = sim.step
        self.dynamic_surface_velocity_step_hook_count = 0

        def _step_with_surface_velocity(*args, **kwargs):
            event = self._apply_surface_velocity_mps_to_target(velocity_mps)
            if event.get("carrier_active", False):
                self.dynamic_surface_velocity_step_hook_count += 1
            return original_step(*args, **kwargs)

        sim.step = _step_with_surface_velocity
        try:
            return self.env.step(action)
        finally:
            sim.step = original_step

    def _step_env_with_dynamic_platform_hold(self, action):
        if not self.dynamic_multi_target_states:
            return self.env.step(action)
        sim = self.env.sim
        original_step = sim.step

        def _step_with_dynamic_platform_hold(*args, **kwargs):
            for state in self.dynamic_multi_target_states:
                platform = state.get("platform")
                if not platform:
                    continue
                base_pos = state.get("platform_base_pos")
                if base_pos is None:
                    continue
                pos = np.asarray(base_pos, dtype=float) + np.asarray(state.get("last_offset", np.zeros(3)), dtype=float)
                velocity = np.asarray(state.get("last_velocity", np.zeros(3)), dtype=float)
                self._set_dynamic_platform_state(platform, pos, velocity)
            return original_step(*args, **kwargs)

        sim.step = _step_with_dynamic_platform_hold
        try:
            return self.env.step(action)
        finally:
            sim.step = original_step

    def _install_dynamic_carrier_if_needed(self):
        carrier_cfg = self._carrier_config()
        if not bool(carrier_cfg.get("enabled", False)):
            return

        carrier_type = str(carrier_cfg.get("type", "invisible_conveyor"))
        if carrier_type == "surface_velocity":
            self.dynamic_carrier = {
                "body_name": str(carrier_cfg.get("body_name", "libero_dynamic_surface_velocity")),
                "geom_name": str(carrier_cfg.get("geom_name", "implicit_conveyor_surface")),
                "kinematic_mode": "surface_velocity",
                "joint_names": {},
                "qpos": [0.0, 0.0, float(carrier_cfg.get("surface_z", 0.895))],
            }
            return
        if carrier_type not in {"invisible_conveyor", "invisible_carrier", "moving_platform"}:
            raise ValueError(f"Unsupported dynamic carrier.type: {carrier_type}")

        body_name = str(carrier_cfg.get("body_name", "libero_dynamic_carrier"))
        geom_name = str(carrier_cfg.get("geom_name", f"{body_name}_collision"))
        kinematic_mode = str(carrier_cfg.get("kinematic_mode", "mocap"))
        joint_names = {
            "x": str(carrier_cfg.get("joint_x", f"{body_name}_x")),
            "y": str(carrier_cfg.get("joint_y", f"{body_name}_y")),
            "z": str(carrier_cfg.get("joint_z", f"{body_name}_z")),
        }

        xml = self.env.model.get_xml()
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Cannot install dynamic carrier: MJCF has no worldbody")

        if root.find(f".//body[@name='{body_name}']") is None:
            size = carrier_cfg.get("size", [0.09, 0.09, 0.006])
            friction = carrier_cfg.get("friction", [3.0, 0.05, 0.001])
            rgba = carrier_cfg.get(
                "rgba",
                [0.1, 0.4, 1.0, 0.0 if not carrier_cfg.get("visible", False) else 0.35],
            )
            body_kwargs = {"name": body_name, "pos": "0 0 0"}
            if kinematic_mode == "mocap":
                body_kwargs["mocap"] = "true"
            body = ET.SubElement(worldbody, "body", **body_kwargs)
            if kinematic_mode == "slide_joints":
                ET.SubElement(
                    body,
                    "joint",
                    name=joint_names["x"],
                    type="slide",
                    axis="1 0 0",
                    limited="false",
                    damping="0",
                )
                ET.SubElement(
                    body,
                    "joint",
                    name=joint_names["y"],
                    type="slide",
                    axis="0 1 0",
                    limited="false",
                    damping="0",
                )
                ET.SubElement(
                    body,
                    "joint",
                    name=joint_names["z"],
                    type="slide",
                    axis="0 0 1",
                    limited="false",
                    damping="0",
                )
            elif kinematic_mode != "mocap":
                raise ValueError(f"Unsupported dynamic carrier.kinematic_mode: {kinematic_mode}")
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="box",
                size=" ".join(str(float(x)) for x in size),
                rgba=" ".join(str(float(x)) for x in rgba),
                friction=" ".join(str(float(x)) for x in friction),
                contype=str(int(carrier_cfg.get("contype", 1))),
                conaffinity=str(int(carrier_cfg.get("conaffinity", 1))),
            )

        self.env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
        self.dynamic_carrier = {
            "body_name": body_name,
            "geom_name": geom_name,
            "kinematic_mode": kinematic_mode,
            "joint_names": joint_names if kinematic_mode == "slide_joints" else {},
        }

    def _set_dynamic_carrier_qpos(self, xyz):
        if self.dynamic_carrier is None:
            return
        if self.dynamic_carrier.get("kinematic_mode") == "surface_velocity":
            self.dynamic_carrier["qpos"] = [float(x) for x in xyz]
            return
        if self.dynamic_carrier.get("kinematic_mode") == "mocap":
            body_name = self.dynamic_carrier["body_name"]
            body_id = self.env.sim.model.body_name2id(body_name)
            mocap_id = int(self.env.sim.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"Dynamic carrier body is not a mocap body: {body_name}")
            self.env.sim.data.mocap_pos[mocap_id] = np.asarray(xyz, dtype=float)
            self.env.sim.data.mocap_quat[mocap_id] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.env.sim.forward()
            return
        joint_names = self.dynamic_carrier["joint_names"]
        for axis, value in zip(("x", "y", "z"), xyz):
            self.env.sim.data.set_joint_qpos(joint_names[axis], float(value))
        self.env.sim.forward()

    def _get_dynamic_carrier_qpos(self):
        if self.dynamic_carrier is None:
            return None
        if self.dynamic_carrier.get("kinematic_mode") == "surface_velocity":
            return list(self.dynamic_carrier.get("qpos", [0.0, 0.0, 0.0]))
        if self.dynamic_carrier.get("kinematic_mode") == "mocap":
            body_name = self.dynamic_carrier["body_name"]
            body_id = self.env.sim.model.body_name2id(body_name)
            mocap_id = int(self.env.sim.model.body_mocapid[body_id])
            return self.env.sim.data.mocap_pos[mocap_id].copy().tolist()
        joint_names = self.dynamic_carrier["joint_names"]
        return [
            float(self.env.sim.data.get_joint_qpos(joint_names[axis]))
            for axis in ("x", "y", "z")
        ]

    def _reset_dynamic_carrier_pose(self):
        if not self._carrier_enabled() or self.dynamic_controller is None:
            return None
        carrier_cfg = self._carrier_config()
        target_pose = self.dynamic_controller.get_body_pose()
        target_pos = np.asarray(target_pose["pos"], dtype=float)
        offset_xyz = np.asarray(carrier_cfg.get("initial_offset_xyz", [0.0, 0.0, -0.075]), dtype=float)
        carrier_pos = target_pos + offset_xyz
        if "initial_z" in carrier_cfg:
            carrier_pos[2] = float(carrier_cfg["initial_z"])
        self._set_dynamic_carrier_qpos(carrier_pos)
        settle_steps = int(carrier_cfg.get("settle_steps", 0))
        for _ in range(max(0, settle_steps)):
            self.env.sim.step()
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        return self._get_dynamic_carrier_qpos()

    def _reset_dynamic_physical_platform_poses(self):
        if not self.dynamic_multi_target_states:
            return []
        events = []
        for state in self.dynamic_multi_target_states:
            name = state["name"]
            platform = state["platform"]
            half_z = float(platform.get("size", [0.06, 0.06, 0.00025])[2])
            surface_epsilon = float(platform.get("surface_epsilon_m", 0.001))
            xy = self._target_platform_xy_center(name)
            bottom_z = self._target_bottom_z(name)
            base_pos = np.asarray([float(xy[0]), float(xy[1]), float(bottom_z - half_z + surface_epsilon)], dtype=float)
            state["platform_base_pos"] = base_pos
            state["elapsed"] = 0.0
            state["last_offset"] = np.zeros(3, dtype=float)
            state["last_velocity"] = np.zeros(3, dtype=float)
            state["cumulative_shift"] = np.zeros(3, dtype=float)
            self._set_dynamic_platform_state(platform, base_pos, np.zeros(3, dtype=float))
            events.append({
                "reason": "physical_platform_reset",
                "mode": "physical_conveyor_patch",
                "step_idx": self.dynamic_step_idx,
                "target_object_name": name,
                "carrier_body_name": platform["body_name"],
                "carrier_geom_name": platform["geom_name"],
                "carrier_qpos": [float(x) for x in base_pos],
            })
        settle_steps = int(self._carrier_config().get("platform_settle_steps", 0))
        for _ in range(max(0, settle_steps)):
            self.env.sim.step()
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        return events

    def _apply_multi_target_dynamic_motion(self, info):
        if not self.dynamic_multi_target_states:
            return
        self._advance_multi_target_platforms_for_dt(
            float(self.dynamic_config.get("motion", {}).get("dt_seconds", self.dynamic_control_dt_seconds)),
            info,
            reason_prefix="dynamic_motion",
            respect_start_end=True,
        )

    def _advance_multi_target_platforms_for_dt(self, dt_seconds, info=None, *, reason_prefix, respect_start_end=False):
        events = []
        for state in self.dynamic_multi_target_states:
            motion_cfg = state.get("motion", {}) or {}
            if respect_start_end:
                start_step = int(motion_cfg.get("start_step", self.dynamic_config.get("motion", {}).get("start_step", 0)))
                end_step = int(motion_cfg.get("end_step", self.dynamic_config.get("motion", {}).get("end_step", 10**12)))
                if self.dynamic_step_idx < start_step or self.dynamic_step_idx > end_step:
                    continue
            old_offset = np.asarray(state.get("last_offset", np.zeros(3)), dtype=float)
            new_elapsed = float(state.get("elapsed", 0.0)) + dt_seconds
            new_offset = self._motion_waypoint_offset(motion_cfg, new_elapsed)
            delta = np.asarray(new_offset, dtype=float) - old_offset
            base_pos = state.get("platform_base_pos")
            if base_pos is None:
                continue
            platform_pos = np.asarray(base_pos, dtype=float) + new_offset
            velocity = delta / max(dt_seconds, 1e-9)
            velocity[2] = 0.0
            self._set_dynamic_platform_state(state["platform"], platform_pos, velocity)
            state["elapsed"] = new_elapsed
            state["last_offset"] = new_offset
            state["last_velocity"] = velocity
            state["cumulative_shift"] = np.asarray(state.get("cumulative_shift", np.zeros(3)), dtype=float) + delta
            event = {
                "reason": f"{reason_prefix}:{motion_cfg.get('type', motion_cfg.get('trajectory', 'waypoints'))}",
                "mode": "physical_conveyor_patch",
                "step_idx": self.dynamic_step_idx,
                "target_object_name": state["name"],
                "target_joint_name": state["joint"],
                "carrier_body_name": state["platform"]["body_name"],
                "carrier_geom_name": state["platform"]["geom_name"],
                "requested_carrier_delta_xyz": [float(x) for x in delta],
                "delta_xyz": [0.0, 0.0, 0.0],
                "carrier_qpos": [float(x) for x in platform_pos],
                "cumulative_shift": [float(x) for x in state["cumulative_shift"]],
            }
            events.append(event)
            if self.dynamic_logger is not None and self.dynamic_episode_active:
                self._log_dynamic_event(event)
        if info is not None:
            for event in events:
                self._append_dynamic_info_event(info, event)
            if events:
                info["dynamic_multi_target_events"] = events
        return events

    def _advance_dynamic_carrier(self, delta_xyz, *, reason, step_idx=None, record_event=True, step_physics=False, delta_seconds=None):
        if self.dynamic_carrier is not None and self.dynamic_carrier.get("kinematic_mode") == "surface_velocity":
            return self._advance_surface_velocity_carrier(
                delta_xyz,
                reason=reason,
                step_idx=step_idx,
                record_event=record_event,
                step_physics=step_physics,
                delta_seconds=delta_seconds,
            )
        before_pose = self.dynamic_controller.get_body_pose()
        before_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        before_carrier_qpos = self._get_dynamic_carrier_qpos()
        if before_carrier_qpos is None:
            raise RuntimeError("Dynamic carrier is not installed")

        delta = np.asarray(delta_xyz, dtype=float)
        after_carrier_qpos = (np.asarray(before_carrier_qpos, dtype=float) + delta).tolist()
        self._set_dynamic_carrier_qpos(after_carrier_qpos)

        if step_physics:
            self.env.sim.step()
            self.env.sim.forward()
            self.env._post_process()
            self.env._update_observables(force=True)

        after_pose = self.dynamic_controller.get_body_pose()
        after_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        self.dynamic_controller.cumulative_shift = [
            self.dynamic_controller.cumulative_shift[i] + float(delta[i]) for i in range(3)
        ]
        event = {
            "reason": reason,
            "mode": "invisible_carrier",
            "step_idx": step_idx,
            "target_object_name": self.dynamic_controller.target.object_name,
            "target_joint_name": self.dynamic_controller.target.joint_name,
            "carrier_body_name": self.dynamic_carrier["body_name"],
            "carrier_geom_name": self.dynamic_carrier["geom_name"],
            "delta_xyz": [float(x) for x in delta],
            "before_pose": before_pose,
            "after_pose": after_pose,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": after_qpos.tolist(),
            "before_carrier_qpos": [float(x) for x in before_carrier_qpos],
            "after_carrier_qpos": [float(x) for x in after_carrier_qpos],
            "cumulative_shift": list(self.dynamic_controller.cumulative_shift),
        }
        if record_event:
            self.dynamic_controller.applied_events.append(event)
        return event

    def _advance_surface_velocity_carrier(self, delta_xyz, *, reason, step_idx=None, record_event=True, step_physics=False, delta_seconds=None):
        carrier_cfg = self._carrier_config()
        before_pose = self.dynamic_controller.get_body_pose()
        before_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        before_carrier_qpos = self._get_dynamic_carrier_qpos()

        target_z = float(before_pose["pos"][2])
        active, carrier_state = self._carrier_contact_state(target_z)
        requested_delta = np.asarray(delta_xyz, dtype=float)
        delta = requested_delta if active else np.zeros(3, dtype=float)
        velocity_event = (
            self._apply_surface_velocity_to_target(
                delta,
                self.dynamic_control_dt_seconds if delta_seconds is None else delta_seconds,
            )
            if active
            else {}
        )
        self.dynamic_surface_velocity_step_mps = (
            velocity_event.get("surface_velocity_mps") if active else None
        )

        if step_physics:
            self.env.sim.step()
            self.env.sim.forward()
            self.env._post_process()
            self.env._update_observables(force=True)

        after_pose = self.dynamic_controller.get_body_pose()
        after_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        self.dynamic_controller.cumulative_shift = [
            self.dynamic_controller.cumulative_shift[i] + float(delta[i])
            for i in range(3)
        ]
        after_carrier_qpos = (np.asarray(before_carrier_qpos, dtype=float) + requested_delta).tolist()
        self._set_dynamic_carrier_qpos(after_carrier_qpos)
        event = {
            "reason": reason,
            "mode": "surface_velocity_carrier",
            "step_idx": step_idx,
            "target_object_name": self.dynamic_controller.target.object_name,
            "target_joint_name": self.dynamic_controller.target.joint_name,
            "carrier_body_name": self.dynamic_carrier["body_name"],
            "carrier_geom_name": self.dynamic_carrier["geom_name"],
            **carrier_state,
            **velocity_event,
            "requested_delta_xyz": [float(x) for x in requested_delta],
            "delta_xyz": [float(x) for x in delta],
            "before_pose": before_pose,
            "after_pose": after_pose,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": after_qpos.tolist(),
            "before_carrier_qpos": [float(x) for x in before_carrier_qpos],
            "after_carrier_qpos": [float(x) for x in after_carrier_qpos],
            "cumulative_shift": list(self.dynamic_controller.cumulative_shift),
        }
        if record_event:
            self.dynamic_controller.applied_events.append(event)
        return event

    def _advance_dynamic_target(self, delta_xyz, *, reason, step_idx=None, record_event=True, step_physics=False, delta_seconds=None):
        if self._use_carrier_for_target_motion():
            return self._advance_dynamic_carrier(
                delta_xyz,
                reason=reason,
                step_idx=step_idx,
                record_event=record_event,
                step_physics=step_physics,
                delta_seconds=delta_seconds,
            )
        event = self.dynamic_controller.shift_target(
            delta_xyz,
            reason=reason,
            step_idx=step_idx,
            record_event=record_event,
        )
        if step_physics:
            self.env.sim.step()
            self.env.sim.forward()
            self.env._post_process()
            self.env._update_observables(force=True)
        return event

    def _apply_dynamic_motion(self, info):
        motion_cfg = self.dynamic_config.get("motion", {})
        motion_type = str(motion_cfg.get("type", "conveyor_linear"))
        start_step = int(motion_cfg.get("start_step", 0))
        end_step = int(motion_cfg.get("end_step", 10**12))
        if self.dynamic_step_idx < start_step or self.dynamic_step_idx > end_step:
            return

        bounded_trajectory = False
        dt_seconds = float(motion_cfg.get("dt_seconds", self.dynamic_control_dt_seconds))
        if motion_type.lower().startswith(("body_yaw_", "body_translation_")):
            event = self._advance_dynamic_fixture_body(
                motion_cfg,
                dt_seconds,
                reason=f"dynamic_motion:{motion_type}",
            )
            self._append_dynamic_info_event(info, event)
            self._log_dynamic_event(event)
            return
        if motion_type == "instant_shift":
            trigger_step = int(motion_cfg.get("trigger_step", start_step))
            already_shifted = any(
                event.get("reason") == "dynamic_motion:instant_shift"
                for event in self.dynamic_controller.applied_events
            )
            if self.dynamic_step_idx != trigger_step or already_shifted:
                return
            delta_xyz = motion_cfg.get("delta_xyz", [0.0, 0.0, 0.0])
        elif motion_type == "conveyor_linear":
            alpha_mps = float(
                motion_cfg.get(
                    "alpha_mps",
                    self._motion_level_alpha_mps(motion_cfg.get("level", "L1")),
                )
            )
            dt_seconds = float(motion_cfg.get("dt_seconds", self.dynamic_control_dt_seconds))
            direction = self._motion_direction_xyz(motion_cfg)
            delta_xyz = (direction * alpha_mps * dt_seconds).tolist()
        elif self._is_pingpong_motion_type(motion_type):
            delta_xyz = self._advance_pingpong_motion_delta(motion_cfg, dt_seconds).tolist()
            bounded_trajectory = True
        elif self._is_bounded_motion_type(motion_type):
            delta_xyz = self._advance_parametric_motion_delta(motion_cfg, dt_seconds).tolist()
            bounded_trajectory = True
        else:
            raise ValueError(f"Unsupported dynamic_motion type: {motion_type}")

        if not bounded_trajectory:
            max_shift = float(motion_cfg.get("max_cumulative_shift_m", 1e9))
            proposed_shift = sum(
                (self.dynamic_controller.cumulative_shift[i] + float(delta_xyz[i])) ** 2
                for i in range(3)
            ) ** 0.5
            if proposed_shift > max_shift:
                info["dynamic_perturbation_skipped"] = {
                    "reason": "max_cumulative_shift_reached",
                    "proposed_shift": proposed_shift,
                    "max_shift": max_shift,
                }
                return

        event = self._advance_dynamic_target(
            delta_xyz,
            reason=f"dynamic_motion:{motion_type}",
            step_idx=self.dynamic_step_idx,
            delta_seconds=float(motion_cfg.get("dt_seconds", self.dynamic_control_dt_seconds)),
        )
        self._append_dynamic_info_event(info, event)
        self._log_dynamic_event(event)

    def _append_dynamic_info_event(self, info, event):
        info.setdefault("dynamic_perturbation_events", []).append(event)
        info["dynamic_perturbation_event"] = event

    def _latency_enabled(self):
        if self.dynamic_config.get("perturb_type", "none") == "latency_drift":
            return True
        return bool(self.dynamic_config.get("latency", {}).get("enabled", False))

    def _get_latency_config(self):
        latency_cfg = dict(self.dynamic_config.get("latency", {}) or {})
        legacy_cfg = self.dynamic_config.get("latency_drift", {}) or {}
        for key, value in legacy_cfg.items():
            latency_cfg.setdefault(key, value)
        if self.dynamic_config.get("perturb_type", "none") == "latency_drift":
            latency_cfg.setdefault("enabled", True)
        return latency_cfg

    def _obs_to_dynamic_video_frame(self, obs):
        frame = {}
        if "agentview_image" in obs:
            frame["image"] = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        if "robot0_eye_in_hand_image" in obs:
            frame["wrist_image"] = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
        return frame if frame else None

    def _stitch_dynamic_video_frame(self, frame):
        images = []
        for key in ("image", "wrist_image"):
            if key not in frame:
                continue
            pil_img = Image.fromarray(np.asarray(frame[key]).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), key, fill=(255, 255, 255))
            images.append(np.asarray(pil_img))
        if not images:
            return None
        return np.concatenate(images, axis=1)

    def _save_latency_video_frame(self, frame, *, sample_idx):
        if self.dynamic_logger is None:
            return None
        stitched = self._stitch_dynamic_video_frame(frame)
        if stitched is None:
            return None
        episode_key = DynamicRunLogger._episode_key(
            self.dynamic_suite_name,
            self.dynamic_task_id,
            self.dynamic_episode_index,
        )
        frame_dir = (
            self.dynamic_logger.output_path.parent
            / f"{self.dynamic_logger.output_path.stem}_latency_frames"
            / episode_key
            / f"step_{int(self.dynamic_step_idx):06d}"
        )
        frame_dir.mkdir(parents=True, exist_ok=True)
        latency_cfg = self._get_latency_config()
        frame_format = str(latency_cfg.get("frame_format", "png")).lower().lstrip(".")
        if frame_format in {"jpeg", "jpg"}:
            suffix = "jpg"
            quality = int(latency_cfg.get("frame_quality", 90))
            frame_path = frame_dir / f"latency_{int(sample_idx):04d}.{suffix}"
            imageio.imwrite(frame_path, stitched, quality=quality)
        else:
            suffix = "png"
            frame_path = frame_dir / f"latency_{int(sample_idx):04d}.{suffix}"
            imageio.imwrite(frame_path, stitched)
        return str(frame_path)

    def _latency_video_segment_path(self):
        if self.dynamic_logger is None:
            return None
        episode_key = DynamicRunLogger._episode_key(
            self.dynamic_suite_name,
            self.dynamic_task_id,
            self.dynamic_episode_index,
        )
        video_dir = (
            self.dynamic_logger.output_path.parent
            / f"{self.dynamic_logger.output_path.stem}_latency_video_segments"
            / episode_key
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir / f"step_{int(self.dynamic_step_idx):06d}.mp4"

    def _latency_state_segment_path(self):
        if self.dynamic_logger is None:
            return None
        episode_key = DynamicRunLogger._episode_key(
            self.dynamic_suite_name,
            self.dynamic_task_id,
            self.dynamic_episode_index,
        )
        state_dir = (
            self.dynamic_logger.output_path.parent
            / f"{self.dynamic_logger.output_path.stem}_latency_state_segments"
            / episode_key
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / f"step_{int(self.dynamic_step_idx):06d}.npz"

    def _latency_model_snapshot(self):
        model = self.env.sim.model
        snapshot = {}
        for name in (
            "body_pos",
            "body_quat",
            "geom_pos",
            "geom_quat",
            "site_pos",
            "site_quat",
            "cam_pos",
            "cam_quat",
        ):
            if hasattr(model, name):
                snapshot[f"model_{name}"] = np.asarray(getattr(model, name)).copy()
        return snapshot

    def _get_latency_substep_count(self, latency_seconds, latency_cfg):
        substep_dt = float(latency_cfg.get("substep_dt", self.dynamic_control_dt_seconds))
        if substep_dt <= 0.0:
            raise ValueError(f"latency.substep_dt must be positive, got {substep_dt}")
        return max(1, int(np.ceil(float(latency_seconds) / substep_dt)))

    def _get_latency_video_dt(self, latency_cfg):
        if "video_dt" in latency_cfg:
            video_dt = float(latency_cfg["video_dt"])
        else:
            video_fps = float(latency_cfg.get("video_fps", 24.0))
            if video_fps <= 0.0:
                raise ValueError(f"latency.video_fps must be positive, got {video_fps}")
            video_dt = 1.0 / video_fps
        if video_dt <= 0.0:
            raise ValueError(f"latency.video_dt must be positive, got {video_dt}")
        return video_dt

    def _latency_freeze_joint_names(self, latency_cfg):
        configured = latency_cfg.get("freeze_joint_names", None)
        if configured is not None:
            return [str(name) for name in configured]
        prefixes = tuple(latency_cfg.get("freeze_joint_prefixes", ["robot0_", "gripper0_"]))
        return [
            self.env.sim.model.joint_id2name(i)
            for i in range(self.env.sim.model.njnt)
            if self.env.sim.model.joint_id2name(i).startswith(prefixes)
        ]

    def _capture_latency_robot_hold_state(self, latency_cfg):
        if not bool(latency_cfg.get("freeze_robot_during_latency", True)):
            return None
        state = []
        for joint_name in self._latency_freeze_joint_names(latency_cfg):
            try:
                qpos = np.asarray(self.env.sim.data.get_joint_qpos(joint_name)).copy()
                qvel = np.asarray(self.env.sim.data.get_joint_qvel(joint_name)).copy()
            except Exception:
                continue
            state.append((joint_name, qpos, qvel))
        return state

    def _restore_latency_robot_hold_state(self, hold_state, *, zero_velocity=True):
        if not hold_state:
            return
        for joint_name, qpos, qvel in hold_state:
            self.env.sim.data.set_joint_qpos(joint_name, qpos)
            if zero_velocity:
                self.env.sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
            else:
                self.env.sim.data.set_joint_qvel(joint_name, qvel)

    def _advance_dynamic_target_for_latency_substep(self, delta_xyz, delta_seconds):
        delta_requested = np.asarray(delta_xyz, dtype=float)
        if self.dynamic_controller is None:
            return {"delta_xyz": delta_requested.tolist()}

        latency_cfg = self._get_latency_config()
        latency_motion_type = self._resolved_motion_type(latency_cfg)
        if latency_motion_type.startswith(("body_yaw_", "body_translation_")):
            return self._advance_dynamic_fixture_body(
                latency_cfg,
                delta_seconds,
                reason="latency_drift_continuous_substep",
            )

        before_pose = self.dynamic_controller.get_body_pose()
        target_z = float(before_pose["pos"][2])
        delta_applied = delta_requested.copy()
        event = {
            "reason": "latency_drift_continuous_substep",
            "step_idx": self.dynamic_step_idx,
            "target_object_name": self.dynamic_controller.target.object_name,
            "target_joint_name": self.dynamic_controller.target.joint_name,
            "delta_xyz": [float(x) for x in delta_applied],
        }

        if self._use_carrier_for_target_motion():
            active, carrier_state = self._carrier_contact_state(target_z)
            event.update({
                "mode": "surface_velocity_carrier" if self.dynamic_carrier and self.dynamic_carrier.get("kinematic_mode") == "surface_velocity" else "invisible_carrier",
                **carrier_state,
            })

            before_carrier_qpos = self._get_dynamic_carrier_qpos()
            if before_carrier_qpos is not None:
                after_carrier_qpos = (
                    np.asarray(before_carrier_qpos, dtype=float) + delta_requested
                ).tolist()
                self._set_dynamic_carrier_qpos(after_carrier_qpos)
                event["before_carrier_qpos"] = [float(x) for x in before_carrier_qpos]
                event["after_carrier_qpos"] = [float(x) for x in after_carrier_qpos]

            if self.dynamic_carrier and self.dynamic_carrier.get("kinematic_mode") != "surface_velocity":
                # A real conveyor/platform moves independently. The target object is
                # not teleported; MuJoCo contact/friction decides whether it follows.
                self.dynamic_controller.cumulative_shift = [
                    self.dynamic_controller.cumulative_shift[i] + float(delta_requested[i])
                    for i in range(3)
                ]
                event["requested_carrier_delta_xyz"] = [float(x) for x in delta_requested]
                event["delta_xyz"] = [0.0, 0.0, 0.0]
                event["cumulative_shift"] = list(self.dynamic_controller.cumulative_shift)
                return event

            if not active:
                delta_applied = np.zeros(3, dtype=float)
            else:
                event.update(
                    self._apply_surface_velocity_to_target(delta_applied, delta_seconds)
                )

            self.dynamic_controller.cumulative_shift = [
                self.dynamic_controller.cumulative_shift[i] + float(delta_applied[i])
                for i in range(3)
            ]
            event["requested_delta_xyz"] = [float(x) for x in delta_requested]
            event["delta_xyz"] = [float(x) for x in delta_applied]
            event["cumulative_shift"] = list(self.dynamic_controller.cumulative_shift)
            self.env.sim.forward()
            return event

        qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        qpos[:3] = qpos[:3] + delta_applied
        self.env.sim.data.set_joint_qpos(self.dynamic_controller.target.joint_name, qpos)

        self.dynamic_controller.cumulative_shift = [
            self.dynamic_controller.cumulative_shift[i] + float(delta_applied[i])
            for i in range(3)
        ]
        event["delta_xyz"] = [float(x) for x in delta_applied]
        event["cumulative_shift"] = list(self.dynamic_controller.cumulative_shift)
        self.env.sim.forward()
        return event

    def _advance_latency_physics(self, delta_xyz, latency_cfg, info):
        hold_state = self._capture_latency_robot_hold_state(latency_cfg)
        self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)

        sim_dt = float(getattr(self.env.sim.model.opt, "timestep", 0.0) or 0.0)
        substep_dt = float(latency_cfg.get("substep_dt", self.dynamic_control_dt_seconds))
        sub_event = self._advance_dynamic_target_for_latency_substep(delta_xyz, substep_dt)
        surface_velocity_mps = sub_event.get("surface_velocity_mps")
        sim_steps = max(1, int(round(substep_dt / sim_dt))) if sim_dt > 0.0 else 1
        for _ in range(sim_steps):
            self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)
            if surface_velocity_mps is not None:
                self._apply_surface_velocity_mps_to_target(surface_velocity_mps)
            self.env.sim.step()
            self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)
        self.env.sim.forward()

        return sub_event

    def _apply_latency_drift(self, info):
        if not self._latency_enabled():
            return
        if self.dynamic_multi_target_states:
            return self._apply_multi_target_latency_drift(info)

        latency_cfg = self._get_latency_config()
        latency_seconds = self._consume_inference_latency()
        latency_measurement = dict(getattr(self, "dynamic_last_latency_measurement", {}) or {})
        info["dynamic_latency_seconds"] = latency_seconds
        info["dynamic_latency_measurement"] = latency_measurement
        if latency_seconds <= 0.0:
            return
        self._log_dynamic_latency(latency_seconds)
        model_input_event = self._save_model_input_observation_frame(
            latency_seconds=latency_seconds
        )
        if model_input_event is not None:
            info["dynamic_model_input_observation"] = model_input_event

        latency_motion_type = self._resolved_motion_type(latency_cfg, "linear")
        latency_uses_bounded_trajectory = self._is_bounded_motion_type(latency_motion_type)
        if latency_uses_bounded_trajectory:
            # For bounded trajectories, do not collapse a long latency interval into
            # a net displacement. A 19s ping-pong segment may have a tiny net delta
            # after many reflections, but the real object should still move at the
            # configured constant speed throughout those 19 seconds.
            delta_xyz = [0.0, 0.0, 0.0]
        else:
            velocity = latency_cfg.get("velocity_xyz_mps", [0.0, 0.0, 0.0])
            delta_xyz = [
                float(velocity[0]) * latency_seconds,
                float(velocity[1]) * latency_seconds,
                float(velocity[2]) * latency_seconds,
            ]
        max_shift = float(latency_cfg.get("max_cumulative_shift_m", 1e9))
        current_shift = sum(x * x for x in self.dynamic_controller.cumulative_shift) ** 0.5
        proposed_shift = sum(
            (self.dynamic_controller.cumulative_shift[i] + delta_xyz[i]) ** 2
            for i in range(3)
        ) ** 0.5

        latency_clipped_by_max_shift = False
        original_delta_xyz = list(delta_xyz)
        if (not latency_uses_bounded_trajectory) and proposed_shift > max_shift:
            remaining_shift = max(0.0, max_shift - current_shift)
            delta_norm = sum(float(x) * float(x) for x in delta_xyz) ** 0.5
            if remaining_shift <= 0.0 or delta_norm <= 0.0:
                # Latency is a control-chain delay even when the dynamic target has
                # reached its motion bound. Keep simulating the waiting period so
                # videos and logs still show the robot frozen during inference.
                delta_xyz = [0.0, 0.0, 0.0]
                latency_clipped_by_max_shift = True
                info["dynamic_perturbation_clipped"] = {
                    "reason": "max_cumulative_shift_reached",
                    "current_shift": current_shift,
                    "proposed_shift": proposed_shift,
                    "max_shift": max_shift,
                    "original_delta_xyz": original_delta_xyz,
                    "clipped_delta_xyz": list(delta_xyz),
                }
            else:
                scale = remaining_shift / delta_norm
                delta_xyz = [float(x) * scale for x in delta_xyz]
                latency_clipped_by_max_shift = True
                info["dynamic_perturbation_clipped"] = {
                    "reason": "max_cumulative_shift_reached",
                    "current_shift": current_shift,
                    "proposed_shift": proposed_shift,
                    "max_shift": max_shift,
                    "original_delta_xyz": original_delta_xyz,
                    "clipped_delta_xyz": list(delta_xyz),
                }

        mode = str(latency_cfg.get("mode", "discrete_shift"))
        if mode != "continuous_physics":
            event = self._advance_dynamic_target(
                delta_xyz,
                reason="latency_drift",
                step_idx=self.dynamic_step_idx,
            )
            self._append_dynamic_info_event(info, event)
            self._log_dynamic_event(event)
            return

        before_pose = self.dynamic_controller.get_body_pose()
        before_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        num_substeps = self._get_latency_substep_count(latency_seconds, latency_cfg)
        effective_substep_dt = float(latency_seconds) / num_substeps
        delta_per_substep = [float(x) / num_substeps for x in delta_xyz]
        total_latency_delta = np.zeros(3, dtype=float)
        save_video_frames = bool(latency_cfg.get("save_video_frames", False))
        return_video_frames = bool(latency_cfg.get("return_video_frames", False))
        save_video_frames_to_disk = bool(latency_cfg.get("save_video_frames_to_disk", save_video_frames))
        frame_storage = str(latency_cfg.get("frame_storage", "state")).lower()
        store_state_segments = frame_storage in {"state", "states", "npz", "state_npz"}
        video_dt = self._get_latency_video_dt(latency_cfg) if save_video_frames else None
        next_video_time = float(video_dt) if save_video_frames else None
        elapsed_time = 0.0
        latency_frame_paths = []
        latency_frame_times = []
        latency_video_path = None
        latency_video_frame_times = []
        latency_video_frame_count = 0
        latency_video_writer = None
        latency_state_path = None
        latency_state_frame_times = []
        latency_state_qpos = []
        latency_state_qvel = []
        latency_state_flat = []
        latency_model_snapshot = self._latency_model_snapshot()
        sample_idx = 0
        try:
            for substep_idx in range(num_substeps):
                if latency_motion_type.startswith(("body_yaw_", "body_translation_")):
                    substep_delta = [0.0, 0.0, 0.0]
                elif self._is_pingpong_motion_type(latency_motion_type):
                    substep_delta = self._advance_pingpong_motion_delta(
                        latency_cfg,
                        effective_substep_dt,
                    ).tolist()
                elif latency_uses_bounded_trajectory:
                    substep_delta = self._advance_parametric_motion_delta(
                        {**latency_cfg, "type": latency_motion_type},
                        effective_substep_dt,
                    ).tolist()
                else:
                    substep_delta = delta_per_substep
                total_latency_delta += np.asarray(substep_delta, dtype=float)
                self._advance_latency_physics(substep_delta, latency_cfg, info)
                elapsed_time = min(float(latency_seconds), (substep_idx + 1) * effective_substep_dt)
                if save_video_frames and elapsed_time + 1e-9 >= next_video_time:
                    if save_video_frames_to_disk and store_state_segments and not return_video_frames:
                        latency_state_qpos.append(self.env.sim.data.qpos.copy())
                        latency_state_qvel.append(self.env.sim.data.qvel.copy())
                        latency_state_flat.append(self.env.sim.get_state().flatten().copy())
                        latency_state_frame_times.append(float(elapsed_time))
                    else:
                        self.env._post_process()
                        self.env._update_observables(force=True)
                        obs = self.env._get_observations()
                        frame = self._obs_to_dynamic_video_frame(obs)
                        if frame is not None:
                            if return_video_frames:
                                info.setdefault("dynamic_latency_frames", []).append(frame)
                            if save_video_frames_to_disk:
                                if frame_storage in {"video", "mp4", "segment", "segments"}:
                                    stitched = self._stitch_dynamic_video_frame(frame)
                                    if stitched is not None:
                                        if latency_video_writer is None:
                                            segment_path = self._latency_video_segment_path()
                                            if segment_path is not None:
                                                latency_video_path = str(segment_path)
                                                latency_video_writer = imageio.get_writer(
                                                    segment_path,
                                                    fps=1.0 / float(video_dt),
                                                    macro_block_size=None,
                                                )
                                        if latency_video_writer is not None:
                                            latency_video_writer.append_data(stitched)
                                            latency_video_frame_count += 1
                                            latency_video_frame_times.append(float(elapsed_time))
                                else:
                                    frame_path = self._save_latency_video_frame(frame, sample_idx=sample_idx)
                                    if frame_path is not None:
                                        latency_frame_paths.append(frame_path)
                                        latency_frame_times.append(float(elapsed_time))
                    sample_idx += 1
                    next_video_time += video_dt
        finally:
            if latency_video_writer is not None:
                latency_video_writer.close()

        if latency_state_qpos:
            state_path = self._latency_state_segment_path()
            if state_path is not None:
                np.savez(
                    state_path,
                    qpos=np.asarray(latency_state_qpos),
                    qvel=np.asarray(latency_state_qvel),
                    states=np.asarray(latency_state_flat),
                    times=np.asarray(latency_state_frame_times, dtype=float),
                    **latency_model_snapshot,
                )
                latency_state_path = str(state_path)


        if latency_uses_bounded_trajectory:
            delta_xyz = total_latency_delta.tolist()
            original_delta_xyz = list(delta_xyz)

        after_pose = self.dynamic_controller.get_body_pose()
        after_qpos = self.env.sim.data.get_joint_qpos(
            self.dynamic_controller.target.joint_name
        ).copy()
        event = {
            "reason": "latency_drift_continuous",
            "step_idx": self.dynamic_step_idx,
            "target_object_name": self.dynamic_controller.target.object_name,
            "target_joint_name": self.dynamic_controller.target.joint_name,
            "delta_xyz": [float(x) for x in delta_xyz],
            "original_delta_xyz": [float(x) for x in original_delta_xyz],
            "clipped_by_max_cumulative_shift": bool(latency_clipped_by_max_shift),
            "before_pose": before_pose,
            "after_pose": after_pose,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": after_qpos.tolist(),
            "cumulative_shift": list(self.dynamic_controller.cumulative_shift),
            "latency_seconds": float(latency_seconds),
            "latency_measurement": latency_measurement,
            "latency_source": latency_measurement.get("source"),
            "notified_latency_seconds": latency_measurement.get("notified_latency_seconds"),
            "freeze_robot_during_latency": bool(latency_cfg.get("freeze_robot_during_latency", True)),
            "freeze_joint_names": self._latency_freeze_joint_names(latency_cfg)
            if bool(latency_cfg.get("freeze_robot_during_latency", True))
            else [],
            "num_substeps": int(num_substeps),
            "substep_dt": float(latency_cfg.get("substep_dt", self.dynamic_control_dt_seconds)),
            "video_dt": float(video_dt) if video_dt is not None else None,
            "video_fps": (1.0 / float(video_dt)) if video_dt is not None else None,
            "expected_video_frames": int(np.floor(float(latency_seconds) / float(video_dt) + 1e-9))
            if video_dt is not None
            else 0,
            "saved_video_frames": len(latency_frame_paths) + int(latency_video_frame_count) + len(latency_state_frame_times),
            "latency_frame_paths": latency_frame_paths,
            "latency_frame_times": latency_frame_times,
            "latency_video_path": latency_video_path,
            "latency_video_frame_count": int(latency_video_frame_count),
            "latency_video_frame_times": latency_video_frame_times,
            "latency_state_path": latency_state_path,
            "latency_state_frame_count": len(latency_state_frame_times),
            "latency_state_frame_times": latency_state_frame_times,
            "insert_before_control_step": int(self.dynamic_step_idx),
        }
        self.dynamic_controller.applied_events.append(event)
        self._append_dynamic_info_event(info, event)
        self._log_dynamic_event(event)

    def _apply_multi_target_latency_drift(self, info):
        latency_cfg = self._get_latency_config()
        latency_seconds = self._consume_inference_latency()
        latency_measurement = dict(getattr(self, "dynamic_last_latency_measurement", {}) or {})
        info["dynamic_latency_seconds"] = latency_seconds
        info["dynamic_latency_measurement"] = latency_measurement
        if latency_seconds <= 0.0:
            return

        self._log_dynamic_latency(latency_seconds)
        model_input_event = self._save_model_input_observation_frame(
            latency_seconds=latency_seconds
        )
        if model_input_event is not None:
            info["dynamic_model_input_observation"] = model_input_event

        hold_state = self._capture_latency_robot_hold_state(latency_cfg)
        self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)
        num_substeps = self._get_latency_substep_count(latency_seconds, latency_cfg)
        effective_substep_dt = float(latency_seconds) / max(1, num_substeps)
        sim_dt = float(getattr(self.env.sim.model.opt, "timestep", 0.0) or 0.0)
        sim_steps = max(1, int(round(effective_substep_dt / sim_dt))) if sim_dt > 0.0 else 1
        all_events = []
        for _ in range(num_substeps):
            events = self._advance_multi_target_platforms_for_dt(
                effective_substep_dt,
                None,
                reason_prefix="latency_drift_continuous_substep",
                respect_start_end=False,
            )
            all_events.extend(events)
            for _sim_idx in range(sim_steps):
                self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)
                self.env.sim.step()
                self._restore_latency_robot_hold_state(hold_state, zero_velocity=True)
        self.env.sim.forward()

        event = {
            "reason": "latency_drift_continuous",
            "mode": "physical_conveyor_patch_multi_target",
            "step_idx": self.dynamic_step_idx,
            "target_object_names": [state["name"] for state in self.dynamic_multi_target_states],
            "latency_seconds": float(latency_seconds),
            "latency_measurement": latency_measurement,
            "latency_source": latency_measurement.get("source"),
            "notified_latency_seconds": latency_measurement.get("notified_latency_seconds"),
            "freeze_robot_during_latency": bool(latency_cfg.get("freeze_robot_during_latency", True)),
            "freeze_joint_names": self._latency_freeze_joint_names(latency_cfg)
            if bool(latency_cfg.get("freeze_robot_during_latency", True))
            else [],
            "num_substeps": int(num_substeps),
            "substep_dt": float(effective_substep_dt),
            "target_events": all_events,
            "insert_before_control_step": int(self.dynamic_step_idx),
        }
        for sub_event in all_events:
            self._append_dynamic_info_event(info, sub_event)
        self._append_dynamic_info_event(info, event)
        info["dynamic_multi_target_latency_event"] = event
        self._log_dynamic_event(event)

    def step(self, action):
        info = {}
        surface_velocity_mps = None
        if self.dynamic_controller is not None:
            self.dynamic_surface_velocity_step_mps = None
            self.dynamic_surface_velocity_step_hook_count = 0
            perturb_type = self.dynamic_config.get("perturb_type", "none")
            self._apply_latency_drift(info)

            if perturb_type == "dynamic_motion":
                if self.dynamic_multi_target_states:
                    self._apply_multi_target_dynamic_motion(info)
                else:
                    self._apply_dynamic_motion(info)
                    surface_velocity_mps = self.dynamic_surface_velocity_step_mps

        if self.dynamic_multi_target_states:
            obs, reward, done, step_info = self._step_env_with_dynamic_platform_hold(action)
        else:
            obs, reward, done, step_info = self._step_env_with_surface_velocity(
                action, surface_velocity_mps
            )
        info.update(step_info)
        if surface_velocity_mps is not None:
            info["dynamic_surface_velocity_step_hook_count"] = int(
                self.dynamic_surface_velocity_step_hook_count
            )
        self.dynamic_surface_velocity_step_mps = None

        if self.dynamic_controller is not None:
            perturb_type = self.dynamic_config.get("perturb_type", "none")
            if perturb_type == "mid_episode_shift":
                shift_cfg = self.dynamic_config.get("mid_episode_shift", {})
                trigger_step = int(shift_cfg.get("trigger_step", -1))
                already_shifted = any(
                    event.get("reason") == "mid_episode_shift"
                    for event in self.dynamic_controller.applied_events
                )
                if self.dynamic_step_idx == trigger_step and not already_shifted:
                    event = self.dynamic_controller.shift_target(
                        shift_cfg.get("delta_xyz", [0.0, 0.0, 0.0]),
                        reason="mid_episode_shift",
                        step_idx=self.dynamic_step_idx,
                    )
                    obs = self.env._get_observations()
                    self._append_dynamic_info_event(info, event)
                    self._log_dynamic_event(event)

        if done:
            self._finalize_dynamic_episode("success")

        self.dynamic_step_idx += 1
        return obs, reward, done, info



    def notify_inference_latency(self, latency_seconds=None):
        # Eval calls this once per policy inference / replan. Chunk-internal
        # env.step calls should not create latency events. If latency_seconds is
        # None, the env uses the configured fixed latency level for this replan.
        self.pending_inference_latency_notified = True
        self.pending_inference_latency_seconds = (
            None if latency_seconds is None else float(latency_seconds)
        )

    def _latency_level_seconds(self, level, latency_cfg):
        default_levels = {
            "NONE": 0.0,
            "L0": 0.0,
            "LOW": 0.10,
            "L1": 0.10,
            "MEDIUM": 0.25,
            "L2": 0.25,
            "HIGH": 0.50,
            "L3": 0.50,
            "EXTREME": 1.00,
            "L4": 1.00,
        }
        configured_levels = latency_cfg.get("level_seconds", {}) or {}
        levels = {**default_levels, **{str(k).upper(): float(v) for k, v in configured_levels.items()}}
        return float(levels.get(str(level).upper(), levels["MEDIUM"]))

    def _fixed_latency_seconds(self, latency_cfg):
        if "fixed_seconds" in latency_cfg:
            return float(latency_cfg["fixed_seconds"])
        if "fixed_latency_seconds" in latency_cfg:
            return float(latency_cfg["fixed_latency_seconds"])
        return self._latency_level_seconds(latency_cfg.get("level", "MEDIUM"), latency_cfg)

    def _cap_dynamic_latency_seconds(self, latency_seconds, latency_cfg):
        max_latency = latency_cfg.get("max_latency_seconds", None)
        if max_latency is not None:
            return min(float(latency_seconds), float(max_latency))
        return float(latency_seconds)

    def _latency_trigger_threshold_seconds(self, latency_cfg):
        return float(
            latency_cfg.get(
                "measured_latency_trigger_seconds",
                latency_cfg.get("trigger_threshold_seconds", 0.05),
            )
        )

    def _consume_inference_latency(self):
        latency_cfg = self._get_latency_config()

        replan_notified = bool(getattr(self, "pending_inference_latency_notified", False))
        notified_latency = getattr(self, "pending_inference_latency_seconds", None)
        self.pending_inference_latency_notified = False
        self.pending_inference_latency_seconds = None

        use_measured_latency = bool(
            latency_cfg.get(
                "use_measured_latency",
                str(latency_cfg.get("source", "")).lower() in {"measured", "notified"},
            )
        )
        source_preference = str(latency_cfg.get("measured_latency_source", "notified")).lower()
        if source_preference not in {"notify", "notified", "eval", "hook"}:
            raise ValueError(
                "LIBERO-dynamic no longer infers latency from env.step wall-clock gaps. "
                "Use measured_latency_source: notified, or disable measured latency "
                "and rely on fixed latency levels at notified replan boundaries."
            )

        selected_source = None
        selected_latency = 0.0
        if replan_notified and notified_latency is not None and float(notified_latency) >= 0.0:
            selected_source = "notified"
            selected_latency = float(notified_latency)

        self.dynamic_last_latency_measurement = {
            "source": selected_source or ("replan_notified" if replan_notified else "none"),
            "selected_latency_seconds": float(selected_latency),
            "notified_latency_seconds": None if notified_latency is None else float(notified_latency),
            "use_measured_latency": bool(use_measured_latency),
            "measured_latency_source": source_preference,
            "replan_notified": bool(replan_notified),
        }

        if use_measured_latency and selected_source is not None:
            return self._cap_dynamic_latency_seconds(selected_latency, latency_cfg)

        fallback_to_fixed = bool(latency_cfg.get("fallback_to_fixed_latency", True))
        if not replan_notified and not bool(latency_cfg.get("apply_fixed_latency_every_step", False)):
            return 0.0
        if use_measured_latency and not fallback_to_fixed:
            return 0.0
        if not (fallback_to_fixed or bool(latency_cfg.get("apply_fixed_latency_every_step", False))):
            return 0.0

        fixed_latency = self._fixed_latency_seconds(latency_cfg)
        self.dynamic_last_latency_measurement.update(
            {
                "source": "fixed_level",
                "selected_latency_seconds": float(fixed_latency),
                "fixed_latency_seconds": float(fixed_latency),
                "fixed_level": str(latency_cfg.get("level", "MEDIUM")),
                "fallback_to_fixed_latency": bool(fallback_to_fixed),
            }
        )
        return self._cap_dynamic_latency_seconds(fixed_latency, latency_cfg)


    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue
        if self.dynamic_controller is not None:
            # robosuite reset can restore the original task XML. Re-install
            # dynamic carrier/platform bodies after reset so formal eval loops
            # see the same augmented model that was created at construction.
            self._install_dynamic_physical_platforms_if_needed()
            ret = self.env._get_observations()

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        expected_len = 1 + int(self.env.sim.model.nq) + int(self.env.sim.model.nv)
        state = np.asarray(mujoco_state)
        if len(state) == expected_len:
            try:
                self.env.sim.set_state_from_flattened(state)
                return
            except ValueError:
                if self._added_dynamic_joint_count() == 0:
                    raise

        if not self._carrier_enabled() and not self.dynamic_physical_platforms:
            self.env.sim.set_state_from_flattened(state)
            return

        added_joints = self._added_dynamic_joint_count()
        if added_joints == 0:
            self.env.sim.set_state_from_flattened(state)
            return
        old_nq = int(self.env.sim.model.nq) - added_joints
        old_nv = int(self.env.sim.model.nv) - added_joints
        old_expected_len = 1 + old_nq + old_nv
        if len(state) != old_expected_len:
            raise ValueError(
                f"Unexpected flattened state length {len(state)} for carrier-augmented model; "
                f"expected {expected_len} or legacy {old_expected_len}"
            )

        if self._restore_legacy_state_by_joint_name(state):
            return

        self.env.sim.data.time = float(state[0])
        self.env.sim.data.qpos[:] = 0.0
        self.env.sim.data.qvel[:] = 0.0
        self.env.sim.data.qpos[:old_nq] = state[1 : 1 + old_nq]
        self.env.sim.data.qvel[:old_nv] = state[1 + old_nq : 1 + old_nq + old_nv]

    def _added_dynamic_joint_count(self):
        count = 0
        if self._carrier_enabled():
            count += len(self.dynamic_carrier.get("joint_names", {}))
        for platform in getattr(self, "dynamic_physical_platforms", {}).values():
            count += len(platform.get("joint_names", {}))
        return count

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        self._finalize_dynamic_episode("failure_or_timeout")
        self._close_model_input_video_writer()
        self.dynamic_episode_index += 1
        self.dynamic_model_input_video_path = None
        self.dynamic_model_input_frame_count = 0
        self.dynamic_step_idx = 0
        self._reset_dynamic_motion_state()
        self.dynamic_episode_active = True
        self.dynamic_episode_finalized = False
        if self.dynamic_controller is not None:
            self.dynamic_controller.applied_events.clear()
            self.dynamic_controller.cumulative_shift = [0.0, 0.0, 0.0]
        for state in self.dynamic_multi_target_states:
            state["elapsed"] = 0.0
            state["last_offset"] = np.zeros(3, dtype=float)
            state["last_velocity"] = np.zeros(3, dtype=float)
            state["cumulative_shift"] = np.zeros(3, dtype=float)
        if self.dynamic_logger is not None:
            self.dynamic_logger.upsert_episode(
                suite_name=self.dynamic_suite_name,
                task_id=self.dynamic_task_id,
                episode_index=self.dynamic_episode_index,
                planned=self._build_dynamic_plan(),
                status="running",
            )
        self._restore_dynamic_fixture_body_canonical_pose()
        obs = self.regenerate_obs_from_state(init_state)
        if self.dynamic_controller is not None:
            adjustment_events = self._apply_dynamic_initial_scene_adjustments()
            self._start_dynamic_fixture_body_episode()
            if adjustment_events and self.dynamic_logger is not None:
                for event in adjustment_events:
                    self.dynamic_logger.append_event(
                        suite_name=self.dynamic_suite_name,
                        task_id=self.dynamic_task_id,
                        episode_index=self.dynamic_episode_index,
                        event={**event, "step_idx": self.dynamic_step_idx},
                    )
            platform_events = self._reset_dynamic_physical_platform_poses()
            if platform_events and self.dynamic_logger is not None:
                for event in platform_events:
                    self.dynamic_logger.append_event(
                        suite_name=self.dynamic_suite_name,
                        task_id=self.dynamic_task_id,
                        episode_index=self.dynamic_episode_index,
                        event=event,
                    )
            carrier_qpos = self._reset_dynamic_carrier_pose()
            if carrier_qpos is not None and self.dynamic_logger is not None:
                self.dynamic_logger.append_event(
                    suite_name=self.dynamic_suite_name,
                    task_id=self.dynamic_task_id,
                    episode_index=self.dynamic_episode_index,
                    event={
                        "reason": "carrier_reset",
                        "mode": "invisible_carrier",
                        "step_idx": self.dynamic_step_idx,
                        "target_object_name": self.dynamic_controller.target.object_name,
                        "carrier_qpos": carrier_qpos,
                    },
                )
            obs = self.env._get_observations()
        return obs

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self._finalize_dynamic_episode("failure_or_timeout")
        self._close_model_input_video_writer()
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
