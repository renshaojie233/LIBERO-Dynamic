#!/usr/bin/env python3
"""Render no-policy preview videos for reviewed LIBERO-dynamic configs.

The goal is visual QA for the benchmark itself.  This script does not run a
robot policy; it loads each LIBERO task at init state 0, applies the reviewed
initial scene adjustments, moves the configured dynamic targets along their
paths, and saves short MP4 previews plus a manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np
import yaml

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def zero_action(env: OffScreenRenderEnv) -> np.ndarray:
    dim = getattr(env.env, "action_dim", None)
    if dim is None:
        dim = sum(int(getattr(robot, "action_dim", 0)) for robot in getattr(env.env, "robots", []) or [])
    return np.zeros(int(dim), dtype=np.float32)


def robot_geom_ids(env: OffScreenRenderEnv) -> list[int]:
    ids = []
    for geom_id in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom_id2name(geom_id) or ""
        body_name = env.sim.model.body_id2name(env.sim.model.geom_bodyid[geom_id]) or ""
        if geom_name.startswith(("robot0_", "gripper0_")) or body_name.startswith(("robot0_", "gripper0_")):
            ids.append(geom_id)
    return ids


def render_named_camera(
    env: OffScreenRenderEnv,
    camera_name: str,
    camera_size: int,
    *,
    hide_robot: bool = False,
) -> np.ndarray | None:
    camera_names = {env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)}
    if camera_name not in camera_names:
        return None
    hidden_ids = robot_geom_ids(env) if hide_robot else []
    old_rgba = None
    try:
        if hidden_ids:
            old_rgba = env.sim.model.geom_rgba[hidden_ids].copy()
            env.sim.model.geom_rgba[hidden_ids, 3] = 0.0
        return np.asarray(env.sim.render(camera_size, camera_size, camera_name=camera_name))[::-1]
    except Exception:
        return None
    finally:
        if old_rgba is not None:
            env.sim.model.geom_rgba[hidden_ids] = old_rgba


def obs_image(obs: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in obs:
        return None
    return np.ascontiguousarray(np.asarray(obs[key])[::-1, ::-1])


def project_world_to_camera(
    env: OffScreenRenderEnv,
    point_xyz: np.ndarray,
    camera_name: str,
    camera_size: int,
) -> tuple[int, int] | None:
    try:
        camera_id = int(env.sim.model.camera_name2id(camera_name))
        camera_pos = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=float)
        camera_xmat = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        cam_point = camera_xmat.T @ (np.asarray(point_xyz, dtype=float) - camera_pos)
        depth = -float(cam_point[2])
        if depth <= 1e-6:
            return None
        fovy = math.radians(float(env.sim.model.cam_fovy[camera_id]))
        half_h = math.tan(fovy / 2.0) * depth
        half_w = half_h
        if half_h <= 1e-9:
            return None
        x_ndc = float(cam_point[0]) / half_w
        y_ndc = float(cam_point[1]) / half_h
        if abs(x_ndc) > 2.0 or abs(y_ndc) > 2.0:
            return None
        u = int(round((x_ndc + 1.0) * 0.5 * (camera_size - 1)))
        v = int(round((1.0 - y_ndc) * 0.5 * (camera_size - 1)))
        if 0 <= u < camera_size and 0 <= v < camera_size:
            return u, v
    except Exception:
        return None
    return None


def draw_disk(frame: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    x0, y0 = center
    h, w = frame.shape[:2]
    x_min, x_max = max(0, x0 - radius), min(w - 1, x0 + radius)
    y_min, y_max = max(0, y0 - radius), min(h - 1, y0 + radius)
    rr = radius * radius
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if (x - x0) * (x - x0) + (y - y0) * (y - y0) <= rr:
                frame[y, x, :] = color


def draw_line(frame: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int], thickness: int = 2) -> None:
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for idx in range(steps + 1):
        t = idx / steps
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        draw_disk(frame, (x, y), max(1, thickness // 2), color)


def draw_polyline(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    for p0, p1 in zip(points, points[1:]):
        draw_line(frame, p0, p1, color, thickness=thickness)


def overlay_motion_traces(
    env: OffScreenRenderEnv,
    birdview: np.ndarray,
    states: list[dict[str, Any]] | None,
    camera_size: int,
) -> np.ndarray:
    if not states:
        return birdview
    out = np.ascontiguousarray(birdview.copy())
    desired_color = (40, 230, 80)
    actual_color = (240, 40, 40)
    desired_dot = (255, 230, 40)
    actual_dot = (40, 140, 255)
    for state in states:
        desired_pts = [
            pt for pt in (
                project_world_to_camera(env, np.asarray(xyz, dtype=float), "birdview", camera_size)
                for xyz in state.get("desired_trace_xyz", [])
            )
            if pt is not None
        ]
        actual_pts = [
            pt for pt in (
                project_world_to_camera(env, np.asarray(xyz, dtype=float), "birdview", camera_size)
                for xyz in state.get("actual_trace_xyz", [])
            )
            if pt is not None
        ]
        draw_polyline(out, desired_pts, desired_color, thickness=2)
        draw_polyline(out, actual_pts, actual_color, thickness=2)
        if desired_pts:
            draw_disk(out, desired_pts[-1], 4, desired_dot)
        if actual_pts:
            draw_disk(out, actual_pts[-1], 4, actual_dot)
    # Tiny legend in the top-left of the birdview panel.
    draw_line(out, (8, 10), (34, 10), desired_color, thickness=3)
    draw_line(out, (8, 22), (34, 22), actual_color, thickness=3)
    return out


def render_frame(
    env: OffScreenRenderEnv,
    obs: dict[str, Any] | None,
    camera_size: int,
    states: list[dict[str, Any]] | None = None,
    overlay_traces: bool = True,
) -> np.ndarray:
    left = render_named_camera(env, "birdview", camera_size, hide_robot=True)
    if left is None:
        left = render_named_camera(env, "birdview", camera_size)
    if left is None:
        left = obs_image(obs or {}, "agentview_image")
    elif overlay_traces:
        left = overlay_motion_traces(env, left, states, camera_size)

    right = render_named_camera(env, "agentview", camera_size)
    if right is None:
        right = obs_image(obs or {}, "agentview_image")

    frames = [f for f in (left, right) if f is not None]
    if not frames:
        raise RuntimeError("Could not render any preview camera frame")
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(frames, axis=1)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return defaults recursively updated by overrides."""
    out = deepcopy(defaults or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def task_bddl(suite_name: str, task_id: int) -> tuple[Any, Path]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return suite, bddl


def object_joint(env: OffScreenRenderEnv, object_name: str) -> str:
    if object_name in getattr(env.env, "objects_dict", {}):
        joints = getattr(env.env.objects_dict[object_name], "joints", None) or []
        if joints:
            return str(joints[0])
    if object_name in getattr(env.env, "fixtures_dict", {}):
        joints = getattr(env.env.fixtures_dict[object_name], "joints", None) or []
        if joints:
            return str(joints[0])
    return object_name


def set_joint_qpos(env: OffScreenRenderEnv, joint_name: str, qpos: np.ndarray | float) -> None:
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()


def set_init_state_allowing_extra_platform_joints(
    env: OffScreenRenderEnv,
    init_state: np.ndarray,
    *,
    added_platforms: int,
    added_platform_qpos: int = 7,
    added_platform_qvel: int = 6,
) -> dict[str, Any]:
    state = np.asarray(init_state, dtype=float)
    expected_len = 1 + int(env.sim.model.nq) + int(env.sim.model.nv)
    if len(state) == expected_len or added_platforms == 0:
        return env.set_init_state(state)

    old_nq = int(env.sim.model.nq) - int(added_platform_qpos) * int(added_platforms)
    old_nv = int(env.sim.model.nv) - int(added_platform_qvel) * int(added_platforms)
    old_expected_len = 1 + old_nq + old_nv
    if len(state) != old_expected_len:
        raise ValueError(
            f"Unexpected init state length {len(state)}; expected {expected_len} or legacy {old_expected_len}"
        )

    env.sim.data.time = float(state[0])
    env.sim.data.qpos[:] = 0.0
    env.sim.data.qvel[:] = 0.0
    env.sim.data.qpos[:old_nq] = state[1 : 1 + old_nq]
    env.sim.data.qvel[:old_nv] = state[1 + old_nq : 1 + old_nq + old_nv]
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()
    if hasattr(env.env, "_post_process"):
        env.env._post_process()
    if hasattr(env.env, "_update_observables"):
        env.env._update_observables(force=True)


def yaw_quat_delta(qpos: np.ndarray, yaw_delta: float) -> np.ndarray:
    if qpos.size < 7 or abs(yaw_delta) < 1e-9:
        return qpos
    half = 0.5 * yaw_delta
    delta = np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)
    w1, x1, y1, z1 = delta
    w2, x2, y2, z2 = qpos[3:7]
    qpos = qpos.copy()
    qpos[3:7] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    qpos[3:7] /= max(1e-9, float(np.linalg.norm(qpos[3:7])))
    return qpos


def yaw_quat_wxyz(yaw_delta: float) -> np.ndarray:
    half = 0.5 * float(yaw_delta)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def apply_initial_adjustments(env: OffScreenRenderEnv, task_cfg: dict[str, Any]) -> None:
    adjustments = task_cfg.get("initial_scene_adjustments", {}) or {}
    if not adjustments.get("enabled", False):
        return
    object_offsets = deepcopy(adjustments.get("object_offsets") or {})

    def _fixture_body_id(name: str) -> int | None:
        candidates = [f"{name}_main", name]
        try:
            if name in getattr(env.env, "fixtures_dict", {}):
                fixture = env.env.fixtures_dict[name]
                root_body = getattr(fixture, "root_body", None)
                if root_body:
                    candidates.insert(0, str(root_body))
        except Exception:
            pass
        for candidate in candidates:
            try:
                return int(env.sim.model.body_name2id(str(candidate)))
            except Exception:
                pass
        for body_idx in range(int(env.sim.model.nbody)):
            body_name = env.sim.model.body_id2name(body_idx)
            if body_name and str(body_name).startswith(f"{name}_"):
                return int(body_idx)
        return None

    fixture_names = set(getattr(env.env, "fixtures_dict", {}) or {})

    for name, item in object_offsets.items():
        offset = np.asarray(item.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=float)
        yaw_delta = float(item.get("yaw_delta_rad", 0.0) or 0.0)
        if str(name) not in fixture_names:
            try:
                joint = object_joint(env, name)
                qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
                if qpos.ndim > 0:
                    qpos[: min(3, qpos.size)] += offset[: min(3, qpos.size)]
                    qpos = yaw_quat_delta(qpos, yaw_delta)
                    set_joint_qpos(env, joint, qpos)
                    continue
            except Exception:
                pass
        body_id = _fixture_body_id(str(name))
        if body_id is None:
            raise KeyError(f"initial_scene_adjustments target not found as object or fixture body: {name}")
        env.sim.model.body_pos[body_id] = np.asarray(env.sim.model.body_pos[body_id], dtype=float) + offset[:3]
        if abs(yaw_delta) > 1e-12:
            env.sim.model.body_quat[body_id] = quat_multiply_wxyz(
                yaw_quat_wxyz(yaw_delta),
                np.asarray(env.sim.model.body_quat[body_id], dtype=float),
            )
        env.sim.forward()


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def polyline_segments(waypoints: list[list[float]], loop: bool) -> list[tuple[np.ndarray, np.ndarray, float]]:
    pts = [np.asarray(p, dtype=float) for p in waypoints if len(p) >= 3]
    if len(pts) <= 1:
        return []
    if loop and float(np.linalg.norm(pts[-1] - pts[0])) > 1e-9:
        pts = [*pts, pts[0]]
    return [(pts[i], pts[i + 1], float(np.linalg.norm(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)]


def segment_accel_duration(length: float, speed: float, accel: float | None) -> float:
    length = float(length)
    speed = max(float(speed), 1e-9)
    accel = 0.0 if accel is None else float(accel)
    if length <= 1e-9:
        return 0.0
    if accel <= 1e-9:
        return length / speed
    accel_dist = speed * speed / accel
    if length <= accel_dist:
        return 2.0 * math.sqrt(length / accel)
    return 2.0 * speed / accel + (length - accel_dist) / speed


def segment_accel_distance(t: float, length: float, speed: float, accel: float | None) -> float:
    """Distance traveled along one segment with zero start/end velocity."""
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
        # Triangular profile: accelerate then immediately decelerate.
        peak_time = math.sqrt(length / accel)
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


def polyline_duration(
    waypoints: list[list[float]],
    speed: float,
    loop: bool,
    stop_seconds: float = 0.0,
    accel_mps2: float | None = None,
) -> float:
    segments = [seg for seg in polyline_segments(waypoints, loop) if seg[2] > 1e-9]
    if not segments:
        return 0.0
    return float(
        sum(
            segment_accel_duration(length, speed, accel_mps2) + max(0.0, float(stop_seconds))
            for _a, _b, length in segments
        )
    )


def polyline_offset(
    waypoints: list[list[float]],
    elapsed: float,
    speed: float,
    loop: bool,
    *,
    stop_seconds: float = 0.0,
    smooth_stop: bool = False,
    accel_mps2: float | None = None,
) -> np.ndarray:
    pts = [np.asarray(p, dtype=float) for p in waypoints if len(p) >= 3]
    if len(pts) == 0:
        return np.zeros(3, dtype=float)
    if len(pts) == 1:
        return pts[0]
    segments = polyline_segments(waypoints, loop)
    total = polyline_duration(waypoints, speed, loop, stop_seconds, accel_mps2)
    if total <= 1e-9:
        return pts[0]
    t = max(0.0, float(elapsed))
    if loop:
        t = t % total
    else:
        t = min(t, total)
    for a, b, length in segments:
        if length <= 1e-9:
            continue
        move_duration = segment_accel_duration(length, speed, accel_mps2)
        if t <= move_duration:
            if accel_mps2 is not None and float(accel_mps2) > 1e-9:
                ratio = segment_accel_distance(t, length, speed, accel_mps2) / max(length, 1e-9)
            else:
                ratio = t / max(move_duration, 1e-9)
                if smooth_stop:
                    ratio = smoothstep(ratio)
            return a + ratio * (b - a)
        t -= move_duration
        dwell = max(0.0, float(stop_seconds))
        if t <= dwell:
            return b
        t -= dwell
    return segments[-1][1] if segments else pts[-1]


def catmull_rom_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def curve_waypoint_samples(
    waypoints: list[list[float]],
    loop: bool,
    samples_per_segment: int = 16,
) -> list[list[float]]:
    pts = [np.asarray(p, dtype=float) for p in waypoints if len(p) >= 3]
    if len(pts) <= 2:
        return [p.tolist() for p in pts]
    if loop and float(np.linalg.norm(pts[-1] - pts[0])) <= 1e-9:
        pts = pts[:-1]
    n = len(pts)
    if n <= 2:
        return [p.tolist() for p in pts]
    samples = []
    segment_count = n if loop else n - 1
    samples_per_segment = max(4, int(samples_per_segment))
    for i in range(segment_count):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p0 = pts[(i - 1) % n] if loop or i > 0 else pts[i]
        p3 = pts[(i + 2) % n] if loop or i + 2 < n else pts[i + 1]
        for j in range(samples_per_segment):
            u = j / float(samples_per_segment)
            samples.append(catmull_rom_point(p0, p1, p2, p3, u).tolist())
    if loop:
        samples.append(samples[0])
    else:
        samples.append(pts[-1].tolist())
    return samples


def curve_offset(
    waypoints: list[list[float]],
    elapsed: float,
    speed: float,
    loop: bool,
    samples_per_segment: int = 16,
) -> np.ndarray:
    sampled = curve_waypoint_samples(waypoints, loop, samples_per_segment)
    return polyline_offset(sampled, elapsed, speed, loop, stop_seconds=0.0, smooth_stop=False, accel_mps2=None)


def movable_offset(motion: dict[str, Any], elapsed: float) -> np.ndarray:
    motion_type = str(motion.get("type", motion.get("trajectory", "zigzag_waypoints")))
    if "waypoints_xyz" in motion:
        if motion_type in {"polynomial_spline", "smoothstep_pingpong"}:
            return curve_offset(
                motion.get("waypoints_xyz") or [],
                elapsed,
                float(motion.get("alpha_mps", 0.055) or 0.055),
                bool(motion.get("loop", True)),
                int(motion.get("curve_samples_per_segment", 16) or 16),
            )
        return polyline_offset(
            motion.get("waypoints_xyz") or [],
            elapsed,
            float(motion.get("alpha_mps", 0.055) or 0.055),
            bool(motion.get("loop", True)),
            stop_seconds=float(motion.get("waypoint_stop_seconds", 0.0) or 0.0),
            smooth_stop=bool(motion.get("waypoint_smooth_stop", False)),
            accel_mps2=(
                float(motion.get("waypoint_accel_mps2"))
                if motion.get("waypoint_accel_mps2", None) is not None
                else None
            ),
        )
    if motion_type == "conveyor_linear":
        direction = np.asarray(motion.get("direction_xyz", [1.0, 0.0, 0.0]), dtype=float)
        norm = max(1e-9, float(np.linalg.norm(direction)))
        return direction / norm * float(motion.get("alpha_mps", 0.055)) * elapsed
    return np.zeros(3, dtype=float)


def polyline_length(waypoints: list[list[float]], loop: bool) -> float:
    pts = [np.asarray(p, dtype=float) for p in waypoints if len(p) >= 3]
    if len(pts) <= 1:
        return 0.0
    if loop and float(np.linalg.norm(pts[-1] - pts[0])) > 1e-9:
        pts = [*pts, pts[0]]
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))


def motion_period_seconds(motion: dict[str, Any]) -> float | None:
    if "waypoints_xyz" in motion:
        speed = float(motion.get("alpha_mps", 0.055) or 0.055)
        motion_type = str(motion.get("type", motion.get("trajectory", "")))
        if motion_type in {"polynomial_spline", "smoothstep_pingpong"}:
            sampled = curve_waypoint_samples(
                motion.get("waypoints_xyz") or [],
                bool(motion.get("loop", True)),
                int(motion.get("curve_samples_per_segment", 16) or 16),
            )
            period = polyline_duration(sampled, speed, bool(motion.get("loop", True)), 0.0, None)
            return None if period <= 1e-9 else period
        period = polyline_duration(
            motion.get("waypoints_xyz") or [],
            speed,
            bool(motion.get("loop", True)),
            float(motion.get("waypoint_stop_seconds", 0.0) or 0.0),
            (
                float(motion.get("waypoint_accel_mps2"))
                if motion.get("waypoint_accel_mps2", None) is not None
                else None
            ),
        )
        return None if period <= 1e-9 else period
    if "waypoints_qpos" in motion:
        pts = [[float(v), 0.0, 0.0] for v in motion.get("waypoints_qpos", [])]
        length = polyline_length(pts, bool(motion.get("loop", True)))
        speed = float(motion.get("alpha_qpos_per_s", 0.025) or 0.025)
        return None if length <= 1e-9 else length / max(speed, 1e-9)
    motion_type = str(motion.get("type", motion.get("trajectory", "")))
    if motion_type == "conveyor_linear":
        return None
    if "amplitude_qpos" in motion:
        amp = float(motion.get("amplitude_qpos", 0.035) or 0.035)
        speed = float(motion.get("alpha_qpos_per_s", 0.025) or 0.025)
        return max(1.0, 4.0 * amp / max(1e-6, speed))
    return None


def apply_speed_override(targets: list[dict[str, Any]], speed_mps: float | None) -> list[dict[str, Any]]:
    if speed_mps is None:
        return targets
    out = deepcopy(targets)
    for target in out:
        motion = target.get("motion")
        if isinstance(motion, dict) and "alpha_mps" in motion:
            motion["alpha_mps"] = float(speed_mps)
    return out


def apply_waypoint_timing_override(
    targets: list[dict[str, Any]],
    stop_seconds: float | None,
    smooth_stop: bool,
    accel_mps2: float | None,
) -> list[dict[str, Any]]:
    if stop_seconds is None and not smooth_stop and accel_mps2 is None:
        return targets
    out = deepcopy(targets)
    for target in out:
        motion = target.get("motion")
        if not isinstance(motion, dict) or "waypoints_xyz" not in motion:
            continue
        if stop_seconds is not None:
            motion["waypoint_stop_seconds"] = float(stop_seconds)
        if smooth_stop:
            motion["waypoint_smooth_stop"] = True
        if accel_mps2 is not None:
            motion["waypoint_accel_mps2"] = float(accel_mps2)
    return out


def apply_carrier_settings_to_targets(
    targets: list[dict[str, Any]],
    carrier_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not carrier_cfg:
        return targets
    out = deepcopy(targets)
    passthrough_keys = (
        "exclude_static_support_contact",
        "friction",
        "solref",
        "solimp",
        "margin",
        "condim",
        "surface_epsilon_m",
        "bottom_contact_margin_m",
        "visible",
        "cover_motion_path",
        "path_cover_margin_m",
        "size",
        "contact_all_target_geoms",
        "stability_proxy_enabled",
        "stability_proxy_size",
        "stability_proxy_size_xy_m",
        "stability_proxy_thickness_m",
        "stability_proxy_bottom_epsilon_m",
        "platform_use_support_plane_z",
    )
    for target in out:
        for key in passthrough_keys:
            # The generated benchmark entries used to set
            # contact_all_target_geoms=false on every target as a conservative
            # default. For the physical conveyor preview this is unsafe: flat
            # objects such as plates and mugs can keep resting on the table
            # while the invisible patch only pairs with one non-support geom.
            # Treat the carrier config as the source of truth for this key so
            # the patch contacts the full target object when requested.
            if key == "contact_all_target_geoms" and key in carrier_cfg:
                target[key] = deepcopy(carrier_cfg[key])
            elif key in carrier_cfg and key not in target:
                target[key] = deepcopy(carrier_cfg[key])
    return out


def force_object_sized_preview_platform(target: dict[str, Any]) -> bool:
    """Keep cabinet/drawer bowl carriers at object footprint size.

    A path-covering preview platform is helpful for many moving supports, but it
    becomes misleadingly large for bowl-on-cabinet / bowl-in-drawer tasks. For
    that family we keep the platform matched to the carried bowl footprint
    instead of expanding it across the full motion envelope.
    """
    support_fields = (
        target.get("target_object_name"),
        target.get("target_joint_name"),
        target.get("target_body_name"),
        target.get("target_object"),
        target.get("object_name"),
    )
    support_text = " ".join(str(item).lower() for item in support_fields if item)
    if not any(token in support_text for token in ("cabinet", "drawer")):
        return False
    coupled_fields = [
        *(target.get("coupled_object_names") or []),
        *(target.get("allowed_collision_body_prefixes") or []),
    ]
    coupled_text = " ".join(str(item).lower() for item in coupled_fields if item)
    return "bowl" in coupled_text


def apply_static_support_contact_override(
    targets: list[dict[str, Any]],
    *,
    keep_static_support_contact: bool,
    static_support_clearance_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Keep table/cabinet support contacts active while the conveyor drives.

    Reviewed configs default to replacing the static tabletop support with the
    hidden conveyor patch.  That is visually stable, but it also means a target
    that tips or loses carrier contact can pass through the table/cabinet.  This
    override preserves the conveyor-platform mechanism while leaving the
    target-vs-support contacts enabled.
    """
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
    return out


def joint_offset(motion: dict[str, Any], elapsed: float) -> float:
    if "waypoints_qpos" in motion:
        pts = [[float(v), 0.0, 0.0] for v in motion.get("waypoints_qpos", [])]
        return float(polyline_offset(pts, elapsed, float(motion.get("alpha_qpos_per_s", 0.025)), True)[0])
    amp = float(motion.get("amplitude_qpos", 0.035) or 0.035)
    speed = float(motion.get("alpha_qpos_per_s", 0.025) or 0.025)
    period = max(1.0, 4.0 * amp / max(1e-6, speed))
    return float(amp * math.sin(2.0 * math.pi * elapsed / period))


def yaw_motion_offset(motion: dict[str, Any], elapsed: float) -> float:
    if "waypoints_yaw_rad" in motion:
        pts = [[float(v), 0.0, 0.0] for v in motion.get("waypoints_yaw_rad", [])]
        speed = float(motion.get("alpha_yaw_rad_per_s", motion.get("alpha_qpos_per_s", 0.20)) or 0.20)
        smooth = str(motion.get("type", "")).lower() in {"body_yaw_polynomial_spline", "joint_polynomial_spline"}
        if smooth:
            return float(curve_offset(pts, elapsed, speed, bool(motion.get("loop", True)))[0])
        return float(polyline_offset(pts, elapsed, speed, bool(motion.get("loop", True)))[0])
    amp = float(motion.get("amplitude_yaw_rad", motion.get("amplitude_qpos", math.radians(15.0))) or math.radians(15.0))
    speed = float(motion.get("alpha_yaw_rad_per_s", motion.get("alpha_qpos_per_s", 0.20)) or 0.20)
    period = max(1.0, 4.0 * abs(amp) / max(1e-6, speed))
    return float(amp * math.sin(2.0 * math.pi * elapsed / period))


def target_entries_for_preview(task_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries = task_cfg.get("target_objects") or []
    if entries:
        return deepcopy(entries)
    motion = deepcopy(task_cfg.get("motion", {}) or {})
    return [
        {
            "target_object_name": task_cfg.get("target_object_name"),
            "target_joint_name": task_cfg.get("target_joint_name"),
            "target_body_name": task_cfg.get("target_body_name"),
            "target_kind": task_cfg.get("target_kind"),
            "dynamic_role": task_cfg.get("dynamic_role"),
            "motion": motion,
        }
    ]


def carrier_type_for_task(task_cfg: dict[str, Any]) -> str:
    carrier_cfg = task_cfg.get("carrier", {}) or {}
    if not bool(carrier_cfg.get("enabled", False)):
        return "none"
    return str(carrier_cfg.get("type", "physical_conveyor_patch"))


def apply_carrier_override(task_cfg: dict[str, Any], carrier_override: str | None) -> dict[str, Any]:
    if not carrier_override:
        return task_cfg
    out = deepcopy(task_cfg)
    carrier = out.setdefault("carrier", {})
    if carrier_override == "none":
        carrier["enabled"] = False
        carrier.pop("type", None)
    else:
        carrier["enabled"] = True
        carrier["type"] = carrier_override
    return out


def preview_items(config_dir: Path, levels: list[str]) -> list[dict[str, Any]]:
    out = []
    for level in levels:
        cfg_path = config_dir / f"libero_dynamic_manual_{level}.yaml"
        if not cfg_path.exists():
            cfg_path = config_dir / f"libero_dynamic_{level}.yaml"
        cfg = load_yaml(cfg_path)
        task_defaults = deepcopy(cfg.get("defaults") or {})
        for suite_name, suite_tasks in (cfg.get("tasks") or {}).items():
            for task_id_raw, task_cfg in (suite_tasks or {}).items():
                task_cfg = deep_merge(task_defaults, task_cfg or {})
                task_id = int(task_id_raw)
                if not task_cfg.get("enabled", False):
                    continue
                variants = task_cfg.get("activation_variants") or []
                if variants:
                    for variant in variants:
                        item_cfg = deepcopy(task_cfg)
                        # Variant-level annotations may have their own initial
                        # object placements.  Without copying these fields the
                        # preview silently reuses the task-level placement for
                        # every trajectory, which makes rendered masks drift
                        # from the manual editor after the first variant.
                        for variant_key in (
                            "initial_scene_adjustments",
                            "manual_review",
                            "logging",
                            "maskfit_alignment",
                        ):
                            if variant_key in variant:
                                item_cfg[variant_key] = deepcopy(variant[variant_key])
                        item_cfg["target_object_name"] = variant.get("target_object_name")
                        item_cfg["target_joint_name"] = variant.get("target_joint_name")
                        item_cfg["target_objects"] = deepcopy(variant.get("target_objects") or [])
                        if item_cfg["target_objects"]:
                            item_cfg["motion"] = deepcopy((item_cfg["target_objects"][0] or {}).get("motion", {}))
                        else:
                            item_cfg["motion"] = deepcopy(variant.get("motion") or item_cfg.get("motion") or {})
                        out.append(
                            {
                                "level": level,
                                "suite": suite_name,
                                "task_id": task_id,
                                "variant_name": variant.get("variant_name", "variant"),
                                "variant_id": variant.get("variant_id"),
                            "task_cfg": item_cfg,
                        }
                    )
                else:
                    out.append(
                        {
                            "level": level,
                            "suite": suite_name,
                            "task_id": task_id,
                            "variant_name": "base",
                            "variant_id": f"{suite_name}_task{task_id:02d}__{level}",
                            "task_cfg": deepcopy(task_cfg),
                        }
                    )
    return out


def target_geom_ids(env: OffScreenRenderEnv, object_name: str) -> list[int]:
    if object_name not in getattr(env.env, "objects_dict", {}):
        return []
    obj = env.env.objects_dict[object_name]
    geom_names = list(getattr(obj, "contact_geoms", []) or []) + list(getattr(obj, "visual_geoms", []) or [])
    ids = []
    for geom_name in geom_names:
        try:
            ids.append(int(env.sim.model.geom_name2id(geom_name)))
        except Exception:
            pass
    return sorted(set(ids))


def body_subtree_geom_ids(env: OffScreenRenderEnv, body_id: int) -> list[int]:
    body_id = int(body_id)
    body_ids = {body_id}
    changed = True
    while changed:
        changed = False
        for idx in range(int(env.sim.model.nbody)):
            if idx in body_ids:
                continue
            parent = int(env.sim.model.body_parentid[idx])
            if parent in body_ids:
                body_ids.add(int(idx))
                changed = True
    return [
        int(geom_id)
        for geom_id in range(int(env.sim.model.ngeom))
        if int(env.sim.model.geom_bodyid[geom_id]) in body_ids
    ]


def capture_joint_states(env: OffScreenRenderEnv) -> dict[str, dict[str, np.ndarray]]:
    """Capture qpos/qvel by joint name so XML edits cannot reorder state."""
    states: dict[str, dict[str, np.ndarray]] = {}
    for joint_id in range(int(env.sim.model.njnt)):
        name = env.sim.model.joint_id2name(joint_id)
        if not name:
            continue
        try:
            qpos = np.asarray(env.sim.data.get_joint_qpos(name).copy(), dtype=float)
            qvel = np.asarray(env.sim.data.get_joint_qvel(name).copy(), dtype=float)
        except Exception:
            continue
        states[str(name)] = {"qpos": qpos, "qvel": qvel}
    return states


def restore_joint_states(env: OffScreenRenderEnv, states: dict[str, dict[str, np.ndarray]]) -> None:
    """Restore captured qpos/qvel by joint name after reset_from_xml_string."""
    for name, values in states.items():
        try:
            env.sim.model.joint_name2id(name)
        except Exception:
            continue
        try:
            env.sim.data.set_joint_qpos(name, values["qpos"])
        except Exception:
            pass
        try:
            env.sim.data.set_joint_qvel(name, values["qvel"])
        except Exception:
            pass
    env.sim.forward()


def capture_geom_rgba(env: OffScreenRenderEnv) -> dict[str, list[float]]:
    """Capture original geom visibility/material alpha before XML reinjection."""
    rgba: dict[str, list[float]] = {}
    for geom_id in range(int(env.sim.model.ngeom)):
        geom_name = env.sim.model.geom_id2name(geom_id)
        if not geom_name:
            continue
        rgba[str(geom_name)] = [float(x) for x in np.asarray(env.sim.model.geom_rgba[geom_id], dtype=float)]
    return rgba


def restore_geom_rgba(env: OffScreenRenderEnv, rgba: dict[str, list[float]]) -> None:
    """Restore original geom rgba so collision geoms do not become visible."""
    for geom_name, values in rgba.items():
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        try:
            env.sim.model.geom_rgba[geom_id] = np.asarray(values, dtype=float)
        except Exception:
            pass
    env.sim.forward()


def target_contact_geom_names(env: OffScreenRenderEnv, object_name: str, bottom_margin_m: float = 0.002) -> list[str]:
    if object_name not in getattr(env.env, "objects_dict", {}):
        return []
    obj = env.env.objects_dict[object_name]
    candidates = []
    for name in list(getattr(obj, "contact_geoms", []) or []):
        try:
            geom_id = int(env.sim.model.geom_name2id(name))
        except Exception:
            continue
        if int(env.sim.model.geom_contype[geom_id]) != 0:
            geom_z = float(env.sim.data.geom_xpos[geom_id][2])
            geom_size_z = float(env.sim.model.geom_size[geom_id][2])
            xy_area = float(env.sim.model.geom_size[geom_id][0] * env.sim.model.geom_size[geom_id][1])
            candidates.append((geom_z - geom_size_z, xy_area, str(name)))
    if not candidates:
        return []
    min_bottom = min(bottom for bottom, _, _ in candidates)
    bottom_margin_m = min(float(bottom_margin_m), 0.002)
    bottom_candidates = [
        (area, name) for bottom, area, name in candidates
        if bottom <= min_bottom + float(bottom_margin_m)
    ]
    if not bottom_candidates:
        return [min(candidates, key=lambda item: item[0])[2]]
    if "moka_pot" in str(object_name):
        # The large moka-pot body geom creates diagonal/side contacts against a
        # moving patch and can kick the object sideways. Use the small bottom
        # support geoms instead; they behave like feet resting on a conveyor.
        foot_names = [
            name for _, name in bottom_candidates
            if re.search(r"_g[5-8]$", str(name))
        ]
        if foot_names:
            return sorted(foot_names)
    # Use every bottom support geom, not just the largest representative geom.
    # Several LIBERO assets (cans, soup boxes, baskets, bowls) touch the table
    # through many small bottom geoms. Pairing the hidden conveyor with only
    # one geom lets the platform move while the real support geoms keep resting
    # on the table, so the object appears static. These pairs are still scoped
    # to the carried object only; the platform remains non-interacting for all
    # other scene bodies.
    return sorted({name for _, name in bottom_candidates})


def target_all_contact_geom_names(env: OffScreenRenderEnv, object_name: str) -> list[str]:
    if object_name not in getattr(env.env, "objects_dict", {}):
        return []
    obj = env.env.objects_dict[object_name]
    names = []
    for name in list(getattr(obj, "contact_geoms", []) or []):
        try:
            geom_id = int(env.sim.model.geom_name2id(name))
        except Exception:
            continue
        if int(env.sim.model.geom_contype[geom_id]) != 0:
            names.append(str(name))
    return names


def target_body_name(env: OffScreenRenderEnv, object_name: str) -> str | None:
    names = target_body_names(env, object_name)
    return names[0] if names else None


def target_body_names(env: OffScreenRenderEnv, object_name: str) -> list[str]:
    if object_name not in getattr(env.env, "objects_dict", {}):
        return []
    obj = env.env.objects_dict[object_name]
    names = []
    for name in list(getattr(obj, "contact_geoms", []) or []):
        try:
            geom_id = int(env.sim.model.geom_name2id(name))
            body_id = int(env.sim.model.geom_bodyid[geom_id])
            body_name = env.sim.model.body_id2name(body_id)
            if body_name:
                names.append(str(body_name))
        except Exception:
            continue
    return sorted(set(names))


def support_exclude_body_names(env: OffScreenRenderEnv) -> list[str]:
    # Only remove the static support planes that the invisible conveyor patch
    # replaces. Fixtures such as stoves, cabinets and drawers must keep their
    # normal contacts so invalid trajectories still show up as physical
    # collisions instead of silently passing through obstacles.
    keywords = (
        "table",
        "counter",
        "floor",
        "arena",
        "living_room_table",
    )
    # Do not exclude the whole world body: preview conveyor platforms are
    # injected under worldbody, and excluding target-vs-world can also suppress
    # the explicit target-vs-platform contact pairs.
    names = set()
    for body_id in range(int(env.sim.model.nbody)):
        body_name = env.sim.model.body_id2name(body_id)
        if not body_name:
            continue
        if any(keyword in body_name for keyword in keywords):
            names.add(str(body_name))
    return sorted(names)


def platform_non_target_exclude_body_names(
    env: OffScreenRenderEnv,
    platform_body_name: str,
    target_body_names: list[str],
) -> list[str]:
    """Bodies that the invisible carrier must never collide with.

    Do not exclude `world`: the target object is also under worldbody, and a
    broad platform-vs-world exclude can suppress the explicit platform-target
    contact pairs we rely on. Instead, enumerate concrete non-target bodies.
    """
    target_set = set(target_body_names)
    exclude = set(support_exclude_body_names(env))
    object_names = set(getattr(env.env, "objects_dict", {}) or {})
    fixture_names = set(getattr(env.env, "fixtures_dict", {}) or {})
    for body_id in range(int(env.sim.model.nbody)):
        body_name = env.sim.model.body_id2name(body_id)
        if not body_name or body_name == "world" or body_name == platform_body_name:
            continue
        if body_name in target_set:
            continue
        if any(body_name.startswith(f"{target}_") for target in target_set):
            continue
        if body_name.startswith(("robot0_", "gripper0_")):
            exclude.add(str(body_name))
            continue
        if body_name in fixture_names or any(body_name.startswith(f"{fixture}_") for fixture in fixture_names):
            exclude.add(str(body_name))
            continue
        if body_name in object_names or any(body_name.startswith(f"{obj}_") for obj in object_names if obj not in target_set):
            exclude.add(str(body_name))
    return sorted(exclude)


def estimate_support_plane_z_under_target(env: OffScreenRenderEnv, object_name: str) -> float | None:
    """Estimate the table/floor top surface directly below a target object."""
    bottom_points = []
    for geom_name in target_contact_geom_names(env, object_name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        geom_z = float(env.sim.data.geom_xpos[geom_id][2])
        geom_size_z = float(env.sim.model.geom_size[geom_id][2])
        bottom_points.append(geom_z - geom_size_z)
    if not bottom_points:
        return None
    target_bottom = float(min(bottom_points))
    support_body_names = set(support_exclude_body_names(env))
    support_tops = []
    for geom_id in range(int(env.sim.model.ngeom)):
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        body_name = env.sim.model.body_id2name(body_id)
        geom_name_value = env.sim.model.geom_id2name(geom_id) or ""
        if body_name not in support_body_names and not any(key in geom_name_value for key in ("floor", "table")):
            continue
        geom_type = int(env.sim.model.geom_type[geom_id])
        geom_z = float(env.sim.data.geom_xpos[geom_id][2])
        geom_size = np.asarray(env.sim.model.geom_size[geom_id], dtype=float)
        if geom_type == 0:  # plane
            top_z = geom_z
        else:
            top_z = geom_z + float(geom_size[2])
        # Ignore distant supports such as the room floor when the object is on a
        # table/cabinet. A too-low support estimate makes the hidden conveyor sit
        # below the scene and lets the carried object fall through its tabletop.
        if target_bottom - 0.08 <= top_z <= target_bottom + 0.06:
            support_tops.append(top_z)
    if not support_tops:
        return None
    return float(max(support_tops))


def align_target_bottom_to_support_plane(
    env: OffScreenRenderEnv,
    object_name: str,
    *,
    max_adjust_m: float = 0.03,
) -> None:
    """Lift objects out of support penetration without pushing valid poses down."""
    support_z = estimate_support_plane_z_under_target(env, object_name)
    if support_z is None:
        return
    bottom_points = []
    for geom_name in target_contact_geom_names(env, object_name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        geom_z = float(env.sim.data.geom_xpos[geom_id][2])
        geom_size_z = float(env.sim.model.geom_size[geom_id][2])
        bottom_points.append(geom_z - geom_size_z)
    if not bottom_points:
        return
    current_bottom = float(min(bottom_points))
    delta_z = float(support_z - current_bottom)
    # Support-plane estimation can be too low for mesh furniture such as the
    # living-room table.  Lowering an already valid annotated pose can bury the
    # object in the tabletop, so this safety pass only fixes penetration by
    # raising objects.  Floating cases are handled by the explicit settle phase.
    if delta_z <= 0.0:
        return
    if abs(delta_z) < 1e-5 or abs(delta_z) > float(max_adjust_m):
        return
    try:
        joint = object_joint(env, object_name)
        qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
        if qpos.size < 3:
            return
        qpos[2] += delta_z
        env.sim.data.set_joint_qpos(joint, qpos)
        try:
            qvel = np.asarray(env.sim.data.get_joint_qvel(joint).copy(), dtype=float)
            qvel[:] = 0.0
            env.sim.data.set_joint_qvel(joint, qvel)
        except Exception:
            pass
        env.sim.forward()
    except Exception:
        return


def geom_name(env: OffScreenRenderEnv, geom_id: int) -> str:
    try:
        return env.sim.model.geom_id2name(int(geom_id)) or ""
    except Exception:
        return ""


def body_name_for_geom(env: OffScreenRenderEnv, geom_id: int) -> str:
    try:
        body_id = int(env.sim.model.geom_bodyid[int(geom_id)])
        return env.sim.model.body_id2name(body_id) or ""
    except Exception:
        return ""


def is_robot_geom(env: OffScreenRenderEnv, geom_id: int) -> bool:
    gname = geom_name(env, geom_id)
    bname = body_name_for_geom(env, geom_id)
    return gname.startswith(("robot0_", "gripper0_", "mount0_")) or bname.startswith(("robot0_", "gripper0_", "mount0_"))


def is_support_geom(env: OffScreenRenderEnv, geom_id: int) -> bool:
    if is_robot_geom(env, geom_id):
        return False
    gname = geom_name(env, geom_id)
    bname = body_name_for_geom(env, geom_id)
    if gname.startswith("libero_dynamic_") or bname.startswith("libero_dynamic_"):
        return False
    support_keywords = ("table", "counter", "stove", "cabinet", "drawer", "shelf", "basket", "plate", "tray")
    if gname.startswith(("floor", "table", "main_table", "arena")):
        return True
    if bname.startswith(("floor", "table", "main_table", "arena")):
        return True
    if any(keyword in gname for keyword in support_keywords):
        return True
    if any(keyword in bname for keyword in support_keywords):
        return True
    fixture_names = set(getattr(env.env, "fixtures_dict", {}) or {})
    if bname in fixture_names:
        return True
    if any(bname.startswith(f"{fixture}_") for fixture in fixture_names):
        return True
    if any(gname.startswith(f"{fixture}_") for fixture in fixture_names):
        return True
    return False


def has_support_contact(env: OffScreenRenderEnv, object_name: str) -> tuple[bool, list[dict[str, Any]]]:
    target_ids = set(target_geom_ids(env, object_name))
    if not target_ids:
        return False, []
    contacts = []
    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in target_ids:
            other = geom2
            target_geom = geom1
        elif geom2 in target_ids:
            other = geom1
            target_geom = geom2
        else:
            continue
        if is_support_geom(env, other):
            contacts.append(
                {
                    "target_geom": geom_name(env, target_geom),
                    "support_geom": geom_name(env, other),
                    "support_body": body_name_for_geom(env, other),
                    "distance": float(contact.dist),
                    "pos": np.asarray(contact.pos, dtype=float).tolist(),
                }
            )
    return bool(contacts), contacts


def platform_geom_ids_for_state(env: OffScreenRenderEnv, state: dict[str, Any]) -> set[int]:
    platform = state.get("platform") or {}
    out: set[int] = set()
    for geom in platform.get("carrier_geom_names", [platform.get("geom_name")]):
        if not geom:
            continue
        try:
            out.add(int(env.sim.model.geom_name2id(str(geom))))
        except Exception:
            pass
    return out


def target_geom_ids_for_state(env: OffScreenRenderEnv, state: dict[str, Any]) -> set[int]:
    if (state.get("is_body_yaw") or state.get("is_body_translation")) and state.get("body_id") is not None:
        return set(body_subtree_geom_ids(env, int(state["body_id"])))
    return set(target_geom_ids(env, str(state.get("name"))))


def allowed_dynamic_contact(env: OffScreenRenderEnv, other_geom: int, platform_geom_ids: set[int]) -> bool:
    if int(other_geom) in platform_geom_ids:
        return True
    if is_robot_geom(env, int(other_geom)):
        # In a real rollout, gripper/robot contact is not a trajectory-design
        # collision. This preview has no active policy, so robot contact mostly
        # indicates static overlap with the parked arm and should not invalidate
        # a dynamic object path.
        return True
    gname = geom_name(env, int(other_geom))
    bname = body_name_for_geom(env, int(other_geom))
    if gname.startswith(("floor", "table", "main_table", "arena")):
        return True
    if bname.startswith(("floor", "table", "main_table", "arena")):
        return True
    return False


def update_safety_diagnostics(env: OffScreenRenderEnv, state: dict[str, Any], camera_size: int) -> None:
    if state.get("safety_target_geom_ids") is None:
        state["safety_target_geom_ids"] = target_geom_ids_for_state(env, state)
        state["safety_platform_geom_ids"] = platform_geom_ids_for_state(env, state)
    target_ids = set(state.get("safety_target_geom_ids") or [])
    platform_ids = set(state.get("safety_platform_geom_ids") or [])
    collision_events = state.setdefault("collision_events", [])
    collision_keys = state.setdefault("collision_keys", set())
    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in target_ids:
            target_geom, other_geom = geom1, geom2
        elif geom2 in target_ids:
            target_geom, other_geom = geom2, geom1
        else:
            continue
        if other_geom in target_ids:
            continue
        if allowed_dynamic_contact(env, other_geom, platform_ids):
            continue
        if float(contact.dist) > -float(state.get("collision_penetration_threshold_m", 1e-4)):
            continue
        key = (geom_name(env, target_geom), geom_name(env, other_geom), body_name_for_geom(env, other_geom))
        if key in collision_keys:
            continue
        collision_keys.add(key)
        if len(collision_events) < 20:
            collision_events.append(
                {
                    "target_geom": key[0],
                    "other_geom": key[1],
                    "other_body": key[2],
                    "distance": float(contact.dist),
                    "pos": np.asarray(contact.pos, dtype=float).tolist(),
                }
            )
    point = None
    if (state.get("is_body_yaw") or state.get("is_body_translation")) and state.get("body_id") is not None:
        point = np.asarray(env.sim.data.body_xpos[int(state["body_id"])], dtype=float)
    elif not state.get("is_joint"):
        try:
            qpos = np.asarray(env.sim.data.get_joint_qpos(state["joint"]).copy(), dtype=float)
            if qpos.size >= 3:
                point = qpos[:3]
        except Exception:
            point = None
    if point is None:
        return

    def _update_camera_visibility(camera_name: str) -> None:
        sample_key = f"{camera_name}_samples"
        offscreen_key = f"{camera_name}_offscreen_samples"
        edge_key = f"{camera_name}_edge_samples"
        state[sample_key] = int(state.get(sample_key, 0)) + 1
        projected = project_world_to_camera(env, np.asarray(point, dtype=float), camera_name, int(camera_size))
        if projected is None:
            state[offscreen_key] = int(state.get(offscreen_key, 0)) + 1
            return
        u, v = projected
        margin = int(max(8, round(0.04 * int(camera_size))))
        if u < margin or v < margin or u >= int(camera_size) - margin or v >= int(camera_size) - margin:
            state[edge_key] = int(state.get(edge_key, 0)) + 1

    _update_camera_visibility("birdview")
    _update_camera_visibility("agentview")


def is_articulated_target(target: dict[str, Any], base_qpos: np.ndarray | None = None) -> bool:
    motion = target.get("motion", {}) or {}
    motion_type = str(motion.get("type", motion.get("trajectory", "")))
    if is_fixture_body_yaw_target(target):
        return True
    if motion_type.startswith("joint_"):
        return True
    if target.get("target_kind") == "articulated_fixture_joint":
        return True
    if target.get("dynamic_role") == "articulated_joint_dynamic":
        return True
    return bool(base_qpos is not None and (np.asarray(base_qpos).ndim == 0 or np.asarray(base_qpos).size < 3))


def is_fixture_body_yaw_target(target: dict[str, Any]) -> bool:
    motion = target.get("motion", {}) or {}
    motion_type = str(motion.get("type", motion.get("trajectory", ""))).lower()
    return (
        str(target.get("dynamic_role", "")) == "fixture_body_yaw_dynamic"
        or str(target.get("target_kind", "")) == "articulated_fixture_body"
        or motion_type.startswith("body_yaw_")
    )

def is_fixture_body_translation_target(target: dict[str, Any]) -> bool:
    motion = target.get("motion", {}) or {}
    motion_type = str(motion.get("type", motion.get("trajectory", ""))).lower()
    return (
        str(target.get("dynamic_role", "")) == "fixture_body_translation_dynamic"
        or str(target.get("target_kind", "")) == "fixture_body"
        or motion_type.startswith("body_translation_")
    )


def fixture_target_body_names(env: OffScreenRenderEnv, target: dict[str, Any]) -> list[str]:
    """Return all concrete MuJoCo bodies belonging to a translated fixture."""
    name = str(target.get("target_object_name") or target.get("object_name") or "")
    prefixes = [f"{name}_"] if name else []
    explicit = [target.get("target_body_name"), target.get("body_name"), f"{name}_main" if name else None, name or None]
    bodies: set[str] = set()
    for candidate in explicit:
        if not candidate:
            continue
        try:
            env.sim.model.body_name2id(str(candidate))
            bodies.add(str(candidate))
        except Exception:
            pass
    for body_id in range(int(env.sim.model.nbody)):
        body_name = env.sim.model.body_id2name(body_id)
        if body_name and any(str(body_name).startswith(prefix) for prefix in prefixes):
            bodies.add(str(body_name))
    return sorted(bodies)


def object_supported_by_any_body(
    env: OffScreenRenderEnv,
    object_name: str,
    support_body_names: set[str],
) -> tuple[bool, list[dict[str, Any]]]:
    """Check whether a free object is resting on / inside a moving fixture."""
    active, contacts = has_support_contact(env, object_name)
    if not active:
        return False, []
    matched = [
        contact
        for contact in contacts
        if str(contact.get("support_body") or "") in support_body_names
    ]
    return bool(matched), matched


def append_fixture_supported_passenger_targets(
    env: OffScreenRenderEnv,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add physical passenger conveyors for objects carried by moving fixtures.

    MuJoCo does not reliably infer tangential surface velocity when a fixture is
    moved by editing model body positions. Without this, a bowl in a drawer can
    move only after being pushed by the cabinet wall. For these supported free
    objects, add a tiny hidden platform under the object that follows the same
    fixture trajectory. The object is still moved by contact/friction, not by
    directly overwriting its qpos.
    """
    out = deepcopy(targets)
    existing = {str(t.get("target_object_name") or t.get("object_name")) for t in out}
    fixture_targets = [target for target in out if is_fixture_body_translation_target(target)]
    if not fixture_targets:
        return out

    for fixture_target in fixture_targets:
        fixture_bodies = set(fixture_target_body_names(env, fixture_target))
        if not fixture_bodies:
            continue
        fixture_name = str(fixture_target.get("target_object_name") or fixture_target.get("object_name") or "")
        for object_name in sorted((getattr(env.env, "objects_dict", {}) or {}).keys()):
            if object_name in existing:
                continue
            try:
                qpos = np.asarray(env.sim.data.get_joint_qpos(object_joint(env, object_name)), dtype=float)
            except Exception:
                continue
            if qpos.size < 3:
                continue
            supported, contacts = object_supported_by_any_body(env, object_name, fixture_bodies)
            if not supported:
                continue
            passenger = deepcopy(fixture_target)
            passenger["target_object_name"] = str(object_name)
            passenger["target_joint_name"] = object_joint(env, object_name)
            passenger["target_kind"] = "movable_object"
            passenger["dynamic_role"] = "fixture_supported_passenger_conveyor"
            passenger["support_fixture_target_name"] = fixture_name
            passenger["support_contact_bodies"] = sorted({str(c.get("support_body")) for c in contacts if c.get("support_body")})
            passenger["support_fixture_body_names"] = sorted(fixture_bodies)
            # The moving fixture is already represented by this passenger's
            # own conveyor patch following the same trajectory. Keep normal
            # object physics with the robot / other objects, but remove direct
            # passenger-vs-fixture contact so cabinet walls do not push or pin
            # the object while the hidden carrier is trying to move it.
            passenger["exclude_support_fixture_contact"] = True
            for key in ("target_body_name", "body_name", "platform_use_support_plane_z"):
                passenger.pop(key, None)
            passenger["exclude_static_support_contact"] = False
            passenger["contact_all_target_geoms"] = True
            passenger["surface_epsilon_m"] = min(float(passenger.get("surface_epsilon_m", 0.001) or 0.001), 0.001)
            passenger["bottom_contact_margin_m"] = max(float(passenger.get("bottom_contact_margin_m", 0.0015) or 0.0015), 0.01)
            passenger["stability_proxy_enabled"] = False
            out.append(passenger)
            existing.add(str(object_name))
    return out


def env_xml(env: OffScreenRenderEnv) -> str:
    model = getattr(getattr(env, "env", None), "model", None) or getattr(env, "model", None)
    if model is None:
        raise RuntimeError("Could not locate robosuite model XML for preview platform injection")
    return model.get_xml()


def capture_body_placements(env: OffScreenRenderEnv) -> dict[str, dict[str, list[float]]]:
    placements = {}
    for body_id in range(int(env.sim.model.nbody)):
        body_name = env.sim.model.body_id2name(body_id)
        if not body_name:
            continue
        placements[str(body_name)] = {
            "pos": np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy().tolist(),
            "quat": np.asarray(env.sim.model.body_quat[body_id], dtype=float).copy().tolist(),
        }
    return placements


def jointed_body_subtree_ids(env: OffScreenRenderEnv) -> set[int]:
    """Bodies whose runtime pose is controlled by qpos/qvel, not model offsets."""
    body_ids = {
        int(env.sim.model.jnt_bodyid[joint_id])
        for joint_id in range(int(env.sim.model.njnt))
    }
    changed = True
    while changed:
        changed = False
        for body_id in range(int(env.sim.model.nbody)):
            if body_id in body_ids:
                continue
            parent_id = int(env.sim.model.body_parentid[body_id])
            if parent_id in body_ids:
                body_ids.add(int(body_id))
                changed = True
    return body_ids


def restore_body_placements(env: OffScreenRenderEnv, placements: dict[str, dict[str, list[float]]]) -> None:
    skip_body_ids = jointed_body_subtree_ids(env)
    for body_name, placement in placements.items():
        try:
            body_id = int(env.sim.model.body_name2id(body_name))
        except Exception:
            continue
        if body_id in skip_body_ids:
            continue
        env.sim.model.body_pos[body_id] = np.asarray(placement["pos"], dtype=float)
        env.sim.model.body_quat[body_id] = np.asarray(placement["quat"], dtype=float)
    env.sim.forward()


def estimate_target_platform_size(env: OffScreenRenderEnv, object_name: str) -> list[float]:
    if "moka_pot" in str(object_name):
        return [0.06, 0.06, 0.00025]
    joint = object_joint(env, object_name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=float)
    center_xy = target_platform_xy_center(env, object_name, fallback_xy=qpos[:2] if qpos.size >= 2 else None)
    radii = []
    for geom_name in target_contact_geom_names(env, object_name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        geom_pos = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
        rbound = float(env.sim.model.geom_rbound[geom_id])
        radii.append(np.abs(geom_pos[:2] - center_xy) + rbound)
    if not radii:
        return [0.06, 0.06, 0.00025]
    extra_margin = 0.025
    lower_name = str(object_name).lower()
    if "bowl" in lower_name:
        # Bowls have a narrow ring-like collision footprint. A slightly larger
        # invisible carrier prevents the ring from tipping through the patch.
        extra_margin = 0.055
    elif "mug" in lower_name:
        extra_margin = 0.04
    elif "basket" in lower_name:
        extra_margin = 0.04
    half_xy = np.max(np.asarray(radii, dtype=float), axis=0) + extra_margin
    half_xy = np.clip(half_xy, 0.06, 0.20)
    return [float(half_xy[0]), float(half_xy[1]), 0.00025]


def expand_platform_size_for_motion_path(size: list[float], target: dict[str, Any]) -> list[float]:
    return [float(x) for x in size]


def target_platform_xy_center(
    env: OffScreenRenderEnv,
    object_name: str,
    fallback_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Center the carrier under the object's bottom support patch, not its joint origin."""
    points = []
    for geom_name in target_contact_geom_names(env, object_name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        points.append(np.asarray(env.sim.data.geom_xpos[geom_id][:2], dtype=float))
    if points:
        return np.median(np.asarray(points, dtype=float), axis=0)
    if fallback_xy is not None:
        return np.asarray(fallback_xy, dtype=float)
    joint = object_joint(env, object_name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=float)
    return qpos[:2] if qpos.size >= 2 else np.zeros(2, dtype=float)


def object_joint_body_name(env: OffScreenRenderEnv, object_name: str) -> str | None:
    try:
        joint_id = int(env.sim.model.joint_name2id(object_joint(env, object_name)))
        body_id = int(env.sim.model.jnt_bodyid[joint_id])
        return env.sim.model.body_id2name(body_id)
    except Exception:
        return None


def world_to_body_local(env: OffScreenRenderEnv, body_name: str, world_xyz: np.ndarray) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    body_pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
    body_rot = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return body_rot.T @ (np.asarray(world_xyz, dtype=float) - body_pos)


def target_bottom_z(env: OffScreenRenderEnv, object_name: str) -> float | None:
    bottoms = []
    for geom_name in target_contact_geom_names(env, object_name):
        try:
            geom_id = int(env.sim.model.geom_name2id(geom_name))
        except Exception:
            continue
        geom_z = float(env.sim.data.geom_xpos[geom_id][2])
        geom_size_z = float(env.sim.model.geom_size[geom_id][2])
        bottoms.append(geom_z - geom_size_z)
    if not bottoms:
        return None
    return float(min(bottoms))


def stability_proxy_enabled(target: dict[str, Any], object_name: str) -> bool:
    if "stability_proxy_enabled" in target:
        return bool(target.get("stability_proxy_enabled"))
    # Backward compatibility for configs generated before the field was
    # renamed. Without this, thin box assets such as butter can miss all
    # platform contacts and fall through the support.
    if "contact_proxy_enabled" in target:
        return bool(target.get("contact_proxy_enabled"))
    return False


def stability_proxy_half_size(env: OffScreenRenderEnv, target: dict[str, Any], object_name: str) -> list[float]:
    if target.get("stability_proxy_size") not in (None, "auto"):
        values = [float(x) for x in target["stability_proxy_size"]]
        if len(values) == 2:
            values.append(float(target.get("stability_proxy_thickness_m", 0.0015)))
        return values[:3]
    if (
        target.get("contact_proxy_size_x_m") is not None
        and target.get("contact_proxy_size_y_m") is not None
    ):
        return [
            float(target.get("contact_proxy_size_x_m")),
            float(target.get("contact_proxy_size_y_m")),
            float(target.get("contact_proxy_size_z_m", target.get("stability_proxy_thickness_m", 0.0015))),
        ]
    if target.get("stability_proxy_size_xy_m") is not None:
        xy = [float(x) for x in target["stability_proxy_size_xy_m"]]
    else:
        platform_size = estimate_target_platform_size(env, object_name)
        xy = [
            float(np.clip(platform_size[0] * 0.75, 0.035, 0.075)),
            float(np.clip(platform_size[1] * 0.75, 0.035, 0.075)),
        ]
    return [xy[0], xy[1], float(target.get("stability_proxy_thickness_m", 0.0015))]


def add_stability_proxy_geom(
    root: ET.Element,
    env: OffScreenRenderEnv,
    target: dict[str, Any],
    object_name: str,
    proxy_idx: int,
) -> str | None:
    """Attach an invisible bottom footprint to the carried object.

    The geom has no normal collision mask. We only create explicit contact
    pairs between this proxy and the object's own moving platform, so the proxy
    improves conveyor support without making the object collide more broadly
    with the robot or unrelated scene objects.
    """
    if not stability_proxy_enabled(target, object_name):
        return None
    body_name = object_joint_body_name(env, object_name) or target_body_name(env, object_name)
    if not body_name:
        return None
    body_xml = root.find(f".//body[@name='{body_name}']")
    if body_xml is None:
        return None
    # Mesh/capsule-like LIBERO assets do not always expose a reliable bottom
    # plane through geom_size. Prefer the measured support contact height when
    # available so the invisible proxy is actually on the carrier surface.
    support_z = target.get("preinstall_support_contact_z")
    bottom_z = float(support_z) if support_z is not None else target_bottom_z(env, object_name)
    if bottom_z is None:
        return None
    joint = object_joint(env, object_name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=float)
    proxy_xy = target_platform_xy_center(env, object_name, fallback_xy=qpos[:2] if qpos.size >= 2 else None)
    half_size = stability_proxy_half_size(env, target, object_name)
    proxy_name = f"libero_dynamic_stability_proxy_{proxy_idx}_{object_name}"
    if root.find(f".//geom[@name='{proxy_name}']") is not None:
        return proxy_name
    world_center = np.asarray(
        [
            float(proxy_xy[0]),
            float(proxy_xy[1]),
            float(bottom_z + half_size[2] + float(target.get("stability_proxy_bottom_epsilon_m", 0.00015))),
        ],
        dtype=float,
    )
    local_center = world_to_body_local(env, body_name, world_center)
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


def install_preview_platform_bodies(env: OffScreenRenderEnv, targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Inject one driven free-joint conveyor patch per movable target.

    Each patch has collision masks disabled and uses explicit MJCF contact
    pairs with only its carried object's contact geoms. Other objects, robot
    links and scene fixtures do not collide with this patch at all.
    """
    placements = capture_body_placements(env)
    joint_states = capture_joint_states(env)
    geom_rgba = capture_geom_rgba(env)
    platforms = {}
    root = ET.fromstring(env_xml(env))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Cannot install preview moving platforms: MJCF has no worldbody")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    for idx, target in enumerate(targets):
        name = str(target.get("target_object_name") or target.get("object_name"))
        if is_fixture_body_translation_target(target) or is_fixture_body_yaw_target(target):
            continue
        joint = str(target.get("target_joint_name") or name)
        try:
            base_qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
        except Exception:
            base_qpos = None
        if is_articulated_target(target, base_qpos):
            continue
        body_name = f"libero_dynamic_preview_platform_{idx}_{name}"
        geom_name = f"{body_name}_geom"
        if "frypan" in name.lower():
            pair_geom_names = target_contact_geom_names(
                env,
                name,
                float(target.get("bottom_contact_margin_m", 0.002)),
            )
        elif bool(target.get("contact_all_target_geoms", False)):
            pair_geom_names = target_all_contact_geom_names(env, name)
        else:
            pair_geom_names = target_contact_geom_names(
                env,
                name,
                float(target.get("bottom_contact_margin_m", 0.002)),
            )
        target_bodies = target_body_names(env, name)
        target_body = target_bodies[0] if target_bodies else None
        if not pair_geom_names:
            continue
        if "size" in target and target["size"] not in (None, "auto"):
            size = [float(x) for x in target["size"]]
        else:
            size = estimate_target_platform_size(env, name)
        size = expand_platform_size_for_motion_path(size, target)
        if root.find(f".//body[@name='{body_name}']") is not None:
            continue
        use_mocap = bool(target.get("use_mocap_platform", False))
        friction = target.get("friction", [80.0, 2.0, 0.1])
        friction_text = " ".join(str(float(x)) for x in friction)
        solref = str(target.get("solref", "0.002 1"))
        solimp = str(target.get("solimp", "0.95 0.99 0.001"))
        margin = str(float(target.get("margin", 0.0001)))
        condim = str(int(target.get("condim", 6)))
        rgba_alpha = 0.35 if bool(target.get("visible", False)) else 0.0
        joint_names = {}
        if use_mocap:
            body = ET.SubElement(worldbody, "body", name=body_name, pos="0 0 0", mocap="true")
        else:
            joint_names = {
                "x": f"{body_name}_x",
                "y": f"{body_name}_y",
                "z": f"{body_name}_z",
            }
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
            rgba=f"1.0 0.0 1.0 {rgba_alpha}",
            friction=friction_text,
            solref=solref,
            solimp=solimp,
            margin=margin,
            condim=condim,
            contype="0",
            conaffinity="0",
            density="10000",
        )
        carrier_geom_names = [geom_name]
        use_rim = False
        rim_geom_names = []
        if use_rim:
            rim_height = float(target.get("stabilizer_rim_height_m", 0.025 if "moka_pot" in str(name) else 0.002))
            rim_thickness = float(target.get("stabilizer_rim_thickness_m", 0.004 if "moka_pot" in str(name) else 0.002))
            rim_z = float(size[2]) + rim_height
            rim_specs = [
                (f"{body_name}_rim_pos_x", [float(size[0]) + rim_thickness, 0.0, rim_z], [rim_thickness, float(size[1]), rim_height]),
                (f"{body_name}_rim_neg_x", [-float(size[0]) - rim_thickness, 0.0, rim_z], [rim_thickness, float(size[1]), rim_height]),
                (f"{body_name}_rim_pos_y", [0.0, float(size[1]) + rim_thickness, rim_z], [float(size[0]), rim_thickness, rim_height]),
                (f"{body_name}_rim_neg_y", [0.0, -float(size[1]) - rim_thickness, rim_z], [float(size[0]), rim_thickness, rim_height]),
            ]
            for rim_name, rim_pos, rim_size in rim_specs:
                ET.SubElement(
                    body,
                    "geom",
                    name=rim_name,
                    type="box",
                    pos=" ".join(str(float(x)) for x in rim_pos),
                    size=" ".join(str(float(x)) for x in rim_size),
                    rgba=f"0.1 0.55 1.0 {rgba_alpha}",
                    friction=friction_text,
                    solref=solref,
                    solimp=solimp,
                    margin=margin,
                    condim=condim,
                    contype="0",
                    conaffinity="0",
                    density="10000",
                )
                carrier_geom_names.append(rim_name)
                rim_geom_names.append(rim_name)
        proxy_geom_name = None
        platform_pair_geom_names = list(pair_geom_names)
        if proxy_geom_name:
            platform_pair_geom_names = sorted(set(platform_pair_geom_names + [proxy_geom_name]))
        for carrier_geom_name in carrier_geom_names:
            carrier_target_geom_names = (
                target_all_contact_geom_names(env, name)
                if carrier_geom_name in rim_geom_names
                else platform_pair_geom_names
            )
            for target_geom_name in carrier_target_geom_names:
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=carrier_geom_name,
                    geom2=target_geom_name,
                        condim=condim,
                        friction=friction_text,
                        solref=solref,
                        solimp=solimp,
                        margin=margin,
                )
        excluded_support_bodies = []
        excluded_platform_bodies = platform_non_target_exclude_body_names(env, body_name, target_bodies)
        for excluded_body_name in excluded_platform_bodies:
            ET.SubElement(
                contact,
                "exclude",
                body1=body_name,
                body2=excluded_body_name,
            )
        keep_static_support = "frypan" in str(name).lower()
        if bool(target.get("exclude_static_support_contact", False)) and not keep_static_support:
            excluded_support_bodies = support_exclude_body_names(env)
            for target_body_name_item in target_bodies:
                for support_body_name in excluded_support_bodies:
                    if support_body_name in target_bodies:
                        continue
                    ET.SubElement(
                        contact,
                        "exclude",
                        body1=target_body_name_item,
                        body2=support_body_name,
                    )
        if bool(target.get("exclude_support_fixture_contact", False)):
            for target_body_name_item in target_bodies:
                for support_body_name in sorted(set(target.get("support_fixture_body_names") or target.get("support_contact_bodies") or [])):
                    if not support_body_name or support_body_name in target_bodies:
                        continue
                    try:
                        env.sim.model.body_name2id(str(support_body_name))
                    except Exception:
                        continue
                    ET.SubElement(
                        contact,
                        "exclude",
                        body1=target_body_name_item,
                        body2=str(support_body_name),
                    )
        platforms[name] = {
            "body_name": body_name,
            "geom_name": geom_name,
            "carrier_geom_names": carrier_geom_names,
            "joint_name": None,
            "joint_names": joint_names,
            "kinematic_mode": "mocap" if use_mocap else "slide_joints",
            "pair_geom_names": pair_geom_names,
            "stability_proxy_geom_name": proxy_geom_name,
            "target_body_name": target_body,
            "target_body_names": target_bodies,
            "excluded_support_bodies": excluded_support_bodies,
            "excluded_platform_bodies": excluded_platform_bodies,
            "size": size,
            "surface_epsilon_m": float(target.get("surface_epsilon_m", 0.0)),
            "preinstall_support_contact_z": target.get("preinstall_support_contact_z"),
            "use_support_plane_z": bool(target.get("platform_use_support_plane_z", False)),
        }

    if platforms:
        env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
        restore_joint_states(env, joint_states)
        restore_geom_rgba(env, geom_rgba)
        restore_body_placements(env, placements)
    return platforms


def set_platform_state(env: OffScreenRenderEnv, platform: dict[str, Any], xyz: np.ndarray, velocity_xyz: np.ndarray) -> None:
    if platform.get("kinematic_mode") == "mocap":
        body_id = int(env.sim.model.body_name2id(platform["body_name"]))
        mocap_id = int(env.sim.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Preview platform body is not mocap: {platform['body_name']}")
        env.sim.data.mocap_pos[mocap_id] = np.asarray(xyz, dtype=float)
        env.sim.data.mocap_quat[mocap_id] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        env.sim.forward()
        return
    if platform.get("kinematic_mode") == "slide_joints":
        joint_names = platform["joint_names"]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            env.sim.data.set_joint_qpos(joint_names[axis], float(xyz[axis_idx]))
            env.sim.data.set_joint_qvel(joint_names[axis], float(velocity_xyz[axis_idx]))
        env.sim.forward()
        return

    joint_id = int(env.sim.model.joint_name2id(platform["joint_name"]))
    qpos_adr = int(env.sim.model.jnt_qposadr[joint_id])
    qvel_adr = int(env.sim.model.jnt_dofadr[joint_id])
    env.sim.data.qpos[qpos_adr : qpos_adr + 7] = np.asarray(
        [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    env.sim.data.qvel[qvel_adr : qvel_adr + 6] = np.asarray(
        [float(velocity_xyz[0]), float(velocity_xyz[1]), float(velocity_xyz[2]), 0.0, 0.0, 0.0],
        dtype=float,
    )
    env.sim.forward()


def set_platform_velocity(
    env: OffScreenRenderEnv,
    platform: dict[str, Any],
    velocity_xyz: np.ndarray,
    fixed_z: float | None = None,
) -> None:
    velocity_xyz = np.asarray(velocity_xyz, dtype=float)
    if platform.get("kinematic_mode") == "mocap":
        return
    if platform.get("kinematic_mode") == "slide_joints":
        joint_names = platform["joint_names"]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            if axis == "z" and fixed_z is not None:
                env.sim.data.set_joint_qpos(joint_names[axis], float(fixed_z))
                env.sim.data.set_joint_qvel(joint_names[axis], 0.0)
            else:
                env.sim.data.set_joint_qvel(joint_names[axis], float(velocity_xyz[axis_idx]))
        return

    joint_id = int(env.sim.model.joint_name2id(platform["joint_name"]))
    qvel_adr = int(env.sim.model.jnt_dofadr[joint_id])
    if fixed_z is not None:
        qpos_adr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qpos_adr + 2] = float(fixed_z)
    env.sim.data.qvel[qvel_adr : qvel_adr + 6] = np.asarray(
        [float(velocity_xyz[0]), float(velocity_xyz[1]), 0.0 if fixed_z is not None else float(velocity_xyz[2]), 0.0, 0.0, 0.0],
        dtype=float,
    )


def get_platform_state(env: OffScreenRenderEnv, platform: dict[str, Any]) -> np.ndarray | None:
    if platform.get("kinematic_mode") == "mocap":
        body_id = int(env.sim.model.body_name2id(platform["body_name"]))
        mocap_id = int(env.sim.model.body_mocapid[body_id])
        if mocap_id < 0:
            return None
        return np.asarray(env.sim.data.mocap_pos[mocap_id].copy(), dtype=float)
    if platform.get("kinematic_mode") == "slide_joints":
        joint_names = platform["joint_names"]
        return np.asarray(
            [float(env.sim.data.get_joint_qpos(joint_names[axis])) for axis in ("x", "y", "z")],
            dtype=float,
        )
    joint_id = int(env.sim.model.joint_name2id(platform["joint_name"]))
    qpos_adr = int(env.sim.model.jnt_qposadr[joint_id])
    return np.asarray(env.sim.data.qpos[qpos_adr : qpos_adr + 3].copy(), dtype=float)


def added_platform_joint_count(platforms: dict[str, dict[str, Any]]) -> int:
    return sum(1 for platform in platforms.values() if platform.get("kinematic_mode") in {"freejoint", "slide_joints"})


def initial_platform_pose(env: OffScreenRenderEnv, object_name: str, platform: dict[str, Any]) -> np.ndarray:
    joint = object_joint(env, object_name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=float)
    platform_xy = target_platform_xy_center(env, object_name, fallback_xy=qpos[:2] if qpos.size >= 2 else None)
    half_z = float(platform.get("size", [0.075, 0.075, 0.004])[2])
    surface_epsilon = float(platform.get("surface_epsilon_m", 0.002))
    if platform.get("use_support_plane_z", False):
        support_z = estimate_support_plane_z_under_target(env, object_name)
        if support_z is not None:
            z = float(support_z) - half_z + surface_epsilon
            return np.asarray([float(platform_xy[0]), float(platform_xy[1]), z], dtype=float)
    if platform.get("preinstall_support_contact_z") is not None:
        contact_z = float(platform["preinstall_support_contact_z"])
        z = contact_z - half_z + surface_epsilon
        return np.asarray([float(platform_xy[0]), float(platform_xy[1]), z], dtype=float)
    active, contacts = has_support_contact(env, object_name)
    if active and contacts:
        contact_z = float(np.median([c.get("pos", [0.0, 0.0, qpos[2]])[2] for c in contacts]))
        # Keep the carrier top nearly flush with the original support plane.
        # This makes the hidden conveyor transfer tangential motion without
        # visibly lifting the object.
        z = contact_z - half_z + surface_epsilon
    else:
        support_z = estimate_support_plane_z_under_target(env, object_name)
        if support_z is not None:
            return np.asarray([float(platform_xy[0]), float(platform_xy[1]), support_z - half_z + surface_epsilon], dtype=float)
        bottoms = []
        for geom_name in target_contact_geom_names(env, object_name):
            try:
                geom_id = int(env.sim.model.geom_name2id(geom_name))
            except Exception:
                continue
            geom_z = float(env.sim.data.geom_xpos[geom_id][2])
            geom_size_z = float(env.sim.model.geom_size[geom_id][2])
            bottoms.append(geom_z - geom_size_z)
        z = (min(bottoms) - half_z + surface_epsilon) if bottoms else float(qpos[2]) - 0.07
    return np.asarray([float(platform_xy[0]), float(platform_xy[1]), z], dtype=float)


def has_platform_contact(env: OffScreenRenderEnv, object_name: str, platform: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    target_ids = set(target_geom_ids(env, object_name))
    proxy_geom_name = platform.get("stability_proxy_geom_name")
    if proxy_geom_name:
        try:
            target_ids.add(int(env.sim.model.geom_name2id(str(proxy_geom_name))))
        except Exception:
            pass
    if not target_ids:
        return False, []
    platform_geom_ids = set()
    for carrier_geom_name in platform.get("carrier_geom_names", [platform.get("geom_name")]):
        if not carrier_geom_name:
            continue
        try:
            platform_geom_ids.add(int(env.sim.model.geom_name2id(carrier_geom_name)))
        except Exception:
            continue
    if not platform_geom_ids:
        return False, []
    contacts = []
    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in platform_geom_ids and geom2 in target_ids:
            target_geom = geom2
            platform_geom = geom1
        elif geom2 in platform_geom_ids and geom1 in target_ids:
            target_geom = geom1
            platform_geom = geom2
        else:
            continue
        contacts.append(
            {
                "target_geom": geom_name(env, target_geom),
                "platform_geom": geom_name(env, platform_geom),
                "distance": float(contact.dist),
                "pos": np.asarray(contact.pos, dtype=float).tolist(),
            }
        )
    return bool(contacts), contacts


def motion_velocity(motion: dict[str, Any], elapsed: float, dt: float) -> np.ndarray:
    current = movable_offset(motion, elapsed)
    nxt = movable_offset(motion, elapsed + dt)
    return (nxt - current) / max(float(dt), 1e-9)


def prepare_target_states(
    env: OffScreenRenderEnv,
    targets: list[dict[str, Any]],
    platforms: dict[str, dict[str, Any]] | None = None,
    carrier_type: str = "physical_conveyor_patch",
) -> list[dict[str, Any]]:
    states = []
    platforms = platforms or {}
    for target in targets:
        name = str(target.get("target_object_name") or target.get("object_name"))
        motion = target.get("motion", {}) or {}
        is_body_yaw = is_fixture_body_yaw_target(target)
        is_body_translation = is_fixture_body_translation_target(target)
        body_id = None
        body_name = None
        base_body_pos = None
        base_body_quat = None
        if is_body_yaw or is_body_translation:
            candidates = [
                target.get("target_body_name"),
                target.get("body_name"),
                f"{name}_main",
                name,
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    body_id = int(env.sim.model.body_name2id(str(candidate)))
                    body_name = str(candidate)
                    break
                except Exception:
                    continue
            if body_id is None:
                for idx in range(int(env.sim.model.nbody)):
                    candidate = env.sim.model.body_id2name(idx)
                    if candidate and str(candidate).startswith(f"{name}_"):
                        body_id = int(idx)
                        body_name = str(candidate)
                        break
            if body_id is None or body_name is None:
                raise KeyError(f"Could not find fixture body for dynamic target: {name}")
            joint = str(target.get("target_joint_name") or name)
            base_qpos = np.asarray([0.0], dtype=float)
            base_body_pos = np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy()
            base_body_quat = np.asarray(env.sim.model.body_quat[body_id], dtype=float).copy()
            is_joint = True
        else:
            joint = str(target.get("target_joint_name") or object_joint(env, name))
            base_qpos = np.asarray(env.sim.data.get_joint_qpos(joint).copy(), dtype=float)
            is_joint = is_articulated_target(target, base_qpos)
        platform = platforms.get(name)
        platform_base_pos = initial_platform_pose(env, name, platform) if platform is not None and not is_joint else None
        if platform_base_pos is not None:
            set_platform_state(
                env,
                platform,
                platform_base_pos + movable_offset(motion, 0.0),
                np.zeros(3, dtype=float),
            )
        states.append(
            {
                "name": name,
                "joint": joint,
                "motion": motion,
                "base_qpos": base_qpos,
                "carrier_type": carrier_type,
                "surface_velocity_tracking_kp": float(target.get("surface_velocity_tracking_kp", 6.0)),
                "surface_velocity_max_mps": float(target.get("surface_velocity_max_mps", 0.18)),
                "platform_tracking_gain": float(target.get("platform_tracking_gain", 4.0)),
                "platform_tracking_max_m": float(target.get("platform_tracking_max_m", 0.0)),
                "platform_velocity_max_mps": float(target.get("platform_velocity_max_mps", 0.12)),
                "collision_penetration_threshold_m": float(target.get("collision_penetration_threshold_m", 1e-4)),
                "intended_base_xy": base_qpos[:2].copy() if base_qpos.size >= 2 else None,
                "is_joint": is_joint,
                "is_body_yaw": is_body_yaw,
                "is_body_translation": is_body_translation,
                "body_name": body_name,
                "body_id": body_id,
                "base_body_pos": base_body_pos,
                "base_body_quat": base_body_quat,
                "platform": platform,
                "target_body_name": target_body_name(env, name),
                "platform_base_pos": platform_base_pos,
                "object_base_xy": base_qpos[:2].copy() if base_qpos.size >= 2 else None,
                "object_base_z": float(base_qpos[2]) if base_qpos.size >= 3 else 0.0,
                "formal_start_z": target.get("formal_start_z"),
                "formal_start_bottom_z": target.get("formal_start_bottom_z"),
                "object_to_platform_xy": (
                    base_qpos[:2].copy() - platform_base_pos[:2]
                    if platform_base_pos is not None and base_qpos.size >= 2
                    else None
                ),
                "last_applied_offset": movable_offset(motion, 0.0),
                "contact_events": 0,
                "joint_events": 0,
                "last_contacts": [],
                "collision_events": [],
                "collision_keys": set(),
                "agentview_samples": 0,
                "agentview_offscreen_samples": 0,
                "agentview_edge_samples": 0,
            }
        )
    env.sim.forward()
    return states


def realign_targets_to_intended_xy(env: OffScreenRenderEnv, states: list[dict[str, Any]]) -> None:
    """Undo tiny XY drift introduced while swapping support contact to platforms.

    Manual plans are drawn in object/world XY, not platform-center XY. The
    invisible carrier is only a physical support mechanism, so the object's
    post-settle XY should still begin at the reviewed waypoint start.
    """
    for state in states:
        if state["is_joint"]:
            continue
        intended_xy = state.get("intended_base_xy")
        if intended_xy is None:
            continue
        qpos = np.asarray(env.sim.data.get_joint_qpos(state["joint"]).copy(), dtype=float)
        if qpos.size < 2:
            continue
        qpos[:2] = np.asarray(intended_xy, dtype=float)[:2]
        env.sim.data.set_joint_qpos(state["joint"], qpos)
        try:
            qvel = np.asarray(env.sim.data.get_joint_qvel(state["joint"]).copy(), dtype=float)
            qvel[:] = 0.0
            env.sim.data.set_joint_qvel(state["joint"], qvel)
        except Exception:
            pass
    env.sim.forward()


def refresh_target_state_anchors(env: OffScreenRenderEnv, states: list[dict[str, Any]]) -> None:
    """Refresh object/platform anchor offsets after support replacement settles."""
    for state in states:
        if state["is_joint"]:
            continue
        qpos = np.asarray(env.sim.data.get_joint_qpos(state["joint"]).copy(), dtype=float)
        if qpos.size < 2:
            continue
        platform = state.get("platform")
        platform_pos = get_platform_state(env, platform) if platform is not None else None
        state["base_qpos"] = qpos.copy()
        state["object_base_xy"] = qpos[:2].copy()
        state["object_base_z"] = float(qpos[2]) if qpos.size >= 3 else float(state.get("object_base_z", 0.0))
        if platform_pos is not None:
            state["platform_base_pos"] = platform_pos.copy()
            state["object_to_platform_xy"] = qpos[:2].copy() - platform_pos[:2]


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(-1)
    if quat.size < 4:
        return np.eye(3)
    w, x, y, z = quat[:4]
    norm = max(float(np.linalg.norm([w, x, y, z])), 1e-9)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def relative_tilt_deg(base_qpos: np.ndarray, qpos: np.ndarray) -> float:
    base_qpos = np.asarray(base_qpos, dtype=float)
    qpos = np.asarray(qpos, dtype=float)
    if base_qpos.size < 7 or qpos.size < 7:
        return 0.0
    base_up = quat_wxyz_to_rotmat(base_qpos[3:7]) @ np.asarray([0.0, 0.0, 1.0])
    cur_up = quat_wxyz_to_rotmat(qpos[3:7]) @ np.asarray([0.0, 0.0, 1.0])
    denom = max(float(np.linalg.norm(base_up) * np.linalg.norm(cur_up)), 1e-9)
    cosang = float(np.clip(np.dot(base_up, cur_up) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def update_motion_diagnostics(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float) -> None:
    if state.get("is_joint"):
        return
    try:
        qpos = np.asarray(env.sim.data.get_joint_qpos(state["joint"]).copy(), dtype=float)
    except Exception:
        return
    if qpos.size < 3:
        return
    object_base_xy = state.get("object_base_xy")
    if object_base_xy is not None:
        offset = movable_offset(state["motion"], elapsed)
        desired_xy = np.asarray(object_base_xy, dtype=float)[:2] + offset[:2]
        xy_error = float(np.linalg.norm(qpos[:2] - desired_xy))
        state["trace_max_xy_error_m"] = max(float(state.get("trace_max_xy_error_m", 0.0)), xy_error)
        desired_z = float(state.get("object_base_z", qpos[2]))
        state.setdefault("desired_trace_xyz", []).append([float(desired_xy[0]), float(desired_xy[1]), desired_z])
        platform = state.get("platform")
        platform_base_pos = state.get("platform_base_pos")
        object_to_platform_xy = state.get("object_to_platform_xy")
        if platform is not None and platform_base_pos is not None and object_to_platform_xy is not None:
            desired_platform_xy = desired_xy - np.asarray(object_to_platform_xy, dtype=float)[:2]
            platform_pos = get_platform_state(env, platform)
            if platform_pos is not None:
                platform_xy = np.asarray(platform_pos, dtype=float)[:2]
                platform_path_error = float(np.linalg.norm(platform_xy - desired_platform_xy))
                object_to_platform_error = float(
                    np.linalg.norm((qpos[:2] - platform_xy) - np.asarray(object_to_platform_xy, dtype=float)[:2])
                )
                state["trace_max_platform_path_error_m"] = max(
                    float(state.get("trace_max_platform_path_error_m", 0.0)),
                    platform_path_error,
                )
                state["trace_max_object_to_platform_error_m"] = max(
                    float(state.get("trace_max_object_to_platform_error_m", 0.0)),
                    object_to_platform_error,
                )
    z_probe, bottom_z = target_z_and_bottom(env, state)
    if z_probe is None:
        z_probe = float(qpos[2])
    if bottom_z is None:
        bottom_z = float(z_probe)
    state.setdefault("actual_trace_xyz", []).append([float(qpos[0]), float(qpos[1]), float(z_probe)])
    state["trace_min_z"] = min(float(state.get("trace_min_z", z_probe)), float(z_probe))
    state["trace_max_z"] = max(float(state.get("trace_max_z", z_probe)), float(z_probe))
    state["trace_min_bottom_z"] = min(float(state.get("trace_min_bottom_z", bottom_z)), float(bottom_z))
    state["trace_max_bottom_z"] = max(float(state.get("trace_max_bottom_z", bottom_z)), float(bottom_z))
    tilt = relative_tilt_deg(np.asarray(state.get("base_qpos", qpos), dtype=float), qpos)
    state["trace_max_tilt_deg"] = max(float(state.get("trace_max_tilt_deg", 0.0)), tilt)
    state["trace_samples"] = int(state.get("trace_samples", 0)) + 1


def target_z_and_bottom(env: OffScreenRenderEnv, state: dict[str, Any]) -> tuple[float | None, float | None]:
    z_probe = None
    target_body = state.get("target_body_name")
    if target_body:
        try:
            body_id = int(env.sim.model.body_name2id(str(target_body)))
            z_probe = float(env.sim.data.body_xpos[body_id][2])
        except Exception:
            pass
    bottoms = []
    try:
        for geom_id in target_geom_ids(env, state["name"]):
            geom_id = int(geom_id)
            if int(env.sim.model.geom_contype[geom_id]) == 0:
                continue
            geom_z = float(env.sim.data.geom_xpos[geom_id][2])
            geom_size_z = float(env.sim.model.geom_size[geom_id][2])
            bottoms.append(geom_z - geom_size_z)
    except Exception:
        pass
    bottom_z = min(bottoms) if bottoms else z_probe
    return z_probe, bottom_z


def record_motion_baselines(env: OffScreenRenderEnv, states: list[dict[str, Any]]) -> None:
    """Record the stable, formal-start height after settling and platform setup."""
    for state in states:
        if state.get("is_joint"):
            continue
        z_probe, bottom_z = target_z_and_bottom(env, state)
        if z_probe is not None:
            state["trace_base_z"] = float(z_probe)
        if bottom_z is not None:
            state["trace_base_bottom_z"] = float(bottom_z)


def set_articulated_joint_preview(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float) -> None:
    base = state["base_qpos"]
    motion = state["motion"]
    if np.asarray(base).ndim == 0:
        set_joint_qpos(env, state["joint"], float(base) + joint_offset(motion, elapsed))
        state["joint_events"] += 1
        return
    qpos = np.asarray(base, dtype=float).copy()
    qpos[0] = float(np.asarray(base).reshape(-1)[0]) + joint_offset(motion, elapsed)
    set_joint_qpos(env, state["joint"], qpos)
    state["joint_events"] += 1


def set_fixture_body_yaw_preview(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float) -> None:
    body_id = state.get("body_id")
    base_quat = state.get("base_body_quat")
    if body_id is None or base_quat is None:
        return
    yaw = yaw_motion_offset(state["motion"], elapsed)
    env.sim.model.body_quat[int(body_id)] = quat_multiply_wxyz(yaw_quat_wxyz(yaw), np.asarray(base_quat, dtype=float))
    env.sim.forward()
    state["joint_events"] += 1


def set_fixture_body_translation_preview(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float) -> None:
    body_id = state.get("body_id")
    base_pos = state.get("base_body_pos")
    if body_id is None or base_pos is None:
        return
    env.sim.model.body_pos[int(body_id)] = np.asarray(base_pos, dtype=float) + movable_offset(state["motion"], elapsed)
    env.sim.forward()
    state["joint_events"] += 1


def set_direct_qpos_preview(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float) -> None:
    """Kinematic preview mode: force the free object center onto the path.

    This is only for debugging annotation/projection quality. It intentionally
    bypasses contact-driven transport, so it should not be used as the final
    physics benchmark behavior.
    """
    if state["is_joint"]:
        return
    base_qpos = np.asarray(state.get("base_qpos"), dtype=float)
    if base_qpos.size < 3:
        return
    qpos = base_qpos.copy()
    offset = movable_offset(state["motion"], elapsed)
    qpos[: min(3, qpos.size)] = base_qpos[: min(3, qpos.size)] + offset[: min(3, qpos.size)]
    set_joint_qpos(env, state["joint"], qpos)
    try:
        qvel = np.asarray(env.sim.data.get_joint_qvel(state["joint"]).copy(), dtype=float)
        qvel[:] = 0.0
        env.sim.data.set_joint_qvel(state["joint"], qvel)
    except Exception:
        pass
    state["joint_events"] += 1


def move_physical_platform_preview(env: OffScreenRenderEnv, state: dict[str, Any], elapsed: float, dt: float) -> None:
    platform = state.get("platform")
    base_pos = state.get("platform_base_pos")
    if platform is None or base_pos is None:
        return

    offset = movable_offset(state["motion"], elapsed)
    velocity = motion_velocity(state["motion"], elapsed, dt)
    velocity[2] = 0.0
    target_pos = np.asarray(base_pos, dtype=float) + offset
    object_base_xy = state.get("object_base_xy")
    object_to_platform_xy = state.get("object_to_platform_xy")
    if object_base_xy is not None and object_to_platform_xy is not None:
        target_pos[:2] = (
            np.asarray(object_base_xy, dtype=float)[:2]
            + offset[:2]
            - np.asarray(object_to_platform_xy, dtype=float)[:2]
        )

    # The conveyor/platform is the prescribed moving support. It must follow
    # the reviewed path exactly; object lag should be exposed as a physics
    # follow error, not hidden by moving the platform off-trajectory.
    set_platform_state(env, platform, target_pos, velocity)
    active, contacts = has_platform_contact(env, state["name"], platform)
    state["last_contacts"] = contacts[:8]
    if active:
        state["contact_events"] += 1


def apply_surface_velocity_preview(
    env: OffScreenRenderEnv,
    state: dict[str, Any],
    elapsed: float,
    dt: float,
) -> None:
    """Apply a conveyor-like horizontal velocity without overwriting position.

    This is intentionally weaker than directly setting qpos: contacts with the
    scene and robot remain part of MuJoCo's solve, so invalid paths show up as
    physical collisions instead of silently tunneling through geometry.
    """
    if state["is_joint"]:
        return
    base_qpos = np.asarray(state.get("base_qpos"), dtype=float)
    if base_qpos.size < 3:
        return
    active, contacts = has_support_contact(env, state["name"])
    state["last_contacts"] = contacts[:8]
    if not active:
        return
    qpos = np.asarray(env.sim.data.get_joint_qpos(state["joint"]).copy(), dtype=float)
    if qpos.size < 3:
        return
    desired_xy = base_qpos[:2] + movable_offset(state["motion"], elapsed)[:2]
    feedforward = motion_velocity(state["motion"], elapsed, dt)[:2]
    tracking_kp = float(state.get("surface_velocity_tracking_kp", 6.0))
    max_speed = float(state.get("surface_velocity_max_mps", 0.18))
    velocity_xy = feedforward + tracking_kp * (desired_xy - qpos[:2])
    speed = float(np.linalg.norm(velocity_xy))
    if speed > max_speed:
        velocity_xy = velocity_xy * (max_speed / max(speed, 1e-9))
    try:
        qvel = np.asarray(env.sim.data.get_joint_qvel(state["joint"]).copy(), dtype=float)
        qvel[:2] = velocity_xy
        if qvel.size >= 3:
            qvel[2] = min(float(qvel[2]), 0.0)
        if qvel.size > 3:
            qvel[3:] = 0.0
        env.sim.data.set_joint_qvel(state["joint"], qvel)
        state["contact_events"] += 1
    except Exception:
        pass


def advance_physical_dynamic_time(
    env: OffScreenRenderEnv,
    states: list[dict[str, Any]],
    elapsed: float,
    duration: float,
    on_substep: Callable[[float], None] | None = None,
) -> float:
    """Advance only MuJoCo physics for an exact wall-clock duration.

    This is used for synchronous policy latency. The robot controller does not
    receive an extra 20 Hz action; callers may wrap ``sim.step`` to hold robot
    joints while moving fixtures, carriers, and free objects continuously.
    """
    requested = max(0.0, float(duration))
    if requested <= 0.0:
        return 0.0

    sim = env.sim
    base_dt = float(getattr(sim.model.opt, "timestep", 0.002) or 0.002)
    full_steps = int(np.floor((requested + 1e-12) / base_dt))
    residual = requested - full_steps * base_dt
    if residual < 1e-10:
        residual = 0.0
    substeps = [base_dt] * full_steps
    if residual > 0.0:
        substeps.append(residual)

    advanced = 0.0
    try:
        for sub_dt in substeps:
            sub_elapsed = float(elapsed) + advanced
            sim.model.opt.timestep = float(sub_dt)
            for state in states:
                if state.get("is_body_yaw"):
                    set_fixture_body_yaw_preview(env, state, sub_elapsed)
                elif state.get("is_body_translation"):
                    set_fixture_body_translation_preview(env, state, sub_elapsed)
                elif state["is_joint"]:
                    set_articulated_joint_preview(env, state, sub_elapsed)
                elif state.get("carrier_type") == "direct_qpos":
                    set_direct_qpos_preview(env, state, sub_elapsed)
                elif state.get("carrier_type") == "surface_velocity":
                    apply_surface_velocity_preview(env, state, sub_elapsed, sub_dt)
                else:
                    move_physical_platform_preview(env, state, sub_elapsed, sub_dt)
            sim.step()
            advanced += float(sub_dt)
            if on_substep is not None:
                on_substep(float(advanced))
    finally:
        sim.model.opt.timestep = base_dt

    final_elapsed = float(elapsed) + advanced
    for state in states:
        if state.get("is_body_yaw"):
            set_fixture_body_yaw_preview(env, state, final_elapsed)
        elif state.get("is_body_translation"):
            set_fixture_body_translation_preview(env, state, final_elapsed)
        elif state["is_joint"]:
            set_articulated_joint_preview(env, state, final_elapsed)
        elif state.get("carrier_type") == "direct_qpos":
            set_direct_qpos_preview(env, state, final_elapsed)
    sim.forward()
    return float(advanced)


def step_with_physical_dynamic(
    env: OffScreenRenderEnv,
    action: np.ndarray,
    states: list[dict[str, Any]],
    elapsed: float,
    dt: float,
    on_substep: Callable[[float], None] | None = None,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    for state in states:
        if state.get("is_body_yaw"):
            set_fixture_body_yaw_preview(env, state, elapsed)
        elif state.get("is_body_translation"):
            set_fixture_body_translation_preview(env, state, elapsed)
        elif state["is_joint"]:
            set_articulated_joint_preview(env, state, elapsed)

    sim = env.sim
    original_step = sim.step

    physics_dt = float(getattr(env.sim.model.opt, "timestep", dt) or dt)
    hook_count = 0

    def _step_with_surface_velocity(*args, **kwargs):
        nonlocal hook_count
        sub_elapsed = float(elapsed) + hook_count * physics_dt
        for state in states:
            if state.get("is_body_yaw"):
                set_fixture_body_yaw_preview(env, state, sub_elapsed)
            elif state.get("is_body_translation"):
                set_fixture_body_translation_preview(env, state, sub_elapsed)
            elif state["is_joint"]:
                set_articulated_joint_preview(env, state, sub_elapsed)
            else:
                if state.get("carrier_type") == "direct_qpos":
                    set_direct_qpos_preview(env, state, sub_elapsed)
                elif state.get("carrier_type") == "surface_velocity":
                    apply_surface_velocity_preview(env, state, sub_elapsed, physics_dt)
                else:
                    move_physical_platform_preview(env, state, sub_elapsed, physics_dt)
        result = original_step(*args, **kwargs)
        hook_count += 1
        if on_substep is not None:
            on_substep(float(hook_count) * physics_dt)
        return result

    sim.step = _step_with_surface_velocity
    try:
        if bool(getattr(env.env, "done", False)):
            # Preview videos are not policy rollouts; keep simulating the
            # dynamic scene even if LIBERO's task success checker terminates.
            env.env.done = False
        obs, reward, done, info = env.step(action)
        if any(state.get("carrier_type") == "direct_qpos" for state in states):
            for state in states:
                if state.get("carrier_type") == "direct_qpos":
                    set_direct_qpos_preview(env, state, float(elapsed) + float(dt))
            env.sim.forward()
            try:
                obs = env.env._get_observations()
            except Exception:
                pass
        return obs, reward, done, info
    finally:
        sim.step = original_step


def step_with_platform_hold(
    env: OffScreenRenderEnv,
    action: np.ndarray,
    states: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    sim = env.sim
    original_step = sim.step

    def _step_with_platform_hold(*args, **kwargs):
        for state in states:
            platform = state.get("platform")
            base_pos = state.get("platform_base_pos")
            if platform is not None and base_pos is not None:
                set_platform_state(env, platform, np.asarray(base_pos, dtype=float), np.zeros(3, dtype=float))
        return original_step(*args, **kwargs)

    sim.step = _step_with_platform_hold
    try:
        return env.step(action)
    finally:
        sim.step = original_step


def render_one(item: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    suite, bddl = task_bddl(item["suite"], item["task_id"])
    task = suite.get_task(item["task_id"])
    init_states = suite.get_task_init_states(item["task_id"])
    control_freq = int(round(1.0 / float(args.sim_dt)))
    if control_freq <= 0 or not np.isclose(1.0 / float(control_freq), float(args.sim_dt), atol=1e-9):
        raise ValueError("--sim-dt must be the reciprocal of a positive integer control frequency")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        control_freq=control_freq,
    )
    targets = apply_speed_override(target_entries_for_preview(item["task_cfg"]), args.speed_mps)
    targets = apply_waypoint_timing_override(
        targets,
        args.waypoint_stop_seconds,
        args.waypoint_smooth_stop,
        args.waypoint_accel_mps2,
    )
    platforms: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    writer = None
    video_frame_count = 0
    next_video_sample_s = 1.0 / float(args.fps)
    started = time.time()
    video_path = output_dir / item["level"] / item["suite"] / f"task{item['task_id']:02d}__{item['variant_name']}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env.reset()
        task_cfg = apply_carrier_override(item["task_cfg"], args.carrier_override)
        carrier_type = carrier_type_for_task(task_cfg)
        targets = apply_carrier_settings_to_targets(targets, task_cfg.get("carrier", {}) or {})
        action = zero_action(env)
        use_physical_platforms = carrier_type in {"physical_conveyor_patch", "physical_platform", "freejoint_platform"}
        settle_on_platform = bool(
            (task_cfg.get("carrier", {}) or {}).get("settle_on_platform", False)
            or task_cfg.get("settle_on_platform", False)
        )
        if use_physical_platforms:
            obs = env.set_init_state(init_states[args.init_state_id])
            apply_initial_adjustments(env, task_cfg)
            targets = apply_static_support_contact_override(
                targets,
                keep_static_support_contact=bool(args.keep_static_support_contact),
                static_support_clearance_m=float(args.static_support_clearance_m),
            )
            for target in targets:
                name = str(target.get("target_object_name") or target.get("object_name"))
                try:
                    base_qpos = np.asarray(
                        env.sim.data.get_joint_qpos(str(target.get("target_joint_name") or object_joint(env, name))).copy(),
                        dtype=float,
                    )
                except Exception:
                    base_qpos = None
                if is_articulated_target(target, base_qpos):
                    continue
                align_target_bottom_to_support_plane(env, name)
            # LIBERO initial states often spawn objects slightly above their
            # final support surface. Usually we settle first, then install
            # conveyor patches at the stable height. Some reviewed starts,
            # however, intentionally place the dynamic object over a caddy /
            # void that only the hidden conveyor should support. For those
            # cases, keep the object's own platform in place during settling
            # so the object does not fall before the formal preview begins.
            if settle_on_platform:
                platforms = install_preview_platform_bodies(env, targets)
                states = prepare_target_states(env, targets, platforms, carrier_type=carrier_type)
                for _ in range(max(0, int(args.settle_steps))):
                    obs, _, _, _ = step_with_platform_hold(env, action, states)
            else:
                for _ in range(max(0, int(args.settle_steps))):
                    obs, _, _, _ = env.step(action)
            env.sim.forward()
            if not settle_on_platform:
                targets = append_fixture_supported_passenger_targets(env, targets)
                targets = apply_carrier_settings_to_targets(targets, task_cfg.get("carrier", {}) or {})
                targets = apply_static_support_contact_override(
                    targets,
                    keep_static_support_contact=bool(args.keep_static_support_contact),
                    static_support_clearance_m=float(args.static_support_clearance_m),
                )
            for target in targets:
                name = str(target.get("target_object_name") or target.get("object_name"))
                try:
                    base_qpos = np.asarray(
                        env.sim.data.get_joint_qpos(str(target.get("target_joint_name") or object_joint(env, name))).copy(),
                        dtype=float,
                    )
                except Exception:
                    base_qpos = None
                if is_articulated_target(target, base_qpos):
                    continue
                active, contacts = has_support_contact(env, name)
                if active and contacts:
                    target["preinstall_support_contact_z"] = float(
                        np.median([c.get("pos", [0.0, 0.0, 0.9])[2] for c in contacts])
                    )
                try:
                    formal_qpos = np.asarray(env.sim.data.get_joint_qpos(object_joint(env, name)).copy(), dtype=float)
                    if formal_qpos.size >= 3:
                        target["formal_start_z"] = float(formal_qpos[2])
                    formal_bottom = target_bottom_z(env, name)
                    if formal_bottom is not None:
                        target["formal_start_bottom_z"] = float(formal_bottom)
                except Exception:
                    pass
        else:
            obs = set_init_state_allowing_extra_platform_joints(
                env,
                init_states[args.init_state_id],
                added_platforms=0,
                added_platform_qpos=3,
                added_platform_qvel=3,
            )
        if use_physical_platforms:
            if not platforms:
                platforms = install_preview_platform_bodies(env, targets)
            try:
                obs = env.env._get_observations()
            except Exception:
                pass
        else:
            apply_initial_adjustments(env, task_cfg)
            for _ in range(max(0, args.settle_steps)):
                obs, _, _, _ = env.step(action)
            platforms = {}

        if not states:
            states = prepare_target_states(env, targets, platforms, carrier_type=carrier_type)
        for _ in range(max(0, int(args.platform_settle_steps))):
            obs, _, _, _ = step_with_platform_hold(env, action, states)
        if bool(args.realign_after_platform_settle) or use_physical_platforms:
            realign_targets_to_intended_xy(env, states)
            refresh_target_state_anchors(env, states)
            for state in states:
                platform = state.get("platform")
                base_pos = state.get("platform_base_pos")
                if platform is not None and base_pos is not None:
                    set_platform_state(env, platform, np.asarray(base_pos, dtype=float), np.zeros(3, dtype=float))
            env.sim.forward()
        refresh_target_state_anchors(env, states)
        record_motion_baselines(env, states)
        sim_dt = float(args.sim_dt)
        duration_seconds = float(args.duration_seconds)
        if args.cycles is not None:
            periods = [
                motion_period_seconds(state["motion"])
                for state in states
                if motion_period_seconds(state["motion"]) is not None
            ]
            if periods:
                duration_seconds = float(args.cycles) * max(periods)
        total_steps = int(math.ceil(duration_seconds / sim_dt))

        if not bool(getattr(args, "validate_only", False)):
            writer = imageio.get_writer(video_path, fps=args.fps, macro_block_size=None)
        for step_idx in range(total_steps):
            elapsed = step_idx * sim_dt

            def _record_video_substep(advanced_s: float) -> None:
                nonlocal next_video_sample_s, video_frame_count
                if writer is None:
                    return
                absolute_time_s = float(elapsed) + float(advanced_s)
                if absolute_time_s + 1e-9 < next_video_sample_s:
                    return
                writer.append_data(
                    render_frame(
                        env,
                        None,
                        args.camera_size,
                        states,
                        overlay_traces=not args.no_overlay_traces,
                    )
                )
                video_frame_count += 1
                next_video_sample_s += 1.0 / float(args.fps)

            obs, _, _, _ = step_with_physical_dynamic(
                env,
                action,
                states,
                elapsed,
                sim_dt,
                on_substep=_record_video_substep,
            )
            for state in states:
                update_motion_diagnostics(env, state, elapsed)
                update_safety_diagnostics(env, state, args.camera_size)
        if writer is not None:
            writer.close()
        status = "ok"
        error = None
    except Exception as exc:
        status = "error"
        error = repr(exc)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
    finally:
        env.close()
    target_modes = {
        state["name"]: (
            "fixture_body_yaw_preview"
            if state.get("is_body_yaw")
            else "fixture_body_translation_preview"
            if state.get("is_body_translation")
            else
            "articulated_joint_qpos_preview"
            if state["is_joint"]
            else "surface_velocity_conveyor"
            if state.get("carrier_type") == "surface_velocity"
            else "physical_conveyor_patch_contact"
        )
        for state in states
    }
    preview_modes = sorted(set(target_modes.values()))
    return {
        "status": status,
        "error": error,
        "level": item["level"],
        "suite": item["suite"],
        "task_id": item["task_id"],
        "variant_name": item["variant_name"],
        "variant_id": item["variant_id"],
        "language": task.language,
        "target_names": [state["name"] for state in states],
        "preview_physics": {
            "mode": preview_modes[0] if len(preview_modes) == 1 else "mixed",
            "keep_static_support_contact": bool(args.keep_static_support_contact),
            "static_support_clearance_m": float(args.static_support_clearance_m),
            "target_modes": target_modes,
            "sim_dt": float(args.sim_dt),
            "contact_events": {state["name"]: int(state["contact_events"]) for state in states},
            "joint_events": {state["name"]: int(state["joint_events"]) for state in states},
            "platforms": {
                state["name"]: state["platform"]
                for state in states
                if state.get("platform") is not None
            },
            "last_contacts": {state["name"]: state["last_contacts"] for state in states},
            "motion_diagnostics": {
                state["name"]: {
                    "max_xy_error_m": round(float(state.get("trace_max_xy_error_m", 0.0)), 6),
                    "max_platform_path_error_m": round(float(state.get("trace_max_platform_path_error_m", 0.0)), 6),
                    "max_object_to_platform_error_m": round(float(state.get("trace_max_object_to_platform_error_m", 0.0)), 6),
                    "max_tilt_deg": round(float(state.get("trace_max_tilt_deg", 0.0)), 3),
                    "min_z": round(float(state.get("trace_min_z", 0.0)), 6),
                    "max_z": round(float(state.get("trace_max_z", 0.0)), 6),
                    "base_z": round(float(state.get("trace_base_z", state.get("object_base_z", 0.0))), 6),
                    "formal_start_z": (
                        round(float(state["formal_start_z"]), 6)
                        if state.get("formal_start_z") is not None
                        else None
                    ),
                    "height_delta_from_formal_start_m": (
                        round(
                            float(state.get("trace_base_z", state.get("object_base_z", 0.0)))
                            - float(state["formal_start_z"]),
                            6,
                        )
                        if state.get("formal_start_z") is not None
                        else None
                    ),
                    "min_bottom_z": round(float(state.get("trace_min_bottom_z", 0.0)), 6),
                    "max_bottom_z": round(float(state.get("trace_max_bottom_z", 0.0)), 6),
                    "base_bottom_z": round(float(state.get("trace_base_bottom_z", 0.0)), 6),
                    "formal_start_bottom_z": (
                        round(float(state["formal_start_bottom_z"]), 6)
                        if state.get("formal_start_bottom_z") is not None
                        else None
                    ),
                    "bottom_delta_from_formal_start_m": (
                        round(
                            float(state.get("trace_base_bottom_z", 0.0))
                            - float(state["formal_start_bottom_z"]),
                            6,
                        )
                        if state.get("formal_start_bottom_z") is not None
                        else None
                    ),
                    "samples": int(state.get("trace_samples", 0)),
                }
                for state in states
            },
            "safety_diagnostics": {
                state["name"]: {
                    "collision_count": len(state.get("collision_events", [])),
                    "collision_events": state.get("collision_events", []),
                    "birdview_samples": int(state.get("birdview_samples", 0)),
                    "birdview_offscreen_samples": int(state.get("birdview_offscreen_samples", 0)),
                    "birdview_edge_samples": int(state.get("birdview_edge_samples", 0)),
                    "birdview_visible_ratio": round(
                        1.0
                        - (
                            float(state.get("birdview_offscreen_samples", 0))
                            / max(1.0, float(state.get("birdview_samples", 0)))
                        ),
                        4,
                    ),
                    "agentview_samples": int(state.get("agentview_samples", 0)),
                    "agentview_offscreen_samples": int(state.get("agentview_offscreen_samples", 0)),
                    "agentview_edge_samples": int(state.get("agentview_edge_samples", 0)),
                    "agentview_visible_ratio": round(
                        1.0
                        - (
                            float(state.get("agentview_offscreen_samples", 0))
                            / max(1.0, float(state.get("agentview_samples", 0)))
                        ),
                        4,
                    ),
                }
                for state in states
            },
        },
        "video": None if bool(getattr(args, "validate_only", False)) else str(video_path),
        "video_fps": int(args.fps),
        "video_sampling_mode": "simulation_time_physics_substeps",
        "video_frame_count": int(video_frame_count),
        "video_duration_s": (
            0.0 if int(args.fps) <= 0 else round(float(video_frame_count) / float(args.fps), 6)
        ),
        "duration_seconds": round(duration_seconds if status == "ok" else float(args.duration_seconds), 3),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--cycles", type=float, default=None)
    parser.add_argument("--speed-mps", type=float, default=None)
    parser.add_argument("--waypoint-stop-seconds", type=float, default=None)
    parser.add_argument("--waypoint-smooth-stop", action="store_true")
    parser.add_argument("--waypoint-accel-mps2", type=float, default=None)
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Saved simulation-time video rate. Default: 60; independent of --sim-dt.",
    )
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--no-overlay-traces", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Run physics diagnostics without writing preview mp4 files.")
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--platform-settle-steps", type=int, default=80)
    parser.add_argument("--realign-after-platform-settle", action="store_true")
    parser.add_argument(
        "--keep-static-support-contact",
        action="store_true",
        help=(
            "Keep target-vs-table/cabinet/floor contacts enabled while the "
            "hidden conveyor patch drives the target. This preserves tabletop "
            "collision/support behavior without switching away from the "
            "physical conveyor carrier."
        ),
    )
    parser.add_argument(
        "--static-support-clearance-m",
        type=float,
        default=0.0,
        help=(
            "When --keep-static-support-contact is enabled, lift the hidden "
            "conveyor surface by this tiny clearance so the conveyor remains "
            "the primary drive while the table/cabinet/floor contact stays "
            "available as a fallback if the object tips or loses support."
        ),
    )
    parser.add_argument(
        "--carrier-override",
        choices=["none", "direct_qpos", "surface_velocity", "physical_conveyor_patch"],
        default=None,
        help="Override carrier mode for debugging. direct_qpos is kinematic and not a physics benchmark mode.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    # Keep the env's built-in dynamic wrapper disabled; this script applies the
    # reviewed motion directly so multi-target activation variants can be shown.
    os.environ["LIBERO_DYNAMIC_CONFIG"] = "/tmp/libero_dynamic_preview_disabled.yaml"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = preview_items(args.config_dir, args.levels)
    selected = items[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    manifest = {
        "config_dir": str(args.config_dir),
        "output_dir": str(args.output_dir),
        "total_planned_all_levels": len(items),
        "start_index": args.start_index,
        "selected_count": len(selected),
        "duration_seconds": args.duration_seconds,
        "fps": args.fps,
        "camera_size": args.camera_size,
        "keep_static_support_contact": bool(args.keep_static_support_contact),
        "static_support_clearance_m": float(args.static_support_clearance_m),
        "records": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    for idx, item in enumerate(selected, start=args.start_index):
        print(f"[{idx + 1}/{len(items)}] {item['level']} {item['suite']} task{item['task_id']:02d} {item['variant_name']}", flush=True)
        record = render_one(item, args, args.output_dir)
        record["index"] = idx
        manifest["records"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if record["status"] != "ok":
            print(f"  ERROR: {record['error']}", flush=True)
        else:
            print(f"  saved {record['video']} ({record['elapsed_seconds']}s)", flush=True)
    print(json.dumps({"manifest": str(manifest_path), "selected_count": len(selected), "total": len(items)}, indent=2), flush=True)
    return 0 if all(r["status"] == "ok" for r in manifest["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
