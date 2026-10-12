#!/usr/bin/env python3
"""Serve FastWAM-Joint through the OpenPI websocket protocol.

This adapter intentionally reuses FastWAM's LIBERO evaluation preprocessing and
postprocessing so realtime LIBERO-Dynamic can call it as an ordinary OpenPI
websocket policy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class FastWAMJointOpenPIWS:
    def __init__(
        self,
        *,
        fastwam_root: Path,
        checkpoint: Path,
        dataset_stats: Path,
        device: str,
        actions_per_chunk: int,
        num_inference_steps: int,
        seed: int | None,
    ) -> None:
        self.fastwam_root = fastwam_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.dataset_stats = dataset_stats.resolve()
        self.device = str(device)
        self.actions_per_chunk = int(actions_per_chunk)
        self.num_inference_steps = int(num_inference_steps)
        if self.num_inference_steps <= 0:
            raise ValueError("--num-inference-steps must be positive")

        for path in (self.fastwam_root, self.fastwam_root / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from experiments.libero.eval_libero_single import (  # pylint: disable=import-outside-toplevel
            _load_model_checkpoint,
            _mixed_precision_to_model_dtype,
            _predict_action_chunk,
        )
        from fastwam.datasets.lerobot.utils.normalizer import (  # pylint: disable=import-outside-toplevel
            load_dataset_stats_from_json,
        )

        self._predict_action_chunk = _predict_action_chunk
        config_dir = self.fastwam_root / "configs"
        seed_override = "seed=null" if seed is None else f"seed={int(seed)}"
        overrides = [
            "task=libero_joint_2cam224_1e-4",
            "model=fastwam_joint",
            f"ckpt={self.checkpoint}",
            f"EVALUATION.dataset_stats_path={self.dataset_stats}",
            f"EVALUATION.num_inference_steps={self.num_inference_steps}",
            f"eval_num_inference_steps={self.num_inference_steps}",
            f"EVALUATION.replan_steps={self.actions_per_chunk}",
            "EVALUATION.visualize_future_video=false",
            "EVALUATION.binarize_gripper=true",
            f"EVALUATION.device={self.device}",
            seed_override,
        ]
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            self.cfg = compose(config_name="sim_libero.yaml", overrides=overrides)

        model_dtype = _mixed_precision_to_model_dtype(self.cfg.get("mixed_precision", "bf16"))
        self.model = instantiate(self.cfg.model, model_dtype=model_dtype, device=self.device)
        _load_model_checkpoint(self.model, str(self.checkpoint))
        self.model = self.model.to(self.device).eval()

        dataset_stats_obj = load_dataset_stats_from_json(str(self.dataset_stats))
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(dataset_stats_obj)

        action_horizon_cfg = self.cfg.EVALUATION.get("action_horizon", None)
        self.action_horizon = int(self.cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
        video_size = self.cfg.data.train.get("video_size", [224, 448])
        self.input_h = int(video_size[0])
        self.input_w = int(video_size[1])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "fastwam_joint",
            "checkpoint": str(self.checkpoint),
            "dataset_stats": str(self.dataset_stats),
            "fastwam_root": str(self.fastwam_root),
            "device": self.device,
            "actions_per_chunk": self.actions_per_chunk,
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "rtc_conditioning_supported": False,
            "image_orientation_fix": "undo_openpi_180deg_before_fastwam_official_preprocess",
            "config_summary": {
                "model_target": str(self.cfg.model.get("_target_", "")),
                "task": "libero_joint_2cam224_1e-4",
                "video_size": list(self.cfg.data.train.get("video_size", [])),
                "num_frames": int(self.cfg.data.train.num_frames),
                "replan_steps": int(self.cfg.EVALUATION.replan_steps),
                "seed": None if self.cfg.get("seed") is None else int(self.cfg.seed),
                "binarize_gripper": bool(self.cfg.EVALUATION.get("binarize_gripper", False)),
            },
        }

    def reset(self) -> None:
        return None

    @staticmethod
    def _axisangle_to_quat(axisangle: np.ndarray) -> np.ndarray:
        vec = np.asarray(axisangle, dtype=np.float32).reshape(3)
        angle = float(np.linalg.norm(vec))
        if angle < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = vec / angle
        half = angle * 0.5
        return np.asarray(
            [axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)],
            dtype=np.float32,
        )

    def _obs_from_openpi(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape[-1] < 8:
            raise ValueError(f"FastWAM expects 8-D LIBERO state, got shape {state.shape}")
        gripper_qpos = np.asarray(state[6:8], dtype=np.float32)

        # The generic OpenPI websocket client rotates LIBERO camera images by
        # 180 degrees before sending them. FastWAM's official LIBERO
        # preprocessing also performs that rotation. Undo the OpenPI-side
        # rotation here so FastWAM sees the same raw camera orientation as its
        # native eval path, then lets its own preprocessing rotate once.
        agentview_image = np.ascontiguousarray(np.asarray(obs["observation/image"])[::-1, ::-1])
        wrist_image = np.ascontiguousarray(np.asarray(obs["observation/wrist_image"])[::-1, ::-1])
        return {
            "agentview_image": agentview_image,
            "robot0_eye_in_hand_image": wrist_image,
            "robot0_eef_pos": np.asarray(state[:3], dtype=np.float32),
            "robot0_eef_quat": self._axisangle_to_quat(state[3:6]),
            "robot0_gripper_qpos": gripper_qpos,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        task = str(obs.get("prompt", ""))
        fastwam_obs = self._obs_from_openpi(obs)
        with torch.inference_mode():
            actions, _imgs, _future = self._predict_action_chunk(
                obs=fastwam_obs,
                task_description=task,
                model=self.model,
                processor=self.processor,
                cfg=self.cfg,
                action_horizon=self.action_horizon,
                input_w=self.input_w,
                input_h=self.input_h,
                model_device=self.device,
            )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return {"actions": actions[: self.actions_per_chunk], "original_actions": actions[: self.actions_per_chunk]}


class WebsocketServer:
    def __init__(self, policy: FastWAMJointOpenPIWS, host: str, port: int) -> None:
        self.policy = policy
        self.host = str(host)
        self.port = int(port)
        self.packer = msgpack.Packer(default=_pack_array)

    async def handler(self, websocket: Any) -> None:
        await websocket.send(self.packer.pack(self.policy.metadata))
        while True:
            try:
                obs = msgpack.unpackb(await websocket.recv(), object_hook=_unpack_array)
                if isinstance(obs, dict) and obs.get("reset"):
                    self.policy.reset()
                    await websocket.send(self.packer.pack({"ok": True}))
                    continue
                started = time.monotonic()
                result = self.policy.infer(obs)
                result["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
                await websocket.send(self.packer.pack(result))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    async def run(self) -> None:
        async with websockets.serve(self.handler, self.host, self.port, compression=None, max_size=None):
            await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    policy = FastWAMJointOpenPIWS(
        fastwam_root=args.fastwam_root,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(policy.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("serving %s", {k: v for k, v in policy.metadata.items() if k != "config"})
    asyncio.run(WebsocketServer(policy, args.host, args.port).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
