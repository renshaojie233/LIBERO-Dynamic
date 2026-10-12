#!/usr/bin/env python3
"""Serve a LeRobot Pi0.5 checkpoint through the OpenPI websocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets

from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import tokenizer_processor
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig


PALIGEMMA_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _force_local_tokenizer(local_tokenizer: str | None) -> None:
    if not local_tokenizer:
        raise ValueError(
            "--local-tokenizer or PALIGEMMA_LOCAL_TOKENIZER is required. "
            "Use a local PaliGemma tokenizer directory to keep evaluation offline and reproducible."
        )
    local_path = Path(local_tokenizer)
    if not local_path.exists():
        raise FileNotFoundError(f"Local PaliGemma tokenizer directory does not exist: {local_path}")
    original_post_init = tokenizer_processor.TokenizerProcessorStep.__post_init__

    def patched_post_init(self: Any) -> None:
        if self.tokenizer_name == PALIGEMMA_TOKENIZER_NAME:
            self.tokenizer_name = str(local_path)
        return original_post_init(self)

    tokenizer_processor.TokenizerProcessorStep.__post_init__ = patched_post_init


def _as_image(value: Any) -> torch.Tensor:
    array = np.array(value, copy=True)
    if array.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape {array.shape}")
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.numel() and float(tensor.max().item()) > 1.5:
        tensor = tensor / 255.0
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous()


def _as_state(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), dtype=torch.float32)


class LeRobotPi05OpenPIWS:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        actions_per_chunk: int,
        enable_rtc: bool,
        rtc_execution_horizon: int | None,
        rtc_max_guidance_weight: float | None,
        local_tokenizer: str | None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self.actions_per_chunk = int(actions_per_chunk)
        _force_local_tokenizer(local_tokenizer)
        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        config.device = self.device
        if hasattr(config, "compile_model"):
            config.compile_model = False
        if hasattr(config, "gradient_checkpointing"):
            config.gradient_checkpointing = False
        if enable_rtc:
            rtc_kwargs: dict[str, Any] = {
                "enabled": True,
                "execution_horizon": int(rtc_execution_horizon or self.actions_per_chunk),
            }
            if rtc_max_guidance_weight is not None:
                rtc_kwargs["max_guidance_weight"] = float(rtc_max_guidance_weight)
            config.rtc_config = RTCConfig(**rtc_kwargs)
        self.policy = PI05Policy.from_pretrained(
            self.checkpoint,
            config=config,
            local_files_only=True,
            strict=False,
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.enable_rtc = bool(enable_rtc)
        self.rtc_config = config.rtc_config if self.enable_rtc else None
        if self.enable_rtc and hasattr(self.policy, "init_rtc_processor"):
            self.policy.init_rtc_processor()
        self.rtc_conditioning_supported = bool(self.enable_rtc and self.policy._rtc_enabled())

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "lerobot_pi05",
            "checkpoint": self.checkpoint,
            "actions_per_chunk": self.actions_per_chunk,
            "device": self.device,
            "rtc_conditioning_supported": self.rtc_conditioning_supported,
            "rtc_config": None
            if self.rtc_config is None
            else {
                "enabled": bool(self.rtc_config.enabled),
                "execution_horizon": int(self.rtc_config.execution_horizon),
                "max_guidance_weight": float(self.rtc_config.max_guidance_weight),
                "prefix_attention_schedule": str(self.rtc_config.prefix_attention_schedule),
            },
        }

    def _batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation.images.image": _as_image(obs["observation/image"]),
            "observation.images.image2": _as_image(obs["observation/wrist_image"]),
            "observation.state": _as_state(obs["observation/state"]),
            "task": str(obs.get("prompt", "")),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        batch = self.preprocessor(self._batch(obs))
        kwargs: dict[str, Any] = {}
        if self.rtc_conditioning_supported:
            leftover = np.array(obs.get("prev_chunk_left_over", []), dtype=np.float32, copy=True)
            kwargs["prev_chunk_left_over"] = (
                torch.as_tensor(leftover, dtype=torch.float32, device=self.device).unsqueeze(0)
                if leftover.size
                else None
            )
            kwargs["inference_delay"] = int(obs.get("inference_delay_steps", 0))

        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch, **kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]
        original_actions = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)

        processed = []
        for idx in range(chunk.shape[1]):
            processed.append(self.postprocessor(chunk[:, idx, :]))
        actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return {"actions": actions, "original_actions": original_actions}


class WebsocketServer:
    def __init__(self, policy: LeRobotPi05OpenPIWS, host: str, port: int) -> None:
        self.policy = policy
        self.host = str(host)
        self.port = int(port)
        self.packer = msgpack.Packer(default=_pack_array)

    async def handler(self, websocket: Any) -> None:
        await websocket.send(self.packer.pack(self.policy.metadata))
        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack.unpackb(await websocket.recv(), object_hook=_unpack_array)
                infer_start = time.monotonic()
                result = self.policy.infer(obs)
                infer_s = time.monotonic() - infer_start
                result["server_timing"] = {"infer_ms": infer_s * 1000.0}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(self.packer.pack(result))
                prev_total_time = time.monotonic() - start
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=10)
    parser.add_argument("--enable-rtc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rtc-execution-horizon", type=int, default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
    parser.add_argument("--local-tokenizer", default=os.environ.get("PALIGEMMA_LOCAL_TOKENIZER"))
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    policy = LeRobotPi05OpenPIWS(
        checkpoint=args.checkpoint,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        enable_rtc=bool(args.enable_rtc),
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        local_tokenizer=args.local_tokenizer,
    )
    logging.info("serving %s", policy.metadata)
    asyncio.run(WebsocketServer(policy, args.host, args.port).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
