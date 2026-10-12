#!/usr/bin/env python3
"""Serve an official VLASH Pi0.5 checkpoint over the OpenPI websocket API."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import time
import traceback
from typing import Any, TypedDict

import msgpack
import numpy as np
import torch
import transformers.utils
import websockets


# Some compatible transformers releases do not export this typing-only helper.
try:
    from transformers.utils import LossKwargs  # noqa: F401
except ImportError:
    class LossKwargs(TypedDict, total=False):
        pass

    transformers.utils.LossKwargs = LossKwargs

import vlash.configs  # noqa: E402,F401 - registers VLASH policy configs
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from vlash.policies.factory import get_policy_class  # noqa: E402


OFFICIAL_ACTIONS_PER_CHUNK = 5


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def _image_tensor(value: Any, device: str) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW image, got shape {array.shape}")
    tensor = torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected a three-channel image, got shape {tuple(tensor.shape)}")
    if tensor.numel() and float(tensor.max().item()) > 1.5:
        tensor = tensor / 255.0
    return tensor.contiguous().unsqueeze(0)


class VLASHOpenPIWS:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.checkpoint = checkpoint.expanduser().resolve()
        self.device = str(device)
        config = PreTrainedConfig.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
        )
        config.device = self.device
        config.n_action_steps = OFFICIAL_ACTIONS_PER_CHUNK
        config.pretrained_path = str(self.checkpoint)
        if hasattr(config, "compile_model"):
            config.compile_model = False
        if hasattr(config, "gradient_checkpointing"):
            config.gradient_checkpointing = False

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(
            str(self.checkpoint),
            config=config,
            local_files_only=True,
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.config = config

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "vlash_pi05",
            "checkpoint": str(self.checkpoint),
            "actions_per_chunk": OFFICIAL_ACTIONS_PER_CHUNK,
            "device": self.device,
            "num_inference_steps": getattr(self.config, "num_inference_steps", None),
            "rtc_conditioning_supported": False,
            "required_observation_history_frames": 1,
            "required_action_history_steps": 0,
            "websocket_connection_resets_policy": True,
            "vlash_state_semantics": "current_state_in_language_prompt",
            "vlash_visual_delay_semantics": "provided_by_client_observation",
            "vlash_official_libero_chunk_size": OFFICIAL_ACTIONS_PER_CHUNK,
            "vlash_official_libero_protocol": "delayed_visual_current_state",
            "vlash_chunk_merge": "intact_no_weighting",
        }

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = torch.as_tensor(
            np.array(observation["observation/state"], dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        batch = {
            "observation.images.image": _image_tensor(
                observation["observation/image"], self.device
            ),
            "observation.images.wrist_image": _image_tensor(
                observation["observation/wrist_image"], self.device
            ),
            "observation.state": state,
            "task": str(observation.get("prompt", "")),
        }

        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        actions = (
            chunk[0, :OFFICIAL_ACTIONS_PER_CHUNK, :]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        if actions.shape[0] != OFFICIAL_ACTIONS_PER_CHUNK or not np.isfinite(actions).all():
            raise ValueError(f"Invalid VLASH action chunk: shape={actions.shape}")
        return {
            "actions": actions,
            "original_actions": np.array(actions, copy=True),
        }


class WebsocketServer:
    def __init__(self, policy: VLASHOpenPIWS, host: str, port: int) -> None:
        self.policy = policy
        self.host = str(host)
        self.port = int(port)
        self.packer = msgpack.Packer(default=_pack_array)
        self.connection_lock = asyncio.Lock()

    async def handler(self, websocket: Any) -> None:
        # A policy instance belongs to exactly one evaluator connection. This
        # lock prevents a second client from resetting or interleaving state.
        async with self.connection_lock:
            self.policy.reset()
            await websocket.send(self.packer.pack(self.policy.metadata))
            previous_total_s = None
            while True:
                try:
                    request_started = time.monotonic()
                    observation = msgpack.unpackb(
                        await websocket.recv(),
                        object_hook=_unpack_array,
                    )
                    inference_started = time.monotonic()
                    result = self.policy.infer(observation)
                    inference_s = time.monotonic() - inference_started
                    result["server_timing"] = {"infer_ms": inference_s * 1000.0}
                    if previous_total_s is not None:
                        result["server_timing"]["prev_total_ms"] = previous_total_s * 1000.0
                    await websocket.send(self.packer.pack(result))
                    previous_total_s = time.monotonic() - request_started
                except websockets.ConnectionClosed:
                    break
                except Exception:
                    logging.exception("VLASH inference failed")
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=1011, reason="Internal server error")
                    raise

    async def run(self) -> None:
        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ):
            await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18240)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"VLASH config.json was not found in {checkpoint}")
    policy = VLASHOpenPIWS(checkpoint, args.device)
    logging.info("Serving VLASH policy: %s", policy.metadata)
    asyncio.run(WebsocketServer(policy, args.host, args.port).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
