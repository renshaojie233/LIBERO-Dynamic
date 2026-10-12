# Multi-GPU Pi0.5 Official-Control Setup

Run these commands from the LIBERO-Dynamic repository root. Choose local paths
for the evaluator environment, the LeRobot policy-server environment, the
Pi0.5 checkpoint, and the PaliGemma tokenizer:

```bash
export REPO=$PWD
export PY=/path/to/libero-dynamic-env/bin/python
export SERVER_PY=/path/to/lerobot-env/bin/python
export CLEAN_CHECKPOINT=/path/to/pi05_libero_pytorch
export RTC_CHECKPOINT=$CLEAN_CHECKPOINT
export PALIGEMMA_LOCAL_TOKENIZER=/path/to/paligemma_tokenizer
```

Use only idle GPUs unless the operator says otherwise. For a single-GPU smoke
run, keep all defaults. For four-GPU throughput, set:

```bash
export PORTS="8124 8125 8126 8127"
export GPUS="0 1 2 3"
export LIBERO_EGL_DEVICE_IDS="0,1,2,3"
export POLICY_SERVER_COUNT=4
export NUM_WORKERS=4
```

## Control Entrypoints

Each entrypoint is one self-contained scheduling strategy. They share the same
LIBERO-Dynamic environment and reviewed old-physics route.

```bash
python scripts/run_pi05_sync_eval.py   ...  # synchronous hold baseline
python scripts/run_pi05_async_eval.py  ...  # LeRobot-style weighted overlap async
python scripts/run_pi05_rtc_eval.py    ...  # delay-aware RTC-style queue update
```

Do not use `naive_async_queue` for official claims. It is only an append-only
baseline for interpreting older runs.

## Speed Sweep

The three speed conditions patch `alpha_mps` in copied dynamic config files:

```text
speed_003 -> 0.03 m/s
speed_006 -> 0.06 m/s
speed_009 -> 0.09 m/s
```

Run the three 1800-item sweeps independently, one strategy at a time:

```bash
bash scripts/launch_pi05_sync_speed_sweep_20260606.sh
bash scripts/launch_pi05_async_speed_sweep_20260606.sh
bash scripts/launch_pi05_rtc_speed_sweep_20260606.sh
```

Each launcher performs one smoke item first, then runs:

```text
3 speeds x 600 items = 1800 items
```

The current output roots are written to:

```text
/tmp/libero_latest_pi05_sync_speed_sweep_root
/tmp/libero_latest_pi05_async_speed_sweep_root
/tmp/libero_latest_pi05_rtc_speed_sweep_root
```

For audit videos, use compact dual model-input video:

```bash
--save-videos \
--no-save-review-video \
--save-model-input-videos \
--model-input-video-layout dual \
--model-input-video-size 128 \
--video-fps 60 \
--record-latency-frames
```

## Policy Service

When four idle GPUs are available, start one LeRobot Pi0.5 websocket adapter
per GPU. The adapter speaks the same OpenPI websocket protocol as the eval
runtime, but internally uses LeRobot's official Pi0.5 preprocessor,
`predict_action_chunk`, optional RTC conditioning, and postprocessor.

The runtime maps workers to ports with `--policy-server-count 4`.

```bash
SERVER_PY=/path/to/lerobot-env/bin/python
CKPT=/path/to/pi05_libero_pytorch
TOKENIZER=/path/to/paligemma_tokenizer

CUDA_VISIBLE_DEVICES=0 $SERVER_PY scripts/serve_lerobot_pi05_openpi_ws.py --port 8124 --checkpoint "$CKPT" --local-tokenizer "$TOKENIZER"
CUDA_VISIBLE_DEVICES=1 $SERVER_PY scripts/serve_lerobot_pi05_openpi_ws.py --port 8125 --checkpoint "$CKPT" --local-tokenizer "$TOKENIZER"
CUDA_VISIBLE_DEVICES=2 $SERVER_PY scripts/serve_lerobot_pi05_openpi_ws.py --port 8126 --checkpoint "$CKPT" --local-tokenizer "$TOKENIZER"
CUDA_VISIBLE_DEVICES=3 $SERVER_PY scripts/serve_lerobot_pi05_openpi_ws.py --port 8127 --checkpoint "$CKPT" --local-tokenizer "$TOKENIZER"
```

Or use the launcher:

```bash
CHECKPOINT=/path/to/pi05_libero_pytorch \
PALIGEMMA_LOCAL_TOKENIZER=/path/to/paligemma_tokenizer \
bash scripts/launch_lerobot_pi05_ws_servers_20260606.sh
```

For RTC-conditioned policy runs, pass `--enable-rtc`. If your RTC implementation
uses a separate checkpoint, set `RTC_CHECKPOINT`; otherwise use the same
LeRobot Pi0.5 checkpoint.

```bash
RTC_CKPT=/path/to/pi05_libero_pytorch
CUDA_VISIBLE_DEVICES=0 $SERVER_PY scripts/serve_lerobot_pi05_openpi_ws.py --port 8124 --checkpoint "$RTC_CKPT" --local-tokenizer "$TOKENIZER" --enable-rtc

CHECKPOINT="$RTC_CKPT" \
PALIGEMMA_LOCAL_TOKENIZER="$TOKENIZER" \
ENABLE_RTC=1 \
bash scripts/launch_lerobot_pi05_ws_servers_20260606.sh
```

The simulated latency is decoupled from server wall-clock throughput by passing
`--latency-source fixed --calibrated-latency-seconds ...` through the launchers.
This lets several environments share policy servers without letting queueing
noise change the simulated world dynamics.
