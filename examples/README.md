# Evaluation examples

## 0. Policy-free simulation

`render_pure_simulation.py` renders the reviewed dynamic scene without loading
a policy or checkpoint. It is the required first validation step for a new
installation.

```bash
MUJOCO_GL=egl python examples/render_pure_simulation.py \
  --output-dir outputs/pure_simulation_smoke \
  --speeds 0.03 0.06 0.09 \
  --limit-per-speed 4 \
  --shard-size 4 \
  --workers 3
```

The command launches each renderer shard in a fresh subprocess, writes a
resumable manifest, materializes a canonical `final_videos/` tree, evaluates
the physical QA diagnostics, and checks MP4 files with `ffprobe` when it is
available. Use `--limit-per-speed 600` for the complete 1800-video sweep.
Policy-free completion confirms scene and rendering health; it is not a robot
task-success metric.

## Model evaluation

These examples launch one policy server and evaluate at least one complete
LIBERO-Dynamic episode. An episode stops on success or after the full 600-step
horizon. Each run saves a dual-view model-input video, per-step traces, the
manifest, and the policy-server log. Review/runtime videos are disabled by
default in these examples to reduce rendering and I/O overhead.

All paths are command-line arguments; no workstation-specific path is embedded
in the examples. Run `python examples/evaluate_<model>.py --help` for the full
interface. Run the commands below from the LIBERO-Dynamic repository root.
For reproducible results, check out the same LIBERO-Dynamic commit or tag used
by the reported evaluation, for example `formal-eval-final-20260622`.

If the model environment intentionally omits a pure-Python runtime dependency,
add its site-packages directory with a repeatable `--server-pythonpath` option.

The default generic latency is zero for a functional smoke test. These examples
do not automatically calibrate latency before each run. For reported results,
measure latency separately with one policy server on an otherwise idle GPU and
pass the chosen statistic through `--calibrated-latency-seconds`. VLASH instead
requires `--measured-latency-seconds`, which is represented once as delayed
visual observations; its generic latency must remain zero.

## Prerequisites checklist

Each example assumes the following are already available:

- a LIBERO-Dynamic evaluator environment that can import this repository,
  construct LIBERO tasks, render videos, and connect to an OpenPI-compatible
  websocket policy server; install `websockets` and `msgpack-numpy` here if
  they are not already present;
- the selected upstream policy repository cloned locally and installed in its
  own Python environment;
- local model assets for that policy; this repository does not download or
  redistribute checkpoints;
- one free CUDA GPU for the policy server and, if desired, a separate GPU or
  process slot for the simulator;
- write access to the chosen `--output-dir`, where dual-view model-input
  videos, traces, manifests, and logs are saved.

## 1. Pi0.5 JAX

Project: [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi).
Clone it with submodules and use its documented `uv sync` environment:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install websockets msgpack-numpy
```

Use the official `pi05_libero` checkpoint or a compatible fine-tuned JAX
checkpoint. The example starts openpi's `scripts/serve_policy.py` and evaluates
through the websocket client:

```bash
python examples/evaluate_pi05_jax.py \
  --openpi-root /path/to/openpi \
  --server-python /path/to/openpi/.venv/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/pi05_libero \
  --output-dir outputs/pi05-jax-smoke \
  --gpu 0
```

## 2. Pi0.5 PyTorch

Project: [Hugging Face LeRobot](https://github.com/huggingface/lerobot). Use a
LeRobot revision that provides `lerobot.policies.pi05.modeling_pi05.PI05Policy`.
The validated path used a LeRobot 0.5.x-compatible Pi0.5 policy stack; current
LeRobot `main` may require Python `>=3.12`, so check its `pyproject.toml`
before creating the environment.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi]"
pip install websockets msgpack-numpy huggingface_hub transformers sentencepiece protobuf tiktoken
```

The server environment must provide a CUDA-compatible PyTorch build. The
evaluator additionally uses openpi's lightweight websocket client, supplied by
`--openpi-root`; it does not load the JAX model in this PyTorch path.
`sentencepiece` and `protobuf` are required to read a local PaliGemma
SentencePiece tokenizer. Installing `tiktoken` as well avoids a second fallback
error path in some `transformers` versions.

The checkpoint directory must contain the LeRobot policy config, model weights,
preprocessor, and postprocessor files. Pass a local PaliGemma tokenizer directory
to avoid an implicit network download:

```text
pi05_libero_pytorch/
  config.json
  model.safetensors
  policy_preprocessor_step_2_normalizer_processor.safetensors
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
paligemma_tokenizer/
  ... tokenizer files ...
```

```bash
python examples/evaluate_pi05_pytorch.py \
  --openpi-root /path/to/openpi \
  --server-python /path/to/lerobot-env/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/pi05_libero_pytorch \
  --local-tokenizer /path/to/paligemma_tokenizer \
  --output-dir outputs/pi05-pytorch-smoke \
  --gpu 0
```

The synchronous, async, and RTC examples use the same checkpoint and tokenizer
arguments. They differ only in the evaluator runtime:

- `evaluate_pi05_pytorch.py`: synchronous hold baseline (`sync_hold`);
- `evaluate_pi05_pytorch_async.py`: LeRobot-style async chunk queue
  (`async_queue`);
- `evaluate_pi05_pytorch_rtc.py`: delay-aware RTC queue replacement
  (`rtc_queue`).

The RTC example passes `--enable-rtc` to the LeRobot policy server and exercises
both model-side RTC conditioning and evaluator-side RTC queue replacement.
Require `rtc_conditioning_supported=True` in the server log and manifests;
otherwise the run is not valid RTC evidence.

```bash
python examples/evaluate_pi05_pytorch_async.py \
  --openpi-root /path/to/openpi \
  --server-python /path/to/lerobot-env/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/pi05_libero_pytorch \
  --local-tokenizer /path/to/paligemma_tokenizer \
  --output-dir outputs/pi05-pytorch-async-smoke \
  --gpu 0

python examples/evaluate_pi05_pytorch_rtc.py \
  --openpi-root /path/to/openpi \
  --server-python /path/to/lerobot-env/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/pi05_libero_pytorch \
  --local-tokenizer /path/to/paligemma_tokenizer \
  --output-dir outputs/pi05-pytorch-rtc-smoke \
  --gpu 0
```

For formal comparisons, keep checkpoint, tokenizer, task range, latency
calibration, and GPU conditions fixed across all three modes. Use distinct
ports if multiple examples run at the same time.

## 3. FastWAM-Joint

Project: [FastWAM](https://github.com/yuantianyuan01/FastWAM). Follow its
environment setup and Wan model-preparation instructions. A joint checkpoint
must be paired with the `dataset_stats.json` used by that checkpoint. Point
`--wan-root` at the model base directory that FastWAM normally exposes as
`DIFFSYNTH_MODEL_BASE_PATH`. Do not point `--wan-root` at the joint checkpoint
directory: first-time FastWAM preparation downloads Wan/T5 assets under this
root, and a clean setup can use roughly 30 GB there.

The public FastWAM release can provide common upstream assets, but the exact
FastWAM-Joint 1-step numbers require the exact joint checkpoint and matching
`dataset_stats.json`. Keep those two files in their own checkpoint directory,
for example:

```text
fastwam-joint/
  weights/
    step_034720.pt
    dataset_stats.json
fastwam-wan/
  ... Wan/T5 model cache used by DIFFSYNTH_MODEL_BASE_PATH ...
```

```bash
git clone https://github.com/yuantianyuan01/FastWAM.git
cd FastWAM
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -e .
pip install websockets msgpack-numpy huggingface_hub future matplotlib
# If FastWAM's LIBERO helpers are not already satisfied by its installer:
pip install robosuite==1.4.0 bddl==1.0.1 mujoco==3.3.2 gym==0.25.2 \
  cloudpickle==2.1.0 easydict==1.9 opencv-python==4.6.0.66
```

Before running the FastWAM-Joint example, generate FastWAM's ActionDiT backbone
payload once. The command below may download large Wan/T5 model assets into
`/path/to/fastwam_wan_models`:

```bash
cd /path/to/FastWAM
mkdir -p checkpoints /path/to/fastwam_wan_models
DIFFSYNTH_MODEL_BASE_PATH=/path/to/fastwam_wan_models \
PYTHONPATH=/path/to/FastWAM:/path/to/FastWAM/src \
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

```bash
python examples/evaluate_fastwam_joint.py \
  --openpi-root /path/to/openpi \
  --fastwam-root /path/to/FastWAM \
  --server-python /path/to/fastwam-env/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/fastwam_joint.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --wan-root /path/to/fastwam_wan_models \
  --num-inference-steps 1 \
  --output-dir outputs/fastwam-joint-smoke \
  --gpu 0
```

## 4. VLASH Pi0.5 Async5

Project: [MIT HAN Lab VLASH](https://github.com/mit-han-lab/vlash). Use the
official `sim/libero` implementation and the matching
`vlash-pi05-libero-async5` checkpoint in a separate Python 3.10 environment.
The evaluator environment still needs `openpi` only for its lightweight
websocket client.

```bash
git clone --branch sim/libero https://github.com/mit-han-lab/vlash.git
cd vlash
conda create -n vlash-libero python=3.10 -y
conda activate vlash-libero
pip install -e .
pip install websockets msgpack huggingface_hub

mkdir -p /path/to/vlash-pi05-libero-async5
huggingface-cli download mit-han-lab/vlash-pi05-libero-async5 \
  --local-dir /path/to/vlash-pi05-libero-async5

export VLASH_HF_HOME=/path/to/vlash-huggingface-cache
HF_HOME="$VLASH_HF_HOME" huggingface-cli download google/paligemma-3b-pt-224
```

Measure batch-one end-to-end latency on one otherwise idle target GPU using the
same checkpoint, two camera inputs, five-action chunk, and websocket request
path. Start exactly one policy server:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME="$VLASH_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/path/to/vlash \
/path/to/vlash-env/bin/python scripts/serve_vlash_openpi_ws.py \
  --checkpoint /path/to/vlash-pi05-libero-async5 \
  --device cuda --port 18240
```

In the evaluator environment, retain steady-state end-to-end samples:

```bash
MUJOCO_GL=egl /path/to/libero-env/bin/python scripts/calibrate_openpi_latency.py \
  --output-json outputs/latency/vlash_rtx5090_run1.json \
  --openpi-port 18240 --camera-size 256 --openpi-resize-size 224 \
  --replan-steps 5 --warmup-requests 8 \
  --num-items 12 --samples-per-item 8
```

Repeat the clean run and freeze the median over both runs' retained samples;
do not select the minimum, maximum, or fastest run. Pass that frozen median to
the example:

```bash
python examples/evaluate_vlash.py \
  --openpi-root /path/to/openpi \
  --vlash-root /path/to/vlash \
  --server-python /path/to/vlash-env/bin/python \
  --eval-python /path/to/libero-env/bin/python \
  --checkpoint /path/to/vlash-pi05-libero-async5 \
  --measured-latency-seconds 0.10398730050292215 \
  --hf-home "$VLASH_HF_HOME" --offline \
  --output-dir outputs/vlash-future-state-async-smoke \
  --gpu 0
```

The example computes `ceil(measured_latency * 20 Hz)` and rejects an explicit
delay outside the official checkpoint's zero-to-four-step training range. For
the audited RTX 5090 measurement above, the mapping is three visual steps
(`150 ms`). Robot state remains current, each five-action chunk is executed
intact, and no LeRobot weighted merge, prefix drop, or second generic latency
delay is applied. The wrapper intentionally requires one evaluator and one
policy server; a parallel sweep must start one independent persistent server
per evaluator worker.

## Common behavior

Use separate ports and GPUs when running examples concurrently. `--task-count`
selects complete episodes, not partial rollout fragments. The default is one
episode with a 600-step horizon and early termination on success. Every example
saves a 256 x 128 dual-view model-input video at 60 fps, traces, manifests, and
`policy_server.log`, then terminates its policy server. This is a downsampled
review copy; policy observations remain at the official 256 x 256 camera
resolution. Frames are sampled from simulation time during both action and
latency physics substeps, so the video preserves the duration of synchronous
holds. The examples do not save review/runtime or separate wrist videos.
`--model-input-video-size` and `--video-fps` may be overridden for diagnostics.
For a fixed half-episode format audit, pass `--max-steps 300
--no-stop-on-success`; normal benchmark runs should retain the default early
termination on success.

For a reproducible stratified audit, place exact item IDs in a JSON list and
pass it with `--item-ids-file`. The example derives the required levels and
trajectory indices, preserves the listed order, and rejects missing or
duplicate IDs. `--num-workers` controls parallel simulator workers:

```json
[
  "libero_10_task01__l1__traj00",
  "libero_goal_task08__l1__traj02",
  "libero_spatial_task01__l1__traj04"
]
```

```bash
$EVAL_PY examples/evaluate_pi05_pytorch_rtc.py \
  ... \
  --item-ids-file review_items.json \
  --num-workers 4
```

Parallel workers use the multiprocessing `spawn` context because MuJoCo/EGL
contexts are not safe to inherit through `fork`. A shared policy server is
appropriate for output or behavior audits with a latency value calibrated
before the run. Do not use policy wall-clock timings collected under shared
parallel load as latency calibration data.

The RTC example requires the server handshake to report
`rtc_conditioning_supported=true`; otherwise the episode fails instead of
silently recording an ordinary async run as RTC. Per-item manifests preserve
the complete server metadata and distinguish RTC conditioning requested,
supported, and actually used.

For functional smoke testing, inspect `evaluation/summary.json`. `ok_status`
and `diagnostic_ok` tell you whether the evaluator/server path ran cleanly;
`success_count` is the raw LIBERO task predicate. The summary also retains the
main repository's Action magnitude and continuity definitions, underflow,
latency, RTC-conditioning, and physical diagnostic aggregates. Per-step traces
retain chunk boundaries, poses, and contacts. A policy can produce a clean
smoke run with `success_count: 0`.

Checkpoint and model licenses remain those of their upstream projects; this
benchmark repository does not redistribute them.
