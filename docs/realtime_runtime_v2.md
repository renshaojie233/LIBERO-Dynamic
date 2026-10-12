# Realtime Runtime V2

Runtime v2 separates MuJoCo physics time, controller time, and policy request
time. MuJoCo integrates at the model timestep (`0.002 s` in the validated
environment), while actions execute at `--control-freq` (`20 Hz`, or `0.05 s`,
by default). Policy modes produce chunks and never mutate dynamic objects.

For synchronous control, policy inference blocks action production. During the
exact simulated latency duration, robot joints are held and the dynamic scene
continues through MuJoCo physics substeps. The duration is not rounded with
`ceil(latency * control_freq)`: `0.03 s` is 15 physics substeps, `0.05 s` is 25,
and zero latency adds no step. Async and RTC action availability remains indexed
in discrete controller slots because those modes execute queued actions while a
request is in flight.

Video fps is a separate observation clock. With
`--record-latency-frames --video-fps 60`, runtime v2 samples MuJoCo state on a
single absolute 60 Hz video timeline during both action execution and
synchronous waits. It does not change the 20 Hz controller and does not update
the policy sensor cache while recording. Formal runs save only a 256 x 128
side-by-side copy of the two policy views (128 x 128 per view); the policy
camera itself remains at 256 x 256. Review renders and separate wrist videos
are disabled by default. `--video-fps` is configurable without changing the
controller; 60 fps is the review default and 100 fps is intended only for short
timing diagnostics.

Dynamic carrier paths, fixture-body transforms, and articulated fixture joints
are evaluated on every MuJoCo substep. At the default rates, each 20 Hz control
action spans 25 separate 500 Hz physics updates. Contact-driven objects are not
teleported to the analytic path: their carriers move at that resolution and the
objects respond through MuJoCo contact, friction, inertia, and slip.

## Control Modes

- `sync_hold` / `hold_latency`: submit a policy request only when the action queue is empty.
  Wait for the request, advance the dynamic scene by the exact inference
  duration while holding the robot, then execute the returned chunk. No
  synthetic zero-action control tick is inserted.
- `async_queue`: LeRobot-style async chunking. A new request is submitted when
  queue length reaches the larger of the configured threshold and the
  calibrated latency budget in action steps; when it arrives, overlapping
  future actions are merged with weighted averaging. The clean Pi0.5 entrypoint
  also enables LeRobot's default 50% chunk threshold.
- `rtc_queue`: delay-aware RTC-style chunking. Actions covered by the inference
  delay remain committed; the returned chunk updates the remaining future queue.
  The clean Pi0.5 RTC entrypoint keeps a conditioned request continuously in
  flight, following LeRobot's RTC background-worker behavior.
  The Pi0.5 RTC entrypoint requires server-side RTC support and passes
  `prev_chunk_left_over` and `inference_delay_steps`; it fails rather than
  silently falling back to ordinary async inference.
- `naive_async_queue`: old append-only queue baseline kept only for old manifests.
- `vlash_future_state`: reserved for future-state-aware controllers.

For compact reference implementations, read:

- `scripts/realtime_eval/sync_loop.py`
- `scripts/realtime_eval/async_queue_loop.py`
- `scripts/realtime_eval/rtc_queue_loop.py`

## Server Parallelism

`--num-workers N` starts N fixed worker slots. By default each worker uses its
own OpenPI endpoint:

```text
worker 0 -> --openpi-port
worker 1 -> --openpi-port + --policy-port-stride
...
```

Use `--no-per-worker-policy-endpoint` only for explicit shared-router
throughput experiments. Manifests mark those runs as latency-contaminated.

For calibrated throughput runs, multiple workers can share a smaller pool of
OpenPI services without letting service queueing change simulated latency:

```text
--per-worker-policy-endpoint --policy-server-count 4

worker 0 -> server 0
worker 1 -> server 1
worker 2 -> server 2
worker 3 -> server 3
worker 4 -> server 0
...
```

`--policy-server-count 0` keeps the one-worker-one-endpoint mapping.

## Latency Sources

Policy wall-clock time and simulated action availability are separate knobs:

- `--latency-source measured_wallclock`: use the actual request elapsed time as
  the simulated inference delay. This is the strict one-env-one-model setting.
- `--latency-source fixed --calibrated-latency-seconds S`: use a fixed
  pre-measured delay.
- `--latency-source empirical --latency-samples-json FILE`: sample simulated
  delay from a pre-measured latency distribution.

In `fixed` and `empirical` modes, wall-clock service time is decoupled from the
configured simulation latency. A synchronous request waits for the response,
then advances simulation by the configured continuous duration. Async and RTC
use calibrated controller-slot availability for queue merging. This preserves
the intended simulated timing while allowing shared-service throughput runs.

Formal parallel sweeps should calibrate with one server and one client on an
idle GPU, freeze a predeclared statistic, and use `fixed` for every worker.
Using `measured_wallclock` with parallel workers or a shared endpoint mixes
policy latency with queueing and resource contention, making simulated task
difficulty depend on evaluator throughput. Preserve the raw calibration JSON
and recalibrate whenever the model request or hardware configuration changes.
RTC-conditioned requests must be measured separately: enable RTC on the model
server, pass representative leftover actions and a control-step delay through
`--rtc-conditioning --rtc-inference-delay-steps D`, and iterate `D` until it
matches `ceil(median_latency * control_freq)`. Do not label a run as model-side
RTC when server metadata reports `rtc_conditioning_supported=false`.

## Example

```bash
PYTHONPATH=$PWD \
LIBERO_CONFIG_PATH=$PWD/libero/libero/configs \
MUJOCO_GL=egl \
python scripts/run_oldphysics_realtime_eval.py \
  --output-dir outputs/realtime_v2_smoke \
  --levels l1 \
  --traj-indices 0 \
  --limit 4 \
  --policy zero \
  --control-mode sync_hold \
  --latency-source measured_wallclock \
  --max-steps 80 \
  --save-videos \
  --no-save-review-video \
  --save-model-input-videos \
  --model-input-video-layout dual \
  --model-input-video-size 128 \
  --record-latency-frames \
  --video-fps 60
```

Outputs include the dual-view model-input video, per-item manifest, optional
step trace, aggregate `manifest.jsonl`, and `summary.json`.

Calibrated shared-service example:

```bash
python scripts/run_oldphysics_realtime_eval.py \
  --output-dir outputs/realtime_v2_openpi_calibrated \
  --levels l1 l2 l3 \
  --traj-indices 0 1 2 3 4 \
  --policy openpi_ws \
  --control-mode async_queue \
  --latency-source empirical \
  --latency-samples-json outputs/openpi_latency_samples.json \
  --num-workers 16 \
  --policy-server-count 4 \
  --openpi-port 8124
```

## Clean Pi0.5 Entrypoints

For formal Pi0.5 runs, prefer the single-strategy entrypoints:

- `scripts/run_pi05_sync_eval.py`
- `scripts/run_pi05_async_eval.py`
- `scripts/run_pi05_rtc_eval.py`

They all use the reviewed old-physics route and differ only in the external
policy scheduling controller. This keeps environment physics and policy
scheduling changes separated.
