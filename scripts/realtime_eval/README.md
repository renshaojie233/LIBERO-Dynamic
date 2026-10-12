# Realtime Evaluation Runtime

This package keeps the simulated world independent from policy scheduling.

## Contract

- Control actions execute once per control tick. Synchronous inference latency
  additionally advances exact continuous MuJoCo physics time while robot joints
  are held.
- Dynamic objects and fixtures move only through the environment step path.
- Policy controllers never call `env.step()` and never mutate the scene.
- Controllers receive observations and may later enqueue action chunks.
- If no action is available at a tick, the action runtime applies the configured
  empty action (`hold` or `zero`) and records an underflow.

This mirrors a real robot setup: the world keeps moving while policy inference,
network delays, RTC queues, or future-state predictors run outside the world.

## Main Pieces

- `control.py`: shared packets, action queue, latency sampler, controller base
  class, and the controller factory.
- `strategies/sync_hold.py`: synchronous hold-latency baseline.
- `strategies/async_queue.py`: LeRobot official async chunking: trigger by
  `queue_len / chunk_size <= chunk_size_threshold`, skip stale timesteps, and
  merge overlapping future actions with `0.3 * old + 0.7 * new`.
- `strategies/rtc_queue.py`: LeRobot official RTC queueing: pass original
  `prev_chunk_left_over` plus inference delay to the RTC-enabled policy, then
  drop the delay prefix and replace the executable future queue.
- `strategies/vlash_future_state.py`: VLASH LIBERO future-state protocol:
  delayed visual observations, current robot state, and intact action chunks.
- `runtime.py`: one episode orchestrator that connects a world, controller, and
  recorder without letting controllers step the world. Avoid editing this for
  policy ideas unless the world contract itself changes.
- `runner.py`: benchmark entrypoint, sharding, manifests, and summaries.

`protocol.py`, `latency.py`, and `controllers.py` are compatibility import
wrappers so older imports keep working. New code should use `control.py`.

To add a new policy scheduling strategy, copy one file under `strategies/`,
override `should_start_request()`, and register the new mode in
`control.py::build_policy_controller()`. Do not edit `runtime.py` for ordinary
strategy experiments.

## Control Modes

- `sync_hold`: submit a policy request when the action queue is empty. The
  request blocks action production; its exact latency advances MuJoCo physics
  while robot joints are held, without inserting a quantized 20 Hz action.
- `async_queue`: LeRobot official async client semantics. The default trigger is
  `--chunk-size-threshold 0.5`; the returned chunk replaces the future action
  queue after stale timesteps are skipped. If an incoming timestep overlaps with
  an existing queued timestep, the default aggregate is `weighted_average =
  0.3 * old + 0.7 * new`.
- `rtc_queue`: LeRobot official RTC ActionQueue semantics. Only this mode sends
  RTC conditioning to the policy. The runtime snapshots unconsumed original
  policy actions for `prev_chunk_left_over`, supplies `inference_delay_steps`,
  then replaces the executable queue with returned actions after dropping the
  delay prefix. It does not do client-side weighted overlap.
- `vlash_future_state`: official VLASH LIBERO semantics. It derives visual
  staleness as `ceil(measured_latency * control_freq)`, pairs delayed camera
  images with current robot state, and executes each returned chunk intact.
  Generic simulated latency must be zero, so the measured delay is represented
  exactly once. This mode does not use LeRobot weighted merging or stale-prefix
  dropping.

`hold_latency` and `naive_async_queue` are kept as backward-compatible aliases
for older manifests. Do not use the old append-only async queue for formal RTC
claims.

Async and RTC request triggers use the configured queue threshold exactly.
Their latency budget is retained as metadata and never silently changes the
trigger. Stale-prefix trimming counts only actions actually dequeued after a
request was submitted; an underflow hold does not age the returned chunk.

`scripts/run_pi05_rtc_eval.py` requires the policy handshake to report
`rtc_conditioning_supported=true`. Manifests separately record whether RTC was
requested, supported, and used, together with the full policy-server metadata.

## Latency Sources

- `measured_wallclock`: use actual policy wall-clock time.
- `fixed`: use a calibrated latency value, allowing shared model services and
  high-throughput simulation without contaminating simulated latency.
- `empirical`: sample from a measured latency distribution.

For throughput benchmarks with shared model servers, prefer `fixed` or
`empirical`; otherwise queueing on the shared service becomes part of the
simulated latency.

VLASH is a protocol-specific exception: its clean-GPU end-to-end measurement
selects an integer delayed-visual offset. Pass the measured value through
`--vlash-measured-latency-seconds`, set the matching
`--vlash-visual-delay-steps`, and keep `--calibrated-latency-seconds 0`.

## Video Modes

The formal default saves only a compact 256 x 128 dual-view video: 128 x 128
`agentview` beside 128 x 128 `robot0_eye_in_hand`. It does not save the review
render or a separate wrist video. The policy camera remains 256 x 256; the
saved copy is downsampled independently. `--record-latency-frames --video-fps
60` samples one simulation-time clock across action execution and synchronous
latency physics substeps, so visible holds have their true duration. An
equivalent explicit configuration is:

```text
--no-save-review-video --save-model-input-videos
--model-input-video-layout dual --model-input-video-size 128
--record-latency-frames --video-fps 60
```
