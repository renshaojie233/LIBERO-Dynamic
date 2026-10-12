# LIBERO-Dynamic Formal Evaluation

`scripts/eval_dynamic_formal.py` is the benchmark-facing synchronous
realtime-v2 entrypoint. It follows the same loop for a zero policy or a real
OpenPI-compatible policy:

1. create a normal LIBERO `OffScreenRenderEnv`;
2. reset and set the selected init state;
3. read the current observation;
4. predict an action chunk;
5. advance exact continuous MuJoCo time for synchronous inference latency
   while holding robot joints;
6. execute the chunk at the 20 Hz action clock;
7. write a per-item manifest, optional video, and aggregate summary.

The zero policy is an infrastructure and physics smoke test, not a task-success
baseline. Formal model results must use a fixed latency calibrated beforehand
on an otherwise idle GPU.

## Clean Baseline

Use the current formal-eval tag on the main repository:

```bash
git clone https://github.com/renshaojie233/libero_dynamic_base.git libero_dynamic_base-formal-eval
cd libero_dynamic_base-formal-eval
git checkout main
```

Older reviewed preview and table-contact snapshots are archived in
`renshaojie233/LIBERO-dynamic-junkyard`.

## Small Smoke Test

```bash
PYTHONPATH=$PWD python scripts/eval_dynamic_formal.py \
  --output-dir outputs/formal_dummy_smoke \
  --levels l1 \
  --traj-indices 0 \
  --limit 4 \
  --num-workers 1 \
  --policy zero \
  --latency-source fixed \
  --calibrated-latency-seconds 0 \
  --keep-static-support-contact \
  --static-support-clearance-m 0.001 \
  --max-steps 40 \
  --save-videos \
  --no-save-review-video \
  --save-model-input-videos \
  --model-input-video-layout dual \
  --model-input-video-size 128 \
  --video-fps 60
```

Expected:

- `summary.json` reports `total: 4`.
- Every item has `status: ok`.
- Every item has non-empty `diagnostics.targets`.
- Every item has `diagnostics.ok: true` for benchmark QA.
- Videos exist if `--save-videos` is enabled.

## Reported Metrics

Every per-item manifest and aggregate `summary.json` preserve more than task
success:

- `success` / `success_count`: raw LIBERO task success;
- platform path error, object-follow error, tilt, relative drop, visibility,
  and diagnostic flags;
- queue underflow, chunk count, policy requests, calibrated simulation latency,
  wall-clock policy latency, and RTC requested/supported/used state;
- `action_l2 = ||a_t||_2` and Action continuity
  `action_delta_l2 = ||a_t-a_(t-1)||_2`.

These are the definitions used by the LIBERO-Dynamic main repository. Action
aggregates are computed even when step-trace files are disabled; the trace keeps
the exact per-control-step values together with chunk boundaries, poses, and
contacts when enabled.

## Full Dummy Eval

Run low concurrency first if the machine has limited EGL/offscreen contexts.

```bash
PYTHONPATH=$PWD python scripts/eval_dynamic_formal.py \
  --output-dir outputs/formal_dummy_full600_fixed_latency \
  --levels l1 l2 l3 \
  --traj-indices 0 1 2 3 4 \
  --num-workers 4 \
  --policy zero \
  --latency-source fixed \
  --calibrated-latency-seconds 0 \
  --keep-static-support-contact \
  --static-support-clearance-m 0.001 \
  --max-steps 600 \
  --save-videos \
  --no-save-review-video \
  --save-model-input-videos \
  --model-input-video-layout dual \
  --model-input-video-size 128 \
  --video-fps 60
```

## Latency Sources

- `--latency-source fixed --calibrated-latency-seconds S`: formal protocol;
  applies the premeasured value `S` to every request.
- `--latency-source measured_wallclock`: single-worker diagnosis only; applies
  the current request wall-clock duration.
- `--latency-source empirical --latency-samples-json FILE`: samples a declared
  empirical latency distribution.

`--video-fps` controls only the saved simulation-time video. It defaults to 60
and may be set to 100 for short timing diagnostics. It never changes the 20 Hz
action clock or the model observation resolution.

## Config Safety

Formal eval builds each item from the reviewed config using the same structure
as preview:

```text
defaults -> task_cfg -> activation_variant
```

Do not replace `task_cfg` with the raw variant. That drops task-level fields
such as `carrier`, which prevents the physical conveyor platform from being
installed.

For the currently reviewed full600 table-contact run, pass:

```bash
--keep-static-support-contact --static-support-clearance-m 0.001
```

The script writes `effective_dynamic_configs/` under the output directory and
points each eval item at those files. This makes the environment consume the
same table-contact/1mm variant that was reviewed in preview videos.

## Diagnostics Baseline

Relative-drop diagnostics default to:

```bash
--bottom-baseline-mode first_sample
```

This uses the first formal rollout diagnostic sample as the height baseline.
It avoids treating initial platform/table-contact stabilization immediately
after `set_init_state(...)` as an object drop, while still flagging later
downward motion. Use `--bottom-baseline-mode reset` only when you explicitly
want the raw reset state as the baseline.
