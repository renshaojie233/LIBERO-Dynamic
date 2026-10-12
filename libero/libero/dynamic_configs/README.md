# LIBERO-dynamic configs

Use `libero_dynamic_l1.yaml`, `libero_dynamic_l2.yaml`, or
`libero_dynamic_l3.yaml` as the current benchmark entry configs. Each level
contains 40 LIBERO tasks x 5 manually reviewed trajectory variants.

The checked-in configs are the final reviewed 2026-06 benchmark revision. They
include the corrected `libero_spatial` task-07 L2 trajectory and the validated
high-friction physical carrier settings. The three YAML files are the canonical
benchmark inputs. They intentionally contain no waypoint acceleration or stop
fields: waypoint motion does not decelerate or pause at corners.

Minimal path-swap usage:

```bash
export PYTHONPATH=/path/to/LIBERO-dynamic:$PYTHONPATH
export LIBERO_CONFIG_PATH=/path/to/LIBERO-dynamic/libero/libero/configs

# Choose one motion level.
export LIBERO_DYNAMIC_CONFIG=libero_dynamic_l1

# Choose one of the five manually reviewed trajectories per task.
# Equivalent values: traj00 ... traj04.
export LIBERO_DYNAMIC_TRAJ_INDEX=0

export LIBERO_DYNAMIC_EVAL_OUTPUT_DIR=/path/to/eval_output
# Optional; if omitted, logs default to $LIBERO_DYNAMIC_EVAL_OUTPUT_DIR/libero_dynamic_logs.
export LIBERO_DYNAMIC_LOG_DIR=/path/to/eval_output/libero_dynamic_logs
```

You can also select an exact variant id, which is useful for debugging:

```bash
export LIBERO_DYNAMIC_VARIANT_ID=libero_10_task02__l2__traj03
```

The env resolves the variant internally from `LIBERO_DYNAMIC_CONFIG` and the
current LIBERO `suite/task_id`, so standard LIBERO eval code can keep creating
`OffScreenRenderEnv` from the original BDDL task. Running all 600 benchmark
variants means sweeping:

- `LIBERO_DYNAMIC_CONFIG in {libero_dynamic_l1, libero_dynamic_l2, libero_dynamic_l3}`
- `LIBERO_DYNAMIC_TRAJ_INDEX in {0, 1, 2, 3, 4}`
- the original 40 LIBERO tasks

For tasks with multiple dynamic objects, the env installs one independent
physical conveyor patch per target object. Each patch has explicit contact
pairs only with its own carried object, so one target's platform is invisible
to the other target objects and to unrelated clutter.

The config uses a compact schema:

- `defaults`: shared dynamic benchmark logic, including latency, video-frame saving, carrier behavior, and default motion style.
- `tasks`: per-task overrides. Each task should mainly specify the target object / joint and any task-specific motion parameters.
- `activation_variants`: five manually reviewed trajectory variants per task.

Unlisted tasks are static by default while the task table is being built.


## Motion families and safety

The benchmark should not be limited to one straight ping-pong motion. We follow the same spirit as DOMINO's dynamic hierarchy: Level 1 predictable constant-velocity motion, Level 2 high-order polynomial / spline motion, and Level 3 segmented stochastic or abrupt motion. In LIBERO-dynamic configs this should map to a `motion.family` field such as:

- `conveyor_pingpong`: bounded constant-velocity surface motion for graspable movable objects.
- `sinusoidal_pingpong`: smooth periodic motion without hard reflection discontinuities.
- `polynomial_spline`: predictable curved trajectory through safe control points.
- `zigzag_waypoints`: irregular bounded waypoint trajectory with direction changes.
- `stochastic_segmented`: two or three sampled Level 1/2 segments with random durations.
- `joint_pingpong`: bounded articulated fixture motion, e.g. opening / closing a drawer.
- `fixture_pose_spline`: small direct XY pose motion for target fixtures such as a stove or rack.

Use `alpha_mps` as the main motion-speed difficulty knob, and keep it separate from latency difficulty. Before promoting a task config into the benchmark, run a preflight rollout without robot motion and reject the config if the target leaves the safe workspace / camera view or collides with non-target clutter. For drawer tasks, prefer driving the drawer joint; do not move the whole cabinet unless that is the explicit benchmark variant. For target-location fixtures like a stove, use direct small fixture pose motion, not surface-velocity conveyor logic.


## Preflight checking

Before using a task config for model evaluation, run the preflight checker. It executes the configured dynamic motion with zero robot actions and reports workspace-bound violations and unexpected target contacts.

```bash
PYTHONPATH=/path/to/LIBERO-dynamic:$PYTHONPATH LIBERO_CONFIG_PATH=/path/to/libero_config LIBERO_DYNAMIC_AUTO_FINALIZE=0 python /path/to/LIBERO-dynamic/scripts/preflight_dynamic_configs.py   --config libero_dynamic   --suite libero_spatial   --task-id 0   --steps 120   --output /tmp/libero_dynamic_preflight.json
```

A failed preflight is useful signal, not just an error. For example, a zigzag path that is safe in one task may hit the plate or cabinet in another. Fix the per-task `axis`, `amplitude_m`, `zigzag_lateral_m`, `waypoints_xyz`, or `safety.workspace_xyz`, then rerun preflight before launching expensive model evaluation.

## Realtime latency semantics

When `latency.use_measured_latency: true`, the eval loop should report each
policy-inference duration by calling:

```python
env.notify_inference_latency(infer_elapsed_time)
```

The env consumes that notified latency exactly once at the next control step.
During this simulated inference interval:

- robot and gripper joints are frozen;
- dynamic objects / target locations continue moving;
- latency frames are saved as MuJoCo state snapshots at `latency.video_fps`;
- the action is executed only after the latency simulation finishes.

LIBERO-dynamic no longer infers inference latency from wall-clock gaps between
`env.step()` calls, because chunk-internal action execution and policy replan
latency cannot be distinguished robustly from inside the env. If no notification
is received, no measured-latency delay is applied. To use fixed delays instead,
call `env.notify_inference_latency(None)` at replan boundaries or set
`latency.use_measured_latency: false` in a controlled eval wrapper.

There is no default max-latency cap in `libero_dynamic.yaml`. If you add `max_latency_seconds`, measured latency will be clipped.

## Result finalization

`libero_dynamic.yaml` enables automatic finalization. After each episode finishes, the env starts a small background watcher; once the normal LIBERO rollout MP4 appears, it writes the two final videos:

- the model-observation video containing only detected inference/replan input moments: `<eval_output>/model_input_observation_videos/<suite>/videos/*--model-input-observations.mp4`;
- the real-time video with latency frames inserted: `<eval_output>/dynamic_real_time_videos/<suite>/videos/*--dynamic-enhanced.mp4`;
- the paired manifest: `<eval_output>/dynamic_results_summary.json`;
- a LIBERO-compatible dynamic result JSON beside the original task result: `<eval_output>/<suite>/gpu*_task*_dynamic_results.json`;
- the merge details: `<eval_output>/dynamic_real_time_videos/dynamic_video_merge_summary.json`.

The original LIBERO/FastWAM rollout video remains in `<eval_output>/<suite>/videos/*.mp4` for compatibility and for rebuilding the real-time video, but it is treated as a raw source artifact rather than a final dynamic-benchmark video.

Intermediate latency PNG folders are deleted by default after a successful merge (`finalize.cleanup_latency_frames: true`). The model-observation video is kept because it is a final artifact, not an intermediate cache. Use `--keep-latency-frames` in the manual finalizer, or set the config field to `false`, when debugging frame alignment.

A process-exit finalizer also runs as a safety net for the last episode. This does not require changing eval code, but the env must know the eval output root, so set `LIBERO_DYNAMIC_EVAL_OUTPUT_DIR` or `LIBERO_DYNAMIC_LOG_DIR=<eval_output>/libero_dynamic_logs`.

Manual finalization remains available and is safe to run repeatedly:

```bash
python /path/to/LIBERO-dynamic/scripts/finalize_dynamic_results.py \
  --eval-output-dir /path/to/eval_output
```

The finalizer defaults to FastWAM LIBERO's `num_steps_wait=30`, so event `step_idx` is mapped to video frame `step_idx - 30`. Override `--num-steps-wait` if your eval uses a different warmup.

Old one-off smoke configs are kept under `deprecated_smoke_configs/` only for reference.

## Benchmark QA scripts

The package includes the same no-policy visual/physics QA path used to build
the reviewed configs:

```bash
PYTHONPATH=/path/to/LIBERO-dynamic:$PYTHONPATH \
LIBERO_CONFIG_PATH=/path/to/LIBERO-dynamic/libero/libero/configs \
python /path/to/LIBERO-dynamic/scripts/render_dynamic_config_previews.py \
  --config-dir /path/to/LIBERO-dynamic/libero/libero/dynamic_configs \
  --levels l1 l2 l3 \
  --start-index 0 \
  --limit 1 \
  --carrier-override physical_conveyor_patch \
  --validate-only
```

`render_dynamic_config_previews.py` supports multi-target variants and is the
source of truth for visual QA before promoting configs into model evaluation.
`rebuild_dynamic_configs_from_manual_json.py` rebuilds the level configs from
manual annotation JSON while preserving the reviewed waypoint geometry.
