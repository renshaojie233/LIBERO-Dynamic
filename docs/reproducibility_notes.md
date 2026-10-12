# Reproducibility Notes

This document records environment constraints and operational requirements for
reproducing the LIBERO-Dynamic Base benchmark. These requirements are part of
the evaluated baseline and should be preserved when reporting comparable
results.

## Validated Baseline

| Component | Validated value |
| --- | --- |
| Source benchmark | `formal-eval-final-20260622` |
| Source commit | `f1a86422d738b1f5759ad871f7e341d99c88f7c9` |
| Python | 3.10 |
| robosuite | 1.4.0 |
| MuJoCo | 2.3.7 |
| PyTorch | 2.13.0+cu130 |
| CUDA user-space packages | 13.0.3 |
| Rendering backend | EGL |
| Pure-simulation speeds | 0.03, 0.06, 0.09 m/s |
| Pure-simulation set size | 600 configurations per speed; 1800 total |

The baseline was reproduced on an NVIDIA RTX 5090 workstation with driver
580.126.18. The policy-free renderer does not depend on a particular PyTorch
build, but model evaluation requires a CUDA build compatible with the policy
environment and host driver.

## Dependency Compatibility

`robosuite==1.4.0` declares only a lower bound on MuJoCo. An unconstrained
installation can therefore select a recent MuJoCo release whose API is not
compatible with robosuite 1.4.0. The evaluator requirement is pinned to
`mujoco==2.3.7`; do not remove this pin without rerunning the complete
simulation and model smoke suites.

Policy implementations must use separate environments. In particular, OpenPI,
LeRobot, and FastWAM have independent Python, CUDA, JAX, PyTorch, tokenizer, and
model-asset constraints. They communicate with the evaluator through the
OpenPI-compatible websocket protocol and should not be installed into the
evaluator environment.

## Headless Rendering and Concurrency

Set `MUJOCO_GL=egl` before importing MuJoCo. The pure-simulation sweep starts
independent Python subprocesses because forking a process after EGL has been
initialized can fail with `EGL_BAD_ALLOC` or produce invalid rendering
contexts. `--workers` controls the number of concurrent subprocesses. Start
with 3 or 4 workers on an unknown host and increase only after the smoke sweep
passes.

The following environment variables are set by the pure-simulation entrypoint:

- `MUJOCO_GL=egl`;
- `MUJOCO_EGL_DEVICE_ID=<gpu>`;
- `CUDA_VISIBLE_DEVICES=<gpu>`;
- `LIBERO_CONFIG_PATH=<repository>/libero/libero/configs`;
- `PYTHONPATH=<repository>:<existing PYTHONPATH>`.

## Pure-Simulation Semantics

`examples/render_pure_simulation.py` calls the reviewed
`scripts/render_dynamic_config_previews.py` implementation. The script applies
the requested speed with `--speed-mps`; it does not rewrite benchmark YAML.
Each configuration is simulated for one complete motion cycle with static
support contact enabled and 1 mm support clearance.

The aggregate QA thresholds are:

| Diagnostic | Threshold |
| --- | ---: |
| Platform path error | <= 0.01 m |
| Object-to-platform error | <= 0.08 m |
| Maximum tilt | <= 35 degrees |
| Maximum drop | <= 0.04 m |
| Birdview visible ratio | >= 0.75 |
| Agentview visible ratio | >= 0.75 |
| Unexpected collision count | 0 |

The default command reports QA findings but exits successfully when every
shard and video is structurally complete. Pass `--fail-on-qa-warning` when the
QA thresholds must be enforced as a nonzero exit status.

The released YAML uses `physical_conveyor_patch` carriers. Model and zero-policy
evaluation must use `scripts/eval_dynamic_formal.py`, which enables the reviewed
physical-carrier profile and realtime-v2 timing for every target. The lower-level
`scripts/eval_dynamic_libero_style.py` retains legacy routing options and is not
the documented baseline entrypoint.

Formal videos use `--video-fps` as a simulation-time sampling rate. The default
is 60 fps at 128 x 128 pixels per view (256 x 128 for the dual-view file).
Changing this option does not change the 20 Hz policy clock or policy input
resolution. Use 100 fps only for short timing diagnostics; numerical latency
claims must come from manifests and traces rather than manual frame counting.

## Reference Full-Sweep Observation

The clean 1800-video reproduction completed all 1800 records and all videos
passed `ffprobe`. Under the thresholds above, 1795 records passed without a QA
warning. The five boundary records were:

- `libero_goal_task06__l2__traj00` at 0.06 and 0.09 m/s: the moving black bowl
  reported one and three sub-millimetre contacts with the stove burner;
- `libero_spatial_task04__l2__traj02` at all three speeds: the second bowl's
  agent-view visibility ratio was 0.7396 to 0.7413, below the 0.75 threshold.

These findings are retained in `summary.json` and must not be silently counted
as strict QA passes. They do not indicate renderer failure, and visual review
may still accept the corresponding motion. Publications should state whether
they use raw completion, strict QA completion, or manual visual acceptance.

## Evaluation Semantics

The following outcomes are distinct:

- **Policy-free completion:** the scene rendered and diagnostics were produced;
- **Infrastructure health:** the evaluator and policy server completed an
  episode without a runtime error;
- **Task success:** the environment success predicate was satisfied;
- **Strict QA pass:** all configured physics and visibility thresholds passed.

Do not report policy-free completion or infrastructure health as task success.
For model evaluations, retain `summary.json`, per-item manifests, model-input
videos, policy-server logs, the exact repository commit, checkpoint identity,
runtime mode, and calibrated latency.

## Network Mirrors

Package and Git mirrors may be used to improve download throughput, but they
must not alter the resolved package versions. Record the final environment with
`pip freeze` or an equivalent lock export. Model checkpoints should be verified
with publisher-provided checksums when available.

LIBERO may print a warning when the optional `libero/datasets` directory is
absent. Policy-free rendering and benchmark evaluation use repository BDDL
files and packaged task initialization states; they do not require the offline
training dataset. Dataset collection or training workflows must install those
datasets separately.
