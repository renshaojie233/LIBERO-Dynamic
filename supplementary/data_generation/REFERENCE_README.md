# Dynamic-data generation

This directory contains the release-side reference implementation used to
construct and partition LIBERO-Dynamic demonstrations.

- `generate_dynamic_demos.py` replays a verified successful static LIBERO
  demonstration as a timing/contact template, tracks the configured moving
  target, applies proximity- and phase-aware end-effector compensation through
  ordinary `env.step(action)` calls, and accepts an episode only after task,
  motion, and physical-validity checks pass.
- `build_phase_partition.py` reconstructs the disjoint phase split: 1,800 exact
  benchmark-phase episodes, the complementary 7,200 non-benchmark-phase
  episodes, and the phase-balanced 1,800-episode compact training subset.

The generator requires the benchmark repository and the official static LIBERO
demonstrations. Set these public, machine-local inputs explicitly:

```bash
export LIBERO_DYNAMIC_REPO_ROOT=/path/to/libero-dynamic
export LIBERO_DYNAMIC_DATASET_ROOT=/path/to/libero_datasets
python data_generation/generate_dynamic_demos.py --help
```

The benchmark grid is `40 tasks x 3 motion families x 5 reviewed trajectories
x 5 phases x 3 speeds = 9,000 episodes`. The formal test split instead uses one
exact benchmark-matched phase for each of the 1,800 task/motion/trajectory/speed
conditions. The compact training split draws one of the four remaining phases
per condition and balances the four phase ordinals.

No checkpoint, account, host, private storage path, or generated dataset is
embedded here. Generated manifests should still be scanned before a double-
blind submission because command-line inputs are supplied by the caller.
