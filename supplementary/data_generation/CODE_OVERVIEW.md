# LIBERO-Dynamic data-generation code

This directory contains the release-side implementation used to create and
audit LIBERO-Dynamic demonstrations. It is the **data-generation code**, not
the supplementary-website builder.

## What the pipeline does

1. Loads a verified successful official static LIBERO demonstration as a
   timing and contact template.
2. Builds one reviewed dynamic condition from the 40 tasks, three motion
   families, five trajectories, five phases, and three object speeds.
3. Advances the moving object through the same physical environment used at
   evaluation time and observes its actual simulator pose.
4. Adds proximity- and phase-aware end-effector compensation to the template
   action, always through ordinary `env.step(action)` calls.
5. Rejects episodes that fail task success, motion, visibility, support,
   contact, tilt, drop, or other physical-validity checks.
6. Writes per-episode arrays, videos, provenance, `manifest.jsonl`,
   `summary.json`, a strict audit, and an optional LIBERO-format HDF5 file.
7. Audits the disjoint benchmark and non-benchmark phase partitions.

The generator is deterministic for a fixed simulator/configuration/seed. It
does not fabricate trajectories or interpolate missing episodes.

## Dataset grid and final split

The reviewed source grid is:

`40 tasks x 3 motion families x 5 trajectories x 5 phases x 3 speeds = 9,000 episodes`.

The final paper protocol uses an exactly benchmark-matched 1,800-episode test
set (one episode per task/motion/trajectory/speed condition) and a disjoint
7,200-episode non-benchmark complement. A deterministic phase-balanced compact
training subset selects one of those four non-benchmark phases per condition,
giving 1,800 training episodes.

## Files

- `generate_dynamic_demos.py`: main generator, strict validator, video/NPZ/HDF5 writer.
- `build_phase_partition.py`: split reconstruction and invariant checks.
- `configs/`: the complete reviewed L1/L2/L3 trajectory annotations.
- `support/eval_dynamic_dummy_policy.py`: condition enumeration and diagnostics.
- `support/eval_dynamic_libero_style.py`: environment and physical rollout utilities.
- `support/render_dynamic_config_previews.py`: canonical object-motion implementation.
- `support/dynamic_motion_invariants.py`: rejects L1/L2/L3 cross-family overrides.
- `support/rebuild_dynamic_configs_from_manual_json.py`: rebuilds configs and keeps
  duplicate compatibility motion blocks synchronized.
- `SOURCE_FILES.json`: hashes and repository-relative provenance for every bundled file.

For double-blind release, obsolete author-machine paths stored only in YAML
annotation-provenance fields are replaced with
`<local-provenance-path-redacted>`. No trajectory point, speed, phase, target,
physics option, or execution parameter is changed.

## Requirements

- The public LIBERO-Dynamic repository, installed as documented by that repository.
- Official static LIBERO demonstration HDF5 files.
- Python packages used by the benchmark, including `numpy`, `h5py`, `PyYAML`,
  `imageio`, LIBERO, robosuite, and MuJoCo.

Set the two machine-local inputs explicitly:

```bash
export LIBERO_DYNAMIC_REPO_ROOT=/path/to/libero-dynamic
export LIBERO_DYNAMIC_DATASET_ROOT=/path/to/libero_datasets
```

The bundled support files mirror their canonical locations. For direct use,
copy them back to `<repo>/scripts/` (or use the same files already present in
the public repository), and place `configs/*.yaml` under the repository's
`libero/libero/dynamic_configs/` directory.

## Example generation call

One invocation generates the 600 task/motion/trajectory conditions for one
speed and one phase:

```bash
python generate_dynamic_demos.py   --output-dir /path/to/output/speed003_phase00   --levels l1 l2 l3   --traj-indices 0 1 2 3 4   --dynamic-speed-mps 0.03   --phase-offset-ratio 0.0   --control-freq 20   --pack-hdf5
```

Repeat with the reviewed speed/phase values recorded by the release manifest.
For the exact benchmark initialization, use `--exact-benchmark-phase` together
with the benchmark initialization options documented by `--help`. Review
`manifest.jsonl`, `strict_audit.json`, and the generated review page before
accepting any episode.

## Phase-partition audit

```bash
python build_phase_partition.py   --legacy-benchmark-selection /path/to/legacy_selection.json   --phase-catalog /path/to/phase_catalog.json   --source-index-root /path/to/source_indices   --training-selection /path/to/train_selection_1800.json   --exact-selection /path/to/exact_benchmark_1800.json   --output-root /path/to/partition_output
```

The script fails rather than silently continuing if counts, uniqueness,
condition coverage, or train/test disjointness are violated.
