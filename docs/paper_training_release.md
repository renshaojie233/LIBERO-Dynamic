# Paper training release

This page indexes the nine model versions used by the LIBERO-Dynamic training
study, the native single-frame FastWAM-Joint checkpoint used in the
cross-model baseline, and the strict non-benchmark training data used for fine-tuning. The
large files are hosted on Hugging Face; this repository contains the code,
evaluation protocol, aggregate results, and this release index.

## Downloads

- **Paper checkpoints and the additional native FastWAM-Joint baseline:** [rsj2003/libero-dynamic-paper-models](https://huggingface.co/rsj2003/libero-dynamic-paper-models)
- **Strict non-benchmark 1,800-episode training data:** [rsj2003/libero-dynamic-strict-nonbenchmark-1800](https://huggingface.co/datasets/rsj2003/libero-dynamic-strict-nonbenchmark-1800)

The model repository keeps all weights under one link. The dataset repository
contains both the original π0.5/LeRobot representation and the gripper-fixed
FastWAM representation of the same strict non-benchmark dynamic data.

The additional native baseline is
`fastwam_joint/original_single_frame/step_034720.pt`, paired with its
`dataset_stats.json`. It is the original single-frame, `history=1` checkpoint
used for the cross-model comparison. It is distinct from
`fastwam_joint/single_frame/static_base/`, the 60K static base used in the
dynamic-data training study.

## Nine paper training versions

| ID | Model family | Training recipe | Paper role | HF path |
|---|---|---|---|---|
| `fastwam_joint_single_static` | Single-frame FastWAM-Joint | Static LIBERO base | Static baseline | `fastwam_joint/single_frame/static_base/` |
| `fastwam_joint_single_dynamic_ft3000` | Single-frame FastWAM-Joint | Dynamic-only fine-tune, step 3000 | Dynamic fine-tune | `fastwam_joint/single_frame/dynamic_ft3000/` |
| `fastwam_joint_single_mixed_ft3000` | Single-frame FastWAM-Joint | Dynamic + static fine-tune, step 3000 | Mixed fine-tune | `fastwam_joint/single_frame/mixed_ft3000/` |
| `fastwam_joint_history_static` | History-frame FastWAM-Joint | Dense history static base | Static baseline | `fastwam_joint/history_frame/static_base/` |
| `fastwam_joint_history_dynamic_ft3000` | History-frame FastWAM-Joint | Dynamic-only fine-tune, step 3000 | Dynamic fine-tune | `fastwam_joint/history_frame/dynamic_ft3000/` |
| `fastwam_joint_history_mixed_ft3000` | History-frame FastWAM-Joint | Dynamic + static fine-tune, step 3000 | Mixed fine-tune | `fastwam_joint/history_frame/mixed_ft3000/` |
| `pi05_static_official` | π0.5 | Official static LIBERO checkpoint | Static baseline | `pi05/static_official/` |
| `pi05_dynamic_ft1500` | π0.5 | Strict non-benchmark dynamic-only fine-tune, step 1500 | Dynamic fine-tune | `pi05/dynamic_ft1500/` |
| `pi05_mixed_ft3000` | π0.5 | Strict non-benchmark dynamic + static fine-tune, step 3000 | Mixed fine-tune | `pi05/mixed_ft3000/` |

FastWAM entries include their paired normalization statistics when those
statistics are part of the trained run. π0.5 entries include the inference
parameters, checkpoint metadata, and matching normalization assets; optimizer
state is intentionally not included in this inference-weight release.

## Data scope

The dataset release is the strict non-benchmark dynamic set: 1,800 episodes
and 284,502 frames at 20 Hz, with no overlap with the 1,800-episode evaluation
population. It has two directory-level representations:

- `pi05_raw/`: LeRobot v2.1 convention used by π0.5 (`[-1, 1]` environment
  gripper action, `-1` open).
- `fastwam_gripperfixed/`: FastWAM convention (`[0, 1]`, `1` open).

The benchmark-phase-1800 training set is deliberately excluded because it
overlaps the evaluation population.

## Reproducibility

The exact evaluation protocol and aggregate numbers are documented in
[`docs/formal_dynamic_eval.md`](formal_dynamic_eval.md),
[`docs/reproducibility_notes.md`](reproducibility_notes.md), and the tables in
[`results/`](../results/). The local audit record is not required to download
or use the public release.
