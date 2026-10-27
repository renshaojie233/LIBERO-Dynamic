# History-frame FastWAM-Joint release package

This directory contains the history-frame FastWAM-Joint training snapshot and
its aligned LIBERO-Dynamic inference adapter. It is the code package requested
for the 60K-step history-frame Joint checkpoint and the input protocol used by
the paper's history-frame evaluations.

## Input contract

The released history configuration uses:

- five consecutive observations: the four preceding observations and the current
  observation (`num_history_frames=5`, `history_frame_stride=1`);
- two 224x224 camera views concatenated horizontally into a 224x448 frame;
- a 32-step action window with the preceding 16 executed actions as a clean
  prefix (`action_clean_prefix_steps=16`) and a 16-step future action horizon;
- video sampling at offsets `[-4, -3, -2, -1, 0, 4, 8, 12, 16]` for the
  current prediction window (`action_video_freq_ratio=4`).

The websocket adapter receives the history explicitly through
`observation/history/image`, `observation/history/wrist_image`,
`observation/history/state`, and `observation/action_history`. It does not
reconstruct the executed-action prefix from predicted actions.

## Contents

- `src/fastwam/`: the FastWAM model, Joint model, dataset/loader, processor,
  trainer, runtime, and supporting modules used by the history-frame run;
- `configs/task/libero_joint_2cam224_history5_dense4_fixed3latent_clean16_1e-4.yaml`:
  the exact history-frame task configuration;
- `configs/data/libero_dynamic_gripperfixed.yaml` and
  `configs/data/libero_mixed_gripperfixed.yaml`: the dynamic-only and mixed
  training data configurations;
- `configs/model/fastwam_joint.yaml` and shared `sim_libero.yaml`/`train.yaml`;
- `serve_fastwam_joint_history_frame_openpi_ws.py`: the aligned websocket
  inference adapter used for the history-frame model.

The adapter expects a FastWAM source checkout at `--fastwam-root` with the
usual `experiments/libero/eval_libero_single.py` evaluation helpers and the
Wan/ActionDiT base weights. The released `src/fastwam` code is the history
training package; it does not redistribute the large Wan base weights.

Example adapter launch:

```bash
python serve_fastwam_joint_history_frame_openpi_ws.py \
  --fastwam-root /path/to/fastwam \
  --checkpoint /path/to/history_frame_checkpoint.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --task-config libero_joint_2cam224_history5_dense4_fixed3latent_clean16_1e-4 \
  --actions-per-chunk 8 \
  --num-inference-steps 1 \
  --port 8124
```

The data configuration retains the absolute paths from the training run as
provenance. Replace `dataset_dirs` and `text_embedding_cache_dir` with local
paths before training on another machine. The released strict non-benchmark
training data is linked from the repository release index; the evaluation
population is not included in that dataset release.

## Provenance

The source snapshot was copied from the training environment used for the
paper's history-frame Joint runs:

- training code root:
  `/home/dataset-local/rsj/training/fastwam-ft-continue-20260914/code`;
- aligned adapter source:
  `/home/dataset-local/rsj/libero-dynamic-eval-workspace/runs/fastwam_joint_histft3000_async_k5-k10_tianyi_20260922/code_snapshot/serve_fastwam_joint_openpi_ws.py`;
- history task:
  `libero_joint_2cam224_history5_dense4_fixed3latent_clean16_1e-4`.

The dynamic-only and mixed Joint checkpoints themselves are hosted in the
Hugging Face model repository under the `fastwam_joint/history_frame/` paths.
