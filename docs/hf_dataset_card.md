---
tags:
- robotics
- libero-dynamic
- lerobot
- nonbenchmark
---

# LIBERO-Dynamic strict non-benchmark training data

This dataset contains the organized strict non-benchmark dynamic training set
used by the paper's fine-tuning experiments: **1,800 episodes and 284,502
frames at 20 Hz**. It has no overlap with the 1,800-episode evaluation
population.

Two representations of the same episode set are provided:

- `pi05_raw/`: LeRobot v2.1 convention for π0.5, with environment gripper
  actions in `[-1, 1]` (`-1` means open).
- `fastwam_gripperfixed/`: FastWAM convention, with gripper actions in
  `[0, 1]` (`1` means open).

The benchmark-phase-1800 training set is deliberately excluded because it
overlaps the evaluation population. The corresponding nine model checkpoints
are in the [paper model repository](https://huggingface.co/rsj2003/libero-dynamic-paper-models).
For code and protocol details, see the [LIBERO-Dynamic GitHub repository](https://github.com/renshaojie233/LIBERO-Dynamic).
