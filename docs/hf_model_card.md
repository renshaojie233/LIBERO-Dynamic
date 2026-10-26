---
library_name: pytorch
tags:
- robotics
- libero-dynamic
- fastwam
- pi05
---

# LIBERO-Dynamic paper checkpoints

This repository contains the nine checkpoint versions used in the
LIBERO-Dynamic dynamic-data training study. They are organized under one
repository so that the paper's release has a single model link.

## Contents

| Family | Variants |
|---|---|
| FastWAM-Joint, single frame | static base; dynamic-only FT3000; dynamic+static FT3000 |
| FastWAM-Joint, history frame | dense-history static base; dynamic-only FT3000; dynamic+static FT3000 |
| π0.5 | official static; dynamic-only FT1500; dynamic+static FT3000 |

The FastWAM files are PyTorch checkpoints and include matching
`dataset_stats.json` files where they were part of the run. The π0.5 entries
contain inference parameters, checkpoint metadata, and normalization assets.
Training optimizer state is intentionally omitted: this is an inference
release, not a resume-training snapshot.

The corresponding strict non-benchmark training data is available at
[`rsj2003/libero-dynamic-strict-nonbenchmark-1800`](https://huggingface.co/datasets/rsj2003/libero-dynamic-strict-nonbenchmark-1800).
The code and evaluation protocol are in the [LIBERO-Dynamic GitHub repository](https://github.com/renshaojie233/LIBERO-Dynamic).
