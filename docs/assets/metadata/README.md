# LIBERO-Dynamic Supplementary Material

Open `index.html` for the compact black/white/gray gallery.

## Dataset coverage

- 40 strict-success videos: exactly one for every benchmark task.
- Four collapsed categories: Long-horizon, Goal, Object, and Spatial; ten tasks each.
- Motion selection is balanced across the page: 14 Linear,
  13 Curvilinear, and 13 Zigzag.
- All selected task-coverage videos use 0.09 m/s object speed and 20 Hz video.
- `assets/videos/dataset/` contains the selected 40-task gallery.
- `assets/metadata/task_coverage_manifest.json` and `.csv` record every selection.

The earlier twelve full-trajectory annotation examples remain under
`assets/videos/trajectory_overlays/` for appendix reuse, but they are intentionally not expanded
on the compact landing page.

## Execution under latency

`assets/videos/execution_protocols/` contains four representative successful runs:

- Sync: Pi0.5 PyTorch, K=7, 147.199 ms. The video preserves inference-hold
  frames so the stop-and-go behavior is directly visible.
- Async: Pi0.5 PyTorch, K=6, 147.199 ms.
- RTC: Pi0.5 PyTorch RTC, K=16, 169.802 ms; RTC conditioning was verified.
- VLASH: VLASH continuous, K=5, 103.987 ms.

Sync, Async, and RTC are matched to the same task and model family. VLASH is a
co-designed model/controller with its own measured latency, so the gallery is a
qualitative system comparison rather than a controller-only causal ablation.

## After dynamic-data fine-tuning

`assets/videos/trained_models/` contains 2 successful evaluation
examples from models trained with strict non-benchmark dynamic data:

- π0.5 dynamic+static fine-tuning: Async K=4, 60.294 ms, L10-03 at 0.09 m/s.
- History FastWAM-Joint dynamic fine-tuning: Async K=8, 89.2 ms, Go-02 at 0.09 m/s.

The exact files and media hashes are recorded in
`assets/metadata/trained_model_manifest.json` and `.csv`.

## Real-robot executions

`assets/videos/real_robot/` contains 8
successful clips grouped into 4 task playlists. Each task
occupies one card on the website; multiple trials advance automatically in the
same player. All packaged filenames and page labels are English.

The web copies retain the source aspect ratio and H.264 encoding, use a maximum
width of 960 px, and receive no color grading. See
`assets/metadata/real_robot_manifest.json` and `.csv` for media properties.

## Data-generation code

`assets/code/data_generation/` contains the release-side dynamic demonstration
generator, phase-partition auditor, direct motion/environment support scripts,
and complete reviewed L1/L2/L3 trajectory configurations. Start with
`assets/code/data_generation/CODE_OVERVIEW.md`; file roles and SHA-256 values
are recorded in `assets/code/data_generation/SOURCE_FILES.json`.

## Evaluation code

`assets/code/evaluation/libero_dynamic_eval_supplement.zip` is the validated,
self-contained evaluation release. It includes the runtime, Sync/Async/RTC/VLASH
execution protocols, model adapters, experiment registry, audit tools, and its
own internal SHA-256 manifest. See `assets/code/evaluation/README.md` for the
archive digest and validation status.

## Integrity

Run `shasum -a 256 -c assets/metadata/SHA256SUMS` from this directory. It covers
all packaged assets except the checksum file itself.
