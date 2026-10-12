# Evaluation Code Map

The release builder copies the following files from the repository. These are
the implementations that define the paper's benchmark behavior.

| Public release path | Repository source | Role |
|---|---|---|
| `code/eval_dynamic_dummy_policy.py` | `scripts/eval_dynamic_dummy_policy.py` | Canonical item enumeration and generator/evaluator diagnostics shared by the data pipeline |
| `code/eval_dynamic_libero_style.py` | `scripts/eval_dynamic_libero_style.py` | Environment construction, canonical item enumeration, diagnostics, and policy I/O |
| `code/render_dynamic_config_previews.py` | `scripts/render_dynamic_config_previews.py` | Canonical physical-motion implementation and no-policy visual diagnostics |
| `code/run_realtime_eval.py` | `scripts/run_realtime_eval.py` | Generic evaluation entry point |
| `code/run_pi05_sync_eval.py` | `scripts/run_pi05_sync_eval.py` | Sync wrapper |
| `code/run_pi05_async_eval.py` | `scripts/run_pi05_async_eval.py` | Async wrapper |
| `code/run_pi05_rtc_eval.py` | `scripts/run_pi05_rtc_eval.py` | RTC wrapper |
| `code/realtime_eval/control.py` | `scripts/realtime_eval/control.py` | Action queue, frozen-latency sampler, and controller base |
| `code/realtime_eval/runtime.py` | `scripts/realtime_eval/runtime.py` | Continuous-world episode runtime and trace recording |
| `code/realtime_eval/runner.py` | `scripts/realtime_eval/runner.py` | Sharding, CLI, manifests, and summaries |
| `code/realtime_eval/strategies/sync_hold.py` | `scripts/realtime_eval/strategies/sync_hold.py` | Sync scheduling rule |
| `code/realtime_eval/strategies/async_queue.py` | `scripts/realtime_eval/strategies/async_queue.py` | Async scheduling and weighted overlap |
| `code/realtime_eval/strategies/rtc_queue.py` | `scripts/realtime_eval/strategies/rtc_queue.py` | RTC conditioning and delay-aware replacement |
| `adapters/serve_lerobot_pi05_openpi_ws.py` | `scripts/serve_lerobot_pi05_openpi_ws.py` | Pi0.5 PyTorch and RTC policy adapter |
| `adapters/serve_fastwam_joint_openpi_ws.py` | `scripts/serve_fastwam_joint_openpi_ws.py` | FastWAM-Joint family adapter |
| `adapters/serve_openpi_jax_ws.py` | public OpenPI policy server | Pi0.5 JAX and Pi0.5 Dynamic-FT adapter |
| `adapters/serve_pi0_jax_ws.py` | public OpenPI policy server | Pi0 JAX adapter |
| `adapters/serve_vla_jepa_ws.py` | public LeRobot integration | VLA-JEPA adapter with checkpoint audit |
| `adapters/serve_openvla_family_ws.py` | public OpenVLA integration | OpenVLA and OpenVLA-OFT adapter |
| `adapters/serve_native_fastwam_ws.py` | public FastWAM integration | Native unconditioned FastWAM adapter |
| `adapters/serve_vlash_continuous_ws.py` | public VLASH integration | VLASH policy endpoint |
| `adapters/run_vlash_continuous_eval.py` | frozen paper integration | Registers the causal continuous-world VLASH controller |
| `adapters/vlash_reference_repo/scripts/realtime_eval/` | frozen paper runtime | Pinned controller interface used by the corrected VLASH run |

Model implementations and checkpoints remain external dependencies; the
benchmark never embeds or redistributes private checkpoint locations.

## Dataset and gallery code

| Public path | Role |
| --- | --- |
| `data_generation/generate_dynamic_demos.py` | Generates strictly validated dynamic demonstrations by compensating verified static templates. |
| `data_generation/build_phase_partition.py` | Reconstructs the exact benchmark phase, complementary non-benchmark phases, and compact training split. |
| `code/dynamic_motion_invariants.py` | Rejects cross-family L1/L2/L3 motion overrides before preview, generation, or evaluation. |
| `code/rebuild_dynamic_configs_from_manual_json.py` | Rebuilds reviewed configs while keeping legacy motion replicas synchronized. |
| `task_gallery/scripts/build_task_gallery.py` | Verifies 12 reviewed clips and regenerates 96 agent-view stills plus 12 contact sheets. |
| `task_gallery/selection.json` | Freezes the appendix task/motion selection and source checksums. |
| `task_gallery/scripts/build_full_video_gallery.py` | Deterministically selects and verifies all 40 tasks by three motion families at 0.09 m/s. |
| `task_gallery/full_video_gallery/manifest.json` | Records the 120-video coverage matrix, exact source selections, media metadata, and checksums. |

The continuous-world VLASH code is pinned separately because it extends the
controller interface with request-time state prediction and request-event
auditing. Its protocol tag is
`vlash_continuous_libero_free_space_osc_v2`. Historical VLASH code using a
delayed image paired with a later true robot state is deliberately excluded.
