#!/usr/bin/env python3
"""Build and audit the final LIBERO-Dynamic phase partition manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def condition_key(row: dict[str, Any]) -> tuple[int, float, int, int]:
    return (
        int(row["task"]),
        round(float(row["speed_mps"]), 2),
        int(row["level"]),
        int(row["trajectory_index"]),
    )


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "suite": dict(sorted(Counter(str(r["suite"]) for r in rows).items())),
        "task": dict(sorted(Counter(f"task{int(r['task']):02d}" for r in rows).items())),
        "speed": dict(sorted(Counter(f"{float(r['speed_mps']):.2f}" for r in rows).items())),
        "level": dict(sorted(Counter(f"l{int(r['level'])}" for r in rows).items())),
        "trajectory": dict(sorted(Counter(f"traj{int(r['trajectory_index']):02d}" for r in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-benchmark-selection", type=Path, required=True)
    parser.add_argument("--phase-catalog", type=Path, required=True)
    parser.add_argument("--source-index-root", type=Path, required=True)
    parser.add_argument("--training-selection", type=Path, required=True)
    parser.add_argument("--exact-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    legacy_doc = read_json(args.legacy_benchmark_selection)
    legacy_rows = legacy_doc["episodes"]
    catalog_doc = read_json(args.phase_catalog)
    catalog = {str(r["item_id"]): r for r in catalog_doc["records"]}
    train_doc = read_json(args.training_selection)
    train_rows = train_doc["episodes"]
    exact_doc = read_json(args.exact_selection)
    exact_rows = exact_doc["episodes"]

    source_rows: list[dict[str, Any]] = []
    source_slots: list[dict[str, Any]] = []
    source_indices = []
    for suite in SUITES:
        path = args.source_index_root / suite / "provenance" / "source_index.json"
        document = read_json(path)
        if document.get("episode_count") != 2250 or document.get("error_count") != 0:
            raise RuntimeError(f"Invalid source index: {path}")
        source_rows.extend(document["episodes"])
        source_slots.extend(document["slots"])
        source_indices.append({"suite": suite, "path": str(path), "sha256": sha256(path)})

    if len(source_rows) != 9000 or len({str(r["item_id"]) for r in source_rows}) != 9000:
        raise RuntimeError("Source release is not exactly 9,000 unique episodes")
    if len(legacy_rows) != 1800 or len(exact_rows) != 1800 or len(train_rows) != 1800:
        raise RuntimeError("All 1,800-row inputs must contain exactly 1,800 episodes")

    grouped: defaultdict[tuple[int, float, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[condition_key(row)].append(row)
    if len(grouped) != 1800 or Counter(map(len, grouped.values())) != {5: 1800}:
        raise RuntimeError("Expected 1,800 conditions with five source phases each")

    legacy_by_condition = {condition_key(r): r for r in legacy_rows}
    exact_by_condition = {condition_key(r): r for r in exact_rows}
    train_by_condition = {condition_key(r): r for r in train_rows}
    if not (set(grouped) == set(legacy_by_condition) == set(exact_by_condition) == set(train_by_condition)):
        raise RuntimeError("Condition grids differ among source, exact, legacy, and training sets")

    legacy_ids = {str(r["item_id"]) for r in legacy_rows}
    nonbenchmark_rows = [r for r in source_rows if str(r["item_id"]) not in legacy_ids]
    if len(nonbenchmark_rows) != 7200 or len({str(r["item_id"]) for r in nonbenchmark_rows}) != 7200:
        raise RuntimeError("Non-benchmark complement is not exactly 7,200 unique episodes")
    nonbenchmark_ids = {str(r["item_id"]) for r in nonbenchmark_rows}
    train_ids = {str(r["item_id"]) for r in train_rows}
    if not train_ids <= nonbenchmark_ids:
        raise RuntimeError("Training 1,800 is not a subset of non-benchmark 7,200")
    if train_ids & legacy_ids:
        raise RuntimeError("Training set overlaps legacy pseudo benchmark selection")

    enriched = []
    for row in sorted(nonbenchmark_rows, key=lambda r: condition_key(r) + (str(r["item_id"]),)):
        item_id = str(row["item_id"])
        phase = catalog[item_id]
        key = condition_key(row)
        enriched.append(
            {
                **row,
                "partition": "nonbenchmark_phase",
                "legacy_pseudo_benchmark_excluded_item_id": str(legacy_by_condition[key]["item_id"]),
                "phase_prime_seconds_actual": float(phase["phase_prime_seconds_actual"]),
                "raw_phase_catalog": phase,
            }
        )

    slot_lookup = {(int(r["task"]), f"{float(r['speed_mps']):.2f}"): r for r in source_slots}
    by_slot: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_slot[(int(row["task"]), f"{float(row['speed_mps']):.2f}")].append(row)
    if len(by_slot) != 120 or Counter(map(len, by_slot.values())) != {60: 120}:
        raise RuntimeError("Expected 120 raw slots with 60 non-benchmark episodes each")
    slots = [{**slot_lookup[key], "selected_episodes": len(by_slot[key])} for key in sorted(by_slot)]

    expected_dist = {
        "suite": {suite: 1800 for suite in SUITES},
        "task": {f"task{i:02d}": 180 for i in range(1, 41)},
        "speed": {"0.03": 2400, "0.06": 2400, "0.09": 2400},
        "level": {"l1": 2400, "l2": 2400, "l3": 2400},
        "trajectory": {f"traj{i:02d}": 1440 for i in range(5)},
    }
    actual_dist = distribution(enriched)
    if actual_dist != expected_dist:
        raise RuntimeError(f"Unexpected 7,200 distribution: {actual_dist}")

    out = args.output_root
    manifest = {
        "format": "libero_dynamic_final_phase_partition_v1",
        "release_name": "libero_dynamic_final_phase_partition_20260911",
        "semantics": {
            "benchmark_phase_1800": "New exact benchmark initialization: init-state-id 0 and exactly 40 control steps (2.0 s at 20 Hz) before policy execution; no nearest-phase approximation.",
            "nonbenchmark_phase_7200": "Complement of the deprecated nearest-phase 1,800 selector inside the reviewed original 9,000-episode source release: four successful source phases per condition.",
            "train_nonbenchmark_phase_1800": "Deterministic phase-balanced subset of nonbenchmark_phase_7200; one episode per condition and 450 selections from each of the four phase-time ordinals.",
            "combined_full_9000": "Logical union of the new exact benchmark_phase_1800 and nonbenchmark_phase_7200. The two components use the same 1,800-condition grid but distinct phase-generation contracts.",
        },
        "counts": {
            "benchmark_phase_1800": 1800,
            "nonbenchmark_phase_7200": 7200,
            "train_nonbenchmark_phase_1800": 1800,
            "combined_full_9000": 9000,
        },
        "invariants": {
            "condition_count": 1800,
            "source_phases_per_condition": 5,
            "nonbenchmark_phases_per_condition": 4,
            "training_episodes_per_condition": 1,
            "training_is_subset_of_nonbenchmark": True,
            "training_legacy_benchmark_item_overlap": 0,
            "exact_condition_grid_matches_nonbenchmark": True,
            "deprecated_pseudo_benchmark_must_not_be_used_as_test_data": True,
        },
        "release_artifacts": {
            "benchmark_phase_1800_raw": "benchmark_phase_1800_raw",
            "benchmark_phase_1800_lerobot": "benchmark_phase_1800_lerobot",
            "nonbenchmark_phase_7200_raw": "nonbenchmark_phase_7200_raw",
            "nonbenchmark_phase_7200_lerobot": "nonbenchmark_phase_7200_lerobot",
            "train_nonbenchmark_phase_1800_raw": "train_nonbenchmark_phase_1800_raw",
            "train_nonbenchmark_phase_1800_lerobot": "train_nonbenchmark_phase_1800_lerobot",
        },
        "inputs": {
            "legacy_pseudo_benchmark_selection": {"file": args.legacy_benchmark_selection.name, "sha256": sha256(args.legacy_benchmark_selection)},
            "phase_catalog": {"file": args.phase_catalog.name, "sha256": sha256(args.phase_catalog)},
            "training_selection": {"file": args.training_selection.name, "sha256": sha256(args.training_selection)},
            "exact_selection": {"file": args.exact_selection.name, "sha256": sha256(args.exact_selection)},
            "source_indices": source_indices,
        },
        "distribution": {
            "benchmark_phase_1800": exact_doc["distribution"],
            "nonbenchmark_phase_7200": actual_dist,
            "train_nonbenchmark_phase_1800": train_doc["distribution"],
        },
    }
    nonbenchmark = {
        "format": "libero_dynamic_nonbenchmark_phase_selection_v1",
        "selection_semantics": manifest["semantics"]["nonbenchmark_phase_7200"],
        "source_episode_count": 9000,
        "excluded_legacy_pseudo_benchmark_episode_count": 1800,
        "selected_episode_count": 7200,
        "selected_slot_count": 120,
        "phases_per_condition": 4,
        "distribution": actual_dist,
        "legacy_pseudo_benchmark_selection_sha256": sha256(args.legacy_benchmark_selection),
        "phase_catalog_sha256": sha256(args.phase_catalog),
        "source_indices": source_indices,
        "slots": slots,
        "episodes": enriched,
    }
    write_json(out / "nonbenchmark_selection_7200.json", nonbenchmark)
    write_json(out / "release_manifest.json", manifest)
    write_json(
        out / "partition_audit.json",
        {
            "status": "pass",
            "source_9000": len(source_rows),
            "legacy_pseudo_benchmark_excluded": len(legacy_ids),
            "nonbenchmark_7200": len(nonbenchmark_ids),
            "exact_benchmark_1800": len(exact_rows),
            "train_nonbenchmark_1800": len(train_ids),
            "training_subset_of_nonbenchmark": train_ids <= nonbenchmark_ids,
            "training_legacy_overlap": len(train_ids & legacy_ids),
            "condition_grid_count": len(grouped),
            "conditions_with_four_nonbenchmark_phases": sum(
                sum(str(r["item_id"]) not in legacy_ids for r in rows) == 4
                for rows in grouped.values()
            ),
        },
    )
    print(json.dumps({"status": "pass", "output": str(out), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
