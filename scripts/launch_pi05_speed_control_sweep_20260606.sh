#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-python3}
: "${CALIB:?Set CALIB to a latency calibration JSON with summary.p90_seconds}"
OPENPI_HOST=${OPENPI_HOST:-127.0.0.1}
OPENPI_PORT=${OPENPI_PORT:-8124}
NUM_WORKERS=${NUM_WORKERS:-6}
MODEL_INPUT_SIZE=${MODEL_INPUT_SIZE:-128}
VIDEO_FPS=${VIDEO_FPS:-60}
RENDER_EVERY=${RENDER_EVERY:-1}
MAX_STEPS=${MAX_STEPS:-600}

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-"$REPO/outputs/pi05_speed_control_sweep_${STAMP}"}

export PYTHONPATH="$REPO:$REPO/scripts"
export LIBERO_CONFIG_PATH="$REPO/libero/libero/configs"
export MUJOCO_GL=egl
export LIBERO_DYNAMIC_AUTO_FINALIZE=false

mkdir -p "$ROOT"
echo "$ROOT" > /tmp/libero_latest_pi05_speed_control_sweep_root

patch_speed_config() {
  local speed="$1"
  local config_dir="$2"
  local patch_json="$3"
  mkdir -p "$config_dir"
  cp "$REPO"/libero/libero/dynamic_configs/libero_dynamic_l*.yaml "$config_dir"/
  "$PY" - "$config_dir" "$speed" "$patch_json" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
speed = float(sys.argv[2])
patch_json = Path(sys.argv[3])
count = 0
old_values = {}

def walk(value):
    global count
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key == "alpha_mps":
                old_values[str(child)] = old_values.get(str(child), 0) + 1
                value[key] = speed
                count += 1
            else:
                walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

for path in sorted(config_dir.glob("libero_dynamic_l*.yaml")):
    data = yaml.safe_load(path.read_text())
    walk(data)
    data["description"] = str(data.get("description", "")) + f" | speed sweep alpha_mps={speed:.3f}"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

patch_json.write_text(json.dumps({
    "alpha_mps": speed,
    "patched_fields": count,
    "old_values": old_values,
    "note": "Absolute translation speed sweep. Rotational alpha_yaw_rad_per_s is intentionally unchanged.",
}, indent=2))
print(json.dumps({"alpha_mps": speed, "patched_fields": count}, indent=2), flush=True)
PY
}

run_eval() {
  local mode="$1"
  local speed_label="$2"
  local speed="$3"
  local out_dir="$4"
  local config_dir="$5"
  local limit_args=("${@:6}")

  cd "$REPO"
  "$PY" scripts/run_oldphysics_realtime_eval.py \
    --config-dir "$config_dir" \
    --output-dir "$out_dir" \
    --levels l1 l2 l3 \
    --traj-indices 0 1 2 3 4 \
    "${limit_args[@]}" \
    --num-workers "$NUM_WORKERS" \
    --policy openpi_ws \
    --openpi-host "$OPENPI_HOST" \
    --openpi-port "$OPENPI_PORT" \
    --no-per-worker-policy-endpoint \
    --control-mode "$mode" \
    --prefetch-threshold 5 \
    --empty-action hold \
    --latency-source fixed \
    --calibrated-latency-seconds "$("$PY" - "$CALIB" <<'PY'
import json
import sys
data = json.loads(open(sys.argv[1]).read())
print(f"{float(data['summary']['p90_seconds']):.6f}")
PY
)" \
    --replan-steps 10 \
    --max-steps "$MAX_STEPS" \
    --save-videos \
    --no-save-review-video \
    --save-model-input-videos \
    --model-input-video-layout dual \
    --model-input-video-size "$MODEL_INPUT_SIZE" \
    --video-fps "$VIDEO_FPS" \
    --render-every "$RENDER_EVERY" \
    --record-latency-frames \
    --camera-size 256 \
    --stop-on-success

  "$PY" - "$out_dir" "$mode" "$speed_label" "$speed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
speed_label = sys.argv[3]
speed = float(sys.argv[4])
rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
summary = {
    "mode": mode,
    "speed_label": speed_label,
    "alpha_mps": speed,
    "count": len(rows),
    "ok_status": sum(1 for row in rows if row.get("status") == "ok"),
    "success": sum(1 for row in rows if row.get("success") is True),
    "success_rate": (sum(1 for row in rows if row.get("success") is True) / len(rows)) if rows else 0.0,
    "diag_ok": sum(1 for row in rows if (row.get("diagnostics") or {}).get("ok") is True),
    "diag_false": sum(1 for row in rows if (row.get("diagnostics") or {}).get("ok") is False),
    "by_level": {},
    "by_suite": {},
}
for level in ["l1", "l2", "l3"]:
    part = [row for row in rows if row.get("level") == level]
    summary["by_level"][level] = {
        "count": len(part),
        "success": sum(1 for row in part if row.get("success") is True),
        "success_rate": (sum(1 for row in part if row.get("success") is True) / len(part)) if part else 0.0,
        "diag_false": sum(1 for row in part if (row.get("diagnostics") or {}).get("ok") is False),
    }
for suite in sorted({row.get("suite") for row in rows}):
    part = [row for row in rows if row.get("suite") == suite]
    summary["by_suite"][suite] = {
        "count": len(part),
        "success": sum(1 for row in part if row.get("success") is True),
        "success_rate": (sum(1 for row in part if row.get("success") is True) / len(part)) if part else 0.0,
        "diag_false": sum(1 for row in part if (row.get("diagnostics") or {}).get("ok") is False),
    }
(root / "speed_control_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

run_smoke() {
  local smoke_root="$ROOT/smoke"
  mkdir -p "$smoke_root"
  local config="$smoke_root/config_speed_006"
  patch_speed_config "0.06" "$config" "$smoke_root/speed_patch_006.json"
  for mode in sync_hold async_queue rtc_queue; do
    local out="$smoke_root/${mode}"
    echo "[smoke] mode=$mode speed=0.06 out=$out"
    NUM_WORKERS=1 MAX_STEPS=80 run_eval "$mode" "006" "0.06" "$out" "$config" --limit 1
  done
}

run_full_sweep() {
  for mode in sync_hold async_queue rtc_queue; do
    local mode_root="$ROOT/$mode"
    mkdir -p "$mode_root"
    for pair in "003 0.03" "006 0.06" "009 0.09"; do
      local speed_label="${pair%% *}"
      local speed="${pair##* }"
      local config="$mode_root/config_speed_${speed_label}"
      local out="$mode_root/speed_${speed_label}"
      echo "[full] mode=$mode speed=$speed out=$out"
      patch_speed_config "$speed" "$config" "$mode_root/speed_patch_${speed_label}.json"
      run_eval "$mode" "$speed_label" "$speed" "$out" "$config"
    done
  done
}

write_combined_summary() {
  "$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = []
for path in sorted(root.glob("*/speed_*/speed_control_summary.json")):
    summaries.append(json.loads(path.read_text()))
combined = {
    "root": str(root),
    "runs": summaries,
    "total_items": sum(row.get("count", 0) for row in summaries),
    "total_success": sum(row.get("success", 0) for row in summaries),
}
combined["total_success_rate"] = (
    combined["total_success"] / combined["total_items"] if combined["total_items"] else 0.0
)
(root / "combined_speed_control_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2))
print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)
PY
}

echo "[root] $ROOT"
run_smoke
run_full_sweep
write_combined_summary
echo "[done] $ROOT"
