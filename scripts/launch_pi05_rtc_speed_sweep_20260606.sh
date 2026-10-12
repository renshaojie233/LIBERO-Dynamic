#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-python3}
OPENPI_HOST=${OPENPI_HOST:-127.0.0.1}
OPENPI_PORT=${OPENPI_PORT:-8124}
POLICY_SERVER_COUNT=${POLICY_SERVER_COUNT:-1}
NUM_WORKERS=${NUM_WORKERS:-1}
MODEL_INPUT_SIZE=${MODEL_INPUT_SIZE:-128}
VIDEO_FPS=${VIDEO_FPS:-60}
RENDER_EVERY=${RENDER_EVERY:-1}
MAX_STEPS=${MAX_STEPS:-600}
CALIBRATED_LATENCY_SECONDS=${CALIBRATED_LATENCY_SECONDS:-0.06}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-"$REPO/outputs/pi05_rtc_speed_sweep_${STAMP}"}

export PYTHONPATH="$REPO:$REPO/scripts"
export LIBERO_CONFIG_PATH="$REPO/libero/libero/configs"
export MUJOCO_GL=egl
export LIBERO_DYNAMIC_AUTO_FINALIZE=false
export LIBERO_EGL_DEVICE_IDS="${LIBERO_EGL_DEVICE_IDS:-0}"

mkdir -p "$ROOT"
echo "$ROOT" > /tmp/libero_latest_pi05_rtc_speed_sweep_root

patch_speed_config() {
  local speed="$1" config_dir="$2" patch_json="$3"
  mkdir -p "$config_dir"
  cp "$REPO"/libero/libero/dynamic_configs/libero_dynamic_l*.yaml "$config_dir"/
  $PY - "$config_dir" "$speed" "$patch_json" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
speed = float(sys.argv[2])
patch_json = Path(sys.argv[3])
count = 0

def walk(value):
    global count
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "alpha_mps":
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
    data["description"] = str(data.get("description", "")) + f" | alpha_mps={speed:.3f}"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
patch_json.write_text(json.dumps({"alpha_mps": speed, "patched_fields": count}, indent=2))
print(json.dumps({"alpha_mps": speed, "patched_fields": count}), flush=True)
PY
}

summarize_run() {
  local out_dir="$1" speed_label="$2" speed="$3"
  $PY - "$out_dir" "$speed_label" "$speed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
speed_label = sys.argv[2]
speed = float(sys.argv[3])
rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
summary = {
    "mode": "rtc_queue",
    "entrypoint": "scripts/run_pi05_rtc_eval.py",
    "strategy": "delay_aware_replace_overlap",
    "speed_label": speed_label,
    "alpha_mps": speed,
    "count": len(rows),
    "success": sum(row.get("success") is True for row in rows),
    "diag_false": sum((row.get("diagnostics") or {}).get("ok") is False for row in rows),
    "rtc_conditioning_supported": sorted({
        bool(chunk.get("metadata", {}).get("rtc_conditioning_supported"))
        for row in rows
        for chunk in row.get("ready_chunks", [])
        if isinstance(chunk, dict)
    }),
}
summary["success_rate"] = summary["success"] / summary["count"] if summary["count"] else 0.0
root.joinpath("speed_control_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY
}

run_one() {
  local speed_label="$1" speed="$2" extra_args=("${@:3}")
  local speed_root="$ROOT/speed_${speed_label}"
  local config_dir="$speed_root/config"
  patch_speed_config "$speed" "$config_dir" "$speed_root/speed_patch.json"
  cd "$REPO"
  $PY scripts/run_pi05_rtc_eval.py \
    --config-dir "$config_dir" \
    --output-dir "$speed_root/run" \
    --levels l1 l2 l3 \
    --traj-indices 0 1 2 3 4 \
    "${extra_args[@]}" \
    --num-workers "$NUM_WORKERS" \
    --policy openpi_ws \
    --openpi-host "$OPENPI_HOST" \
    --openpi-port "$OPENPI_PORT" \
    --policy-server-count "$POLICY_SERVER_COUNT" \
    --latency-source fixed \
    --calibrated-latency-seconds "$CALIBRATED_LATENCY_SECONDS" \
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
  summarize_run "$speed_root/run" "$speed_label" "$speed"
}

run_smoke() {
  local old_root="$ROOT"
  ROOT="$old_root/smoke" NUM_WORKERS=1 MAX_STEPS=80 run_one "006" "0.06" --limit 1
  ROOT="$old_root"
}

run_full() {
  run_one "003" "0.03"
  run_one "006" "0.06"
  run_one "009" "0.09"
}

echo "[rtc root] $ROOT"
if [ "${RUN_SMOKE:-1}" = "1" ]; then
  run_smoke
fi
if [ "${RUN_FULL:-1}" = "1" ]; then
  run_full
fi
echo "[rtc done] $ROOT"
