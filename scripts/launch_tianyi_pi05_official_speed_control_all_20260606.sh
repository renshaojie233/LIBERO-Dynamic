#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-"$REPO/outputs/pi05_official_control_sweep_${STAMP}"}
NUM_WORKERS=${NUM_WORKERS:-1}
POLICY_SERVER_COUNT=${POLICY_SERVER_COUNT:-1}
PORTS=${PORTS:-8124}
GPUS=${GPUS:-0}
LIBERO_EGL_DEVICE_IDS=${LIBERO_EGL_DEVICE_IDS:-0}
PY=${PY:-python3}
SERVER_PY=${SERVER_PY:-python3}
: "${CLEAN_CHECKPOINT:?Set CLEAN_CHECKPOINT to a local LeRobot Pi0.5 checkpoint directory}"
: "${PALIGEMMA_LOCAL_TOKENIZER:?Set PALIGEMMA_LOCAL_TOKENIZER to a local PaliGemma tokenizer directory}"
RTC_CHECKPOINT=${RTC_CHECKPOINT:-$CLEAN_CHECKPOINT}

mkdir -p "$ROOT/logs"
echo "$ROOT" > /tmp/libero_latest_tianyi_pi05_official_control_sweep_root

stop_policy_servers() {
  local old_dir
  old_dir="$(cat /tmp/libero_latest_lerobot_pi05_ws_log_dir 2>/dev/null || true)"
  if [ -z "$old_dir" ]; then
    return
  fi
  for pid_file in "$old_dir"/*.pid; do
    [ -f "$pid_file" ] || continue
    local pid
    pid="$(cat "$pid_file")"
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
      kill "$child" 2>/dev/null || true
    done
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
}

wait_policy_servers() {
  local expected="${1:-4}"
  local log_dir
  log_dir="$(cat /tmp/libero_latest_lerobot_pi05_ws_log_dir)"
  for _ in $(seq 1 120); do
    local ready
    ready="$({ grep -l "serving" "$log_dir"/server_*.log 2>/dev/null || true; } | wc -l)"
    if [ "$ready" -eq "$expected" ]; then
      return
    fi
    sleep 5
  done
  echo "policy servers did not become ready; log_dir=$log_dir" >&2
  tail -n 80 "$log_dir"/server_*.log >&2 || true
  exit 1
}

start_policy_servers() {
  local checkpoint="$1"
  local enable_rtc="$2"
  local name="$3"
  stop_policy_servers
  (
    cd "$REPO"
    CHECKPOINT="$checkpoint" \
    SERVER_PY="$SERVER_PY" \
    PALIGEMMA_LOCAL_TOKENIZER="$PALIGEMMA_LOCAL_TOKENIZER" \
    PORTS="$PORTS" \
    GPUS="$GPUS" \
    ENABLE_RTC="$enable_rtc" \
    LOG_DIR="$ROOT/logs/${name}_policy_servers" \
      bash scripts/launch_lerobot_pi05_ws_servers_20260606.sh
  )
  wait_policy_servers "$POLICY_SERVER_COUNT"
}

run_mode() {
  local mode="$1"
  local script="$2"
  echo "[run] mode=$mode root=$ROOT/$mode"
  (
    cd "$REPO"
    ROOT="$ROOT/$mode" \
    PY="$PY" \
    NUM_WORKERS="$NUM_WORKERS" \
    POLICY_SERVER_COUNT="$POLICY_SERVER_COUNT" \
    RUN_SMOKE=0 \
    RUN_FULL=1 \
      bash "$script"
  )
}

echo "[root] $ROOT"
echo "[config] NUM_WORKERS=$NUM_WORKERS POLICY_SERVER_COUNT=$POLICY_SERVER_COUNT PORTS=$PORTS GPUS=$GPUS LIBERO_EGL_DEVICE_IDS=$LIBERO_EGL_DEVICE_IDS"

start_policy_servers "$CLEAN_CHECKPOINT" 0 "clean"
run_mode sync_hold scripts/launch_pi05_sync_speed_sweep_20260606.sh
run_mode async_queue scripts/launch_pi05_async_speed_sweep_20260606.sh

start_policy_servers "$RTC_CHECKPOINT" 1 "rtc"
run_mode rtc_queue scripts/launch_pi05_rtc_speed_sweep_20260606.sh

stop_policy_servers
echo "[done] $ROOT"
