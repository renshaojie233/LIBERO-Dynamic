#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
SERVER_PY=${SERVER_PY:-python3}
: "${CHECKPOINT:?Set CHECKPOINT to a local LeRobot Pi0.5 checkpoint directory}"
: "${PALIGEMMA_LOCAL_TOKENIZER:?Set PALIGEMMA_LOCAL_TOKENIZER to a local PaliGemma tokenizer directory}"
PORTS=${PORTS:-8124}
GPUS=${GPUS:-0}
LOG_DIR=${LOG_DIR:-"$REPO/logs/policy_servers_$(date +%Y%m%d_%H%M%S)"}
ENABLE_RTC=${ENABLE_RTC:-0}

mkdir -p "$LOG_DIR"
echo "$LOG_DIR" > /tmp/libero_latest_lerobot_pi05_ws_log_dir

read -r -a port_array <<< "$PORTS"
read -r -a gpu_array <<< "$GPUS"
if [ "${#port_array[@]}" -ne "${#gpu_array[@]}" ]; then
  echo "PORTS and GPUS must have the same length" >&2
  exit 2
fi

rtc_args=()
if [ "$ENABLE_RTC" = "1" ]; then
  rtc_args=(--enable-rtc)
fi

for idx in "${!port_array[@]}"; do
  port="${port_array[$idx]}"
  gpu="${gpu_array[$idx]}"
  log="$LOG_DIR/server_${port}_gpu${gpu}.log"
  pid_file="$LOG_DIR/server_${port}.pid"
  echo "[server] gpu=$gpu port=$port log=$log"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PALIGEMMA_LOCAL_TOKENIZER="$PALIGEMMA_LOCAL_TOKENIZER" \
    "$SERVER_PY" scripts/serve_lerobot_pi05_openpi_ws.py \
      --port "$port" \
      --checkpoint "$CHECKPOINT" \
      --local-tokenizer "$PALIGEMMA_LOCAL_TOKENIZER" \
      "${rtc_args[@]}"
  ) >"$log" 2>&1 &
  echo "$!" > "$pid_file"
done

echo "[server log dir] $LOG_DIR"
