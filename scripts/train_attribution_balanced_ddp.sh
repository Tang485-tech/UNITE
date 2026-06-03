#!/usr/bin/env bash
# UNITE E1 class-balanced attribution run (8-way DDP, one process per GPU).
set -euo pipefail

# ---------------- editable settings ----------------
GPUS="0,1,2,3,4,5,6,7"
CONFIG="configs/unite_ffpp_c23_e1_balanced.yaml"
RUN_NAME="e1_balanced"
RESUME="/home/tangbo/code/projects/UNITE/outputs/ffpp_c23/e1_balanced/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0.5/lr0.0001_effbs32_gc1_amp0/e1_balanced/last.ckpt"
PORT_START=29500
NCCL_TIMEOUT=1800
LOG_DIR="logs"
# ----------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

ngpu() { local IFS=','; set -- $1; echo $#; }

pick_free_port() {
    local port="${1:-29500}"
    while ss -tlnH "sport = :$port" | grep -q . 2>/dev/null; do
        port=$((port + 1))
    done
    echo "$port"
}

NPROC="$(ngpu "$GPUS")"
MASTER_PORT="$(pick_free_port "$PORT_START")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_${TIMESTAMP}.log"
EXTRA_ARGS=(--run_name "$RUN_NAME")

if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    EXTRA_ARGS+=(--resume "$RESUME")
    echo "Launching $RUN_NAME on GPUs [$GPUS] (nproc=$NPROC, resume=$RESUME)"
else
    echo "Launching $RUN_NAME on GPUs [$GPUS] (nproc=$NPROC, from scratch)"
fi
echo "Log: $LOG_FILE"

TORCH_NCCL_BLOCKING_WAIT=1 \
NCCL_TIMEOUT="$NCCL_TIMEOUT" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train.py \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    > >(tee -a "$LOG_FILE") 2>&1
