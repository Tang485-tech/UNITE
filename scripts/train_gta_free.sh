#!/usr/bin/env bash
# UNITE GTA-free Universal Detector launcher — edit the vars below to change settings.
set -euo pipefail

# ---------- editable settings ----------
# GPU device IDs (leave empty for single-GPU via --device)
GPUS="2,3,4,5,6,7"

# number of processes (set to 1 for single-GPU, or = ${#GPUS[@]} for multi)
NPROC=6

# path to config
CONFIG="configs/unite_ffpp_c23_gta_free.yaml"

# resume from checkpoint (leave empty to train from scratch)
RESUME="/home/tangbo/code/projects/UNITE/outputs/ffpp_c23/gta_free/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0.5/lr0.0001_effbs32_gc1_amp0/gta_free_exp01/last.ckpt"

# last-level output directory name; leave empty to use seed{seed}
RUN_NAME="gta_free_exp01"

# smoke test: set to a small number (e.g. 20) for quick pipeline check
# leave empty for full training
MAX_TRAIN_STEPS=""
MAX_VAL_STEPS=""

# single-GPU device (only used when NPROC=1 or GPUS is empty)
DEVICE="cuda:0"

# log directory (leave empty to print to terminal only)
LOG_DIR="logs"

# NCCL collective timeout in seconds (default 600, increase for slow validation)
NCCL_TIMEOUT=1800
# ---------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# log file
if [ -n "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/train_gta_free_${TIMESTAMP}.log"
    echo "Logging to: $LOG_FILE"
fi

if [ "$NPROC" -gt 1 ] && [ -n "$GPUS" ]; then
    echo "Launching GTA-free DDP training on GPUs: $GPUS (nproc=$NPROC)"
    EXTRA_ARGS=()
    [ -n "$RESUME" ]          && EXTRA_ARGS+=(--resume "$RESUME")
    [ -n "$RUN_NAME" ]        && EXTRA_ARGS+=(--run_name "$RUN_NAME")
    [ -n "$MAX_TRAIN_STEPS" ] && EXTRA_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
    [ -n "$MAX_VAL_STEPS" ]   && EXTRA_ARGS+=(--max_val_steps "$MAX_VAL_STEPS")
    if [ -n "$LOG_DIR" ]; then
        TORCH_NCCL_BLOCKING_WAIT=1 \
        NCCL_TIMEOUT=$NCCL_TIMEOUT \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" train.py \
            --config "$CONFIG" \
            "${EXTRA_ARGS[@]}" \
            > >(tee -a "$LOG_FILE") 2>&1
    else
        CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" train.py \
            --config "$CONFIG" \
            "${EXTRA_ARGS[@]}"
    fi
else
    echo "Launching GTA-free single-GPU training on: $DEVICE"
    if [ -n "$LOG_DIR" ]; then
        python train.py \
            --config "$CONFIG" \
            --device "$DEVICE" \
            ${RESUME:+--resume "$RESUME"} \
            ${RUN_NAME:+--run_name "$RUN_NAME"} \
            ${MAX_TRAIN_STEPS:+--max_train_steps "$MAX_TRAIN_STEPS"} \
            ${MAX_VAL_STEPS:+--max_val_steps "$MAX_VAL_STEPS"} \
            > >(tee -a "$LOG_FILE") 2>&1
    else
        python train.py \
            --config "$CONFIG" \
            --device "$DEVICE" \
            ${RESUME:+--resume "$RESUME"} \
            ${RUN_NAME:+--run_name "$RUN_NAME"} \
            ${MAX_TRAIN_STEPS:+--max_train_steps "$MAX_TRAIN_STEPS"} \
            ${MAX_VAL_STEPS:+--max_val_steps "$MAX_VAL_STEPS"}
    fi
fi
