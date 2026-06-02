#!/usr/bin/env bash
# UNITE in-domain attribution runs (DDP).
#   E1 = class-balance ON  (configs/unite_ffpp_c23_e1_balanced.yaml, run_name e1_balanced)
#   E2 = AD-loss OFF        (configs/unite_ffpp_c23_e2_ceonly.yaml,   run_name e2_ceonly)
#
# Usage:
#   bash scripts/train_attribution_ddp.sh        # run BOTH (per $MODE)
#   bash scripts/train_attribution_ddp.sh e1     # run only E1 (on $GPUS_ALL)
#   bash scripts/train_attribution_ddp.sh e2     # run only E2 (on $GPUS_ALL)
set -euo pipefail

# ---------------- editable settings ----------------
MODE="parallel_shared"        # "parallel_shared" = BOTH jobs on all of $GPUS_ALL at once
                              #                     (8-way DDP each -> 2 ranks/GPU, most VRAM)
                              # "parallel"        = E1 & E2 concurrently on split GPUs (4+4)
                              # "sequential"      = E1 then E2, each alone on $GPUS_ALL (least VRAM)
GPUS_E1="0,1,2,3"             # GPUs for E1 in "parallel" (split) mode
GPUS_E2="4,5,6,7"             # GPUs for E2 in "parallel" (split) mode (must NOT overlap GPUS_E1)
GPUS_ALL="2,3,4,5,6,7"   # GPUs used by parallel_shared / sequential / single-experiment
PORT_START=29500              # first candidate port (next two free ports are auto-picked)
NCCL_TIMEOUT=1800             # NCCL collective timeout (s); raise if validation is slow
LOG_DIR="logs"
# checkpoint resume (auto-detected from outputs/…/<run_name>/last.ckpt if empty)
# set these to override:  RESUME_E1="outputs/.../e1_balanced/last.ckpt"
RESUME_E1="/home/tangbo/code/projects/UNITE/outputs/ffpp_c23/e1_balanced/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0.5/lr0.0001_effbs32_gc1_amp0/e1_balanced/last.ckpt"
RESUME_E2="/home/tangbo/code/projects/UNITE/outputs/ffpp_c23/e2_ceonly/siglip-so400m-patch14-384_d4_h12/img384_nf64_stride2/ce0.5_ad0/lr0.0001_effbs32_gc1_amp0/e2_ceonly/last.ckpt"
# ----------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

E1_CFG="configs/unite_ffpp_c23_e1_balanced.yaml"; E1_NAME="e1_balanced"
E2_CFG="configs/unite_ffpp_c23_e2_ceonly.yaml";   E2_NAME="e2_ceonly"

ngpu() { local IFS=','; set -- $1; echo $#; }   # count comma-separated GPU ids

# pick the first free TCP port starting from $1
pick_free_port() {
    local port="${1:-29500}"
    while ss -tlnH "sport = :$port" | grep -q . 2>/dev/null; do
        port=$((port + 1))
    done
    echo "$port"
}

# auto-detect last.ckpt for a given run_name (first match under outputs/)
find_last_ckpt() {
    local name="$1" ckpt
    ckpt="$(find outputs -maxdepth 9 -path "*/${name}/last.ckpt" -print -quit 2>/dev/null || true)"
    echo "$ckpt"
}

run_one() {  # args: gpus  master_port  config  run_name  [resume_ckpt]
    local gpus="$1" port="$2" cfg="$3" name="$4" resume="${5:-}"
    local nproc; nproc="$(ngpu "$gpus")"
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local log="$LOG_DIR/train_${name}_${ts}.log"

    # auto-detect checkpoint if not explicitly given
    if [ -z "$resume" ]; then
        resume="$(find_last_ckpt "$name")"
    fi

    local extra_args=()
    if [ -n "$resume" ] && [ -f "$resume" ]; then
        extra_args+=(--resume "$resume")
        echo ">>> [$name] config=$cfg  GPUs=[$gpus]  nproc=$nproc  port=$port  resume=$resume"
    else
        echo ">>> [$name] config=$cfg  GPUs=[$gpus]  nproc=$nproc  port=$port  (from scratch)"
    fi
    echo "    log -> $log   (live view:  tail -f $log)"
    TORCH_NCCL_BLOCKING_WAIT=1 \
    NCCL_TIMEOUT="$NCCL_TIMEOUT" \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    CUDA_VISIBLE_DEVICES="$gpus" \
    torchrun --nproc_per_node="$nproc" --master_port="$port" train.py \
        --config "$cfg" --run_name "$name" \
        "${extra_args[@]}" \
        > "$log" 2>&1
}

launch_both_concurrent() {  # args: gpus_for_e1  gpus_for_e2  — run E1 & E2 at the same time
    local port1; port1="$(pick_free_port "$PORT_START")"
    local port2; port2="$(pick_free_port $((port1 + 1)))"
    run_one "$1" "$port1" "$E1_CFG" "$E1_NAME" "$RESUME_E1" &
    local p1=$!
    sleep 2   # let E1's TCP store bind before E2 picks its own port
    run_one "$2" "$port2" "$E2_CFG" "$E2_NAME" "$RESUME_E2" &
    local p2=$!
    echo "launched: E1 pid=$p1, E2 pid=$p2 — waiting (watch VRAM:  watch -n5 nvidia-smi)..."
    set +e
    wait "$p1"; local r1=$?
    wait "$p2"; local r2=$?
    set -e
    echo "=== done: E1 exit=$r1, E2 exit=$r2 (0 = ok) ==="
    [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]
}

WHICH="${1:-both}"

case "$WHICH" in
  e1) run_one "$GPUS_ALL" "$(pick_free_port "$PORT_START")" "$E1_CFG" "$E1_NAME" "$RESUME_E1"; echo "=== E1 done ==="; exit 0 ;;
  e2) run_one "$GPUS_ALL" "$(pick_free_port "$PORT_START")" "$E2_CFG" "$E2_NAME" "$RESUME_E2"; echo "=== E2 done ==="; exit 0 ;;
  both) : ;;
  *) echo "unknown arg '$WHICH' (use: e1 | e2 | both)"; exit 2 ;;
esac

case "$MODE" in
  parallel_shared)
    echo "=== parallel_shared: BOTH on [$GPUS_ALL], 8-way DDP each (2 ranks/GPU) — watch VRAM ==="
    launch_both_concurrent "$GPUS_ALL" "$GPUS_ALL"
    ;;
  parallel)
    echo "=== parallel: E1 on [$GPUS_E1], E2 on [$GPUS_E2] ==="
    launch_both_concurrent "$GPUS_E1" "$GPUS_E2"
    ;;
  sequential)
    echo "=== sequential on [$GPUS_ALL]: E1 then E2 ==="
    run_one "$GPUS_ALL" "$PORT_E1" "$E1_CFG" "$E1_NAME"
    run_one "$GPUS_ALL" "$PORT_E2" "$E2_CFG" "$E2_NAME"
    echo "=== both done ==="
    ;;
  *) echo "unknown MODE '$MODE' (use: parallel_shared | parallel | sequential)"; exit 2 ;;
esac
