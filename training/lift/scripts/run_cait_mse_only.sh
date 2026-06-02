#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="${DATA_PATH:-/data/datasets/ILSVRC}"
OUTPUT_DIR="${OUTPUT_DIR:-../Output/Lift-GAP-CaiT-MSE-Only}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EPOCHS="${EPOCHS:-300}"

export CUDA_VISIBLE_DEVICES="$GPUS"

python -m torch.distributed.run \
  --nproc_per_node "$NPROC" --master_port "$MASTER_PORT" main.py \
  --output_dir "$OUTPUT_DIR" \
  --data-path "$DATA_PATH" \
  --model deit_tiny_patch16_224 \
  --teacher-model cait_s24_224 \
  --distillation-type soft \
  --distillation-alpha 0.0 \
  --w-fft 0.2 \
  --use-mse-loss \
  --s-id 11 \
  --t-id 23 \
  --drop-path 0 \
  --batch-size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --use-modified-student \
  --expansion-start-layer 11 \
  --expansion-type step \
  --expansion-use-ln \
  --expansion-target-dim 384
