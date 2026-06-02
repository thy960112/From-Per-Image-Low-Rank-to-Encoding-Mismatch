#!/usr/bin/env bash
set -euo pipefail

# Run full feature extraction + SVD + SEP(ECG) analyses for new pretrained ViTs.
# This script is restart-friendly: it skips steps whose output files already exist.

export TMPDIR="${TMPDIR:-/tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-../1000_val}"
DEVICE="${DEVICE:-cuda}"

mkdir -p Output/comparison/svd

count_npy() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  # Fast count without printing file names.
  python - "$dir" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
print(len(list(p.glob('*.npy'))))
PY
}

run_model() {
  local display_name="$1"
  local timm_name="$2"
  local out_root="$3"
  local extra_main_args="${4:-}"

  local feat_root="$out_root/features"
  local feat_dir="$feat_root/$timm_name"
  local svd_dir="$out_root/svd"
  local svd_layers_dir="$out_root/svd_layers"
  local ecg_dir="$out_root/ecg"

  echo "================================================================================"
  echo "MODEL: $display_name"
  echo "timm:  $timm_name"
  echo "out:   $out_root"
  echo "================================================================================"

  # 1) Feature extraction
  local n_feats
  n_feats="$(count_npy "$feat_dir")"
  if [[ "$n_feats" -lt 1000 ]]; then
    echo "[1/4] Extracting features ($n_feats/1000 existing)..."
    # Note: main.py sets HF_HUB_OFFLINE before importing timm.
    python main.py --step 2 \
      --model "$timm_name" \
      --dataset "$DATASET_DIR" \
      --output "$feat_root" \
      --device "$DEVICE" \
      --offline \
      ${extra_main_args}
  else
    echo "[1/4] Features already present ($n_feats/1000). Skipping."
  fi

  # 2) Last-layer SVD (percentiles used by Table 1)
  if [[ ! -f "$svd_dir/svd_results.npz" ]]; then
    echo "[2/4] Running last-layer SVD (percentiles)..."
    python analyze_svd_separate.py \
      --feature-dir "$feat_dir" \
      --output-dir "$svd_dir" \
      --use-gpu
  else
    echo "[2/4] Last-layer SVD results exist. Skipping."
  fi

  # 3) Layer-wise SVD (99% energy curve across depth)
  if [[ ! -f "$svd_layers_dir/layer_wise_svd_results.npz" ]]; then
    echo "[3/4] Running layer-wise SVD (99% energy across layers)..."
    python analyze_layers_svd.py \
      --feature-dir "$feat_dir" \
      --output-dir "$svd_layers_dir" \
      --use-gpu
  else
    echo "[3/4] Layer-wise SVD results exist. Skipping."
  fi

  # 4) SEP/ECG analysis (token-level spectral energy curve)
  if [[ ! -f "$ecg_dir/ecg_results.npz" ]]; then
    echo "[4/4] Running SEP/ECG analysis..."
    python analyze_ecg_spectral.py \
      --feature-dir "$feat_dir" \
      --output-dir "$ecg_dir" \
      --model-name "$display_name" \
      --use-gpu
  else
    echo "[4/4] SEP/ECG results exist. Skipping."
  fi

  echo
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# CLIP (OpenAI)
run_model "ViT-Base (CLIP, OpenAI)"  "vit_base_patch16_clip_224.openai"   "Output/vit_base_patch16_clip_openai"
run_model "ViT-Large (CLIP, OpenAI)" "vit_large_patch14_clip_224.openai"  "Output/vit_large_patch14_clip_openai"

# DINO v1
run_model "ViT-Base (DINO)"          "vit_base_patch16_224.dino"          "Output/vit_base_patch16_224_dino"
run_model "ViT-Small (DINO)"         "vit_small_patch16_224.dino"         "Output/vit_small_patch16_224_dino"

# DINOv2 (force 224 to avoid 518-token explosion)
run_model "ViT-Base (DINOv2)"        "vit_base_patch14_dinov2.lvd142m"    "Output/vit_base_patch14_dinov2"   "--img-size 224"
run_model "ViT-Large (DINOv2)"       "vit_large_patch14_dinov2.lvd142m"   "Output/vit_large_patch14_dinov2"  "--img-size 224"

# MAE
run_model "ViT-Base (MAE)"           "vit_base_patch16_224.mae"           "Output/vit_base_patch16_224_mae"
run_model "ViT-Large (MAE)"          "vit_large_patch16_224.mae"          "Output/vit_large_patch16_224_mae"

# ---------------------------------------------------------------------------
# Cross-model artifacts
# ---------------------------------------------------------------------------

echo "================================================================================"
echo "Cross-model comparison outputs"
echo "================================================================================"

echo "[1/3] Regenerate ECG comparison table/plot (skips per-model analyses)..."
python analyze_ecg_all_models.py --use-gpu --skip-analysis

echo "[2/3] Regenerate ECG comparison plot (paper formatting) + per-model ECG plots..."
python replot_ecg_comparison.py
python replot_ecg_individual.py

echo "[3/3] Generate SVD rank table (Table 1 style, extended)..."
python make_svd_rank_table.py \
  --save-md Output/comparison/svd/svd_rank_table.md \
  --save-tex Output/comparison/svd/svd_rank_table.tex

echo
echo "Done."
