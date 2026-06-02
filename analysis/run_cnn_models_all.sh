#!/usr/bin/env bash
set -euo pipefail

# Run full feature extraction + SVD + SEP(ECG) analyses for CNN models (timm features_only).
# This mirrors `run_new_models_all.sh` but uses `extract_features_cnn.py` + stage-aware analysis.
#
# Restart-friendly: skips steps whose output files already exist.

export TMPDIR="${TMPDIR:-/tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-../1000_val}"
DEVICE="${DEVICE:-cuda}"

KEEP_LAST_K="${KEEP_LAST_K:-4}"   # drop very-high-res stems; keep 56/28/14/7 for ResNet-like models

count_npy() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
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
  local extra_extract_args="${4:-}"

  local feat_root="$out_root/features"
  local feat_dir="$feat_root/$timm_name"
  local svd_dir="$out_root/svd"
  local svd_stages_dir="$out_root/svd_stages"
  local ecg_dir="$out_root/ecg"

  echo "================================================================================"
  echo "MODEL: $display_name"
  echo "timm:  $timm_name"
  echo "out:   $out_root"
  echo "================================================================================"

  # 1) Feature extraction (stage object arrays)
  local n_feats
  n_feats="$(count_npy "$feat_dir")"
  if [[ "$n_feats" -lt 1000 ]]; then
    echo "[1/4] Extracting features ($n_feats/1000 existing)..."
    python extract_features_cnn.py \
      --model "$timm_name" \
      --dataset "$DATASET_DIR" \
      --output "$feat_root" \
      --device "$DEVICE" \
      --offline \
      --keep-last-k "$KEEP_LAST_K" \
      ${extra_extract_args}
  else
    echo "[1/4] Features already present ($n_feats/1000). Skipping."
  fi

  # 2) Last-stage SVD (Table-1 style percentiles)
  if [[ ! -f "$svd_dir/svd_results.npz" ]]; then
    echo "[2/4] Running last-stage SVD (percentiles)..."
    python analyze_svd_separate.py \
      --feature-dir "$feat_dir" \
      --output-dir "$svd_dir" \
      --use-gpu
  else
    echo "[2/4] Last-stage SVD results exist. Skipping."
  fi

  # 3) Stage-wise SVD histograms (per-stage distributions)
  if [[ ! -f "$svd_stages_dir/svd_stage_results.npz" ]]; then
    echo "[3/4] Running stage-wise SVD (per-stage histograms + percentiles)..."
    python analyze_svd_separate_stages.py \
      --feature-dir "$feat_dir" \
      --output-dir "$svd_stages_dir" \
      --use-gpu
  else
    echo "[3/4] Stage-wise SVD results exist. Skipping."
  fi

  # 4) SEP/ECG analysis (uses last stage)
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

# Supervised (ImageNet-1K)
run_model "ResNet-50 (IN1K)"       "resnet50.tv_in1k"               "Output/resnet50_tv_in1k"
run_model "ResNet-101 (IN1K)"      "resnet101.tv_in1k"              "Output/resnet101_tv_in1k"
run_model "ResNet-152 (IN1K)"      "resnet152.tv_in1k"              "Output/resnet152_tv_in1k"

run_model "ConvNeXt-Tiny (IN1K)"   "convnext_tiny.fb_in1k"          "Output/convnext_tiny_fb_in1k"
run_model "ConvNeXt-Base (IN1K)"   "convnext_base.fb_in1k"          "Output/convnext_base_fb_in1k"

# EfficientNet defaults to larger input sizes; override to 224 for manageable token counts.
run_model "EfficientNetV2-S (IN1K)" "tf_efficientnetv2_s.in1k"       "Output/tf_efficientnetv2_s_in1k" "--img-size 224"
run_model "EfficientNet-B4 (IN1K)"  "tf_efficientnet_b4.in1k"        "Output/tf_efficientnet_b4_in1k"  "--img-size 224"

# Self-supervised / masked autoencoding
run_model "ConvNeXtV2-Tiny (FCMAE)" "convnextv2_tiny.fcmae"          "Output/convnextv2_tiny_fcmae"
run_model "ConvNeXtV2-Base (FCMAE)" "convnextv2_base.fcmae"          "Output/convnextv2_base_fcmae"

# Optional extra SSL checkpoints
run_model "ResNet-50 (SWSL IG-1B -> IN1K)"   "resnet50.fb_swsl_ig1b_ft_in1k"    "Output/resnet50_fb_swsl_ig1b_ft_in1k"
run_model "ResNet-50 (SSL YFCC100M -> IN1K)" "resnet50.fb_ssl_yfcc100m_ft_in1k" "Output/resnet50_fb_ssl_yfcc100m_ft_in1k"

# Multimodal (CLIP, OpenAI)
run_model "ResNet-50 (CLIP, OpenAI)"   "resnet50_clip.openai"        "Output/resnet50_clip_openai"
run_model "ResNet-101 (CLIP, OpenAI)"  "resnet101_clip.openai"       "Output/resnet101_clip_openai"
run_model "ResNet-50x4 (CLIP, OpenAI)" "resnet50x4_clip.openai"      "Output/resnet50x4_clip_openai"

# ---------------------------------------------------------------------------
# Cross-model artifacts
# ---------------------------------------------------------------------------

echo "================================================================================"
echo "Cross-model comparison outputs"
echo "================================================================================"

echo "[1/1] Generate SVD rank table (Table 1 style; will include CNN rows when outputs exist)..."
python make_svd_rank_table.py \
  --save-md Output/comparison/svd/svd_rank_table.md \
  --save-tex Output/comparison/svd/svd_rank_table.tex

echo
echo "Done."
