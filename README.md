# From Per-Image Low-Rank to Encoding Mismatch

Code release for the ICML 2026 paper "From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers".

## What Is Included

- `analysis/`: SVD, dataset-level PCA, SEP, SEP permutation robustness, and plotting/table scripts.
- `training/lift/`: Lift implementation using a retained endpoint projector.
- `training/widelast/`: WideLast implementation. 

## Environment

```bash
conda create -n encoding-mismatch python=3.10 -y
conda activate encoding-mismatch
pip install -r requirements.txt
```

For CuPy acceleration in SVD/SEP analysis, install the CuPy wheel matching your CUDA version, for example `cupy-cuda12x`.

## Data Layout

Training scripts expect ImageNet-1K in the standard folder layout:

```text
/path/to/ILSVRC/
  train/
  val/
```

Analysis scripts use a 1,000-image ImageNet validation subset. You can create it from ImageNet val with:

```bash
cd analysis
python main.py \
  --step 1 \
  --source /path/to/ILSVRC/val \
  --dataset /path/to/1000_val
```

## Analysis

Example for CaiT-S24:

```bash
cd analysis

python main.py \
  --step 2 \
  --model cait_s24_224 \
  --dataset /path/to/1000_val \
  --output Output/cait/features \
  --offline

python analyze_svd_separate.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/svd \
  --use-gpu

python analyze_dataset_pca.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/dataset_pca \
  --backend torch \
  --device cuda

python analyze_ecg_spectral.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/ecg \
  --model-name CaiT-S24 \
  --use-gpu

python compare_svd_vs_dataset_pca_hist.py \
  --model-root Output/cait \
  --energy 0.95 \
  --layout overlay
```

SEP permutation robustness:

```bash
python analyze_sep_permutation_robustness.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/sep_permutation_robustness \
  --model-name CaiT-S24 \
  --num-permutations 100 \
  --use-gpu
```

If you only want to regenerate tables/figures from prepared numeric results, use the `.npz`/`.csv` files in `Raw data`.

## Training

All training scripts use environment variables for paths and hardware. Override at launch time:

```bash
export DATA_PATH=/path/to/ILSVRC
export GPUS=0,1,2,3
export NPROC=4
```

Lift, CaiT-S24 teacher, SoftKD + SpectralKD:

```bash
cd training/lift
bash scripts/run_cait_softkd_spectralkd.sh
```

Lift, CaiT-S24 teacher, MSE-only feature KD:

```bash
cd training/lift
bash scripts/run_cait_mse_only.sh
```

WideLast, CaiT-S24 teacher, SoftKD + MSE:

```bash
cd training/widelast
bash scripts/run_cait_softkd_mse.sh
```

WideLast, CaiT-S24 teacher, SpectralKD-only:

```bash
cd training/widelast
bash scripts/run_cait_spectralkd_only.sh
```
