# From Per-Image Low-Rank to Encoding Mismatch

[![arXiv](https://img.shields.io/badge/arXiv-2511.15572-b31b1b.svg)](https://arxiv.org/abs/2511.15572)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICML%202026-8c1b13.svg)](https://openreview.net/forum?id=2Ud1nkQrVZ)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-ffd21e.svg)](https://huggingface.co/Huiyuancs/Encoding_Mismatch)
[![Poster](https://img.shields.io/badge/ICML%202026-Poster-4c78a8.svg)](./icml_poster.pdf)

Official implementation of the ICML 2026 paper:

> **From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers**  
> Huiyuan Tian, Bonan Xu, and Shijian Li

## Resources

- **Paper:** [arXiv](https://arxiv.org/abs/2511.15572) · [OpenReview](https://openreview.net/forum?id=2Ud1nkQrVZ)
- **Pretrained checkpoints and model card:** [Hugging Face](https://huggingface.co/Huiyuancs/Encoding_Mismatch)
- **Poster:** [ICML 2026 poster](./icml_poster.pdf)
- **Prepared analysis results:** [`Raw data`](./Raw%20data) · [Hugging Face dataset and Dataset Viewer](https://huggingface.co/datasets/Huiyuancs/Encoding_Mismatch_Analysis_Data)

## Overview

Feature maps from individual Vision Transformer inputs are often strongly low-rank, but this does not imply that a compact student can match a wider teacher through a small shared feature subspace.

We analyze the distinction between:

1. **Sample-wise low-rank structure**, measured with per-image SVD.
2. **Dataset-level structure**, measured with PCA over many images.
3. **Spectral Energy Patterns (SEP)**, which describe channel-wise feature-energy allocation.

Although each image may occupy a compact subspace, the low-rank subspaces rotate across samples. Their dataset-level union can therefore remain broad. Compact students and wider teachers also allocate feature energy differently across channels, producing an **encoding mismatch** that limits heterogeneous feature distillation.

We introduce two simple remedies:

- **Lift:** retain a learned linear endpoint projector that lifts the student's final feature dimension from 192 to 384.
- **WideLast:** keep the first eleven DeiT-Tiny blocks at 192 dimensions and widen only the final block to 384 dimensions.

## Main Results

ImageNet-1K validation results reported in the paper:

| Student                      | Distillation objective  | Top-1 accuracy |
| ---------------------------- | ----------------------- | -------------: |
| Distilled DeiT-Tiny baseline | Original baseline       |         74.86% |
| Lift                         | SoftKD                  |         77.23% |
| Lift                         | SoftKD + MSE            |         77.50% |
| **Lift**                     | **SoftKD + SpectralKD** |     **77.53%** |
| WideLast                     | SoftKD                  |         77.88% |
| WideLast                     | SoftKD + SpectralKD     |         78.16% |
| **WideLast**                 | **SoftKD + MSE**        |     **78.23%** |

Inference-cost comparison:

| Model                        | Parameters | Change vs. baseline |  FLOPs | Change vs. baseline |
| ---------------------------- | ---------: | ------------------: | -----: | ------------------: |
| Distilled DeiT-Tiny baseline |  5,717,416 |                   — | 2.507G |                   — |
| Lift                         |  5,983,528 |              +4.65% | 2.536G |              +1.17% |
| WideLast                     |  7,239,016 |             +26.61% | 3.089G |             +23.20% |

## Pretrained Checkpoints

The released checkpoints are hosted on Hugging Face:

| Model    | Architecture                                                 | Training objective                    | Checkpoint                                                   |
| -------- | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| Lift     | DeiT-Tiny-derived student with a retained 192→384 endpoint projector | CaiT-S24 teacher, SoftKD + SpectralKD | [Download / view](https://huggingface.co/Huiyuancs/Encoding_Mismatch/blob/main/models/Lift/pytorch_model.pth) |
| WideLast | Blocks 1–11 at 192 dimensions; final block at 384 dimensions | CaiT-S24 teacher, SoftKD + MSE        | [Download / view](https://huggingface.co/Huiyuancs/Encoding_Mismatch/blob/main/models/WideLast/pytorch_model.pth) |

The complete model card, metadata, intended uses, and limitations are available in the [Hugging Face model repository](https://huggingface.co/Huiyuancs/Encoding_Mismatch).

## Repository Structure

```text
.
├── analysis/          # SVD, dataset PCA, SEP, robustness, and plotting scripts
├── Raw data/          # Prepared .npz/.csv results for tables and figures
├── training/
│   ├── lift/          # Lift architecture, training, and evaluation
│   └── widelast/      # WideLast architecture, training, and evaluation
├── icml_poster.pdf    # ICML 2026 poster
├── requirements.txt
└── README.md
```

## Environment

```bash
conda create -n encoding-mismatch python=3.10 -y
conda activate encoding-mismatch
pip install -r requirements.txt
```

For GPU-accelerated SVD/SEP analysis, install the CuPy package matching your CUDA version, for example:

```bash
pip install cupy-cuda12x
```

## Data Layout

Training and evaluation scripts expect ImageNet-1K in the standard folder layout:

```text
/path/to/ILSVRC/
├── train/
└── val/
```

ImageNet is not redistributed by this repository.

The analysis scripts use a 1,000-image subset of the ImageNet validation set. Create it with:

```bash
cd analysis

python main.py \
  --step 1 \
  --source /path/to/ILSVRC/val \
  --dataset /path/to/1000_val
```

## Representation Analysis

Example for CaiT-S24:

```bash
cd analysis

python main.py \
  --step 2 \
  --model cait_s24_224 \
  --dataset /path/to/1000_val \
  --output Output/cait/features \
  --offline
```

### Per-image SVD

```bash
python analyze_svd_separate.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/svd \
  --use-gpu
```

### Dataset-level PCA

```bash
python analyze_dataset_pca.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/dataset_pca \
  --backend torch \
  --device cuda
```

### Spectral Energy Patterns

```bash
python analyze_ecg_spectral.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/ecg \
  --model-name CaiT-S24 \
  --use-gpu
```

### Compare sample-wise and dataset-level ranks

```bash
python compare_svd_vs_dataset_pca_hist.py \
  --model-root Output/cait \
  --energy 0.95 \
  --layout overlay
```

### SEP permutation robustness

```bash
python analyze_sep_permutation_robustness.py \
  --feature-dir Output/cait/features/cait_s24_224 \
  --output-dir Output/cait/sep_permutation_robustness \
  --model-name CaiT-S24 \
  --num-permutations 100 \
  --use-gpu
```

To regenerate paper tables and figures without re-extracting features, use the prepared `.npz` and `.csv` files in [`Raw data`](./Raw%20data).

## Training

All training launchers use environment variables for paths and hardware:

```bash
export DATA_PATH=/path/to/ILSVRC
export GPUS=0,1,2,3
export NPROC=4
```

### Lift: CaiT-S24, SoftKD + SpectralKD

```bash
cd training/lift
bash scripts/run_cait_softkd_spectralkd.sh
```

### Lift: CaiT-S24, MSE-only feature distillation

```bash
cd training/lift
bash scripts/run_cait_mse_only.sh
```

### WideLast: CaiT-S24, SoftKD + MSE

```bash
cd training/widelast
bash scripts/run_cait_softkd_mse.sh
```

### WideLast: CaiT-S24, SpectralKD-only

```bash
cd training/widelast
bash scripts/run_cait_spectralkd_only.sh
```

## Evaluate the Released Checkpoints

Install `huggingface_hub` and download the model repository:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Huiyuancs/Encoding_Mismatch",
    local_dir="./Encoding_Mismatch",
)
```

### Lift

```bash
cd training/lift

python main.py \
  --eval \
  --resume /absolute/path/to/Encoding_Mismatch/models/Lift/pytorch_model.pth \
  --data-path /path/to/ILSVRC \
  --model deit_tiny_patch16_224 \
  --batch-size 256 \
  --num_workers 8 \
  --use-modified-student \
  --expansion-start-layer 11 \
  --expansion-type step \
  --expansion-use-ln \
  --expansion-target-dim 384
```

### WideLast

```bash
cd training/widelast

python main.py \
  --eval \
  --resume /absolute/path/to/Encoding_Mismatch/models/WideLast/pytorch_model.pth \
  --data-path /path/to/ILSVRC \
  --model deit_tiny_patch16_224 \
  --batch-size 256 \
  --num_workers 8 \
  --custom-arch \
  --arch-schedule heads_step
```

## Poster

The [ICML 2026 poster](./icml_poster.pdf) provides a compact visual summary of the motivation, representation analyses, proposed architectures, and main results.

## Citation

```bibtex
@inproceedings{tian2026encodingmismatch,
  title     = {From Per-Image Low-Rank to Encoding Mismatch:
               Rethinking Feature Distillation in Vision Transformers},
  author    = {Tian, Huiyuan and Xu, Bonan and Li, Shijian},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

This repository is licensed under the Apache License 2.0; see
[`LICENSE`](./LICENSE). Third-party dependencies and datasets remain
subject to their respective licenses and terms.
