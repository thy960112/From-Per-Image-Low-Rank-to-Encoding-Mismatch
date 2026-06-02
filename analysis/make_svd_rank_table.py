#!/usr/bin/env python3
"""
Generate a Table-1 style summary from per-model SVD outputs.

This reads `svd_results.npz` produced by:
  - analyze_svd_separate.py (ViT/CaiT/DeiT)
  - analyze_svd_separate_swin.py (Swin)

Optionally (when present), it also reads dataset-level PCA results from:
  - <output_root>/dataset_pca/dataset_pca_results.npz

and prints both Markdown and LaTeX tables to stdout; optionally saves them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class ModelEntry:
    """One row in the table (a model + where to find its outputs)."""
    train_method: str  # SL / SSL / MM
    model: str
    output_root: Path  # e.g. Output/cait
    feature_dir: Path  # e.g. Output/cait/features/cait_s24_224


DEFAULT_MODELS: List[ModelEntry] = [
    # Paper baselines
    ModelEntry(
        train_method="SL",
        model="ViT-Tiny",
        output_root=Path("Output/vit_tiny_patch16_224_21k"),
        feature_dir=Path("Output/vit_tiny_patch16_224_21k/features/vit_tiny_patch16_224.augreg_in21k_ft_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="CaiT-S24",
        output_root=Path("Output/cait"),
        feature_dir=Path("Output/cait/features/cait_s24_224"),
    ),
    ModelEntry(
        train_method="SL",
        model="DeiT-Small",
        output_root=Path("Output/deit_small"),
        feature_dir=Path("Output/deit_small/features/deit_small_patch16_224"),
    ),
    ModelEntry(
        train_method="SL",
        model="Swin-Small",
        output_root=Path("Output/swin_small"),
        feature_dir=Path("Output/swin_small/features/swin_small_patch4_window7_224.ms_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="ViT-Large",
        output_root=Path("Output/vit_large_21k_in1k"),
        feature_dir=Path("Output/vit_large_21k_in1k/features/vit_large_patch16_224.augreg_in21k_ft_in1k"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Huge (MAE)",
        output_root=Path("Output/vit_huge_patch14_224_mae"),
        feature_dir=Path("Output/vit_huge_patch14_224_mae/features/vit_huge_patch14_224.mae"),
    ),
    # New pretraining variants requested by the user (feature dirs follow the same convention)
    ModelEntry(
        train_method="MM",
        model="ViT-Base (CLIP, OpenAI)",
        output_root=Path("Output/vit_base_patch16_clip_openai"),
        feature_dir=Path("Output/vit_base_patch16_clip_openai/features/vit_base_patch16_clip_224.openai"),
    ),
    ModelEntry(
        train_method="MM",
        model="ViT-Large (CLIP, OpenAI)",
        output_root=Path("Output/vit_large_patch14_clip_openai"),
        feature_dir=Path("Output/vit_large_patch14_clip_openai/features/vit_large_patch14_clip_224.openai"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Base (DINOv2)",
        output_root=Path("Output/vit_base_patch14_dinov2"),
        feature_dir=Path("Output/vit_base_patch14_dinov2/features/vit_base_patch14_dinov2.lvd142m"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Large (DINOv2)",
        output_root=Path("Output/vit_large_patch14_dinov2"),
        feature_dir=Path("Output/vit_large_patch14_dinov2/features/vit_large_patch14_dinov2.lvd142m"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Base (DINO)",
        output_root=Path("Output/vit_base_patch16_224_dino"),
        feature_dir=Path("Output/vit_base_patch16_224_dino/features/vit_base_patch16_224.dino"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Small (DINO)",
        output_root=Path("Output/vit_small_patch16_224_dino"),
        feature_dir=Path("Output/vit_small_patch16_224_dino/features/vit_small_patch16_224.dino"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Base (MAE)",
        output_root=Path("Output/vit_base_patch16_224_mae"),
        feature_dir=Path("Output/vit_base_patch16_224_mae/features/vit_base_patch16_224.mae"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ViT-Large (MAE)",
        output_root=Path("Output/vit_large_patch16_224_mae"),
        feature_dir=Path("Output/vit_large_patch16_224_mae/features/vit_large_patch16_224.mae"),
    ),

    # ---------------------------------------------------------------------
    # CNN models (stage features saved as object arrays; last stage used here)
    # ---------------------------------------------------------------------

    # Supervised (ImageNet-1K)
    ModelEntry(
        train_method="SL",
        model="ResNet-50",
        output_root=Path("Output/resnet50_tv_in1k"),
        feature_dir=Path("Output/resnet50_tv_in1k/features/resnet50.tv_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="ResNet-101",
        output_root=Path("Output/resnet101_tv_in1k"),
        feature_dir=Path("Output/resnet101_tv_in1k/features/resnet101.tv_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="ResNet-152",
        output_root=Path("Output/resnet152_tv_in1k"),
        feature_dir=Path("Output/resnet152_tv_in1k/features/resnet152.tv_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="ConvNeXt-Tiny",
        output_root=Path("Output/convnext_tiny_fb_in1k"),
        feature_dir=Path("Output/convnext_tiny_fb_in1k/features/convnext_tiny.fb_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="ConvNeXt-Base",
        output_root=Path("Output/convnext_base_fb_in1k"),
        feature_dir=Path("Output/convnext_base_fb_in1k/features/convnext_base.fb_in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="EfficientNetV2-S",
        output_root=Path("Output/tf_efficientnetv2_s_in1k"),
        feature_dir=Path("Output/tf_efficientnetv2_s_in1k/features/tf_efficientnetv2_s.in1k"),
    ),
    ModelEntry(
        train_method="SL",
        model="EfficientNet-B4",
        output_root=Path("Output/tf_efficientnet_b4_in1k"),
        feature_dir=Path("Output/tf_efficientnet_b4_in1k/features/tf_efficientnet_b4.in1k"),
    ),

    # Self-supervised
    ModelEntry(
        train_method="SSL",
        model="ConvNeXtV2-Tiny (FCMAE)",
        output_root=Path("Output/convnextv2_tiny_fcmae"),
        feature_dir=Path("Output/convnextv2_tiny_fcmae/features/convnextv2_tiny.fcmae"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ConvNeXtV2-Base (FCMAE)",
        output_root=Path("Output/convnextv2_base_fcmae"),
        feature_dir=Path("Output/convnextv2_base_fcmae/features/convnextv2_base.fcmae"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ResNet-50 (SWSL IG-1B -> IN1K)",
        output_root=Path("Output/resnet50_fb_swsl_ig1b_ft_in1k"),
        feature_dir=Path("Output/resnet50_fb_swsl_ig1b_ft_in1k/features/resnet50.fb_swsl_ig1b_ft_in1k"),
    ),
    ModelEntry(
        train_method="SSL",
        model="ResNet-50 (SSL YFCC100M -> IN1K)",
        output_root=Path("Output/resnet50_fb_ssl_yfcc100m_ft_in1k"),
        feature_dir=Path("Output/resnet50_fb_ssl_yfcc100m_ft_in1k/features/resnet50.fb_ssl_yfcc100m_ft_in1k"),
    ),

    # Multimodal (CLIP)
    ModelEntry(
        train_method="MM",
        model="ResNet-50 (CLIP, OpenAI)",
        output_root=Path("Output/resnet50_clip_openai"),
        feature_dir=Path("Output/resnet50_clip_openai/features/resnet50_clip.openai"),
    ),
    ModelEntry(
        train_method="MM",
        model="ResNet-101 (CLIP, OpenAI)",
        output_root=Path("Output/resnet101_clip_openai"),
        feature_dir=Path("Output/resnet101_clip_openai/features/resnet101_clip.openai"),
    ),
    ModelEntry(
        train_method="MM",
        model="ResNet-50x4 (CLIP, OpenAI)",
        output_root=Path("Output/resnet50x4_clip_openai"),
        feature_dir=Path("Output/resnet50x4_clip_openai/features/resnet50x4_clip.openai"),
    ),
]


def _infer_channel_dim(sample_features: np.ndarray) -> int:
    """Infer channel dimension C from one feature file (ViT/CaiT: [L,N,C], Swin: object stages)."""
    if sample_features.dtype == object:
        return int(sample_features[-1].shape[1])
    return int(sample_features.shape[2])


def load_row(entry: ModelEntry):
    svd_file = entry.output_root / "svd" / "svd_results.npz"
    if not svd_file.exists():
        return None

    # Load percentile results.
    data = np.load(svd_file)
    p80 = float(data["percentile_99_dims_80"])
    p90 = float(data["percentile_99_dims_90"])
    p95 = float(data["percentile_99_dims_95"])
    p99 = float(data["percentile_99_dims_99"])

    # Dataset-level PCA (shared projector) results: per-image required d under the shared basis.
    # We report the 99th percentile over images for each energy threshold.
    ds_p80 = ds_p90 = ds_p95 = ds_p99 = "-"
    ds_file = entry.output_root / "dataset_pca" / "dataset_pca_results.npz"
    if ds_file.exists():
        ds = np.load(ds_file)
        curves = ds["curves"]  # [num_images, D] cumulative captured energy for shared PCA basis

        def p99_required_d(threshold: float) -> int:
            hit = curves >= threshold
            first = hit.argmax(axis=1)  # 0-indexed; 0 if never hit
            reached = hit.any(axis=1)
            dims = np.where(reached, first + 1, curves.shape[1])
            return int(round(float(np.percentile(dims, 99))))

        ds_p80 = p99_required_d(0.80)
        ds_p90 = p99_required_d(0.90)
        ds_p95 = p99_required_d(0.95)
        ds_p99 = p99_required_d(0.99)

    # Infer channel dim from a single feature file for consistency.
    feat_files = sorted(entry.feature_dir.glob("*.npy"))
    if not feat_files:
        c = -1
    else:
        sample = np.load(feat_files[0], allow_pickle=True)
        c = _infer_channel_dim(sample)

    return {
        "Train Method": entry.train_method,
        "Model": entry.model,
        "Dims": c if c >= 0 else "-",
        # Per-image SVD (input-dependent) summary
        "80% Energy": int(round(p80)),
        "90% Energy": int(round(p90)),
        "95% Energy": int(round(p95)),
        "99% Energy": int(round(p99)),
        # Dataset-level PCA (shared basis) summary
        "80% Energy (Dataset)": ds_p80,
        "90% Energy (Dataset)": ds_p90,
        "95% Energy (Dataset)": ds_p95,
        "99% Energy (Dataset)": ds_p99,
    }


def to_markdown(rows):
    cols = [
        "Train Method",
        "Model",
        "Dims",
        "SVD 80%",
        "SVD 90%",
        "SVD 95%",
        "SVD 99%",
        "Dataset 80%",
        "Dataset 90%",
        "Dataset 95%",
        "Dataset 99%",
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join([":--" if c in {"Train Method", "Model"} else "--:" for c in cols]) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in [
                    "Train Method",
                    "Model",
                    "Dims",
                    "80% Energy",
                    "90% Energy",
                    "95% Energy",
                    "99% Energy",
                    "80% Energy (Dataset)",
                    "90% Energy (Dataset)",
                    "95% Energy (Dataset)",
                    "99% Energy (Dataset)",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def to_latex(rows):
    # NOTE: In LaTeX, '%' starts a comment, so we must escape it as '\\%'.
    # Two header rows with grouped columns (booktabs + cmidrule recommended).
    lines = []
    lines.append(r"\begin{tabular}{l l r r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"Train Method & Model & Dims & "
        r"\multicolumn{4}{c}{Per-image SVD (99th pct)} & "
        r"\multicolumn{4}{c}{Dataset PCA (shared, 99th pct)} \\"
    )
    lines.append(r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}")
    lines.append(
        r" & & & 80\% & 90\% & 95\% & 99\% & 80\% & 90\% & 95\% & 99\% \\"
    )
    lines.append(r"\midrule")

    # Group rows by training method for readability.
    last_tm = None
    for r in rows:
        tm = r["Train Method"]
        if last_tm is not None and tm != last_tm:
            lines.append(r"\midrule")
        last_tm = tm
        lines.append(
            " & ".join([
                str(r["Train Method"]),
                str(r["Model"]),
                str(r["Dims"]),
                str(r["80% Energy"]),
                str(r["90% Energy"]),
                str(r["95% Energy"]),
                str(r["99% Energy"]),
                str(r["80% Energy (Dataset)"]),
                str(r["90% Energy (Dataset)"]),
                str(r["95% Energy (Dataset)"]),
                str(r["99% Energy (Dataset)"]),
            ]) + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Create Table-1 style SVD rank summary from existing outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-md",
        type=Path,
        default=None,
        help="Optional path to save Markdown table",
    )
    parser.add_argument(
        "--save-tex",
        type=Path,
        default=None,
        help="Optional path to save LaTeX table",
    )
    args = parser.parse_args()

    rows = []
    skipped = []
    for idx, entry in enumerate(DEFAULT_MODELS):
        row = load_row(entry)
        if row is None:
            skipped.append(entry.model)
            continue
        row["_order"] = idx  # stable within-group ordering
        rows.append(row)

    # Sort rows by training method (SL -> SSL -> MM), keep a stable within-group order.
    train_order = {"SL": 0, "SSL": 1, "MM": 2}
    rows.sort(key=lambda r: (train_order.get(r["Train Method"], 99), r["_order"]))

    md = to_markdown(rows)
    tex = to_latex(rows)

    print("\nMarkdown table:\n")
    print(md)
    print("\nLaTeX table:\n")
    print(tex)

    if skipped:
        print("\nSkipped (missing svd_results.npz):")
        for m in skipped:
            print(f"  - {m}")

    if args.save_md is not None:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(md)
        print("\nSaved Markdown:", str(args.save_md))

    if args.save_tex is not None:
        args.save_tex.parent.mkdir(parents=True, exist_ok=True)
        args.save_tex.write_text(tex)
        print("Saved LaTeX:", str(args.save_tex))


if __name__ == "__main__":
    main()
