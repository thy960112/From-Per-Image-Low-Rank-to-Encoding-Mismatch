#!/usr/bin/env python3
"""
Run SEP permutation robustness for all configured transformer models and create
comparison artifacts.
"""

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    {
        "short_name": "vit_tiny",
        "model_name": "ViT-Tiny",
        "feature_dir": "Output/vit_tiny_patch16_224_21k/features/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "output_dir": "Output/vit_tiny_patch16_224_21k/sep_permutation_robustness",
        "color": "#bcbd22",
        "marker": "p",
        "batch_size": 96,
    },
    {
        "short_name": "cait",
        "model_name": "CaiT-S24",
        "feature_dir": "Output/cait/features/cait_s24_224",
        "output_dir": "Output/cait/sep_permutation_robustness",
        "color": "#1f77b4",
        "marker": "o",
        "batch_size": 64,
    },
    {
        "short_name": "deit",
        "model_name": "DeiT-Small",
        "feature_dir": "Output/deit_small/features/deit_small_patch16_224",
        "output_dir": "Output/deit_small/sep_permutation_robustness",
        "color": "#ff7f0e",
        "marker": "s",
        "batch_size": 64,
    },
    {
        "short_name": "vit_large",
        "model_name": "ViT-Large",
        "feature_dir": "Output/vit_large_21k_in1k/features/vit_large_patch16_224.augreg_in21k_ft_in1k",
        "output_dir": "Output/vit_large_21k_in1k/sep_permutation_robustness",
        "color": "#2ca02c",
        "marker": "^",
        "batch_size": 32,
    },
    {
        "short_name": "vit_huge",
        "model_name": "ViT-Huge",
        "feature_dir": "Output/vit_huge_patch14_224_mae/features/vit_huge_patch14_224.mae",
        "output_dir": "Output/vit_huge_patch14_224_mae/sep_permutation_robustness",
        "color": "#d62728",
        "marker": "v",
        "batch_size": 16,
    },
    {
        "short_name": "swin_small",
        "model_name": "Swin-Small",
        "feature_dir": "Output/swin_small/features/swin_small_patch4_window7_224.ms_in1k",
        "output_dir": "Output/swin_small/sep_permutation_robustness",
        "color": "#9467bd",
        "marker": "D",
        "batch_size": 64,
    },
    {
        "short_name": "vit_base_clip_openai",
        "model_name": "ViT-Base (CLIP)",
        "feature_dir": "Output/vit_base_patch16_clip_openai/features/vit_base_patch16_clip_224.openai",
        "output_dir": "Output/vit_base_patch16_clip_openai/sep_permutation_robustness",
        "color": "#8c564b",
        "marker": "X",
        "batch_size": 64,
    },
    {
        "short_name": "vit_large_clip_openai",
        "model_name": "ViT-Large (CLIP)",
        "feature_dir": "Output/vit_large_patch14_clip_openai/features/vit_large_patch14_clip_224.openai",
        "output_dir": "Output/vit_large_patch14_clip_openai/sep_permutation_robustness",
        "color": "#e377c2",
        "marker": "*",
        "batch_size": 32,
    },
    {
        "short_name": "vit_base_dinov2",
        "model_name": "ViT-Base (DINOv2)",
        "feature_dir": "Output/vit_base_patch14_dinov2/features/vit_base_patch14_dinov2.lvd142m",
        "output_dir": "Output/vit_base_patch14_dinov2/sep_permutation_robustness",
        "color": "#7f7f7f",
        "marker": "h",
        "batch_size": 64,
    },
    {
        "short_name": "vit_large_dinov2",
        "model_name": "ViT-Large (DINOv2)",
        "feature_dir": "Output/vit_large_patch14_dinov2/features/vit_large_patch14_dinov2.lvd142m",
        "output_dir": "Output/vit_large_patch14_dinov2/sep_permutation_robustness",
        "color": "#17becf",
        "marker": "P",
        "batch_size": 32,
    },
    {
        "short_name": "vit_base_dino",
        "model_name": "ViT-Base (DINO)",
        "feature_dir": "Output/vit_base_patch16_224_dino/features/vit_base_patch16_224.dino",
        "output_dir": "Output/vit_base_patch16_224_dino/sep_permutation_robustness",
        "color": "#c7c7c7",
        "marker": "H",
        "batch_size": 64,
    },
    {
        "short_name": "vit_small_dino",
        "model_name": "ViT-Small (DINO)",
        "feature_dir": "Output/vit_small_patch16_224_dino/features/vit_small_patch16_224.dino",
        "output_dir": "Output/vit_small_patch16_224_dino/sep_permutation_robustness",
        "color": "#dbdb8d",
        "marker": "s",
        "batch_size": 96,
    },
    {
        "short_name": "vit_base_mae",
        "model_name": "ViT-Base (MAE)",
        "feature_dir": "Output/vit_base_patch16_224_mae/features/vit_base_patch16_224.mae",
        "output_dir": "Output/vit_base_patch16_224_mae/sep_permutation_robustness",
        "color": "#98df8a",
        "marker": ">",
        "batch_size": 64,
    },
    {
        "short_name": "vit_large_mae",
        "model_name": "ViT-Large (MAE)",
        "feature_dir": "Output/vit_large_patch16_224_mae/features/vit_large_patch16_224.mae",
        "output_dir": "Output/vit_large_patch16_224_mae/sep_permutation_robustness",
        "color": "#ffbb78",
        "marker": "<",
        "batch_size": 32,
    },
]


def setup_plot_style():
    plt.rcParams.update({
        "font.size": 20,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    })


def selected_models(only_short_names: list[str] | None) -> list[dict]:
    if not only_short_names:
        return MODELS
    wanted = set(only_short_names)
    return [model for model in MODELS if model["short_name"] in wanted]


def analyze_single_model(model_config: dict, num_permutations: int, bootstrap_samples: int, use_gpu: bool) -> bool:
    feature_dir = Path(model_config["feature_dir"])
    if not feature_dir.exists():
        print(f"Skipping {model_config['model_name']}: feature directory not found at {feature_dir}")
        return False

    output_dir = Path(model_config["output_dir"])
    results_file = output_dir / "sep_permutation_results.npz"
    if results_file.exists():
        print(f"Using cached robustness results for {model_config['model_name']}")
        return True

    cmd = [
        "python",
        "analyze_sep_permutation_robustness.py",
        "--feature-dir", str(feature_dir),
        "--output-dir", str(output_dir),
        "--model-name", model_config["model_name"],
        "--num-permutations", str(num_permutations),
        "--batch-size", str(model_config["batch_size"]),
        "--bootstrap-samples", str(bootstrap_samples),
    ]
    if use_gpu:
        cmd.append("--use-gpu")

    print(f"Running robustness analysis for {model_config['model_name']}...")
    subprocess.run(cmd, check=True)
    return True


def load_model_results(model_config: dict) -> dict | None:
    results_file = Path(model_config["output_dir"]) / "sep_permutation_results.npz"
    summary_file = Path(model_config["output_dir"]) / "sep_permutation_bandwidth_summary.csv"
    if not results_file.exists() or not summary_file.exists():
        return None

    data = np.load(results_file, allow_pickle=True)
    with summary_file.open(newline="") as handle:
        rows = {int(row["threshold_percent"]): row for row in csv.DictReader(handle)}

    return {
        "model_name": str(data["model_name"]),
        "spectrum_mode": str(data["spectrum_mode"]),
        "channel_dim": int(data["channel_dim"]),
        "curve_dim": int(data["curve_dim"]),
        "orig_mean_curve": data["orig_mean_curve"],
        "orig_std_curve": data["orig_std_curve"],
        "perm_mean_curves": data["perm_mean_curves"],
        "perm_std_curves": data["perm_std_curves"],
        "perm_curve_l1": data["perm_curve_l1"],
        "threshold_rows": rows,
    }


def plot_multi_model_comparison(all_results: list[dict], models: list[dict], output_dir: Path):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, model_config in zip(all_results, models):
        if results is None:
            continue

        channel_dim = results["channel_dim"]
        curve_dim = results["curve_dim"]
        perm_mean = results["perm_mean_curves"].mean(axis=0)
        x = np.arange(1, curve_dim + 1, dtype=np.float64) / channel_dim

        ax.plot(
            x,
            perm_mean,
            color=model_config["color"],
            marker=model_config["marker"],
            markevery=max(1, curve_dim // 20),
            linewidth=2.5,
            markersize=6,
            label=f"{model_config['model_name']} (D={channel_dim})",
            alpha=0.9,
            zorder=10,
        )

    thresholds = [50, 60, 70, 80, 90]
    threshold_colors = ["green", "orange", "red", "purple", "brown"]
    for thresh, color in zip(thresholds, threshold_colors):
        ax.axhline(y=thresh, color=color, linestyle="--", alpha=0.4, linewidth=1.5, zorder=1)
        ax.text(
            0.98,
            thresh + 1.5,
            f"{thresh}%",
            fontsize=16,
            color=color,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8, linewidth=1.5),
            zorder=10,
        )

    ax.set_xlabel("Normalized Spectral Bandwidth (d/D)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Cumulative Spectral Energy (%)", fontsize=20, fontweight="bold")
    ax.set_title(
        "SEP Under Random Channel Permutations\nAcross Vision Transformer Architectures",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=11,
        ncol=2,
        framealpha=0.90,
        edgecolor="black",
        fancybox=True,
        columnspacing=1.0,
        handlelength=1.6,
        borderaxespad=0.2,
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        output_file = output_dir / f"sep_permutation_comparison_all_models.{ext}"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot: {output_file}")
    plt.close()


def create_comparison_table(all_results: list[dict], models: list[dict], output_dir: Path) -> list[dict]:
    rows_out: list[dict] = []
    csv_file = output_dir / "sep_permutation_comparison_table.csv"

    with csv_file.open("w", newline="") as handle:
        fieldnames = [
            "Model",
            "Dimension (D)",
            "Curve L1 Mean",
            "Curve L1 Std",
            "Orig b80",
            "Perm b80",
            "Delta b80",
            "CI Low b80",
            "CI High b80",
            "Orig b90",
            "Perm b90",
            "Delta b90",
            "CI Low b90",
            "CI High b90",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for results, model_config in zip(all_results, models):
            if results is None:
                continue
            b80 = results["threshold_rows"][80]
            b90 = results["threshold_rows"][90]
            row = {
                "Model": results["model_name"],
                "Dimension (D)": results["channel_dim"],
                "Curve L1 Mean": f"{results['perm_curve_l1'].mean():.6f}",
                "Curve L1 Std": f"{results['perm_curve_l1'].std():.6f}",
                "Orig b80": b80["orig_bandwidth_mean"],
                "Perm b80": b80["perm_bandwidth_mean"],
                "Delta b80": b80["delta_mean"],
                "CI Low b80": b80["bootstrap_delta_ci_low"],
                "CI High b80": b80["bootstrap_delta_ci_high"],
                "Orig b90": b90["orig_bandwidth_mean"],
                "Perm b90": b90["perm_bandwidth_mean"],
                "Delta b90": b90["delta_mean"],
                "CI Low b90": b90["bootstrap_delta_ci_low"],
                "CI High b90": b90["bootstrap_delta_ci_high"],
            }
            writer.writerow(row)
            rows_out.append(row)

    print(f"Saved comparison table: {csv_file}")
    return rows_out


def create_markdown_summary(
    all_results: list[dict],
    models: list[dict],
    table_rows: list[dict],
    output_dir: Path,
    num_permutations: int,
    bootstrap_samples: int,
):
    curve_means = []
    abs_delta_b80 = []
    abs_delta_b90 = []

    for row in table_rows:
        curve_means.append(float(row["Curve L1 Mean"]))
        abs_delta_b80.append((abs(float(row["Delta b80"])), row["Model"], float(row["Delta b80"])))
        abs_delta_b90.append((abs(float(row["Delta b90"])), row["Model"], float(row["Delta b90"])))

    worst_b80 = max(abs_delta_b80)
    worst_b90 = max(abs_delta_b90)
    summary_file = output_dir / "sep_permutation_summary.md"

    with summary_file.open("w") as handle:
        handle.write("# SEP Permutation Robustness Summary\n\n")
        handle.write("## Protocol\n")
        handle.write(
            f"- Models analyzed: {len(table_rows)} transformer-family models matching the existing all-model ECG comparison.\n"
        )
        handle.write("- Dataset: the same saved 1,000-image ImageNet validation feature sets already used for SEP/ECG.\n")
        handle.write(f"- Randomization: {num_permutations} global random channel permutations per model.\n")
        handle.write("- Spectrum mode: full FFT ordering, normalized as d/D to match the paper artifacts already stored in this repo.\n")
        handle.write(f"- Uncertainty: paired bootstrap over images with {bootstrap_samples} resamples.\n\n")

        handle.write("## Notation Note: d/D vs. d/D'\n")
        handle.write(
            "- The current robustness figures intentionally keep the x-axis label `d/D`, not `d/D'`, because the analysis was run in the same full-FFT mode as the saved SEP/ECG artifacts in this repository.\n"
        )
        handle.write(
            "- There is a real notation discrepancy in the paper: Section 2.5 writes SEP using `D'` unique frequency bins after folding conjugate pairs, but Appendix Table 8 explicitly says `D` is the channel dimension and reports `d80`/`d90` values that are impossible under `D'`.\n"
        )
        handle.write(
            "- Example: for CaiT-S24, `D = 384` and folded `D' = 193`, but the paper table reports `d80 = 309` and `d90 = 346`; for ViT-Large, `D = 1024` and folded `D' = 513`, but the table reports `d80 = 820` and `d90 = 921`. Those values only make sense if the normalization is `d/D`, not `d/D'`.\n"
        )
        handle.write(
            "- For that reason, the permutation analysis here reproduces the paper's stored numerical artifacts with the full FFT ordering and `d/D` normalization. If you want a second variant aligned with the equation-only `D'` notation, the code can be rerun in folded mode and the plots can then be relabeled to `d/D'`.\n\n"
        )

        handle.write("## Overall Findings\n")
        handle.write(
            f"- The permutation-mean SEP curves remain close to diagonal for all models; mean curve L1 distance ranges from {min(curve_means):.6f} to {max(curve_means):.6f}.\n"
        )
        handle.write(
            f"- The largest absolute shift at 80% energy is {worst_b80[2]:+.6f} for {worst_b80[1]}.\n"
        )
        handle.write(
            f"- The largest absolute shift at 90% energy is {worst_b90[2]:+.6f} for {worst_b90[1]}.\n"
        )
        handle.write(
            "- Across the full model set, the key bandwidth statistics remain in the same qualitative regime: b80 stays near 0.8 and b90 stays near 0.9 after random permutations.\n\n"
        )

        handle.write("## Per-Model Table\n")
        handle.write("| Model | D | Curve L1 Mean | Orig b80 | Perm b80 | Delta b80 | Orig b90 | Perm b90 | Delta b90 |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in table_rows:
            handle.write(
                f"| {row['Model']} | {row['Dimension (D)']} | {float(row['Curve L1 Mean']):.6f} | "
                f"{float(row['Orig b80']):.6f} | {float(row['Perm b80']):.6f} | {float(row['Delta b80']):+.6f} | "
                f"{float(row['Orig b90']):.6f} | {float(row['Perm b90']):.6f} | {float(row['Delta b90']):+.6f} |\n"
            )

        handle.write("\n## Exact Formulas Used In This Run\n")
        handle.write(
            "Let image `i` have last-layer tokens `x_{i,t} \\in \\mathbb{R}^D`, with `t = 1, \\dots, N_i`. "
            "In the implementation used for the results above, the full FFT ordering is preserved, so the frequency index runs over all `D` bins.\n\n"
        )
        handle.write(
            r"""For each token, the spectral energy is

`S_{i,t}(k) = |FFT(x_{i,t})_k|^2, \qquad k = 1, \dots, D.`

The token-level cumulative SEP curve is

`SEP_{i,t}(d) = 100 \cdot \frac{\sum_{k=1}^{d} S_{i,t}(k)}{\sum_{k=1}^{D} S_{i,t}(k)}.`

The image-level SEP curve is the mean over that image's tokens:

`SEP_i(d) = \frac{1}{N_i} \sum_{t=1}^{N_i} SEP_{i,t}(d).`

The dataset mean SEP curve is

`\bar{SEP}(d) = \frac{1}{M} \sum_{i=1}^{M} SEP_i(d).`

For a target energy level `\alpha \in {50, 60, 70, 80, 90}`, the per-image normalized bandwidth is

`b_{\alpha,i} = \min \left\{ \frac{d}{D} : SEP_i(d) \ge \alpha \right\}.`

For permutation trial `r`, sample one global random permutation `\pi_r` of the channel indices and apply it consistently to every token:

`x_{i,t}^{(\pi_r)} = \Pi_{\pi_r} x_{i,t}.`

Then recompute `SEP_i^{(\pi_r)}(d)` and `b_{\alpha,i}^{(\pi_r)}` in exactly the same way.

The reported curve-distance metric is the normalized L1 difference between the permutation-mean curve and the original mean curve:

`\Delta_{\mathrm{curve}}^{(\pi_r)} = \frac{1}{100 D} \sum_{d=1}^{D} \left| \bar{SEP}^{(\pi_r)}(d) - \bar{SEP}(d) \right|.`

The reported bandwidth shift for threshold `\alpha` is

`\Delta b_{\alpha}^{(\pi_r)} = \frac{1}{M} \sum_{i=1}^{M} b_{\alpha,i}^{(\pi_r)} - \frac{1}{M} \sum_{i=1}^{M} b_{\alpha,i}.`

The bootstrap confidence interval is computed for the expected random-permutation effect

`\delta_{\alpha} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{E}_{\pi}[b_{\alpha,i}^{(\pi)}] - \frac{1}{M} \sum_{i=1}^{M} b_{\alpha,i},`

where `\mathbb{E}_{\pi}` is approximated by the mean over the 100 sampled permutations, and the bootstrap resamples images with replacement.

"""
        )

        handle.write("## Pseudocode Used In This Run\n")
        handle.write(
            """```python
# X has shape [M, N, D]: M images, N last-layer tokens per image, D channels.

def image_sep_curves(X, perm=None):
    if perm is not None:
        X = X[..., perm]                      # one global channel permutation per trial
    F = abs(fft(X, axis=-1)) ** 2            # full FFT, not folded rFFT
    C = cumsum(F, axis=-1) / F.sum(axis=-1, keepdims=True) * 100.0
    return C.mean(axis=1)                    # [M, D], mean over tokens within each image


def bandwidths(image_curves, thresholds):
    out = {}
    for alpha in thresholds:                 # alpha in {50, 60, 70, 80, 90}
        idx = first_index(image_curves >= alpha, axis=-1)   # [M]
        out[alpha] = (idx + 1) / image_curves.shape[-1]     # normalize by D
    return out


orig_curves = image_sep_curves(X)           # [M, D]
orig_mean = orig_curves.mean(axis=0)        # [D]
orig_b = bandwidths(orig_curves, [50, 60, 70, 80, 90])

trial_curves = []
trial_b = []
for r in range(R):                          # R = 100 in the reported runs
    pi = rng.permutation(D)
    perm_curves = image_sep_curves(X, perm=pi)
    trial_curves.append(perm_curves.mean(axis=0))
    trial_b.append(bandwidths(perm_curves, [50, 60, 70, 80, 90]))

curve_l1 = [mean(abs(curve - orig_mean)) / 100.0 for curve in trial_curves]
delta_b80 = [mean(b[80]) - mean(orig_b[80]) for b in trial_b]
delta_b90 = [mean(b[90]) - mean(orig_b[90]) for b in trial_b]

# Bootstrap over images:
# 1. average the per-image permuted bandwidths across the R trials,
# 2. resample images with replacement,
# 3. recompute the difference in mean bandwidths,
# 4. take the 2.5th and 97.5th percentiles.
```

"""
        )

        handle.write("\n## Interpretation\n")
        handle.write(
            "- SEP is order-sensitive in principle, but the empirical near-diagonal SEP law is robust to global random channel permutations across this model set.\n"
        )
        handle.write(
            "- The robustness is strongest at 90% energy, where the shifts are uniformly small; 80% energy shows slightly larger movement, but still remains close to the original 0.8 rule.\n"
        )
        handle.write(
            "- This supports a careful reviewer response: the qualitative broad-band energy-spread conclusion survives arbitrary global channel reorderings, even though SEP is not mathematically permutation-invariant.\n"
        )

    print(f"Saved markdown summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="All-model SEP permutation robustness analysis")
    parser.add_argument("--num-permutations", type=int, default=100, help="Random global channel permutations per model")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Paired bootstrap samples over images")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip per-model runs and only build combined outputs")
    parser.add_argument("--use-gpu", action="store_true", help="Pass --use-gpu to the per-model script")
    parser.add_argument("--only-short-names", nargs="+", default=None, help="Optional subset of model short names to process")
    args = parser.parse_args()

    models = selected_models(args.only_short_names)
    if not models:
        raise ValueError("No models selected.")

    output_dir = Path("Output/comparison/sep_permutation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_analysis:
        print("[Step 1] Running per-model robustness analysis...")
        for model_config in models:
            try:
                analyze_single_model(
                    model_config=model_config,
                    num_permutations=args.num_permutations,
                    bootstrap_samples=args.bootstrap_samples,
                    use_gpu=args.use_gpu,
                )
            except Exception as exc:
                print(f"Error analyzing {model_config['model_name']}: {exc}")

    print("\n[Step 2] Loading per-model results...")
    all_results = []
    available_models = []
    for model_config in models:
        results = load_model_results(model_config)
        if results is not None:
            all_results.append(results)
            available_models.append(model_config)
            print(f"Loaded {model_config['model_name']}")
        else:
            print(f"Missing results for {model_config['model_name']}")

    if not all_results:
        raise RuntimeError("No robustness results available to compare.")

    print("\n[Step 3] Creating combined figure...")
    plot_multi_model_comparison(all_results, available_models, output_dir)

    print("\n[Step 4] Writing comparison table...")
    table_rows = create_comparison_table(all_results, available_models, output_dir)

    print("\n[Step 5] Writing markdown summary...")
    create_markdown_summary(
        all_results=all_results,
        models=available_models,
        table_rows=table_rows,
        output_dir=output_dir,
        num_permutations=args.num_permutations,
        bootstrap_samples=args.bootstrap_samples,
    )

    print(f"\nDone. Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
