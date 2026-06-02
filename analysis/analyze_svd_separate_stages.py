#!/usr/bin/env python3
"""
Stage-wise SVD analysis for object-array features (Swin / CNN features_only).

Input format (per image):
  - `.npy` file that loads to a NumPy object array of length S (stages)
  - each stage is a float array of shape [N_i, C_i] (tokens x channels)

For each stage, we compute per-image:
  - rank (count of singular values > eps)
  - dimensions needed to reach 80/90/95/99% cumulative energy

Outputs:
  - per-stage histograms (PNG + PDF) under `--output-dir`
  - `svd_stage_results.npz` with raw arrays + 99th percentiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional GPU acceleration (CuPy)
try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("GPU (cupy) is available and will be used for acceleration when --use-gpu is set.")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("GPU (cupy) not available, using CPU (numpy) instead.")


# Match the visual style used in other plotting scripts in this repo.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
    }
)


def _svd_singular_values(x_nc: np.ndarray, use_gpu: bool) -> np.ndarray:
    """Return singular values (sigma) for a matrix [N, C]."""
    if use_gpu and GPU_AVAILABLE:
        x_gpu = cp.asarray(x_nc)
        try:
            sigma = cp.linalg.svd(x_gpu, full_matrices=False, compute_uv=False)
        except TypeError:
            # Older CuPy builds may not expose compute_uv.
            _, sigma, _ = cp.linalg.svd(x_gpu, full_matrices=False)
        return cp.asnumpy(sigma)

    return np.linalg.svd(x_nc, full_matrices=False, compute_uv=False)


def _energy_dims_from_sigma(sigma: np.ndarray) -> dict[str, int]:
    """Compute required dimensions for multiple energy thresholds from singular values."""
    sigma2 = sigma.astype(np.float64) ** 2
    total = float(np.sum(sigma2))
    if total <= 0:
        # Degenerate case: all zeros
        return {"80": 0, "90": 0, "95": 0, "99": 0}

    cumulative = np.cumsum(sigma2) / total
    return {
        "80": int(np.searchsorted(cumulative, 0.80) + 1),
        "90": int(np.searchsorted(cumulative, 0.90) + 1),
        "95": int(np.searchsorted(cumulative, 0.95) + 1),
        "99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def plot_single_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
    filename: str,
    percentile_val: float,
    bins: int,
    output_dir: Path,
):
    """Create histogram with a 99th percentile line (ratio y-axis)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    counts_unnorm, bin_edges = np.histogram(data, bins=bins)
    ratios = counts_unnorm / max(1, len(data))

    ax.bar(
        bin_edges[:-1],
        ratios,
        width=np.diff(bin_edges),
        color=color,
        alpha=0.7,
        edgecolor="black",
        align="edge",
    )

    ax.axvline(
        x=percentile_val,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label=f"99th percentile: {percentile_val:.1f}",
    )

    y_max = float(ratios.max()) if len(ratios) else 0.0
    ax.text(
        percentile_val,
        y_max * 0.95 if y_max > 0 else 0.0,
        f" {percentile_val:.1f}",
        fontsize=14,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel("Ratio", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="upper right")

    mean_val = float(np.mean(data)) if len(data) else float("nan")
    std_val = float(np.std(data)) if len(data) else float("nan")
    median_val = float(np.median(data)) if len(data) else float("nan")
    stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


def analyze_svd_stages(
    feature_dir: Path,
    output_dir: Path,
    stages: list[int] | None = None,
    use_gpu: bool = False,
    rank_eps: float = 1e-10,
    max_files: int | None = None,
):
    feature_files = sorted(feature_dir.glob("*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No .npy files found in: {feature_dir}")

    if max_files is not None:
        feature_files = feature_files[:max_files]

    first = np.load(feature_files[0], allow_pickle=True)
    if first.dtype != object:
        raise ValueError(
            "Expected object-array features (stages). "
            "If you have ViT/CaiT features [L,N,C], use analyze_svd_separate.py / analyze_layers_svd.py."
        )

    num_stages_total = int(first.shape[0])
    if stages is None:
        stages = list(range(num_stages_total))
    else:
        stages = [int(s) for s in stages]
        bad = [s for s in stages if s < 0 or s >= num_stages_total]
        if bad:
            raise ValueError(f"Invalid stage indices {bad}; available stages are [0..{num_stages_total - 1}]")

    # Static shapes per stage (for labeling).
    stage_n = []
    stage_c = []
    for s in stages:
        n, c = first[s].shape
        stage_n.append(int(n))
        stage_c.append(int(c))

    # Storage: stage -> list over images.
    dims_80 = {s: [] for s in stages}
    dims_90 = {s: [] for s in stages}
    dims_95 = {s: [] for s in stages}
    dims_99 = {s: [] for s in stages}
    ranks = {s: [] for s in stages}

    for file_path in tqdm(feature_files, desc="Processing feature maps"):
        feats = np.load(file_path, allow_pickle=True)
        for s in stages:
            x = feats[s]
            sigma = _svd_singular_values(x, use_gpu=use_gpu)
            ranks[s].append(int(np.sum(sigma > rank_eps)))

            ed = _energy_dims_from_sigma(sigma)
            dims_80[s].append(ed["80"])
            dims_90[s].append(ed["90"])
            dims_95[s].append(ed["95"])
            dims_99[s].append(ed["99"])

    # Convert to arrays and report.
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStage-wise Statistics (mean+/-std; 99th percentile in parentheses):")
    header = f"{'Stage':<7} {'N':>6} {'C':>6} | {'80%':>10} {'90%':>10} {'95%':>10} {'99%':>10} {'Rank':>10}"
    print(header)
    print("-" * len(header))

    for idx, s in enumerate(stages):
        a80 = np.asarray(dims_80[s])
        a90 = np.asarray(dims_90[s])
        a95 = np.asarray(dims_95[s])
        a99 = np.asarray(dims_99[s])
        ar = np.asarray(ranks[s])

        def fmt(a: np.ndarray) -> str:
            return f"{np.mean(a):6.1f}+/-{np.std(a):5.1f} ({np.percentile(a, 99):5.1f})"

        print(
            f"{s:<7d} {stage_n[idx]:>6d} {stage_c[idx]:>6d} | "
            f"{fmt(a80):>10} {fmt(a90):>10} {fmt(a95):>10} {fmt(a99):>10} {fmt(ar):>10}"
        )

    # Save npz
    save_dict: dict[str, np.ndarray] = {
        "stage_indices": np.asarray(stages, dtype=np.int32),
        "stage_N": np.asarray(stage_n, dtype=np.int32),
        "stage_C": np.asarray(stage_c, dtype=np.int32),
    }
    for s in stages:
        save_dict[f"stage_{s}_dims_80"] = np.asarray(dims_80[s], dtype=np.int32)
        save_dict[f"stage_{s}_dims_90"] = np.asarray(dims_90[s], dtype=np.int32)
        save_dict[f"stage_{s}_dims_95"] = np.asarray(dims_95[s], dtype=np.int32)
        save_dict[f"stage_{s}_dims_99"] = np.asarray(dims_99[s], dtype=np.int32)
        save_dict[f"stage_{s}_ranks"] = np.asarray(ranks[s], dtype=np.int32)

        save_dict[f"stage_{s}_p99_dims_80"] = np.asarray(np.percentile(save_dict[f"stage_{s}_dims_80"], 99))
        save_dict[f"stage_{s}_p99_dims_90"] = np.asarray(np.percentile(save_dict[f"stage_{s}_dims_90"], 99))
        save_dict[f"stage_{s}_p99_dims_95"] = np.asarray(np.percentile(save_dict[f"stage_{s}_dims_95"], 99))
        save_dict[f"stage_{s}_p99_dims_99"] = np.asarray(np.percentile(save_dict[f"stage_{s}_dims_99"], 99))
        save_dict[f"stage_{s}_p99_ranks"] = np.asarray(np.percentile(save_dict[f"stage_{s}_ranks"], 99))

    npz_path = output_dir / "svd_stage_results.npz"
    np.savez(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")

    # Plots per stage.
    print("\nCreating per-stage histograms...")
    for idx, s in enumerate(stages):
        n = stage_n[idx]
        c = stage_c[idx]

        a80 = np.asarray(dims_80[s])
        a90 = np.asarray(dims_90[s])
        a95 = np.asarray(dims_95[s])
        a99 = np.asarray(dims_99[s])
        ar = np.asarray(ranks[s])

        plot_single_histogram(
            a99,
            f"Dimensions for 99% Energy Restoration\\n(Stage {s} - N={n}, C={c})",
            "Dimension",
            "steelblue",
            f"stage{s}_svd_99_percent_energy",
            float(np.percentile(a99, 99)),
            bins=50,
            output_dir=output_dir,
        )
        plot_single_histogram(
            a95,
            f"Dimensions for 95% Energy Restoration\\n(Stage {s} - N={n}, C={c})",
            "Dimension",
            "seagreen",
            f"stage{s}_svd_95_percent_energy",
            float(np.percentile(a95, 99)),
            bins=40,
            output_dir=output_dir,
        )
        plot_single_histogram(
            a90,
            f"Dimensions for 90% Energy Restoration\\n(Stage {s} - N={n}, C={c})",
            "Dimension",
            "coral",
            f"stage{s}_svd_90_percent_energy",
            float(np.percentile(a90, 99)),
            bins=35,
            output_dir=output_dir,
        )
        plot_single_histogram(
            a80,
            f"Dimensions for 80% Energy Restoration\\n(Stage {s} - N={n}, C={c})",
            "Dimension",
            "mediumpurple",
            f"stage{s}_svd_80_percent_energy",
            float(np.percentile(a80, 99)),
            bins=20,
            output_dir=output_dir,
        )
        plot_single_histogram(
            ar,
            f"Rank Distribution of Feature Maps\\n(Stage {s} - N={n}, C={c})",
            "Rank",
            "crimson",
            f"stage{s}_svd_rank_distribution",
            float(np.percentile(ar, 99)),
            bins=10,
            output_dir=output_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stage-wise SVD histograms + percentiles for object-array features (Swin/CNN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-dir", type=Path, required=True, help="Directory containing stage feature .npy files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save plots + results")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="*",
        default=None,
        help="Optional stage indices to analyze (default: all stages in the feature files).",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration via cupy (if available)")
    parser.add_argument("--rank-eps", type=float, default=1e-10, help="Sigma threshold for numerical rank")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of feature files")
    args = parser.parse_args()

    analyze_svd_stages(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        stages=args.stages,
        use_gpu=args.use_gpu,
        rank_eps=args.rank_eps,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    raise SystemExit(main())
