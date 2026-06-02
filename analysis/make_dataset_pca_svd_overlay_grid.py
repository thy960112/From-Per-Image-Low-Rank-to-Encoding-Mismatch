#!/usr/bin/env python3
"""
Create a single 2x2 grid figure that combines:
  (1) Dataset PCA spectrum (global cumulative energy)
  (2) Energy captured by shared PCA subspace (percentiles over images)
  (3) Overlay histogram for required d @ 95% energy:
        - Per-image SVD (input-dependent basis)
        - Dataset PCA (shared basis)
  (4) Overlay histogram for required d @ 99% energy:
        - Per-image SVD (input-dependent basis)
        - Dataset PCA (shared basis)

Inputs (already produced by this repo's scripts):
  - <model-root>/dataset_pca/dataset_pca_results.npz  (eigvals, curves)
  - <model-root>/svd/svd_results.npz                  (dims_95, dims_99, ...)

Outputs (default):
  - <model-root>/comparison/dataset_pca_svd_overlay_grid.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dims_for_threshold(curves: np.ndarray, threshold: float) -> np.ndarray:
    """Return smallest d (1-indexed) s.t. curves[i, d-1] >= threshold, for each image i."""
    hit = curves >= threshold
    first_idx0 = hit.argmax(axis=1)  # 0-indexed; returns 0 if never hit
    reached = hit.any(axis=1)
    dims = np.where(reached, first_idx0 + 1, curves.shape[1])
    return dims.astype(np.int32, copy=False)


def _setup_style(font_size: int) -> int:
    # Mirror analyze_dataset_pca.py style for consistency with existing figures.
    tick_size = max(int(font_size) - 3, 8)
    legend_size = max(int(font_size) - 7, 8)
    annot_size = max(int(font_size) - 5, 8)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": int(font_size),
            "axes.titlesize": int(font_size),
            "axes.titleweight": "bold",
            "axes.labelsize": int(font_size),
            "axes.labelweight": "bold",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "lines.linewidth": 2.5,
            "grid.alpha": 0.3,
        }
    )
    return annot_size


def _default_bins(energy: float) -> int:
    # Match replot_svd_separate.py / compare_svd_vs_dataset_pca_hist.py.
    mapping = {0.80: 20, 0.90: 35, 0.95: 40, 0.99: 50}
    for k, v in mapping.items():
        if abs(float(energy) - k) < 1e-9:
            return int(v)
    return 50


def _overlay_hist_required_d(
    ax: plt.Axes,
    *,
    svd_dims: np.ndarray,
    ds_dims: np.ndarray,
    energy: float,
    bins: int,
    annot_size: int,
) -> None:
    all_data = np.concatenate([svd_dims, ds_dims], axis=0).astype(np.float64, copy=False)
    lo = float(np.min(all_data))
    hi = float(np.max(all_data))
    if lo == hi:
        hi = lo + 1.0
    bin_edges = np.linspace(lo, hi, int(bins) + 1)

    svd_p99 = float(np.percentile(svd_dims, 99))
    ds_p99 = float(np.percentile(ds_dims, 99))

    def plot_overlay(
        data: np.ndarray,
        *,
        color: str,
        label: str,
        p99_val: float,
        hatch: str | None = None,
    ) -> None:
        weights = np.ones_like(data, dtype=np.float64) / max(len(data), 1)
        ax.hist(
            data,
            bins=bin_edges,
            weights=weights,
            histtype="bar",
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=2.0,
            label=label,
            hatch=hatch,
        )
        ax.hist(
            data,
            bins=bin_edges,
            weights=weights,
            histtype="step",
            color=color,
            linewidth=2.5,
            label=None,
        )
        ax.axvline(p99_val, color=color, linestyle="--", linewidth=2.5)

    plot_overlay(svd_dims, color="seagreen", label="Per-image SVD", p99_val=svd_p99, hatch=None)
    plot_overlay(
        ds_dims,
        color="steelblue",
        label="Dataset PCA (shared)",
        p99_val=ds_p99,
        hatch="//",
    )

    ax.set_title(f"Required d for {int(energy * 100)}% energy", pad=10)
    ax.set_xlabel("Required d")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)

    # Legend inside the axes (top-left).
    ax.legend(loc="upper left", framealpha=0.9)

    # Side-by-side stats blocks near top-center, with backgrounds matching each histogram.
    svd_stats = f"SVD\nMean: {float(np.mean(svd_dims)):.1f}\nStd: {float(np.std(svd_dims)):.1f}"
    ds_stats = f"Dataset PCA\nMean: {float(np.mean(ds_dims)):.1f}\nStd: {float(np.std(ds_dims)):.1f}"
    ax.text(
        0.49,
        0.98,
        svd_stats,
        transform=ax.transAxes,
        fontsize=annot_size,
        verticalalignment="top",
        horizontalalignment="right",
        multialignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="seagreen",
            edgecolor="seagreen",
            linewidth=1.5,
            alpha=0.25,
        ),
    )
    ax.text(
        0.51,
        0.98,
        ds_stats,
        transform=ax.transAxes,
        fontsize=annot_size,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round",
            facecolor="steelblue",
            edgecolor="steelblue",
            linewidth=1.5,
            alpha=0.25,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a 2x2 grid combining PCA spectrum/subspace + overlay required-d histograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Model output root (e.g., Output/cait). Must contain svd/ and dataset_pca/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the grid plot (default: <model-root>/comparison).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=21,
        help="Base font size (match existing repo plots).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95, 0.99],
        help="Energy thresholds drawn on spectrum/subspace plots.",
    )
    parser.add_argument(
        "--bins-95",
        type=int,
        default=None,
        help="Histogram bins for the 95% panel (default: style-matched).",
    )
    parser.add_argument(
        "--bins-99",
        type=int,
        default=None,
        help="Histogram bins for the 99% panel (default: style-matched).",
    )
    args = parser.parse_args()

    model_root: Path = args.model_root
    ds_npz = model_root / "dataset_pca" / "dataset_pca_results.npz"
    svd_npz = model_root / "svd" / "svd_results.npz"
    if not ds_npz.exists():
        raise FileNotFoundError(f"Missing dataset PCA results: {ds_npz}")
    if not svd_npz.exists():
        raise FileNotFoundError(f"Missing SVD results: {svd_npz}")

    out_dir = args.output_dir or (model_root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    annot_size = _setup_style(int(args.font_size))

    ds = np.load(ds_npz)
    eigvals = ds["eigvals"].astype(np.float64, copy=False)
    curves = ds["curves"].astype(np.float64, copy=False)
    D = int(eigvals.shape[0])

    thresholds: Sequence[float] = [float(t) for t in args.thresholds]
    cum_global = np.cumsum(eigvals) / np.maximum(np.sum(eigvals), 1e-12)
    q50, q10, q01 = np.percentile(curves, [50, 10, 1], axis=0)

    svd = np.load(svd_npz)
    svd_dims_95 = svd["dims_95"].astype(np.int32, copy=False)
    svd_dims_99 = svd["dims_99"].astype(np.int32, copy=False)
    ds_dims_95 = dims_for_threshold(curves, 0.95)
    ds_dims_99 = dims_for_threshold(curves, 0.99)

    bins_95 = int(args.bins_95) if args.bins_95 is not None else _default_bins(0.95)
    bins_99 = int(args.bins_99) if args.bins_99 is not None else _default_bins(0.99)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Spectrum.
    ax = axes[0, 0]
    x = np.arange(1, D + 1)
    ax.plot(x, cum_global, linewidth=2.5, label="Global cumulative energy (PCA)")
    for t in thresholds:
        d_t = int(np.searchsorted(cum_global, float(t)) + 1)
        ax.axhline(t, color="gray", linestyle="--", linewidth=1.5)
        ax.axvline(d_t, color="gray", linestyle="--", linewidth=1.5)
        # Put "d=..." below the dashed horizontal line to avoid covering the curve.
        y_txt = max(float(t) - 0.06, 0.02)
        ax.text(
            d_t,
            y_txt,
            f"d={d_t}",
            fontsize=annot_size,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
    ax.set_xlabel("d (components)")
    ax.set_ylabel("Cumulative energy")
    ax.set_title("PCA spectrum")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, D)
    ax.set_ylim(0, 1.0)

    # (2) Captured-energy percentiles (shared basis).
    ax = axes[0, 1]
    xd = np.arange(1, curves.shape[1] + 1)
    ax.plot(xd, q50, linewidth=2.5, label="Median over images")
    ax.plot(xd, q10, linewidth=2.5, label="10th percentile over images")
    ax.plot(xd, q01, linewidth=2.5, label="1st percentile over images")
    for t in thresholds:
        ax.axhline(t, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("d (components)")
    ax.set_ylabel("Energy captured")
    ax.set_title("Energy captured by shared PCA subspace")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, curves.shape[1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")

    # (3) Required-d overlay @ 95% and (4) @ 99%.
    _overlay_hist_required_d(
        axes[1, 0],
        svd_dims=svd_dims_95,
        ds_dims=ds_dims_95,
        energy=0.95,
        bins=bins_95,
        annot_size=annot_size,
    )
    _overlay_hist_required_d(
        axes[1, 1],
        svd_dims=svd_dims_99,
        ds_dims=ds_dims_99,
        energy=0.99,
        bins=bins_99,
        annot_size=annot_size,
    )

    plt.tight_layout()

    stem = "dataset_pca_svd_overlay_grid"
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{stem}.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)

    print("Loaded data from:")
    print(f"  - Dataset PCA: {ds_npz}")
    print(f"  - SVD:         {svd_npz}")


if __name__ == "__main__":
    main()

