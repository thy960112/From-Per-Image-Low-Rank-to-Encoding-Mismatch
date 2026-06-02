#!/usr/bin/env python3
"""
Create a single comparison figure that combines:
  (1) Per-image SVD histogram (input-dependent basis) and
  (2) Dataset-level PCA histogram (shared basis)
for the same target energy threshold (80/90/95/99%).

This is intended to combine plots like:
  - Output/<model>/svd/svd_95_percent_energy_times.pdf
  - Output/<model>/dataset_pca/dataset_pca_analysis_required_d_95.pdf

We regenerate from the original numeric data:
  - SVD: Output/<model>/svd/svd_results.npz (dims_80/90/95/99)
  - Dataset PCA: Output/<model>/dataset_pca/dataset_pca_results.npz (curves)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _dims_for_threshold_from_curves(curves: np.ndarray, threshold: float) -> np.ndarray:
    """curves: [num_images, D] cumulative captured energy; return dims [num_images] (1-indexed)."""
    hit = curves >= threshold
    first = hit.argmax(axis=1)  # 0-indexed; returns 0 if never hit
    reached = hit.any(axis=1)
    dims = np.where(reached, first + 1, curves.shape[1])
    return dims.astype(np.int32, copy=False)


def _energy_to_key(energy: float) -> str:
    # Match saved keys in svd_results.npz
    mapping = {0.80: "dims_80", 0.90: "dims_90", 0.95: "dims_95", 0.99: "dims_99"}
    for k, v in mapping.items():
        if abs(energy - k) < 1e-9:
            return v
    raise ValueError("energy must be one of: 0.80, 0.90, 0.95, 0.99")


def _default_bins(energy: float) -> int:
    # Mirror the choices in replot_svd_separate.py for visual consistency.
    mapping = {0.80: 20, 0.90: 35, 0.95: 40, 0.99: 50}
    for k, v in mapping.items():
        if abs(energy - k) < 1e-9:
            return v
    return 50


def _setup_style(font_size: int) -> None:
    tick_size = max(font_size - 3, 8)
    legend_size = max(font_size - 7, 8)
    annot_size = max(font_size - 6, 8)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": int(font_size),
            "axes.titlesize": int(font_size),
            "axes.titleweight": "bold",
            "axes.labelsize": int(font_size),
            "axes.labelweight": "bold",
            "xtick.labelsize": int(tick_size),
            "ytick.labelsize": int(tick_size),
            "legend.fontsize": int(legend_size),
            "lines.linewidth": 2.5,
            "grid.alpha": 0.3,
        }
    )
    return annot_size


def _load_inputs(model_root: Path) -> Tuple[Path, Path]:
    svd_npz = model_root / "svd" / "svd_results.npz"
    ds_npz = model_root / "dataset_pca" / "dataset_pca_results.npz"
    if not svd_npz.exists():
        raise FileNotFoundError(f"Missing SVD results: {svd_npz}")
    if not ds_npz.exists():
        raise FileNotFoundError(f"Missing dataset PCA results: {ds_npz}")
    return svd_npz, ds_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-image SVD vs dataset PCA (shared) histograms in one figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Model output root (e.g., Output/cait). Must contain svd/svd_results.npz and dataset_pca/dataset_pca_results.npz.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=0.95,
        choices=[0.80, 0.90, 0.95, 0.99],
        help="Target energy threshold.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="subplots",
        choices=["subplots", "overlay"],
        help="How to combine the two histograms: side-by-side subplots or a single overlaid axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the comparison plot (default: <model-root>/comparison).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=25,
        help="Base font size.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Histogram bins (default: match replot_svd_separate.py style per threshold).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title (suptitle).",
    )
    args = parser.parse_args()

    model_root: Path = args.model_root
    svd_npz, ds_npz = _load_inputs(model_root)
    out_dir = args.output_dir or (model_root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    energy = float(args.energy)
    bins = int(args.bins) if args.bins is not None else _default_bins(energy)
    svd_key = _energy_to_key(energy)

    annot_size = _setup_style(int(args.font_size))

    # Load SVD dims (per-image).
    svd = np.load(svd_npz)
    svd_dims = svd[svd_key].astype(np.int32, copy=False)
    svd_p99 = float(np.percentile(svd_dims, 99))

    # Load dataset PCA dims (shared basis).
    ds = np.load(ds_npz)
    curves = ds["curves"]
    ds_dims = _dims_for_threshold_from_curves(curves, energy)
    ds_p99 = float(np.percentile(ds_dims, 99))

    def stats_text(prefix: str, data: np.ndarray, p99_val: float) -> str:
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        return f"{prefix}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}\n99th pct: {p99_val:.0f}"

    title = args.title or (f"Required d for {int(energy*100)}% energy: Per-image SVD vs Dataset PCA")

    stem = f"compare_required_d_{int(energy*100)}_svd_vs_dataset_pca"
    if args.layout == "subplots":
        # Two panels for clean comparison (avoids the huge empty gap on a shared x-axis).
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        def plot_hist(ax, data, *, color, panel_title, p99_val, legend_loc, stats_loc, stats_prefix):
            counts, bin_edges = np.histogram(data, bins=bins)
            ratios = counts / max(len(data), 1)
            ax.bar(
                bin_edges[:-1],
                ratios,
                width=np.diff(bin_edges),
                color=color,
                alpha=0.75,
                edgecolor="black",
                align="edge",
            )
            ax.axvline(
                p99_val,
                color="black",
                linestyle="--",
                linewidth=2.5,
                label=f"99th pct: {p99_val:.0f}",
            )
            ax.set_title(panel_title, pad=10)
            ax.set_xlabel("Required d")
            ax.grid(True, alpha=0.3)
            ax.legend(loc=legend_loc, framealpha=0.9)

            x, y, ha = stats_loc
            ax.text(
                x,
                y,
                stats_text(stats_prefix, data, p99_val),
                transform=ax.transAxes,
                fontsize=annot_size,
                verticalalignment="top",
                horizontalalignment=ha,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
            )

        plot_hist(
            axes[0],
            svd_dims,
            color="seagreen",
            panel_title=f"Per-image SVD ({int(energy*100)}% energy)",
            p99_val=svd_p99,
            legend_loc="upper right",
            stats_loc=(0.02, 0.98, "left"),
            stats_prefix="SVD",
        )
        axes[0].set_ylabel("Ratio")

        plot_hist(
            axes[1],
            ds_dims,
            color="steelblue",
            panel_title=f"Dataset PCA ({int(energy*100)}% energy)",
            p99_val=ds_p99,
            legend_loc="upper left",
            stats_loc=(0.98, 0.98, "right"),
            stats_prefix="Dataset PCA",
        )

        fig.suptitle(
            title,
            fontsize=int(args.font_size) + 2,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
    else:
        # Single axes with two overlaid histograms for direct comparison.
        # Match the PCA spectrum/subspace aspect ratio used in analyze_dataset_pca.py (7:5).
        fig_w = 10.0
        fig_h = fig_w * (5.0 / 7.0)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        all_data = np.concatenate([svd_dims, ds_dims], axis=0)
        lo = float(np.min(all_data))
        hi = float(np.max(all_data))
        if lo == hi:
            hi = lo + 1.0
        bin_edges = np.linspace(lo, hi, bins + 1)

        def plot_overlay(
            data: np.ndarray,
            *,
            color: str,
            label: str,
            p99_val: float,
            hatch: str | None = None,
        ) -> None:
            # Use filled bars (closer to the side-by-side aesthetic) with alpha + optional hatch
            # to keep the two distributions distinguishable.
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
            # Outline on top to keep both distributions visible even where bars overlap.
            ax.hist(
                data,
                bins=bin_edges,
                weights=weights,
                histtype="step",
                color=color,
                linewidth=2.5,
                label=None,
            )
            # Show 99th-percentile markers, but don't add extra legend entries (keeps legend compact).
            ax.axvline(p99_val, color=color, linestyle="--", linewidth=2.5)

        plot_overlay(svd_dims, color="seagreen", label="Per-image SVD", p99_val=svd_p99, hatch=None)
        plot_overlay(
            ds_dims,
            color="steelblue",
            label="Dataset PCA",
            p99_val=ds_p99,
            hatch="//",
        )

        ax.set_xlabel("Required d")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)

        # Put the legend at the top-center, above the stats blocks.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, framealpha=0.9)

        # Separate stats boxes and keep them inside the axes (avoid legend overlap).
        # Place them near the top-center and side-by-side for a compact, paper-friendly layout.
        ax.text(
            0.49,
            0.82,
            stats_text("SVD", svd_dims, svd_p99),
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
            0.82,
            stats_text("Dataset PCA", ds_dims, ds_p99),
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
        # Use an axes title (closer to the plot). Keep it short: "Required d for XX% energy".
        short_title = title.split(":", 1)[0].strip()
        ax.set_title(short_title, fontsize=int(args.font_size) + 1, fontweight="bold", pad=10)

        plt.tight_layout()
        stem = f"{stem}_overlay"

    for ext in ("png", "pdf"):
        out_path = out_dir / f"{stem}.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)

    print("Loaded data from:")
    print(f"  - SVD:        {svd_npz}")
    print(f"  - Dataset PCA: {ds_npz}")


if __name__ == "__main__":
    main()
