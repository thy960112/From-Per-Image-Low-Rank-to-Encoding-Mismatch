#!/usr/bin/env python3
"""
Dataset-level PCA / SVD-style analysis for ViT token feature maps.

This script implements the "global subspace" experiment suggested in `dataset_analysis.md`:
  1) Build a dataset-level (uncentered) second-moment matrix:
        C = (1 / total_tokens) * sum_i X_i^T X_i
     where X_i is the last-layer token feature matrix for image i (tokens x channels).
  2) Eigendecompose C to obtain a *single shared* channel basis V.
  3) Evaluate, per image, how much energy is captured by the same global subspace:
        EnergyCaptured_i(d) = || X_i V_d ||_F^2 / ||X_i||_F^2

Features are expected to be stored one file per image as .npy (default extractor output),
or as .npz with a single array entry. For .npy files produced by `extract_features.py`,
the array shape is typically [L, N, C]; we use the last layer by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def _select_array_from_npz(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    keys = list(npz.keys())
    if not keys:
        raise ValueError("Empty .npz file (no arrays found).")
    # Prefer common names, otherwise fall back to single-entry or first key.
    for k in ("features", "feat", "x", "arr_0"):
        if k in npz:
            return npz[k]
    if len(keys) == 1:
        return npz[keys[0]]
    return npz[keys[0]]


def load_last_layer_feature(
    path: Path,
    *,
    layer: int = -1,
    token_policy: str = "auto",
) -> np.ndarray:
    """
    Load a feature file and return a 2D matrix X of shape [tokens, channels].

    token_policy:
      - "keep": keep tokens as-is.
      - "drop": drop token index 0 (treat as CLS).
      - "auto": drop token 0 only if tokens-1 looks like a patch grid (perfect square).
    """
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            arr = _select_array_from_npz(npz)
    else:
        raise ValueError(f"Unsupported feature file type: {path}")

    # Swin extractor can store an object array with per-stage features.
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr[layer]

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Unexpected payload type in {path}: {type(arr)}")

    if arr.ndim == 3:
        # [L, N, C] -> select layer
        X = arr[layer]
    elif arr.ndim == 2:
        # [N, C]
        X = arr
    else:
        raise ValueError(f"Unsupported feature array shape {arr.shape} in {path}")

    if token_policy not in {"auto", "keep", "drop"}:
        raise ValueError(f"Invalid token_policy={token_policy}. Use auto|keep|drop.")

    if token_policy == "drop":
        if X.shape[0] < 2:
            raise ValueError(f"Cannot drop CLS token: tokens={X.shape[0]} in {path}")
        X = X[1:, :]
    elif token_policy == "auto":
        # Heuristic: if N-1 looks like a square patch grid, assume token[0] is CLS.
        if X.shape[0] >= 2 and _is_perfect_square(X.shape[0] - 1):
            X = X[1:, :]

    # Ensure float32 for compute; accumulate in float64 where needed.
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    return X


def list_feature_files(feature_dir: Path) -> List[Path]:
    files = []
    for ext in ("*.npy", "*.npz"):
        files.extend(sorted(feature_dir.glob(ext)))
    return files


def compute_second_moment(
    feature_files: Sequence[Path],
    *,
    layer: int = -1,
    token_policy: str = "auto",
    batch_images: int = 64,
    center: bool = False,
    backend: str = "numpy",
    device: str = "cuda",
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Compute C = (1/total_tokens) * sum X^T X over all images (uncentered by default).

    Returns:
      C (float64): [D, D]
      mu (float64) or None: [D] mean token feature if center=True
      total_tokens (int)
    """
    if len(feature_files) == 0:
        raise ValueError("No feature files found.")

    if backend not in {"numpy", "torch"}:
        raise ValueError(f"Invalid backend={backend}. Use numpy|torch.")

    # Probe shape from the first file.
    X0 = load_last_layer_feature(feature_files[0], layer=layer, token_policy=token_policy)
    D = int(X0.shape[1])

    if backend == "torch":
        try:
            import torch
        except Exception as e:
            raise ImportError("backend=torch requested but PyTorch is not available.") from e

        torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        dtype = torch.float32
        C_sum_t = torch.zeros((D, D), dtype=dtype, device=torch_device)
        sum_x_t = torch.zeros((D,), dtype=dtype, device=torch_device) if center else None
        total_tokens = 0

        batch: List[np.ndarray] = []

        def flush_batch_t() -> None:
            nonlocal C_sum_t, sum_x_t, total_tokens, batch
            if not batch:
                return
            B_np = np.concatenate(batch, axis=0)
            B_t = torch.from_numpy(B_np).to(torch_device, non_blocking=True)
            C_sum_t += B_t.transpose(0, 1) @ B_t
            if sum_x_t is not None:
                sum_x_t += B_t.sum(dim=0)
            total_tokens += int(B_t.shape[0])
            batch = []

        for p in tqdm(feature_files, desc="Accumulating second moment C", unit="file"):
            X = load_last_layer_feature(p, layer=layer, token_policy=token_policy)
            if X.shape[1] != D:
                raise ValueError(f"Feature dim mismatch in {p}: expected D={D}, got {X.shape[1]}")
            batch.append(X)
            if len(batch) >= batch_images:
                flush_batch_t()
        flush_batch_t()

        C_t = C_sum_t / max(total_tokens, 1)
        mu_t = None
        if center:
            assert sum_x_t is not None
            mu_t = sum_x_t / max(total_tokens, 1)
            C_t = C_t - torch.outer(mu_t, mu_t)

        # Return CPU numpy for downstream (plotting / saving).
        C = C_t.detach().cpu().numpy().astype(np.float64, copy=False)
        mu = mu_t.detach().cpu().numpy().astype(np.float64, copy=False) if mu_t is not None else None
        return C, mu, total_tokens

    C_sum = np.zeros((D, D), dtype=np.float64)
    sum_x = np.zeros((D,), dtype=np.float64) if center else None
    total_tokens = 0

    batch: List[np.ndarray] = []
    batch_tokens = 0

    def flush_batch() -> None:
        nonlocal C_sum, sum_x, total_tokens, batch, batch_tokens
        if not batch:
            return
        B = np.concatenate(batch, axis=0)  # [sum_tokens, D]
        # Accumulate in float64 for stability.
        C_sum += (B.T @ B).astype(np.float64, copy=False)
        if sum_x is not None:
            sum_x += B.sum(axis=0, dtype=np.float64)
        total_tokens += B.shape[0]
        batch = []
        batch_tokens = 0

    for p in tqdm(feature_files, desc="Accumulating second moment C", unit="file"):
        X = load_last_layer_feature(p, layer=layer, token_policy=token_policy)
        if X.shape[1] != D:
            raise ValueError(f"Feature dim mismatch in {p}: expected D={D}, got {X.shape[1]}")
        batch.append(X)
        batch_tokens += X.shape[0]
        if len(batch) >= batch_images:
            flush_batch()

    flush_batch()

    C = C_sum / max(total_tokens, 1)
    mu = None
    if center:
        assert sum_x is not None
        mu = sum_x / max(total_tokens, 1)
        C = C - np.outer(mu, mu)
    return C, mu, total_tokens


def eigh_descending(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with descending eigenvalues."""
    w, V = np.linalg.eigh(C)  # ascending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    return w, V


def per_image_energy_curves(
    feature_files: Sequence[Path],
    V: np.ndarray,
    *,
    layer: int = -1,
    token_policy: str = "auto",
    batch_images: int = 64,
    max_d: Optional[int] = None,
    backend: str = "numpy",
    device: str = "cuda",
    mean_token: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-image cumulative captured-energy curves under the fixed basis V.

    Returns:
      curves: [num_images, d] where curves[i, d-1] = EnergyCaptured_i(d)
    """
    if len(feature_files) == 0:
        raise ValueError("No feature files found.")
    if backend not in {"numpy", "torch"}:
        raise ValueError(f"Invalid backend={backend}. Use numpy|torch.")

    D = int(V.shape[0])
    d = int(D if max_d is None else min(max_d, D))
    Vd = V[:, :d].astype(np.float32, copy=False)
    mu = None
    if mean_token is not None:
        if mean_token.shape != (D,):
            raise ValueError(f"mean_token has shape {mean_token.shape}, expected ({D},)")
        mu = mean_token.astype(np.float32, copy=False)

    if backend == "torch":
        try:
            import torch
        except Exception as e:
            raise ImportError("backend=torch requested but PyTorch is not available.") from e

        torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        Vd_t = torch.from_numpy(Vd).to(torch_device, non_blocking=True)
        mu_t = torch.from_numpy(mu).to(torch_device, non_blocking=True) if mu is not None else None

        X0 = load_last_layer_feature(feature_files[0], layer=layer, token_policy=token_policy)
        N0 = int(X0.shape[0])
        tokens_constant = True
        curves = np.zeros((len(feature_files), d), dtype=np.float32)

        batch: List[np.ndarray] = []
        batch_idx: List[int] = []

        def flush_batch_t() -> None:
            nonlocal batch, batch_idx, tokens_constant
            if not batch:
                return
            if tokens_constant:
                B_np = np.concatenate(batch, axis=0)  # [B*N0, D]
                B_t = torch.from_numpy(B_np).to(torch_device, non_blocking=True)
                if mu_t is not None:
                    B_t = B_t - mu_t
                Y = B_t @ Vd_t  # [B*N0, d]
                Y = Y.view(len(batch), N0, d)
                energies = (Y * Y).sum(dim=1)  # [B, d]
                totals = energies.sum(dim=1, keepdim=True).clamp_min_(1e-12)
                cum = torch.cumsum(energies, dim=1) / totals
                cum_np = cum.detach().cpu().numpy().astype(np.float32, copy=False)
                for bi, img_i in enumerate(batch_idx):
                    curves[img_i] = cum_np[bi]
            else:
                for X, img_i in zip(batch, batch_idx):
                    X_t = torch.from_numpy(X).to(torch_device, non_blocking=True)
                    if mu_t is not None:
                        X_t = X_t - mu_t
                    Y = X_t @ Vd_t
                    e = (Y * Y).sum(dim=0)
                    tot = float(e.sum().item())
                    if tot <= 0:
                        curves[img_i] = 0.0
                    else:
                        curves[img_i] = (torch.cumsum(e, dim=0) / tot).detach().cpu().numpy().astype(
                            np.float32, copy=False
                        )
            batch = []
            batch_idx = []

        for i, p in enumerate(tqdm(feature_files, desc="Computing per-image curves", unit="file")):
            X = load_last_layer_feature(p, layer=layer, token_policy=token_policy)
            if X.shape[0] != N0:
                tokens_constant = False
            batch.append(X)
            batch_idx.append(i)
            if len(batch) >= batch_images:
                flush_batch_t()
        flush_batch_t()
        return curves

    # Probe N to enable fast batching with reshape when token counts are constant.
    X0 = load_last_layer_feature(feature_files[0], layer=layer, token_policy=token_policy)
    N0 = int(X0.shape[0])
    tokens_constant = True

    curves = np.zeros((len(feature_files), d), dtype=np.float32)

    batch: List[np.ndarray] = []
    batch_idx: List[int] = []

    def flush_batch() -> None:
        nonlocal batch, batch_idx, tokens_constant
        if not batch:
            return
        if tokens_constant:
            B = np.concatenate(batch, axis=0)  # [B*N0, D]
            if mu is not None:
                B = B - mu
            Y = B @ Vd  # [B*N0, d]
            Y = Y.reshape(len(batch), N0, d)
            energies = (Y * Y).sum(axis=1)  # [B, d]
            totals = energies.sum(axis=1, keepdims=True)  # [B, 1]
            totals = np.maximum(totals, 1e-12)
            cum = np.cumsum(energies, axis=1) / totals
            for bi, img_i in enumerate(batch_idx):
                curves[img_i] = cum[bi].astype(np.float32, copy=False)
        else:
            for X, img_i in zip(batch, batch_idx):
                if mu is not None:
                    X = X - mu
                Y = X @ Vd
                e = (Y * Y).sum(axis=0)
                tot = float(e.sum())
                if tot <= 0:
                    curves[img_i] = 0.0
                else:
                    curves[img_i] = (np.cumsum(e) / tot).astype(np.float32, copy=False)
        batch = []
        batch_idx = []

    for i, p in enumerate(tqdm(feature_files, desc="Computing per-image curves", unit="file")):
        X = load_last_layer_feature(p, layer=layer, token_policy=token_policy)
        if X.shape[0] != N0:
            tokens_constant = False
        batch.append(X)
        batch_idx.append(i)
        if len(batch) >= batch_images:
            flush_batch()
    flush_batch()
    return curves


def dims_for_threshold(curves: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each image, return the smallest d (1-indexed) such that curves[i, d-1] >= threshold.
    """
    if curves.ndim != 2:
        raise ValueError(f"Expected curves to be 2D, got {curves.shape}")
    hit = curves >= threshold
    # argmax returns 0 when all-False, so we handle "never reaches threshold" explicitly.
    first_idx0 = hit.argmax(axis=1)  # 0-indexed
    reached = hit.any(axis=1)
    dims = np.where(reached, first_idx0 + 1, curves.shape[1])
    return dims.astype(np.int32, copy=False)


def plot_results(
    *,
    eigvals: np.ndarray,
    curves: np.ndarray,
    output_dir: Path,
    title: str,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    centered: bool = False,
    save_subfigures: bool = True,
    font_size: int = 21,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Match the paper plotting style used elsewhere in this repo (see replot_layers_svd*.py).
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

    D = eigvals.shape[0]
    x = np.arange(1, D + 1)
    cum_global = np.cumsum(eigvals) / np.maximum(eigvals.sum(), 1e-12)

    # Per-image percentile curves under the shared basis.
    q50, q10, q01 = np.percentile(curves, [50, 10, 1], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=int(font_size) + 2, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(x, cum_global, linewidth=2.5, label="Global cumulative energy (PCA)")
    for t in thresholds:
        d_t = int(np.searchsorted(cum_global, t) + 1)
        ax.axhline(t, color="gray", linestyle="--", linewidth=1.5)
        ax.axvline(d_t, color="gray", linestyle="--", linewidth=1.5)
        # Place the label *below* the horizontal line to avoid covering the curve.
        y_txt = max(t - 0.06, 0.02)
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
    ax.set_title("Dataset-level PCA spectrum" if not centered else "Dataset-level PCA spectrum (centered)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, D)
    ax.set_ylim(0, 1.0)

    ax = axes[0, 1]
    xd = np.arange(1, curves.shape[1] + 1)
    ax.plot(xd, q50, linewidth=2.5, label="Median over images")
    ax.plot(xd, q10, linewidth=2.5, label="10th percentile over images")
    ax.plot(xd, q01, linewidth=2.5, label="1st percentile over images")
    for t in thresholds:
        ax.axhline(t, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("d (components)")
    ax.set_ylabel("Energy captured")
    ax.set_title("Energy captured by the shared PCA subspace")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, curves.shape[1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")

    # Histograms of required d for selected thresholds under the shared basis.
    for (r, c, t) in [(1, 0, 0.95), (1, 1, 0.99)]:
        ax = axes[r, c]
        dims = dims_for_threshold(curves, t)
        counts, bin_edges = np.histogram(dims, bins=50)
        ratios = counts / max(len(dims), 1)
        ax.bar(
            bin_edges[:-1],
            ratios,
            width=np.diff(bin_edges),
            align="edge",
            color="steelblue" if t == 0.95 else "crimson",
            alpha=0.75,
            edgecolor="black",
        )
        p99 = float(np.percentile(dims, 99))
        ax.axvline(p99, color="black", linestyle="--", linewidth=2.5, label=f"99th pct: {p99:.0f}")
        ax.set_xlabel("Required d")
        ax.set_ylabel("Ratio")
        ax.set_title(f"Required d for {int(t*100)}% energy\n(shared basis)")
        ax.grid(True, alpha=0.3)
        # Avoid overlap between legend and stats box.
        ax.legend(loc="upper left")
        ax.text(
            0.98,
            0.98,
            f"Mean: {dims.mean():.1f}\nStd: {dims.std():.1f}",
            transform=ax.transAxes,
            fontsize=annot_size,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

    plt.tight_layout()
    png_path = output_dir / "dataset_pca_analysis.png"
    pdf_path = output_dir / "dataset_pca_analysis.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plots: {png_path} and {pdf_path}")

    if not save_subfigures:
        return

    # Save each panel as a separate figure to make paper layout easier.
    def save_fig(fig_obj, stem: str) -> None:
        fig_obj.tight_layout()
        fig_obj.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig_obj.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig_obj)

    # (1) Spectrum.
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.plot(x, cum_global, linewidth=2.5, label="Global cumulative energy (PCA)")
    for t in thresholds:
        d_t = int(np.searchsorted(cum_global, t) + 1)
        ax1.axhline(t, color="gray", linestyle="--", linewidth=1.5)
        ax1.axvline(d_t, color="gray", linestyle="--", linewidth=1.5)
        y_txt = max(t - 0.06, 0.02)
        ax1.text(
            d_t,
            y_txt,
            f"d={d_t}",
            fontsize=annot_size,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
    ax1.set_xlabel("d (components)")
    ax1.set_ylabel("Cumulative energy")
    ax1.set_title("PCA spectrum" if not centered else "PCA spectrum (centered)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, D)
    ax1.set_ylim(0, 1.0)
    save_fig(fig1, "dataset_pca_analysis_spectrum")

    # (2) Captured-energy percentiles over images (shared basis).
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    xd = np.arange(1, curves.shape[1] + 1)
    ax2.plot(xd, q50, linewidth=2.5, label="Median over images")
    ax2.plot(xd, q10, linewidth=2.5, label="10th percentile over images")
    ax2.plot(xd, q01, linewidth=2.5, label="1st percentile over images")
    for t in thresholds:
        ax2.axhline(t, color="gray", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("d (components)")
    ax2.set_ylabel("Energy captured")
    ax2.set_title("Energy captured by shared PCA subspace")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, curves.shape[1])
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")
    save_fig(fig2, "dataset_pca_analysis_captured_energy")

    # (3) Required d histogram for 95% and (4) 99%.
    for t in (0.95, 0.99):
        dims = dims_for_threshold(curves, t)
        fig_h, ax_h = plt.subplots(1, 1, figsize=(7, 5))
        counts, bin_edges = np.histogram(dims, bins=50)
        ratios = counts / max(len(dims), 1)
        ax_h.bar(
            bin_edges[:-1],
            ratios,
            width=np.diff(bin_edges),
            align="edge",
            color="steelblue" if t == 0.95 else "crimson",
            alpha=0.75,
            edgecolor="black",
        )
        p99 = float(np.percentile(dims, 99))
        ax_h.axvline(p99, color="black", linestyle="--", linewidth=2.5, label=f"99th pct: {p99:.0f}")
        ax_h.set_xlabel("Required d")
        ax_h.set_ylabel("Ratio")
        ax_h.set_title(f"Required d for {int(t*100)}% energy (shared basis)")
        ax_h.grid(True, alpha=0.3)
        ax_h.legend(loc="upper left")
        ax_h.text(
            0.98,
            0.98,
            f"Mean: {dims.mean():.1f}\nStd: {dims.std():.1f}",
            transform=ax_h.transAxes,
            fontsize=annot_size,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )
        save_fig(fig_h, f"dataset_pca_analysis_required_d_{int(t*100)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-level PCA analysis on last-layer token features (shared subspace).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory containing per-image feature files (.npy/.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: <feature-dir>/../dataset_pca).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Which layer index to use (default: -1 for last layer).",
    )
    parser.add_argument(
        "--token-policy",
        type=str,
        default="auto",
        choices=["auto", "keep", "drop"],
        help="Whether to drop token 0 as CLS.",
    )
    parser.add_argument(
        "--batch-images",
        type=int,
        default=64,
        help="How many images to concatenate per matmul batch (speed/memory trade-off).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of feature files used (for quick runs).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Use centered covariance (variance) instead of uncentered energy (default off).",
    )
    parser.add_argument(
        "--max-d",
        type=int,
        default=None,
        help="Compute per-image curves only up to max-d components (default: full D).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="Computation backend. auto=use torch if available, otherwise numpy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use when backend=torch (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=21,
        help="Base font size for plots (use a larger value for paper subfigures).",
    )
    args = parser.parse_args()

    feature_dir: Path = args.feature_dir
    if not feature_dir.exists():
        raise SystemExit(f"Feature directory not found: {feature_dir}")

    feature_files = list_feature_files(feature_dir)
    if args.max_files is not None:
        feature_files = feature_files[: args.max_files]
    if not feature_files:
        raise SystemExit(f"No feature files found in {feature_dir} (expected *.npy or *.npz).")

    out_dir = args.output_dir
    if out_dir is None:
        out_name = "dataset_pca_centered" if args.center else "dataset_pca"
        # Heuristic: if the directory structure looks like .../<model_run>/features/<model_name>/,
        # save alongside other analyses in .../<model_run>/dataset_pca/.
        if feature_dir.parent.name == "features":
            out_dir = feature_dir.parent.parent / out_name
        else:
            out_dir = feature_dir.parent / out_name

    print(f"Using {len(feature_files)} files from: {feature_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Layer: {args.layer} (last layer if -1)")
    print(f"Token policy: {args.token_policy}")
    print(f"Centering: {args.center}")

    backend = args.backend
    if backend == "auto":
        try:
            import torch

            wants_cuda = str(args.device).startswith("cuda")
            backend = "torch" if (not wants_cuda or torch.cuda.is_available()) else "numpy"
        except Exception:
            backend = "numpy"

    print(f"Backend: {backend}" + (f" (device={args.device})" if backend == "torch" else ""))

    C, mu, total_tokens = compute_second_moment(
        feature_files,
        layer=args.layer,
        token_policy=args.token_policy,
        batch_images=args.batch_images,
        center=args.center,
        backend=backend,
        device=args.device,
    )
    eigvals, V = eigh_descending(C)

    # Per-image curves under the shared PCA basis.
    curves = per_image_energy_curves(
        feature_files,
        V,
        layer=args.layer,
        token_policy=args.token_policy,
        batch_images=args.batch_images,
        max_d=args.max_d,
        backend=backend,
        device=args.device,
        mean_token=mu,
    )

    # Print key numbers for paper-writing / sanity checks.
    cum_global = np.cumsum(eigvals) / np.maximum(eigvals.sum(), 1e-12)
    for t in (0.80, 0.90, 0.95, 0.99):
        d_t = int(np.searchsorted(cum_global, t) + 1)
        dims_t = dims_for_threshold(curves, t)
        print(
            f"[t={t:.2f}] global d={d_t}, shared-basis required d over images: "
            f"mean={dims_t.mean():.1f}, std={dims_t.std():.1f}, 99th pct={np.percentile(dims_t, 99):.0f}"
        )

    # Save numeric results.
    out_dir.mkdir(parents=True, exist_ok=True)
    save_payload = dict(
        eigvals=eigvals.astype(np.float32, copy=False),
        eigvecs=V.astype(np.float32, copy=False),
        curves=curves.astype(np.float32, copy=False),
        total_tokens=np.int64(total_tokens),
        feature_dir=str(feature_dir),
        layer=np.int32(args.layer),
        token_policy=str(args.token_policy),
        centered=np.bool_(args.center),
    )
    if mu is not None:
        save_payload["mean_token"] = mu.astype(np.float32, copy=False)
    np.savez(out_dir / "dataset_pca_results.npz", **save_payload)
    print(f"Saved results: {out_dir / 'dataset_pca_results.npz'}")

    plot_results(
        eigvals=eigvals,
        curves=curves,
        output_dir=out_dir,
        title=f"Dataset-level PCA (layer={args.layer}, token_policy={args.token_policy})",
        centered=bool(args.center),
        font_size=int(args.font_size),
    )


if __name__ == "__main__":
    main()
