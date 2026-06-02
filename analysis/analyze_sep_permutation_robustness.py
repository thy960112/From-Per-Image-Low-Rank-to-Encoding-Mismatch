#!/usr/bin/env python3
"""
Permutation robustness analysis for Spectral Energy Patterns (SEP).

This script reuses saved last-layer token features and measures how the SEP
curve changes under global random permutations of the channel dimension.

By default it matches the paper artifacts currently stored in this repository:
the full FFT ordering is preserved and normalized bandwidth is reported as d/D.
An optional folded mode is also provided for experiments based on unique
frequency bins after conjugate folding.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def setup_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    })


def extract_last_tokens(feature_array: np.ndarray) -> np.ndarray:
    """Return the last-layer or last-stage token matrix with shape [N, D]."""
    tokens = feature_array[-1]
    if tokens.ndim != 2:
        raise ValueError(f"Expected last-layer tokens with shape [N, D], got {tokens.shape}")
    return np.asarray(tokens, dtype=np.float32)


def discover_feature_files(feature_dir: str, max_files: int | None = None) -> list[Path]:
    feature_path = Path(feature_dir)
    feature_files = sorted(feature_path.glob("*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No .npy files found in {feature_dir}")
    if max_files is not None:
        feature_files = feature_files[:max_files]
    return feature_files


def load_feature_tensor(feature_dir: str, max_files: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Load all last-layer token matrices into a single tensor [M, N, D]."""
    feature_files = discover_feature_files(feature_dir, max_files=max_files)

    first = np.load(feature_files[0], allow_pickle=True)
    first_tokens = extract_last_tokens(first)
    num_images = len(feature_files)
    num_tokens, channel_dim = first_tokens.shape

    all_tokens = np.empty((num_images, num_tokens, channel_dim), dtype=np.float32)
    file_names: list[str] = []

    for idx, file_path in enumerate(tqdm(feature_files, desc="Loading features")):
        tokens = extract_last_tokens(np.load(file_path, allow_pickle=True))
        if tokens.shape != (num_tokens, channel_dim):
            raise ValueError(
                "All feature files must share the same last-layer token shape. "
                f"Expected {(num_tokens, channel_dim)}, got {tokens.shape} for {file_path}"
            )
        all_tokens[idx] = tokens
        file_names.append(file_path.name)

    return all_tokens, file_names


def folded_energy_from_rfft(fft_coeffs, signal_dim: int):
    """Fold conjugate pairs into the unique bins represented by rFFT."""
    energies = np.abs(fft_coeffs) ** 2 if isinstance(fft_coeffs, np.ndarray) else cp.abs(fft_coeffs) ** 2

    if signal_dim <= 1:
        return energies

    if signal_dim % 2 == 0:
        if energies.shape[-1] > 2:
            energies[..., 1:-1] *= 2.0
    else:
        if energies.shape[-1] > 1:
            energies[..., 1:] *= 2.0
    return energies


def compute_sep_image_curves(
    tokens_mnd: np.ndarray,
    batch_size: int,
    spectrum_mode: str,
    permutation: np.ndarray | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute image-level SEP curves with shape [M, L].

    Each image curve is the mean token-level cumulative spectral energy across
    the last-layer tokens of that image.
    """
    num_images, _, channel_dim = tokens_mnd.shape
    curve_dim = channel_dim if spectrum_mode == "full" else (channel_dim // 2 + 1)
    image_curves = np.empty((num_images, curve_dim), dtype=np.float64)

    perm_gpu = cp.asarray(permutation) if (use_gpu and CUPY_AVAILABLE and permutation is not None) else None

    for start in tqdm(range(0, num_images, batch_size), desc="Computing SEP curves", leave=False):
        end = min(start + batch_size, num_images)
        batch = tokens_mnd[start:end]

        if use_gpu and CUPY_AVAILABLE:
            x = cp.asarray(batch)
            if perm_gpu is not None:
                x = x[..., perm_gpu]

            if spectrum_mode == "full":
                coeffs = cp.fft.fft(x, axis=-1)
                energies = cp.abs(coeffs) ** 2
            else:
                coeffs = cp.fft.rfft(x, axis=-1)
                energies = folded_energy_from_rfft(coeffs, signal_dim=channel_dim)

            totals = cp.sum(energies, axis=-1, keepdims=True)
            if bool(cp.any(totals <= 0)):
                raise ValueError("Encountered tokens with zero total spectral energy.")
            curves = cp.cumsum(energies, axis=-1) / totals * 100.0
            image_curves[start:end] = cp.asnumpy(curves.mean(axis=1))
            continue

        x = np.take(batch, permutation, axis=-1) if permutation is not None else batch
        if spectrum_mode == "full":
            coeffs = np.fft.fft(x, axis=-1)
            energies = np.abs(coeffs) ** 2
        else:
            coeffs = np.fft.rfft(x, axis=-1)
            energies = folded_energy_from_rfft(coeffs, signal_dim=channel_dim)

        totals = np.sum(energies, axis=-1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Encountered tokens with zero total spectral energy.")
        curves = np.cumsum(energies, axis=-1) / totals * 100.0
        image_curves[start:end] = curves.mean(axis=1)

    return image_curves


def compute_bandwidths(
    image_curves: np.ndarray,
    thresholds: np.ndarray,
    normalization_denominator: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first crossing dimensions and normalized bandwidths for each image."""
    hits = image_curves[:, None, :] >= thresholds[None, :, None]
    if not np.all(hits.any(axis=-1)):
        raise ValueError("At least one threshold was not reached by some SEP curves.")
    dims = hits.argmax(axis=-1) + 1
    bandwidths = dims / normalization_denominator
    return dims.astype(np.int32), bandwidths.astype(np.float64)


def bootstrap_expected_delta(
    orig_bandwidths: np.ndarray,
    perm_bandwidths: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray | None:
    """
    Paired bootstrap over images for the expected random-permutation effect.

    Returns an array of shape [A, 2] with lower/upper 95% confidence bounds.
    """
    if bootstrap_samples <= 0:
        return None

    num_images = orig_bandwidths.shape[0]
    rng = np.random.default_rng(seed)
    expected_perm = perm_bandwidths.mean(axis=0)  # [M, A]
    sample_indices = rng.integers(0, num_images, size=(bootstrap_samples, num_images))

    orig_sample_means = orig_bandwidths[sample_indices].mean(axis=1)
    perm_sample_means = expected_perm[sample_indices].mean(axis=1)
    delta_samples = perm_sample_means - orig_sample_means

    lower = np.percentile(delta_samples, 2.5, axis=0)
    upper = np.percentile(delta_samples, 97.5, axis=0)
    return np.stack([lower, upper], axis=1)


def plot_curve_envelope(
    output_dir: Path,
    model_name: str,
    x_values: np.ndarray,
    x_label: str,
    orig_mean_curve: np.ndarray,
    orig_std_curve: np.ndarray,
    perm_mean_curves: np.ndarray,
):
    setup_plot_style()

    perm_mean = perm_mean_curves.mean(axis=0)
    perm_p05 = np.percentile(perm_mean_curves, 5, axis=0)
    perm_p95 = np.percentile(perm_mean_curves, 95, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(x_values, orig_mean_curve, color="tab:blue", label="Original mean SEP", linewidth=2.5)
    ax.fill_between(
        x_values,
        orig_mean_curve - orig_std_curve,
        orig_mean_curve + orig_std_curve,
        color="tab:blue",
        alpha=0.18,
        label="Original ±1 std over images",
    )

    ax.plot(
        x_values,
        perm_mean,
        color="tab:orange",
        linestyle="--",
        linewidth=2.5,
        label="Permutation mean SEP",
    )
    ax.fill_between(
        x_values,
        perm_p05,
        perm_p95,
        color="tab:orange",
        alpha=0.22,
        label="Permutation 5-95 percentile band",
    )

    ax.set_xlim([x_values[0], x_values[-1]])
    ax.set_ylim([0, 105])
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Spectral Energy (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"SEP Permutation Robustness - {model_name}", fontsize=16, fontweight="bold", pad=18)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"sep_permutation_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_bandwidth_summary(
    output_dir: Path,
    model_name: str,
    thresholds: np.ndarray,
    orig_bandwidths: np.ndarray,
    perm_bandwidth_means: np.ndarray,
):
    setup_plot_style()

    orig_mean = orig_bandwidths.mean(axis=0)
    orig_std = orig_bandwidths.std(axis=0)
    perm_mean = perm_bandwidth_means.mean(axis=0)
    perm_std = perm_bandwidth_means.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    x_orig = thresholds - 0.6
    x_perm = thresholds + 0.6

    ax.errorbar(x_orig, orig_mean, yerr=orig_std, fmt="o-", color="tab:blue", capsize=4, label="Original")
    ax.errorbar(
        x_perm,
        perm_mean,
        yerr=perm_std,
        fmt="s--",
        color="tab:orange",
        capsize=4,
        label="Permutation ensemble",
    )

    ax.set_xlabel("Energy Threshold (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Bandwidth", fontsize=14, fontweight="bold")
    ax.set_title(f"Bandwidth Shifts Under Channel Permutations - {model_name}", fontsize=16, fontweight="bold", pad=18)
    ax.set_xticks(thresholds)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"sep_permutation_bandwidths.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def write_bandwidth_summary_csv(
    output_dir: Path,
    thresholds: np.ndarray,
    orig_dims: np.ndarray,
    orig_bandwidths: np.ndarray,
    perm_bandwidths: np.ndarray,
    bootstrap_ci: np.ndarray | None,
):
    perm_bandwidth_means = perm_bandwidths.mean(axis=1)
    delta_means = perm_bandwidth_means - orig_bandwidths.mean(axis=0, keepdims=True)

    csv_path = output_dir / "sep_permutation_bandwidth_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "threshold_percent",
            "orig_dim_mean",
            "orig_dim_std",
            "orig_bandwidth_mean",
            "orig_bandwidth_std",
            "perm_bandwidth_mean",
            "perm_bandwidth_std_over_permutations",
            "delta_mean",
            "delta_std_over_permutations",
            "delta_p05",
            "delta_p95",
            "bootstrap_delta_ci_low",
            "bootstrap_delta_ci_high",
        ])

        for idx, threshold in enumerate(thresholds):
            ci_low = ""
            ci_high = ""
            if bootstrap_ci is not None:
                ci_low = f"{bootstrap_ci[idx, 0]:.6f}"
                ci_high = f"{bootstrap_ci[idx, 1]:.6f}"

            writer.writerow([
                int(threshold),
                f"{orig_dims[:, idx].mean():.3f}",
                f"{orig_dims[:, idx].std():.3f}",
                f"{orig_bandwidths[:, idx].mean():.6f}",
                f"{orig_bandwidths[:, idx].std():.6f}",
                f"{perm_bandwidth_means[:, idx].mean():.6f}",
                f"{perm_bandwidth_means[:, idx].std():.6f}",
                f"{delta_means[:, idx].mean():.6f}",
                f"{delta_means[:, idx].std():.6f}",
                f"{np.percentile(delta_means[:, idx], 5):.6f}",
                f"{np.percentile(delta_means[:, idx], 95):.6f}",
                ci_low,
                ci_high,
            ])


def write_permutation_summary_csv(
    output_dir: Path,
    thresholds: np.ndarray,
    perm_bandwidths: np.ndarray,
    perm_curve_l1: np.ndarray,
):
    perm_bandwidth_means = perm_bandwidths.mean(axis=1)

    csv_path = output_dir / "sep_permutation_trials.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["permutation_id", "curve_l1_fraction"]
        header.extend([f"b{int(threshold)}_mean" for threshold in thresholds])
        writer.writerow(header)

        for idx in range(perm_bandwidth_means.shape[0]):
            row = [idx, f"{perm_curve_l1[idx]:.8f}"]
            row.extend(f"{value:.6f}" for value in perm_bandwidth_means[idx])
            writer.writerow(row)


def write_text_summary(
    output_dir: Path,
    model_name: str,
    spectrum_mode: str,
    channel_dim: int,
    curve_dim: int,
    num_images: int,
    num_tokens: int,
    thresholds: np.ndarray,
    orig_bandwidths: np.ndarray,
    perm_bandwidths: np.ndarray,
    perm_curve_l1: np.ndarray,
    bootstrap_ci: np.ndarray | None,
    elapsed_seconds: float,
):
    perm_bandwidth_means = perm_bandwidths.mean(axis=1)
    delta_means = perm_bandwidth_means - orig_bandwidths.mean(axis=0, keepdims=True)

    summary_path = output_dir / "sep_permutation_summary.txt"
    with summary_path.open("w") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Spectrum mode: {spectrum_mode}\n")
        handle.write(f"Channel dimension D: {channel_dim}\n")
        handle.write(f"Curve dimension L: {curve_dim}\n")
        handle.write(f"Images: {num_images}\n")
        handle.write(f"Tokens per image: {num_tokens}\n")
        handle.write(f"Permutations: {perm_bandwidths.shape[0]}\n")
        handle.write(f"Elapsed seconds: {elapsed_seconds:.2f}\n")
        handle.write("\n")
        handle.write(
            "Curve L1 fraction between permutation mean curve and original mean curve:\n"
            f"  mean={perm_curve_l1.mean():.6f}, std={perm_curve_l1.std():.6f}, "
            f"p05={np.percentile(perm_curve_l1, 5):.6f}, p95={np.percentile(perm_curve_l1, 95):.6f}\n"
        )
        handle.write("\n")
        handle.write("Bandwidth summary:\n")

        for idx, threshold in enumerate(thresholds):
            line = (
                f"  b{int(threshold)}: "
                f"orig={orig_bandwidths[:, idx].mean():.6f} +/- {orig_bandwidths[:, idx].std():.6f}, "
                f"perm={perm_bandwidth_means[:, idx].mean():.6f} +/- {perm_bandwidth_means[:, idx].std():.6f}, "
                f"delta={delta_means[:, idx].mean():.6f} +/- {delta_means[:, idx].std():.6f}, "
                f"delta_p05={np.percentile(delta_means[:, idx], 5):.6f}, "
                f"delta_p95={np.percentile(delta_means[:, idx], 95):.6f}"
            )
            if bootstrap_ci is not None:
                line += (
                    f", bootstrap95=[{bootstrap_ci[idx, 0]:.6f}, "
                    f"{bootstrap_ci[idx, 1]:.6f}]"
                )
            handle.write(line + "\n")


def save_results_npz(
    output_dir: Path,
    model_name: str,
    spectrum_mode: str,
    file_names: list[str],
    channel_dim: int,
    curve_dim: int,
    thresholds: np.ndarray,
    orig_image_curves: np.ndarray,
    orig_dims: np.ndarray,
    orig_bandwidths: np.ndarray,
    perm_mean_curves: np.ndarray,
    perm_std_curves: np.ndarray,
    perm_bandwidths: np.ndarray,
    perm_curve_l1: np.ndarray,
    bootstrap_ci: np.ndarray | None,
):
    np.savez(
        output_dir / "sep_permutation_results.npz",
        model_name=np.array(model_name),
        spectrum_mode=np.array(spectrum_mode),
        file_names=np.array(file_names, dtype=object),
        channel_dim=np.array(channel_dim),
        curve_dim=np.array(curve_dim),
        thresholds=thresholds,
        orig_image_curves=orig_image_curves,
        orig_mean_curve=orig_image_curves.mean(axis=0),
        orig_std_curve=orig_image_curves.std(axis=0),
        orig_dims=orig_dims,
        orig_bandwidths=orig_bandwidths,
        perm_mean_curves=perm_mean_curves,
        perm_std_curves=perm_std_curves,
        perm_bandwidths=perm_bandwidths,
        perm_curve_l1=perm_curve_l1,
        bootstrap_expected_delta_ci=np.array([]) if bootstrap_ci is None else bootstrap_ci,
    )


def analyze_sep_permutation_robustness(
    feature_dir: str,
    output_dir: str,
    model_name: str,
    thresholds: list[float],
    num_permutations: int,
    spectrum_mode: str,
    batch_size: int,
    seed: int,
    max_files: int | None,
    bootstrap_samples: int,
    use_gpu: bool,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SEP Permutation Robustness Analysis")
    print("=" * 72)
    print(f"Model: {model_name}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Spectrum mode: {spectrum_mode}")
    print(f"Permutations: {num_permutations}")
    print(f"Batch size: {batch_size}")
    print(f"GPU acceleration: {use_gpu and CUPY_AVAILABLE}")
    print(f"Bootstrap samples: {bootstrap_samples}")

    start_time = time.time()

    tokens_mnd, file_names = load_feature_tensor(feature_dir, max_files=max_files)
    num_images, num_tokens, channel_dim = tokens_mnd.shape
    curve_dim = channel_dim if spectrum_mode == "full" else (channel_dim // 2 + 1)
    norm_denominator = channel_dim if spectrum_mode == "full" else curve_dim
    x_label = "Normalized Spectral Bandwidth (d/D)" if spectrum_mode == "full" else "Normalized Spectral Bandwidth (d/D')"
    x_values = np.arange(1, curve_dim + 1, dtype=np.float64) / norm_denominator
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)

    print("\n[1/4] Computing original SEP curves...")
    orig_image_curves = compute_sep_image_curves(
        tokens_mnd=tokens_mnd,
        batch_size=batch_size,
        spectrum_mode=spectrum_mode,
        permutation=None,
        use_gpu=use_gpu,
    )
    orig_dims, orig_bandwidths = compute_bandwidths(
        image_curves=orig_image_curves,
        thresholds=thresholds_arr,
        normalization_denominator=norm_denominator,
    )
    orig_mean_curve = orig_image_curves.mean(axis=0)
    orig_std_curve = orig_image_curves.std(axis=0)

    print("\n[2/4] Running random channel permutations...")
    perm_mean_curves = np.empty((num_permutations, curve_dim), dtype=np.float64)
    perm_std_curves = np.empty((num_permutations, curve_dim), dtype=np.float64)
    perm_bandwidths = np.empty((num_permutations, num_images, len(thresholds_arr)), dtype=np.float64)
    perm_curve_l1 = np.empty((num_permutations,), dtype=np.float64)

    rng = np.random.default_rng(seed)
    for perm_idx in tqdm(range(num_permutations), desc="Permutation trials"):
        permutation = rng.permutation(channel_dim)
        perm_image_curves = compute_sep_image_curves(
            tokens_mnd=tokens_mnd,
            batch_size=batch_size,
            spectrum_mode=spectrum_mode,
            permutation=permutation,
            use_gpu=use_gpu,
        )
        _, perm_bandwidth = compute_bandwidths(
            image_curves=perm_image_curves,
            thresholds=thresholds_arr,
            normalization_denominator=norm_denominator,
        )
        perm_mean_curves[perm_idx] = perm_image_curves.mean(axis=0)
        perm_std_curves[perm_idx] = perm_image_curves.std(axis=0)
        perm_bandwidths[perm_idx] = perm_bandwidth
        perm_curve_l1[perm_idx] = np.mean(np.abs(perm_mean_curves[perm_idx] - orig_mean_curve)) / 100.0

    print("\n[3/4] Computing summaries and plots...")
    bootstrap_ci = bootstrap_expected_delta(
        orig_bandwidths=orig_bandwidths,
        perm_bandwidths=perm_bandwidths,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
    )

    plot_curve_envelope(
        output_dir=output_path,
        model_name=model_name,
        x_values=x_values,
        x_label=x_label,
        orig_mean_curve=orig_mean_curve,
        orig_std_curve=orig_std_curve,
        perm_mean_curves=perm_mean_curves,
    )

    plot_bandwidth_summary(
        output_dir=output_path,
        model_name=model_name,
        thresholds=thresholds_arr,
        orig_bandwidths=orig_bandwidths,
        perm_bandwidth_means=perm_bandwidths.mean(axis=1),
    )

    elapsed_seconds = time.time() - start_time

    print("\n[4/4] Saving results...")
    save_results_npz(
        output_dir=output_path,
        model_name=model_name,
        spectrum_mode=spectrum_mode,
        file_names=file_names,
        channel_dim=channel_dim,
        curve_dim=curve_dim,
        thresholds=thresholds_arr,
        orig_image_curves=orig_image_curves,
        orig_dims=orig_dims,
        orig_bandwidths=orig_bandwidths,
        perm_mean_curves=perm_mean_curves,
        perm_std_curves=perm_std_curves,
        perm_bandwidths=perm_bandwidths,
        perm_curve_l1=perm_curve_l1,
        bootstrap_ci=bootstrap_ci,
    )
    write_bandwidth_summary_csv(
        output_dir=output_path,
        thresholds=thresholds_arr,
        orig_dims=orig_dims,
        orig_bandwidths=orig_bandwidths,
        perm_bandwidths=perm_bandwidths,
        bootstrap_ci=bootstrap_ci,
    )
    write_permutation_summary_csv(
        output_dir=output_path,
        thresholds=thresholds_arr,
        perm_bandwidths=perm_bandwidths,
        perm_curve_l1=perm_curve_l1,
    )
    write_text_summary(
        output_dir=output_path,
        model_name=model_name,
        spectrum_mode=spectrum_mode,
        channel_dim=channel_dim,
        curve_dim=curve_dim,
        num_images=num_images,
        num_tokens=num_tokens,
        thresholds=thresholds_arr,
        orig_bandwidths=orig_bandwidths,
        perm_bandwidths=perm_bandwidths,
        perm_curve_l1=perm_curve_l1,
        bootstrap_ci=bootstrap_ci,
        elapsed_seconds=elapsed_seconds,
    )

    print("\nSummary:")
    print(f"  Curve L1 fraction mean: {perm_curve_l1.mean():.6f}")
    print(f"  Curve L1 fraction std : {perm_curve_l1.std():.6f}")
    perm_bandwidth_means = perm_bandwidths.mean(axis=1)
    for idx, threshold in enumerate(thresholds_arr):
        print(
            f"  b{int(threshold)}: "
            f"orig={orig_bandwidths[:, idx].mean():.6f}, "
            f"perm={perm_bandwidth_means[:, idx].mean():.6f}, "
            f"delta={perm_bandwidth_means[:, idx].mean() - orig_bandwidths[:, idx].mean():.6f}"
        )

    print(f"\nElapsed time: {elapsed_seconds:.2f} seconds")
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SEP robustness analysis under random channel permutations."
    )
    parser.add_argument("--feature-dir", type=str, required=True, help="Directory containing saved feature .npy files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save robustness artifacts")
    parser.add_argument("--model-name", type=str, required=True, help="Display name for plots and tables")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[50, 60, 70, 80, 90],
        help="Energy thresholds in percent for b_alpha summaries",
    )
    parser.add_argument("--num-permutations", type=int, default=50, help="Number of random global channel permutations")
    parser.add_argument(
        "--spectrum-mode",
        type=str,
        choices=("full", "folded"),
        default="full",
        help="Use the full FFT ordering (paper-matching) or folded unique-frequency bins",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of images per FFT batch")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for channel permutations")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick debugging")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Paired bootstrap samples over images for the expected permutation effect; set 0 to disable",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use CuPy for FFT batches if available")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_gpu and not CUPY_AVAILABLE:
        print("Warning: --use-gpu specified but CuPy is not available. Falling back to CPU.")

    analyze_sep_permutation_robustness(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        thresholds=args.thresholds,
        num_permutations=args.num_permutations,
        spectrum_mode=args.spectrum_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        max_files=args.max_files,
        bootstrap_samples=args.bootstrap_samples,
        use_gpu=args.use_gpu and CUPY_AVAILABLE,
    )


if __name__ == "__main__":
    main()
