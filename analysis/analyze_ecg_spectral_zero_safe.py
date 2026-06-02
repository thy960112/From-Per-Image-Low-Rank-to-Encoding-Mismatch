#!/usr/bin/env python3
"""
Spectral Energy Concentration Gap (ECG) Analysis with zero-energy-token handling.

This variant matches `analyze_ecg_spectral.py` except that it skips tokens whose
total spectral energy is exactly zero. This is useful for CNN last-stage feature
maps where some spatial locations can be identically zero after ReLU.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
    })


def find_threshold_dimension(ecg_curve, threshold_percent):
    """
    Find the dimension index where ECG reaches the threshold.

    Args:
        ecg_curve: numpy array of shape (D,)
        threshold_percent: target percentage (e.g., 50.0 for 50%)

    Returns:
        dimension_index: first index where ECG >= threshold_percent
    """
    indices = np.where(ecg_curve >= threshold_percent)[0]
    if len(indices) == 0:
        return len(ecg_curve) - 1
    return indices[0]


def load_last_layer_features(feature_dir):
    """
    Load last-layer token matrices from all feature files.

    Returns the list of files and the inferred channel dimension.
    """
    feature_path = Path(feature_dir)
    feature_files = sorted(list(feature_path.glob('*.npy')))

    if len(feature_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {feature_dir}")

    first = np.load(feature_files[0], allow_pickle=True)
    model_dim = int(first[-1].shape[1])

    print(f"Found {len(feature_files)} feature files")
    print(f"Inferred model dimension (D): {model_dim}")
    return feature_files, model_dim


def _ecg_curves_for_tokens(tokens_nc: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Compute ECG curves for a token matrix of shape [N, D] (frequency order preserved).

    Tokens with zero total spectral energy are removed before normalization.
    """
    if use_gpu and CUPY_AVAILABLE:
        x = cp.asarray(tokens_nc)
        fft_coeffs = cp.fft.fft(x, axis=1)
        energies = cp.abs(fft_coeffs) ** 2
        totals = cp.sum(energies, axis=1, keepdims=True)
        valid = cp.squeeze(totals, axis=1) > 0
        if not bool(cp.any(valid)):
            return np.empty((0, int(tokens_nc.shape[1])), dtype=np.float64)
        energies = energies[valid]
        totals = totals[valid]
        cumulative = cp.cumsum(energies, axis=1)
        ecg = (cumulative / totals) * 100.0
        return cp.asnumpy(ecg)

    fft_coeffs = np.fft.fft(tokens_nc, axis=1)
    energies = np.abs(fft_coeffs) ** 2
    totals = np.sum(energies, axis=1, keepdims=True)
    valid = np.squeeze(totals, axis=1) > 0
    if not np.any(valid):
        return np.empty((0, int(tokens_nc.shape[1])), dtype=np.float64)
    energies = energies[valid]
    totals = totals[valid]
    cumulative = np.cumsum(energies, axis=1)
    ecg = (cumulative / totals) * 100.0
    return ecg


def analyze_ecg(feature_dir, output_dir, model_name="Model", use_gpu=False):
    """Main ECG analysis function."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Spectral Energy Concentration Gap Analysis (Zero-Safe)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GPU acceleration: {use_gpu and CUPY_AVAILABLE}")

    print("\n[1/4] Loading last layer features...")
    feature_files, model_dim = load_last_layer_features(feature_dir)

    print("\n[2/4] Computing spectral energy for all tokens...")
    sum_ecg = np.zeros((model_dim,), dtype=np.float64)
    sumsq_ecg = np.zeros((model_dim,), dtype=np.float64)
    num_tokens = 0
    num_zero_energy_tokens = 0

    for file_path in tqdm(feature_files, desc="Processing feature files"):
        features = np.load(file_path, allow_pickle=True)
        tokens = features[-1]

        ecg_curves = _ecg_curves_for_tokens(tokens, use_gpu=use_gpu)
        num_zero_energy_tokens += int(tokens.shape[0] - ecg_curves.shape[0])
        if ecg_curves.shape[0] == 0:
            continue

        sum_ecg += ecg_curves.sum(axis=0)
        sumsq_ecg += (ecg_curves ** 2).sum(axis=0)
        num_tokens += int(ecg_curves.shape[0])

    if num_tokens == 0:
        raise ValueError("All tokens had zero spectral energy; ECG is undefined for this feature set.")

    print("\n[3/4] Computing average ECG curve...")
    mean_ecg_curve = sum_ecg / num_tokens
    var = (sumsq_ecg / num_tokens) - (mean_ecg_curve ** 2)
    var = np.maximum(var, 0.0)
    std_ecg_curve = np.sqrt(var)

    thresholds = [50, 60, 70, 80, 90]
    threshold_dims = {}

    for thresh in thresholds:
        dim_idx = find_threshold_dimension(mean_ecg_curve, thresh)
        threshold_dims[thresh] = {
            'dimension': dim_idx + 1,
            'normalized': (dim_idx + 1) / model_dim,
        }

    print("\n[4/4] Results:")
    print(f"{'='*60}")
    print(f"Model Dimension (D): {model_dim}")
    print(f"Number of tokens analyzed: {num_tokens}")
    print(f"Skipped zero-energy tokens: {num_zero_energy_tokens}")
    print(f"\nThreshold Dimensions:")
    print(f"{'Threshold':<12} {'Dimension':<12} {'Normalized (d/D)':<20}")
    print(f"{'-'*44}")
    for thresh in thresholds:
        dim = threshold_dims[thresh]['dimension']
        norm = threshold_dims[thresh]['normalized']
        print(f"{thresh}%{' ':<9} {dim:<12} {norm:.4f}")

    results = {
        'model_name': model_name,
        'model_dim': model_dim,
        'num_tokens': num_tokens,
        'num_zero_energy_tokens': num_zero_energy_tokens,
        'mean_ecg_curve': mean_ecg_curve,
        'std_ecg_curve': std_ecg_curve,
        'threshold_dims': threshold_dims,
        'thresholds': thresholds,
    }

    results_file = output_path / 'ecg_results.npz'
    np.savez(results_file, **results)
    print(f"\nSaved results to {results_file}")

    print("\nGenerating plots...")
    plot_ecg_curve(mean_ecg_curve, std_ecg_curve, model_dim, model_name,
                   threshold_dims, num_tokens, output_path)
    save_threshold_table(threshold_dims, model_dim, model_name, output_path)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
    return results


def plot_ecg_curve(mean_ecg, std_ecg, model_dim, model_name, threshold_dims, num_tokens, output_dir):
    """Plot the Energy Concentration Gap curve."""
    setup_plot_style()

    dimensions = np.arange(1, model_dim + 1)
    normalized_dims = dimensions / model_dim
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(normalized_dims, mean_ecg, 'b-', linewidth=2.5, label='Mean ECG')
    ax.fill_between(
        normalized_dims,
        mean_ecg - std_ecg,
        mean_ecg + std_ecg,
        alpha=0.2,
        color='blue',
        label='±1 std',
    )

    colors = ['green', 'orange', 'red', 'purple', 'brown']
    for thresh, color in zip([50, 60, 70, 80, 90], colors):
        norm_dim = threshold_dims[thresh]['normalized']
        ax.axhline(y=thresh, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.axvline(x=norm_dim, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(0.02, thresh + 1, f'{thresh}%', fontsize=10, color=color,
                verticalalignment='bottom', fontweight='bold')

    ax.set_xlabel('Normalized Frequency (f/D)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Spectral Energy (%)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Cumulative Spectral Energy Pattern - {model_name}\n'
        f'(Averaged over {num_tokens:,} tokens, frequency order preserved)',
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

    textstr = f'Model Dimension: {model_dim}\n'
    textstr += f'd/D at 80%: {threshold_dims[80]["normalized"]:.3f}\n'
    textstr += f'd/D at 90%: {threshold_dims[90]["normalized"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.25, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'ecg_curve.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
    plt.close()


def save_threshold_table(threshold_dims, model_dim, model_name, output_dir):
    """Save threshold dimensions table to CSV."""
    import csv

    csv_file = output_dir / 'ecg_thresholds.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Model_Dimension', 'Threshold (%)', 'Dimension (d)', 'Normalized (d/D)'])
        for thresh in sorted(threshold_dims.keys()):
            dim = threshold_dims[thresh]['dimension']
            norm = threshold_dims[thresh]['normalized']
            writer.writerow([model_name, model_dim, thresh, dim, f'{norm:.6f}'])

    print(f"Saved threshold table: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Spectral Energy Concentration Gap Analysis with zero-energy-token handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Path to directory containing extracted features')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory for results and plots')
    parser.add_argument('--model-name', type=str, default='Model',
                        help='Name of the model for plot titles (default: Model)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration with CuPy if available')

    args = parser.parse_args()

    if args.use_gpu and not CUPY_AVAILABLE:
        print("Warning: --use-gpu specified but CuPy is not available.")
        print("Falling back to CPU (NumPy) computation.")
        print("To enable GPU: pip install cupy-cuda11x  # or cupy-cuda12x\n")

    analyze_ecg(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_gpu=args.use_gpu,
    )


if __name__ == '__main__':
    main()
