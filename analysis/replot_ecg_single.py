#!/usr/bin/env python3
"""
Replot ECG curve for a single model with cleaner formatting.

Usage:
    python replot_ecg_single.py --results-file Output/vit_huge_patch14_224_mae/ecg/ecg_results.npz
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


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


def plot_ecg_curve_clean(results_file, output_dir=None):
    """
    Plot ECG curve with clean formatting and no overlapping labels.

    Args:
        results_file: path to ecg_results.npz
        output_dir: optional output directory (defaults to same dir as results_file)
    """
    # Load results
    data = np.load(results_file, allow_pickle=True)

    model_name = str(data['model_name'])
    model_dim = int(data['model_dim'])
    num_tokens = int(data['num_tokens'])
    mean_ecg = data['mean_ecg_curve']
    std_ecg = data['std_ecg_curve']
    threshold_dims = data['threshold_dims'].item()

    # Set output directory
    if output_dir is None:
        output_dir = Path(results_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup plotting
    setup_plot_style()

    # Create normalized dimension axis
    dimensions = np.arange(1, model_dim + 1)
    normalized_dims = dimensions / model_dim

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot mean ECG curve
    ax.plot(normalized_dims, mean_ecg, 'b-', linewidth=2.5, label='Mean ECG', zorder=10)

    # Plot standard deviation as shaded region
    ax.fill_between(normalized_dims,
                     mean_ecg - std_ecg,
                     mean_ecg + std_ecg,
                     alpha=0.2, color='blue', label='±1 std', zorder=5)

    # Add threshold lines and labels
    thresholds = [50, 60, 70, 80, 90]
    colors = ['green', 'orange', 'red', 'purple', 'brown']

    for thresh, color in zip(thresholds, colors):
        norm_dim = threshold_dims[thresh]['normalized']

        # Horizontal line at threshold
        ax.axhline(y=thresh, color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

        # Vertical line at corresponding d/D
        ax.axvline(x=norm_dim, color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

        # Label on the LEFT side of the plot (y-axis area)
        ax.text(-0.02, thresh, f'{thresh}%',
                fontsize=11, color=color,
                verticalalignment='center',
                horizontalalignment='right',
                fontweight='bold',
                transform=ax.get_yaxis_transform())

    # Formatting
    ax.set_xlabel('Normalized Dimension (d/D)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Concentration Gap (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Spectral Energy Concentration Gap - {model_name}',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

    # Add text box with key statistics in UPPER LEFT
    textstr = f'Model Dimension: {model_dim}\n'
    textstr += f'Tokens Analyzed: {num_tokens:,}\n'
    textstr += f'd/D at 80%: {threshold_dims[80]["normalized"]:.3f}\n'
    textstr += f'd/D at 90%: {threshold_dims[90]["normalized"]:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save figure
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'ecg_curve.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")

    plt.close()

    print(f"\nReplotted ECG curve for {model_name}")
    print(f"Model Dimension: {model_dim}")
    print(f"Tokens Analyzed: {num_tokens:,}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Replot ECG curve with clean formatting'
    )
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to ecg_results.npz file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as results file)')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return

    # Replot
    plot_ecg_curve_clean(args.results_file, args.output_dir)


if __name__ == '__main__':
    main()
