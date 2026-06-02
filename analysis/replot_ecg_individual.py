#!/usr/bin/env python3
"""
Replot individual ECG curves for all models with updated formatting.

Updates:
- Figure size: (8, 6)
- Font size: 25
- X-axis: "Normalized Spectral Bandwidth (d/D)"
- Title: "Cumulative SEP - {Model Name}"
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def discover_ecg_output_dirs(output_root='Output'):
    """Return every existing ECG or zero-safe ECG output directory with saved results."""
    root = Path(output_root)
    patterns = ('*/ecg/ecg_results.npz', '*/ecg_zero_safe/ecg_results.npz')
    results = {
        results_file.parent
        for pattern in patterns
        for results_file in root.glob(pattern)
    }
    return sorted(results)


def load_model_results(output_dir):
    """Load ECG results for a model."""
    results_file = Path(output_dir) / 'ecg_results.npz'

    if not results_file.exists():
        return None

    data = np.load(results_file, allow_pickle=True)

    results = {
        'model_name': str(data['model_name']),
        'model_dim': int(data['model_dim']),
        'num_tokens': int(data['num_tokens']),
        'mean_ecg_curve': data['mean_ecg_curve'],
        'std_ecg_curve': data['std_ecg_curve'],
        'threshold_dims': data['threshold_dims'].item(),
        'thresholds': data['thresholds']
    }

    return results


def plot_ecg_curve_updated(results, model_name, output_dir):
    """
    Plot individual ECG curve with updated formatting.

    Updates:
    - Figure size: (8, 6)
    - Font size: 25
    - X-axis: "Normalized Spectral Bandwidth (d/D)"
    - Title: "Cumulative SEP - {Model Name}"
    """
    # Set up plot style with larger font
    plt.rcParams.update({
        'font.size': 25,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
    })

    mean_ecg = results['mean_ecg_curve']
    std_ecg = results['std_ecg_curve']
    model_dim = results['model_dim']
    threshold_dims = results['threshold_dims']

    # Create normalized dimension axis
    dimensions = np.arange(1, model_dim + 1)
    normalized_dims = dimensions / model_dim

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot mean ECG curve
    ax.plot(normalized_dims, mean_ecg, 'b-', linewidth=2.5, label='Mean SEP', zorder=10)

    # Plot standard deviation as shaded region
    ax.fill_between(normalized_dims,
                     mean_ecg - std_ecg,
                     mean_ecg + std_ecg,
                     alpha=0.2, color='blue', label='±1 std', zorder=5)

    # Add threshold lines
    thresholds = [50, 60, 70, 80, 90]
    colors = ['green', 'orange', 'red', 'purple', 'brown']

    for thresh, color in zip(thresholds, colors):
        norm_dim = threshold_dims[thresh]['normalized']

        # Horizontal line at threshold
        ax.axhline(y=thresh, color=color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)

        # Vertical line at normalized dimension
        ax.axvline(x=norm_dim, color=color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)

        # Label INSIDE the plot on the right side with smaller font
        ax.text(0.97, thresh + 1, f'{thresh}%',
                fontsize=14,
                color=color,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor=color, alpha=0.85, linewidth=1.2),
                zorder=10)

    # Formatting with updated labels
    ax.set_xlabel('Normalized Spectral Bandwidth (d/D)', fontsize=25, fontweight='bold')
    ax.set_ylabel('Cumulative Spectral Energy (%)', fontsize=25, fontweight='bold')
    ax.set_title(f'Cumulative SEP - {model_name}',
                 fontsize=25, fontweight='bold', pad=20)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.95,
              edgecolor='black', fancybox=True)

    # Add text box with key statistics at top left
    textstr = f'Model Dim: {model_dim}\n'
    textstr += f'd/D at 80%: {threshold_dims[80]["normalized"]:.3f}\n'
    textstr += f'd/D at 90%: {threshold_dims[90]["normalized"]:.3f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save figure with _times suffix
    output_path = Path(output_dir)
    for ext in ['png', 'pdf']:
        output_file = output_path / f'ecg_curve_times.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def main():
    print("Replotting individual ECG curves with new formatting...")
    print(f"  - Figure size: (8, 6)")
    print(f"  - Font size: 25")
    print(f"  - X-axis: Normalized Spectral Bandwidth (d/D)")
    print(f"  - Title: Cumulative SEP - {{Model Name}}")
    print()

    output_dirs = discover_ecg_output_dirs()
    if not output_dirs:
        print("No ECG result directories found.")
        return

    print(f"  - Found {len(output_dirs)} ECG result directories")
    print()

    # Process each discovered model output
    for output_dir in output_dirs:
        print(f"Processing {output_dir.parent.name}...")

        # Load results
        results = load_model_results(output_dir)

        if results is None:
            print(f"  ✗ Skipped (no results found)")
            continue

        # Replot
        plot_ecg_curve_updated(results, results['model_name'], output_dir)
        print(f"  ✓ Completed")
        print()

    print("All ECG curves regenerated successfully!")


if __name__ == '__main__':
    main()
