import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Times New Roman font globally with larger size
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 25,
})

def plot_single_histogram(data, title, xlabel, color, filename, percentile_val, bins=50, output_dir="."):
    """Create histogram with 99th percentile line."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create histogram
    counts_unnorm, bin_edges = np.histogram(data, bins=bins)
    ratios = counts_unnorm / len(data)

    ax.bar(bin_edges[:-1], ratios, width=np.diff(bin_edges),
           color=color, alpha=0.7, edgecolor='black', align='edge')

    # Add vertical dotted line at 99th percentile
    ax.axvline(x=percentile_val, color='black', linestyle='--', linewidth=2.5,
               label=f'99th percentile: {percentile_val:.1f}')

    # Add text annotation for the 99th percentile
    y_max = ratios.max()
    ax.text(percentile_val, y_max * 0.70, f' {percentile_val:.1f}',
            fontsize=20, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=25, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=25, fontweight='bold')
    ax.set_title(title, fontsize=25, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='upper right')

    # Add statistics text
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    png_path = output_dir / f'{filename}.png'
    pdf_path = output_dir / f'{filename}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


def main():
    # Load existing results
    results_path = Path("Output/swin_small/svd/svd_results.npz")
    output_dir = Path("Output/swin_small/svd")

    print(f"Loading results from {results_path}")
    data = np.load(results_path)

    dims_99 = data['dims_99']
    dims_95 = data['dims_95']
    dims_90 = data['dims_90']
    dims_80 = data['dims_80']
    ranks = data['ranks']

    # Load percentiles
    p99_dims_99 = data['percentile_99_dims_99']
    p99_dims_95 = data['percentile_99_dims_95']
    p99_dims_90 = data['percentile_99_dims_90']
    p99_dims_80 = data['percentile_99_dims_80']
    p99_ranks = data['percentile_99_ranks']

    # Swin has 4 stages, last is Stage 3
    last_stage_idx = 3

    print("\nRegenerating plots with new formatting...")

    # Plot 1: 99% energy restoration
    plot_single_histogram(
        dims_99,
        f'Dimensions for 99% Energy Restoration\n(Last Stage - Stage {last_stage_idx})',
        'Dimension',
        'steelblue',
        'svd_99_percent_energy_times',
        p99_dims_99,
        bins=50,
        output_dir=output_dir
    )

    # Plot 2: 95% energy restoration
    plot_single_histogram(
        dims_95,
        f'Dimensions for 95% Energy Restoration\n(Last Stage - Stage {last_stage_idx})',
        'Dimension',
        'seagreen',
        'svd_95_percent_energy_times',
        p99_dims_95,
        bins=40,
        output_dir=output_dir
    )

    # Plot 3: 90% energy restoration
    plot_single_histogram(
        dims_90,
        f'Dimensions for 90% Energy Restoration\n(Last Stage - Stage {last_stage_idx})',
        'Dimension',
        'coral',
        'svd_90_percent_energy_times',
        p99_dims_90,
        bins=35,
        output_dir=output_dir
    )

    # Plot 4: 80% energy restoration
    plot_single_histogram(
        dims_80,
        f'Dimensions for 80% Energy Restoration\n(Last Stage - Stage {last_stage_idx})',
        'Dimension',
        'mediumpurple',
        'svd_80_percent_energy_times',
        p99_dims_80,
        bins=20,
        output_dir=output_dir
    )

    # Plot 5: Rank distribution
    plot_single_histogram(
        ranks,
        f'Rank Distribution of Feature Maps\n(Last Stage - Stage {last_stage_idx})',
        'Rank',
        'crimson',
        'svd_rank_distribution_times',
        p99_ranks,
        bins=10,
        output_dir=output_dir
    )

    print("\nAll plots regenerated successfully!")


if __name__ == "__main__":
    main()
