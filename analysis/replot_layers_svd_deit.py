import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(stats, output_dir):
    """Create and save line chart of layer-wise dimension requirements."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Times New Roman font with larger size
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 20,
    })

    mean_dims = stats['mean_dims']
    std_dims = stats['std_dims']
    percentile_25 = stats['percentile_25']
    percentile_75 = stats['percentile_75']
    num_layers = len(mean_dims)

    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Layer indices (0 to num_layers-1)
    layers = np.arange(num_layers)

    # Plot main line
    ax.plot(layers, mean_dims, marker='o', linewidth=2.5, markersize=8,
            color='steelblue', label='Mean dimension', zorder=3)

    # Add shaded region for standard deviation
    ax.fill_between(layers, mean_dims - std_dims, mean_dims + std_dims,
                    alpha=0.2, color='steelblue', label='±1 Std Dev')

    # Add shaded region for 25th-75th percentile
    ax.fill_between(layers, percentile_25, percentile_75,
                    alpha=0.15, color='coral', label='25th-75th percentile')

    # Styling
    ax.set_xlabel(f'Layer Index (0-{num_layers-1})', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Dimension for 99% Energy', fontsize=20, fontweight='bold')
    ax.set_title('Layer-wise Dimension Requirements for 99% Energy Restoration',
                fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='best')

    # Set x-axis ticks to show all layers
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])

    # Add annotations for first, middle, and last layers
    middle_idx = num_layers // 2  # Middle layer
    for idx, label in [(0, 'First layer'), (middle_idx, 'Middle layer'), (num_layers-1, 'Last layer')]:
        ax.annotate(f'{label}\n({mean_dims[idx]:.1f})',
                    xy=(idx, mean_dims[idx]),
                    xytext=(10, 20) if idx != (num_layers-1) else (-80, 20),
                    textcoords='offset points',
                    fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

    # Add horizontal line at last layer value for reference
    ax.axhline(y=mean_dims[-1], color='red', linestyle='--', linewidth=1.5,
              alpha=0.5, label=f'Last layer baseline: {mean_dims[-1]:.1f}')

    plt.tight_layout()
    png_path = output_dir / 'layer_wise_99_energy_dimensions_times.png'
    pdf_path = output_dir / 'layer_wise_99_energy_dimensions_times.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nPlot saved as '{png_path}' and '{pdf_path}'")
    plt.close()


def main():
    # Load existing results
    results_path = Path("Output/deit_small/svd_layers/layer_wise_svd_results.npz")
    output_dir = Path("Output/deit_small/svd_layers")

    print(f"Loading results from {results_path}")
    data = np.load(results_path, allow_pickle=True)

    stats = {
        'mean_dims': data['mean_dims'],
        'std_dims': data['std_dims'],
        'median_dims': data['median_dims'],
        'percentile_25': data['percentile_25'],
        'percentile_75': data['percentile_75'],
    }

    print(f"Loaded data for {len(stats['mean_dims'])} layers")
    print("\nRegenerating plots with new formatting...")

    # Create plots
    plot_results(stats, output_dir)

    print("\nPlots regenerated successfully!")


if __name__ == "__main__":
    main()
