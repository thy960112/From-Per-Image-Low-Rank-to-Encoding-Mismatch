import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(data, output_dir):
    """Create and save line chart of stage-wise dimension requirements."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Times New Roman font with larger size
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 20,
    })

    # Extract data for all stages
    num_stages = int(data['num_stages'])
    stage_dims = [data[f'stage_{i}_dims'] for i in range(num_stages)]

    # Calculate statistics
    means = [np.mean(dims) for dims in stage_dims]
    stds = [np.std(dims) for dims in stage_dims]
    percentile_25 = [np.percentile(dims, 25) for dims in stage_dims]
    percentile_75 = [np.percentile(dims, 75) for dims in stage_dims]

    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stage indices (0 to num_stages-1)
    stages = np.arange(num_stages)

    # Plot main line
    ax.plot(stages, means, marker='o', linewidth=2.5, markersize=10,
            color='steelblue', label='Mean dimension', zorder=3)

    # Add shaded region for standard deviation
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(stages, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.2, color='steelblue', label='±1 Std Dev')

    # Add shaded region for 25th-75th percentile
    ax.fill_between(stages, percentile_25, percentile_75,
                    alpha=0.15, color='coral', label='25th-75th percentile')

    # Styling
    ax.set_xlabel(f'Stage Index (0-{num_stages-1})', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Dimension for 99% Energy', fontsize=20, fontweight='bold')
    ax.set_title('Swin Transformer: Stage-wise Dimension Requirements\nfor 99% Energy Restoration',
                fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='upper right')

    # Set x-axis ticks to show all stages
    ax.set_xticks(stages)
    ax.set_xticklabels([f'Stage {i}' for i in stages])

    # Add annotations for all stages (only 4 stages, so annotate all)
    # Swin channel dimensions: [96, 192, 384, 768]
    channel_dims = [96, 192, 384, 768]
    # Custom positioning to avoid title overlap
    annotation_offsets = [
        (10, -40),    # Stage 0: right and up
        (0, -60),   # Stage 1: right and down (avoid title overlap)
        (0, 40),   # Stage 2: left and up
        (-60, 40)   # Stage 3: left and up (avoid exceeding border)
    ]
    for idx in range(num_stages):
        ax.annotate(f'Stage {idx}\n({means[idx]:.1f} dims)\n({channel_dims[idx]} ch)',
                    xy=(idx, means[idx]),
                    xytext=annotation_offsets[idx],
                    textcoords='offset points',
                    fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

    # Add horizontal line at last stage value for reference
    ax.axhline(y=means[-1], color='red', linestyle='--', linewidth=1.5,
              alpha=0.5, label=f'Last stage baseline: {means[-1]:.1f}')

    plt.tight_layout()
    png_path = output_dir / 'stage_wise_99_energy_dimensions_times.png'
    pdf_path = output_dir / 'stage_wise_99_energy_dimensions_times.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nPlot saved as '{png_path}' and '{pdf_path}'")
    plt.close()


def main():
    # Load existing results
    results_path = Path("Output/swin_small/svd_layers/stage_wise_svd_results.npz")
    output_dir = Path("Output/swin_small/svd_layers")

    print(f"Loading results from {results_path}")
    data = np.load(results_path, allow_pickle=True)

    print(f"Loaded data for {int(data['num_stages'])} stages")
    print("\nRegenerating plots with new formatting...")

    # Create plots
    plot_results(data, output_dir)

    print("\nPlots regenerated successfully!")


if __name__ == "__main__":
    main()
