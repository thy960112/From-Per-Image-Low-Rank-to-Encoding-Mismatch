import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (cupy) is available and will be used for acceleration.")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("GPU (cupy) not available, using CPU (numpy) instead.")


def analyze_svd_last_layer(feature_dir, use_gpu=False):
    """Analyze SVD of last layer features.

    Args:
        feature_dir: Path to directory containing feature .npy files
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dictionary with analysis results
    """
    feature_files = list(Path(feature_dir).glob("*.npy"))
    print(f"Found {len(feature_files)} feature map files")

    # Storage for results
    dims_99 = []
    dims_95 = []
    dims_90 = []
    dims_80 = []
    ranks = []

    # Process each feature map
    for file_path in tqdm(feature_files, desc="Processing feature maps"):
        # Load feature map (shape: [24, 196, 384])
        all_layers = np.load(file_path)

        # Extract ONLY the last layer (index -1 or 23)
        # Shape: (196, 384) - 196 spatial tokens, 384 feature dimensions
        X = all_layers[-1]

        # Perform SVD on the last layer feature map
        if use_gpu and GPU_AVAILABLE:
            X_gpu = cp.asarray(X)
            U, sigma, Vt = cp.linalg.svd(X_gpu, full_matrices=False)
            # Move results back to CPU
            sigma = cp.asnumpy(sigma)
        else:
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Calculate rank (number of non-zero singular values)
        rank = np.sum(sigma > 1e-10)
        ranks.append(int(rank))

        # Calculate cumulative energy
        sigma_squared = sigma ** 2
        total_energy = np.sum(sigma_squared)
        cumulative_energy = np.cumsum(sigma_squared) / total_energy

        # Find dimensions needed for different energy levels
        dim_99 = int(np.searchsorted(cumulative_energy, 0.99) + 1)
        dim_95 = int(np.searchsorted(cumulative_energy, 0.95) + 1)
        dim_90 = int(np.searchsorted(cumulative_energy, 0.90) + 1)
        dim_80 = int(np.searchsorted(cumulative_energy, 0.80) + 1)

        dims_99.append(dim_99)
        dims_95.append(dim_95)
        dims_90.append(dim_90)
        dims_80.append(dim_80)

    # Convert to numpy arrays
    dims_99 = np.array(dims_99)
    dims_95 = np.array(dims_95)
    dims_90 = np.array(dims_90)
    dims_80 = np.array(dims_80)
    ranks = np.array(ranks)

    return {
        'dims_99': dims_99,
        'dims_95': dims_95,
        'dims_90': dims_90,
        'dims_80': dims_80,
        'ranks': ranks,
    }


def print_statistics(results):
    """Print statistics of SVD analysis results."""
    print(f"\nStatistics:")
    print(f"Dimensions for 99% energy: mean={np.mean(results['dims_99']):.2f}, std={np.std(results['dims_99']):.2f}")
    print(f"Dimensions for 95% energy: mean={np.mean(results['dims_95']):.2f}, std={np.std(results['dims_95']):.2f}")
    print(f"Dimensions for 90% energy: mean={np.mean(results['dims_90']):.2f}, std={np.std(results['dims_90']):.2f}")
    print(f"Dimensions for 80% energy: mean={np.mean(results['dims_80']):.2f}, std={np.std(results['dims_80']):.2f}")
    print(f"Ranks: mean={np.mean(results['ranks']):.2f}, std={np.std(results['ranks']):.2f}")

    # Find the dimension where 99% of samples have rank lower than this value
    rank_99_percentile = np.percentile(results['ranks'], 99)
    print(f"99th percentile of ranks: {rank_99_percentile:.2f}")


def plot_results(results, output_dir):
    """Create and save plots of SVD analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Times New Roman font
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
    })

    dims_99 = results['dims_99']
    dims_95 = results['dims_95']
    dims_90 = results['dims_90']
    dims_80 = results['dims_80']
    ranks = results['ranks']
    rank_99_percentile = np.percentile(ranks, 99)

    # Create figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SVD Analysis of Last Layer Feature Maps (Layer 23)', fontsize=16, fontweight='bold')

    # Helper function to create histogram with ratio on y-axis
    def plot_histogram(ax, data, title, xlabel, color, bins=50):
        counts_unnorm, bin_edges = np.histogram(data, bins=bins)
        ratios = counts_unnorm / len(data)

        ax.bar(bin_edges[:-1], ratios, width=np.diff(bin_edges),
               color=color, alpha=0.7, edgecolor='black', align='edge')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Ratio', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.95, 0.95, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 1: 99% energy restoration
    plot_histogram(axes[0, 0], dims_99,
                   'Dimensions for 99% Energy Restoration',
                   'Dimension', 'steelblue')

    # Plot 2: 95% energy restoration
    plot_histogram(axes[0, 1], dims_95,
                   'Dimensions for 95% Energy Restoration',
                   'Dimension', 'seagreen')

    # Plot 3: 90% energy restoration
    plot_histogram(axes[0, 2], dims_90,
                   'Dimensions for 90% Energy Restoration',
                   'Dimension', 'coral')

    # Plot 4: 80% energy restoration
    plot_histogram(axes[1, 0], dims_80,
                   'Dimensions for 80% Energy Restoration',
                   'Dimension', 'mediumpurple')

    # Plot 5: Rank distribution with 99th percentile line
    ax = axes[1, 1]
    counts_unnorm, bin_edges = np.histogram(ranks, bins=50)
    ratios = counts_unnorm / len(ranks)
    ax.bar(bin_edges[:-1], ratios, width=np.diff(bin_edges),
           color='crimson', alpha=0.7, edgecolor='black', align='edge')

    # Add vertical dotted line at 99th percentile
    ax.axvline(x=rank_99_percentile, color='black', linestyle='--', linewidth=2,
               label=f'99th percentile: {rank_99_percentile:.0f}')

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Ratio', fontsize=12)
    ax.set_title('Rank Distribution of Feature Maps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add statistics text
    mean_val = np.mean(ranks)
    std_val = np.std(ranks)
    ax.text(0.95, 0.95, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove the empty subplot
    axes[1, 2].remove()

    plt.tight_layout()
    png_path = output_dir / 'svd_analysis_times.png'
    pdf_path = output_dir / 'svd_analysis_times.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nPlots saved as '{png_path}' and '{pdf_path}'")
    plt.close()


def save_results(results, output_dir):
    """Save numerical results to .npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_99_percentile = np.percentile(results['ranks'], 99)
    npz_path = output_dir / 'svd_results.npz'
    np.savez(npz_path,
             dims_99=results['dims_99'],
             dims_95=results['dims_95'],
             dims_90=results['dims_90'],
             dims_80=results['dims_80'],
             ranks=results['ranks'],
             rank_99_percentile=rank_99_percentile)
    print(f"Results saved to '{npz_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SVD of last layer features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("features/cait_s24_224"),
        help="Directory containing feature .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output plots and results"
    )
    parser.add_argument(
        "--use-gpu",
        action='store_true',
        help="Use GPU acceleration via cupy (requires cupy installation)"
    )
    args = parser.parse_args()

    # Run analysis
    results = analyze_svd_last_layer(args.feature_dir, use_gpu=args.use_gpu)

    # Print statistics
    print_statistics(results)

    # Create plots
    plot_results(results, args.output_dir)

    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
