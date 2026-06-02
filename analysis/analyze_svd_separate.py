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

# Set Times New Roman font globally
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
})

def analyze_svd_last_layer(feature_dir, use_gpu=False):
    """Analyze SVD on the last layer."""
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
        # Load feature map:
        # - ViT/CaiT: numeric array [L, N, C]
        # - Swin/CNN: object array of stages, each [N_i, C_i]
        all_layers = np.load(file_path, allow_pickle=True)

        # Extract ONLY the last layer (index -1)
        # Shape: (N, C) for ViT/CaiT or (N_i, C_i) for Swin/CNN
        X = all_layers[-1]

        # Perform SVD on the last layer feature map
        if use_gpu and GPU_AVAILABLE:
            X_gpu = cp.asarray(X)
            U, sigma, Vt = cp.linalg.svd(X_gpu, full_matrices=False)
            sigma = cp.asnumpy(sigma)
        else:
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Calculate rank (number of non-zero singular values)
        rank = np.sum(sigma > 1e-10)
        ranks.append(rank)

        # Calculate cumulative energy
        sigma_squared = sigma ** 2
        total_energy = np.sum(sigma_squared)
        cumulative_energy = np.cumsum(sigma_squared) / total_energy

        # Find dimensions needed for different energy levels
        dim_99 = np.searchsorted(cumulative_energy, 0.99) + 1
        dim_95 = np.searchsorted(cumulative_energy, 0.95) + 1
        dim_90 = np.searchsorted(cumulative_energy, 0.90) + 1
        dim_80 = np.searchsorted(cumulative_energy, 0.80) + 1

        dims_99.append(dim_99)
        dims_95.append(dim_95)
        dims_90.append(dim_90)
        dims_80.append(dim_80)

    # Convert to numpy arrays
    return {
        'dims_99': np.array(dims_99),
        'dims_95': np.array(dims_95),
        'dims_90': np.array(dims_90),
        'dims_80': np.array(dims_80),
        'ranks': np.array(ranks)
    }

def print_statistics(results):
    """Print statistics."""
    dims_99 = results['dims_99']
    dims_95 = results['dims_95']
    dims_90 = results['dims_90']
    dims_80 = results['dims_80']
    ranks = results['ranks']

    print(f"\nStatistics:")
    print(f"Dimensions for 99% energy: mean={np.mean(dims_99):.2f}, std={np.std(dims_99):.2f}")
    print(f"Dimensions for 95% energy: mean={np.mean(dims_95):.2f}, std={np.std(dims_95):.2f}")
    print(f"Dimensions for 90% energy: mean={np.mean(dims_90):.2f}, std={np.std(dims_90):.2f}")
    print(f"Dimensions for 80% energy: mean={np.mean(dims_80):.2f}, std={np.std(dims_80):.2f}")
    print(f"Ranks: mean={np.mean(ranks):.2f}, std={np.std(ranks):.2f}")

    # Calculate 99th percentiles
    percentile_99_dims_99 = np.percentile(dims_99, 99)
    percentile_99_dims_95 = np.percentile(dims_95, 99)
    percentile_99_dims_90 = np.percentile(dims_90, 99)
    percentile_99_dims_80 = np.percentile(dims_80, 99)
    percentile_99_ranks = np.percentile(ranks, 99)

    print(f"\n99th Percentiles:")
    print(f"99% energy restoration: {percentile_99_dims_99:.2f}")
    print(f"95% energy restoration: {percentile_99_dims_95:.2f}")
    print(f"90% energy restoration: {percentile_99_dims_90:.2f}")
    print(f"80% energy restoration: {percentile_99_dims_80:.2f}")
    print(f"Ranks: {percentile_99_ranks:.2f}")

# Helper function to create histogram with 99th percentile line
def plot_single_histogram(data, title, xlabel, color, filename, percentile_val, bins=50, output_dir="."):
    """Create histogram with 99th percentile line."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 7))

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
    ax.text(percentile_val, y_max * 0.95, f' {percentile_val:.1f}',
            fontsize=14, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')

    # Add statistics text
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    png_path = output_dir / f'{filename}.png'
    pdf_path = output_dir / f'{filename}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


def create_plots(results, output_dir, last_layer_idx, feat_kind="Layer"):
    """Create all SVD plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dims_99 = results['dims_99']
    dims_95 = results['dims_95']
    dims_90 = results['dims_90']
    dims_80 = results['dims_80']
    ranks = results['ranks']

    # Calculate 99th percentiles
    p99_dims_99 = np.percentile(dims_99, 99)
    p99_dims_95 = np.percentile(dims_95, 99)
    p99_dims_90 = np.percentile(dims_90, 99)
    p99_dims_80 = np.percentile(dims_80, 99)
    p99_ranks = np.percentile(ranks, 99)

    print("\nCreating individual plots...")

    # Plot 1: 99% energy restoration
    plot_single_histogram(
        dims_99,
        f'Dimensions for 99% Energy Restoration\n(Last {feat_kind} - {feat_kind} {last_layer_idx})',
        'Dimension',
        'steelblue',
        'svd_99_percent_energy',
        p99_dims_99,
        bins=50,
        output_dir=output_dir
    )

    # Plot 2: 95% energy restoration
    plot_single_histogram(
        dims_95,
        f'Dimensions for 95% Energy Restoration\n(Last {feat_kind} - {feat_kind} {last_layer_idx})',
        'Dimension',
        'seagreen',
        'svd_95_percent_energy',
        p99_dims_95,
        bins=40,
        output_dir=output_dir
    )

    # Plot 3: 90% energy restoration
    plot_single_histogram(
        dims_90,
        f'Dimensions for 90% Energy Restoration\n(Last {feat_kind} - {feat_kind} {last_layer_idx})',
        'Dimension',
        'coral',
        'svd_90_percent_energy',
        p99_dims_90,
        bins=35,
        output_dir=output_dir
    )

    # Plot 4: 80% energy restoration
    plot_single_histogram(
        dims_80,
        f'Dimensions for 80% Energy Restoration\n(Last {feat_kind} - {feat_kind} {last_layer_idx})',
        'Dimension',
        'mediumpurple',
        'svd_80_percent_energy',
        p99_dims_80,
        bins=20,
        output_dir=output_dir
    )

    # Plot 5: Rank distribution
    plot_single_histogram(
        ranks,
        f'Rank Distribution of Feature Maps\n(Last {feat_kind} - {feat_kind} {last_layer_idx})',
        'Rank',
        'crimson',
        'svd_rank_distribution',
        p99_ranks,
        bins=10,
        output_dir=output_dir
    )

    print("\nAll plots saved successfully!")


def save_results(results, output_dir):
    """Save results to npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / 'svd_results.npz'
    np.savez(npz_path,
             dims_99=results['dims_99'],
             dims_95=results['dims_95'],
             dims_90=results['dims_90'],
             dims_80=results['dims_80'],
             ranks=results['ranks'],
             percentile_99_dims_99=np.percentile(results['dims_99'], 99),
             percentile_99_dims_95=np.percentile(results['dims_95'], 99),
             percentile_99_dims_90=np.percentile(results['dims_90'], 99),
             percentile_99_dims_80=np.percentile(results['dims_80'], 99),
             percentile_99_ranks=np.percentile(results['ranks'], 99))
    print(f"Results saved to '{npz_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SVD on last layer of extracted features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory containing feature .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output plots and results"
    )
    parser.add_argument(
        "--use-gpu",
        action='store_true',
        help="Use GPU acceleration via cupy (requires cupy installation)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SVD Analysis on Last Layer")
    print("=" * 70)
    print(f"Feature directory: {args.feature_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use GPU: {args.use_gpu}")
    print("=" * 70)
    print()

    # Run analysis
    results = analyze_svd_last_layer(args.feature_dir, use_gpu=args.use_gpu)

    # Determine last layer index from a sample file
    sample_file = list(Path(args.feature_dir).glob("*.npy"))[0]
    sample_features = np.load(sample_file, allow_pickle=True)
    last_layer_idx = sample_features.shape[0] - 1
    feat_kind = "Stage" if sample_features.dtype == object else "Layer"

    # Print statistics
    print_statistics(results)

    # Create plots
    create_plots(results, args.output_dir, last_layer_idx, feat_kind=feat_kind)

    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
