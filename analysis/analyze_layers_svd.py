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


def analyze_svd_all_layers(feature_dir, use_gpu=False):
    """Analyze SVD across all layers.

    Args:
        feature_dir: Path to directory containing feature .npy files
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dictionary with analysis results per layer
    """
    feature_files = list(Path(feature_dir).glob("*.npy"))
    print(f"Found {len(feature_files)} feature map files")

    # Detect number of layers from first file
    first_features = np.load(feature_files[0])
    num_layers = first_features.shape[0]
    print(f"Detected {num_layers} layers in the model")

    # Storage for results - dimensions needed for 99% energy per layer
    # dims_per_layer[layer_idx] = list of dimensions across all samples
    dims_per_layer = [[] for _ in range(num_layers)]

    # Process each feature map
    for file_path in tqdm(feature_files, desc="Processing feature maps"):
        # Load feature map (shape: [L, 196, 384])
        all_layers = np.load(file_path)

        # Process each layer
        for layer_idx in range(num_layers):
            # Extract features for this layer
            # Shape: (196, 384) - 196 spatial tokens, 384 feature dimensions
            X = all_layers[layer_idx]

            # Perform SVD
            if use_gpu and GPU_AVAILABLE:
                X_gpu = cp.asarray(X)
                U, sigma, Vt = cp.linalg.svd(X_gpu, full_matrices=False)
                # Move results back to CPU
                sigma = cp.asnumpy(sigma)
            else:
                U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

            # Calculate cumulative energy
            sigma_squared = sigma ** 2
            total_energy = np.sum(sigma_squared)
            cumulative_energy = np.cumsum(sigma_squared) / total_energy

            # Find dimension needed for 99% energy
            dim_99 = int(np.searchsorted(cumulative_energy, 0.99) + 1)

            dims_per_layer[layer_idx].append(dim_99)

    return {'dims_per_layer': dims_per_layer}


def compute_statistics(results):
    """Compute statistics from layer-wise results."""
    dims_per_layer = results['dims_per_layer']
    num_layers = len(dims_per_layer)

    mean_dims = []
    std_dims = []
    median_dims = []
    percentile_25 = []
    percentile_75 = []

    for layer_idx in range(num_layers):
        dims_array = np.array(dims_per_layer[layer_idx])
        mean_dims.append(np.mean(dims_array))
        std_dims.append(np.std(dims_array))
        median_dims.append(np.median(dims_array))
        percentile_25.append(np.percentile(dims_array, 25))
        percentile_75.append(np.percentile(dims_array, 75))

    return {
        'mean_dims': np.array(mean_dims),
        'std_dims': np.array(std_dims),
        'median_dims': np.array(median_dims),
        'percentile_25': np.array(percentile_25),
        'percentile_75': np.array(percentile_75),
        'dims_per_layer': dims_per_layer
    }


def print_statistics(stats):
    """Print layer-wise statistics."""
    mean_dims = stats['mean_dims']
    std_dims = stats['std_dims']
    median_dims = stats['median_dims']
    num_layers = len(mean_dims)

    print(f"\nLayer-wise Statistics (99% Energy Restoration):")
    print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("=" * 40)
    for layer_idx in range(num_layers):
        print(f"{layer_idx:<8} {mean_dims[layer_idx]:<10.2f} {std_dims[layer_idx]:<10.2f} {median_dims[layer_idx]:<10.2f}")


def plot_results(stats, output_dir):
    """Create and save line chart of layer-wise dimension requirements."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Times New Roman font
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
    })

    mean_dims = stats['mean_dims']
    std_dims = stats['std_dims']
    percentile_25 = stats['percentile_25']
    percentile_75 = stats['percentile_75']
    num_layers = len(mean_dims)

    # Create line chart
    fig, ax = plt.subplots(figsize=(14, 8))

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
    ax.set_xlabel(f'Layer Index (0-{num_layers-1})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Dimension for 99% Energy', fontsize=14, fontweight='bold')
    ax.set_title('Layer-wise Dimension Requirements for 99% Energy Restoration',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')

    # Set x-axis ticks to show all layers
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])

    # Add annotations for first, middle, and last layers
    middle_idx = num_layers // 2
    for idx, label in [(0, 'First layer'), (middle_idx, 'Middle layer'), (num_layers-1, 'Last layer')]:
        ax.annotate(f'{label}\n({mean_dims[idx]:.1f})',
                    xy=(idx, mean_dims[idx]),
                    xytext=(10, 20) if idx != (num_layers-1) else (-80, 20),
                    textcoords='offset points',
                    fontsize=10,
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


def save_results(stats, output_dir):
    """Save numerical results to .npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / 'layer_wise_svd_results.npz'
    np.savez(npz_path,
             mean_dims=stats['mean_dims'],
             std_dims=stats['std_dims'],
             median_dims=stats['median_dims'],
             percentile_25=stats['percentile_25'],
             percentile_75=stats['percentile_75'],
             dims_per_layer=stats['dims_per_layer'])
    print(f"Data saved to '{npz_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SVD across all layers",
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
    results = analyze_svd_all_layers(args.feature_dir, use_gpu=args.use_gpu)

    # Compute statistics
    stats = compute_statistics(results)

    # Print statistics
    print_statistics(stats)

    # Create plots
    plot_results(stats, args.output_dir)

    # Save results
    save_results(stats, args.output_dir)


if __name__ == "__main__":
    main()
