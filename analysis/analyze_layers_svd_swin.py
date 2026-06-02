import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# Set Times New Roman font globally
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
})

def analyze_svd_all_stages(feature_dir):
    """Analyze SVD across all 4 stages of Swin Transformer."""
    feature_files = list(Path(feature_dir).glob("*.npy"))
    print(f"Found {len(feature_files)} feature map files")
    
    # Load first file to detect number of stages
    first_file = np.load(feature_files[0], allow_pickle=True)
    num_stages = len(first_file)
    print(f"Detected {num_stages} stages in the model")
    
    # Storage for layer-wise dimensions (for 99% energy)
    layer_wise_dims = [[] for _ in range(num_stages)]
    
    # Process each feature map
    for file_path in tqdm(feature_files, desc="Processing feature maps"):
        # Load all stages (object array with 4 stages)
        all_stages = np.load(file_path, allow_pickle=True)
        
        # Analyze each stage
        for stage_idx in range(num_stages):
            X = all_stages[stage_idx]  # Shape: (N_i, C_i)
            
            # Perform SVD
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
            
            # Calculate cumulative energy
            sigma_squared = sigma ** 2
            total_energy = np.sum(sigma_squared)
            cumulative_energy = np.cumsum(sigma_squared) / total_energy
            
            # Find dimension needed for 99% energy
            dim_99 = np.searchsorted(cumulative_energy, 0.99) + 1
            layer_wise_dims[stage_idx].append(dim_99)
    
    # Convert lists to arrays
    layer_wise_dims = [np.array(dims) for dims in layer_wise_dims]
    
    return {
        'num_stages': num_stages,
        'layer_wise_dims': layer_wise_dims
    }

def print_statistics(results):
    """Print layer-wise statistics."""
    num_stages = results['num_stages']
    layer_wise_dims = results['layer_wise_dims']
    
    print(f"\nStage-wise Statistics (99% Energy Restoration):")
    print(f"{'Stage':<8} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("=" * 40)
    
    for stage_idx in range(num_stages):
        dims = layer_wise_dims[stage_idx]
        mean_val = np.mean(dims)
        std_val = np.std(dims)
        median_val = np.median(dims)
        print(f"{stage_idx:<8} {mean_val:<10.2f} {std_val:<10.2f} {median_val:<10.2f}")

def create_plot(results, output_dir):
    """Create stage-wise SVD plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_stages = results['num_stages']
    layer_wise_dims = results['layer_wise_dims']
    
    # Calculate statistics for each stage
    means = [np.mean(dims) for dims in layer_wise_dims]
    stds = [np.std(dims) for dims in layer_wise_dims]
    medians = [np.median(dims) for dims in layer_wise_dims]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    stage_indices = list(range(num_stages))
    
    # Plot mean with error bars (std)
    ax.errorbar(stage_indices, means, yerr=stds, marker='o', markersize=8,
                linewidth=2, capsize=5, capthick=2, label='Mean ± Std', color='steelblue')
    
    # Plot median
    ax.plot(stage_indices, medians, marker='s', markersize=8,
            linewidth=2, linestyle='--', label='Median', color='coral')
    
    ax.set_xlabel('Stage Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dimensions for 99% Energy', fontsize=14, fontweight='bold')
    ax.set_title('Swin Transformer: Stage-wise SVD Analysis\\n(99% Energy Restoration)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xticks(stage_indices)
    
    plt.tight_layout()
    
    # Save plots
    png_path = output_dir / 'stage_wise_99_energy_dimensions.png'
    pdf_path = output_dir / 'stage_wise_99_energy_dimensions.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as '{png_path}' and '{pdf_path}'")

def save_results(results, output_dir):
    """Save results to npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_dict = {
        'num_stages': results['num_stages'],
    }
    
    # Add each stage's dimensions
    for stage_idx, dims in enumerate(results['layer_wise_dims']):
        save_dict[f'stage_{stage_idx}_dims'] = dims
    
    npz_path = output_dir / 'stage_wise_svd_results.npz'
    np.savez(npz_path, **save_dict)
    print(f"Data saved to '{npz_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SVD across all stages of Swin Transformer",
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
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_svd_all_stages(args.feature_dir)
    
    # Print statistics
    print_statistics(results)
    
    # Create plot
    create_plot(results, args.output_dir)
    
    # Save results
    save_results(results, args.output_dir)

if __name__ == "__main__":
    main()
