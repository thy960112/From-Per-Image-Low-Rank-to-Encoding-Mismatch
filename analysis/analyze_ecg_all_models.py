#!/usr/bin/env python3
"""
Multi-Model Spectral Energy Concentration Gap Analysis

Analyzes ECG for all available models and creates comparison plots and tables.

Usage:
    python analyze_ecg_all_models.py --use-gpu
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess
import csv

# Try to import cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


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


# Model configurations
MODELS = [
    {
        'short_name': 'vit_tiny',
        'model_name': 'ViT-Tiny',
        'feature_dir': 'Output/vit_tiny_patch16_224_21k/features/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        'output_dir': 'Output/vit_tiny_patch16_224_21k/ecg',
        'color': '#bcbd22',
        'marker': 'p'
    },
    {
        'short_name': 'cait',
        'model_name': 'CaiT-S24',
        'feature_dir': 'Output/cait/features/cait_s24_224',
        'output_dir': 'Output/cait/ecg',
        'color': '#1f77b4',
        'marker': 'o'
    },
    {
        'short_name': 'deit',
        'model_name': 'DeiT-Small',
        'feature_dir': 'Output/deit_small/features/deit_small_patch16_224',
        'output_dir': 'Output/deit_small/ecg',
        'color': '#ff7f0e',
        'marker': 's'
    },
    {
        'short_name': 'vit_large',
        'model_name': 'ViT-Large',
        'feature_dir': 'Output/vit_large_21k_in1k/features/vit_large_patch16_224.augreg_in21k_ft_in1k',
        'output_dir': 'Output/vit_large_21k_in1k/ecg',
        'color': '#2ca02c',
        'marker': '^'
    },
    {
        'short_name': 'vit_huge',
        'model_name': 'ViT-Huge',
        'feature_dir': 'Output/vit_huge_patch14_224_mae/features/vit_huge_patch14_224.mae',
        'output_dir': 'Output/vit_huge_patch14_224_mae/ecg',
        'color': '#d62728',
        'marker': 'v'
    },
    {
        'short_name': 'swin_small',
        'model_name': 'Swin-Small',
        'feature_dir': 'Output/swin_small/features/swin_small_patch4_window7_224.ms_in1k',
        'output_dir': 'Output/swin_small/ecg',
        'color': '#9467bd',
        'marker': 'D'
    },
    # Pretraining variants (CLIP / DINOv2 / MAE)
    {
        'short_name': 'vit_base_clip_openai',
        'model_name': 'ViT-Base (CLIP)',
        'feature_dir': 'Output/vit_base_patch16_clip_openai/features/vit_base_patch16_clip_224.openai',
        'output_dir': 'Output/vit_base_patch16_clip_openai/ecg',
        'color': '#8c564b',
        'marker': 'X'
    },
    {
        'short_name': 'vit_large_clip_openai',
        'model_name': 'ViT-Large (CLIP)',
        'feature_dir': 'Output/vit_large_patch14_clip_openai/features/vit_large_patch14_clip_224.openai',
        'output_dir': 'Output/vit_large_patch14_clip_openai/ecg',
        'color': '#e377c2',
        'marker': '*'
    },
    {
        'short_name': 'vit_base_dinov2',
        'model_name': 'ViT-Base (DINOv2)',
        'feature_dir': 'Output/vit_base_patch14_dinov2/features/vit_base_patch14_dinov2.lvd142m',
        'output_dir': 'Output/vit_base_patch14_dinov2/ecg',
        'color': '#7f7f7f',
        'marker': 'h'
    },
    {
        'short_name': 'vit_large_dinov2',
        'model_name': 'ViT-Large (DINOv2)',
        'feature_dir': 'Output/vit_large_patch14_dinov2/features/vit_large_patch14_dinov2.lvd142m',
        'output_dir': 'Output/vit_large_patch14_dinov2/ecg',
        'color': '#17becf',
        'marker': 'P'
    },
    {
        'short_name': 'vit_base_dino',
        'model_name': 'ViT-Base (DINO)',
        'feature_dir': 'Output/vit_base_patch16_224_dino/features/vit_base_patch16_224.dino',
        'output_dir': 'Output/vit_base_patch16_224_dino/ecg',
        'color': '#c7c7c7',
        'marker': 'H'
    },
    {
        'short_name': 'vit_small_dino',
        'model_name': 'ViT-Small (DINO)',
        'feature_dir': 'Output/vit_small_patch16_224_dino/features/vit_small_patch16_224.dino',
        'output_dir': 'Output/vit_small_patch16_224_dino/ecg',
        'color': '#dbdb8d',
        'marker': 's'
    },
    {
        'short_name': 'vit_base_mae',
        'model_name': 'ViT-Base (MAE)',
        'feature_dir': 'Output/vit_base_patch16_224_mae/features/vit_base_patch16_224.mae',
        'output_dir': 'Output/vit_base_patch16_224_mae/ecg',
        'color': '#98df8a',
        'marker': '>'
    },
    {
        'short_name': 'vit_large_mae',
        'model_name': 'ViT-Large (MAE)',
        'feature_dir': 'Output/vit_large_patch16_224_mae/features/vit_large_patch16_224.mae',
        'output_dir': 'Output/vit_large_patch16_224_mae/ecg',
        'color': '#ffbb78',
        'marker': '<'
    },
]


def analyze_single_model(model_config, use_gpu):
    """
    Analyze a single model using analyze_ecg_spectral.py

    Args:
        model_config: dict with model configuration
        use_gpu: whether to use GPU
    """
    feature_dir = model_config['feature_dir']
    output_dir = model_config['output_dir']
    model_name = model_config['model_name']

    # Check if feature directory exists
    if not Path(feature_dir).exists():
        print(f"Skipping {model_name}: feature directory not found at {feature_dir}")
        return False

    # Check if analysis already done
    results_file = Path(output_dir) / 'ecg_results.npz'
    if results_file.exists():
        print(f"Results already exist for {model_name}, loading from cache...")
        return True

    # Run analysis
    cmd = [
        'python', 'analyze_ecg_spectral.py',
        '--feature-dir', feature_dir,
        '--output-dir', output_dir,
        '--model-name', model_name
    ]

    if use_gpu:
        cmd.append('--use-gpu')

    print(f"\nAnalyzing {model_name}...")
    subprocess.run(cmd, check=True)

    return True


def load_model_results(model_config):
    """
    Load ECG results for a model.

    Args:
        model_config: dict with model configuration

    Returns:
        results dict or None if not found
    """
    results_file = Path(model_config['output_dir']) / 'ecg_results.npz'

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


def plot_multi_model_comparison(all_results, models, output_dir):
    """
    Plot ECG curves for all models on the same plot.

    Args:
        all_results: list of results dicts
        models: list of model configs
        output_dir: output directory
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each model
    for results, model_config in zip(all_results, models):
        if results is None:
            continue

        model_dim = results['model_dim']
        mean_ecg = results['mean_ecg_curve']

        # Create normalized dimension axis
        dimensions = np.arange(1, model_dim + 1)
        normalized_dims = dimensions / model_dim

        # Plot
        ax.plot(normalized_dims, mean_ecg,
                color=model_config['color'],
                marker=model_config['marker'],
                markevery=max(1, model_dim // 20),
                linewidth=2.5,
                markersize=6,
                label=f"{model_config['model_name']} (D={model_dim})",
                alpha=0.9,
                zorder=10)

    # Add threshold lines
    thresholds = [50, 60, 70, 80, 90]
    threshold_colors = ['green', 'orange', 'red', 'purple', 'brown']

    for thresh, color in zip(thresholds, threshold_colors):
        # Horizontal line at threshold
        ax.axhline(y=thresh, color=color, linestyle='--',
                   alpha=0.4, linewidth=1.5, zorder=1)

        # Label INSIDE the plot on the right side to avoid overlap with legend
        ax.text(0.98, thresh + 1.5, f'{thresh}%',
                fontsize=11,
                color=color,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, alpha=0.8, linewidth=1.5),
                zorder=10)

    # Formatting
    ax.set_xlabel('Normalized Frequency (f/D)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Spectral Energy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Cumulative Spectral Energy Pattern Comparison\n'
                 'Across Vision Transformer Architectures (Frequency Order Preserved)',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])  # Changed from 105 to 100 to avoid extra space
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    # Compact legend: 2 columns, smaller font, placed where it doesn't cover the curves.
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        fontsize=10,
        ncol=2,
        framealpha=0.90,
        edgecolor='black',
        fancybox=True,
        columnspacing=1.0,
        handlelength=1.6,
        borderaxespad=0.2,
    )

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        output_file = Path(output_dir) / f'ecg_comparison_all_models.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {output_file}")

    plt.close()


def create_comparison_table(all_results, models, output_dir):
    """
    Create comparison table with threshold dimensions for all models.

    Args:
        all_results: list of results dicts
        models: list of model configs
        output_dir: output directory
    """
    csv_file = Path(output_dir) / 'ecg_comparison_table.csv'

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Model', 'Dimension (D)', 'Num Tokens',
            'd/D @ 50%', 'd @ 50%',
            'd/D @ 60%', 'd @ 60%',
            'd/D @ 70%', 'd @ 70%',
            'd/D @ 80%', 'd @ 80%',
            'd/D @ 90%', 'd @ 90%'
        ])

        # Data rows
        for results, model_config in zip(all_results, models):
            if results is None:
                continue

            model_name = results['model_name']
            model_dim = results['model_dim']
            num_tokens = results['num_tokens']
            threshold_dims = results['threshold_dims']

            row = [model_name, model_dim, num_tokens]

            for thresh in [50, 60, 70, 80, 90]:
                if thresh in threshold_dims:
                    norm = threshold_dims[thresh]['normalized']
                    dim = threshold_dims[thresh]['dimension']
                    row.extend([f'{norm:.6f}', dim])
                else:
                    row.extend(['N/A', 'N/A'])

            writer.writerow(row)

    print(f"Saved comparison table: {csv_file}")

    # Also print to console
    print("\n" + "="*100)
    print("COMPARISON TABLE: Normalized Dimensions (d/D) at Energy Thresholds")
    print("="*100)
    print(f"{'Model':<15} {'D':<6} {'Tokens':<8} {'50%':<8} {'60%':<8} {'70%':<8} {'80%':<8} {'90%':<8}")
    print("-"*100)

    for results, model_config in zip(all_results, models):
        if results is None:
            continue

        model_name = results['model_name']
        model_dim = results['model_dim']
        num_tokens = results['num_tokens']
        threshold_dims = results['threshold_dims']

        row_str = f"{model_name:<15} {model_dim:<6} {num_tokens:<8}"

        for thresh in [50, 60, 70, 80, 90]:
            if thresh in threshold_dims:
                norm = threshold_dims[thresh]['normalized']
                row_str += f" {norm:.4f}  "
            else:
                row_str += " N/A     "

        print(row_str)

    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Spectral Energy Concentration Gap Analysis'
    )
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration with CuPy if available')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip individual model analysis, only create comparison plots')

    args = parser.parse_args()

    print("="*80)
    print("Multi-Model Spectral Energy Concentration Gap Analysis")
    print("="*80)

    # Create output directory
    output_dir = Path('Output/comparison/ecg')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    if not args.skip_analysis:
        print("\n[Step 1] Analyzing individual models...")
        for model_config in MODELS:
            try:
                analyze_single_model(model_config, args.use_gpu)
            except Exception as e:
                print(f"Error analyzing {model_config['model_name']}: {e}")
                continue

    # Load all results
    print("\n[Step 2] Loading results...")
    all_results = []
    available_models = []

    for model_config in MODELS:
        results = load_model_results(model_config)
        if results is not None:
            all_results.append(results)
            available_models.append(model_config)
            print(f"✓ Loaded results for {model_config['model_name']}")
        else:
            print(f"✗ No results found for {model_config['model_name']}")

    if len(all_results) == 0:
        print("\nError: No model results found. Please run analysis first.")
        return

    # Create comparison plots
    print("\n[Step 3] Creating comparison plots...")
    plot_multi_model_comparison(all_results, available_models, output_dir)

    # Create comparison table
    print("\n[Step 4] Creating comparison table...")
    create_comparison_table(all_results, available_models, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Comparison results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
