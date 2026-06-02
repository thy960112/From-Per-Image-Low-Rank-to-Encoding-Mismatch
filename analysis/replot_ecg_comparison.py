#!/usr/bin/env python3
"""
Replot ECG comparison with updated formatting for publication.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Model configurations (same as analyze_ecg_all_models.py)
MODELS = [
    {
        'short_name': 'vit_tiny',
        'model_name': 'ViT-Tiny',
        'output_dir': 'Output/vit_tiny_patch16_224_21k/ecg',
        'color': '#bcbd22',
        'marker': 'p'
    },
    {
        'short_name': 'cait',
        'model_name': 'CaiT-S24',
        'output_dir': 'Output/cait/ecg',
        'color': '#1f77b4',
        'marker': 'o'
    },
    {
        'short_name': 'deit',
        'model_name': 'DeiT-Small',
        'output_dir': 'Output/deit_small/ecg',
        'color': '#ff7f0e',
        'marker': 's'
    },
    {
        'short_name': 'vit_large',
        'model_name': 'ViT-Large',
        'output_dir': 'Output/vit_large_21k_in1k/ecg',
        'color': '#2ca02c',
        'marker': '^'
    },
    {
        'short_name': 'vit_huge',
        'model_name': 'ViT-Huge',
        'output_dir': 'Output/vit_huge_patch14_224_mae/ecg',
        'color': '#d62728',
        'marker': 'v'
    },
    {
        'short_name': 'swin_small',
        'model_name': 'Swin-Small',
        'output_dir': 'Output/swin_small/ecg',
        'color': '#9467bd',
        'marker': 'D'
    },
    # Pretraining variants (CLIP / DINOv2 / MAE)
    {
        'short_name': 'vit_base_clip_openai',
        'model_name': 'ViT-Base (CLIP)',
        'output_dir': 'Output/vit_base_patch16_clip_openai/ecg',
        'color': '#8c564b',
        'marker': 'X'
    },
    {
        'short_name': 'vit_large_clip_openai',
        'model_name': 'ViT-Large (CLIP)',
        'output_dir': 'Output/vit_large_patch14_clip_openai/ecg',
        'color': '#e377c2',
        'marker': '*'
    },
    {
        'short_name': 'vit_base_dinov2',
        'model_name': 'ViT-Base (DINOv2)',
        'output_dir': 'Output/vit_base_patch14_dinov2/ecg',
        'color': '#7f7f7f',
        'marker': 'h'
    },
    {
        'short_name': 'vit_large_dinov2',
        'model_name': 'ViT-Large (DINOv2)',
        'output_dir': 'Output/vit_large_patch14_dinov2/ecg',
        'color': '#17becf',
        'marker': 'P'
    },
    {
        'short_name': 'vit_base_dino',
        'model_name': 'ViT-Base (DINO)',
        'output_dir': 'Output/vit_base_patch16_224_dino/ecg',
        'color': '#c7c7c7',
        'marker': 'H'
    },
    {
        'short_name': 'vit_small_dino',
        'model_name': 'ViT-Small (DINO)',
        'output_dir': 'Output/vit_small_patch16_224_dino/ecg',
        'color': '#dbdb8d',
        'marker': 's'
    },
    {
        'short_name': 'vit_base_mae',
        'model_name': 'ViT-Base (MAE)',
        'output_dir': 'Output/vit_base_patch16_224_mae/ecg',
        'color': '#98df8a',
        'marker': '>'
    },
    {
        'short_name': 'vit_large_mae',
        'model_name': 'ViT-Large (MAE)',
        'output_dir': 'Output/vit_large_patch16_224_mae/ecg',
        'color': '#ffbb78',
        'marker': '<'
    },
]


def load_model_results(model_config):
    """Load ECG results for a model."""
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
    Plot ECG curves for all models with updated formatting.

    New specifications:
    - Figure size: (10, 6)
    - Font size: 20
    - X-axis: "Normalized Spectral Bandwidth (d/D)"
    - Title: "Cumulative Spectral Energy Pattern Comparison Across Vision Transformer Architectures"
    """
    # Set up plot style with larger font
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

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

        # Label INSIDE the plot on the right side
        ax.text(0.98, thresh + 1.5, f'{thresh}%',
                fontsize=16,
                color=color,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, alpha=0.8, linewidth=1.5),
                zorder=10)

    # Formatting with updated labels
    ax.set_xlabel('Normalized Spectral Bandwidth (d/D)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Cumulative Spectral Energy (%)', fontsize=20, fontweight='bold')
    ax.set_title('Cumulative Spectral Energy Pattern Comparison\nAcross Vision Transformer Architectures',
                 fontsize=20, fontweight='bold', pad=20)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    # Compact legend: 2 columns, smaller font, placed where it doesn't cover the curves.
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        fontsize=11,
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
        print(f"Saved: {output_file}")

    plt.close()


def main():
    output_dir = Path('Output/comparison/ecg')

    print("Loading ECG results from individual models...")

    # Load all results
    all_results = []
    available_models = []

    for model_config in MODELS:
        results = load_model_results(model_config)
        if results is not None:
            all_results.append(results)
            available_models.append(model_config)
            print(f"✓ Loaded {model_config['model_name']}")
        else:
            print(f"✗ Skipped {model_config['model_name']} (no results)")

    if len(all_results) == 0:
        print("\nError: No model results found.")
        return

    print(f"\nRegenerating comparison plot with new formatting...")
    print(f"  - Figure size: (10, 6)")
    print(f"  - Font size: 20")
    print(f"  - X-axis: Normalized Spectral Bandwidth (d/D)")
    print(f"  - Title: Cumulative Spectral Energy Pattern Comparison Across Vision Transformer Architectures")

    # Create comparison plot with new formatting
    plot_multi_model_comparison(all_results, available_models, output_dir)

    print("\nPlot regenerated successfully!")


if __name__ == '__main__':
    main()
