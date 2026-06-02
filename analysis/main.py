#!/usr/bin/env python3
"""
Main script to prepare dataset and extract ViT features.

Usage:
    # Step 1: Prepare dataset (only needs to be run once)
    python main.py --step 1

    # Step 2: Extract features
    python main.py --step 2 --model vit_base_patch16_224

    # Or run both steps
    python main.py --step all --model vit_base_patch16_224

    # Use offline mode for HuggingFace
    HF_HUB_OFFLINE=1 python main.py --step 2 --model vit_base_patch16_224
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='ImageNet ViT Feature Extraction Pipeline')

    parser.add_argument(
        '--step',
        type=str,
        choices=['1', '2', 'all'],
        default='all',
        help='Which step to run: 1=prepare dataset, 2=extract features, all=both'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='vit_base_patch16_224',
        help='ViT model name from timm (e.g., vit_base_patch16_224, vit_large_patch16_224)'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='data/ILSVRC/val',
        help='Source ImageNet validation directory'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/1000_val',
        help='Target dataset directory (1000 images)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='Output/features',
        help='Output directory for features'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for feature extraction'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='Optional square input resolution override passed to the timm model and transforms '
             '(e.g., 224). Useful for models whose default pretrained config uses a larger '
             'resolution (e.g., DINOv2 at 518).'
    )

    parser.add_argument(
        '--offline',
        action='store_true',
        help='Use offline mode for HuggingFace Hub'
    )

    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional cap on number of images processed (useful for quick smoke tests)'
    )

    args = parser.parse_args()

    # Set offline mode if requested
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        print("HuggingFace Hub offline mode enabled")

    print("=" * 80)
    print("ImageNet ViT Feature Extraction Pipeline")
    print("=" * 80)

    # Step 1: Prepare dataset
    if args.step in ['1', 'all']:
        from prepare_dataset import prepare_test_dataset
        print("\n" + "=" * 80)
        print("STEP 1: Preparing Test Dataset")
        print("=" * 80)

        if not os.path.exists(args.source):
            print(f"Error: Source directory not found: {args.source}")
            sys.exit(1)

        prepare_test_dataset(args.source, args.dataset)
        print(f"\n✓ Dataset preparation complete!")

    # Step 2: Extract features
    if args.step in ['2', 'all']:
        # Import timm/torch only after offline flags are set (HF_HUB_OFFLINE is read at import time
        # in some huggingface_hub versions).
        from extract_features import process_dataset
        print("\n" + "=" * 80)
        print("STEP 2: Extracting Features")
        print("=" * 80)

        if not os.path.exists(args.dataset):
            print(f"Error: Dataset directory not found: {args.dataset}")
            print("Please run step 1 first to prepare the dataset")
            sys.exit(1)

        process_dataset(
            model_name=args.model,
            dataset_dir=args.dataset,
            output_dir=args.output,
            device=args.device,
            pretrained=True,
            img_size=args.img_size,
            max_images=args.max_images,
        )
        print(f"\n✓ Feature extraction complete!")

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)

    if args.step in ['2', 'all']:
        output_path = Path(args.output) / args.model
        print(f"\nFeatures saved to: {output_path}")
        print(f"Feature format: [L, N, C] numpy arrays")
        print(f"  L = number of transformer layers")
        print(f"  N = number of tokens (excluding CLS token)")
        print(f"  C = feature dimension")


if __name__ == "__main__":
    main()
