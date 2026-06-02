#!/usr/bin/env python3
"""
Script to prepare test dataset by selecting the first image from each class.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def prepare_test_dataset(source_dir, target_dir):
    """
    Select the first image from each class in ImageNet validation set.

    Args:
        source_dir: Path to ImageNet validation directory
        target_dir: Path to save the new dataset
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Get all class directories
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])

    print(f"Found {len(class_dirs)} classes")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name

        # Get all images in this class directory (sorted)
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpeg', '.jpg', '.png']])

        if not images:
            print(f"Warning: No images found in {class_name}")
            continue

        # Select the first image
        first_image = images[0]

        # Create target class directory
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy the image
        target_image_path = target_class_dir / first_image.name
        shutil.copy2(first_image, target_image_path)

        copied_count += 1

    print(f"\nSuccessfully copied {copied_count} images to {target_dir}")
    print(f"Dataset structure: {target_dir}/class_name/image.JPEG")


if __name__ == "__main__":
    source_dir = "data/ILSVRC/val"
    target_dir = "data/1000_val"

    prepare_test_dataset(source_dir, target_dir)
