#!/usr/bin/env python3
"""
Script to verify the setup before running the feature extraction pipeline.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = {
        'torch': 'PyTorch',
        'timm': 'timm (PyTorch Image Models)',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'tqdm': 'tqdm'
    }

    all_installed = True

    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\nSome dependencies are missing. Install them with:")
        print("  pip install -r requirements.txt")
        return False

    # Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (Device: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ⚠ Could not check CUDA availability: {e}")

    print("\n✓ All dependencies are installed")
    return True


def check_paths():
    """Check if required directories exist."""
    print("\nChecking paths...")

    # Check source ImageNet directory
    source_dir = Path("data/ILSVRC/val")
    if source_dir.exists():
        # Count subdirectories
        class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        print(f"  ✓ Source ImageNet directory exists: {source_dir}")
        print(f"    Found {len(class_dirs)} class directories")

        # Check a sample class directory
        if class_dirs:
            sample_class = class_dirs[0]
            images = list(sample_class.glob("*.JPEG")) + list(sample_class.glob("*.jpg"))
            print(f"    Sample class '{sample_class.name}' has {len(images)} images")
    else:
        print(f"  ✗ Source ImageNet directory NOT found: {source_dir}")
        print("    Please ensure the ImageNet validation set is at this location")
        return False

    # Check if target dataset directory exists
    target_dir = Path("data/1000_val")
    if target_dir.exists():
        class_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
        print(f"  ✓ Target dataset directory exists: {target_dir}")
        print(f"    Found {len(class_dirs)} class directories")
        if len(class_dirs) == 1000:
            print(f"    ✓ Dataset appears to be complete (1000 classes)")
        else:
            print(f"    ⚠ Expected 1000 classes, found {len(class_dirs)}")
    else:
        print(f"  ⚠ Target dataset directory does not exist: {target_dir}")
        print(f"    Will be created when running step 1")

    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        print(f"  ✓ HuggingFace cache directory exists: {hf_cache}")
        # Check if there are any models cached
        cached_models = list(hf_cache.glob("models--*"))
        if cached_models:
            print(f"    Found {len(cached_models)} cached models")
        else:
            print(f"    ⚠ No models cached yet")
    else:
        print(f"  ⚠ HuggingFace cache directory does not exist: {hf_cache}")
        print(f"    Models will be downloaded on first run")

    print("\n✓ Path checks complete")
    return True


def check_timm_models():
    """Check if timm can list available models."""
    print("\nChecking timm models...")

    try:
        import timm

        # List some popular ViT models
        vit_models = [
            'vit_base_patch16_224',
            'vit_large_patch16_224',
            'vit_small_patch16_224',
        ]

        all_models = timm.list_models('vit*')
        print(f"  ✓ timm can access model list")
        print(f"    Found {len(all_models)} ViT variants")

        print(f"\n  Popular ViT models:")
        for model in vit_models:
            if model in all_models:
                print(f"    ✓ {model}")
            else:
                print(f"    ✗ {model} - Not found")

    except Exception as e:
        print(f"  ✗ Error checking timm models: {e}")
        return False

    print("\n✓ timm models check complete")
    return True


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Setup Verification for ImageNet ViT Feature Extraction")
    print("=" * 80)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("Paths", check_paths),
        ("timm Models", check_timm_models),
    ]

    results = []

    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 80)
    print("Verification Summary")
    print("=" * 80)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Prepare dataset: python main.py --step 1")
        print("  2. Extract features: python main.py --step 2 --model vit_base_patch16_224")
        print("  or run both: python main.py --step all --model vit_base_patch16_224")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
