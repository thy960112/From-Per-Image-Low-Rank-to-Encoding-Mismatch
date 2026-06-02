#!/usr/bin/env python3
"""
Extract stage feature maps from CNN models (ResNet / ConvNeXt / EfficientNet, incl. CLIP & SSL variants).

For each image, we save a single `.npy` file containing a NumPy *object* array of length S (stages).
Each element is a float32 array of shape [N_i, C_i] where N_i = H_i * W_i (spatial tokens).

This mirrors the Swin feature format used elsewhere in the repo (variable stage shapes).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _collect_images(dataset_dir: Path):
    image_paths = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        # Accept the same extensions as the ViT extractor.
        for img_path in sorted(class_dir.glob("*.JPEG")) + sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
            image_paths.append(img_path)
    return image_paths


def _flatten_feature_map(feat):
    """Convert a feature tensor [1, C, H, W] -> numpy [H*W, C] float32."""
    # Delay torch import for CLI flags (offline mode) before importing timm/torch.
    import torch

    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(feat)}")
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {tuple(feat.shape)}")
    if feat.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {feat.shape[0]}")

    _, c, h, w = feat.shape
    tokens = feat.squeeze(0).permute(1, 2, 0).reshape(h * w, c)
    return tokens.detach().cpu().numpy().astype(np.float32, copy=False)


def process_dataset_cnn(
    model_name: str,
    dataset_dir: str,
    output_dir: str,
    device: str = "cuda",
    pretrained: bool = True,
    offline: bool = True,
    img_size: int | None = None,
    keep_last_k: int = 4,
    max_images: int | None = None,
):
    # Set offline mode before importing timm/huggingface_hub.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_root = Path(output_dir) / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Create a features-only model; most CNNs support this and return a list of stage feature maps.
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
    model = model.to(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.eval()

    # Build transform from pretrained config; optionally override spatial resolution.
    cfg = resolve_data_config({}, model=model)
    if img_size is not None:
        cfg["input_size"] = (3, img_size, img_size)
    transform = create_transform(**cfg)

    image_paths = _collect_images(dataset_path)
    print(f"Model: {model_name}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Output: {out_root}")
    print(f"keep_last_k: {keep_last_k}")
    if img_size is not None:
        print(f"img_size override: {img_size}x{img_size}")

    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"Limiting to first {len(image_paths)} images (max_images={max_images})")

    saved = 0
    last_obj = None

    for img_path in tqdm(image_paths, desc="Extracting CNN features"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        x = transform(image).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            feats = model(x)  # list of tensors

        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            print(f"Warning: no features returned for {img_path}")
            continue

        if keep_last_k is not None and keep_last_k > 0:
            feats = feats[-keep_last_k:]

        obj = np.empty(len(feats), dtype=object)
        for i, f in enumerate(feats):
            obj[i] = _flatten_feature_map(f)

        class_name = img_path.parent.name
        image_name = img_path.stem
        out_file = out_root / f"{class_name}_{image_name}.npy"
        np.save(out_file, obj)
        saved += 1
        last_obj = obj

    print("\nFeature extraction complete!")
    print(f"Saved {saved} feature files to {out_root}")
    if last_obj is not None:
        print(f"Saved feature format: object array with {len(last_obj)} stages")
        for i, stage in enumerate(last_obj):
            print(f"  Stage {i}: N={stage.shape[0]}, C={stage.shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract stage features for CNN models (timm features_only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="timm model name (e.g., resnet50.tv_in1k)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory (1000-val)")
    parser.add_argument("--output", type=str, required=True, help="Output base directory (will create <output>/<model>/)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for extraction")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 (recommended on this machine)")
    parser.add_argument("--img-size", type=int, default=None, help="Optional square input resolution override (e.g., 224)")
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=4,
        help="Keep only the last K stage features returned by timm features_only (use 4 to drop ResNet stem output).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images (smoke tests)")
    args = parser.parse_args()

    process_dataset_cnn(
        model_name=args.model,
        dataset_dir=args.dataset,
        output_dir=args.output,
        device=args.device,
        pretrained=True,
        offline=args.offline,
        img_size=args.img_size,
        keep_last_k=args.keep_last_k,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    sys.exit(main())

