#!/usr/bin/env python3
"""
Script to extract feature maps from all layers of a ViT model for ImageNet images.
"""

import os
import sys
import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class FeatureExtractor:
    """Extract features from all transformer blocks of a ViT model."""

    def __init__(self, model_name, pretrained=True, device='cuda', img_size=None, model_kwargs=None):
        """
        Initialize the feature extractor.

        Args:
            model_name: Name of the timm model (e.g., 'vit_base_patch16_224')
            pretrained: Whether to load pretrained weights
            device: Device to run the model on
            img_size: Optional square input resolution override (e.g., 224). Useful for models
                      whose default pretrained config uses a larger resolution (e.g., DINOv2 518).
            model_kwargs: Optional dict of extra kwargs forwarded to timm.create_model
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size
        self.model_kwargs = model_kwargs or {}

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        if self.img_size is not None:
            print(f"Input image size override: {self.img_size}x{self.img_size}")

        # Load model
        create_kwargs = dict(self.model_kwargs)
        if self.img_size is not None:
            # Some models (e.g., DINOv2) assert on the default training resolution.
            # Passing img_size here forces the patch grid / pos-embed shape to match.
            create_kwargs.setdefault('img_size', self.img_size)
        self.model = timm.create_model(model_name, pretrained=pretrained, **create_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Detect model type
        # CaiT models have SA blocks that don't include CLS token
        self.is_cait = 'cait' in model_name.lower()
        # Swin Transformer has hierarchical 4-stage architecture
        self.is_swin = 'swin' in model_name.lower()

        # Setup data transform
        config = resolve_data_config({}, model=self.model)
        if self.img_size is not None:
            # Keep the model's mean/std/etc but force the spatial resolution.
            config['input_size'] = (3, self.img_size, self.img_size)
        self.transform = create_transform(**config)

        # Storage for features
        self.features = []
        self.hooks = []

        # Register hooks
        self._register_hooks()

        print(f"Model loaded successfully")
        if self.is_swin:
            print(f"Model type: Swin Transformer (4-stage hierarchical)")
        elif self.is_cait:
            print(f"Model type: CaiT")
        else:
            print(f"Model type: Standard ViT")
        print(f"Number of feature extraction points: {len(self.hooks)}")

    def _register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs."""

        def hook_fn(module, input, output):
            """Hook function to capture layer output."""
            if isinstance(output, torch.Tensor):
                if self.is_swin:
                    # Swin Transformer outputs features in [B, H, W, C] format
                    # Reshape to [B, H*W, C] to match ViT format
                    B, H, W, C = output.shape
                    features = output.view(B, H * W, C).detach().cpu()
                elif self.is_cait:
                    # CaiT SA blocks output only spatial tokens (no CLS token)
                    # Shape: [B, 196, C] for 224x224 images
                    features = output.detach().cpu()
                else:
                    # Standard ViT includes CLS token as first token
                    # Shape: [B, 197, C] for 224x224 images (1 CLS + 196 spatial)
                    # Remove CLS token (index 0) to get only spatial tokens
                    features = output[:, 1:, :].detach().cpu()
                self.features.append(features)

        # Register hooks based on model architecture
        if self.is_swin:
            # Swin Transformer: hook into the 4 stages
            if hasattr(self.model, 'layers'):
                for i, layer in enumerate(self.model.layers):
                    hook = layer.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
                    print(f"Registered hook on stage {i}")
            else:
                print("Warning: Could not find 'layers' attribute in Swin model")
                print("Model structure:", self.model)
                raise ValueError("Unsupported Swin model structure")
        elif hasattr(self.model, 'blocks'):
            # ViT/CaiT: hook into transformer blocks
            for i, block in enumerate(self.model.blocks):
                hook = block.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                print(f"Registered hook on block {i}")
        else:
            print("Warning: Could not find 'blocks' or 'layers' attribute in model")
            print("Model structure:", self.model)
            raise ValueError("Unsupported model structure")

    def extract_features(self, image_path):
        """
        Extract features from all layers for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of shape [L, N, C] where:
            - L is the number of layers
            - N is the number of tokens (excluding CLS)
            - C is the feature dimension
        """
        # Clear previous features
        self.features = []

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Stack features from all layers
        # features: list of [1, N, C] tensors
        if len(self.features) == 0:
            print("Warning: No features were captured")
            return None

        if self.is_swin:
            # For Swin: stages have different spatial dimensions
            # Return as numpy object array to preserve variable sizes
            # Each stage: [N_i, C_i] where N_i and C_i vary by stage
            feature_list = [f.squeeze(0).numpy() for f in self.features]
            # Create object array with shape (L,) where each element is [N_i, C_i]
            feature_array = np.empty(len(feature_list), dtype=object)
            for i, f in enumerate(feature_list):
                feature_array[i] = f
            return feature_array
        else:
            # For ViT/CaiT: all layers have same N, can stack normally
            # Stack to [L, 1, N, C] then squeeze to [L, N, C]
            feature_array = torch.stack(self.features, dim=0).squeeze(1).numpy()
            return feature_array

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def process_dataset(model_name, dataset_dir, output_dir, device='cuda', pretrained=True, img_size=None, model_kwargs=None, max_images=None):
    """
    Process all images in the dataset and save feature maps.

    Args:
        model_name: Name of the timm model
        dataset_dir: Directory containing the 1000 validation images
        output_dir: Base directory to save features
        device: Device to run the model on
        pretrained: Whether to load pretrained weights
        img_size: Optional square input resolution override
        model_kwargs: Optional dict of extra kwargs forwarded to timm.create_model
        max_images: Optional cap on number of images processed (useful for quick smoke tests)
    """
    # Create output directory with model name
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_path}")

    # Initialize feature extractor
    extractor = FeatureExtractor(
        model_name,
        pretrained=pretrained,
        device=device,
        img_size=img_size,
        model_kwargs=model_kwargs,
    )

    # Get all images from dataset
    dataset_path = Path(dataset_dir)
    image_paths = []

    # Collect all images from class subdirectories
    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            for img_path in sorted(class_dir.glob('*.JPEG')) + sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.png')):
                image_paths.append(img_path)

    print(f"Found {len(image_paths)} images to process")
    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"Limiting to first {len(image_paths)} images (max_images={max_images})")

    saved = 0
    last_features = None

    # Process each image
    for img_path in tqdm(image_paths, desc="Extracting features"):
        # Extract features
        features = extractor.extract_features(img_path)

        if features is None:
            print(f"Failed to extract features for {img_path}")
            continue

        # Create output filename
        # Use class_name_imagename.npy format
        class_name = img_path.parent.name
        image_name = img_path.stem
        output_filename = f"{class_name}_{image_name}.npy"
        output_file = output_path / output_filename

        # Save features
        np.save(output_file, features)
        saved += 1
        last_features = features

    print(f"\nFeature extraction complete!")
    print(f"Saved {saved} feature files to {output_path}")

    if last_features is None:
        print("Warning: No features were saved (0 images processed or all failed).")
    elif 'swin' in model_name.lower():
        print(f"Feature format: Object array with L={last_features.shape[0]} stages")
        print(f"Each stage has shape [N_i, C_i] with variable dimensions:")
        for i, stage_feat in enumerate(last_features):
            print(f"  Stage {i}: N={stage_feat.shape[0]}, C={stage_feat.shape[1]}")
    else:
        print(
            f"Feature shape: [L, N, C] where L={last_features.shape[0]}, N={last_features.shape[1]}, C={last_features.shape[2]}")

    # Cleanup
    extractor.remove_hooks()


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"  # ViT-Tiny pretrained on ImageNet-21k, finetuned on ImageNet-1k
    DATASET_DIR = "data/1000_val"
    OUTPUT_DIR = "Output/vit_tiny_patch16_224_21k/features"

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found: {DATASET_DIR}")
        print("Please run prepare_dataset.py first")
        sys.exit(1)

    # Set offline mode if needed
    os.environ['HF_HUB_OFFLINE'] = '1'

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")
    print()

    # Process dataset
    process_dataset(
        model_name=MODEL_NAME,
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
        pretrained=True
    )
