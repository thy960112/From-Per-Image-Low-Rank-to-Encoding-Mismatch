"""
Modified student architectures with channel expansion for testing hypothesis:
"ViT models may be redundant in parameter space, but not redundant in channel space"

This module provides modified student models with gradually expanding channel dimensions
to match teacher's channel capacity in later layers.

UPDATE (2025): The original ChannelExpansionWrapper approach is DEPRECATED due to
MHA dimension constraints (embed_dim % num_heads must equal 0). Use the new custom
ViT implementations from custom_vit.py instead, which explicitly modify the DeiT-Tiny
architecture with proper dimension handling and projection shortcuts.
"""

import torch
import torch.nn as nn
from timm import create_model
from typing import List, Optional, Tuple
import numpy as np

# Import new custom ViT implementations
from custom_vit import (
    create_deit_tiny_heads_linear,
    create_deit_tiny_fixed_linear,
    create_deit_tiny_heads_step,
    create_deit_tiny_fixed_step,
    CustomVisionTransformer
)


class ChannelExpansionWrapper(nn.Module):
    """
    Wraps a student model and adds channel expansion layers to gradually increase
    embedding dimensions in later transformer blocks.

    The channel dimensions increase gradually following the entropy shape pattern,
    with the final layer matching the teacher's embedding dimension.
    """

    def __init__(
        self,
        base_model: nn.Module,
        base_embed_dim: int,
        target_embed_dim: int,
        expansion_start_layer: int,
        total_layers: int,
        expansion_type: str = 'linear',
        use_layer_norm: bool = True
    ):
        """
        Args:
            base_model: The base student model (e.g., DeiT-Tiny)
            base_embed_dim: Original embedding dimension (e.g., 192 for DeiT-Tiny)
            target_embed_dim: Target embedding dimension to match teacher (e.g., 384 for CaiT-S24)
            expansion_start_layer: Layer index to start expansion (e.g., 9 for last 3 layers)
            total_layers: Total number of transformer blocks
            expansion_type: 'linear' or 'exponential' expansion pattern
            use_layer_norm: Whether to add LayerNorm after projection
        """
        super().__init__()
        self.base_model = base_model
        self.base_embed_dim = base_embed_dim
        self.target_embed_dim = target_embed_dim
        self.expansion_start_layer = expansion_start_layer
        self.total_layers = total_layers
        self.expansion_type = expansion_type
        self.use_layer_norm = use_layer_norm

        # Calculate channel dimensions for each layer
        self.layer_dims = self._calculate_layer_dimensions()

        # Create projection layers
        # Each layer after expansion_start_layer gets projected from base_embed_dim to target dim
        self.projections = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict() if use_layer_norm else None

        for layer_idx in range(expansion_start_layer, total_layers):
            out_dim = self.layer_dims[layer_idx + 1]

            if out_dim != self.base_embed_dim:
                # Project from base embedding dim to target dim for this layer
                self.projections[str(layer_idx)] = nn.Linear(self.base_embed_dim, out_dim, bias=False)

                if use_layer_norm:
                    self.layer_norms[str(layer_idx)] = nn.LayerNorm(out_dim)

        # Create a new classification head for the expanded dimension
        # The base model's head expects base_embed_dim, but we output target_embed_dim
        num_classes = base_model.num_classes
        self.head = nn.Linear(target_embed_dim, num_classes)

        print(f"Channel expansion architecture:")
        print(f"  Base dim: {base_embed_dim}, Target dim: {target_embed_dim}")
        print(f"  Expansion starts at layer: {expansion_start_layer}")
        print(f"  Layer dimensions: {self.layer_dims}")
        print(f"  Total parameters added: {self._count_expansion_params():,}")

    def _calculate_layer_dimensions(self) -> List[int]:
        """
        Calculate channel dimension for each layer based on expansion pattern.
        Follows entropy curve shape: gradual increase to match teacher's encoding capacity.
        """
        dims = [self.base_embed_dim] * (self.expansion_start_layer + 1)

        num_expansion_layers = self.total_layers - self.expansion_start_layer

        if self.expansion_type == 'linear':
            # Linear interpolation
            for i in range(num_expansion_layers):
                progress = (i + 1) / num_expansion_layers
                dim = int(self.base_embed_dim + (self.target_embed_dim - self.base_embed_dim) * progress)
                dims.append(dim)

        elif self.expansion_type == 'exponential':
            # Exponential growth (mimics entropy curve)
            for i in range(num_expansion_layers):
                progress = (i + 1) / num_expansion_layers
                # Use exponential interpolation
                ratio = self.target_embed_dim / self.base_embed_dim
                dim = int(self.base_embed_dim * (ratio ** progress))
                dims.append(dim)

        elif self.expansion_type == 'step':
            # Step function: expand only at the last layer
            for i in range(num_expansion_layers - 1):
                dims.append(self.base_embed_dim)
            dims.append(self.target_embed_dim)

        else:
            raise ValueError(f"Unknown expansion_type: {self.expansion_type}")

        return dims

    def _count_expansion_params(self) -> int:
        """Count additional parameters from projection layers and new head"""
        total = 0
        for proj in self.projections.values():
            total += sum(p.numel() for p in proj.parameters())
        if self.layer_norms is not None:
            for ln in self.layer_norms.values():
                total += sum(p.numel() for p in ln.parameters())
        # Add classification head parameters
        total += sum(p.numel() for p in self.head.parameters())
        return total

    def forward_intermediates(self, x, indices: Optional[List[int]] = None):
        """
        Forward pass with intermediate feature extraction and channel expansion.

        Args:
            x: Input tensor [B, 3, H, W]
            indices: Layer indices to extract (after expansion)

        Returns:
            final_output: Final feature tensor
            block_outs: List of intermediate features at specified indices (after expansion)
        """
        # Get all intermediate features from base model
        # For ViT models, these are in [B, C, H, W] format where C = base_embed_dim
        _, base_block_outs = self.base_model.forward_intermediates(x)

        # Apply channel expansion to intermediate features
        # Each layer output is projected independently from base_embed_dim to target dim
        expanded_block_outs = []

        for layer_idx in range(len(base_block_outs)):
            # Get features from base model (always base_embed_dim channels)
            current_features = base_block_outs[layer_idx]

            # Apply projection if this layer should be expanded
            if str(layer_idx) in self.projections:
                # Permute to [B, H, W, C] for linear layer
                B, C, H, W = current_features.shape
                current_features = current_features.permute(0, 2, 3, 1)  # [B, H, W, C]

                # Apply projection from base_embed_dim to target dimension
                current_features = self.projections[str(layer_idx)](current_features)

                if self.use_layer_norm and str(layer_idx) in self.layer_norms:
                    current_features = self.layer_norms[str(layer_idx)](current_features)

                # Permute back to [B, C, H, W]
                current_features = current_features.permute(0, 3, 1, 2)  # [B, C', H, W]

            expanded_block_outs.append(current_features)

        # Select specified indices
        if indices is not None:
            selected_outs = [expanded_block_outs[i] for i in indices if i < len(expanded_block_outs)]
        else:
            selected_outs = expanded_block_outs

        # Final output uses the last expanded features
        final_output = expanded_block_outs[-1] if expanded_block_outs else None

        return final_output, selected_outs

    def forward_features(self, x):
        """Forward through feature extraction only"""
        return self.base_model.forward_features(x)

    def forward_head(self, x):
        """
        Forward through classification head.
        x is in [B, C, H, W] format, need to pool and classify.
        """
        # Global average pooling [B, C, H, W] -> [B, C]
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        return self.head(x)

    def forward(self, x, indices: Optional[List[int]] = None, require_feat: bool = True):
        """
        Full forward pass compatible with customized_forward.py

        Args:
            x: Input images
            indices: Layer indices to extract
            require_feat: Whether to return intermediate features

        Returns:
            If require_feat: (logits, block_outs)
            Else: logits
        """
        if require_feat:
            features, block_outs = self.forward_intermediates(x, indices)
            logits = self.forward_head(features)
            return logits, block_outs
        else:
            # Get the last layer features with expansion
            features, _ = self.forward_intermediates(x, indices=[self.total_layers - 1])
            logits = self.forward_head(features)
            return logits


class UniformWideStudent(nn.Module):
    """
    Control baseline: Student with uniform wide channels across all layers.
    This serves as a parameter-count-matched control to isolate the effect
    of late-layer channel expansion vs. simply having more parameters.

    For DeiT-Tiny (192 base dim, 3 heads), we use 204 dim (3*68) to roughly match
    the parameter count of the modified student (~6.4M parameters).
    """

    def __init__(
        self,
        model_name: str,
        uniform_embed_dim: int = 204,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pretrained: bool = False
    ):
        """
        Args:
            model_name: Base model architecture (e.g., 'deit_tiny_patch16_224')
            uniform_embed_dim: Uniform embedding dimension across all layers (must be divisible by num_heads)
            num_classes: Number of output classes
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            pretrained: Load pretrained weights (not applicable for custom embed_dim)
        """
        super().__init__()

        # Determine num_heads based on the base model
        # DeiT-Tiny has 3 heads, so we scale accordingly
        if 'tiny' in model_name.lower():
            # For DeiT-Tiny, base is 192 dim with 3 heads (64 dim per head)
            # We keep the same head dimension, so num_heads = uniform_embed_dim / 64
            assert uniform_embed_dim % 3 == 0, f"For DeiT-Tiny variants, embed_dim must be divisible by 3, got {uniform_embed_dim}"
            num_heads = 3
            assert uniform_embed_dim % num_heads == 0, f"embed_dim {uniform_embed_dim} must be divisible by num_heads {num_heads}"
        else:
            raise NotImplementedError(f"UniformWideStudent currently only supports 'tiny' variants, got {model_name}")

        # Create the model with custom embed_dim and num_heads
        self.model = create_model(
            model_name,
            embed_dim=uniform_embed_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=False  # Can't use pretrained weights with custom dimensions
        )

        self.uniform_embed_dim = uniform_embed_dim
        self.num_heads = num_heads

    def forward(self, x):
        """Standard forward pass"""
        return self.model(x)

    def forward_intermediates(
        self,
        x,
        indices: Optional[List[int]] = None,
        norm: bool = False,
        stop_early: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with intermediate outputs.
        Compatible with customized_forward.py interface.
        """
        return self.model.forward_intermediates(
            x,
            indices=indices,
            norm=norm,
            stop_early=stop_early
        )

    def get_num_layers(self) -> int:
        """Get total number of transformer blocks"""
        return len(self.model.blocks)

    def get_embed_dim(self) -> int:
        """Get embedding dimension"""
        return self.uniform_embed_dim

    def forward_head(self, x, pre_logits: bool = False):
        """
        Forward through classification head.
        Compatible with customized_forward.py interface.
        """
        return self.model.forward_head(x, pre_logits=pre_logits)

    def forward_features(self, x):
        """Forward through feature extraction only"""
        return self.model.forward_features(x)


def create_modified_student(
    base_model_name: str = 'deit_tiny_patch16_224',
    teacher_embed_dim: int = 384,
    expansion_start_layer: int = 9,
    expansion_type: str = 'linear',
    use_layer_norm: bool = True,
    num_classes: int = 1000,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    pretrained: bool = False
) -> ChannelExpansionWrapper:
    """
    Factory function to create a modified student model with channel expansion.

    Args:
        base_model_name: Name of base student model from timm
        teacher_embed_dim: Target embedding dimension (from teacher)
        expansion_start_layer: Layer to start channel expansion
        expansion_type: Type of expansion pattern ('linear', 'exponential', 'step')
        use_layer_norm: Whether to use LayerNorm after projections
        num_classes: Number of output classes
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
        pretrained: Whether to load pretrained weights for base model

    Returns:
        Modified student model with channel expansion

    Example:
        >>> model = create_modified_student(
        ...     base_model_name='deit_tiny_patch16_224',
        ...     teacher_embed_dim=384,  # CaiT-S24
        ...     expansion_start_layer=9,  # Expand in last 3 layers
        ...     expansion_type='linear'
        ... )
    """
    # Create base student model
    base_model = create_model(
        base_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )

    # Get model properties
    base_embed_dim = base_model.embed_dim

    # Count total transformer blocks
    total_layers = len(base_model.blocks)

    # Wrap with channel expansion
    modified_model = ChannelExpansionWrapper(
        base_model=base_model,
        base_embed_dim=base_embed_dim,
        target_embed_dim=teacher_embed_dim,
        expansion_start_layer=expansion_start_layer,
        total_layers=total_layers,
        expansion_type=expansion_type,
        use_layer_norm=use_layer_norm
    )

    # Copy useful attributes
    modified_model.num_classes = num_classes
    modified_model.embed_dim = teacher_embed_dim  # Final embedding dim

    return modified_model


def create_uniform_wide_student(
    base_model_name: str = 'deit_tiny_patch16_224',
    uniform_embed_dim: int = 204,
    num_classes: int = 1000,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0
) -> UniformWideStudent:
    """
    Factory function to create a uniform-wide student model.
    This serves as a parameter-matched control for the modified student experiments.

    Args:
        base_model_name: Name of base model architecture (e.g., 'deit_tiny_patch16_224')
        uniform_embed_dim: Uniform embedding dimension across all layers (default: 204 for ~6.4M params)
        num_classes: Number of output classes
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate

    Returns:
        Uniform-wide student model

    Example:
        >>> # Create a control student with uniform 204 channels (~6.4M params)
        >>> model = create_uniform_wide_student(
        ...     base_model_name='deit_tiny_patch16_224',
        ...     uniform_embed_dim=204
        ... )
    """
    model = UniformWideStudent(
        model_name=base_model_name,
        uniform_embed_dim=uniform_embed_dim,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pretrained=False
    )

    return model


# ============================================================================
# NEW: Custom Architecture Factory Functions (2025 Update)
# ============================================================================

def create_custom_architecture(
    arch_schedule: str,
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.
) -> CustomVisionTransformer:
    """
    Factory function to create custom DeiT-Tiny variants with explicit architecture modifications.

    This replaces the deprecated ChannelExpansionWrapper approach with proper explicit
    modifications to the ViT architecture, handling MHA dimension constraints correctly.

    Args:
        arch_schedule: Architecture expansion schedule, one of:
            - 'heads_linear': Plan A, linear expansion with changing heads
            - 'fixed_linear': Plan B, linear expansion with fixed 3 heads
            - 'heads_step': Plan A, step expansion with changing heads
            - 'fixed_step': Plan B, step expansion with fixed 3 heads
        num_classes: Number of output classes
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate

    Returns:
        CustomVisionTransformer with specified expansion schedule

    Example:
        >>> # Create Plan A (heads-change, linear expansion)
        >>> model = create_custom_architecture('heads_linear')
        >>> # [192×3h]×9 → 256×4h → 320×5h → 384×6h
    """
    schedule_map = {
        'heads_linear': create_deit_tiny_heads_linear,
        'fixed_linear': create_deit_tiny_fixed_linear,
        'heads_step': create_deit_tiny_heads_step,
        'fixed_step': create_deit_tiny_fixed_step,
    }

    if arch_schedule not in schedule_map:
        raise ValueError(
            f"Unknown arch_schedule: {arch_schedule}. "
            f"Must be one of {list(schedule_map.keys())}"
        )

    factory_fn = schedule_map[arch_schedule]
    model = factory_fn(
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )

    # Print architecture info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created custom architecture: {arch_schedule}")
    print(f"  Layer dimensions: {model.layer_dims}")
    print(f"  Layer heads: {model.layer_heads}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Final embedding dim: {model.num_features}")

    return model


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Custom Architectures (NEW - 2025 Update)")
    print("=" * 70)

    x = torch.randn(2, 3, 224, 224)

    # Test all 4 expansion schedules
    schedules = ['heads_linear', 'fixed_linear', 'heads_step', 'fixed_step']

    for schedule in schedules:
        print(f"\n{'='*70}")
        print(f"Testing: {schedule}")
        print(f"{'='*70}")

        model = create_custom_architecture(
            arch_schedule=schedule,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        # Test forward pass
        logits, features = model(x, indices=[11], require_feat=True)
        print(f"\n✓ Forward pass successful!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Number of features: {len(features)}")
        print(f"  Last feature shape: {features[-1].shape}")

    print("\n" + "="*70)
    print("✓ All custom architecture tests passed!")
    print("="*70)

    print("\n" + "="*70)
    print("Testing DEPRECATED ChannelExpansionWrapper (for backward compatibility)")
    print("="*70)

    model = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear',
        use_layer_norm=True
    )

    print("\nTesting forward with features:")
    logits, features = model(x, indices=[11], require_feat=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Number of feature maps: {len(features)}")
    print(f"Last feature map shape: {features[-1].shape}")

    print("\n✓ Deprecated wrapper test passed (but use custom architectures instead)!")
