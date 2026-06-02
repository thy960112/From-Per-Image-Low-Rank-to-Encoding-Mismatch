"""
Custom Vision Transformer implementation with heterogeneous block dimensions.

This module provides ViT building blocks where each transformer block can have
different embed_dim and num_heads, enabling gradual channel expansion experiments.

Key differences from standard ViT:
- Each block can have different embed_dim and num_heads
- Projection shortcuts added for residual connections when dimensions change
- MLP hidden size adjusted per layer (4× embed_dim)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import DropPath


class CustomAttention(nn.Module):
    """
    Multi-head attention module with configurable dimensions.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads (must divide dim evenly)
        qkv_bias: Whether to use bias in qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by num_heads {num_heads}'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CustomBlock(nn.Module):
    """
    Transformer block with configurable dimensions and optional projection shortcut.

    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio (typically 4.0)
        qkv_bias: Whether to use bias in attention qkv
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
        act_layer: Activation layer
        norm_layer: Normalization layer
        proj_in_dim: If specified, add input projection from proj_in_dim to dim
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        proj_in_dim: Optional[int] = None
    ):
        super().__init__()

        self.dim = dim
        self.proj_in_dim = proj_in_dim

        # Input projection if dimensions change
        if proj_in_dim is not None and proj_in_dim != dim:
            self.proj_in = nn.Linear(proj_in_dim, dim, bias=False)
            self.proj_norm = norm_layer(dim)
        else:
            self.proj_in = None
            self.proj_norm = None

        # Standard transformer block components
        self.norm1 = norm_layer(dim)
        self.attn = CustomAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional input projection for dimension change.

        Args:
            x: Input tensor [B, N, C_in] where C_in = proj_in_dim or dim

        Returns:
            Output tensor [B, N, C_out] where C_out = dim
        """
        # Project input if dimensions change
        if self.proj_in is not None:
            x = self.proj_in(x)
            x = self.proj_norm(x)

        # Standard transformer block with residual connections
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CustomVisionTransformer(nn.Module):
    """
    Vision Transformer with heterogeneous block dimensions.

    Supports gradual channel expansion by allowing different embed_dim and num_heads
    for each transformer block. Automatically adds projection shortcuts when dimensions change.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        layer_dims: List of embedding dimensions for each layer (length = num_layers)
        layer_heads: List of number of heads for each layer (length = num_layers)
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Use bias in attention qkv
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        layer_dims: List[int] = None,
        layer_heads: List[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = None
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Validate inputs
        assert layer_dims is not None and len(layer_dims) > 0, "layer_dims must be provided"
        assert layer_heads is not None and len(layer_heads) == len(layer_dims), \
            "layer_heads must match length of layer_dims"

        # Validate embed_dim % num_heads == 0 for each layer
        for i, (dim, heads) in enumerate(zip(layer_dims, layer_heads)):
            assert dim % heads == 0, \
                f"Layer {i}: embed_dim {dim} must be divisible by num_heads {heads}"

        self.num_classes = num_classes
        self.num_features = layer_dims[-1]  # Final layer dimension
        self.embed_dim = layer_dims[0]  # Initial embedding dimension
        self.layer_dims = layer_dims
        self.layer_heads = layer_heads
        self.num_layers = len(layer_dims)

        # Patch embedding - always starts with first layer's dimension
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embeddings use initial dimension
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # Build transformer blocks with varying dimensions
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            dim = layer_dims[i]
            num_heads = layer_heads[i]

            # Determine if we need input projection
            proj_in_dim = layer_dims[i-1] if i > 0 else None

            block = CustomBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                proj_in_dim=proj_in_dim
            )
            self.blocks.append(block)

        # Final norm and classification head use final dimension
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]

        # Add position embeddings (may need projection if pos_embed dim doesn't match)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Classification head."""
        # Extract class token
        x = x[:, 0]  # [B, num_features]

        if pre_logits:
            return x

        return self.head(x)

    def forward(self, x: torch.Tensor, indices: Optional[List[int]] = None, require_feat: bool = True):
        """
        Forward pass compatible with customized_forward.py interface.

        Args:
            x: Input images [B, C, H, W]
            indices: Layer indices to extract intermediate features
            require_feat: Whether to return intermediate features

        Returns:
            If require_feat: (logits, block_outs)
            Else: logits
        """
        if require_feat:
            x, block_outs = self.forward_intermediates(x, indices)
            logits = self.forward_head(x)
            return logits, block_outs
        else:
            x = self.forward_features(x)
            logits = self.forward_head(x)
            return logits

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[List[int]] = None,
        norm: bool = False,
        stop_early: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with intermediate feature extraction.

        Compatible with timm and customized_forward.py interfaces.

        Args:
            x: Input images [B, C, H, W]
            indices: Layer indices to extract (e.g., [11] for last layer)
            norm: Whether to apply norm before extraction (not used, kept for compatibility)
            stop_early: Whether to stop after last index (not used, kept for compatibility)

        Returns:
            final_features: Final features [B, num_patches+1, num_features]
            block_outs: List of intermediate features in [B, C, H, W] format
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        num_patches = x.shape[1]
        H = W = int(num_patches ** 0.5)

        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Collect intermediate outputs
        block_outs = []
        indices_set = set(indices) if indices is not None else set(range(self.num_layers))

        for i, block in enumerate(self.blocks):
            x = block(x)

            # Extract intermediate features if requested
            if i in indices_set:
                # Remove class token and reshape to [B, C, H, W]
                patch_tokens = x[:, 1:, :]  # [B, num_patches, dim]
                features = patch_tokens.permute(0, 2, 1).reshape(B, self.layer_dims[i], H, W)
                block_outs.append(features)

        # Final norm
        final_features = self.norm(x)

        return final_features, block_outs


def create_custom_deit_tiny(
    layer_dims: List[int],
    layer_heads: List[int],
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.,
    **kwargs
) -> CustomVisionTransformer:
    """
    Factory function to create custom DeiT-Tiny with specified dimensions.

    Args:
        layer_dims: Embedding dimension for each layer (length = 12)
        layer_heads: Number of heads for each layer (length = 12)
        num_classes: Number of output classes
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate

    Returns:
        CustomVisionTransformer model
    """
    model = CustomVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        layer_dims=layer_dims,
        layer_heads=layer_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        **kwargs
    )

    return model


# ============================================================================
# Factory Functions for Experimental Expansion Schedules
# ============================================================================

def create_deit_tiny_heads_linear(
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.
) -> CustomVisionTransformer:
    """
    Plan A - Heads-change schedule (Linear expansion):
    [192×3h]×9 → 256×4h → 320×5h → 384×6h

    Keeps head_dim ≈ 64 constant, increases number of heads as width grows.
    Matches standard ViT scaling practice (d_model = heads × head_dim).

    Total layers: 12
    Final dimension: 384 (matches CaiT-S24 teacher)
    """
    layer_dims = [192] * 9 + [256, 320, 384]
    layer_heads = [3] * 9 + [4, 5, 6]

    return create_custom_deit_tiny(
        layer_dims=layer_dims,
        layer_heads=layer_heads,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


def create_deit_tiny_fixed_linear(
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.
) -> CustomVisionTransformer:
    """
    Plan B - Fixed-3-heads schedule (Linear expansion):
    [192×3h]×9 → 240×3h → 312×3h → 384×3h

    Keeps 3 heads fixed, grows head_dim (64 → 80 → 104 → 128).
    Uses multiples of 3 to maintain divisibility constraint.

    Total layers: 12
    Final dimension: 384 (matches CaiT-S24 teacher)
    """
    layer_dims = [192] * 9 + [240, 312, 384]
    layer_heads = [3] * 12

    return create_custom_deit_tiny(
        layer_dims=layer_dims,
        layer_heads=layer_heads,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


def create_deit_tiny_heads_step(
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.
) -> CustomVisionTransformer:
    """
    Plan A - Heads-change schedule (Step expansion):
    [192×3h]×11 → 384×6h

    Parameter-efficient baseline: expands only at the final layer.
    Keeps head_dim = 64 constant.

    Total layers: 12
    Final dimension: 384 (matches CaiT-S24 teacher)
    """
    layer_dims = [192] * 11 + [384]
    layer_heads = [3] * 11 + [6]

    return create_custom_deit_tiny(
        layer_dims=layer_dims,
        layer_heads=layer_heads,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


def create_deit_tiny_fixed_step(
    num_classes: int = 1000,
    drop_rate: float = 0.,
    drop_path_rate: float = 0.
) -> CustomVisionTransformer:
    """
    Plan B - Fixed-3-heads schedule (Step expansion):
    [192×3h]×11 → 384×3h

    Parameter-efficient baseline: expands only at the final layer.
    Keeps 3 heads fixed, head_dim grows to 128.

    Total layers: 12
    Final dimension: 384 (matches CaiT-S24 teacher)
    """
    layer_dims = [192] * 11 + [384]
    layer_heads = [3] * 12

    return create_custom_deit_tiny(
        layer_dims=layer_dims,
        layer_heads=layer_heads,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


if __name__ == '__main__':
    # Test custom architecture
    print("Testing Custom Vision Transformer with heterogeneous dimensions...\n")

    # Test 1: Heads-change schedule (Linear)
    # [192×3h]×9 → 256×4h → 320×5h → 384×6h = 12 layers total
    print("=" * 60)
    print("Test 1: Heads-change schedule - Linear (Plan A)")
    print("=" * 60)
    layer_dims = [192] * 9 + [256, 320, 384]
    layer_heads = [3] * 9 + [4, 5, 6]

    model = create_custom_deit_tiny(layer_dims, layer_heads)
    print(f"Layer dimensions: {layer_dims}")
    print(f"Layer heads: {layer_heads}")
    print(f"Total layers: {len(layer_dims)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x, indices=[11], require_feat=True)
    print(f"\nForward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Number of features: {len(features)}")
    print(f"  Last feature shape: {features[-1].shape} (expected: [2, 384, 14, 14])")

    # Test 2: Fixed-3-heads schedule (Linear)
    # [192×3h]×9 → 240×3h → 312×3h → 384×3h = 12 layers total
    print("\n" + "=" * 60)
    print("Test 2: Fixed-3-heads schedule - Linear (Plan B)")
    print("=" * 60)
    layer_dims = [192] * 9 + [240, 312, 384]
    layer_heads = [3] * 12

    model = create_custom_deit_tiny(layer_dims, layer_heads)
    print(f"Layer dimensions: {layer_dims}")
    print(f"Layer heads: {layer_heads}")
    print(f"Total layers: {len(layer_dims)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits, features = model(x, indices=[11], require_feat=True)
    print(f"\nForward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Last feature shape: {features[-1].shape} (expected: [2, 384, 14, 14])")

    # Test 3: Step expansion - Heads-change
    # [192×3h]×11 → 384×6h = 12 layers total
    print("\n" + "=" * 60)
    print("Test 3: Step expansion - Heads-change (Plan A)")
    print("=" * 60)
    layer_dims = [192] * 11 + [384]
    layer_heads = [3] * 11 + [6]

    model = create_custom_deit_tiny(layer_dims, layer_heads)
    print(f"Layer dimensions: {layer_dims}")
    print(f"Layer heads: {layer_heads}")
    print(f"Total layers: {len(layer_dims)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits, features = model(x, indices=[11], require_feat=True)
    print(f"\nForward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Last feature shape: {features[-1].shape} (expected: [2, 384, 14, 14])")

    # Test 4: Step expansion - Fixed-3-heads
    # [192×3h]×11 → 384×3h = 12 layers total
    print("\n" + "=" * 60)
    print("Test 4: Step expansion - Fixed-3-heads (Plan B)")
    print("=" * 60)
    layer_dims = [192] * 11 + [384]
    layer_heads = [3] * 12

    model = create_custom_deit_tiny(layer_dims, layer_heads)
    print(f"Layer dimensions: {layer_dims}")
    print(f"Layer heads: {layer_heads}")
    print(f"Total layers: {len(layer_dims)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits, features = model(x, indices=[11], require_feat=True)
    print(f"\nForward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Last feature shape: {features[-1].shape} (expected: [2, 384, 14, 14])")

    print("\n✓ All tests passed!")
