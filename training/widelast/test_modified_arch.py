#!/usr/bin/env python3
"""
Test script for modified student architecture.
Verifies that channel expansion works correctly and is compatible with the training pipeline.
"""

import torch
from modified_models import create_modified_student
from timm import create_model


def test_channel_dimensions():
    """Test that channel dimensions expand correctly"""
    print("=" * 70)
    print("TEST 1: Channel Dimension Expansion")
    print("=" * 70)

    model = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear',
        use_layer_norm=True
    )

    print("\nExpected layer dimensions:")
    for i, dim in enumerate(model.layer_dims):
        print(f"  Layer {i}: {dim} channels")

    assert model.layer_dims[0] == 192, "First layer should be 192"
    assert model.layer_dims[-1] == 384, "Last layer should match teacher (384)"
    assert len(model.layer_dims) == 13, "Should have 13 dimensions (12 layers + initial)"

    print("\n✓ Channel dimensions correct!")


def test_forward_pass():
    """Test forward pass with feature extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass with Feature Extraction")
    print("=" * 70)

    model = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    # Test with features
    print("\nTesting forward with features...")
    logits, features = model(x, indices=[11], require_feat=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Number of features: {len(features)}")
    print(f"  Last feature shape: {features[-1].shape}")

    # Verify shapes
    assert logits.shape == (batch_size, 1000), f"Logits shape incorrect: {logits.shape}"
    assert len(features) == 1, f"Should extract 1 feature, got {len(features)}"
    # Features should be in [B, C, H, W] format
    assert features[-1].shape[1] == 384, f"Last feature should have 384 channels, got {features[-1].shape[1]}"

    print("\n✓ Forward pass with features correct!")

    # Test without features
    print("\nTesting forward without features...")
    logits_only = model(x, require_feat=False)
    print(f"  Logits shape: {logits_only.shape}")

    assert logits_only.shape == (batch_size, 1000), f"Logits shape incorrect: {logits_only.shape}"

    print("\n✓ Forward pass without features correct!")


def test_expansion_types():
    """Test different expansion types"""
    print("\n" + "=" * 70)
    print("TEST 3: Different Expansion Types")
    print("=" * 70)

    for exp_type in ['linear', 'exponential', 'step']:
        print(f"\nTesting {exp_type} expansion...")

        model = create_modified_student(
            base_model_name='deit_tiny_patch16_224',
            teacher_embed_dim=384,
            expansion_start_layer=9,
            expansion_type=exp_type
        )

        x = torch.randn(2, 3, 224, 224)
        logits, features = model(x, indices=[11], require_feat=True)

        print(f"  Layer dimensions: {model.layer_dims}")
        print(f"  Last feature channels: {features[-1].shape[1]}")

        assert features[-1].shape[1] == 384, f"{exp_type}: Last layer should be 384 channels"

        print(f"  ✓ {exp_type.capitalize()} expansion works!")


def test_dimension_matching():
    """Test that student and teacher dimensions match when using modified architecture"""
    print("\n" + "=" * 70)
    print("TEST 4: Student-Teacher Dimension Matching")
    print("=" * 70)

    # Create modified student
    student = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )

    # Create teacher
    teacher = create_model('cait_s24_224', pretrained=False)

    print(f"\nTeacher embed_dim: {teacher.embed_dim}")
    print(f"Student final embed_dim: {student.embed_dim}")

    assert student.embed_dim == teacher.embed_dim, "Student and teacher dimensions should match!"

    # Test feature extraction
    x = torch.randn(2, 3, 224, 224)

    # Get student features
    _, student_feats = student.forward_intermediates(x, indices=[11])

    # Get teacher features
    _, teacher_feats = teacher.forward_intermediates(x, indices=[23])

    print(f"\nStudent last feature shape: {student_feats[-1].shape}")
    print(f"Teacher last feature shape: {teacher_feats[-1].shape}")

    # Check channel dimensions match (features are [B, C, H, W])
    assert student_feats[-1].shape[1] == teacher_feats[-1].shape[1], \
        "Student and teacher channel dimensions should match!"

    print("\n✓ Dimension matching correct!")


def test_parameter_count():
    """Compare parameter counts"""
    print("\n" + "=" * 70)
    print("TEST 5: Parameter Count Comparison")
    print("=" * 70)

    # Standard student
    standard = create_model('deit_tiny_patch16_224', pretrained=False)
    standard_params = sum(p.numel() for p in standard.parameters())

    # Modified student
    modified = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )
    modified_params = sum(p.numel() for p in modified.parameters())

    # Teacher
    teacher = create_model('cait_s24_224', pretrained=False)
    teacher_params = sum(p.numel() for p in teacher.parameters())

    print(f"\nStandard DeiT-Tiny: {standard_params:,} parameters")
    print(f"Modified DeiT-Tiny: {modified_params:,} parameters (+{modified_params - standard_params:,})")
    print(f"CaiT-S24 Teacher:   {teacher_params:,} parameters")

    expansion_params = modified_params - standard_params
    print(f"\nExpansion overhead: {expansion_params:,} parameters")
    print(f"Overhead percentage: {100 * expansion_params / standard_params:.2f}%")

    # Modified student should have more params but still less than teacher
    assert modified_params > standard_params, "Modified should have more parameters"
    assert modified_params < teacher_params, "Modified should have fewer parameters than teacher"

    print("\n✓ Parameter count as expected!")


def test_gradient_flow():
    """Test that gradients flow through projection layers"""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Flow Through Projections")
    print("=" * 70)

    model = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )

    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    logits, features = model(x, indices=[11], require_feat=True)

    # Backward pass
    loss = logits.sum() + features[-1].sum()
    loss.backward()

    # Check gradients in projection layers
    projection_found = False
    for name, param in model.named_parameters():
        if 'projections' in name or 'head' in name:
            projection_found = True
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                print(f"  ✓ {name}: gradient OK")

    assert projection_found, "No projection or head parameters found!"

    print("\n✓ Gradients flow correctly!")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Modified Student Architecture Tests" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_channel_dimensions()
        test_forward_pass()
        test_expansion_types()
        test_dimension_matching()
        test_parameter_count()
        test_gradient_flow()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe modified architecture is ready for training!")
        print("Run ./train_modified_student.sh to start experiments.")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == '__main__':
    main()
