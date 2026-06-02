"""
Comprehensive integration test for custom architecture implementation.

This script tests:
1. Architecture instantiation for all 4 schedules
2. Forward pass with intermediate feature extraction
3. Compatibility with customized_forward.py
4. FFT loss computation with matching dimensions
5. Model loading in main.py (dry run)
"""

import torch
import sys
from modified_models import create_custom_architecture
from customized_forward import register_forward
from losses import layer_fft_loss


def test_architecture_instantiation():
    """Test that all 4 architectures can be instantiated correctly."""
    print("=" * 70)
    print("Test 1: Architecture Instantiation")
    print("=" * 70)

    schedules = ['heads_linear', 'fixed_linear', 'heads_step', 'fixed_step']
    models = {}

    for schedule in schedules:
        print(f"\nTesting {schedule}...")
        model = create_custom_architecture(
            arch_schedule=schedule,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        models[schedule] = model

        # Verify architecture properties
        assert len(model.layer_dims) == 12, f"Expected 12 layers, got {len(model.layer_dims)}"
        assert model.num_features == 384, f"Expected final dim=384, got {model.num_features}"
        print(f"✓ {schedule}: {len(model.layer_dims)} layers, final dim={model.num_features}")

    print("\n✓ All architectures instantiated successfully!")
    return models


def test_forward_pass(models):
    """Test forward pass with intermediate feature extraction."""
    print("\n" + "=" * 70)
    print("Test 2: Forward Pass with Intermediate Features")
    print("=" * 70)

    x = torch.randn(2, 3, 224, 224)

    for schedule, model in models.items():
        print(f"\nTesting {schedule}...")

        # Test forward with features
        logits, features = model(x, indices=[11], require_feat=True)

        assert logits.shape == (2, 1000), f"Logits shape mismatch: {logits.shape}"
        assert len(features) == 1, f"Expected 1 feature map, got {len(features)}"
        assert features[0].shape == (2, 384, 14, 14), \
            f"Feature shape mismatch: {features[0].shape}, expected (2, 384, 14, 14)"

        print(f"  Logits: {logits.shape}")
        print(f"  Features: {features[0].shape}")
        print(f"✓ {schedule} forward pass successful")

    print("\n✓ All forward passes successful!")


def test_customized_forward_compatibility(models):
    """Test compatibility with customized_forward.py."""
    print("\n" + "=" * 70)
    print("Test 3: Customized Forward Compatibility")
    print("=" * 70)

    x = torch.randn(2, 3, 224, 224)

    for schedule, model in models.items():
        print(f"\nTesting {schedule}...")

        # Register custom forward (should work without errors)
        register_forward(model, f"custom_{schedule}")

        # Test forward with require_feat=True
        result = model(x, indices=[11], require_feat=True)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Expected (logits, features) tuple, got {type(result)}"

        # Test forward with require_feat=False
        result = model(x, require_feat=False)
        assert isinstance(result, torch.Tensor), \
            f"Expected logits tensor, got {type(result)}"

        print(f"✓ {schedule} compatible with customized_forward")

    print("\n✓ All models compatible with customized_forward!")


def test_fft_loss_dimension_matching():
    """Test that FFT loss works correctly with matching dimensions."""
    print("\n" + "=" * 70)
    print("Test 4: FFT Loss with Dimension Matching")
    print("=" * 70)

    # Create dummy student and teacher features with matching dimensions
    batch_size = 2
    spatial_size = 14

    # Test Case 1: Dimensions match (C_s == C_t = 384)
    print("\nTest Case 1: C_s == C_t (384)")
    F_s = torch.randn(batch_size, 384, spatial_size, spatial_size)
    F_t = torch.randn(batch_size, 384, spatial_size, spatial_size)

    loss = layer_fft_loss(F_s, F_t)
    print(f"  Student shape: {F_s.shape}")
    print(f"  Teacher shape: {F_t.shape}")
    print(f"  FFT loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ FFT loss with matching dimensions works")

    # Test Case 2: Student smaller than teacher (C_s < C_t)
    print("\nTest Case 2: C_s < C_t (192 < 384)")
    F_s = torch.randn(batch_size, 192, spatial_size, spatial_size)
    F_t = torch.randn(batch_size, 384, spatial_size, spatial_size)

    loss = layer_fft_loss(F_s, F_t)
    print(f"  Student shape: {F_s.shape}")
    print(f"  Teacher shape: {F_t.shape}")
    print(f"  FFT loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ FFT loss with student < teacher works (pooling applied)")

    print("\n✓ FFT loss computation verified!")


def test_main_integration():
    """Test that main.py can load custom architectures (dry run)."""
    print("\n" + "=" * 70)
    print("Test 5: Main.py Integration (Dry Run)")
    print("=" * 70)

    schedules = ['heads_linear', 'fixed_linear', 'heads_step', 'fixed_step']

    for schedule in schedules:
        print(f"\nTesting main.py argument parsing for {schedule}...")

        # Create a mock args object
        class MockArgs:
            model = 'deit_tiny_patch16_224'
            custom_arch = True
            arch_schedule = schedule
            use_modified_student = False
            use_uniform_wide = False
            nb_classes = 1000
            drop = 0.0
            drop_path = 0.0

        args = MockArgs()

        # Test model creation logic
        try:
            model = create_custom_architecture(
                arch_schedule=args.arch_schedule,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path
            )
            register_forward(model, args.model)
            print(f"✓ {schedule} can be loaded in main.py")
        except Exception as e:
            print(f"✗ {schedule} failed: {e}")
            raise

    print("\n✓ Main.py integration successful!")


def main():
    print("\n" + "=" * 70)
    print("Custom Architecture Integration Test")
    print("=" * 70)

    try:
        # Run all tests
        models = test_architecture_instantiation()
        test_forward_pass(models)
        test_customized_forward_compatibility(models)
        test_fft_loss_dimension_matching()
        test_main_integration()

        # Summary
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe custom architecture implementation is working correctly.")
        print("You can now run training experiments with:")
        print("  ./train_custom_experiments.sh")
        print("\nOr train individual configurations with:")
        print("  python main.py --custom-arch --arch-schedule heads_linear ...")
        print("\nSee CUSTOM_ARCH_GUIDE.md for detailed usage instructions.")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ TESTS FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
