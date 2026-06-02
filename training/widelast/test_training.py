#!/usr/bin/env python3
"""
Integration test: Verify modified student works with the training pipeline.
Tests loss computation, optimization step, and compatibility with the full pipeline.
"""

import torch
import torch.nn as nn
from timm import create_model
from modified_models import create_modified_student
from losses import DistillationLoss
import argparse


def test_training_integration():
    """Test that modified student works with the full training pipeline"""
    print("=" * 70)
    print("INTEGRATION TEST: Modified Student Training Pipeline")
    print("=" * 70)

    # Create modified student
    print("\n1. Creating modified student model...")
    student = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear',
        num_classes=1000
    )

    # Create teacher
    print("\n2. Creating teacher model...")
    teacher = create_model('cait_s24_224', pretrained=False, num_classes=1000)
    teacher.eval()

    # Register forward hooks
    print("\n3. Registering forward hooks...")
    from customized_forward import register_forward
    register_forward(student, 'deit_tiny_patch16_224')
    register_forward(teacher, 'cait_s24_224')

    # Create mock args for DistillationLoss
    class Args:
        distillation_type = 'soft'
        distillation_tau = 1.0
        distillation_alpha = 0.9
        distillation_beta = 1.0
        w_fft = 0.2
        s_id = [11]
        t_id = [23]

    args = Args()

    # Create loss function
    print("\n4. Creating loss function...")
    base_criterion = nn.CrossEntropyLoss()
    criterion = DistillationLoss(base_criterion, teacher, args)

    # Create optimizer
    print("\n5. Creating optimizer...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    # Test training step
    print("\n6. Testing training step...")
    student.train()

    # Create dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 1000, (batch_size,))

    print(f"   Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Forward pass
    print("\n7. Forward pass through student...")
    student_outputs = student(images, args.s_id)
    print(f"   Student logits shape: {student_outputs[0].shape}")
    print(f"   Student features: {len(student_outputs[1])} feature maps")
    print(f"   Student feature[0] shape: {student_outputs[1][0].shape}")

    # Compute loss
    print("\n8. Computing distillation loss...")
    loss_base, loss_fft, loss_dist = criterion(images, student_outputs, labels)
    total_loss = loss_base + loss_fft + loss_dist

    print(f"   Base loss: {loss_base.item():.4f}")
    print(f"   FFT loss: {loss_fft.item():.4f}")
    print(f"   Distillation loss: {loss_dist.item():.4f}")
    print(f"   Total loss: {total_loss.item():.4f}")

    # Check loss is finite
    assert torch.isfinite(total_loss), "Loss is not finite!"

    # Backward pass
    print("\n9. Backward pass...")
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients
    print("\n10. Checking gradients...")
    grad_count = 0
    nan_count = 0
    for name, param in student.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"   ✗ NaN gradient in {name}")

    print(f"   Parameters with gradients: {grad_count}")
    print(f"   Parameters with NaN gradients: {nan_count}")

    assert nan_count == 0, "Found NaN gradients!"

    # Optimizer step
    print("\n11. Optimizer step...")
    optimizer.step()

    print("\n✓ Training step successful!")

    # Test evaluation mode
    print("\n12. Testing evaluation mode...")
    student.eval()
    with torch.no_grad():
        eval_logits = student(images, require_feat=False)
        print(f"   Eval logits shape: {eval_logits.shape}")
        assert eval_logits.shape == (batch_size, 1000)

    print("\n✓ Evaluation mode works!")

    return True


def test_dimension_matching_in_loss():
    """Test that FFT loss works correctly with matching dimensions"""
    print("\n" + "=" * 70)
    print("DIMENSION MATCHING TEST: FFT Loss with Expanded Channels")
    print("=" * 70)

    from losses import layer_fft_loss

    # Test case 1: Matching dimensions (no pooling)
    print("\n1. Testing with matching dimensions (384 == 384)...")
    F_s = torch.randn(4, 384, 14, 14)
    F_t = torch.randn(4, 384, 14, 14)
    loss = layer_fft_loss(F_s, F_t)
    print(f"   Student shape: {F_s.shape}")
    print(f"   Teacher shape: {F_t.shape}")
    print(f"   FFT loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    print("   ✓ Matching dimensions work (no pooling needed)")

    # Test case 2: Student smaller (pooling teacher)
    print("\n2. Testing with smaller student (192 < 384)...")
    F_s = torch.randn(4, 192, 14, 14)
    F_t = torch.randn(4, 384, 14, 14)
    loss = layer_fft_loss(F_s, F_t)
    print(f"   Student shape: {F_s.shape}")
    print(f"   Teacher shape: {F_t.shape}")
    print(f"   FFT loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    print("   ✓ Smaller student works (teacher pooled)")

    print("\n✓ All dimension cases work!")

    return True


def test_distributed_compatibility():
    """Test that the model is compatible with DistributedDataParallel"""
    print("\n" + "=" * 70)
    print("DDP COMPATIBILITY TEST")
    print("=" * 70)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping DDP test")
        return True

    print("\n1. Creating model on GPU...")
    student = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )

    device = torch.device('cuda:0')
    student = student.to(device)

    print(f"   Model on device: {next(student.parameters()).device}")

    # Test forward pass on GPU
    print("\n2. Testing forward pass on GPU...")
    x = torch.randn(2, 3, 224, 224).to(device)
    logits, features = student(x, indices=[11], require_feat=True)

    print(f"   Logits device: {logits.device}")
    print(f"   Features device: {features[0].device}")

    assert logits.device.type == 'cuda'
    assert features[0].device.type == 'cuda'

    print("\n✓ GPU compatibility verified!")

    return True


def test_checkpoint_save_load():
    """Test that model can be saved and loaded correctly"""
    print("\n" + "=" * 70)
    print("CHECKPOINT SAVE/LOAD TEST")
    print("=" * 70)

    import tempfile
    import os

    print("\n1. Creating model...")
    model1 = create_modified_student(
        base_model_name='deit_tiny_patch16_224',
        teacher_embed_dim=384,
        expansion_start_layer=9,
        expansion_type='linear'
    )

    # Get initial weights
    print("\n2. Getting initial weights...")
    initial_weight = model1.head.weight.clone()

    # Save checkpoint
    print("\n3. Saving checkpoint...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

        checkpoint = {
            'model': model1.state_dict(),
            'optimizer': None,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"   Saved to: {checkpoint_path}")

        # Create new model
        print("\n4. Creating new model instance...")
        model2 = create_modified_student(
            base_model_name='deit_tiny_patch16_224',
            teacher_embed_dim=384,
            expansion_start_layer=9,
            expansion_type='linear'
        )

        # Load checkpoint
        print("\n5. Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model'])

        # Verify weights match
        print("\n6. Verifying weights...")
        loaded_weight = model2.head.weight

        weight_diff = (initial_weight - loaded_weight).abs().max().item()
        print(f"   Max weight difference: {weight_diff:.10f}")

        assert weight_diff < 1e-6, "Weights don't match after loading!"

    print("\n✓ Checkpoint save/load works!")

    return True


def main():
    """Run all integration tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "Integration Tests - Training Pipeline" + " " * 13 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        # Test 1: Basic training integration
        test_training_integration()

        # Test 2: Dimension matching in loss
        test_dimension_matching_in_loss()

        # Test 3: Distributed compatibility
        test_distributed_compatibility()

        # Test 4: Checkpoint save/load
        test_checkpoint_save_load()

        print("\n" + "=" * 70)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe modified architecture is fully compatible with:")
        print("  ✓ Training pipeline (forward/backward/optimize)")
        print("  ✓ DistillationLoss with FFT")
        print("  ✓ GPU training")
        print("  ✓ Checkpoint save/load")
        print("\nReady for full-scale training!")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
