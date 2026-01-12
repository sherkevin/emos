"""
Tests for the loss module.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def test_supervised_vmf_loss_creation():
    """Test SupervisedVMFNLoss creation and initialization."""
    import torch

    from src.loss import SupervisedVMFNLoss

    loss_fn = SupervisedVMFNLoss(num_emotions=28, embedding_dim=64)

    # Check prototypes are created
    assert loss_fn.prototypes.shape == (28, 64)

    # Check prototypes are L2 normalized
    norms = torch.norm(loss_fn.prototypes, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(28), atol=1e-6)

    print("✓ SupervisedVMFNLoss creation test passed")


def test_supervised_vmf_loss_forward():
    """Test SupervisedVMFNLoss forward pass."""
    import torch

    from src.loss import SupervisedVMFNLoss

    loss_fn = SupervisedVMFNLoss(num_emotions=28, embedding_dim=64)

    # Create dummy inputs
    batch_size = 4
    mu = F.normalize(torch.randn(batch_size, 64), p=2, dim=1)
    kappa = torch.rand(batch_size, 1) * 50 + 1  # [1, 51]
    soft_labels = torch.rand(batch_size, 28)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)  # Normalize

    # Compute loss
    loss = loss_fn(mu, kappa, soft_labels)

    # Check loss is scalar and finite
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0

    print(f"✓ SupervisedVMFNLoss forward test passed (loss={loss.item():.4f})")


def test_supervised_vmf_loss_detach_kappa():
    """Test that kappa is detached in L_vMF (no gradient flow to kappa)."""
    import torch

    from src.loss import SupervisedVMFNLoss
    import torch.nn.functional as F

    loss_fn = SupervisedVMFNLoss(num_emotions=28, embedding_dim=64)

    # Create inputs with requires_grad
    mu_raw = torch.randn(2, 64, requires_grad=True)
    mu = F.normalize(mu_raw, p=2, dim=1)
    kappa = torch.randn(2, 1, requires_grad=True) * 10 + 10
    soft_labels = F.softmax(torch.randn(2, 28), dim=1)

    # Enable gradient retention for non-leaf mu
    mu.retain_grad()

    # Compute loss
    loss = loss_fn(mu, kappa, soft_labels)

    # Backward
    loss.backward()

    # Check gradient flow
    # mu should have gradients
    assert mu.grad is not None
    assert torch.any(mu.grad != 0)

    # kappa should NOT have gradients (detached)
    assert kappa.grad is None or torch.all(kappa.grad == 0)

    print("✓ Kappa detach test passed (no gradient leakage)")


def test_calibration_loss():
    """Test calibration loss with neutral exclusion."""
    import torch

    from src.loss import calibration_loss

    # Test case 1: Neutral dominant (should have LOW kappa)
    neutral_dominant = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.9]]).float()  # 0.9 neutral at index 4
    kappa_pred = torch.tensor([5.0])

    loss_neutral = calibration_loss(
        kappa_pred, neutral_dominant, alpha_scale=50.0, neutral_idx=4
    )

    # Target kappa should be based on 0.1 (excluding neutral)
    # target = 1.0 + 50.0 * 0.1 = 6.0
    # loss = (5.0 - 6.0)^2 = 1.0
    assert abs(loss_neutral.item() - 1.0) < 1e-4

    # Test case 2: High intensity emotion (should have HIGH kappa)
    high_intensity = torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0]]).float()  # 0.9 joy at index 1
    kappa_pred = torch.tensor([40.0])

    loss_high = calibration_loss(
        kappa_pred, high_intensity, alpha_scale=50.0, neutral_idx=4
    )

    # target = 1.0 + 50.0 * 0.9 = 46.0
    # loss = (40.0 - 46.0)^2 = 36.0
    assert abs(loss_high.item() - 36.0) < 1e-4

    print("✓ Calibration loss test passed")


def test_auxiliary_loss():
    """Test auxiliary loss computation."""
    import torch

    from src.loss import auxiliary_loss
    import torch.nn.functional as F

    # Create soft labels
    soft_labels = F.softmax(torch.randn(4, 28), dim=1)

    # Test that loss is finite and positive
    aux_logits = torch.randn(4, 28)  # Logits
    loss = auxiliary_loss(aux_logits, soft_labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0

    # Test with very large logits (confidence) - should have different loss
    aux_logits2 = torch.randn(4, 28) * 10  # More confident predictions
    loss2 = auxiliary_loss(aux_logits2, soft_labels)

    # Both should be valid losses
    assert torch.isfinite(loss2)

    print("✓ Auxiliary loss test passed")


def test_combined_loss():
    """Test combined ProbabilisticGBERTLoss."""
    import torch

    from src.loss import ProbabilisticGBERTLoss
    import torch.nn.functional as F

    loss_fn = ProbabilisticGBERTLoss(
        num_emotions=28,
        embedding_dim=64,
        lambda_cal=0.1,
        lambda_aux=0.05,
    )

    # Create dummy inputs
    batch_size = 4
    mu = F.normalize(torch.randn(batch_size, 64), p=2, dim=1)
    kappa = torch.rand(batch_size, 1) * 50 + 1
    aux_logits = torch.randn(batch_size, 28)
    soft_labels = F.softmax(torch.randn(batch_size, 28), dim=1)

    # Compute loss
    total_loss, loss_dict = loss_fn(mu, kappa, aux_logits, soft_labels)

    # Check total loss
    assert total_loss.dim() == 0
    assert torch.isfinite(total_loss)

    # Check loss dictionary
    assert set(loss_dict.keys()) == {"total", "vmf", "cal", "aux"}
    assert all(isinstance(v, float) for v in loss_dict.values())

    # Check total matches
    assert abs(loss_dict["total"] - total_loss.item()) < 1e-6

    # Check weighted components
    expected_total = loss_dict["vmf"] + 0.1 * loss_dict["cal"] + 0.05 * loss_dict["aux"]
    assert abs(loss_dict["total"] - expected_total) < 1e-4

    print(f"✓ Combined loss test passed (total={loss_dict['total']:.4f})")


def test_compute_intensity():
    """Test intensity computation with neutral exclusion."""
    import torch

    from src.loss import compute_intensity

    # Test with neutral dominant
    labels = torch.tensor([[0.05] * 27 + [0.65]]).float()  # neutral=0.65, others=0.05
    intensity = compute_intensity(labels, neutral_idx=27, exclude_neutral=True)

    # Should be 0.05 (excluding neutral), not 0.65
    assert abs(intensity.item() - 0.05) < 1e-6

    # Test without exclusion
    intensity_all = compute_intensity(labels, neutral_idx=27, exclude_neutral=False)
    assert abs(intensity_all.item() - 0.65) < 1e-6

    # Test with high emotion
    labels2 = torch.tensor([[0.8] + [0.01] * 26 + [0.19]]).float()  # joy=0.8
    intensity2 = compute_intensity(labels2, neutral_idx=27, exclude_neutral=True)
    assert abs(intensity2.item() - 0.8) < 1e-6

    print("✓ Compute intensity test passed")


def test_prototypes_gradient_flow():
    """Test that prototypes receive gradients."""
    import torch

    from src.loss import SupervisedVMFNLoss
    import torch.nn.functional as F

    loss_fn = SupervisedVMFNLoss(num_emotions=28, embedding_dim=64)

    # Create inputs
    mu = F.normalize(torch.randn(4, 64, requires_grad=False), p=2, dim=1)
    kappa = torch.randn(4, 1, requires_grad=False)
    soft_labels = F.softmax(torch.randn(4, 28), dim=1)

    # Enable gradients for prototypes
    loss_fn.prototypes.requires_grad = True

    # Compute loss
    loss = loss_fn(mu, kappa, soft_labels)

    # Backward
    loss.backward()

    # Check prototypes have gradients
    assert loss_fn.prototypes.grad is not None
    assert torch.any(loss_fn.prototypes.grad != 0)

    print("✓ Prototypes gradient flow test passed")


if __name__ == "__main__":
    import torch.nn.functional as F

    test_supervised_vmf_loss_creation()
    test_supervised_vmf_loss_forward()
    test_supervised_vmf_loss_detach_kappa()
    test_calibration_loss()
    test_auxiliary_loss()
    test_combined_loss()
    test_compute_intensity()
    test_prototypes_gradient_flow()
    print("\n✅ All loss tests passed!")
