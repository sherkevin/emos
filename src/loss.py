"""
Loss functions for Probabilistic G-BERT V4.

Implements:
1. Supervised vMF-NCE Loss with Class Prototypes
2. Calibration Loss (κ alignment with Soft Label Max-Norm)
3. Auxiliary Loss (semantic preservation)

Key design decisions from PRD V5:
- Exclude neutral category when computing intensity
- Use detach() on kappa in L_vMF to prevent gradient leakage
- Normalize prototypes at each forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SupervisedVMFNLoss(nn.Module):
    """
    Supervised vMF-NCE Loss with Class Prototypes.

    Contrastive learning strategy: pull samples closer to their emotion class centers.

    Key design (V5):
    1. Prototypes are normalized at each forward pass (prevents magnitude cheating)
    2. Kappa uses detach() (prevents gradient leakage - kappa only updated by L_Cal)

    Args:
        num_emotions: Number of emotion categories (default: 28)
        embedding_dim: Dimension of the embedding space (default: 64)
    """

    def __init__(self, num_emotions: int = 28, embedding_dim: int = 64):
        super().__init__()
        # Learnable class prototypes: center vectors for each emotion category
        self.prototypes = nn.Parameter(torch.randn(num_emotions, embedding_dim))

        # Initialize with L2 normalization
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, p=2, dim=1))

        self.num_emotions = num_emotions
        self.embedding_dim = embedding_dim

    def forward(
        self, mu: torch.Tensor, kappa: torch.Tensor, soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Supervised vMF-NCE loss.

        Args:
            mu: (B, D) sample semantic directions (already L2-normalized)
            kappa: (B, 1) sample concentration parameters
            soft_labels: (B, num_emotions) soft label distributions

        Returns:
            L_vMF: scalar loss value
        """
        batch_size = mu.shape[0]

        # Step A: Normalize prototypes at each forward pass
        # This prevents the model from "cheating" by increasing prototype magnitude
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # (num_emotions, D)

        # Step B: Compute cosine similarity between samples and all prototypes
        # (B, D) @ (D, num_emotions) -> (B, num_emotions)
        cosine_sim = torch.matmul(mu, prototypes_norm.T)

        # Step C: Apply dynamic temperature (CRITICAL: Detach Kappa!)
        # L_vMF should only update mu and prototypes, not kappa
        # Kappa is only updated by L_Cal (calibration loss)
        kappa_fixed = kappa.detach()  # (B, 1)
        scaled_logits = cosine_sim * kappa_fixed  # (B, num_emotions)

        # Step D: KL Divergence with soft labels as target
        # This supports multi-label scenarios (e.g., "joyful surprise")
        log_probs = F.log_softmax(scaled_logits, dim=1)  # (B, num_emotions)
        l_vmf = F.kl_div(log_probs, soft_labels, reduction="batchmean")

        return l_vmf


def calibration_loss(
    predicted_kappa: torch.Tensor,
    soft_labels: torch.Tensor,
    alpha_scale: float = 50.0,
    neutral_idx: int = 27,
) -> torch.Tensor:
    """
    Calibration Loss: align predicted κ with Soft Label Max-Norm intensity.

    CRITICAL (V5): Exclude neutral category from Max-Norm calculation.
    This prevents neutral/boring sentences from getting high κ.

    Args:
        predicted_kappa: (B, 1) or (B,) predicted concentration parameters
        soft_labels: (B, num_emotions) soft label distributions
        alpha_scale: Physical scaling coefficient (default: 50.0)
        neutral_idx: Index of neutral category (default: 27)

    Returns:
        L_Cal: scalar MSE loss
    """
    # Squeeze predicted_kappa if needed
    if predicted_kappa.dim() > 1:
        predicted_kappa = predicted_kappa.squeeze(-1)  # (B,)

    # Step A: Exclude neutral from Max-Norm calculation (The Neutrality Paradox)
    # Neutral-dominant sentences should have LOW kappa, not high
    num_emotions = soft_labels.shape[-1]
    if neutral_idx >= num_emotions:
        # If neutral_idx is out of bounds, don't filter
        soft_labels_no_neutral = soft_labels
    else:
        # Create mask for all columns except neutral
        mask = torch.ones(num_emotions, dtype=torch.bool, device=soft_labels.device)
        mask[neutral_idx] = False
        soft_labels_no_neutral = soft_labels[:, mask]  # (B, num_emotions - 1)

    # Step B: Compute intensity as Max-Norm (excluding neutral)
    i_raw = torch.max(soft_labels_no_neutral, dim=1).values  # (B,)

    # Step C: Compute target kappa with physical scaling
    target_kappa = 1.0 + alpha_scale * i_raw  # (B,)

    # Step D: MSE loss
    l_cal = F.mse_loss(predicted_kappa, target_kappa)

    return l_cal


def auxiliary_loss(
    aux_logits: torch.Tensor, soft_labels: torch.Tensor
) -> torch.Tensor:
    """
    Auxiliary Loss: KL divergence for semantic preservation.

    Ensures the bottleneck vector μ retains recoverable emotion category information.
    Prevents semantic collapse during early training.

    Args:
        aux_logits: (B, num_emotions) auxiliary classifier logits
        soft_labels: (B, num_emotions) soft label distributions

    Returns:
        L_Aux: scalar KL divergence loss
    """
    log_pred = F.log_softmax(aux_logits, dim=1)
    l_aux = F.kl_div(log_pred, soft_labels, reduction="batchmean")
    return l_aux


class ProbabilisticGBERTLoss(nn.Module):
    """
    Combined loss function for Probabilistic G-BERT V4.

    L_Total = L_vMF + λ_Cal * L_Cal + λ_Aux * L_Aux

    Args:
        num_emotions: Number of emotion categories (default: 28)
        embedding_dim: Dimension of the embedding space (default: 64)
        lambda_cal: Weight for calibration loss (default: 0.1)
        lambda_aux: Weight for auxiliary loss (default: 0.05)
        alpha_scale: Physical scaling coefficient (default: 50.0)
        neutral_idx: Index of neutral category (default: 27)
    """

    def __init__(
        self,
        num_emotions: int = 28,
        embedding_dim: int = 64,
        lambda_cal: float = 0.1,
        lambda_aux: float = 0.05,
        alpha_scale: float = 50.0,
        neutral_idx: int = 27,
    ):
        super().__init__()
        self.vmf_loss = SupervisedVMFNLoss(num_emotions, embedding_dim)
        self.lambda_cal = lambda_cal
        self.lambda_aux = lambda_aux
        self.alpha_scale = alpha_scale
        self.neutral_idx = neutral_idx

    def forward(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        aux_logits: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual components.

        Args:
            mu: (B, embedding_dim) sample semantic directions
            kappa: (B, 1) sample concentration parameters
            aux_logits: (B, num_emotions) auxiliary classifier logits
            soft_labels: (B, num_emotions) soft label distributions

        Returns:
            total_loss: scalar total loss
            loss_dict: Dictionary containing individual loss components
        """
        # L_vMF: Supervised Prototype Loss
        l_vmf = self.vmf_loss(mu, kappa, soft_labels)

        # L_Cal: Calibration Loss (physics constraint)
        l_cal = calibration_loss(
            kappa, soft_labels, self.alpha_scale, self.neutral_idx
        )

        # L_Aux: Auxiliary Loss (semantic regularization)
        l_aux = auxiliary_loss(aux_logits, soft_labels)

        # Total Loss
        total_loss = l_vmf + self.lambda_cal * l_cal + self.lambda_aux * l_aux

        # Return individual components for logging
        loss_dict = {
            "total": total_loss.item(),
            "vmf": l_vmf.item(),
            "cal": l_cal.item(),
            "aux": l_aux.item(),
        }

        return total_loss, loss_dict


def compute_intensity(
    soft_labels: torch.Tensor, neutral_idx: int = 27, exclude_neutral: bool = True
) -> torch.Tensor:
    """
    Compute intensity from soft labels.

    Args:
        soft_labels: (B, num_emotions) soft label distributions
        neutral_idx: Index of neutral category (default: 27)
        exclude_neutral: Whether to exclude neutral from Max-Norm (default: True)

    Returns:
        intensity: (B,) intensity values
    """
    if exclude_neutral and neutral_idx < soft_labels.shape[-1]:
        # Exclude neutral from calculation
        mask = torch.ones(soft_labels.shape[-1], dtype=torch.bool, device=soft_labels.device)
        mask[neutral_idx] = False
        soft_labels_filtered = soft_labels[:, mask]
        return torch.max(soft_labels_filtered, dim=1).values
    else:
        return torch.max(soft_labels, dim=1).values
