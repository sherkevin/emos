"""
Configuration module for Probabilistic G-BERT V4.

Centralized hyperparameter management ensuring experimental reproducibility.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def get_device() -> str:
    """
    Dynamically detect and return the available device.

    Returns:
        "cuda" if GPU is available, otherwise "cpu"
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"
    # Note: CUDA detection will be done at runtime to avoid import errors
    # when torch is not installed
    return "cuda"  # Default, will be validated at runtime


@dataclass
class Config:
    """
    Centralized configuration for Probabilistic G-BERT V4.

    All hyperparameters are defined here for easy experimentation and reproducibility.
    """

    # ==================== Model Architecture ====================
    model_name: str = "roberta-base"
    embedding_dim: int = 64
    num_emotions: int = 28
    alpha_scale: float = 50.0  # Physical scaling coefficient for kappa

    # ==================== Data Processing ====================
    max_length: int = 128
    neutral_idx: int = 27  # Index of neutral emotion in the 28 categories

    # ==================== Training ====================
    physical_batch_size: int = 64
    effective_batch_size: int = 256
    grad_accum_steps: int = 4  # physical_batch_size * grad_accum_steps = effective_batch_size

    # Learning rates
    lr_backbone: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01

    # Scheduler
    epochs: int = 5
    warmup_ratio: float = 0.1

    # Early stopping
    patience: int = 3

    # ==================== Loss Weights ====================
    lambda_cal: float = 0.1  # Calibration loss weight
    lambda_aux: float = 0.05  # Auxiliary loss weight

    # ==================== Hardware ====================
    device: str = field(default_factory=get_device)
    use_cudnn_benchmark: bool = True

    # ==================== Logging ====================
    log_interval: int = 100  # Log every N steps
    save_interval: int = 500  # Save checkpoint every N steps

    # ==================== Paths ====================
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"

    # ==================== HuggingFace Mirror (China) ====================
    # Use HF Endpoint for faster download in China
    hf_endpoint: Optional[str] = "https://hf-mirror.com"

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Validate batch size relationship
        if self.physical_batch_size * self.grad_accum_steps != self.effective_batch_size:
            # Auto-adjust grad_accum_steps
            self.grad_accum_steps = max(1, self.effective_batch_size // self.physical_batch_size)

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def device_actual(self) -> str:
        """
        Get the actual device at runtime.

        This property checks CUDA availability at call time rather than
        at config creation time.
        """
        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                import warnings
                warnings.warn("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return self.device
        except ImportError:
            return "cpu"

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_emotions": self.num_emotions,
            "alpha_scale": self.alpha_scale,
            "max_length": self.max_length,
            "physical_batch_size": self.physical_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "lr_backbone": self.lr_backbone,
            "lr_heads": self.lr_heads,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "warmup_ratio": self.warmup_ratio,
            "patience": self.patience,
            "lambda_cal": self.lambda_cal,
            "lambda_aux": self.lambda_aux,
            "device": self.device,
        }


# GoEmotions 28 categories index mapping
EMOTION_INDEX = {
    # Positive (12)
    "admiration": 0,
    "amusement": 1,
    "approval": 2,
    "caring": 3,
    "desire": 4,
    "excitement": 5,
    "gratitude": 6,
    "joy": 7,
    "love": 8,
    "optimism": 9,
    "pride": 10,
    "relief": 11,
    # Negative (11)
    "anger": 12,
    "annoyance": 13,
    "disappointment": 14,
    "disapproval": 15,
    "disgust": 16,
    "embarrassment": 17,
    "fear": 18,
    "grief": 19,
    "nervousness": 20,
    "remorse": 21,
    "sadness": 22,
    # Ambiguous / Cognitive (4)
    "confusion": 23,
    "curiosity": 24,
    "realization": 25,
    "surprise": 26,
    # Neutral (1)
    "neutral": 27,
}

# Reverse mapping for inference
INDEX_TO_EMOTION = {v: k for k, v in EMOTION_INDEX.items()}
