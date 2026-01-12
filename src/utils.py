"""
Utility functions for Probabilistic G-BERT V4.

Includes logging, device management, seed setting, and metrics.
"""

import os
import random
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np


class Logger:
    """
    Unified logging interface supporting both console and file output.

    Automatically detects and initializes WandB if available.
    """

    def __init__(
        self,
        name: str = "probabilistic_gbert",
        log_dir: str = "logs",
        use_wandb: bool = True,
        project_name: str = "probabilistic-gbert",
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory to save log files
            use_wandb: Whether to use Weights & Biases (if available)
            project_name: WandB project name
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"train_{timestamp}.log"

        # Configure Python logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        self.logger.addHandler(file_handler)

        # WandB initialization (optional)
        self.wandb = None
        if use_wandb:
            self._init_wandb(project_name)

        self.info(f"Logger initialized. Log file: {log_file}")

    def _init_wandb(self, project_name: str):
        """Initialize Weights & Biases if available."""
        try:
            import wandb

            wandb.init(project=project_name, reinit=True)
            self.wandb = wandb
            self.info("WandB initialized successfully")
        except ImportError:
            self.info("WandB not available. Skipping WandB logging.")
        except Exception as e:
            self.warning(f"Failed to initialize WandB: {e}")

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg)
        if self.wandb:
            self.wandb.log({"info": msg}, **kwargs)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def log_metrics(self, metrics: dict[str, Any], step: int):
        """Log metrics to both console and WandB."""
        # Format metrics for console
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        self.info(f"Step {step}: {metrics_str}")

        # Log to WandB
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        """Log configuration parameters."""
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        if self.wandb:
            self.wandb.config.update(config)

    def close(self):
        """Close the logger and WandB."""
        if self.wandb:
            self.wandb.finish()
        for handler in self.logger.handlers:
            handler.close()


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True) -> str:
    """
    Detect and return the available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        "cuda" if available and preferred, otherwise "cpu"
    """
    if not prefer_cuda or os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            # Enable cuDNN benchmark for faster training
            torch.backends.cudnn.benchmark = True
            return device
    except ImportError:
        pass

    return "cpu"


def format_number(n: float, precision: int = 4) -> str:
    """
    Format a number for display.

    Args:
        n: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if isinstance(n, float):
        if abs(n) < 1e-4:
            return f"{n:.2e}"
        return f"{n:.{precision}f}"
    return str(n)


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking training metrics.
    """

    def __init__(self):
        """Initialize the meter."""
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.

        Args:
            val: New value to add
            n: Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class MetricsTracker:
    """
    Track multiple metrics during training.
    """

    def __init__(self):
        """Initialize the tracker."""
        self.metrics = {}

    def update(self, name: str, value: float, n: int = 1):
        """
        Update a metric.

        Args:
            name: Metric name
            value: Metric value
            n: Number of items this value represents
        """
        if name not in self.metrics:
            self.metrics[name] = AverageMeter()
        self.metrics[name].update(value, n)

    def get(self, name: str) -> float:
        """Get the current average of a metric."""
        if name in self.metrics:
            return self.metrics[name].avg
        return 0.0

    def get_all(self) -> dict[str, float]:
        """Get all current averages."""
        return {name: meter.avg for name, meter in self.metrics.items()}

    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
