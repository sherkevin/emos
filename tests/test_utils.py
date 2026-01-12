"""
Tests for the utils module.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_logger_creation():
    """Test logger creation and basic logging."""
    from src.utils import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_wandb=False)
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.debug("Test debug message")
        logger.close()

        # Check log file was created
        log_files = list(Path(tmpdir).glob("*.log"))
        assert len(log_files) > 0
    print("✓ Logger creation test passed")


def test_logger_metrics():
    """Test logger metrics logging."""
    from src.utils import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_wandb=False)

        metrics = {"loss": 0.5, "accuracy": 0.85, "kappa": 25.3}
        logger.log_metrics(metrics, step=100)
        logger.close()
    print("✓ Logger metrics test passed")


def test_logger_config():
    """Test logger config logging."""
    from src.utils import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_wandb=False)

        config = {
            "model_name": "roberta-base",
            "embedding_dim": 64,
            "lr": 0.001,
        }
        logger.log_config(config)
        logger.close()
    print("✓ Logger config test passed")


def test_set_seed():
    """Test random seed setting."""
    from src.utils import set_seed
    import random
    import numpy as np

    set_seed(42)

    # Test Python random - verify reproducibility
    val1 = random.randint(0, 1000)
    set_seed(42)
    val2 = random.randint(0, 1000)
    assert val1 == val2, f"Expected same value with same seed, got {val1} != {val2}"

    # Test numpy reproducibility
    set_seed(42)
    val3 = np.random.randint(0, 1000)
    set_seed(42)
    val4 = np.random.randint(0, 1000)
    assert val3 == val4, f"Expected same numpy value with same seed, got {val3} != {val4}"

    print("✓ Set seed test passed")


def test_get_device():
    """Test device detection."""
    from src.utils import get_device

    device = get_device()
    assert device in ["cuda", "cpu"]
    print(f"✓ Get device test passed (device={device})")


def test_format_number():
    """Test number formatting."""
    from src.utils import format_number

    assert format_number(0.5) == "0.5000"
    assert format_number(1.23456789, precision=2) == "1.23"
    assert format_number(0.00001) == "1.00e-05"
    assert format_number(12345) == "12345"
    print("✓ Format number test passed")


def test_average_meter():
    """Test AverageMeter class."""
    from src.utils import AverageMeter

    meter = AverageMeter()
    assert meter.avg == 0.0

    meter.update(10.0)
    assert meter.avg == 10.0

    meter.update(20.0)
    assert meter.avg == 15.0

    meter.update(30.0, n=2)
    assert meter.avg == 22.5  # (10 + 20 + 30*2) / 4

    meter.reset()
    assert meter.avg == 0.0
    print("✓ Average meter test passed")


def test_metrics_tracker():
    """Test MetricsTracker class."""
    from src.utils import MetricsTracker

    tracker = MetricsTracker()

    tracker.update("loss", 1.0)
    tracker.update("loss", 2.0)
    assert tracker.get("loss") == 1.5

    tracker.update("accuracy", 0.8)
    tracker.update("accuracy", 0.9)
    assert abs(tracker.get("accuracy") - 0.85) < 1e-6

    all_metrics = tracker.get_all()
    assert all_metrics["loss"] == 1.5
    assert abs(all_metrics["accuracy"] - 0.85) < 1e-6

    tracker.reset()
    assert tracker.get("loss") == 0.0
    print("✓ Metrics tracker test passed")


if __name__ == "__main__":
    from pathlib import Path

    test_logger_creation()
    test_logger_metrics()
    test_logger_config()
    test_set_seed()
    test_get_device()
    test_format_number()
    test_average_meter()
    test_metrics_tracker()
    print("\n✅ All utils tests passed!")
