"""
Tests for the config module.
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_creation():
    """Test basic config creation."""
    from src.config import Config

    config = Config()
    assert config.model_name == "roberta-base"
    assert config.embedding_dim == 64
    assert config.num_emotions == 28
    assert config.alpha_scale == 50.0
    print("✓ Config creation test passed")


def test_config_dict_conversion():
    """Test config to dictionary conversion."""
    from src.config import Config

    config = Config()
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "model_name" in config_dict
    assert config_dict["embedding_dim"] == 64
    print("✓ Config dict conversion test passed")


def test_emotion_index_mapping():
    """Test emotion index mapping."""
    from src.config import EMOTION_INDEX, INDEX_TO_EMOTION

    # Check forward mapping
    assert EMOTION_INDEX["joy"] == 7
    assert EMOTION_INDEX["anger"] == 12
    assert EMOTION_INDEX["neutral"] == 27

    # Check reverse mapping
    assert INDEX_TO_EMOTION[7] == "joy"
    assert INDEX_TO_EMOTION[12] == "anger"
    assert INDEX_TO_EMOTION[27] == "neutral"

    # Check all 28 categories
    assert len(EMOTION_INDEX) == 28
    assert len(INDEX_TO_EMOTION) == 28
    print("✓ Emotion index mapping test passed")


def test_batch_size_adjustment():
    """Test automatic grad_accum_steps adjustment."""
    from src.config import Config

    config = Config(
        physical_batch_size=32,
        effective_batch_size=256,
        grad_accum_steps=4  # This will be auto-adjusted to 8
    )
    expected_grad_accum = 256 // 32  # = 8
    assert config.grad_accum_steps == expected_grad_accum
    print(f"✓ Batch size adjustment test passed (grad_accum_steps={config.grad_accum_steps})")


def test_device_detection():
    """Test device detection."""
    from src.config import Config, get_device

    device = get_device()
    assert device in ["cuda", "cpu"]
    print(f"✓ Device detection test passed (device={device})")


def test_temp_directory_creation():
    """Test that config creates directories."""
    from src.config import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            log_dir=os.path.join(tmpdir, "logs"),
            data_dir=os.path.join(tmpdir, "data"),
        )
        # Directories should be created
        assert os.path.exists(config.checkpoint_dir)
        assert os.path.exists(config.log_dir)
        assert os.path.exists(config.data_dir)
    print("✓ Directory creation test passed")


if __name__ == "__main__":
    test_config_creation()
    test_config_dict_conversion()
    test_emotion_index_mapping()
    test_batch_size_adjustment()
    test_device_detection()
    test_temp_directory_creation()
    print("\n✅ All config tests passed!")
