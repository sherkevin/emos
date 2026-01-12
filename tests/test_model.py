"""
Tests for the model module.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_creation():
    """Test ProbabilisticGBERTV4 creation."""
    import torch

    from src.model import ProbabilisticGBERTV4

    # Create model (use smaller model for testing)
    model = ProbabilisticGBERTV4(
        model_name="roberta-base",
        embedding_dim=64,
        num_emotions=28,
        alpha_scale=50.0,
        use_hf_mirror=True,
    )

    assert model.embedding_dim == 64
    assert model.num_emotions == 28
    assert model.alpha_scale == 50.0

    print("✓ Model creation test passed")


def test_model_forward_sentence():
    """Test model forward pass (sentence-level)."""
    import torch

    from src.model import ProbabilisticGBERTV4

    model = ProbabilisticGBERTV4(use_hf_mirror=True)

    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    outputs = model(input_ids, attention_mask)

    # Check outputs
    assert "mu" in outputs
    assert "kappa" in outputs
    assert "aux_logits" in outputs

    assert outputs["mu"].shape == (batch_size, 64)
    assert outputs["kappa"].shape == (batch_size, 1)
    assert outputs["aux_logits"].shape == (batch_size, 28)

    # Check mu is normalized
    mu_norms = torch.norm(outputs["mu"], p=2, dim=1)
    assert torch.allclose(mu_norms, torch.ones(batch_size), atol=1e-5)

    # Check kappa is positive
    assert torch.all(outputs["kappa"] > 0)

    print(f"✓ Model forward (sentence) test passed (kappa={outputs['kappa'].mean().item():.2f})")


def test_model_forward_entity():
    """Test model forward pass (entity-level)."""
    import torch

    from src.model import ProbabilisticGBERTV4

    model = ProbabilisticGBERTV4(use_hf_mirror=True)

    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Create entity mask (first half of sequence)
    entity_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    entity_mask[:, :10] = True

    # Forward pass with entity mask
    outputs = model(input_ids, attention_mask, entity_mask)

    # Check outputs include entity-level
    assert "mu" in outputs
    assert "kappa" in outputs
    assert "mu_entity" in outputs
    assert "kappa_entity" in outputs
    assert "aux_logits" in outputs

    assert outputs["mu_entity"].shape == (batch_size, 64)
    assert outputs["kappa_entity"].shape == (batch_size, 1)

    # Check entity mu is also normalized
    mu_entity_norms = torch.norm(outputs["mu_entity"], p=2, dim=1)
    assert torch.allclose(mu_entity_norms, torch.ones(batch_size), atol=1e-5)

    print("✓ Model forward (entity) test passed")


def test_model_empty_entity_mask():
    """Test model with empty entity mask (fallback behavior)."""
    import torch

    from src.model import ProbabilisticGBERTV4

    model = ProbabilisticGBERTV4(use_hf_mirror=True)

    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Create entity mask with one empty sample
    entity_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    entity_mask[0, :10] = True  # First sample has entity
    # Second sample has NO entity (empty)

    # Forward pass (should warn but not crash)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        outputs = model(input_ids, attention_mask, entity_mask)

        # Should have warning about empty mask
        assert any("Empty entity_mask" in str(warning.message) for warning in w)

    # Check outputs are still valid
    assert "mu_entity" in outputs
    assert "kappa_entity" in outputs
    assert torch.all(torch.isfinite(outputs["mu_entity"]))
    assert torch.all(torch.isfinite(outputs["kappa_entity"]))

    print("✓ Empty entity mask fallback test passed")


def test_pool_tokens():
    """Test token pooling function."""
    import torch

    from src.model import ProbabilisticGBERTV4

    model = ProbabilisticGBERTV4(use_hf_mirror=True)

    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    dim = 8
    token_vectors = torch.randn(batch_size, seq_len, dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, 8:] = False  # Mask out second half

    # Test mean pooling
    pooled_mean = model._pool_tokens(token_vectors, mask, pooling_type="mean")
    assert pooled_mean.shape == (batch_size, dim)

    # Test max pooling
    pooled_max = model._pool_tokens(token_vectors, mask, pooling_type="max")
    assert pooled_max.shape == (batch_size, dim)

    # Test sum pooling
    pooled_sum = model._pool_tokens(token_vectors, mask, pooling_type="sum")
    assert pooled_sum.shape == (batch_size, dim)

    print("✓ Pool tokens test passed")


def test_pool_tokens_empty_mask():
    """Test pooling with completely empty mask."""
    import torch

    from src.model import ProbabilisticGBERTV4

    model = ProbabilisticGBERTV4(use_hf_mirror=True)

    # Create dummy inputs with one empty mask
    batch_size = 2
    seq_len = 16
    dim = 8
    token_vectors = torch.randn(batch_size, seq_len, dim)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, :8] = True  # First sample has some valid tokens
    # Second sample is all False (empty)

    # Test max pooling with empty mask (should handle -inf)
    pooled_max = model._pool_tokens(token_vectors, mask, pooling_type="max")
    assert pooled_max.shape == (batch_size, dim)
    assert torch.all(torch.isfinite(pooled_max))

    print("✓ Pool tokens empty mask test passed")


def test_hf_mirror_setup():
    """Test HuggingFace mirror setup."""
    import os

    from src.model import setup_hf_mirror

    setup_hf_mirror("https://test-endpoint.com")

    assert os.environ.get("HF_ENDPOINT") == "https://test-endpoint.com"

    # Clean up
    del os.environ["HF_ENDPOINT"]

    print("✓ HF mirror setup test passed")


def test_predictor_creation():
    """Test GbertPredictor creation."""
    import torch

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from transformers import AutoTokenizer

    model = ProbabilisticGBERTV4(use_hf_mirror=False)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device="cpu")

    assert predictor.model is model
    assert predictor.tokenizer is tokenizer
    assert predictor.device == "cpu"

    print("✓ Predictor creation test passed")


def test_predictor_sentence():
    """Test GbertPredictor sentence-level prediction."""
    import torch

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from transformers import AutoTokenizer

    model = ProbabilisticGBERTV4(use_hf_mirror=False)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device="cpu")

    # Test sentence prediction
    result = predictor.predict("I am absolutely furious right now!")

    assert result["text"] == "I am absolutely furious right now!"
    assert result["mode"] == "sentence"
    assert result["span_text"] is None
    assert isinstance(result["mu"], list)
    assert len(result["mu"]) == 64
    assert isinstance(result["kappa"], float)

    print(f"✓ Predictor sentence test passed (kappa={result['kappa']:.2f})")


def test_predictor_entity():
    """Test GbertPredictor entity-level prediction."""
    import torch

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from transformers import AutoTokenizer

    model = ProbabilisticGBERTV4(use_hf_mirror=False)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device="cpu")

    # Test entity prediction
    text = "The movie was fantastic but acting terrible"
    result = predictor.predict(text, span_text="fantastic")

    assert result["text"] == text
    assert result["mode"] == "entity"
    assert result["span_text"] == "fantastic"
    assert isinstance(result["mu"], list)
    assert len(result["mu"]) == 64

    print(f"✓ Predictor entity test passed (kappa={result['kappa']:.2f})")


def test_predictor_entity_not_found():
    """Test predictor when entity span is not found."""
    import torch

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from transformers import AutoTokenizer
    import warnings

    model = ProbabilisticGBERTV4(use_hf_mirror=False)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device="cpu")

    # Test with non-existent span
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = predictor.predict("Hello world", span_text="nonexistent")

    # Should fall back to sentence-level
    assert result["mode"] == "sentence"

    print("✓ Predictor entity not found test passed")


if __name__ == "__main__":
    import warnings

    # Suppress transformer warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)

    test_model_creation()
    test_model_forward_sentence()
    test_model_forward_entity()
    test_model_empty_entity_mask()
    test_pool_tokens()
    test_pool_tokens_empty_mask()
    test_hf_mirror_setup()
    test_predictor_creation()
    test_predictor_sentence()
    test_predictor_entity()
    test_predictor_entity_not_found()
    print("\n✅ All model tests passed!")
