"""
Tests for the inference script functionality.
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_inference_imports():
    """Test that inference.py can be imported."""
    import torch
    assert torch.__version__ is not None
    print("✓ Inference imports test passed")


def test_inference_argparse():
    """Test argument parsing for inference.py."""
    # Simulate sys.argv
    original_argv = sys.argv
    sys.argv = [
        "inference.py",
        "--checkpoint", "checkpoints/best_model.pt",
        "--text", "Hello world",
    ]

    try:
        from inference import parse_args
        args = parse_args()

        assert args.checkpoint == "checkpoints/best_model.pt"
        assert args.text == "Hello world"
        assert args.model_name == "roberta-base"
        assert args.entity is None

        print("✓ Inference argparse test passed")
    finally:
        sys.argv = original_argv


def test_predictor_from_scratch():
    """Test predictor creation without checkpoint."""
    import torch
    from transformers import AutoTokenizer

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from src.utils import get_device

    device = get_device()

    # Create untrained model
    model = ProbabilisticGBERTV4(
        model_name="roberta-base",
        use_hf_mirror=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device)

    # Test sentence prediction
    result = predictor.predict("I am absolutely furious!")

    assert result["text"] == "I am absolutely furious!"
    assert result["mode"] == "sentence"
    assert isinstance(result["kappa"], float)
    assert len(result["mu"]) == 64

    print(f"✓ Predictor from scratch test passed (kappa={result['kappa']:.2f})")


def test_predictor_entity():
    """Test entity-level prediction."""
    import torch
    from transformers import AutoTokenizer

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from src.utils import get_device

    device = get_device()

    model = ProbabilisticGBERTV4(
        model_name="roberta-base",
        use_hf_mirror=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    predictor = GbertPredictor(model, tokenizer, device)

    # Test entity prediction
    text = "The movie was fantastic but acting terrible"
    result = predictor.predict(text, span_text="fantastic")

    assert result["text"] == text
    assert result["mode"] == "entity"
    assert result["span_text"] == "fantastic"
    assert len(result["mu"]) == 64

    print(f"✓ Predictor entity test passed (kappa={result['kappa']:.2f})")


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    import torch
    from transformers import AutoTokenizer

    from src.model import ProbabilisticGBERTV4, GbertPredictor
    from src.utils import get_device

    device = get_device()

    # Create model
    model = ProbabilisticGBERTV4(
        model_name="roberta-base",
        embedding_dim=32,
        use_hf_mirror=False,
    )

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        torch.save(model.state_dict(), checkpoint_path)

        # Load model with same parameters then load checkpoint
        model2 = ProbabilisticGBERTV4(
            model_name="roberta-base",
            embedding_dim=32,  # Must match saved model
            use_hf_mirror=False,
        )
        model2.load_state_dict(torch.load(checkpoint_path, map_location=device))

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        predictor = GbertPredictor(model2, tokenizer, device)

        # Test prediction
        result = predictor.predict("Test text")
        assert result["mu"] is not None

        print("✓ Checkpoint save/load test passed")
    finally:
        os.unlink(checkpoint_path)


if __name__ == "__main__":
    test_inference_imports()
    test_inference_argparse()
    test_predictor_from_scratch()
    test_predictor_entity()
    test_checkpoint_save_load()
    print("\n✅ All inference tests passed!")
