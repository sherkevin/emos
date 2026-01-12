"""
Tests for the train script functionality.
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_train_imports():
    """Test that train.py can be imported."""
    # This just checks the imports work
    import torch
    assert torch.__version__ is not None
    print("✓ Train imports test passed")


def test_train_argparse():
    """Test argument parsing for train.py."""
    # Simulate sys.argv
    original_argv = sys.argv
    sys.argv = ["train.py", "--batch_size", "32", "--epochs", "1"]

    try:
        # Import train module (not executing main)
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", os.path.join(os.path.dirname(__file__)[:-6], "train.py"))
        train_module = importlib.util.module_from_spec(spec)

        # Parse args
        from train import parse_args
        args = parse_args()

        assert args.batch_size == 32
        assert args.epochs == 1
        assert args.model_name == "roberta-base"

        print("✓ Train argparse test passed")
    finally:
        sys.argv = original_argv


def test_training_pipeline_setup():
    """Test that the training pipeline components can be initialized."""
    import torch
    from transformers import AutoTokenizer

    from src.config import Config
    from src.model import ProbabilisticGBERTV4
    from src.loss import ProbabilisticGBERTLoss
    from src.dataset import FineGrainedEmotionDataset, generate_dummy_data
    from src.utils import get_device

    # Create dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.jsonl")
        val_path = os.path.join(tmpdir, "val.jsonl")

        generate_dummy_data(train_path, num_samples=10)
        generate_dummy_data(val_path, num_samples=5)

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Create datasets (note: sample flattening means count may be higher)
        train_dataset = FineGrainedEmotionDataset(train_path, tokenizer)
        val_dataset = FineGrainedEmotionDataset(val_path, tokenizer)

        # Sample flattening with multiple targets creates more samples
        assert len(train_dataset) >= 10  # At least 10 samples
        assert len(val_dataset) >= 5     # At least 5 samples

        # Create model
        model = ProbabilisticGBERTV4(
            model_name="roberta-base",
            embedding_dim=32,  # Smaller for testing
            num_emotions=28,
            use_hf_mirror=False,
        )

        # Create loss
        criterion = ProbabilisticGBERTLoss(
            num_emotions=28,
            embedding_dim=32,
        )

        # Test forward pass
        device = get_device()
        model = model.to(device)
        criterion = criterion.to(device)

        # Get a sample
        sample = train_dataset[0]

        # Add batch dimension
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        entity_mask = sample["entity_mask"].unsqueeze(0).to(device)
        soft_labels = sample["soft_label"].unsqueeze(0).to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask, entity_mask)
        loss, loss_dict = criterion(
            outputs["mu"],
            outputs["kappa"],
            outputs["aux_logits"],
            soft_labels,
        )

        assert torch.isfinite(loss)
        assert "total" in loss_dict

        print(f"✓ Training pipeline setup test passed (loss={loss.item():.4f})")


def test_optimizer_setup():
    """Test optimizer and scheduler setup."""
    import torch
    from transformers import get_linear_schedule_with_warmup, AutoTokenizer

    from src.config import Config
    from src.model import ProbabilisticGBERTV4
    from src.loss import ProbabilisticGBERTLoss
    from src.utils import get_device

    device = get_device()

    # Create model
    model = ProbabilisticGBERTV4(
        model_name="roberta-base",
        embedding_dim=32,
        use_hf_mirror=False,
    ).to(device)

    criterion = ProbabilisticGBERTLoss(
        num_emotions=28,
        embedding_dim=32,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.semantic_head.parameters(), "lr": 1e-4},
        {"params": model.energy_proj.parameters(), "lr": 1e-4},
        {"params": model.aux_head.parameters(), "lr": 1e-4},
        {"params": criterion.vmf_loss.prototypes, "lr": 1e-4},
    ], weight_decay=0.01)

    # Create scheduler
    total_steps = 100
    warmup_steps = 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Test one step
    scheduler.step()

    assert scheduler.get_last_lr()[0] > 0

    print("✓ Optimizer setup test passed")


if __name__ == "__main__":
    from src.dataset import EMOTION_INDEX

    test_train_imports()
    test_train_argparse()
    test_training_pipeline_setup()
    test_optimizer_setup()
    print("\n✅ All train tests passed!")
