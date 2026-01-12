"""
Tests for the dataset module.
"""

import json
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_emotion_index():
    """Test emotion index mapping."""
    from src.dataset import EMOTION_INDEX, INDEX_TO_EMOTION

    assert EMOTION_INDEX["joy"] == 7
    assert EMOTION_INDEX["anger"] == 12
    assert EMOTION_INDEX["neutral"] == 27

    assert INDEX_TO_EMOTION[7] == "joy"
    assert INDEX_TO_EMOTION[12] == "anger"
    assert INDEX_TO_EMOTION[27] == "neutral"

    assert len(EMOTION_INDEX) == 28
    assert len(INDEX_TO_EMOTION) == 28

    print("✓ Emotion index test passed")


def test_soft_label_to_vector():
    """Test soft label to vector conversion."""
    import torch

    from src.dataset import FineGrainedEmotionDataset, EMOTION_INDEX
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"text": "I love this!", "targets": [{"span_text": "love", "char_start": 2, "char_end": 6, "soft_label": {"joy": 0.7, "love": 0.3}}]}')
        temp_path = f.name

    try:
        dataset = FineGrainedEmotionDataset(temp_path, tokenizer)

        # Check sample was created
        assert len(dataset) == 1

        # Check soft label conversion
        sample = dataset[0]
        assert sample["soft_label"].shape == (28,)
        assert sample["soft_label"][EMOTION_INDEX["joy"]] > 0
        assert sample["soft_label"][EMOTION_INDEX["love"]] > 0
    finally:
        os.unlink(temp_path)

    print("✓ Soft label to vector test passed")


def test_sample_flattening():
    """Test Sample Flattening (1 sentence → N samples)."""
    import torch

    from src.dataset import FineGrainedEmotionDataset, EMOTION_INDEX
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Create data with multiple targets
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = {
            "text": "The movie was fantastic but acting terrible",
            "targets": [
                {"span_text": "fantastic", "char_start": 16, "char_end": 26, "soft_label": {"joy": 0.8}},
                {"span_text": "terrible", "char_start": 37, "char_end": 45, "soft_label": {"anger": 0.7}},
            ]
        }
        f.write(json.dumps(data))
        temp_path = f.name

    try:
        dataset = FineGrainedEmotionDataset(temp_path, tokenizer)

        # Should have 2 samples (one per target)
        assert len(dataset) == 2

        # Check both samples have the same text
        sample0 = dataset[0]
        sample1 = dataset[1]
        assert sample0["soft_label"][EMOTION_INDEX["joy"]] > 0.5
        assert sample1["soft_label"][EMOTION_INDEX["anger"]] > 0.5
    finally:
        os.unlink(temp_path)

    print("✓ Sample flattening test passed")


def test_entity_mask_generation():
    """Test entity mask generation from character offsets."""
    import torch

    from src.dataset import FineGrainedEmotionDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Create data with known entity position
    text = "The fantastic movie"
    char_start = 4  # "fantastic" starts at index 4
    char_end = 14    # "fantastic" ends at index 14

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = {
            "text": text,
            "targets": [
                {"span_text": "fantastic", "char_start": char_start, "char_end": char_end, "soft_label": {"joy": 0.8}},
            ]
        }
        f.write(json.dumps(data))
        temp_path = f.name

    try:
        dataset = FineGrainedEmotionDataset(temp_path, tokenizer)
        sample = dataset[0]

        # Check entity mask
        entity_mask = sample["entity_mask"]
        assert entity_mask.shape == (128,)  # max_len

        # Find tokens that correspond to "fantastic"
        # The exact position depends on tokenization, but some should be marked
        assert entity_mask.sum() > 0, "At least some tokens should be marked as entity"

    finally:
        os.unlink(temp_path)

    print("✓ Entity mask generation test passed")


def test_no_targets_fallback():
    """Test dataset with no targets (sentence-level)."""
    import torch

    from src.dataset import FineGrainedEmotionDataset, EMOTION_INDEX
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Create data without targets
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = {
            "text": "Hello world",
            "soft_label": {"neutral": 1.0}
        }
        f.write(json.dumps(data))
        temp_path = f.name

    try:
        dataset = FineGrainedEmotionDataset(temp_path, tokenizer)

        # Should create one sample with full text as entity
        assert len(dataset) == 1

        sample = dataset[0]
        assert sample["soft_label"][EMOTION_INDEX["neutral"]] > 0.5

    finally:
        os.unlink(temp_path)

    print("✓ No targets fallback test passed")


def test_postprocess_llm_output():
    """Test LLM output post-processing."""
    from src.dataset import postprocess_llm_output

    text = "The fantastic movie"

    llm_output = {
        "targets": [
            {"span_text": "fantastic", "soft_label": {"joy": 0.8}},
            {"span_text": "movie", "soft_label": {"neutral": 0.9}},
        ]
    }

    result = postprocess_llm_output(text, llm_output)

    assert result["targets"][0]["char_start"] == 4
    assert result["targets"][0]["char_end"] == 13  # 4 + 9
    assert result["targets"][1]["char_start"] == 14
    assert result["targets"][1]["char_end"] == 19  # 14 + 5

    print("✓ Postprocess LLM output test passed")


def test_generate_dummy_data():
    """Test dummy data generation."""
    from src.dataset import generate_dummy_data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    try:
        generate_dummy_data(temp_path, num_samples=10)

        # Check file was created and has content
        with open(temp_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 10

        # Check each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "text" in data
            assert "targets" in data

    finally:
        os.unlink(temp_path)

    print("✓ Generate dummy data test passed")


if __name__ == "__main__":
    import json
    from src.dataset import EMOTION_INDEX

    test_emotion_index()
    test_soft_label_to_vector()
    test_sample_flattening()
    test_entity_mask_generation()
    test_no_targets_fallback()
    test_postprocess_llm_output()
    test_generate_dummy_data()
    print("\n✅ All dataset tests passed!")
