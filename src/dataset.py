"""
Dataset module for Probabilistic G-BERT V4.

Implements:
1. FineGrainedEmotionDataset - Main dataset with Sample Flattening
2. Data generation utilities for LLM-based soft label creation
3. GoEmotions data processing utilities

Key features:
- Sample Flattening: 1 sentence → N training samples (one per entity)
- Character-offset alignment for entity masks
- Support for sparse soft labels (dict format)
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Emotion name to index mapping (copied from config for standalone use)
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

INDEX_TO_EMOTION = {v: k for k, v in EMOTION_INDEX.items()}


class FineGrainedEmotionDataset(Dataset):
    """
    V4 Dataset with Character-Offset Alignment and Sample Flattening.

    Key features:
    1. Sample Flattening: 1 sentence with N targets → N training samples
    2. Character-offset entity mask generation for precise token alignment

    Data format (JSONL):
    {
      "text": "The movie was fantastic but acting terrible",
      "targets": [
        {
          "span_text": "fantastic",
          "char_start": 16,
          "char_end": 26,
          "soft_label": {"joy": 0.8, "neutral": 0.2}
        },
        ...
      ]
    }

    Args:
        data_path: Path to JSONL data file
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length (default: 128)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # Load and flatten data
        self._load_and_flatten(data_path)

    def _load_and_flatten(self, data_path: str):
        """
        Load JSONL data and apply Sample Flattening.

        Each entry may have multiple targets (entities). We create one training
        sample per target, so a sentence with 3 annotated entities becomes
        3 separate training samples.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

                text = entry.get("text", "")
                if not text:
                    continue

                targets = entry.get("targets", [])
                if not targets:
                    # If no targets specified, create a sentence-level sample
                    soft_label = entry.get("soft_label", {})
                    if not soft_label:
                        # Default to neutral if no label
                        soft_label = {"neutral": 1.0}

                    self.samples.append({
                        "text": text,
                        "char_start": 0,
                        "char_end": len(text),
                        "soft_label": soft_label,
                    })
                else:
                    # Sample Flattening: create one sample per target
                    for target in targets:
                        # Support both old and new format
                        if "char_start" in target and "char_end" in target:
                            # New format with explicit character offsets
                            char_start = target["char_start"]
                            char_end = target["char_end"]
                            soft_label = target.get("soft_label", {})
                        else:
                            # Old format - compute offsets from span_text
                            span_text = target.get("span_text", "")
                            idx = text.find(span_text)
                            if idx == -1:
                                # Try regex search
                                match = re.search(re.escape(span_text), text)
                                if match:
                                    idx = match.start()
                                else:
                                    print(f"Warning: Span '{span_text}' not found in text: {text[:50]}...")
                                    continue

                            char_start = idx
                            char_end = idx + len(span_text)
                            soft_label = target.get("soft_label", {})

                        self.samples.append({
                            "text": text,
                            "char_start": char_start,
                            "char_end": char_end,
                            "soft_label": soft_label,
                        })

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def _soft_label_to_vector(self, label_input) -> torch.Tensor:
        """
        Convert soft label to 28-dim tensor.

        Args:
            label_input: Can be:
                - list/tuple of 28 floats (array format)
                - dict mapping emotion names to probabilities (dict format, for compatibility)

        Returns:
            28-dim torch tensor
        """
        # Array format (preferred)
        if isinstance(label_input, (list, tuple)):
            label_vector = torch.tensor(label_input, dtype=torch.float32)
            # Ensure it's 28-dimensional
            if label_vector.shape[0] != 28:
                raise ValueError(f"Soft label must have 28 elements, got {label_vector.shape[0]}")
            # Normalize to ensure sum = 1
            total = label_vector.sum()
            if total > 0:
                label_vector = label_vector / total
            return label_vector

        # Dict format (for backward compatibility)
        elif isinstance(label_input, dict):
            label_vector = torch.zeros(28)
            for emotion, value in label_input.items():
                if emotion in EMOTION_INDEX:
                    label_vector[EMOTION_INDEX[emotion]] = value
                else:
                    # Unknown emotion - try case-insensitive match
                    emotion_lower = emotion.lower()
                    if emotion_lower in EMOTION_INDEX:
                        label_vector[EMOTION_INDEX[emotion_lower]] = value
            # Normalize to ensure sum = 1
            total = label_vector.sum()
            if total > 0:
                label_vector = label_vector / total
            return label_vector

        else:
            raise TypeError(f"Soft label must be list or dict, got {type(label_input)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys:
            - input_ids: (max_len,) token IDs
            - attention_mask: (max_len,) attention mask
            - entity_mask: (max_len,) entity mask (1 for entity tokens)
            - soft_label: (28,) soft label vector
        """
        item = self.samples[idx]
        text = item["text"]

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        # Create entity mask from character offsets
        char_start = item["char_start"]
        char_end = item["char_end"]

        # Handle offset format (list of tuples or tensor)
        if isinstance(offsets, list):
            token_starts = torch.tensor([o[0] for o in offsets])
            token_ends = torch.tensor([o[1] for o in offsets])
        else:
            token_starts = offsets[:, 0]
            token_ends = offsets[:, 1]

        # Entity mask: tokens that overlap with the character span
        entity_mask = (
            (token_starts < char_end)
            & (token_ends > char_start)
            & attention_mask.bool()
        ).float()

        # Convert soft label to vector
        soft_label = self._soft_label_to_vector(item["soft_label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_mask": entity_mask,
            "soft_label": soft_label,
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 32,
    max_len: int = 128,
    num_workers: int = 0,
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        train_path: Path to training data JSONL
        val_path: Path to validation data JSONL
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length
        num_workers: Number of worker processes

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = FineGrainedEmotionDataset(train_path, tokenizer, max_len)
    val_dataset = FineGrainedEmotionDataset(val_path, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def postprocess_llm_output(text: str, llm_output: Dict) -> Dict:
    """
    Post-process LLM output to add character offsets.

    This function implements the V5 fix for LLM character counting hallucination.
    LLM generates content (span_text), Python computes coordinates (char_start, char_end).

    Args:
        text: Original input text
        llm_output: LLM output with targets containing span_text and soft_label

    Returns:
        llm_output with added char_start and char_end for each target
    """
    if "targets" not in llm_output:
        return llm_output

    for target in llm_output["targets"]:
        span_text = target.get("span_text", "")
        if not span_text:
            continue

        # Find character position
        idx = text.find(span_text)
        if idx == -1:
            # Try fuzzy match
            match = re.search(re.escape(span_text), text)
            if match:
                idx = match.start()
            else:
                raise ValueError(f"Span '{span_text}' not found in text")

        target["char_start"] = idx
        target["char_end"] = idx + len(span_text)

    return llm_output


def generate_dummy_data(
    output_path: str,
    num_samples: int = 100,
    max_targets: int = 3,
):
    """
    Generate dummy training data for testing.

    Args:
        output_path: Path to save the generated data
        num_samples: Number of samples to generate
        max_targets: Maximum number of targets per sample
    """
    import random

    # Sample texts
    texts = [
        "I am absolutely furious right now!",
        "The movie was fantastic but acting terrible",
        "I feel so happy and joyful today",
        "This is a terrible disaster",
        "I'm not sure how I feel about this",
        "What a wonderful surprise!",
        "I am deeply disappointed and sad",
        "This is just okay, nothing special",
        "I love this so much!",
        "I hate when this happens",
    ]

    emotions = list(EMOTION_INDEX.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            text = random.choice(texts)

            # Decide number of targets
            num_targets = random.randint(1, max_targets)
            targets = []

            for _ in range(num_targets):
                # Pick random span (word or phrase)
                words = text.split()
                if not words:
                    continue

                start_idx = random.randint(0, len(words) - 1)
                end_idx = min(start_idx + random.randint(1, 2), len(words))
                span_text = " ".join(words[start_idx:end_idx])

                # Find character position
                char_start = text.find(span_text)
                if char_start == -1:
                    continue
                char_end = char_start + len(span_text)

                # Generate soft label
                num_emotions = random.randint(1, 3)
                chosen_emotions = random.sample(emotions, num_emotions)
                probs = [random.random() for _ in range(num_emotions)]
                total = sum(probs)
                probs = [p / total for p in probs]

                soft_label = {chosen_emotions[i]: probs[i] for i in range(num_emotions)}

                targets.append({
                    "span_text": span_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "soft_label": soft_label,
                })

            f.write(json.dumps({"text": text, "targets": targets}))
            f.write("\n")

    print(f"Generated {num_samples} dummy samples to {output_path}")


def generate_goemotions_from_raw(
    raw_data_path: str,
    output_path: str,
):
    """
    Process GoEmotions raw annotations to soft labels.

    GoEmotions raw format has multiple annotator votes per emotion.
    We normalize these to create soft labels.

    Args:
        raw_data_path: Path to GoEmotions raw data
        output_path: Path to save processed JSONL
    """
    # This is a placeholder - actual implementation depends on the specific
    # GoEmotions data format you have
    raise NotImplementedError(
        "GoEmotions raw processing depends on data format. "
        "Use the datasets library: pip install datasets"
    )
