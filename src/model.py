"""
Model architecture for Probabilistic G-BERT V4: Entity-Aware.

Implements the Token-Level Bottlenecked Tri-Branch architecture:
- Branch A: Token-Level Semantic Core → Late Pooling → μ (64d)
- Branch B: Token-Level Energy → Max Pooling → κ (1d)
- Branch C: Auxiliary Semantic Head (64 → 28)

V4 Key Features:
- Project-then-Pool architecture (enables entity-level inference)
- Max Pooling for κ (intensive property, not extensive)
- Empty Mask protection for robust entity-level inference
"""

import warnings
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_hf_mirror(endpoint: str = "https://hf-mirror.com"):
    """
    Setup HuggingFace mirror for faster downloads in China.

    Args:
        endpoint: Mirror endpoint URL
    """
    import os

    os.environ["HF_ENDPOINT"] = endpoint


class ProbabilisticGBERTV4(nn.Module):
    """
    Probabilistic G-BERT V4: Entity-Aware vMF Distribution for Text Embedding.

    Architecture: Token-Level Bottlenecked Tri-Branch
    - Branch A: Token-Level Semantic Core → Late Pooling → μ (64d)
    - Branch B: Token-Level Energy → Max Pooling → κ (1d)
    - Branch C: Auxiliary Semantic Head (64 → 28)

    V4 Update: Projection-First architecture
    - Process each token independently to get (B, L, 64) vectors
    - Support both Sentence-Level and Entity-Level inference via flexible masking

    Args:
        model_name: HuggingFace model name (default: "roberta-base")
        embedding_dim: Bottleneck dimension (default: 64)
        num_emotions: Number of emotion categories (default: 28)
        alpha_scale: Physical scaling coefficient for κ (default: 50.0)
        use_hf_mirror: Whether to use HF mirror for download (default: True)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        embedding_dim: int = 64,
        num_emotions: int = 28,
        alpha_scale: float = 50.0,
        use_hf_mirror: bool = True,
    ):
        super().__init__()

        if use_hf_mirror:
            setup_hf_mirror()

        # Load backbone
        try:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
        except ImportError:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
        self.alpha_scale = alpha_scale
        self.model_name = model_name

        # Branch A: Token-Level Semantic Bottleneck (768 → 64 per token)
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        )

        # Branch B: Token-Level Energy Projection (768 → 1 per token)
        self.energy_proj = nn.Linear(hidden_size, 1)

        # Branch C: Auxiliary Semantic Head (64 → 28)
        self.aux_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_emotions),
        )

    def _pool_tokens(
        self,
        token_vectors: torch.Tensor,
        mask: torch.Tensor,
        pooling_type: str = "mean",
    ) -> torch.Tensor:
        """
        Pool token vectors using the specified mask.

        Args:
            token_vectors: (B, L, D) token-level vectors
            mask: (B, L) boolean mask (True = include in pooling)
            pooling_type: 'mean', 'sum', or 'max'

        Returns:
            (B, D) pooled vectors
        """
        device = token_vectors.device
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)

        # Check for empty masks
        valid_counts = mask.sum(dim=1)  # (B,)
        empty_mask = (valid_counts == 0)  # (B,) bool

        if pooling_type == "mean":
            # Mean pooling: sum / count
            summed = torch.sum(token_vectors * mask_expanded, dim=1)
            count = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            pooled = summed / count
        elif pooling_type == "sum":
            # Sum pooling
            pooled = torch.sum(token_vectors * mask_expanded, dim=1)
        else:  # 'max'
            # Max pooling: set mask=0 positions to -inf
            masked = torch.where(
                mask_expanded.bool(),
                token_vectors,
                torch.tensor(float("-inf"), device=device),
            )
            pooled, _ = masked.max(dim=1)
            # Replace -inf with 0 for empty masks
            pooled = torch.where(empty_mask.unsqueeze(-1), torch.zeros_like(pooled), pooled)

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with support for entity-level inference.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask (1=valid, 0=pad)
            entity_mask: (B, L) optional entity mask (True=entity token)

        Returns:
            Dictionary containing:
            - mu: (B, embedding_dim) sentence semantic direction
            - kappa: (B, 1) sentence concentration parameter
            - mu_entity: (B, embedding_dim) entity semantic direction (if entity_mask provided)
            - kappa_entity: (B, 1) entity concentration (if entity_mask provided)
            - aux_logits: (B, num_emotions) auxiliary logits
        """
        # Backbone encoding
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, hidden_size)

        # Convert attention_mask to boolean (True = valid token)
        valid_mask = attention_mask.bool()  # (B, L)

        # ===== Branch A: Token-Level Semantic Vectors =====
        # Project each token to embedding_dim
        token_vectors = self.semantic_head(last_hidden)  # (B, L, embedding_dim)

        # Sentence-level pooling (default)
        mu = self._pool_tokens(token_vectors, valid_mask, pooling_type="mean")
        mu = F.normalize(mu, p=2, dim=1)  # (B, embedding_dim), ||μ|| = 1

        # Entity-level pooling (optional) with Empty Mask protection
        mu_entity = None
        kappa_entity = None
        entity_mask_safe = None

        if entity_mask is not None:
            # Check for empty entity masks
            empty_mask = (entity_mask.sum(dim=1) == 0)  # (B,) bool

            if empty_mask.any():
                warnings.warn(
                    f"Empty entity_mask detected for {empty_mask.sum().item()} samples. "
                    "Falling back to sentence-level pooling."
                )
                # Create safe mask
                entity_mask_safe = entity_mask.clone()
                entity_mask_safe[empty_mask] = valid_mask[empty_mask]
            else:
                entity_mask_safe = entity_mask

            mu_entity = self._pool_tokens(token_vectors, entity_mask_safe, pooling_type="mean")
            mu_entity = F.normalize(mu_entity, p=2, dim=1)

        # ===== Branch B: Token-Level Energy =====
        token_energies = F.softplus(self.energy_proj(last_hidden))  # (B, L, 1)

        # Sentence-level energy aggregation (max pooling - concentration is intensive)
        energy_sentence = self._pool_tokens(token_energies, valid_mask, pooling_type="max")
        kappa = 1.0 + self.alpha_scale * energy_sentence  # (B, 1)

        # Entity-level energy aggregation (optional)
        if entity_mask_safe is not None:
            energy_entity = self._pool_tokens(token_energies, entity_mask_safe, pooling_type="max")
            kappa_entity = 1.0 + self.alpha_scale * energy_entity  # (B, 1)

        # ===== Branch C: Auxiliary logits (uses sentence-level mu) =====
        aux_logits = self.aux_head(mu)  # (B, num_emotions)

        result = {
            "mu": mu,
            "kappa": kappa,
            "aux_logits": aux_logits,
        }

        if mu_entity is not None:
            result["mu_entity"] = mu_entity
        if kappa_entity is not None:
            result["kappa_entity"] = kappa_entity

        return result

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim

    def get_num_emotions(self) -> int:
        """Return the number of emotion categories."""
        return self.num_emotions


class GbertPredictor:
    """
    Predictor class for inference with Probabilistic G-BERT V4.

    Supports both sentence-level and entity-level emotion analysis.
    """

    def __init__(
        self,
        model: ProbabilisticGBERTV4,
        tokenizer,
        device: str = "cpu",
    ):
        """
        Initialize the predictor.

        Args:
            model: Trained ProbabilisticGBERTV4 model
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = "roberta-base",
        device: str = "cpu",
    ):
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model_name: HuggingFace model name
            device: Device to run inference on

        Returns:
            GbertPredictor instance
        """
        from transformers import AutoTokenizer

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ProbabilisticGBERTV4(model_name=model_name)
        model.load_state_dict(checkpoint)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(model, tokenizer, device)

    def _create_entity_mask(
        self, text: str, span_text: str, encoding: dict
    ) -> torch.Tensor:
        """
        Create entity mask through character alignment.

        Args:
            text: Input text
            span_text: Entity text to find
            encoding: Tokenizer output with offset_mapping

        Returns:
            entity_mask: (L,) tensor, 1 for entity tokens, 0 otherwise
        """
        # Find character offsets
        idx = text.find(span_text)
        if idx == -1:
            # Try fuzzy match with regex
            import re

            match = re.search(re.escape(span_text), text)
            if match:
                idx = match.start()
            else:
                raise ValueError(f"Span '{span_text}' not found in text")

        c_start, c_end = idx, idx + len(span_text)

        # Get token offsets
        offsets = encoding["offset_mapping"]  # List of (start, end) tuples

        # Handle both list and tensor formats
        if isinstance(offsets, list):
            # Remove batch dimension if present
            if len(offsets) > 0 and isinstance(offsets[0], list):
                offsets = offsets[0]
            token_starts = torch.tensor([o[0] for o in offsets])
            token_ends = torch.tensor([o[1] for o in offsets])
        else:
            # Tensor format
            if offsets.dim() > 2:
                offsets = offsets.squeeze(0)
            token_starts = offsets[:, 0]
            token_ends = offsets[:, 1]

        # Get attention mask
        attention_mask = encoding["attention_mask"]
        if isinstance(attention_mask, list):
            if len(attention_mask) > 0 and isinstance(attention_mask[0], list):
                attention_mask = attention_mask[0]
            attention_mask = torch.tensor(attention_mask)
        else:
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)

        # Create entity mask (tokens that overlap with the span)
        entity_mask = (token_starts < c_end) & (token_ends > c_start) & attention_mask.bool()

        return entity_mask.float()

    def predict(
        self, text: str, span_text: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Predict emotion for text or entity.

        Args:
            text: Input text
            span_text: Entity text for entity-level analysis (optional)

        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Create entity mask if needed
        entity_mask = None
        if span_text is not None:
            try:
                entity_mask = self._create_entity_mask(text, span_text, encoding)
            except ValueError as e:
                warnings.warn(f"Entity masking failed: {e}. Falling back to sentence-level.")
                entity_mask = None

        # Prepare inputs
        inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

        if entity_mask is not None:
            inputs["entity_mask"] = entity_mask.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Format results
        if span_text is not None and "mu_entity" in outputs:
            mu = outputs["mu_entity"]
            kappa = outputs["kappa_entity"]
            mode = "entity"
        else:
            mu = outputs["mu"]
            kappa = outputs["kappa"]
            mode = "sentence"

        return {
            "text": text,
            "mode": mode,
            "span_text": span_text,
            "mu": mu.cpu().squeeze(0).tolist(),
            "kappa": kappa.cpu().item(),
        }
