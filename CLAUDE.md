# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Probabilistic G-BERT (System V3)** is an academic research project for Natural Language Processing. It proposes a novel text embedding approach that represents text as **Von Mises-Fisher (vMF) distributions** on hyperspheres rather than deterministic point vectors.

**Core Innovation:** "Intensity implies Certainty" - emotional intensity is modeled as the concentration parameter (kappa) of a vMF distribution, not as a scalar property of a fixed vector.

**Status:** Early planning/documentation phase. No implementation code exists yet.

## Planned Architecture

The model uses a **Bottlenecked Tri-Branch** design:

1. **Branch A (Semantic):** RoBERTa (768d) → bottleneck → 64d unit vector (μ, the mean direction)
2. **Branch B (Mass):** Gravitational Attention → concentration parameter κ (physical quality/uncertainty)
3. **Branch C (Auxiliary):** 64d → 28d logits for emotion classification (prevents semantic collapse)

**Training Objective:** 3-Part Loss
- `L_vMF`: vMF-NCE contrastive loss with dynamic temperature (τ = 1/κ)
- `L_Cal`: Calibration loss (aligns predicted κ with soft-label max-norm intensity)
- `L_Aux`: KL divergence (preserves emotion semantics in bottleneck)

## Key Technical Details

### Soft Label Max-Norm Intensity
- Intensity is defined geometrically: `I_raw = max(soft_label)` where soft_label is a 28-dim probability distribution over GoEmotions categories
- High intensity (sharp distribution, max → 1.0) → high κ → low temperature → high specificity
- Low intensity (flat distribution, max → 0.04) → low κ → high temperature → diverse results

### Physical Scaling
- α_scale = 50.0 maps intensity [0,1] to κ in [1, 51]
- This yields temperature τ ≈ 0.02 for high-intensity samples (optimal for hard negative mining)

### Device Strategy
- Always use `config.device` (dynamically set to "cuda" or "cpu")
- Never hardcode `.cuda()`; use `.to(device)` instead
- Enable `torch.backends.cudnn.benchmark = True` when GPU is available

### Numerical Stability
- Add epsilon (1e-6) when computing τ = 1/(κ + 1e-6) to prevent division by zero
- Use `F.softplus()` or `ELU + 1` to ensure non-negative energies

### Gradient Accumulation
- Effective batch size should be ~256 for contrastive learning
- With GPU memory constraints, use physical batch_size=64 with grad_accum_steps=4

## Planned Directory Structure

```
probabilistic_gbert/
├── data/
│   ├── raw/                     # Raw GoEmotions data
│   ├── processed/               # JSONL with soft labels
│   └── generate_data.py         # LLM soft label generation
├── src/
│   ├── config.py                # Centralized hyperparameters
│   ├── dataset.py               # EmotionDataset with intensity computation
│   ├── model.py                 # ProbabilisticGBERT tri-branch model
│   ├── loss.py                  # GBERTLoss (MATS, calibration, auxiliary)
│   └── utils.py                 # Device detection, metrics, logging
├── checkpoints/                 # Model checkpoints
├── train.py                     # Training entry point
├── inference.py                 # Retrieval/evaluation script
└── requirements.txt
```

## GoEmotions Categories (28 total)

The model uses 28 emotion categories for soft labels:

**Positive (12):** admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief

**Negative (11):** anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness

**Ambiguous/Cognitive (4):** confusion, curiosity, realization, surprise

**Neutral (1):** neutral

## Development Workflow (Planned)

1. Create `requirements.txt` with dependencies
2. Implement `data/generate_data.py` for LLM-based soft label generation
3. Build core modules in order: `config.py` → `model.py` → `loss.py`
4. Implement `train.py` with gradient accumulation and hardware fallback
5. Add `inference.py` for retrieval testing with κ-weighted scoring

## Key References

- **DESIGN.md:** Chinese implementation design document with detailed module specifications
- **PRD.md:** Complete theoretical formulation, math derivations, and reference PyTorch implementation
