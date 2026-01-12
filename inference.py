#!/usr/bin/env python
"""
Inference script for Probabilistic G-BERT V4.

Usage:
    # Interactive mode
    python inference.py --checkpoint checkpoints/best_model.pt

    # Single sentence
    python inference.py --checkpoint checkpoints/best_model.pt --text "I am furious!"

    # Entity-level
    python inference.py --checkpoint checkpoints/best_model.pt --text "The movie was fantastic" --entity "fantastic"

Features:
- Sentence-level emotion analysis
- Entity-level emotion analysis
- Interactive demo mode
- Batch processing from file
"""

import argparse
import json
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, INDEX_TO_EMOTION
from src.model import ProbabilisticGBERTV4, GbertPredictor
from src.utils import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Probabilistic G-BERT V4")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="HuggingFace model name (default: roberta-base)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text for analysis",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Entity span for entity-level analysis",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with texts (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Show top K emotions",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    return parser.parse_args()


def format_result(result: dict, top_k: int = 5) -> str:
    """Format inference result for display."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Mode: {result['mode'].upper()}")
    lines.append(f"Text: {result['text']}")

    if result['mode'] == 'entity':
        lines.append(f"Entity: {result['span_text']}")

    lines.append(f"Kappa (Concentration): {result['kappa']:.2f}")

    # Interpret kappa
    kappa = result['kappa']
    if kappa > 40:
        lines.append(f"Intensity: HIGH (Solid State - Precise)")
    elif kappa > 20:
        lines.append(f"Intensity: MEDIUM")
    else:
        lines.append(f"Intensity: LOW (Gas State - Diverse)")

    lines.append("-" * 60)

    # Show top emotions
    mu = result['mu']
    lines.append(f"Embedding: {len(mu)}d vector")
    lines.append(f"First 5 dims: {[f'{x:.4f}' for x in mu[:5]]}")

    lines.append("=" * 60)
    return "\n".join(lines)


def interactive_mode(predictor: GbertPredictor, top_k: int):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("Probabilistic G-BERT V4 - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  <text>           - Analyze a sentence")
    print("  <text> -- <span>  - Analyze an entity in the sentence")
    print("  quit             - Exit")
    print("\n" + "-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Check for entity span
            span_text = None
            if " -- " in user_input:
                parts = user_input.split(" -- ")
                text = parts[0]
                span_text = parts[1]
            else:
                text = user_input

            # Run prediction
            result = predictor.predict(text, span_text=span_text)
            print(format_result(result, top_k))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_processing(
    predictor: GbertPredictor,
    input_file: str,
    output_file: str = None,
):
    """Process texts from a file."""
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue

            try:
                result = predictor.predict(text)
                results.append(result)
                print(f"[{line_num}] {text[:50]}... -> kappa={result['kappa']:.2f}")
            except Exception as e:
                print(f"[{line_num}] Error processing: {e}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {output_file}")

    return results


def main():
    """Main inference function."""
    args = parse_args()

    # Setup device
    device = args.device if args.device else get_device()

    print("=" * 60)
    print("Probabilistic G-BERT V4 - Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")

    # Load predictor
    print("\nLoading model...")
    predictor = GbertPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=device,
    )
    print("Model loaded successfully!")

    # Run appropriate mode
    if args.interactive:
        interactive_mode(predictor, args.top_k)

    elif args.input_file:
        print(f"\nProcessing file: {args.input_file}")
        batch_processing(predictor, args.input_file, args.output_file)

    elif args.text:
        result = predictor.predict(args.text, span_text=args.entity)
        print(format_result(result, args.top_k))

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output_file}")

    else:
        print("\nNo input specified. Use --text, --input_file, or --interactive")
        print("Example: python inference.py --checkpoint checkpoints/best_model.pt --interactive")


if __name__ == "__main__":
    main()
