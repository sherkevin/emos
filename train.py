#!/usr/bin/env python
"""
Training script for Probabilistic G-BERT V4.

Usage:
    python train.py --train_data data/train.jsonl --val_data data/val.jsonl

Features:
- Automatic hardware detection (GPU/CPU)
- Gradient accumulation for effective batch size
- Early stopping with checkpoint saving
- WandB logging (optional)
- Resume from checkpoint
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import ProbabilisticGBERTV4
from src.loss import ProbabilisticGBERTLoss
from src.dataset import FineGrainedEmotionDataset, generate_dummy_data
from src.utils import Logger, set_seed, get_device, MetricsTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Probabilistic G-BERT V4")

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train.jsonl",
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/val.jsonl",
        help="Path to validation data JSONL",
    )
    parser.add_argument(
        "--generate_dummy",
        action="store_true",
        help="Generate dummy training data for testing",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Bottleneck embedding dimension",
    )
    parser.add_argument(
        "--num_emotions",
        type=int,
        default=28,
        help="Number of emotion categories",
    )
    parser.add_argument(
        "--alpha_scale",
        type=float,
        default=50.0,
        help="Physical scaling coefficient for kappa",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Physical batch size",
    )
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=256,
        help="Target effective batch size (for gradient accumulation)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=2e-5,
        help="Learning rate for backbone",
    )
    parser.add_argument(
        "--lr_heads",
        type=float,
        default=1e-4,
        help="Learning rate for projection heads",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience",
    )

    # Loss weights
    parser.add_argument(
        "--lambda_cal",
        type=float,
        default=0.1,
        help="Calibration loss weight",
    )
    parser.add_argument(
        "--lambda_aux",
        type=float,
        default=0.05,
        help="Auxiliary loss weight",
    )

    # System arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Other arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log every N steps",
    )

    return parser.parse_args()


def validate_model(model, dataloader, criterion, device, config):
    """
    Validate the model.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        config: Configuration object

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    metrics = MetricsTracker()
    all_kappas = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_mask = batch["entity_mask"].to(device)
            soft_labels = batch["soft_label"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, entity_mask)

            # Compute loss
            loss, loss_dict = criterion(
                outputs["mu"],
                outputs["kappa"],
                outputs["aux_logits"],
                soft_labels,
            )

            # Update metrics
            batch_size = input_ids.shape[0]
            metrics.update("total", loss_dict["total"], batch_size)
            metrics.update("vmf", loss_dict["vmf"], batch_size)
            metrics.update("cal", loss_dict["cal"], batch_size)
            metrics.update("aux", loss_dict["aux"], batch_size)

            # Track kappa values (flatten (B, 1) to list)
            all_kappas.extend(outputs["kappa"].squeeze(-1).cpu().tolist())

    # Compute average kappa
    avg_kappa = sum(all_kappas) / len(all_kappas)

    return {
        "loss": metrics.get("total"),
        "loss_vmf": metrics.get("vmf"),
        "loss_cal": metrics.get("cal"),
        "loss_aux": metrics.get("aux"),
        "avg_kappa": avg_kappa,
    }


def train_epoch(
    model, dataloader, criterion, optimizer, scheduler, device, epoch, config, logger
):
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch number
        config: Configuration object
        logger: Logger instance

    Returns:
        Dictionary with training metrics
    """
    model.train()

    metrics = MetricsTracker()
    num_steps = len(dataloader)
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        entity_mask = batch["entity_mask"].to(device)
        soft_labels = batch["soft_label"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask, entity_mask)

        # Compute loss
        loss, loss_dict = criterion(
            outputs["mu"],
            outputs["kappa"],
            outputs["aux_logits"],
            soft_labels,
        )

        # Scale loss for gradient accumulation
        loss = loss / config.grad_accum_steps

        # Backward pass
        loss.backward()

        # Update weights
        if (step + 1) % config.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics
        batch_size = input_ids.shape[0]
        metrics.update("total", loss_dict["total"] * config.grad_accum_steps, batch_size)
        metrics.update("vmf", loss_dict["vmf"] * config.grad_accum_steps, batch_size)
        metrics.update("cal", loss_dict["cal"] * config.grad_accum_steps, batch_size)
        metrics.update("aux", loss_dict["aux"] * config.grad_accum_steps, batch_size)

        # Log progress
        if (step + 1) % config.log_interval == 0:
            global_step = epoch * num_steps + step
            lr = scheduler.get_last_lr()[0]

            logger.log_metrics(
                {
                    "train/loss": metrics.get("total"),
                    "train/loss_vmf": metrics.get("vmf"),
                    "train/loss_cal": metrics.get("cal"),
                    "train/loss_aux": metrics.get("aux"),
                    "train/lr": lr,
                },
                step=global_step,
            )

    return {
        "loss": metrics.get("total"),
        "loss_vmf": metrics.get("vmf"),
        "loss_cal": metrics.get("cal"),
        "loss_aux": metrics.get("aux"),
    }


def main():
    """Main training function."""
    args = parse_args()

    # Create config
    config = Config(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        num_emotions=args.num_emotions,
        alpha_scale=args.alpha_scale,
        max_length=args.max_length,
        physical_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        lambda_cal=args.lambda_cal,
        lambda_aux=args.lambda_aux,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Override device if specified
    if args.device:
        config.device = args.device
    if args.no_cuda:
        config.device = "cpu"

    # Setup device
    device = config.device_actual
    logger = Logger(use_wandb=args.use_wandb)

    logger.info("=" * 60)
    logger.info("Probabilistic G-BERT V4 - Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Effective batch size: {config.effective_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.grad_accum_steps}")
    logger.log_config(config.to_dict())

    # Set random seed
    set_seed(args.seed)

    # Generate dummy data if requested
    if args.generate_dummy:
        logger.info("Generating dummy training data...")
        os.makedirs("data", exist_ok=True)
        generate_dummy_data(args.train_data, num_samples=100)
        generate_dummy_data(args.val_data, num_samples=20)

    # Check data files exist
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Run with --generate_dummy to create test data")
        return

    if not os.path.exists(args.val_data):
        logger.info(f"Validation data not found: {args.val_data}")
        logger.info("Using training data for validation")
        args.val_data = args.train_data

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = FineGrainedEmotionDataset(
        args.train_data, tokenizer, config.max_length
    )
    val_dataset = FineGrainedEmotionDataset(
        args.val_data, tokenizer, config.max_length
    )
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.physical_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Create model
    logger.info("Creating model...")
    model = ProbabilisticGBERTV4(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        num_emotions=config.num_emotions,
        alpha_scale=config.alpha_scale,
        use_hf_mirror=True,
    )

    # Create loss function
    criterion = ProbabilisticGBERTLoss(
        num_emotions=config.num_emotions,
        embedding_dim=config.embedding_dim,
        lambda_cal=config.lambda_cal,
        lambda_aux=config.lambda_aux,
        alpha_scale=config.alpha_scale,
    )

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.lr_backbone},
            {"params": model.semantic_head.parameters(), "lr": config.lr_heads},
            {"params": model.energy_proj.parameters(), "lr": config.lr_heads},
            {"params": model.aux_head.parameters(), "lr": config.lr_heads},
            {"params": criterion.vmf_loss.prototypes, "lr": config.lr_heads},
        ],
        weight_decay=config.weight_decay,
    )

    # Create scheduler
    total_steps = len(train_loader) // config.grad_accum_steps * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, config, logger
        )

        logger.info(f"Train Loss: {train_metrics['loss']:.4f} "
                   f"(vmf={train_metrics['loss_vmf']:.4f}, "
                   f"cal={train_metrics['loss_cal']:.4f}, "
                   f"aux={train_metrics['loss_aux']:.4f})")

        # Validate
        val_metrics = validate_model(model, val_loader, criterion, device, config)

        logger.info(f"Val Loss: {val_metrics['loss']:.4f} "
                   f"(vmf={val_metrics['loss_vmf']:.4f}, "
                   f"cal={val_metrics['loss_cal']:.4f}, "
                   f"aux={val_metrics['loss_aux']:.4f})")
        logger.info(f"Avg Kappa: {val_metrics['avg_kappa']:.2f}")

        # Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epoch(s)")

        # Save last checkpoint
        last_checkpoint_path = os.path.join(config.checkpoint_dir, "last_checkpoint.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "config": config.to_dict(),
        }, last_checkpoint_path)

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered (patience={config.patience})")
            break

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

    logger.close()


if __name__ == "__main__":
    main()
