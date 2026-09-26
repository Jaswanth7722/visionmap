"""
PointNet++ Training Pipeline for SemanticKITTI
Optimized for Kaggle 2x T4 GPUs (32GB VRAM total), Mixed Precision (AMP fp16),
Multi-GPU (DataParallel), Frequent Checkpointing, and Auto-Resume.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.model import PointNet2SemSeg
from python.training.dataset import (
    SemanticKITTIDataset,
    compute_dataset_class_distribution,
    CLASS_NAMES
)
from python.training.evaluate import evaluate_model, print_metrics_table


def get_gpu_memory_info() -> List[Dict[str, float]]:
    """Returns memory allocated and reserved for each available CUDA GPU in MB."""
    info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            info.append({"gpu_id": i, "name": torch.cuda.get_device_name(i), "allocated_mb": allocated, "reserved_mb": reserved})
    return info


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 20
) -> float:
    """Trains model for one epoch using mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (xyz, targets) in enumerate(train_loader):
        xyz = xyz.to(device, non_blocking=True)          # [B, N, 3]
        targets = targets.to(device, non_blocking=True)  # [B, N]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(xyz)  # [B, N, num_classes]
            # Flatten for cross-entropy
            B, N, C = logits.shape
            loss = criterion(logits.view(B * N, C), targets.view(B * N))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] Step {batch_idx}: NaN or Inf loss encountered! Skipping backward step.")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            gpu_mem = get_gpu_memory_info()
            mem_str = " | ".join([f"GPU{g['gpu_id']}: {g['allocated_mb']:.0f}MB" for g in gpu_mem])
            print(
                f"[Epoch {epoch:02d}] Step [{batch_idx+1:04d}/{len(train_loader):04d}] "
                f"Loss: {loss.item():.4f} | Avg Loss: {total_loss / num_batches:.4f} | "
                f"Speed: {rate:.1f} batch/s | {mem_str}"
            )

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_miou: float,
    metrics: Optional[Dict[str, Any]] = None
):
    """Saves model state, optimizer, scaler, and training metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    state = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "metrics": metrics or {}
    }
    torch.save(state, checkpoint_path)
    print(f"[Checkpoint] Successfully saved to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[int, float]:
    """Loads checkpoint and returns (start_epoch, best_miou)."""
    if not os.path.exists(checkpoint_path):
        print(f"[Resume] No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 1, 0.0

    print(f"[Resume] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    raw_model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[Resume] Optimizer state restored.")
        except Exception as e:
            print(f"[Resume Warning] Could not restore optimizer state: {e}")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("[Resume] GradScaler state restored.")
        except Exception as e:
            print(f"[Resume Warning] Could not restore scaler state: {e}")

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_miou = checkpoint.get("best_miou", 0.0)
    print(f"[Resume] Resuming at Epoch {start_epoch} with prior best mIoU: {best_miou:.2f}%")
    return start_epoch, best_miou


def run_pipeline(args):
    # Determine devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[Setup] Available compute device: {device} (Total GPUs: {num_gpus})")
    for g in get_gpu_memory_info():
        print(f"  - Device {g['gpu_id']}: {g['name']}")

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset setup
    train_subsample = args.subsample if not args.smoke_run else 0.05
    val_subsample = args.subsample if not args.smoke_run else 0.05

    print(f"[Dataset] Loading SemanticKITTI from: {args.data_dir}")
    train_dataset = SemanticKITTIDataset(
        root_dir=args.data_dir,
        sequences=args.train_sequences,
        num_points=args.num_points,
        split="train",
        subsample_ratio=train_subsample
    )
    val_dataset = SemanticKITTIDataset(
        root_dir=args.data_dir,
        sequences=args.val_sequences,
        num_points=args.num_points,
        split="val",
        subsample_ratio=val_subsample
    )

    print(f"[Dataset] Train scans: {len(train_dataset)} | Val scans: {len(val_dataset)}")

    # Class distribution analysis
    print("[Dataset] Inspecting class distribution...")
    dist_stats = compute_dataset_class_distribution(train_dataset, num_samples=min(100, len(train_dataset)))
    print("Class percentages in sample:")
    for c_name, pct in dist_stats["class_percentages"].items():
        print(f"  - {c_name:<16}: {pct:5.2f}%")
    if dist_stats["underrepresented_classes"]:
        print(f"[WARNING] Underrepresented classes (<2%): {dist_stats['underrepresented_classes']}")
        print(f"[Weighting] Applying recommended class weights: {dist_stats['recommended_weights']}")
        class_weights = torch.tensor(dist_stats["recommended_weights"], dtype=torch.float32, device=device)
    else:
        class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    # Dataloaders
    batch_size = args.batch_size
    num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 2. Model setup
    model = PointNet2SemSeg(num_classes=3, in_channels=args.in_channels).to(device)

    # Multi-GPU DataParallel wrapper
    if num_gpus > 1:
        print(f"[Multi-GPU] Wrapping model with torch.nn.DataParallel across {num_gpus} GPUs...")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Auto-resume
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pth")
    start_epoch, best_miou = load_checkpoint(latest_ckpt_path, model, optimizer, scaler, device=device)

    total_epochs = 2 if args.smoke_run else args.epochs
    epoch_times = []

    print("\n" + "=" * 60)
    print(f">>> STARTING TRAINING: Epochs {start_epoch} to {total_epochs} (Smoke: {args.smoke_run}) <<<")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval
        )

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        print(f"\n[Epoch {epoch:02d} Complete] Avg Loss: {train_loss:.4f} | Time: {epoch_duration:.1f}s")

        # Validation
        print(f"[Epoch {epoch:02d}] Running Validation on {len(val_dataset)} scans...")
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        val_metrics = evaluate_model(raw_model, val_loader, device=device, num_classes=3)
        print_metrics_table(val_metrics)

        current_miou = val_metrics["mean_iou"]

        # Save latest checkpoint
        save_checkpoint(latest_ckpt_path, model, optimizer, scaler, epoch, best_miou, val_metrics)

        # Save best checkpoint
        if current_miou > best_miou:
            print(f">>> New best mIoU: {current_miou:.2f}% (Previous: {best_miou:.2f}%). Updating checkpoint_best.pth <<<")
            best_miou = current_miou
            save_checkpoint(best_ckpt_path, model, optimizer, scaler, epoch, best_miou, val_metrics)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            periodic_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(periodic_path, model, optimizer, scaler, epoch, best_miou, val_metrics)

        # In smoke run mode, report projection and verify round-trip
        if args.smoke_run:
            print("\n" + "*" * 60)
            print(">>> SMOKE RUN TIMING & PROJECTION <<<")
            print(f"Measured time for 1 smoke epoch: {epoch_duration:.1f}s")
            full_epoch_est = (epoch_duration / max(train_subsample, 0.01))
            total_proj_min = (full_epoch_est * args.epochs) / 60.0
            print(f"Projected time per FULL epoch (100% data): {full_epoch_est:.1f}s ({full_epoch_est/60:.1f} min)")
            print(f"Projected total training time ({args.epochs} epochs): {total_proj_min:.1f} minutes ({total_proj_min/60:.2f} hours)")
            print("*" * 60 + "\n")
            break

    # Persist final best checkpoint to output dir
    final_output_ckpt = os.path.join(args.output_dir, "pointnet2_trained_best.pth")
    if os.path.exists(best_ckpt_path):
        import shutil
        shutil.copyfile(best_ckpt_path, final_output_ckpt)
        print(f"[Done] Copied durable best checkpoint to: {final_output_ckpt}")

    print("\n==========================================")
    print(">>> TRAINING PIPELINE EXECUTION FINISHED <<<")
    print(f"Best mIoU achieved: {best_miou:.2f}%")
    print("==========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet++ SemanticKITTI Training on Kaggle 2x T4")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/semantickitti", help="SemanticKITTI dataset root path")
    parser.add_argument("--train-sequences", nargs="+", default=["00"], help="Training sequences (default: 00)")
    parser.add_argument("--val-sequences", nargs="+", default=["08"], help="Validation sequences (default: 08)")
    parser.add_argument("--checkpoint-dir", type=str, default="/kaggle/working/checkpoints", help="Checkpoint dir")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/models", help="Output model directory")
    parser.add_argument("--num-points", type=int, default=4096, help="Point count per cloud (default: 4096)")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels (default: 3 for xyz)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size across GPUs (default: 16)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of full training epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio of scans (default: 1.0)")
    parser.add_argument("--save-every", type=int, default=1, help="Save periodic checkpoint every N epochs")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log outputs")
    parser.add_argument("--smoke-run", action="store_true", help="Run 1-2 smoke epochs on slice to verify pipeline")
    args = parser.parse_args()

    run_pipeline(args)
