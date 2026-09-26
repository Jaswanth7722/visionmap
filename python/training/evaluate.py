"""
Evaluation Module for PointNet++ Semantic Segmentation
Computes confusion matrix, per-class IoU, precision, recall, mIoU, and overall accuracy.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.model import PointNet2SemSeg
from python.training.dataset import SemanticKITTIDataset, CLASS_NAMES


def compute_metrics(confusion_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Computes per-class IoU, Precision, Recall, mean IoU, and Overall Accuracy from confusion matrix.
    Args:
        confusion_matrix: 3x3 ndarray where rows = true, cols = predicted
    """
    num_classes = confusion_matrix.shape[0]
    total_samples = np.sum(confusion_matrix)
    correct_samples = np.trace(confusion_matrix)
    overall_acc = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0

    per_class_iou = {}
    per_class_precision = {}
    per_class_recall = {}
    ious = []

    for c in range(num_classes):
        c_name = CLASS_NAMES.get(c, f"class_{c}")
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp

        denom_iou = tp + fp + fn
        iou = (tp / denom_iou * 100.0) if denom_iou > 0 else 0.0
        per_class_iou[c_name] = round(float(iou), 2)
        ious.append(iou)

        denom_prec = tp + fp
        prec = (tp / denom_prec * 100.0) if denom_prec > 0 else 0.0
        per_class_precision[c_name] = round(float(prec), 2)

        denom_rec = tp + fn
        rec = (tp / denom_rec * 100.0) if denom_rec > 0 else 0.0
        per_class_recall[c_name] = round(float(rec), 2)

    mean_iou = round(float(np.mean(ious)), 2)

    return {
        "overall_accuracy": round(float(overall_acc), 2),
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion_matrix.tolist()
    }


def print_metrics_table(metrics: Dict[str, Any]):
    """Prints a clean ASCII report table for evaluation metrics."""
    print("\n" + "=" * 70)
    print(">>> POINTNET++ SEMANTIC SEGMENTATION EVALUATION REPORT <<<")
    print("=" * 70)
    print(f"Overall Accuracy (OA): {metrics['overall_accuracy']:.2f}%")
    print(f"Mean IoU (mIoU):        {metrics['mean_iou']:.2f}%")
    print("-" * 70)
    print(f"{'Class':<20} | {'IoU (%)':<12} | {'Precision (%)':<15} | {'Recall (%)':<12}")
    print("-" * 70)
    for c_name in ["terrain", "static_obstacle", "dynamic_object"]:
        iou = metrics["per_class_iou"].get(c_name, 0.0)
        prec = metrics["per_class_precision"].get(c_name, 0.0)
        rec = metrics["per_class_recall"].get(c_name, 0.0)
        print(f"{c_name:<20} | {iou:<12.2f} | {prec:<15.2f} | {rec:<12.2f}")
    print("=" * 70 + "\n")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 3
) -> Dict[str, Any]:
    """
    Evaluates model on dataloader.
    Returns dictionary with IoU, Precision, Recall, mIoU, and OA.
    """
    model.eval()
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for xyz, targets in dataloader:
        xyz = xyz.to(device)           # [B, N, 3]
        targets = targets.numpy()      # [B, N]

        # Model forward
        logits = model(xyz)            # [B, N, num_classes]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # [B, N]

        valid_mask = (targets >= 0) & (targets < num_classes)
        valid_targets = targets[valid_mask]
        valid_preds = preds[valid_mask]

        if len(valid_targets) > 0:
            for t, p in zip(valid_targets, valid_preds):
                confusion_mat[t, p] += 1

    metrics = compute_metrics(confusion_mat)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PointNet++ Checkpoint on SemanticKITTI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to SemanticKITTI dataset root")
    parser.add_argument("--sequences", nargs="+", default=["08"], help="Sequences for evaluation (default: 08)")
    parser.add_argument("--num-points", type=int, default=4096, help="Points per cloud")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--output-json", type=str, default="evaluation_metrics.json", help="Path to save metrics JSON")
    parser.add_argument("--no-label-download", action="store_true",
                        help="Do not attempt to download SemanticKITTI labels when missing "
                             "(useful offline; evaluation then refuses without ground truth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load model
    model = PointNet2SemSeg(num_classes=3, in_channels=3)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    # Load dataset
    val_dataset = SemanticKITTIDataset(
        root_dir=args.data_dir,
        sequences=args.sequences,
        num_points=args.num_points,
        split="val",
        auto_download_labels=not args.no_label_download,
    )

    # H7: never score a model against its own heuristic output. Without
    # ground-truth .label files the "targets" would be geometric guesses, and
    # any accuracy number computed from them is circular. Refuse loudly and
    # write no metrics file, so no fabricated report can circulate.
    if not val_dataset.has_ground_truth:
        print("=" * 70)
        print("NO GROUND TRUTH AVAILABLE -- evaluation refused.")
        print(f"Sequences {args.sequences} under {args.data_dir} contain no .label files;")
        print("scoring against heuristic labels would be circular validation.")
        print("Provide SemanticKITTI .label files (or run with labels) and retry.")
        print("=" * 70)
        sys.exit(2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )

    metrics = evaluate_model(model, val_loader, device=device, num_classes=3)
    print_metrics_table(metrics)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output_json}")
