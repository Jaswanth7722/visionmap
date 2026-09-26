"""
Unit tests for SemanticKITTI dataset pipeline & label remapping.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.dataset import (
    SemanticKITTIDataset,
    remap_labels_vectorized,
    SEMANTICKITTI_REMAP_TABLE,
    CLASS_NAMES,
    compute_dataset_class_distribution
)


def test_remap_labels_vectorized():
    # Test specific key classes
    raw = np.array([0, 1, 10, 40, 50, 70, 252, 999], dtype=np.uint32)
    remapped = remap_labels_vectorized(raw)
    assert remapped[0] == -1  # 0 -> unlabeled
    assert remapped[1] == -1  # 1 -> outlier
    assert remapped[2] == 2   # 10 -> car (dynamic_object)
    assert remapped[3] == 0   # 40 -> road (terrain)
    assert remapped[4] == 1   # 50 -> building (static_obstacle)
    assert remapped[5] == 1   # 70 -> vegetation (static_obstacle)
    assert remapped[6] == 2   # 252 -> moving-car (dynamic_object)
    assert remapped[7] == -1  # unmapped -> ignore


def test_dataset_loading_and_sampling():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sequence 00 directory structure
        seq_dir = Path(temp_dir) / "sequences" / "00"
        velo_dir = seq_dir / "velodyne"
        lbl_dir = seq_dir / "labels"
        velo_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        num_points = 5000
        # Write dummy .bin (float32 [M, 4])
        raw_pts = np.random.randn(num_points, 4).astype(np.float32)
        raw_pts.tofile(str(velo_dir / "000000.bin"))

        # Write dummy .label (uint32 [M])
        # Mix of terrain (40), static (50), dynamic (10), unlabeled (0)
        labels_raw = np.random.choice([0, 10, 40, 50, 70], size=num_points).astype(np.uint32)
        labels_raw.tofile(str(lbl_dir / "000000.label"))

        # Instantiate dataset with N=2048
        ds = SemanticKITTIDataset(root_dir=temp_dir, sequences=["00"], num_points=2048)
        assert len(ds) == 1

        xyz, lbl = ds[0]
        assert xyz.shape == (2048, 3)
        assert lbl.shape == (2048,)
        assert xyz.dtype == torch.float32
        assert lbl.dtype == torch.int64

        # Verify class range is within {-1, 0, 1, 2}
        unique_classes = set(lbl.numpy().tolist())
        assert unique_classes.issubset({-1, 0, 1, 2})

        # Distribution calculation
        stats = compute_dataset_class_distribution(ds, num_samples=1)
        assert stats["total_points"] == 2048
        assert len(stats["recommended_weights"]) == 3
        # Ground-truth provenance is flagged for consumers (H7).
        assert ds.has_ground_truth is True
        assert ds.label_provenance == "ground-truth"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_heuristic_label_provenance_without_label_files():
    """Scans without .label files must be flagged heuristic, never silently
    presented as ground truth (H7: blocks circular evaluation)."""
    temp_dir = tempfile.mkdtemp()
    try:
        seq_dir = Path(temp_dir) / "sequences" / "00" / "velodyne"
        seq_dir.mkdir(parents=True, exist_ok=True)
        raw_pts = np.random.randn(500, 4).astype(np.float32)
        raw_pts.tofile(str(seq_dir / "000000.bin"))

        ds = SemanticKITTIDataset(
            root_dir=temp_dir, sequences=["00"], num_points=64,
            auto_download_labels=False,
        )
        assert ds.has_ground_truth is False
        assert ds.label_provenance == "heuristic"
        xyz, lbl = ds[0]
        assert xyz.shape == (64, 3)
        assert set(lbl.numpy().tolist()).issubset({-1, 0, 1, 2})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
