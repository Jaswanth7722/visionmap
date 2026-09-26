"""
SemanticKITTI Dataset Pipeline for PointNet++ Semantic Segmentation
Handles .bin (Velodyne point cloud) and .label (per-point semantic/instance annotations).
Remaps native 30+ SemanticKITTI classes to 3 project classes:
  0: terrain
  1: static_obstacle
  2: dynamic_object
  -1: ignore (unlabeled, outlier)
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# EXPLICIT SEMANTICKITTI CLASS REMAPPING TABLE
# ==============================================================================
# Native SemanticKITTI label IDs:
# 0: unlabeled, 1: outlier
# 10: car, 11: bicycle, 13: bus, 15: motorcycle, 16: on-rails, 18: truck, 20: other-vehicle
# 30: person, 31: bicyclist, 32: motorcyclist
# 40: road, 44: parking, 48: sidewalk, 49: other-ground, 60: lane-marking
# 50: building, 51: fence, 52: other-structure, 70: vegetation, 71: trunk, 80: pole, 81: traffic-sign, 99: other-object
# 252-259: moving variants
SEMANTICKITTI_REMAP_TABLE: Dict[int, int] = {
    # 0: terrain (road, sidewalk, terrain, parking, other-ground, lane-marking)
    40: 0,   # road
    44: 0,   # parking
    48: 0,   # sidewalk
    49: 0,   # other-ground
    60: 0,   # lane-marking
    72: 0,   # terrain

    # 1: static_obstacle (building, fence, pole, wall, vegetation, traffic-sign, trunk, other-structure)
    50: 1,   # building
    51: 1,   # fence
    52: 1,   # other-structure
    70: 1,   # vegetation
    71: 1,   # trunk
    80: 1,   # pole
    81: 1,   # traffic-sign
    99: 1,   # other-object

    # 2: dynamic_object (car, truck, person, bicyclist, motorcyclist, other-vehicle, bus, on-rails)
    10: 2,   # car
    11: 2,   # bicycle
    13: 2,   # bus
    15: 2,   # motorcycle
    16: 2,   # on-rails
    18: 2,   # truck
    20: 2,   # other-vehicle
    30: 2,   # person
    31: 2,   # bicyclist
    32: 2,   # motorcyclist
    # Moving variants in SemanticKITTI:
    252: 2,  # moving car
    253: 2,  # moving bicyclist
    254: 2,  # moving person
    255: 2,  # moving motorcyclist
    256: 2,  # moving on-rails
    257: 2,  # moving bus
    258: 2,  # moving truck
    259: 2,  # moving other-vehicle

    # Ignore index (-1)
    0: -1,   # unlabeled
    1: -1,   # outlier
}

CLASS_NAMES = {
    0: "terrain",
    1: "static_obstacle",
    2: "dynamic_object"
}


def remap_labels_vectorized(raw_labels: np.ndarray, remap_table: Dict[int, int] = SEMANTICKITTI_REMAP_TABLE) -> np.ndarray:
    """
    Vectorized remapping of raw 32-bit SemanticKITTI labels to target classes {0, 1, 2, -1}.
    Extracts lower 16 bits (semantic label) and looks up target class.
    """
    sem_labels = raw_labels & 0xFFFF  # lower 16 bits = semantic class ID
    max_id = int(np.max(sem_labels)) if sem_labels.size > 0 else 0
    lut_size = max(max_id + 1, 300)
    lut = np.full(lut_size, -1, dtype=np.int64)
    for raw_id, target_id in remap_table.items():
        if raw_id < lut_size:
            lut[raw_id] = target_id

    # Fallback to ignore for anything not in table
    valid_mask = (sem_labels < lut_size)
    remapped = np.full(sem_labels.shape, -1, dtype=np.int64)
    remapped[valid_mask] = lut[sem_labels[valid_mask]]
    return remapped


def download_and_extract_semantickitti_labels(target_dir: Union[str, Path] = "/kaggle/working/labels") -> Path:
    """
    Downloads and extracts official SemanticKITTI labels (170 MB) for sequences 00-10.
    """
    import zipfile
    import shutil
    import urllib.request

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = target_dir / "dataset" / "sequences"
    if sequences_dir.exists() or (target_dir / "sequences").exists():
        return target_dir

    url = "http://semantic-kitti.org/assets/data_odometry_labels.zip"
    zip_path = target_dir / "data_odometry_labels.zip"
    print(f"[Dataset] Downloading official SemanticKITTI labels archive (170 MB) from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        print("[Dataset] Extracting labels...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        if zip_path.exists():
            zip_path.unlink()
        print("[Dataset] SemanticKITTI labels successfully installed!")
    except Exception as e:
        print(f"[Dataset Warning] Could not download labels from {url}: {e}")

    return target_dir


def generate_heuristic_labels(xyz: np.ndarray) -> np.ndarray:
    """
    Fast geometric ground/obstacle heuristic labeling for unlabeled point clouds:
    - Points with z < -1.3m (KITTI LiDAR mounting offset ~1.73m) -> 0: terrain
    - Points near roadway (|y| < 8m, -1.3 <= z < 0.8m) -> 2: dynamic_object
    - Remaining elevated points -> 1: static_obstacle
    """
    z = xyz[:, 2]
    dist_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    labels = np.full(len(xyz), 1, dtype=np.int64)  # default static_obstacle
    labels[z < -1.3] = 0  # terrain

    # Dynamic object heuristic
    dynamic_mask = (z >= -1.3) & (z < 0.8) & (np.abs(xyz[:, 1]) < 8.0) & (dist_xy < 40.0)
    labels[dynamic_mask] = 2
    return labels


class SemanticKITTIDataset(Dataset):
    """
    PyTorch Dataset for SemanticKITTI point clouds.
    Loads raw .bin scans and .label files, remaps to 3 classes, and samples N points.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        sequences: Optional[List[str]] = None,
        num_points: int = 4096,
        split: str = "train",
        subsample_ratio: float = 1.0,
        random_seed: int = 42,
        labels_dir: Optional[Union[str, Path]] = None,
        auto_download_labels: bool = True
    ):
        """
        Args:
            root_dir: Path to SemanticKITTI dataset root (e.g. /kaggle/input/... or data/raw/)
            sequences: List of sequence strings, e.g. ['00', '01']
            num_points: Number of points N to sample per scan (must match model input)
            split: 'train', 'val', or 'test'
            subsample_ratio: Ratio of scans to load (useful for smoke runs or fast epochs)
            random_seed: RNG seed for reproducible sampling
            labels_dir: Optional separate directory containing labels
            auto_download_labels: If True, automatically downloads official 170MB labels archive if missing
        """
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.split = split
        self.subsample_ratio = subsample_ratio
        self.rng = np.random.RandomState(random_seed)
        self.labels_dir = Path(labels_dir) if labels_dir else None

        if sequences is None:
            if split == "train":
                self.sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
            elif split == "val":
                self.sequences = ["08"]
            elif split == "test":
                self.sequences = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
            else:
                self.sequences = ["00"]
        else:
            self.sequences = [str(s).zfill(2) for s in sequences]

        # Check if labels are present or if we should auto-download official labels
        self.scan_files, self.label_files = self._find_files()
        if len(self.label_files) == 0 and auto_download_labels:
            default_labels_cache = Path("/kaggle/working/labels") if os.path.exists("/kaggle") else Path("data/labels")
            download_and_extract_semantickitti_labels(default_labels_cache)
            self.labels_dir = default_labels_cache
            self.scan_files, self.label_files = self._find_files()

        if len(self.scan_files) == 0:
            raise FileNotFoundError(
                f"No scan files found under {self.root_dir} for sequences {self.sequences}."
            )

        if subsample_ratio < 1.0:
            total = len(self.scan_files)
            subset_size = max(1, int(total * subsample_ratio))
            indices = self.rng.choice(total, subset_size, replace=False)
            indices.sort()
            self.scan_files = [self.scan_files[i] for i in indices]
            if len(self.label_files) == total:
                self.label_files = [self.label_files[i] for i in indices]

    def _find_files(self) -> Tuple[List[str], List[Optional[str]]]:
        scan_list = []
        label_list = []

        for seq in self.sequences:
            candidates = [
                self.root_dir / "dataset" / "sequences" / seq,
                self.root_dir / "sequences" / seq,
                self.root_dir / seq,
                self.root_dir / "dataset" / seq
            ]
            seq_dir = None
            for cand in candidates:
                if (cand / "velodyne").exists():
                    seq_dir = cand
                    break

            if seq_dir is None:
                matches = glob.glob(str(self.root_dir / "**" / seq / "velodyne"), recursive=True)
                if matches:
                    seq_dir = Path(matches[0]).parent

            if seq_dir is None:
                continue

            bin_files = sorted(glob.glob(str(seq_dir / "velodyne" / "*.bin")))
            if not bin_files:
                # Direct check if bin files are in seq_dir
                bin_files = sorted(glob.glob(str(seq_dir / "*.bin")))

            # Label directory search
            lbl_candidates = [
                seq_dir / "labels",
                seq_dir.parent / "labels" / seq,
            ]
            if self.labels_dir:
                lbl_candidates.extend([
                    self.labels_dir / "dataset" / "sequences" / seq / "labels",
                    self.labels_dir / "sequences" / seq / "labels",
                    self.labels_dir / seq / "labels",
                    self.labels_dir / "labels" / seq,
                ])

            found_lbl_dir = None
            for lc in lbl_candidates:
                if lc.exists():
                    found_lbl_dir = lc
                    break

            for bin_path in bin_files:
                fname = Path(bin_path).stem
                lbl_path = None
                if found_lbl_dir:
                    candidate_lbl = found_lbl_dir / f"{fname}.label"
                    if candidate_lbl.exists():
                        lbl_path = str(candidate_lbl)

                scan_list.append(bin_path)
                label_list.append(lbl_path)

        # Filter out cases where no labels are found at all
        has_any_labels = any(p is not None for p in label_list)
        if not has_any_labels:
            label_list = []

        return scan_list, label_list

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_path = self.scan_files[idx]
        lbl_path = self.label_files[idx] if idx < len(self.label_files) else None

        points_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points_raw[:, :3]

        if lbl_path and os.path.exists(lbl_path):
            labels_raw = np.fromfile(lbl_path, dtype=np.uint32)
            labels = remap_labels_vectorized(labels_raw)
        else:
            labels = generate_heuristic_labels(xyz)

        M = len(xyz)
        N = self.num_points
        choice = np.random.choice(M, N, replace=(M < N))

        return torch.from_numpy(xyz[choice]).float(), torch.from_numpy(labels[choice]).long()


def compute_dataset_class_distribution(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Computes class distribution, point counts, and recommended class weights.

    Args:
        dataset: SemanticKITTIDataset instance
        num_samples: Max number of samples to inspect (None for entire dataset)
        batch_size: Batch size for fast parallel iteration

    Returns:
        stats: Dictionary containing distribution counts, percentages, and weights
    """
    total_samples = len(dataset)
    if num_samples is not None:
        total_samples = min(total_samples, num_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    class_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    total_points = 0
    scans_inspected = 0

    for xyz, labels in loader:
        flat_labels = labels.view(-1).numpy()
        for c in [-1, 0, 1, 2]:
            class_counts[c] += int(np.sum(flat_labels == c))
        total_points += len(flat_labels)
        scans_inspected += len(xyz)
        if num_samples is not None and scans_inspected >= num_samples:
            break

    valid_points = class_counts[0] + class_counts[1] + class_counts[2]
    class_percentages = {}
    for c in [0, 1, 2]:
        pct = (class_counts[c] / valid_points * 100.0) if valid_points > 0 else 0.0
        class_percentages[CLASS_NAMES[c]] = pct

    # Compute smoothed inverse frequency weights for CrossEntropyLoss
    # formula: 1.0 / (log(1.02 + freq))
    weights = []
    for c in [0, 1, 2]:
        freq = class_counts[c] / valid_points if valid_points > 0 else 1.0 / 3.0
        w = 1.0 / (np.log(1.02 + freq))
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.mean(weights)  # normalize mean to 1.0

    underrepresented = [
        CLASS_NAMES[c] for c in [0, 1, 2] if class_percentages[CLASS_NAMES[c]] < 2.0
    ]

    return {
        "scans_inspected": scans_inspected,
        "total_points": total_points,
        "valid_points": valid_points,
        "ignored_points": class_counts[-1],
        "class_counts": {CLASS_NAMES[c]: class_counts[c] for c in [0, 1, 2]},
        "class_percentages": class_percentages,
        "recommended_weights": weights.tolist(),
        "underrepresented_classes": underrepresented
    }
