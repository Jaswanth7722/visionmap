"""
PS26053 LiDAR Semantic Perception - Single Scan Inference & Verification Demo
Loads a raw Velodyne .bin file (e.g. 000000.bin), executes PointNet++ semantic inference
via ONNX Runtime, applies geometric heuristics when needed, and exports 3D visualizations.

Usage:
    python python/inference_demo.py --bin_path 000000.bin
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.dataset import (
    CLASS_NAMES,
    generate_heuristic_labels,
    remap_labels_vectorized,
)

# Color palette for 3D PLY visualization (RGB in 0..255)
CLASS_COLORS = {
    0: (76, 175, 80),    # Terrain / Road: Lush Green
    1: (244, 67, 54),    # Static Obstacle / Building / Vegetation: Coral Red
    2: (33, 150, 243),   # Dynamic Object / Car / Truck / Bicyclist: Electric Blue
    -1: (158, 158, 158), # Ignored / Noise: Slate Grey
}


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Loads a KITTI / SemanticKITTI binary point cloud (.bin)."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"LiDAR scan file not found: {bin_path}")
    raw = np.fromfile(bin_path, dtype=np.float32)
    if len(raw) % 4 != 0:
        raise ValueError(f"Corrupt scan: float count {len(raw)} is not a multiple of 4.")
    points = raw.reshape(-1, 4)
    return points


def export_ply(filename: str, points: np.ndarray, labels: np.ndarray) -> None:
    """
    Exports 3D points and semantic class colors to standard ASCII PLY format.
    Can be opened directly in MeshLab, CloudCompare, Blender, or web viewers.
    """
    N = len(points)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int class_id\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i, :3]
            c_id = int(labels[i])
            r, g, b = CLASS_COLORS.get(c_id, (158, 158, 158))
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {c_id}\n")


def run_inference(
    bin_path: str,
    onnx_path: str,
    num_points: int = 4096,
    label_path: Optional[str] = None,
    output_ply: Optional[str] = None,
    output_label: Optional[str] = None,
    full_scan: bool = False,
) -> None:
    print("=" * 70)
    print("PS26053 SEMANTIC PERCEPTION - INFERENCE & VALIDATION")
    print("=" * 70)

    # 1. Load scan
    print(f"[*] Reading scan: {bin_path}")
    points_raw = load_kitti_bin(bin_path)
    xyz = points_raw[:, :3]
    intensity = points_raw[:, 3]
    total_pts = len(points_raw)

    print(f"    Total points in scan:  {total_pts:,}")
    print(f"    X coordinate bounds:   [{xyz[:, 0].min():.2f}m, {xyz[:, 0].max():.2f}m]")
    print(f"    Y coordinate bounds:   [{xyz[:, 1].min():.2f}m, {xyz[:, 1].max():.2f}m]")
    print(f"    Z coordinate bounds:   [{xyz[:, 2].min():.2f}m, {xyz[:, 2].max():.2f}m]")
    print(f"    Intensity bounds:      [{intensity.min():.2f}, {intensity.max():.2f}]")

    # 2. Check Ground Truth or apply heuristic
    has_gt = False
    if label_path and os.path.exists(label_path):
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        ground_truth = remap_labels_vectorized(raw_labels)
        has_gt = True
        print(f"[*] Loaded Ground Truth labels from: {label_path}")
    else:
        ground_truth = generate_heuristic_labels(xyz)
        print("[*] No .label file specified. Computed geometric heuristic labels.")

    print("\n--- Point Cloud Class Distribution (Full Scan) ---")
    for cid in [0, 1, 2, -1]:
        cnt = np.sum(ground_truth == cid)
        pct = (cnt / total_pts) * 100.0
        name = CLASS_NAMES.get(cid, "ignored")
        print(f"    {cid:2d} | {name:<16}: {cnt:7,d} points ({pct:5.1f}%)")

    # 3. ONNX Runtime Model Session
    print(f"\n[*] Initializing ONNX Runtime Session: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"    Input Name:  '{input_meta.name}', Shape: {input_meta.shape}, Dtype: {input_meta.type}")
    print(f"    Output Name: '{output_meta.name}', Shape: {output_meta.shape}, Dtype: {output_meta.type}")

    # 4. Sampling N points for Model Inference
    # PointNet++ consumes shape (Batch, NumPoints, 3)
    np.random.seed(42)
    sample_idx = np.random.choice(total_pts, num_points, replace=False)
    sample_xyz = xyz[sample_idx]
    model_input = sample_xyz[np.newaxis, :, :].astype(np.float32)  # shape (1, N, 3)

    # 5. Measure Latency
    # Warmup
    _ = session.run([output_meta.name], {input_meta.name: model_input})

    runs = 5
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = session.run([output_meta.name], {input_meta.name: model_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    print(f"\n[*] ONNX Runtime CPU Inference Benchmark (N={num_points}):")
    print(f"    Latency: {avg_ms:.2f} ms +/- {std_ms:.2f} ms over {runs} iterations")
    print(f"    Throughput: {1000.0 / avg_ms:.1f} scans/sec (CPU single-session)")

    # 6. Predictions
    logits = outputs[0]  # shape (1, N, 3)
    predicted_classes = np.argmax(logits, axis=-1)[0]  # shape (N,)

    print(f"\n--- Model Prediction Summary (N={num_points} sampled points) ---")
    for cid in [0, 1, 2]:
        cnt = np.sum(predicted_classes == cid)
        pct = (cnt / num_points) * 100.0
        print(f"    {cid:2d} | {CLASS_NAMES[cid]:<16}: {cnt:5,d} points ({pct:5.1f}%)")

    # 7. Optional Full Scan Segmentation
    full_predictions = None
    if full_scan:
        print(f"\n[*] Running full-scan chunked segmentation on all {total_pts:,} points...")
        full_predictions = np.zeros(total_pts, dtype=np.int64)
        chunk_size = num_points
        num_chunks = int(np.ceil(total_pts / chunk_size))
        t_start = time.perf_counter()

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, total_pts)
            chunk_pts = xyz[start:end]
            act_n = len(chunk_pts)

            # Pad if needed to match chunk_size or dynamic axis
            if act_n < chunk_size:
                pad_pts = np.pad(chunk_pts, ((0, chunk_size - act_n), (0, 0)), mode="edge")
            else:
                pad_pts = chunk_pts

            chunk_in = pad_pts[np.newaxis, :, :].astype(np.float32)
            chunk_out = session.run([output_meta.name], {input_meta.name: chunk_in})[0]
            chunk_preds = np.argmax(chunk_out[0], axis=-1)[:act_n]
            full_predictions[start:end] = chunk_preds

        t_elapsed = time.perf_counter() - t_start
        print(f"    Full-scan segmentation completed in {t_elapsed:.2f} s ({total_pts / t_elapsed:,.0f} pts/s)")
        print("\n--- Full-Scan Prediction Breakdown ---")
        for cid in [0, 1, 2]:
            cnt = np.sum(full_predictions == cid)
            pct = (cnt / total_pts) * 100.0
            print(f"    {cid:2d} | {CLASS_NAMES[cid]:<16}: {cnt:7,d} points ({pct:5.1f}%)")

    # 8. Export PLY
    if output_ply:
        print(f"\n[*] Saving 3D colored PLY visualization to: {output_ply}")
        pts_to_save = xyz if full_scan else sample_xyz
        labels_to_save = full_predictions if full_scan else predicted_classes
        export_ply(output_ply, pts_to_save, labels_to_save)
        print(f"    [OK] Saved {len(pts_to_save):,d} colored points. Open in 3D viewer.")

    # 9. Export binary .label file (low 16-bits class, high 16-bits instance ID 0)
    if output_label:
        print(f"\n[*] Saving SemanticKITTI format .label file to: {output_label}")
        lbl_data = (full_predictions if full_scan else ground_truth).astype(np.uint32)
        lbl_data.tofile(output_label)
        print(f"    [OK] Saved {len(lbl_data):,d} labels to {output_label}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Scan Verification & Inference Completed.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PS26053 LiDAR Inference Demo")
    parser.add_argument("--bin_path", type=str, default="000000.bin", help="Path to .bin scan")
    parser.add_argument("--onnx_path", type=str, default="models/onnx/pointnet2_semseg.onnx", help="Path to ONNX model")
    parser.add_argument("--num_points", type=int, default=4096, help="Point sample count N")
    parser.add_argument("--label_path", type=str, default=None, help="Optional ground truth .label file")
    parser.add_argument("--output_ply", type=str, default="sample_prediction.ply", help="Output PLY file")
    parser.add_argument("--output_label", type=str, default=None, help="Optional output .label path")
    parser.add_argument("--full_scan", action="store_true", help="Segment all points in scan via chunking")
    args = parser.parse_args()

    run_inference(
        bin_path=args.bin_path,
        onnx_path=args.onnx_path,
        num_points=args.num_points,
        label_path=args.label_path,
        output_ply=args.output_ply,
        output_label=args.output_label,
        full_scan=args.full_scan,
    )


if __name__ == "__main__":
    main()
