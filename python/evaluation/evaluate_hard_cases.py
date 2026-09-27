"""
PS26053 - Hard Cases 3D Perception Evaluation Suite
Evaluates trained PointNet++ ONNX model on ultra-challenging driving scenarios:
  1. Dense Downtown Traffic Jam (multi-agent occlusion: cars, bus, pedestrians, skyscrapers)
  2. Active Construction Zone (excavator, workers, concrete barriers, cones, uneven road)
  3. Narrow Forest Winding Road (guardrail, dense trees, overhanging canopy, zero traffic)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.dataset import CLASS_NAMES
from python.inference_demo import export_ply

ONNX_MODEL = "models/onnx/pointnet2_semseg.onnx"
OUTPUT_DIR = "results/hard_cases"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_traffic_jam_pointcloud(num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downtown traffic jam:
    - Road surface: X in [5, 45], Y in [-7, 7], Z ~ -1.73m (Class 0: terrain)
    - Front taxi: X in [7, 11], Y in [-1.0, 1.0], Z in [-1.5, -0.3] (Class 2: dynamic)
    - Left bus: X in [14, 24], Y in [-5.5, -2.5], Z in [-1.5, 1.2] (Class 2: dynamic)
    - Crossing pedestrians: X in [11, 14], Y in [2.0, 5.0], Z in [-1.7, -0.1] (Class 2: dynamic)
    - Buildings on sides: |Y| > 7.5, Z in [-1.5, 15.0] (Class 1: static)
    """
    pts_list, gt_list = [], []

    # 1. Road terrain (40%)
    n_road = int(num_points * 0.40)
    x = np.random.uniform(5, 45, n_road)
    y = np.random.uniform(-7.5, 7.5, n_road)
    z = np.random.normal(-1.73, 0.04, n_road)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_road, 0))

    # 2. Taxi (15%)
    n_taxi = int(num_points * 0.15)
    x = np.random.uniform(7.5, 11.5, n_taxi)
    y = np.random.uniform(-1.0, 1.0, n_taxi)
    z = np.random.uniform(-1.5, -0.2, n_taxi)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_taxi, 2))

    # 3. City Bus (15%)
    n_bus = int(num_points * 0.15)
    x = np.random.uniform(14.0, 24.0, n_bus)
    y = np.random.uniform(-5.5, -2.5, n_bus)
    z = np.random.uniform(-1.5, 1.2, n_bus)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_bus, 2))

    # 4. Pedestrians on crosswalk (8%)
    n_ped = int(num_points * 0.08)
    x = np.random.uniform(11.5, 14.0, n_ped)
    y = np.random.uniform(2.0, 5.5, n_ped)
    z = np.random.uniform(-1.7, -0.1, n_ped)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_ped, 2))

    # 5. Flanking buildings (22%)
    n_bldg = num_points - sum(len(p) for p in pts_list)
    side = np.random.choice([-1.0, 1.0], n_bldg)
    x = np.random.uniform(5, 45, n_bldg)
    y = side * np.random.uniform(8.0, 14.0, n_bldg)
    z = np.random.uniform(-1.5, 12.0, n_bldg)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_bldg, 1))

    pts = np.vstack(pts_list).astype(np.float32)
    gt = np.concatenate(gt_list).astype(np.int64)
    return pts, gt


def build_construction_zone_pointcloud(num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construction zone:
    - Road & work trench: X in [5, 45], Y in [-5, 5], Z ~ -1.73m (Class 0: terrain)
    - Front SUV: X in [13, 17], Y in [-1.5, 0.5], Z in [-1.5, -0.1] (Class 2: dynamic)
    - Concrete barriers & safety cones along lane edge: Y in [-3.5, -2.8] and [2.0, 2.5] (Class 1: static)
    - Heavy Excavator: X in [22, 28], Y in [3.0, 6.0], Z in [-1.5, 3.5] (Class 1: static)
    - Workers: X in [18, 22], Y in [2.5, 4.0], Z in [-1.7, 0.0] (Class 2: dynamic)
    """
    pts_list, gt_list = [], []

    # 1. Roadway (45%)
    n_road = int(num_points * 0.45)
    x = np.random.uniform(5, 45, n_road)
    y = np.random.uniform(-4.0, 4.0, n_road)
    z = np.random.normal(-1.73, 0.05, n_road)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_road, 0))

    # 2. SUV (12%)
    n_suv = int(num_points * 0.12)
    x = np.random.uniform(13.0, 17.5, n_suv)
    y = np.random.uniform(-1.5, 0.5, n_suv)
    z = np.random.uniform(-1.5, -0.1, n_suv)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_suv, 2))

    # 3. Concrete barriers & cones (18%)
    n_barr = int(num_points * 0.18)
    side = np.random.choice([-1.0, 1.0], n_barr)
    x = np.random.uniform(5, 40, n_barr)
    y = np.where(side > 0, np.random.uniform(2.0, 2.5, n_barr), np.random.uniform(-3.5, -3.0, n_barr))
    z = np.random.uniform(-1.73, -0.8, n_barr)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_barr, 1))

    # 4. Excavator (18%)
    n_exc = int(num_points * 0.18)
    x = np.random.uniform(22.0, 28.0, n_exc)
    y = np.random.uniform(3.0, 6.0, n_exc)
    z = np.random.uniform(-1.5, 3.0, n_exc)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_exc, 1))

    # 5. Workers (7%)
    n_work = num_points - sum(len(p) for p in pts_list)
    x = np.random.uniform(18.0, 22.0, n_work)
    y = np.random.uniform(2.5, 4.0, n_work)
    z = np.random.uniform(-1.7, 0.0, n_work)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_work, 2))

    pts = np.vstack(pts_list).astype(np.float32)
    gt = np.concatenate(gt_list).astype(np.int64)
    return pts, gt


def build_forest_road_pointcloud(num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Curved forest road:
    - Winding road: X in [5, 45], Y ~ sin(X/8)*3.0, width 4m, Z ~ -1.73m (Class 0: terrain)
    - Guardrail along road edge (Class 1: static)
    - Grassy embankment & pine trees on both sides (Class 1: static + Class 0: terrain)
    - Zero dynamic objects!
    """
    pts_list, gt_list = [], []

    # 1. Winding Road (45%)
    n_road = int(num_points * 0.45)
    x = np.random.uniform(5, 45, n_road)
    road_center_y = np.sin(x / 7.0) * 3.5
    y = road_center_y + np.random.uniform(-2.2, 2.2, n_road)
    z = np.random.normal(-1.73, 0.03, n_road)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_road, 0))

    # 2. Guardrail along right curve (12%)
    n_guard = int(num_points * 0.12)
    x = np.random.uniform(5, 40, n_guard)
    road_center_y = np.sin(x / 7.0) * 3.5
    y = road_center_y + 2.5 + np.random.normal(0, 0.05, n_guard)
    z = np.random.uniform(-1.73, -1.0, n_guard)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_guard, 1))

    # 3. Dense Pine Trees & Overhanging Canopy (43%)
    n_trees = num_points - sum(len(p) for p in pts_list)
    x = np.random.uniform(5, 45, n_trees)
    road_center_y = np.sin(x / 7.0) * 3.5
    side = np.random.choice([-1.0, 1.0], n_trees)
    y = road_center_y + side * np.random.uniform(3.5, 12.0, n_trees)
    z = np.random.uniform(-1.5, 8.0, n_trees)
    pts_list.append(np.stack([x, y, z], -1))
    gt_list.append(np.full(n_trees, 1))

    pts = np.vstack(pts_list).astype(np.float32)
    gt = np.concatenate(gt_list).astype(np.int64)
    return pts, gt


def run_evaluation():
    print("=" * 80)
    print("PS26053 - HARD CASES BENCHMARK EVALUATION (ONNX OPSET 17)")
    print("=" * 80)

    session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    scenarios = [
        ("traffic_jam", "Dense Urban Traffic Jam (Taxi, Bus, Pedestrians, Buildings)", build_traffic_jam_pointcloud),
        ("construction_zone", "Active Road Construction Zone (SUV, Excavator, Barriers, Workers)", build_construction_zone_pointcloud),
        ("forest_road", "Curved Rural Forest Road (Winding Road, Guardrail, Pine Trees)", build_forest_road_pointcloud),
    ]

    report_data = []

    for key, title, generator in scenarios:
        print(f"\n[*] Evaluating Scenario: {title}")
        pts, gt = generator(num_points=4096)

        # Inference
        t0 = time.perf_counter()
        logits = session.run([out_name], {in_name: pts[np.newaxis, :, :]})[0]
        latency = (time.perf_counter() - t0) * 1000.0
        preds = np.argmax(logits, axis=-1)[0]

        # Metrics
        acc = (np.sum(preds == gt) / len(gt)) * 100.0
        print(f"    Inference Latency: {latency:.2f} ms")
        print(f"    Scenario Accuracy: {acc:.2f}%")

        # Class Breakdown
        class_stats = {}
        for c in [0, 1, 2]:
            pred_cnt = int(np.sum(preds == c))
            gt_cnt = int(np.sum(gt == c))
            tp = int(np.sum((preds == c) & (gt == c)))
            fp = pred_cnt - tp
            fn = gt_cnt - tp
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou * 100.0) if denom_iou > 0 else 0.0
            class_stats[CLASS_NAMES[c]] = {
                "gt_count": gt_cnt,
                "pred_count": pred_cnt,
                "iou": round(iou, 2),
                "precision": round(tp / (tp + fp) * 100.0, 2) if (tp + fp) > 0 else 0.0,
                "recall": round(tp / (tp + fn) * 100.0, 2) if (tp + fn) > 0 else 0.0,
            }
            print(f"    - {CLASS_NAMES[c]:<16}: IoU {iou:5.2f}% | Prec {class_stats[CLASS_NAMES[c]]['precision']:5.2f}% | Rec {class_stats[CLASS_NAMES[c]]['recall']:5.2f}%")

        # Export PLY
        ply_path = os.path.join(OUTPUT_DIR, f"{key}_3d_prediction.ply")
        export_ply(ply_path, pts, preds)
        print(f"    Exported 3D colored PLY: {ply_path}")

        report_data.append({
            "key": key,
            "title": title,
            "accuracy": round(acc, 2),
            "latency_ms": round(latency, 2),
            "class_stats": class_stats,
            "ply_path": ply_path
        })

    with open(os.path.join(OUTPUT_DIR, "hard_cases_results.json"), "w") as f:
        json.dump(report_data, f, indent=2)
    print("\nSaved evaluation results to: results/hard_cases/hard_cases_results.json")


if __name__ == "__main__":
    from typing import Tuple
    run_evaluation()
