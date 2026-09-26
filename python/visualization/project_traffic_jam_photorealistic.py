"""
PS26053 - Photorealistic 3D LiDAR Projection of traffic_jam.jpg
Directly converts pixels and semantic elements from traffic_jam.jpg into
a realistic 3D LiDAR point cloud in vehicle sensor coordinates:
  - Roadway & Crosswalk (Z = -1.73m)
  - Yellow Taxi (X ~ 9m, Y ~ 0m)
  - Blue Transit Bus (X ~ 16m, Y ~ -4.2m)
  - Pedestrians Crossing (X ~ 11m, Y ~ +2.5m to +6.0m)
  - Trailing Sedans/SUVs (X ~ 14m - 25m)
  - Urban Canyon Skyscrapers (|Y| > 7.5m, Z up to 15m)

Runs inference using trained PointNet++ ONNX model and exports:
  1. traffic_jam_3d_prediction.ply (Colored by predicted semantic classes: 0: Terrain, 1: Static, 2: Dynamic)
  2. traffic_jam_photo_colors.ply (Colored by actual RGB photo pixels)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.dataset import CLASS_NAMES
try:
    from python.inference_demo import export_ply
except ImportError:
    from scripts.inference_demo import export_ply

IMG_PATH = "images/hard_cases/traffic_jam.jpg"
ONNX_MODEL = "models/onnx/pointnet2_semseg.onnx"
OUTPUT_DIR = "images/hard_cases"


def generate_traffic_jam_photorealistic(num_points: int = 4096):
    img = Image.open(IMG_PATH).convert("RGB")
    W, H = img.size
    img_np = np.array(img, dtype=np.float32)

    # Intrinsic camera parameters
    fx = 1200.0
    fy = 1200.0
    cx = W / 2.0
    cy = H / 2.0

    points_3d = []
    rgb_colors = []
    expected_classes = []

    # 1. Road & Crosswalk surface (Z ~ -1.73m ground)
    # Spans bottom half of image: v in [450, 720]
    n_road = int(num_points * 0.45)
    for _ in range(n_road):
        u = np.random.uniform(50, W - 50)
        v = np.random.uniform(460, 720)
        # Depth from ground plane geometry
        depth_x = (fy * 1.73) / max(v - cy, 10.0)
        depth_x = np.clip(depth_x, 4.0, 45.0)
        y_lat = - (u - cx) * depth_x / fx
        z_height = -1.73 + np.random.normal(0, 0.02)

        c = img_np[int(v), int(u)].astype(np.uint8)
        points_3d.append([depth_x, y_lat, z_height])
        rgb_colors.append(c)
        expected_classes.append(0)

    # 2. Yellow Taxi (Centered in front, u in [520, 820], v in [450, 680])
    n_taxi = int(num_points * 0.18)
    for _ in range(n_taxi):
        u = np.random.uniform(530, 810)
        v = np.random.uniform(460, 670)
        # Taxi sits at depth 8.0m to 11.5m
        depth_x = 8.0 + (670 - v) / 210.0 * 3.5
        y_lat = - (u - cx) * 9.5 / fx
        z_height = -1.73 + (670 - v) / 210.0 * 1.4

        c = img_np[int(v), int(u)].astype(np.uint8)
        points_3d.append([depth_x, y_lat, z_height])
        rgb_colors.append(c)
        expected_classes.append(2)

    # 3. Blue City Bus (Left lane, u in [160, 440], v in [300, 520])
    n_bus = int(num_points * 0.15)
    for _ in range(n_bus):
        u = np.random.uniform(160, 440)
        v = np.random.uniform(310, 510)
        # Bus sits at depth 13.0m to 22.0m
        depth_x = 13.0 + (510 - v) / 200.0 * 9.0
        y_lat = - (u - cx) * 16.0 / fx
        z_height = -1.73 + (510 - v) / 200.0 * 2.8

        c = img_np[int(v), int(u)].astype(np.uint8)
        points_3d.append([depth_x, y_lat, z_height])
        rgb_colors.append(c)
        expected_classes.append(2)

    # 4. Crossing Pedestrians (Right foreground, u in [840, 1300], v in [430, 620])
    n_peds = int(num_points * 0.10)
    for _ in range(n_peds):
        u = np.random.uniform(840, 1280)
        v = np.random.uniform(440, 610)
        # Pedestrians walk at depth 10.0m to 14.0m
        depth_x = 10.0 + (610 - v) / 170.0 * 4.0
        y_lat = - (u - cx) * 11.5 / fx
        z_height = -1.73 + (610 - v) / 170.0 * 1.75

        c = img_np[int(v), int(u)].astype(np.uint8)
        points_3d.append([depth_x, y_lat, z_height])
        rgb_colors.append(c)
        expected_classes.append(2)

    # 5. Flanking Skyscrapers / Commercial Buildings (Left & Right upper halves)
    n_bldg = num_points - len(points_3d)
    for _ in range(n_bldg):
        side_right = np.random.choice([False, True])
        if side_right:
            u = np.random.uniform(1150, W - 10)
        else:
            u = np.random.uniform(10, 250)
        v = np.random.uniform(20, 420)
        # Buildings are 20m - 50m away, standing up to 15m high
        depth_x = np.random.uniform(18.0, 50.0)
        y_lat = - (u - cx) * depth_x / fx
        z_height = -1.5 + (420 - v) / 400.0 * 14.0

        c = img_np[int(v), int(u)].astype(np.uint8)
        points_3d.append([depth_x, y_lat, z_height])
        rgb_colors.append(c)
        expected_classes.append(1)

    pts = np.array(points_3d, dtype=np.float32)
    colors = np.array(rgb_colors, dtype=np.uint8)
    gt = np.array(expected_classes, dtype=np.int64)
    return pts, colors, gt


def export_colored_ply(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Exports 3D points with actual RGB image colors."""
    N = len(points)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def run():
    print("=" * 70)
    print("PROJECTING TRAFFIC_JAM.JPG TO REALISTIC 3D LIDAR POINT CLOUD")
    print("=" * 70)

    pts, photo_colors, gt = generate_traffic_jam_photorealistic(num_points=4096)
    print(f"Generated {len(pts)} 3D points from traffic_jam.jpg pixels:")
    print(f"  X (Forward): [{pts[:,0].min():.2f}m, {pts[:,0].max():.2f}m]")
    print(f"  Y (Lateral): [{pts[:,1].min():.2f}m, {pts[:,1].max():.2f}m]")
    print(f"  Z (Height):  [{pts[:,2].min():.2f}m, {pts[:,2].max():.2f}m]")

    # Run trained ONNX Model
    session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    logits = session.run([out_name], {in_name: pts[np.newaxis, :, :]})[0]
    preds = np.argmax(logits, axis=-1)[0]

    print("\n--- Model Semantic Prediction Breakdown ---")
    for c in [0, 1, 2]:
        cnt = np.sum(preds == c)
        pct = (cnt / 4096) * 100.0
        print(f"  Class {c} ({CLASS_NAMES[c]:<16}): {cnt:5d} points ({pct:5.1f}%)")

    # Save Semantic PLY
    sem_ply_path = os.path.join(OUTPUT_DIR, "traffic_jam_3d_prediction.ply")
    export_ply(sem_ply_path, pts, preds)
    print(f"\n[OK] Updated semantic 3D prediction: {sem_ply_path}")

    # Save Photo Colors PLY
    photo_ply_path = os.path.join(OUTPUT_DIR, "traffic_jam_photo_colors.ply")
    export_colored_ply(photo_ply_path, pts, photo_colors)
    print(f"[OK] Exported real photo colored 3D point cloud: {photo_ply_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
