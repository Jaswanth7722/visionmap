"""
PS26053 Open3D Live 2.5D / 3D World Model Viewer
Visualizes exported 2.5D PLY maps and point clouds with Open3D desktop renderer.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def view_map_ply(ply_path: str = "results/maps/adaptive_map_frame000000.ply"):
    """Loads and renders 2.5D world model map using Open3D."""
    if not HAS_OPEN3D:
        print("[Error] Open3D is not installed.")
        return

    full_path = os.path.join(str(REPO_ROOT), ply_path)
    if not os.path.exists(full_path):
        print(f"[Error] Map file not found: {full_path}")
        return

    print(f"[Open3D] Loading 2.5D Adaptive World Model: {full_path} ...")
    pcd = o3d.io.read_point_cloud(full_path)
    print(f"[Open3D] Loaded {len(pcd.points)} active map cells.")

    # Create coordinate frame at sensor origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # Visualizer window setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PS26053 — 2.5D Adaptive LiDAR World Model (Open3D)", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)

    opt = vis.get_render_option()
    opt.background_color = [0.08, 0.10, 0.14]
    opt.point_size = 4.0
    opt.show_coordinate_frame = True

    print("[Open3D] Viewer launched. Close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    target_ply = sys.argv[1] if len(sys.argv) > 1 else "results/maps/adaptive_map_frame000000.ply"
    view_map_ply(target_ply)
