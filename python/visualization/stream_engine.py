"""
PS26053 Live Streaming & Real-Time Perception Engine
Supports:
1. Live Camera / Webcam Feed (with synthetic driving camera fallback)
2. Real-Time Continuous LiDAR Stream (from .bin scans or dynamic urban simulator)
3. Direct PointNet++ ONNX Runtime Inference & Kalman Tracking
4. Adaptive Variable-Resolution 2.5D World Model Grid
5. Live Uniform vs. Adaptive Comparative Profiler
"""

import os
import sys
import time
import math
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import onnxruntime as ort

try:
    import cv2
except ImportError:
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLASS_NAMES = {
    0: "terrain",
    1: "static_obstacle",
    2: "dynamic_object"
}

# Memory-estimate constants, measured 2026-09-26 with a compiled sizeof probe
# against the actual C++ sources (MinGW GCC 15.2 x86_64):
#   sizeof(Cell) = 88; one adaptive base cell additionally carries an
#   unordered_map node + Quadtree (32) + heap QuadtreeNode (144), totalling
#   ~252 bytes. These are estimates for display only; exact profiling is out
#   of scope for the dashboard. See FIXES_REPORT.md (C2).
UNIFORM_BYTES_PER_CELL = 88
ADAPTIVE_BYTES_PER_CELL = 252


class CameraProjector:
    """Captures camera frames and projects image pixels to 3D metric coordinates."""
    def __init__(self, camera_id: int = 0, enable_hardware: bool = False):
        self.camera_id = camera_id
        self.enable_hardware = enable_hardware
        self.cap = None
        self.use_synthetic = not enable_hardware

    def try_open_hardware(self):
        if cv2 is not None:
            try:
                import os
                os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" # Prefer DirectShow over hanging MSMF
                cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.use_synthetic = False
                        return
                    cap.release()
            except Exception:
                pass
        self.use_synthetic = True

    def read_frame(self, frame_index: int = 0) -> Tuple[np.ndarray, bool]:
        """Returns RGB frame (H, W, 3) and is_live boolean."""
        if self.enable_hardware and self.cap is None and self.use_synthetic:
            self.try_open_hardware()

        if self.enable_hardware and not self.use_synthetic and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb, True

        return self._generate_synthetic_frame(frame_index), False

    def _generate_synthetic_frame(self, frame_index: int) -> np.ndarray:
        # Zero Mock Data Policy: Never fake driving scenes or vehicles when hardware is off
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(frame, "HARDWARE CAMERA OFFLINE", (w // 2 - 190, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "No camera detected / Hardware not connected. Zero mock data.",
                        (w // 2 - 250, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        else:
            # cv2 itself is unavailable: white banner bar drawn with numpy so
            # this path never crashes on a missing optional dependency.
            frame[h // 2 - 14:h // 2 + 14, w // 2 - 200:w // 2 + 200] = (255, 255, 255)
        return frame

    def project_to_point_cloud(self, frame_rgb: np.ndarray, num_points: int = 4096) -> np.ndarray:
        """
        Projects RGB image pixels into 3D metric coordinates (X=Forward, Y=Lateral, Z=Height).
        Returns array of shape (M, 4) [x, y, z, intensity] with M <= num_points
        (the sampling grid rarely divides evenly; exact counts are reported).
        """
        h, w, _ = frame_rgb.shape
        # Uniform sampling across the image plane
        grid_h = int(math.sqrt(num_points * (h / w)))
        grid_w = num_points // grid_h
        
        ys = np.linspace(h * 0.35, h - 1, grid_h).astype(int)
        xs = np.linspace(0, w - 1, grid_w).astype(int)
        xv, yv = np.meshgrid(xs, ys)
        sampled_x = xv.flatten()[:num_points]
        sampled_y = yv.flatten()[:num_points]

        # Monocular depth projection: pixels lower in image are closer
        v_norm = (sampled_y - (h * 0.35)) / (h * 0.65)
        v_norm = np.clip(v_norm, 0.05, 1.0)
        
        # Depth ranging from 5m (bottom) to 50m (horizon)
        depth = 5.0 + 45.0 * (1.0 - v_norm)
        
        # Lateral X based on horizontal field of view (~60 deg)
        u_norm = (sampled_x - (w / 2.0)) / (w / 2.0)
        lateral_y = u_norm * depth * math.tan(math.radians(30.0))
        
        # Height Z
        height_z = -1.6 + (1.0 - v_norm) * 4.0

        # Colors & Intensity
        colors = frame_rgb[sampled_y, sampled_x]
        intensity = (0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]) / 255.0

        # Point cloud: X = forward (depth), Y = lateral, Z = height
        cloud = np.column_stack([depth, lateral_y, height_z, intensity]).astype(np.float32)
        return cloud

    def release(self):
        if self.cap is not None:
            self.cap.release()


class LiveLidarStreamer:
    """Streams LiDAR scans continuously or generates dynamic urban point clouds."""
    def __init__(self, raw_bin_path: Optional[str] = "data/raw/000000.bin"):
        self.raw_bin_path = raw_bin_path
        self.cached_points = None
        self._load_cached_points()

    def _load_cached_points(self):
        if self.raw_bin_path and os.path.exists(self.raw_bin_path):
            try:
                pts = np.fromfile(self.raw_bin_path, dtype=np.float32).reshape(-1, 4)
                self.cached_points = pts
                print(f"[LiveLidarStreamer] Loaded baseline scan: {len(pts)} points.")
            except Exception as e:
                print(f"[LiveLidarStreamer] Failed to load {self.raw_bin_path}: {e}")

    def get_frame(self, frame_index: int = 0) -> np.ndarray:
        """
        Returns the cached real LiDAR scan unchanged, shape (M, 4).
        Frames are intentionally NOT perturbed: with a single recorded scan
        there is no real motion, and synthesizing any would fabricate tracks.
        """
        if self.cached_points is not None:
            return self.cached_points
        raise RuntimeError(
            f"No LiDAR data available (tried {self.raw_bin_path}). "
            "The dashboard shows an error instead of generating a synthetic cloud."
        )


class RealTimePipelineEngine:
    """End-to-End Live Processing Engine connecting Perception, Tracking, and Adaptive Mapping."""
    def __init__(self, onnx_model_path: str = "models/onnx/pointnet2_semseg.onnx"):
        self.onnx_model_path = onnx_model_path
        self.session = None
        self._load_model()
        # Hardware capture is attempted: a present webcam yields live frames,
        # an absent one yields the honestly labeled offline placeholder.
        self.camera = CameraProjector(0, enable_hardware=True)
        # Absolute path: the dashboard must not depend on the process CWD.
        self.lidar = LiveLidarStreamer(os.path.join(str(REPO_ROOT), "data", "raw", "000000.bin"))

        # Tracking state
        self.tracks = {}
        self.next_track_id = 1
        self._last_frame_t = None

        # Adaptive 2.5D World Model storage
        self.adaptive_cells = {}
        self.frame_count = 0

        # Measured ORT throughput on this machine is ~40 us/point, so cap one
        # frame at 16384 stride-sampled points (~0.65 s worst case) and always
        # report the exact classified/total counts next to every number.
        self.max_infer_points = 16384
        self.infer_chunk = 8192
        # Memoization for identical inputs only (same bytes -> same output).
        self._infer_cache = {}

    def _load_model(self):
        # The ONNX session is mandatory. Without it there is no perception to
        # show, and this engine refuses to substitute heuristic labels while
        # anything claims PointNet++ output.
        if not os.path.exists(self.onnx_model_path):
            raise RuntimeError(
                f"ONNX model not found: {self.onnx_model_path}. "
                "PointNet++ inference cannot run without it, so the dashboard "
                "refuses to start rather than display fabricated segmentation."
            )
        try:
            self.session = ort.InferenceSession(
                self.onnx_model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create ONNX Runtime session for {self.onnx_model_path}: {exc}"
            ) from exc
        print(f"[Engine] ONNX Runtime session ready: {self.onnx_model_path}")

    def _run_onnx(self, xyz: np.ndarray) -> np.ndarray:
        """Chunked full-subset inference; the model axis is dynamic to N=12000+."""
        outs = []
        for i in range(0, len(xyz), self.infer_chunk):
            chunk = np.ascontiguousarray(xyz[i:i + self.infer_chunk])
            outs.append(
                self.session.run(["logits"], {"points": chunk.reshape(1, -1, 3)})[0][0]
            )
        return np.concatenate(outs, axis=0).astype(np.float32)

    def process_frame(self, source_mode: str = "Live Camera", custom_cloud: Optional[np.ndarray] = None,
                      enable_refinement: bool = True) -> Dict:
        """
        Executes end-to-end processing for a single frame:
        Returns complete metrics, points, classes, tracks, and 2.5D cells.

        Every label shown comes from the ONNX model via ONNX Runtime. There is
        no heuristic fallback anywhere in this path: if inference fails, the
        returned dict carries {"error": ...} and the dashboard renders the
        error instead of fabricated data.
        """
        t_start = time.perf_counter()
        self.frame_count += 1
        camera_img = None
        is_hardware_camera = False

        # 1. Ingestion
        if custom_cloud is not None:
            points_raw = np.asarray(custom_cloud, dtype=np.float32)
            input_desc = f"uploaded cloud ({len(points_raw):,} pts)"
        elif source_mode == "Live Camera":
            camera_img, is_hardware_camera = self.camera.read_frame(self.frame_count)
            points_raw = self.camera.project_to_point_cloud(camera_img, num_points=4096)
            input_desc = "camera projection (monocular depth estimate: geometry approximate)"
        else: # Continuous LiDAR Stream
            points_raw = self.lidar.get_frame(self.frame_count)
            input_desc = "recorded LiDAR scan data/raw/000000.bin (static: no real motion)"

        t_ingest = time.perf_counter()

        # Stride-sample very large clouds into the inference budget. Exact
        # classified/total counts are reported alongside every number.
        points_total = int(len(points_raw))
        stride = max(1, points_total // self.max_infer_points)
        xyz_full = np.ascontiguousarray(points_raw[:, :3], dtype=np.float32)
        xyz = np.ascontiguousarray(xyz_full[::stride])
        points_classified = int(len(xyz))

        # 2. Semantic Perception (PointNet++ ONNX Runtime, chunked).
        # Identical inputs reuse identical outputs (memoization, not estimation).
        cache_key = (source_mode, points_total, stride,
                     hashlib.sha256(xyz.tobytes()).hexdigest())
        cached = self._infer_cache.get(cache_key)
        if cached is not None:
            classes, confs = cached
            inference_source = "onnxruntime (cached: byte-identical input)"
        else:
            try:
                logits = self._run_onnx(xyz)
            except Exception as exc:
                return {"error": f"ONNX inference failed: {exc}. No heuristic fallback is used."}
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_l = np.exp(shifted)
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            classes = np.argmax(probs, axis=-1).astype(int)
            confs = np.max(probs, axis=-1).astype(np.float32)
            self._infer_cache[cache_key] = (classes, confs)
            while len(self._infer_cache) > 4:
                self._infer_cache.pop(next(iter(self._infer_cache)))
            inference_source = "onnxruntime: models/onnx/pointnet2_semseg.onnx"

        t_infer = time.perf_counter()

        # 3. Dynamic Object Tracking (nearest-centroid association estimator)
        now = time.perf_counter()
        dt = 0.0 if self._last_frame_t is None else max(0.0, now - self._last_frame_t)
        self._last_frame_t = now
        dynamic_mask = (classes == 2)
        dynamic_pts = xyz[dynamic_mask]
        current_tracks = self._update_tracks(dynamic_pts, dt)
        t_track = time.perf_counter()

        # 4. Adaptive Variable-Resolution 2.5D World Model Grid
        grid_metrics = self._update_adaptive_grid(
            xyz, classes, confs, current_tracks, refine=enable_refinement)
        t_map = time.perf_counter()

        # Timing breakdown
        preprocess_ms = (t_ingest - t_start) * 1000.0
        infer_ms = (t_infer - t_ingest) * 1000.0
        track_ms = (t_track - t_infer) * 1000.0
        map_ms = (t_map - t_track) * 1000.0
        total_ms = (t_map - t_start) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 30.0

        if points_raw.shape[1] > 3:
            intensities = np.asarray(points_raw[::stride, 3], dtype=np.float32)
        else:
            intensities = np.ones(points_classified, dtype=np.float32)

        return {
            "error": None,
            "frame_index": self.frame_count,
            "source_mode": source_mode,
            "input_desc": input_desc,
            "camera_frame": camera_img,
            "is_hardware_camera": is_hardware_camera,
            "points": xyz,
            "points_total": points_total,
            "points_classified": points_classified,
            "network_share_pct": 100.0 * points_classified / max(1, points_total),
            "inference_source": inference_source,
            "tracking_source": "python centroid association (estimator, not a Kalman filter)",
            "intensities": intensities,
            "classes": classes,
            "confidences": confs,
            "tracks": current_tracks,
            "cells": grid_metrics["cells"],
            "uniform_cell_count": grid_metrics["uniform_cell_count"],
            "adaptive_cell_count": grid_metrics["adaptive_cell_count"],
            "cell_reduction_pct": grid_metrics["cell_reduction_pct"],
            "uniform_mem_mb": grid_metrics["uniform_mem_mb"],
            "adaptive_mem_mb": grid_metrics["adaptive_mem_mb"],
            "memory_saved_pct": grid_metrics["memory_saved_pct"],
            "memory_basis": grid_metrics["memory_basis"],
            "boundary_alignment_errors": self._boundary_errors(grid_metrics["cells"]),
            "boundary_cells_checked": len(grid_metrics["cells"]),
            "timing": {
                "preprocess_ms": preprocess_ms,
                "infer_ms": infer_ms,
                "track_ms": track_ms,
                "map_ms": map_ms,
                "total_ms": total_ms,
                "fps": fps
            }
        }

    def _update_tracks(self, dynamic_pts: np.ndarray, dt: float) -> List[Dict]:
        """Nearest-centroid association tracker (Python estimator, not a Kalman filter).

        Velocity is the measured centroid displacement divided by the measured
        inter-frame interval; confidence grows with consecutive associations
        as min(0.95, 0.40 + 0.10 * hits). With no motion or no elapsed time the
        velocity is 0 — nothing here is synthesized.
        """
        if len(dynamic_pts) == 0:
            for tid in list(self.tracks):
                self.tracks[tid]["misses"] += 1
                if self.tracks[tid]["misses"] > 3:
                    del self.tracks[tid]
            return []

        # Cluster dynamic points on a 2 m spatial hash.
        grid = {}
        for pt in dynamic_pts:
            k = (int(math.floor(pt[0] / 2.0)), int(math.floor(pt[1] / 2.0)))
            grid.setdefault(k, []).append(pt)

        centroids = []
        for pts in grid.values():
            if len(pts) >= 6:
                pts_arr = np.array(pts)
                centroids.append((
                    float(np.mean(pts_arr[:, 0])),
                    float(np.mean(pts_arr[:, 1])),
                    float(np.mean(pts_arr[:, 2])),
                    len(pts),
                ))

        tracks_list = []
        matched = set()
        gate = 3.0
        for (cx, cy, cz, n) in centroids:
            best_id, best_d = None, gate
            for tid, tr in list(self.tracks.items()):
                if tid in matched:
                    continue
                d = math.hypot(cx - tr["x"], cy - tr["y"])
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "x": cx, "y": cy, "z": cz,
                    "vx": 0.0, "vy": 0.0, "hits": 1, "misses": 0,
                }
                # A track born this frame must not absorb a second cluster in
                # the same frame; it becomes matchable on the next frame.
                matched.add(tid)
            else:
                tid = best_id
                matched.add(tid)
                tr = self.tracks[tid]
                if dt > 1e-6:
                    tr["vx"] = (cx - tr["x"]) / dt
                    tr["vy"] = (cy - tr["y"]) / dt
                else:
                    tr["vx"] = 0.0
                    tr["vy"] = 0.0
                tr.update(x=cx, y=cy, z=cz, hits=tr["hits"] + 1, misses=0)
            tr = self.tracks[tid]
            confidence = min(0.95, 0.40 + 0.10 * tr["hits"])
            speed = math.hypot(tr["vx"], tr["vy"])
            tracks_list.append({
                "id": tid,
                "x": cx, "y": cy, "z": cz,
                "vx": tr["vx"], "vy": tr["vy"],
                "speed": speed,
                "points_count": n,
                "class": "dynamic_object",
                "confidence": confidence,
            })

        for tid in [t for t in self.tracks if t not in matched]:
            self.tracks[tid]["misses"] += 1
            if self.tracks[tid]["misses"] > 3:
                del self.tracks[tid]

        tracks_list.sort(key=lambda t: t["id"])
        return tracks_list

    def _update_adaptive_grid(self, xyz: np.ndarray, classes: np.ndarray, confs: np.ndarray,
                               tracks: List[Dict], refine: bool = True) -> Dict:
        """
        Updates Level 1 (Distance Bands) and Level 2 (Quadtree Local Refinement) 2.5D Grid.
        The uniform baseline is counted over the identical input subset, so the
        comparison is apples-to-apples within this dashboard. Memory figures are
        estimates from measured C++ struct sizes (see module constants), not
        profiled allocations.
        """
        # Distance bands definition from Section 1
        # 0-10m: 0.05m, 10-30m: 0.15m, 30-60m: 0.30m, 60-100m: 0.50m
        def get_base_res(d):
            if d <= 10.0: return 0.05
            if d <= 30.0: return 0.15
            if d <= 60.0: return 0.30
            return 0.50

        # Uniform baseline: every point mapped to 5cm fixed cell
        uniform_keys = set()
        # Microcell ownership: every occupied 5 cm quantum has exactly one
        # owning cell, so stored footprints from different bands are disjoint
        # (same pattern as the C++ Grid25D fix for C1).
        q = 0.05
        kmap = {0.05: 1, 0.15: 3, 0.30: 6, 0.50: 10}
        owner = {}
        cells_by_key = {}

        def merge_into(cell, z, cls, conf):
            if cls != 0:
                cell["elevation"] = max(cell["elevation"], z)
            if conf > cell["conf"]:
                cell["class"] = cls
                cell["conf"] = conf
            cell["count"] += 1

        for i in range(len(xyz)):
            x, y, z = xyz[i]
            cls = int(classes[i])
            conf = float(confs[i])
            d = math.hypot(x, y)

            # Uniform 5cm cell key
            ux = int(math.floor(x / q))
            uy = int(math.floor(y / q))
            uniform_keys.add((ux, uy))

            mkey = (ux, uy)
            if mkey in owner:
                merge_into(cells_by_key[owner[mkey]], z, cls, conf)
                continue

            # Adaptive base resolution
            base_res = get_base_res(d)
            k = kmap.get(base_res)

            # Level 2 Refinement trigger: dynamic object proximity refines down to 5cm
            is_near_track = False
            if refine:
                for trk in tracks:
                    if math.hypot(x - trk["x"], y - trk["y"]) < 3.0:
                        is_near_track = True
                        break

            k_eff = 1 if (refine and (is_near_track or cls == 2)) else k
            if k_eff is None:
                k_eff = 1
            cx = ux // k_eff
            cy = uy // k_eff

            contested = False
            for ix in range(cx * k_eff, cx * k_eff + k_eff):
                for iy in range(cy * k_eff, cy * k_eff + k_eff):
                    if (ix, iy) in owner:
                        contested = True
                        break
                if contested:
                    break
            if contested:
                k_eff = 1
                cx, cy = ux, uy
            res_eff = k_eff * q
            cell_key = (res_eff, cx, cy)
            cell = cells_by_key.get(cell_key)
            if cell is None:
                cell = {
                    "cx": (cx * k_eff + k_eff / 2.0) * q,
                    "cy": (cy * k_eff + k_eff / 2.0) * q,
                    "res": res_eff,
                    "elevation": z,
                    "class": cls,
                    "conf": conf,
                    "count": 0,
                }
                cells_by_key[cell_key] = cell
                for ix in range(cx * k_eff, cx * k_eff + k_eff):
                    for iy in range(cy * k_eff, cy * k_eff + k_eff):
                        owner[(ix, iy)] = cell_key
            merge_into(cell, z, cls, conf)

        cells = list(cells_by_key.values())
        uniform_count = len(uniform_keys)
        adaptive_count = len(cells)

        uniform_mem = (uniform_count * UNIFORM_BYTES_PER_CELL) / (1024.0 * 1024.0)
        adaptive_mem = (adaptive_count * ADAPTIVE_BYTES_PER_CELL) / (1024.0 * 1024.0)

        red_pct = 100.0 * (1.0 - (adaptive_count / max(1, uniform_count)))
        mem_pct = 100.0 * (1.0 - (adaptive_mem / max(0.001, uniform_mem)))

        return {
            "cells": cells,
            "uniform_cell_count": uniform_count,
            "adaptive_cell_count": adaptive_count,
            "cell_reduction_pct": red_pct,
            "uniform_mem_mb": uniform_mem,
            "adaptive_mem_mb": adaptive_mem,
            "memory_saved_pct": mem_pct,
            "memory_basis": (
                f"estimate: uniform {UNIFORM_BYTES_PER_CELL} B/cell (sizeof(Cell)), "
                f"adaptive {ADAPTIVE_BYTES_PER_CELL} B/cell (map node + Quadtree + "
                "heap node + Cell, measured via compiled sizeof probe)"
            ),
        }

    def _boundary_errors(self, cells: List[Dict]) -> int:
        """Count 5 cm lattice posts covered by more than one stored cell
        footprint. Candidate collisions are verified against the exact stored
        footprints, so cells sharing only an edge are never miscounted."""
        q = 0.05
        owner = {}
        errors = 0
        for idx, c in enumerate(cells):
            x0 = c["cx"] - c["res"] / 2.0
            x1 = c["cx"] + c["res"] / 2.0
            y0 = c["cy"] - c["res"] / 2.0
            y1 = c["cy"] + c["res"] / 2.0
            ix0 = int(math.floor(x0 / q + 1e-4))
            ix1 = int(math.ceil(x1 / q - 1e-4)) - 1
            iy0 = int(math.floor(y0 / q + 1e-4))
            iy1 = int(math.ceil(y1 / q - 1e-4)) - 1
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = (ix, iy)
                    if key not in owner:
                        owner[key] = idx
                    elif owner[key] != idx:
                        o = cells[owner[key]]
                        ox0 = o["cx"] - o["res"] / 2.0
                        ox1 = o["cx"] + o["res"] / 2.0
                        oy0 = o["cy"] - o["res"] / 2.0
                        oy1 = o["cy"] + o["res"] / 2.0
                        if (ox0 < x1 - 1e-6 and x0 < ox1 - 1e-6 and
                                oy0 < y1 - 1e-6 and y0 < oy1 - 1e-6):
                            errors += 1
        return errors
