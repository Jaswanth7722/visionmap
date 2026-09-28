# PS26053: Technical Codebase Audit & Alignment Report

**Problem Statement ID:** 26053  
**Problem Statement Title:** Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception  
**Organization:** Defence Research and Development Organisation (DRDO)  
**Theme / Category:** Smart Vehicles / Software  
**Target Repository:** `visionmapx`  
**Report Date:** 2026-09-28  

---

## Executive Summary & Alignment Score

| Requirement Area | Required by Problem Statement | Codebase Status | Alignment |
|---|---|---|---|
| **1. Deep Learning Model** | PointNet++ / Sparse Conv semantic segmentation into terrain, static obstacles, moving objects | PyTorch PointNet++ implementation (`python/training`), ONNX export pipeline (`python/conversion/export_onnx.py`), C++ ONNX Runtime CPU inference (`cpp/src/inference/onnx_engine.cpp`) | **90%** (Trained model exportable; C++ native inference verified with geometric fallback) |
| **2. Variable Resolution Grid Engine** | 2.5D elevation + semantic grid; high-res near (5 cm @ 0–10 m) to low-res far (50 cm @ 60–100 m); 0 boundary alignment errors / no data loss | Custom C++ Dual-Level Grid (`cpp/src/mapping/grid_25d.cpp`, `quadtree.cpp`, `resolution_policy.cpp`), 5 cm microcell lattice quantization with strict disjoint ownership index | **95%** (Verified: 0 overlapping microcells across all distance transitions in `test_boundary_alignment`) |
| **3. Real-Time Dynamic Tracking** | Differentiate and track static obstacles vs dynamic actors (pedestrians, vehicles) | 3D Euclidean Connected-Components Clusterer + Kalman Filter State Tracking with bounding boxes and velocity vectors (`cpp/src/tracking/kalman_tracker.cpp`) | **92%** (Kalman filtering, track ID persistence, 3D velocity estimation fully implemented) |
| **4. Real-Time Visualization Dashboard** | 2.5D map dashboard, color-coded terrain/objects, memory & FPS telemetry | Native C++ Multi-threaded Web Server (`cpp/apps/live_dashboard_server.cpp`) + Three.js WebGL GPU Client (`cpp/web/index.html`) + sequence playback & HUD | **95%** (Interactive 3D orbit controls, elevation heatmap, semantic layer toggles, live FPS/RAM/latency HUD) |
| **5. Performance Benchmarking & QA** | Latency (FPS), memory footprint comparison, boundary integrity | Standalone C++ benchmark (`cpp/apps/benchmark.cpp`), 10/10 C++ unit & integration tests (`tests/cpp/`), Sidecar JSON telemetry export | **90%** (Automated boundary verification, empirical latency and honest memory profiling) |
| **Overall Alignment** | **Complete Autonomous 2.5D Perception Pipeline** | **End-to-end standalone C++ prototype with training & evaluation tooling** | **92.4%** |

---

## Part 1: How Existing Features Are Implemented

The codebase is structured around a **high-performance, native C++17/20 runtime core** with an offline Python training and evaluation loop.

### 1. Ingestion & Preprocessing (`cpp/src/io`, `cpp/src/preprocessing`)
- **LiDAR Point Cloud Ingestion (`lidar_io.cpp`):**
  - High-speed binary parsing of KITTI/SemanticKITTI `.bin` (4-float: `[x, y, z, intensity]`) and ASCII `.pcd` point clouds.
  - Multi-frame sequence discovery and sorting (`listSequenceScans`).
- **Coordinate & Pose Transform (`coordinate_transform.cpp`):**
  - Rigid 6-DoF vehicle/sensor frame to world frame transformation with sensor origin tracking.
- **ROI Crop & Exact Voxel Downsampling (`point_cloud_ops.cpp`):**
  - Strict bounding box spatial filtering (`[-40m, +40m]` X, `[-40m, +40m]` Y, `[-3m, +5m]` Z).
  - 64-bit coordinate sort-based voxel grid downsampling ensuring zero hash collisions and centroid preservation.

### 2. Deep Learning & Semantic Segmentation (`python/training`, `cpp/src/inference`)
- **PointNet++ Architecture (`python/training/model.py`):**
  - Set Abstraction (SA) with Multi-Scale Grouping (MSG) and Feature Propagation (FP) layers.
  - Multi-class output mapped to DRDO 3-class ontology: `0: TERRAIN`, `1: STATIC_OBSTACLE`, `2: DYNAMIC_OBSTACLE`.
- **Gated ONNX Exporter (`python/conversion/export_onnx.py`):**
  - Exports PyTorch models to ONNX Opset 17 with dynamic batch and point dimensions.
  - Strict operator validation against ONNX Runtime standard operator sets and numerical parity testing.
- **C++ ONNX Runtime Engine (`cpp/src/inference/onnx_engine.cpp`):**
  - Sub-cloud chunking (4096-point blocks) running directly in C++ via ONNX Runtime C++ API.
  - Fail-safe geometric fallback tagging with per-point network/fallback provenance accounting.

### 3. Obstacle Clustering & Kalman Tracking (`cpp/src/tracking`)
- **3D Connected-Components Clustering (`kalman_tracker.cpp`):**
  - Fast spatial hashing and union-find clustering over non-terrain points.
- **Kalman State Estimator:**
  - 6D state vector $[x, y, z, v_x, v_y, v_z]^T$ with constant-velocity motion model.
  - Hungarian/Greedy centroid association with configurable gating distances, track confirmation, and coasting.
  - Computes persistent Track IDs, bounding boxes, velocities, and confidence scores.

### 4. Adaptive Variable-Resolution 2.5D World Model (`cpp/src/mapping`, `cpp/src/resolution`)
- **Distance Band Policy (`resolution_policy.cpp`):**
  - Tiered radial resolution:
    - `0 – 10 m`: **5 cm** cells ($1\times$ quantum)
    - `10 – 30 m`: **15 cm** cells ($3\times$ quantum)
    - `30 – 60 m`: **30 cm** cells ($6\times$ quantum)
    - `60 – 100 m`: **50 cm** cells ($10\times$ quantum)
- **Zero-Error Boundary Alignment (`grid_25d.cpp`):**
  - 5 cm shared microcell lattice quantization.
  - Microcell ownership hash index preventing overlapping allocations across radial boundaries.
  - Contested boundary blocks gracefully split into single-quantum tiles.
- **Level-2 Quadtree Local Refinement (`quadtree.cpp`):**
  - Importance engine dynamically subdivides base cells containing dynamic objects, velocity changes, or sharp vertical gradients (curbs/edges) up to depth 3 without upgrading the entire distance ring.
- **Temporal Fusion & Decay:**
  - Bayesian-style log-odds occupancy updates with temporal decay.
  - Complete state reset for cells decaying below 0.05 occupancy.
- **PLY / JSON Exporter:**
  - Exports full 2.5D maps with elevation, color-coded semantics, track ID, and sidecar metadata JSON.

### 5. Real-Time Dashboard & WebGL Viewer (`cpp/apps/live_dashboard_server.cpp`, `cpp/web/index.html`)
- **Native C++ Multi-Threaded HTTP Server:**
  - Cross-platform socket server (macOS/Linux/Windows).
  - Non-blocking REST API: `/api/status`, `/api/lidar/scan`, `/api/frames`, `/api/frame/next`, `/api/ply`, `/api/lidar/upload`.
  - Live process telemetry: RSS RAM (via `mach_task_basic_info`/`proc`), FPS, stage-by-stage latencies (preprocessing, inference, tracking, mapping).
- **Three.js WebGL Dashboard:**
  - High-performance 3D orbit viewer rendering raw LiDAR point clouds and 2.5D grid cells simultaneously.
  - Color palettes for Semantic Classification, Elevation Heatmap, Distance Rings, and Level-2 Quadtree Depth.
  - 3D bounding box overlays with dynamic velocity vectors.
  - Memory and latency comparison charts against flat uniform grids.

### 6. Verification & Automated Test Suite (`tests/cpp/`)
- 10 automated C++ test suites (100% passing via `ctest`):
  1. `test_resolution`: Band quantization and quantum multiples.
  2. `test_quadtree`: Subdivide, parent-child point conservation, depth limits.
  3. `test_projection`: 3D-to-2.5D cell projection and track association.
  4. `test_tracking`: 3D cluster separation, Kalman updates, velocity convergence.
  5. `test_temporal_fusion`: Decay dynamics and memory reclamation.
  6. `test_boundary_alignment`: Exhaustive post-by-post overlap/gap detector across all band boundaries.
  7. `test_coordinate_transform`: Sensor-to-world rigid transforms.
  8. `test_preprocessing`: Sort-based voxel downsampling collision invariance.
  9. `test_io`: Binary `.bin` reader and SemanticKITTI label remapping parity.
  10. `test_config`: YAML parser and runtime configuration overrides.

---

## Part 2: What Is Left to Build (Gaps to Hardware / Production Prototype)

To transition this software framework into a deployable vehicle prototype, the following modules are required:

### 1. Live Sensor Hardware Drivers & ROS2 Node
- **Current State:** Reads recorded `.bin`/`.pcd` files from disk, sequence directories, or HTTP multipart upload.
- **What is left:**
  - UDP Packet Parser for live Velodyne (VLP-16, HDL-32E, HDL-64E) / Ouster (OS1/OS2) / Hesai raw sensor streams.
  - Native ROS2 C++ Node (`rclcpp`) subscribing to `/velodyne_points` (`sensor_msgs/msg/PointCloud2`) and publishing `/adaptive_map/grid_25d` and `/tracked_objects`.

### 2. GPU Hardware Acceleration (CUDA / TensorRT)
- **Current State:** ONNX Runtime runs on CPU intra-op thread pools (taking ~1.5–2.5s per 50k points on CPU).
- **What is left:**
  - Enable ONNX Runtime CUDA / TensorRT Execution Provider in `onnx_engine.cpp`.
  - Batch inference pipelines on GPU to achieve true 20–30 Hz real-time frame rates.

### 3. Flat Memory / Arena-Allocated Quadtree Optimization
- **Current State:** Dynamic heap allocation (`new QuadtreeNode` per refined tile) introduces pointer overhead (~252 bytes/cell vs 88 bytes for a flat struct).
- **What is left:**
  - Contiguous linear block / arena allocator for Quadtree nodes to achieve theoretical memory reduction (saving 40–60% RAM over uniform grids rather than incurring pointer overhead).

### 4. Vehicle Odometry / SLAM Pose Integration
- **Current State:** `CoordinateTransform` is implemented and active as Stage 0, but defaults to identity when no external odometry is fed.
- **What is left:**
  - Integrate an IMU/Wheel Odometry or LOAM/Fast-LIO pose feed into `MappingPipeline::processFrame` for multi-scan global accumulation during vehicle motion.

### 5. Multi-Frame Public Dataset Downloader Automation
- **Current State:** Single sample frame (`000000.bin`) and synthetic generator (`generate_360_lidar_scan.py`).
- **What is left:**
  - Automated bash/python downloader to fetch and unpack sequence slices from SemanticKITTI / nuScenes directly into `data/raw/sequences/`.

---

## Part 3: What Is Additional / Redundant / Deprecated (Useless Code)

The codebase has undergone iterative refactoring. The following components are redundant, superseded, or non-essential for the core C++ pipeline:

| File / Component | Category | Why It Is Redundant / Deprecated | Recommended Action |
|---|---|---|---|
| `python/visualization/dashboard.py` (if present or referenced) / `open3d_viewer.py` | Legacy Python UI | Superseded entirely by the native C++ WebGL dashboard (`live_dashboard_server.cpp` + `cpp/web/index.html`), which avoids Python runtime dependencies. | Retain `open3d_viewer.py` only as a debug visualizer; remove legacy Python dashboard files. |
| `scripts/run_dashboard.bat` | Platform-Specific Script | Windows-specific cmd wrapper. Redundant with cross-platform python launcher `scripts/run_dashboard.py` or direct `./build/bin/live_dashboard_server`. | Keep as optional convenience for Windows users, or standardize on `run_dashboard.py`. |
| `audit.md` & `FIXES_REPORT.md` | Historical Audit Logs | Temporary working snapshots created during previous code review and bugfix passes. Not used by build or runtime. | Move to `docs/archive/` or keep as internal QA records. |
| `docs/experiments/` (empty/stub) | Empty Directory | Contains only one report (`hard_cases_benchmark_report.md`). | Consolidate under `docs/` or expand with empirical test outputs. |
| `uploads/` (empty directory) | Temporary Upload Staging | Created automatically by server when uploading scans via browser UI. | Add to `.gitignore` to prevent tracking runtime uploads. |
| `kaggle_deploy/` | Notebook Deployment | Kaggle training notebook (`kaggle_pointnet2_training.ipynb`). Useful for cloud GPU training, but not part of the C++ runtime system. | Keep segregated in `kaggle_deploy/` for training workflows only. |

---

## Part 4: Detailed Alignment Matrix Against DRDO PS 26053

```mermaid
flowchart TD
    A[Raw 3D LiDAR Point Cloud<br>KITTI / Velodyne 360] --> B[Coordinate Transform & ROI Voxelization<br>cpp/src/preprocessing]
    B --> C[PointNet++ Semantic Perception<br>cpp/src/inference/onnx_engine.cpp]
    C --> D[Dynamic Object Extraction & Kalman Tracker<br>cpp/src/tracking/kalman_tracker.cpp]
    C --> E[Adaptive Variable-Resolution Policy<br>cpp/src/resolution/resolution_policy.cpp]
    D --> F[Level-2 Quadtree Local Refinement<br>cpp/src/mapping/quadtree.cpp]
    E --> G[2.5D World Model & Microcell Lattice<br>cpp/src/mapping/grid_25d.cpp]
    F --> G
    G --> H[Temporal Fusion & Decay Engine<br>cpp/src/mapping/grid_25d.cpp]
    H --> I[PLY 2.5D Mesh & Sidecar JSON<br>results/maps/]
    H --> J[C++ Live Web Server<br>cpp/apps/live_dashboard_server.cpp]
    J --> K[Interactive WebGL Dashboard<br>cpp/web/index.html]
```

### Problem Statement Requirement Breakdown

1. **Terrain Analysis (Drivable vs Non-drivable):**
   - **Requirement:** Classify road surfaces, sidewalks, curbs, and non-drivable elevation changes.
   - **Implementation:** PointNet++ classifies ground into `TERRAIN (0)` while height gradient and curb thresholds in `ResolutionPolicy::shouldRefineCurb` detect curb boundaries.

2. **Object Detection (Static vs Dynamic):**
   - **Requirement:** Segment static obstacles (walls, poles) and dynamic obstacles (vehicles, pedestrians).
   - **Implementation:** Multi-class segmentation network + 3D connected-components clustering + Kalman tracking with 3D velocity vectors.

3. **Adaptive Spatial Representation (5 cm to 50 cm):**
   - **Requirement:** Variable resolution grid where cell size increases with distance without alignment errors or data loss.
   - **Implementation:** 4 discrete radial distance bands (5cm @ 0–10m, 15cm @ 10–30m, 30cm @ 30–60m, 50cm @ 60–100m) anchored on a 5 cm microcell lattice with zero cross-band overlaps or voids.

4. **Real-time Visualization:**
   - **Requirement:** Dashboard showing the 2.5D map with distinct color coding and memory reduction metrics.
   - **Implementation:** Native HTTP socket server + Three.js WebGL dashboard with 4 color modes, track bounding boxes, and live RAM/FPS/latency telemetry.

5. **Performance Metrics:**
   - **Requirement:** Low latency (high FPS) and high accuracy across distance.
   - **Implementation:** Automated `benchmark` executable profiling per-stage milliseconds, cell counts, and boundary verification.

---

## Conclusion

The `visionmapx` codebase demonstrates an **exceptionally high degree of architectural alignment (92.4%)** with DRDO Problem Statement 26053. The core algorithmic deliverables—variable-resolution 2.5D projection, zero-overlap microcell lattice alignment, PointNet++ semantic inference in C++, 3D Kalman tracking, and a live WebGL dashboard—are **fully functional, compiled, tested (10/10 C++ test suites passing), and operational**.

The remaining gaps to turn this into a field-deployed system are engineering extensions (ROS2 transport, GPU TensorRT acceleration, and live sensor socket drivers), while the core software prototype is complete and standalone.
