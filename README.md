# PS 26053 — Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception
**DRDO | Smart Vehicles | Software Autonomous Navigation Pipeline**

This repository implements the production-grade **C++17/20 runtime architecture** specified in `PS26053_Final_Architecture_C++.md`. It balances perception precision and memory/computational latency through a dual-level spatial adaptation framework:

- **Level 1 (Baseline Distance Bands):** Sensor-centric radial foveation (0–10m @ 5cm, 10–30m @ 15cm, 30–60m @ 30cm, 60–100m @ 50cm).
- **Level 2 (Semantic & Motion Innovation):** Dynamic obstacle velocity and elevation gradient trigger Quadtree local refinement inside distance bands without upgrading entire bands.
- **Zero-Python Hot Path:** Model training is done offline in PyTorch, deployed via ONNX export and executed purely in **C++17 via ONNX Runtime C++**.

---

## 0. Quick Start (Windows PowerShell, from the repo root)

```powershell
# Build everything (MinGW + Ninja, already configured in build/)
cmake --build build --parallel 4

# Run all C++ tests
ctest --test-dir build --output-on-failure

# Run all Python tests (training/export/eval utilities only)
pytest tests/python/

# Map one frame -> 2.5D world model PLY
.\build\bin\lidar_mapper.exe data\raw\000000.bin models\onnx\pointnet2_semseg.onnx results\maps\adaptive_map_frame000000.ply

# Uniform-vs-adaptive benchmark with QA + coverage report
.\build\bin\benchmark.exe data\raw\000000.bin models\onnx\pointnet2_semseg.onnx

# Live dashboard: 3D view + 2.5D BEV + tracks + measured metrics (allow ~1 min
# for the first ONNX load + full-frame CPU inference, then open the URL)
.\build\bin\live_dashboard_server.exe 8080
# or: python scripts\run_dashboard.py [port]
# then open http://localhost:8080
```

---

## 1. Project Directory Structure

```
ps26053-lidar-mapping/
├── README.md
├── CMakeLists.txt
├── requirements.txt
├── config/
│   ├── system.yaml
│   ├── lidar.yaml
│   ├── model.yaml
│   ├── resolution.yaml
│   └── tracking.yaml
├── data/
│   ├── raw/ (e.g. 000000.bin)
│   ├── sequences/ (SemanticKITTI format)
│   ├── processed/
│   └── labels/ (000000.label)
├── models/
│   ├── checkpoints/ (pointnet2_trained_best.pth)
│   ├── onnx/ (pointnet2_semseg.onnx, pointnet2_semseg.json)
│   └── metadata/ (model_metadata.json)
├── cpp/
│   ├── include/ps26053/
│   │   ├── common/ (cell.hpp, types.hpp)
│   │   ├── io/ (lidar_io.hpp)
│   │   ├── preprocessing/ (point_cloud_ops.hpp)
│   │   ├── resolution/ (resolution_policy.hpp)
│   │   ├── mapping/ (grid_25d.hpp, quadtree.hpp)
│   │   ├── inference/ (onnx_engine.hpp)
│   │   ├── tracking/ (kalman_tracker.hpp)
│   │   └── runtime/ (pipeline.hpp)
│   ├── src/ (implementation of core modules)
│   └── apps/
│       ├── lidar_mapper.cpp (main pipeline executable)
│       └── benchmark.cpp (uniform vs adaptive benchmark)
├── python/
│   ├── training/ (dataset.py, model.py, train.py, evaluate.py)
│   ├── conversion/ (export_onnx.py)
│   ├── evaluation/ (evaluate_hard_cases.py)
│   └── visualization/ (open3d_viewer.py, project_traffic_jam_photorealistic.py)
│       # NOTE: the live runtime and dashboard are C++-only
│       # (cpp/apps/live_dashboard_server.cpp + cpp/web/); no Python in the loop.
├── tests/
│   ├── cpp/ (test_resolution, test_quadtree, test_projection, test_tracking,
│   │         test_temporal_fusion, test_boundary_alignment,
│   │         test_coordinate_transform, test_preprocessing, test_io, test_config)
│   └── python/ (test_model.py, test_onnx.py, test_dataset.py, test_images_as_pointclouds.py)
├── third_party/
│   └── onnxruntime/ (prebuilt ONNX Runtime C++ v1.20.1 headers & libs)
├── results/
│   ├── maps/ (exported 2.5D PLY world models)
│   ├── benchmarks/
│   ├── metrics/
│   └── figures/
└── docs/
    ├── architecture/
    └── experiments/ (hard_cases_benchmark_report.md)
```

---

## 2. Environment Setup

### C++ Toolchain & Libraries
- **Compiler:** GCC 15.2 (MinGW-w64) / Clang / MSVC with C++17 support.
- **Build System:** CMake >= 3.16 + Ninja.
- **Eigen3:** Linear algebra and coordinate transformations (`mingw-w64-x86_64-eigen3`).
- **yaml-cpp:** Configuration parsing (`mingw-w64-x86_64-yaml-cpp`).
- **ONNX Runtime C++:** Native C++ deployment engine (included under `third_party/onnxruntime`).

### Python Environment (Offline Training & Benchmarking Only)
Install required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Building the C++ Pipeline

To configure and build with CMake and Ninja:
```powershell
# Configure (only needed if build/ is deleted)
cmake -B build -G "Ninja" -DCMAKE_CXX_COMPILER=g++

# Build all libraries, executables, and tests
cmake --build build --parallel 4
```

---

## 4. Running the C++ Applications

### A. Run Main LiDAR Mapper
Runs full ingestion, PointNet++ C++ inference, Kalman tracking, and 2.5D world model generation:
```powershell
.\build\bin\lidar_mapper.exe data\raw\000000.bin models\onnx\pointnet2_semseg.onnx results\maps\adaptive_map_frame000000.ply
```

### B. Run Uniform vs Adaptive Benchmark
Direct proof-of-value benchmark comparing uniform fine grid (5 cm) against PS26053 adaptive variable-resolution:
```powershell
.\build\bin\benchmark.exe data\raw\000000.bin models\onnx\pointnet2_semseg.onnx
```

### C. Run Live Dashboard Server (C++-only runtime and dashboard)
```powershell
.\build\bin\live_dashboard_server.exe 8080
# or: python scripts\run_dashboard.py [port]
```
then open http://localhost:8080 — 3D semantic view, 2.5D BEV, tracks, measured metrics and boundary QA, all served from the native pipeline with no Python in the loop.

### Benchmark Results on Example Scan (122,626 Points, measured 2026-09-26):
| Metric | Uniform Baseline (5 cm) | Adaptive (PS26053) | Delta |
|---|---|---|---|
| **Shared Input Points** | 49,661 | 49,661 | identical input |
| **Active Stored Cells** | 35,907 | 27,476 | **−23.48% fewer cells** |
| **Memory Footprint** | 3.01 MB | 14.95 MB | **−396.24% (adaptive uses MORE — Quadtree-per-cell overhead, tracked design issue)** |
| **Processing Latency** | 17 ms (key-counting only) | ~2.4 s (full CPU inference) | ~0.4 FPS; GPU required for real time |
| **Network-Labeled Points** | n/a | 49,661 (100.00%) | 0 fallback |
| **Boundary Alignment Errors** | n/a (stores no cells) | 0 | **0 Errors (PASS)** |

---

## 5. Running the Test Suite

Run all C++ unit tests:
```powershell
ctest --test-dir build --output-on-failure
```

Run all Python tests (training/export/eval utilities):
```powershell
pytest tests/python/
```
Verified passing tests (10 C++ + Python suite):
- `test_resolution`: Distance-band resolution assignment and Level-2 motion/gradient triggers.
- `test_quadtree`: Quadtree subdivision, deterministic state redistribution, and refine-to-depth descent.
- `test_projection`: 3D point cloud to 2.5D elevation and clearance calculation; track association preserves network semantics.
- `test_tracking`: 3D connected-component clustering, real-timestamp dt, history-based confidence.
- `test_temporal_fusion`: Occupancy decay plus full clearing of departed cells.
- `test_boundary_alignment`, `test_coordinate_transform`, `test_preprocessing`, `test_io`, `test_config`: band-transition alignment QA, sensor transform, exact voxelization, shared label remap, YAML config loading.
