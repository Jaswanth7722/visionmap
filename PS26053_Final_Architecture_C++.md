# PS 26053 — FINAL ARCHITECTURE (LOCKED, C++ RUNTIME)
**Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception**
DRDO | Smart Vehicles | Software

No further changes to language or stack. This is the build spec.

---

## 1. Core Principle

Two-level spatial adaptation:

- **Level 1 — PS baseline policy:** cell resolution set purely by distance from sensor.
- **Level 2 — Project innovation:** semantic importance + motion trigger local refinement inside a distance band, without upgrading the whole band.

Base resolution bands (configurable, PS examples are 5cm near / 50cm far):

| Range | Cell size |
|---|---|
| 0–10 m | 5 cm |
| 10–30 m | 15 cm |
| 30–60 m | 30 cm |
| 60–100 m | 50 cm |

---

## 2. Final Tech Stack (no substitutions)

| Layer | Technology | Why |
|---|---|---|
| Runtime language | **C++17/20** | Real-time pipeline, mapping core, all hot-path logic |
| Point cloud ops | **PCL + Eigen** | Filtering, downsampling, geometry, coordinate transforms |
| ML training (offline only) | **Python + PyTorch** | Only place Python appears — model training is not a runtime component |
| Segmentation model | **PointNet++** | Per-point semantic labels: terrain / static / dynamic |
| Model deployment | **ONNX export → ONNX Runtime C++** | Trained once in Python, executed natively in C++; zero Python in the live pipeline |
| Tracking | **Custom Kalman filter + association (C++)** | Track ID, position, velocity, temporal continuity |
| Spatial structure | **Custom Quadtree (C++)** | Local resolution refinement inside a base band |
| World model | **Custom C++ 2.5D grid** | Elevation + occupancy + semantics + motion, per cell |
| Visualization | **Open3D (C++/Python bindings)** | Live 3D/2.5D map render |
| Dashboard | **Streamlit + Plotly** | Metrics, resolution legend, tracked objects, benchmark charts (reads data emitted by the C++ core — not itself part of the real-time hot path) |
| Benchmarking | **C++ timers + NumPy/scikit-learn (Python, offline analysis)** | FPS, latency, memory, accuracy computation and plotting |
| Build | **CMake** | C++ dependency/build management |
| Config | **YAML** | Resolution bands, model paths, sensor params, tracking params |
| Version control | **Git/GitHub** | — |

**Explicitly out of core (optional/future only):** ROS 2, TensorRT, LibTorch (fallback only if ONNX export fails), pybind11/nanobind (tooling only), cloud backend, database, full SLAM, path planning, physical hardware dependency.

> Note: Python appears **only** in offline model training and offline benchmark plotting. The live runtime — ingestion through mapping through visualization data feed — is 100% C++. This is not a compromise; it's the standard split even in production autonomous stacks (train offline, deploy compiled).

---

## 3. Runtime Pipeline (per frame)

```
LiDAR Data / Recorded Replay
        |
        v
+-------------------------------+
| C++ INGESTION                 |
| ROI crop | noise filter |     |
| voxel downsampling            |
+---------------+---------------+
                |
                v
+-------------------------------+
| COORDINATE / POSE HANDLING    |
| sensor -> vehicle/world frame |
+---------------+---------------+
                |
                v
+-------------------------------+
| SEMANTIC PERCEPTION           |
| PointNet++ (trained Python)   |
| ONNX Runtime C++ inference    |
+---------------+---------------+
                |
                v
+-------------------------------+
| OBJECT EXTRACTION             |
| terrain regions | clusters    |
+---------------+---------------+
                |
                v
+-------------------------------+
| TRACKING                      |
| association | Kalman | ID     |
+---------------+---------------+
                |
                v
+-------------------------------+
| IMPORTANCE ENGINE             |
| distance + motion + semantic  |
+---------------+---------------+
                |
                v
+-------------------------------+
| BASE RESOLUTION POLICY        |
| 5/15/30/50 cm by distance     |
+---------------+---------------+
                |
                v
+-------------------------------+
| LOCAL REFINEMENT DECISION     |
+-------+------------+----------+
    YES |            | NO
        v            v
 +-------------+  +----------------+
 | QUADTREE    |  | KEEP BASE CELL |
 | REFINEMENT  |  | REPRESENTATION |
 +------+------+  +--------+-------+
        |                  |
        +--------+---------+
                 v
+-------------------------------+
| 3D -> 2.5D PROJECTION         |
| XY -> cell | Z -> elevation   |
| class -> semantic layer       |
+---------------+---------------+
                |
                v
+-------------------------------+
| 2.5D WORLD MODEL              |
| elevation|occupancy|semantics |
| confidence|object ID|motion   |
+---------------+---------------+
                |
                v
+-------------------------------+
| TEMPORAL FUSION                |
| update | predict/move | decay  |
+---------------+---------------+
        |                |
        v                v
+---------------+  +------------------+
| OPEN3D VIEW   |  | PERFORMANCE / QA |
| adaptive map  |  | FPS|latency|mem  |
+---------------+  +--------+---------+
                            |
                            v
                +---------------------+
                | UNIFORM vs ADAPTIVE |
                | BENCHMARK           |
                +---------------------+
```

---

## 4. Cell Data Structure (2.5D World Model)

```
Cell {
    bounds              // explicit spatial extent, no implicit sizing
    resolution          // this cell's actual size
    elevation           // summarized height, not raw 3D
    ground_z, min_z, max_z, clearance   // optional vertical summary
    occupancy
    semantic_class      // see SemanticClass enum below
    semantic_confidence
    object_id
    velocity_x, velocity_y
    importance
    timestamp
}
```

### 4.1 Semantic Classes (PointNet++ output, C++ enum)

| Value | C++ Enum | Meaning | Visualization Color |
|---|---|---|---|
| `0` | `TERRAIN` | **Road surface / ground plane** — drivable area, flat ground, dirt tracks | 🟢 Emerald Green `#10b981` |
| `1` | `STATIC_OBSTACLE` | **Fixed structures** — buildings, walls, trees, poles, parked objects, kerbs | 🔵 Cyan `#00f3ff` |
| `2` | `DYNAMIC_OBSTACLE` | **Moving objects** — vehicles, pedestrians, cyclists; Kalman-tracked with ID | 🟠 Amber `#f59e0b` |
| `255` | `UNKNOWN` | Unclassified / low-confidence points | Grey |

These labels come directly from **PointNet++ ONNX Runtime inference** (zero Python in the live path). The same class codes propagate from the per-point label into `Cell.semantic_class` during 3D→2.5D projection. The dashboard renders them with the colours above in **Semantics mode**, and all bounding boxes around Tracked Obstacles (class 2) are drawn in **Amber**.

**Alignment rules (mandatory — this is the PS's core technical trap):**
1. One single world/vehicle coordinate convention for every cell, always.
2. Every cell defined by explicit bounds + resolution — never inferred.
3. Deterministic point-to-cell mapping so adjacent resolution levels never gap or overlap.
4. On subdivision, parent cell state is preserved or deterministically redistributed to children — never dropped.
5. Timestamp + confidence retained per cell so temporal fusion can separate fresh vs stale data.

---

## 5. AI Training → Deployment Flow

```
OFFLINE (Python):
Dataset (SemanticKITTI-style) → PyTorch → PointNet++ training
→ evaluation → ONNX export

RUNTIME (C++, no Python):
C++ point cloud → ONNX Runtime C++ → semantic labels → C++ mapping pipeline
```

**Critical safeguard:** validate ONNX export + operator support on Day 1, before building anything else downstream. If the PointNet++ implementation uses unsupported ops, swap implementation or fall back to LibTorch — do not discover this after the pipeline is built around it.

---

## 6. Component Responsibility Table

| Component | Language | Responsibility |
|---|---|---|
| LiDAR ingestion | C++ | Read sensor/recorded frames, maintain timing |
| Preprocessing | C++/PCL/Eigen | ROI crop, filtering, voxel downsampling, coordinate prep |
| Semantic perception | Python(train)/C++(infer) | Per-point labels via PointNet++ → ONNX |
| Object extraction | C++ | Points → terrain regions + object clusters |
| Tracking | C++ | Track IDs, position, velocity, association |
| Importance engine | C++ | Distance + motion + semantic + relevance score |
| Base resolution policy | C++ | PS distance-band resolution assignment |
| Local refinement | C++ | Quadtree subdivision where justified |
| 3D→2.5D projection | C++ | XY→cell, Z→elevation summary |
| World model | C++ | Persistent grid: elevation/occupancy/semantics/confidence |
| Temporal fusion | C++ | Update, predict, decay stale data |
| Visualization | Open3D + Streamlit/Plotly | Live map + metrics dashboard |
| Evaluation | Python + C++ instrumentation | Uniform-vs-adaptive comparison |

---

## 7. Benchmark Plan (mandatory deliverable, not optional polish)

Run identical LiDAR sequence through two modes:
- **Uniform high-resolution** (fixed fine cell size everywhere)
- **Adaptive variable-resolution** (this system)

Report, for both:
- Memory: peak + average
- Cell count: total stored cells
- Latency: end-to-end + per-stage breakdown
- FPS: measured from actual frame timing
- Semantic accuracy: IoU / precision / recall / F1
- Mapping quality: elevation error, object localization error
- **Boundary-alignment QA**: gap/overlap count across N frames (0 expected) — cheap to log, few teams will think to report it

---

## 8. Build Order (de-risked)

1. Day 1: PointNet++ → ONNX → ONNX Runtime C++ minimal load-and-run test (highest-risk link, validate first). In parallel: PCL/Eigen/CMake "hello point cloud" compiles.
2. Ingestion + preprocessing (verify visually in Open3D).
3. Coordinate/pose transform.
4. PointNet++ full training + ONNX inference wired into pipeline.
5. Base distance-band resolution grid only (no quadtree yet) — this alone satisfies the PS minimum.
6. Object extraction + Kalman tracking.
7. Importance engine + Quadtree local refinement.
8. Temporal fusion.
9. Uniform-resolution baseline pipeline (for benchmark comparison).
10. Streamlit dashboard (both modes side by side) + Open3D live view.
11. Full benchmark run + boundary-alignment QA logging.

Commit to git after each stage passes its own standalone test. Never integrate two unverified modules at once.

---

## 9. Project File Structure

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
│   ├── raw/ | sequences/ | processed/ | labels/
├── models/
│   ├── checkpoints/ | onnx/ | metadata/
├── cpp/
│   ├── include/ps26053/{common,io,preprocessing,coordinate,inference,perception,tracking,importance,resolution,mapping,runtime}/
│   ├── src/{io,preprocessing,coordinate,inference,perception,tracking,importance,resolution,mapping,runtime}/
│   └── apps/
│       ├── lidar_mapper.cpp
│       └── benchmark.cpp
├── python/
│   ├── training/{dataset.py, model.py, train.py, evaluate.py}
│   ├── conversion/export_onnx.py
│   ├── evaluation/{metrics.py, benchmark_analysis.py, plots.py}
│   └── visualization/{open3d_viewer.py, dashboard.py, plot_builder.py}
├── tests/
│   ├── cpp/{test_quadtree.cpp, test_resolution.cpp, test_projection.cpp, test_tracking.cpp, test_temporal_fusion.cpp}
│   └── python/{test_model.py, test_onnx.py, test_metrics.py}
├── scripts/
│   ├── preprocess.py | train.py | export_onnx.py | run_mapper.sh | benchmark.sh
├── results/
│   ├── maps/ | benchmarks/ | metrics/ | figures/
└── docs/
    ├── architecture/ | experiments/ | presentation/
```

---

## 10. Innovation Statement (final)

> "We dynamically allocate map resolution where the environment needs detail — combining PS-mandated distance-based foveation with semantic and motion-driven local refinement — while preserving elevation, semantics, and temporal continuity in a lightweight 2.5D world model, and we prove the efficiency gain with a direct uniform-vs-adaptive benchmark rather than a claim."

---

## 11. Frozen Decisions (do not revisit)

```
RUNTIME              = C++17/20, no exceptions
MODEL TRAINING       = Python/PyTorch, offline only
MODEL DEPLOYMENT     = ONNX Runtime C++, no Python in hot path
DISTANCE BANDS       = PS base resolution policy
QUADTREE             = local refinement mechanism only
2.5D WORLD MODEL     = elevation + occupancy + semantics + motion
OPEN3D               = live map view
STREAMLIT/PLOTLY     = metrics/status dashboard (reads C++ output, not in hot path)
UNIFORM vs ADAPTIVE  = primary proof-of-value experiment
```

Do not add frameworks beyond this list unless a profiled bottleneck demands it.
