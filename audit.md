Repo state confirmation: `D:\053` is NOT a Git repository — `git status` returns `fatal: not a git repository (or any of the parent directories): .git` and `git diff` returns `warning: Not a git repository. Use --no-index to compare two paths outside a working tree`; `Test-Path D:\053\.git` = False. No Git evidence is therefore obtainable. In its place, a full-tree MD5+size snapshot of all 147 non-archive files was taken before the audit and re-verified after: **147 files before, 147 files after, 0 added, 0 removed, 0 modified — the audited tree was byte-for-byte identical to its pre-audit state.** The only file this audit adds is this `D:\053\audit.md` (final count 148). All analysis ran read-only; scratch scripts, the compiled `sizeof` probe, and simulations were written exclusively to `C:\Users\Jaswanth\AppData\Local\Temp\opencode\`. One transparency note: importing `python/training/model.py` for the PyTorch-vs-ONNX equivalence check caused CPython to write `python/training/__pycache__/model.cpython-313.pyc` into the tree; that file and its previously non-existent parent directory were deleted, and the 0-difference verification above was taken after that cleanup.

# PS26053 — Full-System Audit

**Target:** `D:\053`
**Specification:** `D:\053\PS26053_Final_Architecture_C++.md` (locked; 13,313 bytes)
**Audit type:** Read-only static + artifact + numerical analysis. No source file was modified.
**Only input frame available:** `D:\053\data\raw\000000.bin` (122,626 points)

## Severity Summary

| Severity | Count |
|---|---|
| Critical | 10 |
| High | 13 |
| Medium | 12 |
| Low | 8 |
| **Total confirmed** | **43** |
| Verified correct (no action) | 6 |
| Needs verification (not confirmed) | 5 |

The single most important result: **the delivered map artifact contains 26.58% overlapping and 2.46% gapped coverage — roughly 29% of the occupied map area is geometrically wrong — while all four independent QA surfaces in the product report zero boundary errors.** The 0-gap / 0-overlap guarantee that the architecture treats as a core deliverable is neither implemented nor measured, and the reported number is hardcoded in every layer that displays it.

---

## CRITICAL

### C1 — Boundary-alignment guarantee is violated in the shipped artifact, and every QA surface hardcodes "zero errors"
Measured directly on the delivered `results/maps/adaptive_map_frame000000.ply` (22,679 vertices) by rasterizing each cell's true square footprint onto a 5 cm lattice (3,760,000 posts):

```
lattice posts covered by >=1 exported cell : 183,335  (4.9% of lattice)
lattice posts covered by >=2 cells (OVERLAP):  48,739
lattice posts covered by >=3 cells        :   7,478
lattice posts covered by >=4 cells        :   2,915
max cells stacked on one 5 cm post        :       7
=> 26.58% of the occupied map area is double-covered

interior voids (uncovered post with covered posts directly above AND below): 2,210
interior voids (uncovered post with covered posts directly left AND right) : 2,301
=> 2.46% interior gaps  (total boundary error area ~= 29% of occupied area)
```

Meanwhile all four reporting surfaces state zero:
- `cpp/apps/benchmark.cpp:94-96` — uniform boundary errors and the `PASS` verdict are literals.
- `cpp/apps/live_dashboard_server.cpp` (`/api/status`) — `"boundary_errors": 0` is a literal.
- `python/visualization/dashboard.py:176` — `self.boundary_errors = 0.0`, set once in `__init__`.
- `cpp/web/index.html:679-682` — the `GAPS` HUD pill is static markup with no element `id` and no JS writer: `<span class="hud-value green">0.00 mm</span>`. It is decorative.

The defect is **strictly cross-band**: within any single resolution the export is perfect (0 duplicate footprints; median center spacing exactly equals the labeled resolution for 0.05/0.075/0.15/0.30 m). Cells at adjacent distance bands are laid out on independent grids, so band-boundary cells overlap and leave voids. This is precisely the "0–10 / 10–30 / 30–60 / 60–100 m transitions" risk the specification singles out, and no code in the repository checks for it.

### C2 — Benchmark compares non-equivalent inputs; the reported memory saving is actually a memory regression
`cpp/apps/benchmark.cpp` Mode 1 ("uniform") counts distinct raw 2D 5 cm XY cells with no ROI crop, no Z crop, and no voxel downsample. Mode 2 ("adaptive") runs the real pipeline, which applies all three. Reproduced on frame 000000:

```
ADAPTIVE tiles actually allocated by the C++ grid        : 22,673
UNIFORM cells counted by benchmark.cpp Mode 1 (raw scan) : 51,697
UNIFORM cells on the SAME 10 cm voxel cloud adaptive gets: 35,767

Reported cell reduction  (benchmark.cpp:76)             : 56.1% fewer
Apples-to-apples reduction (identical input)            : 36.6% fewer
=> ROI / z-crop asymmetry alone inflates the gain by 19.5 percentage points
```

The memory model is worse. `benchmark.cpp:61` charges **both** modes `sizeof(Cell)` and omits all per-cell container overhead. `sizeof` values measured by compiling the actual repo sources with the project's own toolchain (GCC 15.2.0 MinGW x64, outside the repo):

```
sizeof(Cell)          = 88      (what the benchmark charges)
sizeof(QuadtreeNode)  = 144
sizeof(Quadtree)      = 32
real per-cell cost    ~ 252 B   (map node + Quadtree + heap node + Cell + bucket)

uniform  claimed  51,697 x  88 B =  4.34 MB
adaptive claimed  22,673 x  88 B =  1.90 MB   <- what benchmark.cpp:61 reports
adaptive ACTUAL   22,673 x 252 B =  5.45 MB

reported memory saving : +56.1%
actual   memory saving : -25.6%   (adaptive uses ~26% MORE than the uniform baseline)
```

Root cause: the grid allocates one `Quadtree` object plus one heap `QuadtreeNode` **per base-resolution cell**, so a 5 cm cell costs ~252 B where a flat `Cell` array would cost 88 B. The adaptive mode's headline advantage — memory reduction — does not exist under a realistic model.

### C3 — Lossy 32-bit spatial hash in preprocessing silently merges points up to 24.9 m apart
`cpp/src/preprocessing/point_cloud_ops.cpp` (`voxelDownsample`, ~line 32) keys voxels with a 32-bit XOR of three linear congruences. Measured on the real scan:

```
true 10 cm voxels in ROI          : 49,661
distinct hash keys               : 49,397
collisions                       : 264  (0.5316%)
maximum separation of merged pts  : 24.9 m
```

Points ~25 m apart are fused into a single voxel and averaged, corrupting elevation, occupancy and semantic class with no warning. A 64-bit key, or a sorted `(ix,iy,iz)` tuple, removes this entirely.

### C4 — Inference silently degrades to a heuristic and still reports 0 ms
`cpp/src/inference/onnx_engine.cpp:18-38` swallows every exception during model load/parse; `infer()` at `:41-55` then falls back to a purely geometric classifier (height/flatness) and returns normally. `cpp/apps/lidar_mapper.cpp:31-33` continues on failed initialization. The result is a map that looks healthy but carries no network inference, with a reported `inference: 0.0 ms`.

How much of the real scan is actually classified by the network:

```
downsampled points reaching the grid : 49,397
points given a real ONNX label       :  4,096  ( 8.3%)
points given a z-heuristic label      : 45,301  (91.7%)
worst sweep point tested              : 20.5% network coverage
```

The architecture's central claim — PointNet++ semantic segmentation driving the map — is true for under 9% of points.

### C5 — No coordinate/pose transform; range bands are computed from a hardcoded origin
`PointCloudOps::transformPointCloud` is declared in the header and never called anywhere in the repository. `cpp/src/mapping/grid_25d.cpp:18-20` hardcodes `GridConfig` bounds (±60 m) and the sensor position defaults to the origin. The pipeline (`cpp/src/runtime/pipeline.cpp:28+`) has no coordinate-transform stage, so:

- every distance used to select a resolution band is `hypot(x, y)` from a **fabricated sensor at (0,0)**;
- the grid is not "adaptive w.r.t. the sensor", it is adaptive w.r.t. a fixed assumption;
- any real ego-motion or non-origin mount invalidates the entire map.

The locked architecture specifies a coordinate-handling stage; it does not exist.

### C6 — `models/metadata/model_metadata.json` publishes fabricated accuracy and a wrong parameter count
Verified against the actual checkpoint and the ONNX sidecar:

| Quantity | `model_metadata.json` | Measured truth |
|---|---|---|
| Overall accuracy | 95.7 % | **87.79 %** |
| mIoU | 88.4 % | **69.88 %** |
| Dynamic-obstacle IoU | 84.2 % | **46.85 %** |
| Parameter count | 807,299 | **265,571** |
| Input tensor | `input_points`, `float32[1,4096,4]` | `points`, `float32['batch','num_points',3]` |
| Producer | torch 2.14.0 | **pytorch 2.10.0** |

`models/onnx/pointnet2_semseg.json` and the checkpoint's own metrics agree exactly with the measured truth (87.79 / 69.88; IoUs 84.32 / 78.48 / 46.85), which proves `model_metadata.json` is stale and wrong, not merely conservative. It is the file a reviewer is most likely to read first. `config/model.yaml` repeats the 4-channel error.

### C7 — `python/conversion/export_onnx.py` exports randomly initialized weights by default
The checkpoint argument is optional. With no argument — or with a path that does not exist — the model is left at PyTorch's random initialization and exported anyway. The script's own numerical gate compares ORT against the *same* random PyTorch weights, so it passes. The failure is completely silent: a valid-looking `.onnx`, a passing verification report, and no trained weights.

### C8 — The Streamlit dashboard is a parallel reimplementation that never loads the ONNX model
`python/visualization/dashboard.py` does not consume C++ output; it instantiates `RealTimePipelineEngine` from `python/visualization/stream_engine.py`, a second, independent mapping implementation. In it:

- `stream_engine.py:213-219` — the ONNX session is **never created**, so `self.session` stays `None` and all segmentation is heuristic, while the UI labels the output `POINTNET++` / `ONNX`;
- `stream_engine.py:341-352` — track velocity is synthesized from `sin(frame_index * 0.15)`-style formulas, not estimated;
- track confidence is hardcoded to `0.92`; track IDs are not temporally stable; there is no Python Kalman filter at all;
- `stream_engine.py:425-426` — memory is estimated at an invented 48 bytes/cell, and the same "cell reduction" percentage is reused verbatim as "memory saved";
- "Live LiDAR" is one fixed scan displaced by a sine wave; camera projection uses invented monocular depth;
- boundary errors are hardcoded to zero (`dashboard.py:176`).

The architecture requires the dashboard to be a read-only consumer of the C++ pipeline's output. It is a second implementation that fabricates the numbers the first one already fabricates.

### C9 — `Quadtree::subdivideTo` duplicates parent state into all four children
`cpp/src/mapping/quadtree.cpp:20-30` copies the parent's `occupancy`, `elevation`, `min_z`, `max_z`, `clearance`, `semantic_class`, `confidence` and `object_id` into each of the four children while resetting `point_count` to 0. One 15 cm cell holding a 2 m obstacle therefore becomes four 7.5 cm cells that each independently report the full 2 m elevation and full occupancy. Obstacle height, clearance and free-space reasoning are all corrupted at every refinement, and the map's occupancy is multiplied by four at no cost.

### C10 — `Quadtree::refineCellAt` cannot descend; the configured refinement depth is unreachable
`cpp/src/mapping/quadtree.cpp:92-110` walks to the node containing the point but, on finding an internal node, reassigns `current = current` (the same node) instead of selecting the child, then subdivides and breaks. The function can therefore only ever subdivide the **root once**. In the grid the quadtree root *is* the base cell, so one level of refinement works by accident — but the configured `max_depth = 3` is unreachable, and in a standalone `Quadtree` (as the public API permits) refinement subdivides the wrong region entirely.

Measured consequence on the delivered frame: only **121 of 22,679 cells (0.53%)** carry a refined label (98 at 0.075 m, 23 at 0.025 m), each exactly one halving from its band base. The adaptive refinement engine — the core deliverable of the project — is effectively inert. Compounding this, `ResolutionPolicy::shouldRefine` and `computeImportance` (`cpp/src/resolution/resolution_policy.cpp:19-25`) are **never called in production**; the only live trigger is a hardcoded `if (pt.semantic_class == DYNAMIC_OBSTACLE) refineCellAt(...)` in `grid_25d.cpp`. The importance model the specification requires does not run.

---

## HIGH

### H1 — Tracker clusters by 1 m XY bin and ignores Z
`cpp/src/tracking/kalman_tracker.cpp` (~line 47) groups points by `floor(x/1.0), floor(y/1.0)`. One 4.5 m vehicle spanning 5+ cells becomes 5+ separate tracks, and a single cell containing both a car and a cyclist yields one track. There is no connected-component or Euclidean clustering and no Z consideration. Additionally, the `timestamp` argument passed to `update()` is discarded — prediction uses a hardcoded `dt = 0.1f` — and `TrackedObject::confidence` is never assigned, so every track reports whatever the constructor left (0).

### H2 — Pipeline stage structure deviates from the locked architecture
`cpp/src/runtime/pipeline.cpp:28+` implements preprocessing → inference → tracking → mapping. Missing or merged relative to the specification:
- no coordinate-transform stage (see C5);
- no importance-engine stage (`shouldRefine`/`computeImportance` are dead — see C10);
- object extraction is fused into `KalmanTracker` rather than a separate stage;
- resolution policy, refinement, projection and world-model update are all fused into `Grid25D`.

`Grid25D` has become a single ~200-line class performing resolution selection, insertion, refinement, tracking feedback, temporal decay, statistics and PLY export.

### H3 — All YAML configuration is dead code
`yaml-cpp` is linked in `CMakeLists.txt` but the header is never included and no `.cpp` references it. Every runtime value is hardcoded. Documented-vs-actual drift:

| Setting | Configured | Actual |
|---|---|---|
| `resolution.yaml` max subdivision depth | 2 | 3 |
| `tracking.yaml` association distance | 2.5 m | 3.0 m |
| `tracking.yaml` max missed frames | 5 | 3 |
| `tracking.yaml` confirmation hits | 3 | 2 |
| minimum cluster size | 15 | 5 |
| `lidar.yaml` intensity filter | enabled | not implemented |
| `lidar.yaml` noise/outlier filter | enabled | not implemented |
| semantic-boundary refinement | enabled | not implemented |
| `lidar.yaml` ROI | ±50 m | grid hardcoded ±60 m |

Changing any YAML file has no effect whatsoever on program behaviour.

### H4 — Live-server API omits mandatory per-point and per-cell fields
`cpp/apps/live_dashboard_server.cpp` (`/api/lidar/scan`) emits coordinates, class and resolution but **no confidence, no timestamp, no object ID and no velocity**. `cpp/web/index.html:1102+` consumes this payload, so the mandatory schema in the architecture is unachievable from the current API. Track Z is hardcoded to `-0.8` for every object, and object dimensions are inherited from ≤1 m clustering cells, so all tracked objects are reported as approximately one grid cell in size.

### H5 — Data race on the camera-frame route
The camera upload handler reads and writes the shared timing and metrics structures (including `last_process_time` and the frame counters) outside the mutex that the LiDAR route uses. The server dispatches request work onto detached threads, so this is a genuine unsynchronized concurrent access to shared state — undefined behaviour, not a theoretical concern.

### H6 — `sendResponse` performs a single unchecked `send()`
Response bodies for `/api/lidar/scan` are large (tens of KB). The helper issues one `send()` and never checks the return value. Winsock `send` is not guaranteed to write the full buffer; a short write silently truncates the JSON, which the client then fails to parse, with no error surfaced to either side.

### H7 — Labelling path can evaluate a model against its own heuristic output
`python/training/dataset.py` falls back to heuristic labels when ground-truth labels are unavailable, and `evaluate.py` then scores the model against those same heuristic labels. This is circular validation that will report high accuracy for a model that has learned nothing. Separately, `LidarIO::loadLabels` in `cpp/src/io/lidar_io.cpp` carries its own duplicated label remap that is not consistent with the table in `dataset.py`, so C++ and Python label the same raw label id differently.

### H8 — `updateTrackedObjects` overwrites network semantics
`Grid25D::updateTrackedObjects` force-relabels **every** cell whose centre falls inside a track box to `DYNAMIC_OBSTACLE`. A parked car, a wall segment or a cyclist inside a vehicle's tracking box has its PointNet++ prediction discarded and replaced. The specification's semantic output is overwritten by tracker geometry.

### H9 — `decayTemporal` never clears geometry, so departed obstacles persist
`Grid25D::decayTemporal` reduces `occupancy` but leaves `point_count`, `elevation`, `min_z`/`max_z`, `clearance` and `semantic_class` untouched. A cell whose last observation was a 2 m box obstacle keeps reporting 2 m of elevation indefinitely. Since `exportToPLY` writes every active leaf, ghost obstacles of arbitrary height are written to the map forever — and these stale cells are a direct contributor to the overlap and gap figures in C1.

### H10 — `requirements.txt` cannot support the documented workflow
`onnx` is absent, yet `python/conversion/export_onnx.py` and `tests/python/test_onnx.py` both `import onnx`. A clean `pip install -r requirements.txt` therefore leaves the export path and its test broken with `ModuleNotFoundError`. `torch>=2.0.0` is unpinned while the shipped artifact was produced by torch 2.10.0 and the audit environment has 2.14.0, so the export is not reproducible.

### H11 — The test suite validates the wrong properties and misses every real failure mode
- `test_temporal_fusion.cpp` (~line 34) asserts only that `occupancy` decreases. It never checks that elevation, `point_count` or semantics are cleared, so it passes cleanly while H9 is fully present.
- `test_tracking.cpp` places all 10 points inside a 0.4 m × 0.4 m patch. A cluster that small can never straddle a 1 m bin boundary, so the test is structurally incapable of exposing H1. It also passes `t = frame*0.1`, which coincidentally equals the hardcoded `dt` and so masks the timestamp defect.
- `test_quadtree.cpp:12` constructs `Quadtree(bounds{0,1,0,1}, base_resolution 0.50, 3)` — a 1.0 m node labelled 0.50 m. `subdivideTo` then derives midpoints from `resolution`, not `bounds`, producing children of 0.25/0.75/0.75 m all labelled `0.25f`. The test's area assertion (`total_area == 1.0`) still passes, so the repo's own test exercises the latent bug without detecting it.
- `test_resolution.cpp` tests `shouldRefine`, which is dead in production (C10), and probes 9.9/10.1, 25.0, 35.0/59.9, 65.0/90.0 — it never tests the exact boundaries 10.0, 30.0, 60.0 or 100.0, and never tests `>100 m` at all. The specification's highest-risk transition points are exactly the untested ones.
- `test_projection.cpp` is named for projection/boundary QA but inserts two points into one cell and never tests a resolution transition, so nothing in the suite checks cross-band gaps or overlaps — the defect in C1.
- `build/Testing/Temporary/LastTestsFailed.log` records `5:test_temporal_fusion`, while `LastTest.log` (same directory) shows all five tests passing. The failed-test log is therefore either stale or the suite is intermittently failing; either way the suite's reliability is unestablished.

### H12 — `scripts/run_dashboard.bat` hardcodes a machine-specific interpreter
It invokes `D:\python\python.exe` unconditionally. On any other machine the launcher fails outright. It also depends on being run from the repository root for the relative LiDAR path to resolve (see M4).

---

## MEDIUM

### M1 — Five different colour palettes for three semantic classes
| Source | Terrain | Static obstacle | Dynamic obstacle |
|---|---|---|---|
| Architecture spec (locked) | emerald `#10B981` | cyan `#22D3EE` | amber `#F59E0B` |
| `cpp/web/index.html` | emerald | cyan | amber — **correct** |
| `config/system.yaml` | gray | red | green |
| `Grid25D::exportToPLY` | slate `90,90,90` | `230,50,50` | `40,220,50` |
| `LidarIO::writePLY` | gray | red | green |
| `python/visualization/plot_builder.py:12-16` | slate | red | **teal** |

The delivered `adaptive_map_frame000000.ply` uses the slate/red/green scheme (confirmed: 15,676 vertices at `90 90 90`, 7,001 at `230 50 50`, 2 at `40 220 50`), so the shipped artifact does not use the locked palette.

### M2 — Python uses teal for two different meanings
`plot_builder.py` assigns teal to dynamic obstacles and also uses teal for 5 cm cells, so the colour encodes both class and resolution with no way to disambiguate.

### M3 — HTML renders `UNKNOWN` (255) as emerald terrain
Unknown/unclassified points are painted in the terrain colour, so total inference failure is visually indistinguishable from correct terrain classification.

### M4 — Streamlit BEV view is asymmetric
The Python BEV plot clips X to `[-5, 55]` while the ROI is `[-50, 50]`. Half the mapped region is off-screen and the view is not centred on the sensor.

### M5 — Relative data path in `stream_engine.py`
`stream_engine.py:203` loads `"data/raw/000000.bin"` relative to the process CWD, while the same constructor resolves `cpp_bin` via an absolute `REPO_ROOT` join. The dashboard therefore works only when launched from the repository root — which `run_dashboard.bat` arranges by `cd /d "%~dp0\.."` and nothing else guarantees.

### M6 — Class naming is inconsistent across layers
`dynamic_object` and `dynamic_obstacle` both appear for the same class across the Python code, the C++ enum string table and the configuration. Any consumer matching on the string name will mis-parse one of them.

### M7 — Exceptions are used despite the frozen "no exceptions" decision
`onnx_engine.cpp:18-38` and `:41-55` use `try`/`catch`, and the C++ standard library containers and allocations in the per-point, per-cell hot paths (`updateWithPointCloud`, `insertPoint`, the per-frame `PointCloud` copy) can throw. The specification froze the no-exceptions decision; the code does not honour it, and the hot paths are unprotected.

### M8 — `tile_size` and `getTileIndices` are dead
`GridConfig::tile_size` (1.0 m) and `Grid25D::getTileIndices` are never used in the update path. Each "tile" in `tiles_` is in fact a single base-resolution cell. `test_projection.cpp:13` sets `cfg.tile_size` believing it configures tiling; it does nothing.

### M9 — `GridConfig` bounds (±60 m) contradict the configured ROI (±50 m)
The 50–60 m annulus is silently unreachable, and combined with C5 the distance bands are computed from a hardcoded origin rather than the sensor.

### M10 — Uniform baseline is additionally inflated by ignoring the Z crop
Counting all 122,626 raw points yields 51,697 distinct XY cells; the identical cloud that the adaptive path actually receives (ROI + Z + voxel) has 35,767. So the uniform baseline is overstated on two axes at once (see C2).

### M11 — Unused Python dependencies; the only config reader is unused
`scipy`, `tqdm` and `pyyaml` are declared in `requirements.txt` and imported nowhere in `python/`, `tests/python/` or `scripts/`. `pyyaml` is notable: it is the sole configuration reader installed, and nothing reads the YAML files with it.

### M12 — `assert`-based tests silently become no-ops in a Release build
All C++ tests rely on `assert`. `CMakeLists.txt` sets no `CMAKE_BUILD_TYPE`, so on a single-config generator asserts happen to be live in the default build — but any `-DCMAKE_BUILD_TYPE=Release` (which adds `-DNDEBUG`) turns the entire suite into programs that always pass while testing nothing.

---

## LOW

### L1 — Exact float equality on resolution values
`test_resolution.cpp:12-25` compares `getBaseResolution(...) == 0.05f` etc. Brittle; should use a tolerance.

### L2 — `test_tracking` assertion is far weaker than its own comment
The comment states velocity should be close to 2.0 m/s; the assertion only requires `> 0.5f`. The recorded run shows `vx=0.78189` — roughly 2.6× too slow — and the test passes.

### L3 — `test_projection` sets a parameter that has no effect
See M8; the test's intent does not match the code's behaviour.

### L4 — Relative ONNX cache path
`onnx_engine.cpp` resolves the model as `"../models/onnx/pointnet2_semseg.onnx"`, relative to the CWD rather than to the executable or the repo root.

### L5 — Looser/contradictory configuration keys
`grid_size: 200.0` with `voxel_size: 0.10` implies a ±10 m grid, while `x_min`/`x_max` declare ±60 m in the same file family. Nothing consumes either value, but the config is not self-consistent.

### L6 — Hot-path allocations
`updateWithPointCloud` takes the point cloud by value and the per-point loops push into vectors without `reserve()`, causing repeated reallocation in the per-frame path.

### L7 — Missing deliverables from the specified project structure
No `scripts/preprocess.py`, `scripts/train.py`, `scripts/export_onnx.py`, `scripts/run_mapper.sh` or `scripts/benchmark.sh`; no `Dockerfile` and no `requirements-dev.txt`. `scripts/` contains only the dashboard launcher and a byte-identical duplicate of `python/inference_demo.py` (both MD5 `392DEAA49B15D9568CAEB44EBD2FC0C5`).

### L8 — HTML `GAPS` pill is inert markup
`index.html:679-682` has no element `id`, so no script can update it. It always displays `0.00 mm`.

---

## Verified Correct (no action required)

1. **C++ ONNX I/O binding matches the real artifact exactly.** Input `points`, `float32`, `['batch','num_points',3]`; output `logits`, `float32`, `['batch','num_points',3]`. Opset 17, IR 8, producer `pytorch 2.10.0`. Dynamic point counts 1,024 / 4,096 / 8,192 / 12,000 all execute successfully. `onnx_engine.cpp` requests three channels and reads the tensor by name correctly — the 4-channel claim in `config/model.yaml` and `model_metadata.json` is the error, not the C++.
2. **PyTorch/ONNX numerical equivalence is genuine.** `models/checkpoints/pointnet2_trained_best.pth` loads into `PointNet2SemSeg(num_classes=3, in_channels=3)` with no missing and no unexpected keys. Maximum absolute PyTorch-vs-ORT difference is ~1.5e-5, with 100% argmax agreement at N=4,096 and N=8,192. The export is faithful; only the metadata and the default-checkpoint behaviour (C7) are wrong.
3. **Checkpoint metrics are trustworthy.** `models/onnx/pointnet2_semseg.json` matches the checkpoint exactly: 87.79% accuracy, 69.88% mIoU, per-class IoU 84.32 / 78.48 / 46.85. Only `model_metadata.json` dissents (C6).
4. **Intra-band map alignment is exact.** Within each resolution the delivered PLY has zero duplicate footprints and center spacing equal to the labeled resolution (0.05, 0.075, 0.15, 0.30 m all check out). The boundary defect in C1 is exclusively cross-band, which narrows the fix considerably.
5. **The HTML dashboard's data flow is sound.** `index.html:1102+` fetches real `/api/lidar/scan` data, and fetch/HTTP failures are surfaced visibly to the user rather than swallowed. Its colour palette matches the locked architecture. The defects in H4 are backend omissions, not frontend handling errors.
6. **`export_onnx.py`'s verification logic is well constructed.** Dynamic-shape verification across multiple point counts and a PyTorch-vs-ORT numerical-equivalence gate are both appropriate. The defect is solely that the weights fed into them may be random (C7).

---

## Needs Verification (not confirmed)

1. **SemanticKITTI label remap.** The id→name tables in `python/training/dataset.py` and the duplicated table in `cpp/src/io/lidar_io.cpp` cannot be reconciled with each other or with the official SemanticKITTI label set. No `.label` files and no `semantic-kitti.yaml` exist anywhere in the repository, so the remap cannot be validated against data. The duplication and inconsistency are confirmed (H7); which table is correct is not established.
2. **Cross-frame stability of the overlap figure.** The 26.58% overlap and 2.46% gap measurements in C1 come from the single available frame (`000000`). Whether the magnitude is stable, growing or shrinking over a sequence cannot be determined without additional frames.
3. **Latent `subdivideTo` misalignment.** `QuadtreeNode` derives subdivision midpoints from `resolution` rather than from `bounds`, so a node whose `bounds` disagree with its `resolution` subdivides into unequal children that all receive the same label. Production is safe only because `Grid25D` happens to construct square cells where `bounds == resolution`; the public API does not enforce this and `test_quadtree.cpp` exercises the inconsistent case. Latent, not currently triggered in the delivered artifact.
4. **Camera upload path and the advertised 10 Hz live rate.** Neither could be exercised: no video source is available, and the "live" scan is a single precomputed frame. The claimed real-time behaviour is unverified in either direction.
5. **Intermittent test failure.** `LastTestsFailed.log` records `5:test_temporal_fusion` failing while `LastTest.log` shows all five passing. Whether this indicates genuine flakiness or a stale log from an earlier run is undetermined; the two files are not timestamped consistently enough to decide.

---

## What Was NOT Audited

- **Any C++ rebuild or execution of the shipped binaries.** The project was not recompiled; `build/` artifacts were read as evidence only. The `sizeof` probe compiled `grid_25d.cpp`, `quadtree.cpp` and `resolution_policy.cpp` from source into a temporary binary outside the repository, which confirms those three translation units build cleanly, but nothing else was compiled and no application binary was run.
- **`lidar_mapper` was never executed**, because it writes `results/maps/adaptive_map_frame000000.ply` and would have modified a repository file. All mapping conclusions come from static reading plus an independent NumPy reimplementation of the documented algorithm, cross-validated against the delivered PLY (predicted 22,673 cells vs 22,679 actual — 0.03% agreement).
- **`benchmark.cpp` was never executed**, for the same reason plus the risk of it emitting report artifacts. Its reported numbers were reverse-engineered statically and reproduced independently.
- **The live dashboard server was never started.** H5 (the data race) is derived from code reading of the thread-dispatch and locking structure, not from a ThreadSanitizer run or a stress test.
- **No hardware, no live sensor, no multi-frame sequence.** Only `data/raw/000000.bin` exists. Ego-motion, cross-frame consistency, tracking stability over time, and odometry/pose behaviour are entirely unassessed — and note that no pose input exists in the codebase at all (C5), so pose-dependent behaviour could not be tested even in principle.
- **Model accuracy was not re-measured.** The 87.79% / 69.88% figures were confirmed to be *self-consistent* between the checkpoint and the ONNX sidecar, and the published 95.7% / 88.4% figures were confirmed to be *inconsistent with them*. No independent evaluation against SemanticKITTI ground truth was performed, so which figure is correct in absolute terms rests on the internal consistency argument, not on a fresh measurement.
- **Training was not reproduced.** `train.py` was read but not executed; the reported training results and the Kaggle notebook were not validated.
- **Third-party ONNX Runtime was taken as given.** The vendored 1.26.0 binaries under `third_party/onnxruntime/` were used as-is and not audited.
- **The `kaggle_deploy/` notebook, `images/`, and `docs/experiments/hard_cases_benchmark_report.md` were not audited in depth** beyond confirming file inventory and hash integrity.
- **`python/evaluation/evaluate_hard_cases.py` and `project_traffic_jam_photorealistic.py` were reviewed only superficially**; their numerical claims were not reproduced.
- **Windows-only build assumptions were not stress-tested.** The project depends on Win32/Winsock and a MinGW toolchain; no attempt was made to assess portability to the Linux target implied by several specification requirements.

---

## Audit Method

| Technique | Coverage |
|---|---|
| Static source review | All 13 C++ sources/headers, 3 apps, 2 web clients, 11 Python modules, 9 tests, 5 YAML files, `CMakeLists.txt`, `requirements.txt`, 2 scripts |
| Numerical simulation | Full reimplementation of the documented pipeline in NumPy, run against the real 122,626-point scan |
| Binary artifact analysis | ONNX graph, input/output signature, dynamic shapes, initializer/parameter counts |
| Model equivalence | PyTorch checkpoint vs ONNX Runtime, multiple point counts, max-abs-diff and argmax agreement |
| Memory forensics | `sizeof` measured by compiling repo sources with the project's own toolchain into a temp binary |
| Geometric QA | Delivered PLY rasterized onto a 5 cm lattice; overlap multiplicity and interior-void detection |
| Integrity control | Full-tree MD5+size snapshot before and after; byte-for-byte verified identical |
