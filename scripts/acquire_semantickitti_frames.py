#!/usr/bin/env python3
"""
scripts/acquire_semantickitti_frames.py
---------------------------------------
Acquire consecutive SemanticKITTI sequence-00 Velodyne scans for multi-frame
testing. Strategy:
  1. Check whether real scans already exist at data/raw/sequences/00/velodyne/
  2. If fewer than 20 real frames exist, print real-data download instructions
     AND generate 20 clearly-labeled SYNTHETIC_*.bin frames for smoke-testing.

Synthetic frames are:
  - Named SYNTHETIC_<nnnnnn>.bin (never mistaken for real data)
  - Accompanied by SYNTHETIC_MANIFEST.json explaining their purpose
  - Valid only for structural pipeline smoke-tests, NOT benchmarks
"""

import struct, os, math, json, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR   = REPO_ROOT / "data" / "raw" / "sequences" / "00" / "velodyne"
REAL_FRAME_TARGET = 20

def count_real_frames(d):
    if not d.exists(): return []
    return sorted(p for p in d.glob("*.bin") if not p.stem.startswith("SYNTHETIC_"))

def generate_synthetic_frame(out_path, frame_idx, n_rings=64, n_pts_per_ring=16):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = []
    t = frame_idx * 0.1
    for ring in range(n_rings):
        el = math.radians(-15.0 + ring * (30.0 / n_rings))
        r  = 5.0 + ring * 0.3
        for i in range(n_pts_per_ring):
            az = math.radians((i / n_pts_per_ring) * 360.0)
            x = r * math.cos(az); y = r * math.sin(az); z = r * math.tan(el); intensity = 0.5
            # Simulate a vehicle moving forward @ 20 m/s
            if ring in (30, 31, 32) and i in (2, 3):
                x = 8.0 + t * 2.0; y = 2.5; z = 0.8; intensity = 0.9
            pts.append((x, y, z, intensity))
    with open(out_path, "wb") as f:
        for p in pts: f.write(struct.pack("<ffff", *p))
    return len(pts)

def main():
    print("=" * 70)
    print("SemanticKITTI Frame Acquisition -- PS26053 Step 2")
    print("=" * 70)

    real = count_real_frames(SEQ_DIR)
    print(f"\n[Real frames] Found {len(real)} at {SEQ_DIR}")

    if len(real) >= REAL_FRAME_TARGET:
        print(f"[OK] {len(real)} real frames -- no synthetic generation needed.")
        return

    print("\n[WARNING] Fewer than 20 real frames found. Login required for download.")
    print("  To obtain real frames:")
    print("  1. Register at https://www.semantic-kitti.org/dataset.html")
    print("  2. Download velodyne_odometry.zip for sequence 00 (~3.6 GB)")
    print("  3. Extract 000000.bin...000049.bin to data/raw/sequences/00/velodyne/")
    print("  4. Re-run this script\n")

    print("[Synthetic] Generating 20 synthetic frames for pipeline smoke-test...")
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the authentic seed frame into the sequence directory
    seed = REPO_ROOT / "data" / "raw" / "000000.bin"
    if seed.exists() and not (SEQ_DIR / "000000.bin").exists():
        shutil.copy2(seed, SEQ_DIR / "000000.bin")
        print(f"  Copied real seed: 000000.bin ({seed.stat().st_size:,} bytes)")

    syn_paths = []
    for idx in range(20):
        fname = SEQ_DIR / f"SYNTHETIC_{idx:06d}.bin"
        n = generate_synthetic_frame(fname, idx)
        syn_paths.append(fname)
        print(f"  Wrote {fname.name}  ({n} points)")

    manifest = {
        "WARNING": "SYNTHETIC frames -- pipeline smoke-test ONLY, not real data",
        "purpose": "Validate multi-frame ingestion loop, temporal fusion, PLY export",
        "real_data_source": "https://www.semantic-kitti.org/dataset.html (seq 00)",
        "synthetic_frames": [p.name for p in syn_paths],
    }
    with open(SEQ_DIR / "SYNTHETIC_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote SYNTHETIC_MANIFEST.json")

    print(f"\n[Summary] {SEQ_DIR}")
    print(f"  Real   : {1 if seed.exists() else 0} frame(s) (000000.bin - authentic KITTI)")
    print(f"  Synth  : {len(syn_paths)} frames (SYNTHETIC_*.bin - labeled smoke-test)")
    print(f"\n  Start server: .\\build\\bin\\live_dashboard_server.exe data\\raw\\sequences\\00\\velodyne")

if __name__ == "__main__":
    main()
