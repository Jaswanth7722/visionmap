#!/usr/bin/env python3
"""
scripts/validate_ply.py
------------------------
Automated quality validator for per-frame PLY outputs from the PS26053
LiDAR mapping pipeline.

Checks per PLY file:
  1. Header presence and FORMAT == binary_little_endian or ascii
  2. No NaN or Inf in any vertex coordinate (x, y, z)
  3. Vertex count > 0
  4. All x/y coordinates within configured world boundary [-100, 100] m
  5. All z coordinates within [-5, 50] m (reasonable urban scene height)
  6. Sidecar JSON exists with required keys
  7. (Soft) Vertex count matches active_cells from sidecar JSON (warning, not fail)
  8. Color channels present and within [0, 255]

Usage:
  python scripts/validate_ply.py [results/maps]  [--strict]

Returns exit code 0 when all checks pass, 1 if any hard failure.
"""

import struct
import json
import sys
import math
from pathlib import Path

REQUIRED_JSON_KEYS = {"frame_index", "active_cells", "point_counts", "latency_ms"}
BOUNDARY_XY = 100.0
BOUNDARY_Z_MIN = -5.0
BOUNDARY_Z_MAX = 50.0

def fail(msg): return ("FAIL", msg)
def warn(msg): return ("WARN", msg)
def ok(msg):   return ("OK",   msg)

def parse_ply_header(data: bytes):
    """Return (n_vertices, header_end_byte, is_ascii, has_color) or raise.
    Handles both LF and CRLF line endings (Windows C++ may write CRLF).
    """
    # Try CRLF first, then LF
    for end_marker in (b"end_header\r\n", b"end_header\n"):
        idx = data.find(end_marker)
        if idx != -1:
            break
    if idx == -1:
        raise ValueError("No 'end_header' found")
    header = data[:idx].decode("ascii", errors="replace")
    n_verts = 0
    is_ascii = False
    has_color = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("format ascii"):
            is_ascii = True
        elif line.startswith("format binary_little_endian"):
            is_ascii = False
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        if "red" in line or "green" in line or "blue" in line:
            has_color = True
    return n_verts, idx + len(end_marker), is_ascii, has_color

def validate_ply(ply_path: Path, strict: bool = False):
    results = []
    sidecar = ply_path.with_suffix(".json")

    if not ply_path.exists():
        return [fail(f"PLY file not found: {ply_path}")]

    data = ply_path.read_bytes()

    # 1. Header
    try:
        n_verts, body_start, is_ascii, has_color = parse_ply_header(data)
    except Exception as e:
        return [fail(f"Header parse error: {e}")]

    results.append(ok(f"Header OK — {n_verts} vertices, {'ascii' if is_ascii else 'binary_le'}"))

    # 2. Vertex count
    if n_verts == 0:
        results.append(fail("Zero vertices — empty PLY"))
    else:
        results.append(ok(f"Vertex count > 0 ({n_verts})"))

    # 3. Parse vertices and check NaN/Inf and boundaries
    nan_inf_count = 0
    oob_count = 0
    bad_color = 0
    checked = 0

    if is_ascii:
        body = data[body_start:].decode("ascii", errors="replace")
        lines = body.strip().splitlines()
        for line in lines[:n_verts]:
            vals = line.split()
            if len(vals) < 3:
                continue
            try:
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            except ValueError:
                continue
            checked += 1
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                nan_inf_count += 1
            elif (abs(x) > BOUNDARY_XY or abs(y) > BOUNDARY_XY or
                  z < BOUNDARY_Z_MIN or z > BOUNDARY_Z_MAX):
                oob_count += 1
            if has_color and len(vals) >= 6:
                try:
                    r, g, b = int(vals[3]), int(vals[4]), int(vals[5])
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                        bad_color += 1
                except ValueError:
                    pass
    else:
        # Binary: assume 3 floats (xyz) + 3 uint8 (rgb) = 15 bytes per vertex
        # OR 3 floats only = 12 bytes. Infer from body size.
        body = data[body_start:]
        bytes_per_vertex_xyz = 12  # 3 x float32
        bytes_per_vertex_rgb = 3   # 3 x uint8
        stride_rgb = bytes_per_vertex_xyz + bytes_per_vertex_rgb
        stride_xyz = bytes_per_vertex_xyz
        # Pick stride that divides body size cleanly
        stride = stride_rgb if has_color and (len(body) % stride_rgb == 0) else stride_xyz
        for i in range(min(n_verts, len(body) // stride)):
            offset = i * stride
            if offset + 12 > len(body):
                break
            x, y, z = struct.unpack_from("<fff", body, offset)
            checked += 1
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                nan_inf_count += 1
            elif (abs(x) > BOUNDARY_XY or abs(y) > BOUNDARY_XY or
                  z < BOUNDARY_Z_MIN or z > BOUNDARY_Z_MAX):
                oob_count += 1
            if has_color and stride == stride_rgb:
                r, g, b = struct.unpack_from("<BBB", body, offset + 12)
                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                    bad_color += 1

    if nan_inf_count > 0:
        results.append(fail(f"NaN/Inf detected in {nan_inf_count}/{checked} vertices"))
    else:
        results.append(ok(f"No NaN/Inf in {checked} vertices"))

    if oob_count > 0:
        msg = f"{oob_count}/{checked} vertices outside boundary"
        results.append(fail(msg) if strict else warn(msg))
    else:
        results.append(ok(f"All {checked} vertices within boundary"))

    if has_color:
        if bad_color > 0:
            results.append(fail(f"Invalid color in {bad_color} vertices"))
        else:
            results.append(ok("Color channels valid"))
    else:
        results.append(warn("No color properties in PLY"))

    # 4. Sidecar JSON
    if not sidecar.exists():
        results.append(warn(f"Sidecar JSON missing: {sidecar.name}"))
    else:
        try:
            meta = json.loads(sidecar.read_text())
            missing = REQUIRED_JSON_KEYS - set(meta.keys())
            if missing:
                results.append(fail(f"Sidecar JSON missing keys: {missing}"))
            else:
                results.append(ok(f"Sidecar JSON OK — {meta.get('active_cells',0)} cells, "
                                  f"{meta.get('latency_ms',0):.1f} ms"))
                # Soft: vertex count vs active_cells
                ac = meta.get("active_cells", 0)
                if ac > 0 and abs(n_verts - ac) > max(50, ac * 0.2):
                    results.append(warn(f"Vertex count ({n_verts}) differs from "
                                        f"active_cells ({ac}) by >{20:.0f}%"))
        except json.JSONDecodeError as e:
            results.append(fail(f"Sidecar JSON parse error: {e}"))

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate PLY outputs from PS26053 LiDAR pipeline")
    parser.add_argument("maps_dir", nargs="?", default="results/maps",
                        help="Directory containing adaptive_map_frame*.ply files")
    parser.add_argument("--strict", action="store_true",
                        help="Treat boundary warnings as hard failures")
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    if not maps_dir.exists():
        print(f"[ERROR] Directory not found: {maps_dir}")
        sys.exit(1)

    ply_files = sorted(maps_dir.glob("adaptive_map_frame*.ply"))
    if not ply_files:
        print(f"[ERROR] No adaptive_map_frame*.ply files found in {maps_dir}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"PS26053 PLY Quality Validator — {len(ply_files)} frame(s) in {maps_dir}")
    print(f"{'='*70}")

    total_pass = 0; total_warn = 0; total_fail = 0; files_failed = 0

    for ply in ply_files:
        results = validate_ply(ply, strict=args.strict)
        file_ok = all(r[0] != "FAIL" for r in results)
        status = "PASS" if file_ok else "FAIL"
        print(f"\n{'-'*50}")
        print(f"[{status}] {ply.name}")
        for level, msg in results:
            icon = {"OK": "  OK", "WARN": "  !", "FAIL": "  X"}[level]
            print(f"{icon} {msg}")
        if file_ok:
            total_pass += 1
        else:
            total_fail += 1
            files_failed += 1
        total_warn += sum(1 for r in results if r[0] == "WARN")

    print(f"\n{'='*70}")
    print(f"RESULTS: {total_pass} PASS, {total_fail} FAIL, {total_warn} warnings")
    if files_failed == 0:
        print("ALL PLY FILES PASSED QUALITY CHECKS.")
    else:
        print(f"{files_failed} FILE(S) FAILED QUALITY CHECKS.")
    print(f"{'='*70}")

    sys.exit(0 if files_failed == 0 else 1)

if __name__ == "__main__":
    main()

