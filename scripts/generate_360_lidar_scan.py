"""
PS26053 - Authentic 360-Degree Velodyne HDL-64E LiDAR Scan Generator
Generates a realistic 360-degree LiDAR point cloud scan in SemanticKITTI format:
  - 64 laser channels (elevation -24.8 deg to +2.0 deg)
  - 360-degree surround horizontal azimuth (-180 to +180 deg)
  - Full ground plane road surface (TERRAIN, class 0) at z = -1.60m
  - 5 realistic surrounding vehicles with contoured 3D chassis, hoods, and roofs (DYNAMIC_OBSTACLE, class 2)
  - Ground properly shadowed beneath vehicles (no false ground slicers)
  - Roadside curbs, guardrails, poles, and trees (STATIC_OBSTACLE, class 1)
  - Total ~120,000 points saved to data/raw/000000.bin (float32 [x, y, z, intensity])
"""

import os
import numpy as np

def generate_360_scan(output_path="data/raw/000000.bin"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    points = []
    
    sensor_height = 1.60  # meters above ground
    
    # 5 authentic surrounding vehicles in 360 surround
    vehicles_meta = [
        {"name": "Lead Car Ahead",        "cx": 13.0,  "cy":  0.0, "l": 4.5, "w": 1.9, "h": 1.50, "pts": 1400},
        {"name": "Left Forward Car",      "cx": 17.5,  "cy":  3.5, "l": 4.5, "w": 1.9, "h": 1.55, "pts": 1300},
        {"name": "Right Passing Car",     "cx":  6.0,  "cy": -3.5, "l": 4.3, "w": 1.85, "h": 1.45, "pts": 1300},
        {"name": "Rear Follower Car",     "cx": -14.0, "cy":  0.0, "l": 4.5, "w": 1.9, "h": 1.50, "pts": 1300},
        {"name": "Rear Left Follower",    "cx": -19.5, "cy":  3.5, "l": 4.6, "w": 1.9, "h": 1.55, "pts": 1200},
    ]

    # Velodyne HDL-64E: 64 vertical beams from -24.8 deg to +2.0 deg
    elev_angles = np.linspace(np.radians(-24.8), np.radians(2.0), 64)
    num_azimuth = 1800  # 0.2 deg angular resolution
    azimuths = np.linspace(-np.pi, np.pi, num_azimuth, endpoint=False)
    
    # 1. 360-degree Ground Plane (Road Surface at z = -1.60m)
    for elev in elev_angles:
        if elev < -0.018:  # Beams pointing downward hit ground
            r_ground = sensor_height / (-np.sin(elev))
            if r_ground > 70.0:
                continue
            
            for az in azimuths:
                r = r_ground + np.random.normal(0, 0.015)
                x = r * np.cos(elev) * np.cos(az)
                y = r * np.cos(elev) * np.sin(az)
                z = -sensor_height + np.random.normal(0, 0.015)
                
                # Ego vehicle footprint (clear)
                if abs(x) < 2.6 and abs(y) < 1.3:
                    continue
                
                # Check if ground is shadowed by any surrounding vehicle
                shadowed = False
                for v in vehicles_meta:
                    if abs(x - v["cx"]) <= (v["l"] / 2.0 + 0.15) and abs(y - v["cy"]) <= (v["w"] / 2.0 + 0.15):
                        shadowed = True
                        break
                if shadowed:
                    continue
                
                # Road intensity: asphalt ~0.24, lane markings ~0.80
                intensity = 0.22 + np.random.uniform(0, 0.06)
                if abs(y) < 0.12 or (abs(y - 3.5) < 0.12 and int(abs(x)) % 6 < 3) or (abs(y + 3.5) < 0.12 and int(abs(x)) % 6 < 3):
                    intensity = 0.82  # Road lane stripes
                
                points.append([x, y, z, intensity])
    
    # Helper to create contoured 3D vehicle returns (hood, cabin, roof, sides, bumpers)
    def add_realistic_vehicle(cx, cy, length, width, height, num_pts):
        z_base = -sensor_height + 0.35  # wheels/chassis base at -1.25m
        z_hood = -sensor_height + height * 0.55  # hood at -0.75m
        z_roof = -sensor_height + height         # roof at -0.10m
        
        # 30% roof points
        n_roof = int(num_pts * 0.30)
        rx = np.random.uniform(cx - length * 0.25, cx + length * 0.25, n_roof)
        ry = np.random.uniform(cy - width * 0.42, cy + width * 0.42, n_roof)
        rz = np.random.normal(z_roof, 0.03, n_roof)
        
        # 25% hood & trunk points
        n_hood = int(num_pts * 0.25)
        hx = np.concatenate([
            np.random.uniform(cx + length * 0.25, cx + length * 0.48, n_hood // 2),
            np.random.uniform(cx - length * 0.48, cx - length * 0.25, n_hood - n_hood // 2)
        ])
        hy = np.random.uniform(cy - width * 0.45, cy + width * 0.45, n_hood)
        hz = np.random.normal(z_hood, 0.04, n_hood)
        
        # 25% lateral door surfaces
        n_sides = int(num_pts * 0.25)
        sx = np.random.uniform(cx - length * 0.48, cx + length * 0.48, n_sides)
        side_sign = np.random.choice([-1.0, 1.0], size=n_sides)
        sy = cy + side_sign * (width * 0.48 + np.random.normal(0, 0.02, n_sides))
        sz = np.random.uniform(z_base, z_hood, n_sides)
        
        # 20% front & rear bumper faces
        n_ends = num_pts - (n_roof + n_hood + n_sides)
        end_sign = np.random.choice([-1.0, 1.0], size=n_ends)
        ex = cx + end_sign * (length * 0.48 + np.random.normal(0, 0.02, n_ends))
        ey = np.random.uniform(cy - width * 0.45, cy + width * 0.45, n_ends)
        ez = np.random.uniform(z_base, z_hood, n_ends)
        
        all_vx = np.concatenate([rx, hx, sx, ex])
        all_vy = np.concatenate([ry, hy, sy, ey])
        all_vz = np.concatenate([rz, hz, sz, ez])
        all_vi = np.random.uniform(0.68, 0.92, len(all_vx))
        
        for i in range(len(all_vx)):
            points.append([float(all_vx[i]), float(all_vy[i]), float(all_vz[i]), float(all_vi[i])])

    # 2. Add the 5 authentic surround vehicles
    for v in vehicles_meta:
        add_realistic_vehicle(v["cx"], v["cy"], v["l"], v["w"], v["h"], v["pts"])

    # 3. Add 360 roadside barriers, guardrails, curbs, and light poles (at |y| >= 5.2m)
    def add_roadside_structures(x_start=-50, x_end=65, step=4.0):
        for x in np.arange(x_start, x_end, step):
            # Left roadside barrier & trees (y = 5.2m to 12m)
            for _ in range(32):
                y = np.random.uniform(5.2, 11.5)
                z = np.random.uniform(-sensor_height + 0.35, 2.4)
                intens = np.random.uniform(0.35, 0.65)
                points.append([x + np.random.normal(0, 0.4), y, z, intens])
            # Right roadside barrier & trees (y = -11.5m to -5.2m)
            for _ in range(32):
                y = np.random.uniform(-11.5, -5.2)
                z = np.random.uniform(-sensor_height + 0.35, 2.4)
                intens = np.random.uniform(0.35, 0.65)
                points.append([x + np.random.normal(0, 0.4), y, z, intens])

    add_roadside_structures()

    pts_arr = np.array(points, dtype=np.float32)
    print(f"[360 LiDAR] Generated {len(pts_arr):,} points across 360 degrees.")
    print(f"  X (Forward/Back): [{pts_arr[:,0].min():.1f}m, {pts_arr[:,0].max():.1f}m]")
    print(f"  Y (Left/Right):   [{pts_arr[:,1].min():.1f}m, {pts_arr[:,1].max():.1f}m]")
    print(f"  Z (Elevation):    [{pts_arr[:,2].min():.1f}m, {pts_arr[:,2].max():.1f}m]")
    
    with open(output_path, "wb") as f:
        pts_arr.tofile(f)
    print(f"[360 LiDAR] Successfully saved to {output_path} ({os.path.getsize(output_path):,} bytes)")

if __name__ == "__main__":
    generate_360_scan()
