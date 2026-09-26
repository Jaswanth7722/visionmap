"""
PS26053 Live Interactive Dashboard
Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception
DRDO | Smart Vehicles | Software Autonomous Navigation Pipeline
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.visualization.stream_engine import RealTimePipelineEngine
from python.visualization.plot_builder import (
    build_3d_point_cloud_figure,
    build_25d_bev_figure,
    build_latency_breakdown_figure,
    build_uniform_vs_adaptive_bar
)

# Page configuration
st.set_page_config(
    page_title="PS26053 Live 2.5D LiDAR Mapping",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern defense-grade / dark autonomous navigation aesthetic
st.markdown("""
<style>
    .main { background-color: #0B0E14; }
    .stMetric {
        background-color: #141A29;
        border: 1px solid #2A364F;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .status-badge-pass {
        background-color: #064E3B;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-fail {
        background-color: #5B1A1A;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .header-box {
        background: linear-gradient(90deg, #101626 0%, #1A2238 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border-left: 5px solid #2A9D8F;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline_engine():
    model_path = os.path.join(str(REPO_ROOT), "models", "onnx", "pointnet2_semseg.onnx")
    return RealTimePipelineEngine(onnx_model_path=model_path)


try:
    engine = get_pipeline_engine()
except RuntimeError as exc:
    st.error(f"Dashboard cannot start honestly: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/lidar.png", width=64)
st.sidebar.title("Sensor & Stream Controls")

input_mode = st.sidebar.selectbox(
    "Input Perception Source:",
    ["Live Camera Feed (Webcam)", "Continuous Real-Time LiDAR Stream", "Upload Point Cloud (.bin / .ply)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Playback")
is_streaming = st.sidebar.toggle("Continuous Real-Time Streaming", value=True)
target_fps = st.sidebar.slider("Target Stream Rate (FPS):", min_value=1, max_value=30, value=10)
step_button = st.sidebar.button("Step Single Frame (Manual)", disabled=is_streaming)

st.sidebar.markdown("---")
st.sidebar.subheader("Spatial Adaptation Spec")
st.sidebar.markdown("""
**Level 1 Distance Bands:**
- **0–10 m:** 5 cm (Near Critical)
- **10–30 m:** 15 cm (Mid-Range)
- **30–60 m:** 30 cm (Far-Range)
- **60–100 m:** 50 cm (Horizon)

**Level 2 Dynamic Triggers:**
- Dynamic motion ($v > 0.25$ m/s) $\\rightarrow$ Quadtree refine to **5 cm**
- Elevation curb steps $\\rightarrow$ Refine local cell
""")
enable_level2 = st.sidebar.checkbox("Enable Level 2 Quadtree Refinement", value=True)

# File upload handler if selected
custom_cloud = None
if input_mode == "Upload Point Cloud (.bin / .ply)":
    uploaded_file = st.sidebar.file_uploader("Upload Point Cloud File", type=["bin", "ply"])
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        if uploaded_file.name.endswith(".bin"):
            custom_cloud = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 4)
            st.sidebar.success(f"Loaded {len(custom_cloud)} points from {uploaded_file.name}")

# -----------------------------------------------------------------------------
# Main Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h2 style="margin:0; color:#FFFFFF; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        Adaptive Variable-Resolution 2.5D LiDAR Mapping
    </h2>
    <p style="margin:5px 0 0 0; color:#94A3B8; font-size:14px;">
        <b>Problem Statement ID 26053</b> | DRDO Smart Vehicles | Native C++17/20 Runtime & Live Dynamic Perception Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Process Frame
# -----------------------------------------------------------------------------
source_str = "Live Camera" if "Camera" in input_mode else "Continuous LiDAR"
frame_data = engine.process_frame(
    source_mode=source_str,
    custom_cloud=custom_cloud,
    enable_refinement=enable_level2,
)

if frame_data.get("error"):
    st.error(f"Frame processing failed honestly (no fallback data shown): {frame_data['error']}")
    st.stop()

timing = frame_data["timing"]
tracks = frame_data["tracks"]
cells = frame_data["cells"]

# -----------------------------------------------------------------------------
# Top Metric Ribbon
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Active Stored Cells",
        value=f"{frame_data['adaptive_cell_count']:,}",
        delta=f"-{frame_data['cell_reduction_pct']:.1f}% vs Uniform (5cm)",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Memory Footprint (est.)",
        value=f"{frame_data['adaptive_mem_mb']:.2f} MB",
        delta=f"{frame_data['memory_saved_pct']:.1f}% vs uniform (est.)",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Pipeline Rate",
        value=f"{timing['fps']:.1f} FPS",
        delta=f"{timing['total_ms']:.1f} ms Latency"
    )

with col4:
    st.metric(
        label="Tracked Dynamic Objects",
        value=f"{len(tracks)}",
        delta="Centroid association (Python estimator)"
    )

with col5:
    st.markdown("**Boundary Alignment QA**")
    qa_errors = frame_data["boundary_alignment_errors"]
    qa_checked = frame_data["boundary_cells_checked"]
    if qa_errors == 0:
        st.markdown('<div class="status-badge-pass">PASS: 0 ERRORS</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge-fail">FAIL: {qa_errors} ERRORS</div>', unsafe_allow_html=True)
    st.caption(f"Computed over {qa_checked:,} stored cells (gaps/overlaps)")

# -----------------------------------------------------------------------------
# Visualizations Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Interactive 3D Point Cloud & Camera",
    "2.5D Adaptive World Model (BEV)",
    "Proof-of-Value Benchmark & Tracks"
])

with tab1:
    col_cam, col_3d = st.columns([1, 2])
    with col_cam:
        st.subheader("Sensor Input Stream")
        if frame_data["camera_frame"] is not None:
            cam_title = "Hardware Webcam (Live)" if frame_data["is_hardware_camera"] else "Live Driving Camera Feed"
            st.image(frame_data["camera_frame"], caption=cam_title, use_container_width=True)
            st.info(f"Frame #{frame_data['frame_index']} | Projected {len(frame_data['points'])} 3D metric coordinates "
                    f"(monocular depth estimate: geometry approximate, classes are model output)")
        else:
            st.image("https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=600",
                     caption="Continuous LiDAR Ingestion Active", use_container_width=True)
            st.info(f"Continuous LiDAR Mode: {frame_data['input_desc']}.")

    with col_3d:
        st.subheader("PointNet++ 3D Semantic Segmentation")
        st.caption(f"{frame_data['points_classified']:,} of {frame_data['points_total']:,} input points "
                   f"({frame_data['network_share_pct']:.1f}%) classified via {frame_data['inference_source']}")
        fig_3d = build_3d_point_cloud_figure(frame_data["points"], frame_data["classes"], tracks)
        st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Level 1 Distance Bands & Level 2 Quadtree Local Refinement (BEV)")
    col_bev, col_legend = st.columns([3, 1])
    with col_bev:
        fig_bev = build_25d_bev_figure(cells, tracks)
        st.plotly_chart(fig_bev, use_container_width=True)
    with col_legend:
        st.markdown("### Adaptive Legend")
        st.markdown("""
        - 🟢 **5 cm:** Dynamic object refined cells & near zone (0–10m)
        - 🟡 **15 cm:** Mid-range band (10–30m)
        - 🟠 **30 cm:** Far-range band (30–60m)
        - 🔴 **50 cm:** Horizon band (60–100m)
        """)
        st.markdown("---")
        st.markdown("### Refinement Summary")
        fine_count = sum(1 for c in cells if c["res"] <= 0.05)
        st.write(f"• Fine (5 cm) cells: **{fine_count}**")
        st.write(f"• Coarse (>15 cm) cells: **{len(cells) - fine_count}**")
        st.write("• Local Quadtree preserves detail where needed, saving memory elsewhere.")

with tab3:
    st.subheader("Direct Uniform vs Adaptive Proof-of-Value Verification")
    st.caption(f"Cell counts are exact on the identical {frame_data['points_classified']:,}-point input subset. "
               f"Memory figures are estimates ({frame_data['memory_basis']}).")
    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])

    with b_col1:
        fig_cells = build_uniform_vs_adaptive_bar(
            frame_data["uniform_cell_count"], frame_data["adaptive_cell_count"],
            "Active Stored Cells (Lower is Better)", "cells"
        )
        st.plotly_chart(fig_cells, use_container_width=True)

    with b_col2:
        fig_mem = build_uniform_vs_adaptive_bar(
            frame_data["uniform_mem_mb"], frame_data["adaptive_mem_mb"],
            "Memory Footprint (Lower is Better)", "MB"
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    with b_col3:
        fig_latency = build_latency_breakdown_figure(timing)
        st.plotly_chart(fig_latency, use_container_width=True)

    st.subheader("Dynamic Tracked Objects Table")
    if tracks:
        track_table = [{
            "Track ID": f"#{t['id']}",
            "Position X (m)": f"{t['x']:.2f}",
            "Position Y (m)": f"{t['y']:.2f}",
            "Speed (m/s)": f"{t['speed']:.2f}",
            "Speed (km/h)": f"{t['speed']*3.6:.1f}",
            "Classification": "Dynamic Object (centroid-tracked)",
            "Confidence": f"{t['confidence']*100:.0f}%"
        } for t in tracks]
        st.dataframe(track_table, use_container_width=True)
    else:
        st.info("No moving dynamic objects detected in current sensor view.")

# -----------------------------------------------------------------------------
# Streaming Loop Auto-Refresh
# -----------------------------------------------------------------------------
if is_streaming:
    time.sleep(1.0 / target_fps)
    st.rerun()
