"""
PS26053 Plot Builder
Provides interactive Plotly visualizations for 3D point clouds, 2.5D BEV adaptive grids,
and performance benchmarking analytics.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List


CLASS_COLOR_MAP = {
    0: ("#10B981", "Terrain"),
    1: ("#22D3EE", "Static Obstacle"),
    2: ("#F59E0B", "Dynamic Object")
}


def build_3d_point_cloud_figure(points: np.ndarray, classes: np.ndarray, tracks: List[Dict]) -> go.Figure:
    """Builds an interactive 3D WebGL point cloud plot with semantic coloring and track markers."""
    fig = go.Figure()

    # Subsample for smooth 60fps browser rendering if points > 3000
    if len(points) > 3000:
        step = len(points) // 3000
        pts_vis = points[::step]
        cls_vis = classes[::step]
    else:
        pts_vis = points
        cls_vis = classes

    # Add points grouped by class for clean legend toggling
    for c_id, (color, name) in CLASS_COLOR_MAP.items():
        mask = (cls_vis == c_id)
        if np.any(mask):
            fig.add_trace(go.Scatter3d(
                x=pts_vis[mask, 0],
                y=pts_vis[mask, 1],
                z=pts_vis[mask, 2],
                mode='markers',
                marker=dict(
                    size=2.5,
                    color=color,
                    opacity=0.85
                ),
                name=f"{name} ({np.sum(mask)})",
                hoverinfo='text',
                hovertext=[f"Class: {name}<br>XYZ: ({x:.1f}, {y:.1f}, {z:.1f})m" for x, y, z in pts_vis[mask]]
            ))

    # Add Sensor Origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=6, color='#FFD166', symbol='diamond'),
        name="Ego Vehicle / Sensor",
        text=["EGO SENSOR"],
        textposition="top center"
    ))

    # Add Tracked Objects (Velocity vectors / cones)
    if tracks:
        tx = [t["x"] for t in tracks]
        ty = [t["y"] for t in tracks]
        tz = [t["z"] for t in tracks]
        speeds = [t["speed"] for t in tracks]
        t_text = [f"Track #{t['id']}<br>Speed: {t['speed']:.1f} m/s ({t['speed']*3.6:.0f} km/h)<br>Vel: ({t['vx']:.1f}, {t['vy']:.1f}) m/s" for t in tracks]

        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode='markers+text',
            marker=dict(size=8, color='#06D6A0', symbol='square'),
            name="Tracked Dynamic Objects",
            text=[f"#{t['id']} ({t['speed']:.1f}m/s)" for t in tracks],
            textposition="top center",
            hoverinfo='text',
            hovertext=t_text
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Forward X (m)', backgroundcolor="#111625", gridcolor="#2A324B"),
            yaxis=dict(title='Lateral Y (m)', backgroundcolor="#111625", gridcolor="#2A324B"),
            zaxis=dict(title='Height Z (m)', backgroundcolor="#111625", gridcolor="#2A324B"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor="#0E1117",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#E2E8F0", size=11)
        ),
        height=480
    )
    return fig


def build_25d_bev_figure(cells: List[Dict], tracks: List[Dict]) -> go.Figure:
    """Builds the 2.5D Bird's Eye View (BEV) world model showing variable resolution cells and distance bands."""
    fig = go.Figure()

    # Draw concentric PS distance band circles (0-10m, 10-30m, 30-60m)
    theta = np.linspace(0, 2*np.pi, 100)
    for radius, color, label in [(10.0, "rgba(78, 204, 163, 0.35)", "Band 1: 0–10m (5 cm)"),
                                 (30.0, "rgba(255, 217, 61, 0.30)", "Band 2: 10–30m (15 cm)"),
                                 (60.0, "rgba(255, 107, 107, 0.25)", "Band 3: 30–60m (30 cm)")]:
        fig.add_trace(go.Scatter(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            mode='lines',
            line=dict(color=color, width=1.5, dash='dot'),
            name=label,
            hoverinfo='name'
        ))

    if cells:
        # Group cells by resolution to visualize the variable cell sizes
        cells_by_res = {}
        for c in cells:
            r = c["res"]
            cells_by_res.setdefault(r, []).append(c)

        res_colors = {
            0.05: ("#2A9D8F", 7.0, "5 cm (Fine / Refined)"),
            0.15: ("#E9C46A", 5.0, "15 cm (Mid-Range)"),
            0.30: ("#F4A261", 4.0, "30 cm (Far-Range)"),
            0.50: ("#E76F51", 3.0, "50 cm (Horizon)")
        }

        for res, clist in cells_by_res.items():
            color, size, label = res_colors.get(res, ("#8A8A8A", 4.0, f"{res*100:.0f} cm"))
            cx = [c["cx"] for c in clist]
            cy = [c["cy"] for c in clist]
            hover = [f"Res: {c['res']*100:.1f}cm<br>Pos: ({c['cx']:.1f}, {c['cy']:.1f})m<br>Elev: {c['elevation']:.2f}m<br>Pts: {c['count']}" for c in clist]

            fig.add_trace(go.Scatter(
                x=cx, y=cy,
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    symbol='square',
                    opacity=0.8
                ),
                name=label,
                hoverinfo='text',
                hovertext=hover
            ))

    # Add Tracked Object Markers & Velocity Arrows in BEV
    if tracks:
        for trk in tracks:
            fig.add_trace(go.Scatter(
                x=[trk["x"]], y=[trk["y"]],
                mode='markers+text',
                marker=dict(size=11, color="#00F5D4", symbol="circle-open-dot", line=dict(width=2, color="#00F5D4")),
                text=[f"#{trk['id']} {trk['speed']:.1f}m/s"],
                textposition="top center",
                name=f"Track #{trk['id']}"
            ))

    fig.update_layout(
        xaxis=dict(title='Forward X (m)', range=[-55, 55], gridcolor="#2A324B"),
        yaxis=dict(title='Lateral Y (m)', range=[-55, 55], gridcolor="#2A324B"),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111625",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#E2E8F0", size=10)
        ),
        height=480
    )
    return fig


def build_latency_breakdown_figure(timing: Dict[str, float]) -> go.Figure:
    """Builds horizontal bar chart of stage-by-stage latencies."""
    stages = ["Preprocessing", "PointNet++ ONNX", "Tracking (assoc.)", "2.5D Mapping"]
    times = [
        timing.get("preprocess_ms", 0.0),
        timing.get("infer_ms", 0.0),
        timing.get("track_ms", 0.0),
        timing.get("map_ms", 0.0)
    ]
    colors = ["#4D96FF", "#6BCB77", "#FFD93D", "#FF6B6B"]

    fig = go.Figure(go.Bar(
        x=times,
        y=stages,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{t:.1f} ms" for t in times],
        textposition='auto',
    ))

    fig.update_layout(
        title=dict(text=f"Total Pipeline Latency: {timing.get('total_ms', 0):.1f} ms ({timing.get('fps', 0):.1f} FPS)",
                   font=dict(color="#E2E8F0", size=12)),
        xaxis=dict(title="Time (ms)", gridcolor="#2A324B"),
        yaxis=dict(gridcolor="#2A324B"),
        margin=dict(l=10, r=10, b=20, t=35),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111625",
        font=dict(color="#CBD5E1"),
        height=200
    )
    return fig


def build_uniform_vs_adaptive_bar(uniform_val: float, adaptive_val: float, title: str, unit: str) -> go.Figure:
    """Builds a direct comparison bar chart showing Uniform Baseline vs Adaptive Ours."""
    fig = go.Figure(data=[
        go.Bar(name='Uniform Baseline (5 cm)', x=['Mode'], y=[uniform_val], marker_color='#E76F51', text=[f"{uniform_val:,.0f} {unit}" if unit != "MB" else f"{uniform_val:.2f} MB"], textposition='auto'),
        go.Bar(name='Adaptive (PS26053 Ours)', x=['Mode'], y=[adaptive_val], marker_color='#2A9D8F', text=[f"{adaptive_val:,.0f} {unit}" if unit != "MB" else f"{adaptive_val:.2f} MB"], textposition='auto')
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E2E8F0", size=12)),
        barmode='group',
        margin=dict(l=10, r=10, b=20, t=35),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111625",
        font=dict(color="#CBD5E1"),
        height=200,
        showlegend=False
    )
    return fig
