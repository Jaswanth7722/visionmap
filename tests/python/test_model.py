"""
Unit Tests for PointNet++ Semantic Segmentation Architecture
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.training.model import (
    PointNet2SemSeg,
    square_distance,
    farthest_point_sample,
    query_ball_point,
    gather_points,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)


def test_square_distance():
    src = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    dst = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    dist = square_distance(src, dst)
    assert dist.shape == (1, 2, 2)
    assert torch.isclose(dist[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(dist[0, 0, 1], torch.tensor(1.0))
    assert torch.isclose(dist[0, 1, 0], torch.tensor(1.0))
    assert torch.isclose(dist[0, 1, 1], torch.tensor(2.0))


def test_farthest_point_sample():
    B, N = 2, 100
    npoint = 16
    xyz = torch.randn(B, N, 3)
    fps_idx = farthest_point_sample(xyz, npoint)
    assert fps_idx.shape == (B, npoint)
    assert fps_idx.dtype == torch.int64
    assert fps_idx.min() >= 0
    assert fps_idx.max() < N


def test_query_ball_point():
    B, N, S, K = 2, 100, 16, 8
    xyz = torch.randn(B, N, 3)
    new_xyz = xyz[:, :S, :]
    group_idx = query_ball_point(radius=0.5, nsample=K, xyz=xyz, new_xyz=new_xyz)
    assert group_idx.shape == (B, S, K)
    assert group_idx.min() >= 0
    assert group_idx.max() < N


def test_gather_points():
    B, N, C = 2, 100, 16
    S, K = 8, 4
    points = torch.randn(B, N, C)
    
    # 2D index test
    idx2 = torch.randint(0, N, (B, S))
    g2 = gather_points(points, idx2)
    assert g2.shape == (B, S, C)
    assert torch.allclose(g2[0, 0], points[0, idx2[0, 0]])

    # 3D index test
    idx3 = torch.randint(0, N, (B, S, K))
    g3 = gather_points(points, idx3)
    assert g3.shape == (B, S, K, C)
    assert torch.allclose(g3[0, 1, 2], points[0, idx3[0, 1, 2]])


def test_pointnet_set_abstraction():
    B, N = 2, 256
    sa = PointNetSetAbstraction(npoint=32, radius=0.2, nsample=16, in_channel=3, mlp_channels=[16, 32])
    xyz = torch.randn(B, N, 3)
    new_xyz, new_points = sa(xyz, points=None)
    assert new_xyz.shape == (B, 32, 3)
    assert new_points.shape == (B, 32, 32)


def test_pointnet_feature_propagation():
    B, N, S = 2, 128, 32
    fp = PointNetFeaturePropagation(in_channel=32 + 16, mlp_channels=[32, 16])
    xyz1 = torch.randn(B, N, 3)
    xyz2 = torch.randn(B, S, 3)
    points1 = torch.randn(B, N, 16)
    points2 = torch.randn(B, S, 32)
    out = fp(xyz1, xyz2, points1, points2)
    assert out.shape == (B, N, 16)


def test_model_forward_nominal():
    model = PointNet2SemSeg(num_classes=3, in_channels=3).eval()
    x = torch.randn(1, 4096, 3)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4096, 3)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_model_forward_dynamic_counts():
    model = PointNet2SemSeg(num_classes=3, in_channels=3).eval()
    for N in [1024, 2048, 4096]:
        x = torch.randn(1, N, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, N, 3)


def test_model_backward_gradient_flow():
    model = PointNet2SemSeg(num_classes=3, in_channels=3).train()
    x = torch.randn(1, 1024, 3, requires_grad=True)
    target = torch.randint(0, 3, (1, 1024))
    
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, 3), target.view(-1))
    loss.backward()

    assert not torch.isnan(loss)
    # Check that classifier and backbone parameters have valid gradients
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad, "Model gradients failed to flow during backward pass!"
