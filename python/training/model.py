"""
Pure PyTorch PointNet++ Semantic Segmentation Model
Compliant with ONNX Opset 17 export and dynamic point-cloud sizing.
No custom CUDA kernels required.

Classes:
  0: terrain (road, sidewalk, terrain, parking)
  1: static_obstacle (building, fence, pole, wall, vegetation, traffic sign)
  2: dynamic_object (car, truck, person, bicyclist, motorcyclist, other-vehicle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean squared distance between points.
    src: [B, N, C]
    dst: [B, M, C]
    Returns:
        dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return torch.clamp(dist, min=0.0)


def farthest_point_sample(xyz, npoint: int):
    """
    Farthest Point Sampling implemented using pure PyTorch tensor operations.
    Fully traceable and exportable to ONNX with dynamic input points N.

    Args:
        xyz: [B, N, 3] coordinates
        npoint: int, number of centroids to sample
    Returns:
        centroids_idx: [B, npoint] indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)
    farthest = torch.zeros(B, dtype=torch.int64, device=device)
    centroids_list = []

    for _ in range(npoint):
        centroids_list.append(farthest.unsqueeze(1))
        centroid = torch.gather(xyz, 1, farthest.view(B, 1, 1).expand(B, 1, 3))
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return torch.cat(centroids_list, dim=1)


def query_ball_point(radius: float, nsample: int, xyz, new_xyz):
    """
    Ball query grouping implemented with PyTorch TopK and Where.
    Selects up to nsample points within radius; if fewer, duplicates nearest point.

    Args:
        radius: search radius float
        nsample: maximum number of neighbors
        xyz: all points, [B, N, 3]
        new_xyz: query centroids, [B, S, 3]
    Returns:
        group_idx: [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    dists, group_idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False)
    first_idx = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    group_idx = torch.where(dists > (radius ** 2), first_idx, group_idx)
    return group_idx


def gather_points(points, idx):
    """
    Gather points/features by index using torch.gather.
    Standard ONNX GatherElements friendly.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]
    Returns:
        gathered: [B, S, C] or [B, S, K, C]
    """
    B = points.shape[0]
    C = points.shape[-1]
    if idx.dim() == 2:
        S = idx.shape[1]
        flat_idx = idx.unsqueeze(-1).expand(B, S, C)
        return torch.gather(points, 1, flat_idx)
    elif idx.dim() == 3:
        S = idx.shape[1]
        K = idx.shape[2]
        flat_idx = idx.reshape(B, S * K, 1).expand(B, S * K, C)
        return torch.gather(points, 1, flat_idx).view(B, S, K, C)
    else:
        raise ValueError(f"Unsupported idx dim: {idx.dim()}")


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Module: Sampling (FPS) + Grouping (Ball Query) + PointNet MLP
    """
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp_channels: list):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel
        for out_channel in mlp_channels:
            layers.append(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        """
        Args:
            xyz: [B, N, 3] point coordinates
            points: [B, N, D] point features (or None)
        Returns:
            new_xyz: [B, npoint, 3]
            new_points: [B, npoint, mlp[-1]]
        """
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = gather_points(xyz, fps_idx)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = gather_points(xyz, idx)  # [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = gather_points(points, idx)
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm

        # [B, npoint, nsample, D] -> [B, D, npoint, nsample]
        grouped_points = grouped_points.permute(0, 3, 1, 2)
        new_points = self.mlp(grouped_points)
        # Max-pool across local neighborhood
        new_points = torch.max(new_points, dim=-1)[0]  # [B, mlp[-1], npoint]
        new_points = new_points.permute(0, 2, 1)        # [B, npoint, mlp[-1]]

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation Module: Inverse distance weighted interpolation + Skip connection + Shared MLP
    """
    def __init__(self, in_channel: int, mlp_channels: list):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp_channels:
            layers.append(nn.Conv1d(last_channel, out_channel, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolates features from xyz2/points2 to xyz1, concatenates with points1.

        Args:
            xyz1: [B, N, 3] coordinates of denser points
            xyz2: [B, S, 3] coordinates of sparser centroids
            points1: [B, N, C1] features of denser points (or None)
            points2: [B, S, C2] features of sparser centroids
        Returns:
            new_points: [B, N, mlp[-1]]
        """
        dists = square_distance(xyz1, xyz2)
        dists, idx = torch.topk(dists, 3, dim=-1, largest=False)
        dists = torch.clamp(dists, min=1e-4)
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)

        gathered_points2 = gather_points(points2, idx)  # [B, N, 3, C2]
        interpolated_points = torch.sum(gathered_points2 * weight.unsqueeze(-1), dim=2)  # [B, N, C2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, C, N]
        new_points = self.mlp(new_points)
        new_points = new_points.permute(0, 2, 1)  # [B, N, C_out]
        return new_points


class PointNet2SemSeg(nn.Module):
    """
    PointNet++ Semantic Segmentation Network.
    Fully compatible with standard ONNX Runtime C++ and dynamic input point counts.

    Outputs per-point logits for `num_classes`:
      0: terrain
      1: static_obstacle
      2: dynamic_object
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Set Abstraction hierarchy
        # SA1: N -> 512 points
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3 + (in_channels - 3 if in_channels > 3 else 0),
            mlp_channels=[32, 32, 64]
        )
        # SA2: 512 -> 128 points
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=32,
            in_channel=64 + 3,
            mlp_channels=[64, 64, 128]
        )
        # SA3: 128 -> 32 points
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=16,
            in_channel=128 + 3,
            mlp_channels=[128, 128, 256]
        )

        # Feature Propagation hierarchy
        # FP3: 32 -> 128
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp_channels=[256, 128])
        # FP2: 128 -> 512
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 64, mlp_channels=[128, 64])
        # FP1: 512 -> N
        extra_dim = (in_channels - 3) if in_channels > 3 else 3
        self.fp1 = PointNetFeaturePropagation(in_channel=64 + extra_dim, mlp_channels=[64, 64])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, xyz):
        """
        Forward pass.
        Args:
            xyz: [B, N, C] where C >= 3 (first 3 channels are x, y, z)
        Returns:
            logits: [B, N, num_classes]
        """
        coords = xyz[:, :, :3]
        features = xyz[:, :, 3:] if self.in_channels > 3 else None

        # Encoder (Set Abstraction)
        l1_xyz, l1_points = self.sa1(coords, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder (Feature Propagation)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = coords if features is None else torch.cat([coords, features], dim=-1)
        l0_points = self.fp1(coords, l1_xyz, l0_points, l1_points)

        # Classifier
        B, N, C = l0_points.shape
        flat_points = l0_points.reshape(B * N, C)
        logits = self.classifier(flat_points)
        return logits.reshape(B, N, self.num_classes)
