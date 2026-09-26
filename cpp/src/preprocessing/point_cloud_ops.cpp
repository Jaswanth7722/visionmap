#include "ps26053/preprocessing/point_cloud_ops.hpp"
#include <unordered_map>
#include <cmath>
#include <random>

namespace ps26053 {

PointCloud PointCloudOps::filterROI(const PointCloud& input, const PreprocessingConfig& config) {
    PointCloud filtered;
    filtered.reserve(input.size());

    for (const auto& pt : input) {
        if (pt.x >= config.x_min && pt.x <= config.x_max &&
            pt.y >= config.y_min && pt.y <= config.y_max &&
            pt.z >= config.z_min && pt.z <= config.z_max) {
            filtered.push_back(pt);
        }
    }
    return filtered;
}

struct VoxelCentroid {
    float sum_x{0.0f};
    float sum_y{0.0f};
    float sum_z{0.0f};
    float sum_intensity{0.0f};
    int count{0};
    SemanticClass semantic_class{SemanticClass::UNKNOWN};
    float confidence{0.0f};
};

PointCloud PointCloudOps::voxelGridDownsample(const PointCloud& input, float leaf_size) {
    if (leaf_size <= 0.0f || input.empty()) return input;

    float inv_leaf = 1.0f / leaf_size;
    std::unordered_map<int64_t, VoxelCentroid> grid;

    for (const auto& pt : input) {
        int64_t vx = static_cast<int64_t>(std::floor(pt.x * inv_leaf));
        int64_t vy = static_cast<int64_t>(std::floor(pt.y * inv_leaf));
        int64_t vz = static_cast<int64_t>(std::floor(pt.z * inv_leaf));

        // Spatial hash combination
        int64_t key = (vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791);
        auto& cell = grid[key];
        cell.sum_x += pt.x;
        cell.sum_y += pt.y;
        cell.sum_z += pt.z;
        cell.sum_intensity += pt.intensity;
        if (pt.confidence > cell.confidence) {
            cell.semantic_class = pt.semantic_class;
            cell.confidence = pt.confidence;
        }
        cell.count++;
    }

    PointCloud downsampled;
    downsampled.reserve(grid.size());

    for (const auto& [key, cell] : grid) {
        float inv_c = 1.0f / cell.count;
        Point3D pt;
        pt.x = cell.sum_x * inv_c;
        pt.y = cell.sum_y * inv_c;
        pt.z = cell.sum_z * inv_c;
        pt.intensity = cell.sum_intensity * inv_c;
        pt.semantic_class = cell.semantic_class;
        pt.confidence = cell.confidence;
        downsampled.push_back(pt);
    }

    return downsampled;
}

void PointCloudOps::transformPointCloud(PointCloud& cloud, const Eigen::Matrix4f& transform) {
    Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);

    for (auto& pt : cloud) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f tp = R * p + t;
        pt.x = tp.x();
        pt.y = tp.y();
        pt.z = tp.z();
    }
}

PointCloud PointCloudOps::samplePoints(const PointCloud& input, size_t target_points) {
    PointCloud sampled;
    if (input.empty()) return sampled;

    sampled.reserve(target_points);

    if (input.size() <= target_points) {
        sampled = input;
        // Pad by repeating if fewer points
        while (sampled.size() < target_points) {
            sampled.push_back(input[sampled.size() % input.size()]);
        }
        return sampled;
    }

    // Uniform stride sampling with reproducible step
    size_t step = input.size() / target_points;
    for (size_t i = 0; i < target_points; ++i) {
        sampled.push_back(input[i * step]);
    }
    return sampled;
}

} // namespace ps26053
