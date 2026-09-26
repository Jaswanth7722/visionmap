#include "ps26053/preprocessing/point_cloud_ops.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

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

PointCloud PointCloudOps::voxelGridDownsample(const PointCloud& input, float leaf_size) {
    if (leaf_size <= 0.0f || input.empty()) return input;

    // Exact voxelization (C3): sort points by integer voxel coordinates and
    // accumulate runs. The previous 32-bit XOR spatial hash merged voxels up
    // to 24.9 m apart; sorting uses the full 64-bit coordinates, so distinct
    // voxels can never collide. Output order is deterministic (sorted voxel
    // order), which also removes hash-iteration nondeterminism downstream.
    float inv_leaf = 1.0f / leaf_size;

    struct VoxelRef {
        int64_t vx, vy, vz;
        size_t idx;
        bool operator<(const VoxelRef& o) const {
            if (vx != o.vx) return vx < o.vx;
            if (vy != o.vy) return vy < o.vy;
            if (vz != o.vz) return vz < o.vz;
            return idx < o.idx;
        }
    };

    std::vector<VoxelRef> order;
    order.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const auto& pt = input[i];
        order.push_back({
            static_cast<int64_t>(std::floor(pt.x * inv_leaf)),
            static_cast<int64_t>(std::floor(pt.y * inv_leaf)),
            static_cast<int64_t>(std::floor(pt.z * inv_leaf)),
            i
        });
    }
    std::sort(order.begin(), order.end());

    PointCloud downsampled;
    downsampled.reserve(input.size());

    size_t run_start = 0;
    auto flush_run = [&](size_t begin, size_t end) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f, sum_i = 0.0f;
        SemanticClass cls = SemanticClass::UNKNOWN;
        float conf = 0.0f;
        for (size_t k = begin; k < end; ++k) {
            const auto& pt = input[order[k].idx];
            sum_x += pt.x; sum_y += pt.y; sum_z += pt.z;
            sum_i += pt.intensity;
            if (pt.confidence > conf) {
                conf = pt.confidence;
                cls = pt.semantic_class;
            }
        }
        float inv_c = 1.0f / static_cast<float>(end - begin);
        Point3D pt;
        pt.x = sum_x * inv_c;
        pt.y = sum_y * inv_c;
        pt.z = sum_z * inv_c;
        pt.intensity = sum_i * inv_c;
        pt.semantic_class = cls;
        pt.confidence = conf;
        downsampled.push_back(pt);
    };

    for (size_t i = 1; i <= order.size(); ++i) {
        if (i == order.size() ||
            order[i].vx != order[run_start].vx ||
            order[i].vy != order[run_start].vy ||
            order[i].vz != order[run_start].vz) {
            flush_run(run_start, i);
            run_start = i;
        }
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
