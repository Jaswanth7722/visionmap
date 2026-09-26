#pragma once

#include "ps26053/common/types.hpp"
#include <Eigen/Dense>

namespace ps26053 {

struct PreprocessingConfig {
    float x_min{-50.0f};
    float x_max{50.0f};
    float y_min{-50.0f};
    float y_max{50.0f};
    float z_min{-4.0f};
    float z_max{6.0f};
    float voxel_leaf_size{0.10f}; // 10 cm voxel downsampling
    bool enable_voxel{true};
};

class PointCloudOps {
public:
    /**
     * @brief Crop point cloud to Region of Interest (ROI)
     */
    static PointCloud filterROI(const PointCloud& input, const PreprocessingConfig& config);

    /**
     * @brief Fast voxel grid downsampling with centroid computation
     */
    static PointCloud voxelGridDownsample(const PointCloud& input, float leaf_size);

    /**
     * @brief Transform point cloud by 4x4 rigid transformation matrix
     */
    static void transformPointCloud(PointCloud& cloud, const Eigen::Matrix4f& transform);

    /**
     * @brief Subsample point cloud to exactly N points for PointNet++ inference
     */
    static PointCloud samplePoints(const PointCloud& input, size_t target_points = 4096);
};

} // namespace ps26053
