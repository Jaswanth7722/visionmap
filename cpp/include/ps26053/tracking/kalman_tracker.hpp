#pragma once

#include "ps26053/common/types.hpp"
#include <vector>
#include <Eigen/Dense>

namespace ps26053 {

struct SingleKalmanFilter {
    Eigen::Vector4f x; // [px, py, vx, vy]
    Eigen::Matrix4f P; // State covariance
    Eigen::Matrix4f F; // Transition matrix
    Eigen::Matrix4f Q; // Process noise
    Eigen::Matrix<float, 2, 4> H; // Measurement matrix
    Eigen::Matrix2f R; // Measurement noise

    SingleKalmanFilter(const Eigen::Vector2f& initial_pos, float dt = 0.1f);
    void predict(float dt);
    void update(const Eigen::Vector2f& measurement);
};

class KalmanTracker {
public:
    KalmanTracker();

    /**
     * @brief Extract clusters from dynamic points and update tracks
     */
    void update(const PointCloud& dynamic_cloud, double timestamp);

    const std::vector<TrackedObject>& getActiveTracks() const { return tracks_; }

private:
    int32_t next_track_id_{1};
    std::vector<TrackedObject> tracks_;
    std::vector<SingleKalmanFilter> filters_;

    std::vector<BoundingBox2D> clusterDynamicPoints(const PointCloud& dynamic_cloud) const;
    void associateAndFilter(const std::vector<BoundingBox2D>& clusters, double timestamp);
};

} // namespace ps26053
