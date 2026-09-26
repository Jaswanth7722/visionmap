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

    SingleKalmanFilter(const Eigen::Vector2f& initial_pos, float dt = 0.1f,
                       float q_pos = 0.1f, float q_vel = 0.5f, float r_pos = 0.2f);
    void predict(float dt);
    void update(const Eigen::Vector2f& measurement);
};

/**
 * @brief Tracker tunables, loaded from config/tracking.yaml (H3).
 * Compiled defaults mirror the shipped YAML so behavior is identical with
 * or without the config files; tests configure explicitly where they need
 * smaller clusters.
 */
struct TrackerConfig {
    float association_distance{2.5f};
    int max_missed_frames{5};
    int min_hits_to_confirm{3};
    size_t min_cluster_size{15};
    float cluster_radius{0.65f};
    // Maximum lateral (Y) extent of one dynamic object in metres.
    // Components wider than this are rejected as background sheets, not
    // discrete objects. A mis-segmented LiDAR plane (e.g. a flat ground
    // return labeled DYNAMIC by inference) typically spans the full scan
    // width while staying thin in depth — the lateral extent gate rejects it.
    // Road vehicles are narrow across the road (<3 m) and long along it;
    // a 12 m bus is ~2.5 m in Y and passes. Trade-off, stated openly: a
    // laterally-spread group (e.g. a crowd shoulder-to-shoulder wider than
    // this) is rejected too. 0 disables.
    float max_cluster_width{6.0f};
    float nominal_dt{0.1f};
    float process_noise_pos{0.1f};
    float process_noise_vel{0.5f};
    float measurement_noise_pos{0.2f};
};

/**
 * @brief One 3D-connected dynamic cluster: 2D footprint plus mean height.
 */
struct DynamicCluster {
    BoundingBox2D bbox;
    float mean_z{0.0f};
    size_t point_count{0};
};

class KalmanTracker {
public:
    KalmanTracker();

    void configure(const TrackerConfig& config) { config_ = config; }
    const TrackerConfig& config() const { return config_; }

    /**
     * @brief Extract clusters from dynamic points and update tracks
     */
    void update(const PointCloud& dynamic_cloud, double timestamp);

    const std::vector<TrackedObject>& getActiveTracks() const { return tracks_; }

private:
    int32_t next_track_id_{1};
    std::vector<TrackedObject> tracks_;
    std::vector<SingleKalmanFilter> filters_;
    double last_timestamp_{0.0};
    bool has_last_timestamp_{false};
    TrackerConfig config_;

    /**
     * @brief Track confidence from association history: a new track starts
     * at 0.50 and gains 0.10 per consecutive hit, capped at 0.95.
     */
    static float trackConfidence(int32_t hits) {
        float c = 0.40f + 0.10f * static_cast<float>(hits);
        return (c > 0.95f) ? 0.95f : c;
    }

    std::vector<DynamicCluster> clusterDynamicPoints(const PointCloud& dynamic_cloud) const;
    void associateAndFilter(const std::vector<DynamicCluster>& clusters, double timestamp);
};

} // namespace ps26053
