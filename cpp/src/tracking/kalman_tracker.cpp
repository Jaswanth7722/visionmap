#include "ps26053/tracking/kalman_tracker.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>

namespace ps26053 {

SingleKalmanFilter::SingleKalmanFilter(const Eigen::Vector2f& initial_pos, float dt) {
    x = Eigen::Vector4f(initial_pos.x(), initial_pos.y(), 0.0f, 0.0f);
    P = Eigen::Matrix4f::Identity() * 1.0f;
    P(2, 2) = 10.0f;
    P(3, 3) = 10.0f;

    F = Eigen::Matrix4f::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;

    Q = Eigen::Matrix4f::Identity() * 0.1f;
    Q(2, 2) = 0.5f;
    Q(3, 3) = 0.5f;

    H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;

    R = Eigen::Matrix2f::Identity() * 0.2f;
}

void SingleKalmanFilter::predict(float dt) {
    F(0, 2) = dt;
    F(1, 3) = dt;
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void SingleKalmanFilter::update(const Eigen::Vector2f& z) {
    Eigen::Vector2f y = z - H * x; // innovation
    Eigen::Matrix2f S = H * P * H.transpose() + R;
    Eigen::Matrix<float, 4, 2> K = P * H.transpose() * S.inverse();
    x = x + K * y;
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    P = (I - K * H) * P;
}

KalmanTracker::KalmanTracker() {}

std::vector<BoundingBox2D> KalmanTracker::clusterDynamicPoints(const PointCloud& dynamic_cloud) const {
    std::vector<BoundingBox2D> clusters;
    if (dynamic_cloud.empty()) return clusters;

    // Fast grid-based clustering
    float grid_size = 1.0f;
    std::unordered_map<int64_t, std::vector<Point3D>> grid;

    for (const auto& pt : dynamic_cloud) {
        if (pt.semantic_class != SemanticClass::DYNAMIC_OBSTACLE) continue;
        int gx = static_cast<int>(std::floor(pt.x / grid_size));
        int gy = static_cast<int>(std::floor(pt.y / grid_size));
        int64_t key = (static_cast<int64_t>(gx) << 32) | (static_cast<int64_t>(gy) & 0xFFFFFFFFLL);
        grid[key].push_back(pt);
    }

    for (const auto& [key, pts] : grid) {
        if (pts.size() < 5) continue;
        BoundingBox2D bbox{pts[0].x, pts[0].x, pts[0].y, pts[0].y};
        for (const auto& p : pts) {
            bbox.min_x = std::min(bbox.min_x, p.x);
            bbox.max_x = std::max(bbox.max_x, p.x);
            bbox.min_y = std::min(bbox.min_y, p.y);
            bbox.max_y = std::max(bbox.max_y, p.y);
        }
        clusters.push_back(bbox);
    }

    return clusters;
}

void KalmanTracker::associateAndFilter(const std::vector<BoundingBox2D>& clusters, double timestamp) {
    // 1. Predict existing tracks
    float dt = 0.1f;
    for (auto& kf : filters_) {
        kf.predict(dt);
    }

    std::vector<bool> cluster_matched(clusters.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    // 2. Greedy association
    for (size_t t = 0; t < tracks_.size(); ++t) {
        float min_dist = 3.0f; // 3 meter gating threshold
        int best_c = -1;

        for (size_t c = 0; c < clusters.size(); ++c) {
            if (cluster_matched[c]) continue;
            Eigen::Vector2f c_pos(clusters[c].centerX(), clusters[c].centerY());
            Eigen::Vector2f pred_pos(filters_[t].x(0), filters_[t].x(1));
            float d = (c_pos - pred_pos).norm();
            if (d < min_dist) {
                min_dist = d;
                best_c = static_cast<int>(c);
            }
        }

        if (best_c >= 0) {
            cluster_matched[best_c] = true;
            track_matched[t] = true;
            Eigen::Vector2f meas(clusters[best_c].centerX(), clusters[best_c].centerY());
            filters_[t].update(meas);

            tracks_[t].position = Eigen::Vector2f(filters_[t].x(0), filters_[t].x(1));
            tracks_[t].velocity = Eigen::Vector2f(filters_[t].x(2), filters_[t].x(3));
            tracks_[t].bbox = clusters[best_c];
            tracks_[t].hits++;
            tracks_[t].misses = 0;
            if (tracks_[t].hits >= 2) {
                tracks_[t].confirmed = true;
            }
        } else {
            tracks_[t].misses++;
        }
    }

    // 3. Spawn new tracks for unmatched clusters
    for (size_t c = 0; c < clusters.size(); ++c) {
        if (!cluster_matched[c]) {
            Eigen::Vector2f pos(clusters[c].centerX(), clusters[c].centerY());
            SingleKalmanFilter new_kf(pos, dt);
            filters_.push_back(new_kf);

            TrackedObject track;
            track.id = next_track_id_++;
            track.position = pos;
            track.velocity = Eigen::Vector2f(0.0f, 0.0f);
            track.bbox = clusters[c];
            track.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
            track.hits = 1;
            track.misses = 0;
            track.confirmed = false;
            tracks_.push_back(track);
        }
    }

    // 4. Remove dead tracks
    for (int i = static_cast<int>(tracks_.size()) - 1; i >= 0; --i) {
        if (tracks_[i].misses > 3) {
            tracks_.erase(tracks_.begin() + i);
            filters_.erase(filters_.begin() + i);
        }
    }
}

void KalmanTracker::update(const PointCloud& dynamic_cloud, double timestamp) {
    auto clusters = clusterDynamicPoints(dynamic_cloud);
    associateAndFilter(clusters, timestamp);
}

} // namespace ps26053
