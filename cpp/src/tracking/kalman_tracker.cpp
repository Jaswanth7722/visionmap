#include "ps26053/tracking/kalman_tracker.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>

namespace ps26053 {

namespace {

// Exact 3D integer cell key: no hash truncation, so distant points can never
// share a cell through key collision.
struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct CellKeyHash {
    size_t operator()(const CellKey& k) const noexcept {
        size_t h = std::hash<int64_t>{}(k.x);
        h ^= std::hash<int64_t>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

} // namespace

SingleKalmanFilter::SingleKalmanFilter(const Eigen::Vector2f& initial_pos, float dt,
                                         float q_pos, float q_vel, float r_pos) {
    x = Eigen::Vector4f(initial_pos.x(), initial_pos.y(), 0.0f, 0.0f);
    P = Eigen::Matrix4f::Identity() * 1.0f;
    P(2, 2) = 10.0f;
    P(3, 3) = 10.0f;

    F = Eigen::Matrix4f::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;

    Q = Eigen::Matrix4f::Identity() * q_pos;
    Q(2, 2) = q_vel;
    Q(3, 3) = q_vel;

    H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;

    R = Eigen::Matrix2f::Identity() * r_pos;
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

std::vector<DynamicCluster> KalmanTracker::clusterDynamicPoints(const PointCloud& dynamic_cloud) const {
    std::vector<DynamicCluster> clusters;
    std::vector<const Point3D*> pts;
    pts.reserve(dynamic_cloud.size());
    for (const auto& pt : dynamic_cloud) {
        if (pt.semantic_class != SemanticClass::DYNAMIC_OBSTACLE) continue;
        pts.push_back(&pt);
    }
    if (pts.empty()) return clusters;

    // 3D Euclidean connected components (H1): points within the configured
    // cluster radius link up, so one vehicle forms one cluster regardless of
    // its span, and objects stacked in Z never merge. Union-find over a 3D
    // spatial hash.
    const float kClusterRadius = config_.cluster_radius;
    const size_t kMinClusterPoints = config_.min_cluster_size;

    std::unordered_map<CellKey, std::vector<size_t>, CellKeyHash> grid;
    for (size_t i = 0; i < pts.size(); ++i) {
        CellKey key{
            static_cast<int64_t>(std::floor(pts[i]->x / kClusterRadius)),
            static_cast<int64_t>(std::floor(pts[i]->y / kClusterRadius)),
            static_cast<int64_t>(std::floor(pts[i]->z / kClusterRadius))};
        grid[key].push_back(i);
    }

    std::vector<size_t> parent(pts.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::function<size_t(size_t)> find = [&](size_t a) -> size_t {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    auto unite = [&](size_t a, size_t b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[a] = b;
    };

    const float r2 = kClusterRadius * kClusterRadius;
    for (size_t i = 0; i < pts.size(); ++i) {
        int64_t ix = static_cast<int64_t>(std::floor(pts[i]->x / kClusterRadius));
        int64_t iy = static_cast<int64_t>(std::floor(pts[i]->y / kClusterRadius));
        int64_t iz = static_cast<int64_t>(std::floor(pts[i]->z / kClusterRadius));
        for (int64_t dx = -1; dx <= 1; ++dx) {
            for (int64_t dy = -1; dy <= 1; ++dy) {
                for (int64_t dz = -1; dz <= 1; ++dz) {
                    auto it = grid.find(CellKey{ix + dx, iy + dy, iz + dz});
                    if (it == grid.end()) continue;
                    for (size_t j : it->second) {
                        if (j <= i) continue; // each unordered pair once
                        float ddx = pts[i]->x - pts[j]->x;
                        float ddy = pts[i]->y - pts[j]->y;
                        float ddz = pts[i]->z - pts[j]->z;
                        if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) {
                            unite(i, j);
                        }
                    }
                }
            }
        }
    }

    std::unordered_map<size_t, BoundingBox2D> boxes;
    std::unordered_map<size_t, size_t> counts;
    std::unordered_map<size_t, double> sum_z;
    for (size_t i = 0; i < pts.size(); ++i) {
        size_t root = find(i);
        auto it = boxes.find(root);
        if (it == boxes.end()) {
            boxes.emplace(root, BoundingBox2D{pts[i]->x, pts[i]->x, pts[i]->y, pts[i]->y});
            counts.emplace(root, 1);
            sum_z.emplace(root, pts[i]->z);
        } else {
            BoundingBox2D& bbox = it->second;
            bbox.min_x = std::min(bbox.min_x, pts[i]->x);
            bbox.max_x = std::max(bbox.max_x, pts[i]->x);
            bbox.min_y = std::min(bbox.min_y, pts[i]->y);
            bbox.max_y = std::max(bbox.max_y, pts[i]->y);
            ++counts[root];
            sum_z[root] += pts[i]->z;
        }
    }

    for (const auto& [root, bbox] : boxes) {
        if (counts[root] >= kMinClusterPoints) {
            // Object-size prior: reject components wider (laterally, Y) than
            // any plausible single dynamic object (see
            // TrackerConfig::max_cluster_width). Width is measured laterally
            // because that is the axis on which background sheets appear:
            // monocular-projected frames quantize depth per image row, so a
            // mislabeled row becomes a band spanning the full image width
            // while staying thin in depth (X).
            if (config_.max_cluster_width > 0.0f &&
                (bbox.max_y - bbox.min_y) > config_.max_cluster_width) {
                continue;
            }
            DynamicCluster cluster;
            cluster.bbox = bbox;
            cluster.mean_z = static_cast<float>(sum_z[root] / static_cast<double>(counts[root]));
            cluster.point_count = counts[root];
            clusters.push_back(cluster);
        }
    }

    // Deterministic cluster order (unordered_map iteration is not), so track
    // IDs are reproducible for identical inputs.
    std::sort(clusters.begin(), clusters.end(), [](const DynamicCluster& a, const DynamicCluster& b) {
        if (a.bbox.min_x != b.bbox.min_x) return a.bbox.min_x < b.bbox.min_x;
        return a.bbox.min_y < b.bbox.min_y;
    });

    return clusters;
}

void KalmanTracker::associateAndFilter(const std::vector<DynamicCluster>& clusters, double timestamp) {
    // 1. Predict existing tracks using the REAL inter-frame interval (H1).
    // Non-positive intervals (repeated timestamps) predict with dt = 0, i.e.
    // no motion assumed; large gaps are clamped to 1 s to keep the
    // covariance update bounded. With no history yet, the configured nominal
    // step applies (harmless: no tracks exist to predict).
    float dt = config_.nominal_dt;
    if (has_last_timestamp_) {
        double raw_dt = timestamp - last_timestamp_;
        dt = (raw_dt > 0.0) ? static_cast<float>(std::min(raw_dt, 1.0)) : 0.0f;
    }
    for (auto& kf : filters_) {
        kf.predict(dt);
    }

    std::vector<bool> cluster_matched(clusters.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    // 2. Greedy association
    for (size_t t = 0; t < tracks_.size(); ++t) {
        float min_dist = config_.association_distance; // configured gating threshold
        int best_c = -1;

        for (size_t c = 0; c < clusters.size(); ++c) {
            if (cluster_matched[c]) continue;
            Eigen::Vector2f c_pos(clusters[c].bbox.centerX(), clusters[c].bbox.centerY());
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
            Eigen::Vector2f meas(clusters[best_c].bbox.centerX(), clusters[best_c].bbox.centerY());
            filters_[t].update(meas);

            tracks_[t].position = Eigen::Vector2f(filters_[t].x(0), filters_[t].x(1));
            tracks_[t].position_z = clusters[best_c].mean_z;
            tracks_[t].velocity = Eigen::Vector2f(filters_[t].x(2), filters_[t].x(3));
            tracks_[t].bbox = clusters[best_c].bbox;
            tracks_[t].hits++;
            tracks_[t].misses = 0;
            // Confidence is computed from association history (H1), never
            // left at the constructor default.
            tracks_[t].confidence = trackConfidence(tracks_[t].hits);
            if (tracks_[t].hits >= config_.min_hits_to_confirm) {
                tracks_[t].confirmed = true;
            }
        } else {
            tracks_[t].misses++;
        }
    }

    // 3. Spawn new tracks for unmatched clusters
    for (size_t c = 0; c < clusters.size(); ++c) {
        if (!cluster_matched[c]) {
            Eigen::Vector2f pos(clusters[c].bbox.centerX(), clusters[c].bbox.centerY());
            SingleKalmanFilter new_kf(pos, dt, config_.process_noise_pos,
                                      config_.process_noise_vel,
                                      config_.measurement_noise_pos);
            filters_.push_back(new_kf);

            TrackedObject track;
            track.id = next_track_id_++;
            track.position = pos;
            track.position_z = clusters[c].mean_z;
            track.velocity = Eigen::Vector2f(0.0f, 0.0f);
            track.bbox = clusters[c].bbox;
            track.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
            track.hits = 1;
            track.misses = 0;
            track.confidence = trackConfidence(1);
            track.confirmed = false;
            tracks_.push_back(track);
        }
    }

    // 4. Remove dead tracks
    for (int i = static_cast<int>(tracks_.size()) - 1; i >= 0; --i) {
        if (tracks_[i].misses > config_.max_missed_frames) {
            tracks_.erase(tracks_.begin() + i);
            filters_.erase(filters_.begin() + i);
        }
    }
}

void KalmanTracker::update(const PointCloud& dynamic_cloud, double timestamp) {
    auto clusters = clusterDynamicPoints(dynamic_cloud);
    associateAndFilter(clusters, timestamp);
    last_timestamp_ = timestamp;
    has_last_timestamp_ = true;
}

} // namespace ps26053
