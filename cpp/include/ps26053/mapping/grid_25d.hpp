#pragma once

#include "ps26053/common/cell.hpp"
#include "ps26053/resolution/resolution_policy.hpp"
#include "ps26053/mapping/quadtree.hpp"
#include <unordered_map>
#include <vector>
#include <string>

namespace ps26053 {

struct GridConfig {
    float x_min{-60.0f};
    float x_max{60.0f};
    float y_min{-60.0f};
    float y_max{60.0f};
    float tile_size{1.0f}; // 1.0 meter master tile
};

class Grid25D {
public:
    explicit Grid25D(const GridConfig& config = GridConfig());
    ~Grid25D() = default;

    /**
     * @brief Ingest processed & segmented point cloud into the 2.5D world model
     */
    void updateWithPointCloud(const PointCloud& cloud, double timestamp, const Eigen::Vector3f& sensor_pos = Eigen::Vector3f::Zero());

    /**
     * @brief Associate and update cells with dynamic tracks
     */
    void updateTrackedObjects(const std::vector<TrackedObject>& tracks);

    /**
     * @brief Temporal decay for stale dynamic cells
     */
    void decayTemporal(double current_time, double max_staleness_sec = 2.0);

    /**
     * @brief Export the 2.5D elevation & semantic map as a 3D colored PLY
     */
    bool exportToPLY(const std::string& filepath, bool occupied_only = true) const;

    /**
     * @brief Get all cells across all tiles (for benchmarking & visualization)
     */
    std::vector<Cell> getAllCells() const;

    size_t totalActiveCells() const;
    size_t totalAllocatedTiles() const { return tiles_.size(); }

    const ResolutionPolicy& getResolutionPolicy() const { return policy_; }

private:
    GridConfig config_;
    ResolutionPolicy policy_;

    // Master tile hash key: (tile_ix, tile_iy)
    int64_t tileKey(int ix, int iy) const {
        return (static_cast<int64_t>(ix) << 32) | (static_cast<int64_t>(iy) & 0xFFFFFFFFLL);
    }

    std::pair<int, int> getTileIndices(float x, float y) const;

    std::unordered_map<int64_t, std::unique_ptr<Quadtree>> tiles_;
};

} // namespace ps26053
