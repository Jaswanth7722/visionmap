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
    int max_depth{3};      // quadtree subdivision depth (resolution.yaml)
};

/**
 * @brief Result of the boundary-alignment QA check (architecture spec, section 7).
 *
 * This is computed from the actual stored cells every time it is requested:
 * overlapping_quanta counts 5 cm lattice microcells covered by more than one
 * stored cell footprint, and misaligned_cells counts cells whose stored width
 * or height disagrees with their stored resolution. It must never be replaced
 * by a hardcoded constant at any reporting surface.
 */
struct BoundaryQA {
    size_t overlapping_quanta{0};
    size_t misaligned_cells{0};
    size_t checked_cells{0};
    size_t totalErrors() const { return overlapping_quanta + misaligned_cells; }
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

    /**
     * @brief Compute boundary-alignment errors over the stored cells.
     */
    BoundaryQA checkBoundaryAlignment() const;

    /**
     * @brief Measured memory footprint of the live grid in bytes.
     *
     * Sums exact sizeof terms (Quadtree objects, heap nodes, leaf Cells)
     * plus a documented per-entry allowance for the unordered_map node
     * itself (libstdc++: next pointer + key + value ≈ 24 B, plus malloc
     * overhead; charged as 40 B and labeled approximate). Bucket-array
     * storage is included exactly, as is the microcell ownership index
     * that guarantees disjoint footprints. Used by the honest benchmark (C2).
     */
    size_t estimateMemoryBytes() const;

    /**
     * @brief Apply a new configuration (H3: from config files). Clears all
     * stored cells, since bounds, bands and depth define cell identity.
     * Must be called before the first frame, not mid-stream.
     */
    void configure(const GridConfig& config) {
        config_ = config;
        tiles_.clear();
        micro_owner_.clear();
    }

    const ResolutionPolicy& getResolutionPolicy() const { return policy_; }
    ResolutionPolicy& mutablePolicy() { return policy_; }

private:
    GridConfig config_;
    ResolutionPolicy policy_;

    // Deterministic cell key: quantum multiple tag + cell indices at that
    // resolution. The tag keeps identically indexed cells from different
    // distance bands distinct instead of aliasing them into one quadtree.
    static int64_t cellKey(int quantum_multiple, int cx, int cy) {
        return (static_cast<int64_t>(quantum_multiple) << 56) |
               ((static_cast<int64_t>(cx) & 0xFFFFFFLL) << 28) |
               (static_cast<int64_t>(cy) & 0xFFFFFFLL);
    }

    static int floorDiv(int a, int b) {
        int q = a / b;
        int r = a % b;
        if ((r != 0) && ((r < 0) != (b < 0))) --q;
        return q;
    }

    // M8: the old master-tile index helper was dead code (the microcell
    // lattice supersedes tiling) and has been removed. tile_size remains a
    // loaded config field reserved for future use.

    // Level-2 refinement for one observation, driven by the importance
    // engine (distance + motion + semantic relevance).
    void refineByImportance(Quadtree& tree, const Point3D& pt, const Eigen::Vector3f& sensor_pos);

    std::unordered_map<int64_t, std::unique_ptr<Quadtree>> tiles_;

    // Microcell ownership: 5 cm lattice key -> base-cell key. Every occupied
    // microcell has exactly one owner, so a new base cell is only ever
    // created over free microcells and stored footprints cannot overlap.
    std::unordered_map<int64_t, int64_t> micro_owner_;

    static int64_t microKey(int mx, int my) {
        return (static_cast<int64_t>(mx) << 32) | (static_cast<int64_t>(my) & 0xFFFFFFFFLL);
    }
};

} // namespace ps26053
