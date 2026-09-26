#pragma once

#include "ps26053/common/cell.hpp"
#include <vector>
#include <string>

namespace ps26053 {

struct DistanceBand {
    float max_range;
    float cell_size;
    std::string name;
};

class ResolutionPolicy {
public:
    ResolutionPolicy();

    /**
     * @brief Finest alignment quantum in meters. Every stored cell edge lies on
     * the global lattice anchored at the grid origin with this step, so cells
     * from different distance bands share exact edges instead of overlapping.
     * All default band sizes are exact integer multiples of this quantum.
     */
    static constexpr float alignmentQuantum() { return 0.05f; }

    /**
     * @brief Level 1 — PS baseline policy: get base resolution purely by distance from sensor.
     * 0–10 m  -> 5 cm
     * 10–30 m -> 15 cm
     * 30–60 m -> 30 cm
     * 60–100 m -> 50 cm
     */
    float getBaseResolution(float distance) const;

    /**
     * @brief Integer multiple of the alignment quantum for a band resolution.
     * @return k such that resolution == k * alignmentQuantum(), or -1 when the
     * resolution is not an exact multiple (custom bands added at runtime).
     */
    int quantumMultiple(float resolution) const;

    /**
     * @brief Level 2 — Project innovation: semantic importance + motion trigger local refinement
     * inside a distance band without upgrading the entire band.
     * @return true if the cell warrants Quadtree local refinement to finer resolution
     */
    bool shouldRefine(const Cell& cell, float distance_to_sensor) const;

    /**
     * @brief Compute quantitative importance score in range [0.0, 1.0]
     */
    float computeImportance(const Cell& cell, float distance_to_sensor) const;

    void addBand(float max_range, float cell_size, const std::string& name);

    /**
     * @brief Replace the band table wholesale (H3: loaded from
     * config/resolution.yaml). Bands should be sorted by ascending max_range.
     */
    void setBands(const std::vector<DistanceBand>& bands) { bands_ = bands; }
    void setMotionVelocityThreshold(float v) { motion_velocity_threshold_ = v; }
    void setCurbThreshold(float c) { curb_threshold_ = c; }

private:
    std::vector<DistanceBand> bands_;
    float curb_threshold_{0.12f};        // 12 cm curb height trigger
    float motion_velocity_threshold_{0.25f}; // 0.25 m/s velocity trigger
};

} // namespace ps26053
