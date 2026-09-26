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
     * @brief Level 1 — PS baseline policy: get base resolution purely by distance from sensor.
     * 0–10 m  -> 5 cm
     * 10–30 m -> 15 cm
     * 30–60 m -> 30 cm
     * 60–100 m -> 50 cm
     */
    float getBaseResolution(float distance) const;

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

private:
    std::vector<DistanceBand> bands_;
    float curb_threshold_{0.12f};        // 12 cm curb height trigger
    float motion_velocity_threshold_{0.25f}; // 0.25 m/s velocity trigger
};

} // namespace ps26053
