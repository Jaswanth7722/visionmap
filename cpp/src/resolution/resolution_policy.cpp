#include "ps26053/resolution/resolution_policy.hpp"
#include <cmath>
#include <algorithm>

namespace ps26053 {

ResolutionPolicy::ResolutionPolicy() {
    // Exact distance bands from Section 1 of PS26053 Architecture Spec
    bands_.push_back({10.0f,  0.05f, "near_critical"}); // 0–10m  -> 5 cm
    bands_.push_back({30.0f,  0.15f, "mid_range"});     // 10–30m -> 15 cm
    bands_.push_back({60.0f,  0.30f, "far_range"});     // 30–60m -> 30 cm
    bands_.push_back({100.0f, 0.50f, "horizon"});       // 60–100m -> 50 cm
}

void ResolutionPolicy::addBand(float max_range, float cell_size, const std::string& name) {
    bands_.push_back({max_range, cell_size, name});
}

float ResolutionPolicy::getBaseResolution(float distance) const {
    for (const auto& band : bands_) {
        if (distance <= band.max_range) {
            return band.cell_size;
        }
    }
    return bands_.empty() ? 0.50f : bands_.back().cell_size;
}

float ResolutionPolicy::computeImportance(const Cell& cell, float distance_to_sensor) const {
    float importance = 0.0f;

    // 1. Proximity score (decay with distance)
    float prox_score = std::max(0.0f, 1.0f - (distance_to_sensor / 60.0f));
    importance += 0.25f * prox_score;

    // 2. Dynamic obstacle motion score (Level 2 project innovation)
    float speed = std::hypot(cell.velocity_x, cell.velocity_y);
    if (cell.semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
        if (speed >= motion_velocity_threshold_ || cell.object_id >= 0) {
            importance += 0.45f;
        } else {
            importance += 0.25f;
        }
    } else if (cell.semantic_class == SemanticClass::STATIC_OBSTACLE) {
        importance += 0.15f;
    }

    // 3. Elevation gradient / curb / cliff detection
    if (cell.clearance >= curb_threshold_ && cell.clearance <= 1.5f) {
        // High priority: obstacle edge or curb
        importance += 0.20f;
    }

    // 4. Uncertainty score (low confidence needs higher spatial sampling)
    if (cell.semantic_confidence > 0.0f && cell.semantic_confidence < 0.70f) {
        importance += 0.10f;
    }

    return std::clamp(importance, 0.0f, 1.0f);
}

bool ResolutionPolicy::shouldRefine(const Cell& cell, float distance_to_sensor) const {
    // Never refine if already at maximum resolution (5 cm)
    if (cell.resolution <= 0.05f) {
        return false;
    }

    float importance = computeImportance(cell, distance_to_sensor);
    
    // Triggers for local refinement:
    // 1. Moving dynamic object inside mid/far bands (e.g. vehicle at 25m refined to 5cm/7.5cm)
    if (cell.semantic_class == SemanticClass::DYNAMIC_OBSTACLE && 
        (std::hypot(cell.velocity_x, cell.velocity_y) >= motion_velocity_threshold_ || cell.object_id >= 0)) {
        return true;
    }

    // 2. High importance score threshold
    if (importance >= 0.55f) {
        return true;
    }

    return false;
}

} // namespace ps26053
