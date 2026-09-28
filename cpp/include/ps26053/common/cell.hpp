#pragma once

#include "ps26053/common/types.hpp"
#include <algorithm>
#include <cmath>

namespace ps26053 {

/**
 * @brief 2.5D World Model Cell
 * Implements Section 4 of the PS26053 Architecture Specification.
 * Every cell has explicit bounds and resolution to prevent gaps or overlaps.
 */
struct Cell {
    BoundingBox2D bounds;       ///< Explicit spatial extent (min_x, max_x, min_y, max_y)
    float resolution{0.05f};    ///< This cell's actual edge size in meters
    float elevation{0.0f};      ///< Summarized top obstacle height in meters
    float ground_z{0.0f};       ///< Estimated ground height
    float min_z{0.0f};          ///< Minimum z recorded in this cell
    float max_z{0.0f};          ///< Maximum z recorded in this cell
    float clearance{0.0f};      ///< Vertical clearance (max_z - min_z)
    float occupancy{0.0f};      ///< Occupancy probability [0.0 = free, 1.0 = occupied]
    
    SemanticClass semantic_class{SemanticClass::TERRAIN};
    float semantic_confidence{0.0f};
    int32_t object_id{-1};      ///< Associated tracked dynamic object ID (-1 if none)
    float velocity_x{0.0f};     ///< Estimated velocity in m/s
    float velocity_y{0.0f};
    float importance{0.0f};     ///< Evaluated Level-2 importance score [0.0, 1.0]
    int depth{0};               ///< Quadtree refinement depth (0 = base band cell, 1..max_depth = refined)
    double timestamp{0.0};      ///< Timestamp of last update
    uint32_t point_count{0};    ///< Number of LiDAR points contributing to this cell

    bool is_occupied() const { return occupancy >= 0.5f; }
    bool is_dynamic() const { return (semantic_class == SemanticClass::DYNAMIC_OBSTACLE && (std::hypot(velocity_x, velocity_y) > 0.2f || object_id >= 0)); }

    /**
     * @brief Update cell with a new 3D point observation
     */
    void addPoint(const Point3D& pt, double stamp) {
        if (point_count == 0) {
            min_z = pt.z;
            max_z = pt.z;
            elevation = pt.z;
            ground_z = (pt.semantic_class == SemanticClass::TERRAIN) ? pt.z : pt.z - 0.2f;
            semantic_class = pt.semantic_class;
            semantic_confidence = pt.confidence;
        } else {
            min_z = std::min(min_z, pt.z);
            max_z = std::max(max_z, pt.z);
            // Elevation summarizes obstacle height
            if (pt.semantic_class != SemanticClass::TERRAIN) {
                elevation = std::max(elevation, pt.z);
            }
            if (pt.confidence > semantic_confidence) {
                semantic_class = pt.semantic_class;
                semantic_confidence = pt.confidence;
            }
        }
        clearance = max_z - min_z;
        occupancy = std::min(1.0f, occupancy + 0.35f);
        timestamp = stamp;
        point_count++;
    }

    /**
     * @brief Preserve and subdivide state to 4 quadtree children.
     *
     * Alignment rule 4: parent state is preserved or deterministically
     * redistributed — never dropped. Each child inherits the parent's belief
     * state (occupancy, semantics, confidence, motion, importance, timestamp)
     * and its geometric summary (elevation, min/max z, clearance, ground)
     * as a prior, because the observations that produced them are not tracked
     * per quadrant. Later point insertions into individual children refine
     * them away from the prior. point_count is split as evenly as possible so
     * the total is preserved exactly (remainder goes to the lowest child
     * indices, deterministically).
     */
    void subdivideTo(Cell children[4]) const {
        float half_res = resolution * 0.5f;
        float mid_x = bounds.min_x + half_res;
        float mid_y = bounds.min_y + half_res;

        // Child 0: Bottom-Left (SW)
        children[0].bounds = {bounds.min_x, mid_x, bounds.min_y, mid_y};
        // Child 1: Bottom-Right (SE)
        children[1].bounds = {mid_x, bounds.max_x, bounds.min_y, mid_y};
        // Child 2: Top-Left (NW)
        children[2].bounds = {bounds.min_x, mid_x, mid_y, bounds.max_y};
        // Child 3: Top-Right (NE)
        children[3].bounds = {mid_x, bounds.max_x, mid_y, bounds.max_y};

        uint32_t base_share = point_count / 4;
        uint32_t remainder = point_count % 4;

        for (int i = 0; i < 4; ++i) {
            children[i].resolution = half_res;
            children[i].elevation = elevation;
            children[i].ground_z = ground_z;
            children[i].min_z = min_z;
            children[i].max_z = max_z;
            children[i].clearance = clearance;
            children[i].occupancy = occupancy;
            children[i].semantic_class = semantic_class;
            children[i].semantic_confidence = semantic_confidence;
            children[i].object_id = object_id;
            children[i].velocity_x = velocity_x;
            children[i].velocity_y = velocity_y;
            children[i].importance = importance;
            children[i].depth = depth + 1;
            children[i].timestamp = timestamp;
            children[i].point_count = base_share + (static_cast<uint32_t>(i) < remainder ? 1u : 0u);
        }
    }
};

} // namespace ps26053
