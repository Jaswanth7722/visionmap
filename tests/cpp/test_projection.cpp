#include "ps26053/mapping/grid_25d.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_projection ...\n";

    ps26053::GridConfig cfg;
    cfg.x_min = -10.0f; cfg.x_max = 10.0f;
    cfg.y_min = -10.0f; cfg.y_max = 10.0f;
    // L3: cfg.tile_size is intentionally not set here — it is a reserved
    // config field with no effect on cell layout (see M8).

    ps26053::Grid25D grid(cfg);

    // Create 3D points
    ps26053::PointCloud cloud;
    // Ground point at (2.0, 3.0, -1.5)
    cloud.push_back({2.0f, 3.0f, -1.5f, 0.2f, ps26053::SemanticClass::TERRAIN, 0.95f});
    // Obstacle point at (2.0, 3.0, 0.5) with the highest confidence, so the
    // cell adopts the STATIC classification (needed for the H8 check below).
    cloud.push_back({2.0f, 3.0f, 0.5f, 0.8f, ps26053::SemanticClass::STATIC_OBSTACLE, 0.98f});

    grid.updateWithPointCloud(cloud, 0.0);

    auto cells = grid.getAllCells();
    assert(!cells.empty());

    // Find the cell containing (2.0, 3.0)
    const ps26053::Cell* target_cell = nullptr;
    for (const auto& c : cells) {
        if (c.bounds.contains(2.0f, 3.0f)) {
            target_cell = &c;
            break;
        }
    }

    assert(target_cell != nullptr);
    // Elevation should record the highest obstacle height
    assert(target_cell->elevation == 0.5f);
    assert(target_cell->min_z == -1.5f);
    assert(target_cell->clearance == (0.5f - (-1.5f)));

    // H8: a confirmed track overlapping the cell associates identity and
    // motion but must NOT overwrite the network's STATIC classification.
    {
        ps26053::TrackedObject track;
        track.id = 7;
        track.position = Eigen::Vector2f(2.0f, 3.0f);
        track.velocity = Eigen::Vector2f(1.5f, 0.0f);
        track.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
        track.confidence = 0.9f;
        track.bbox = {1.0f, 3.0f, 2.0f, 4.0f};
        track.hits = 3;
        track.confirmed = true;
        std::vector<ps26053::TrackedObject> tracks{track};
        grid.updateTrackedObjects(tracks);

        auto cells_after_track = grid.getAllCells();
        const ps26053::Cell* tracked_cell = nullptr;
        for (const auto& c : cells_after_track) {
            if (c.bounds.contains(2.0f, 3.0f)) {
                tracked_cell = &c;
                break;
            }
        }
        assert(tracked_cell != nullptr);
        assert(tracked_cell->semantic_class == ps26053::SemanticClass::STATIC_OBSTACLE);
        assert(tracked_cell->object_id == 7);
        assert(tracked_cell->velocity_x == 1.5f);
    }

    std::cout << "[Test PASS] test_projection succeeded!\n";
    return 0;
}
