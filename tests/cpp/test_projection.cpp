#include "ps26053/mapping/grid_25d.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_projection ...\n";

    ps26053::GridConfig cfg;
    cfg.x_min = -10.0f; cfg.x_max = 10.0f;
    cfg.y_min = -10.0f; cfg.y_max = 10.0f;
    cfg.tile_size = 1.0f;

    ps26053::Grid25D grid(cfg);

    // Create 3D points
    ps26053::PointCloud cloud;
    // Ground point at (2.0, 3.0, -1.5)
    cloud.push_back({2.0f, 3.0f, -1.5f, 0.2f, ps26053::SemanticClass::TERRAIN, 0.95f});
    // Obstacle point at (2.0, 3.0, 0.5)
    cloud.push_back({2.0f, 3.0f, 0.5f, 0.8f, ps26053::SemanticClass::STATIC_OBSTACLE, 0.90f});

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

    std::cout << "[Test PASS] test_projection succeeded!\n";
    return 0;
}
