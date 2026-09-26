#include "ps26053/mapping/grid_25d.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_temporal_fusion ...\n";

    ps26053::GridConfig cfg;
    cfg.x_min = -10.0f; cfg.x_max = 10.0f;
    cfg.y_min = -10.0f; cfg.y_max = 10.0f;
    ps26053::Grid25D grid(cfg);

    // Frame 0 at t = 0.0s: Dynamic obstacle observed
    ps26053::PointCloud cloud0;
    cloud0.push_back({3.0f, 3.0f, 0.5f, 0.8f, ps26053::SemanticClass::DYNAMIC_OBSTACLE, 0.9f});
    grid.updateWithPointCloud(cloud0, 0.0);

    auto cells0 = grid.getAllCells();
    float init_occ = 0.0f;
    for (const auto& c : cells0) {
        if (c.bounds.contains(3.0f, 3.0f)) {
            init_occ = c.occupancy;
            break;
        }
    }
    assert(init_occ > 0.0f);

    // Fast forward 5 seconds with no new observations -> stale cell should decay
    grid.decayTemporal(5.0, 2.0);

    auto cells1 = grid.getAllCells();
    float decayed_occ = 0.0f;
    for (const auto& c : cells1) {
        if (c.bounds.contains(3.0f, 3.0f)) {
            decayed_occ = c.occupancy;
            break;
        }
    }

    assert(decayed_occ < init_occ);
    std::cout << "Initial occupancy: " << init_occ << ", Decayed occupancy: " << decayed_occ << "\n";

    std::cout << "[Test PASS] test_temporal_fusion succeeded!\n";
    return 0;
}
