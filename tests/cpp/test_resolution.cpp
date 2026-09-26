#include "ps26053/resolution/resolution_policy.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_resolution ...\n";

    ps26053::ResolutionPolicy policy;

    // Test Level 1: Distance bands
    // 0-10m -> 5cm
    assert(policy.getBaseResolution(5.0f) == 0.05f);
    assert(policy.getBaseResolution(9.9f) == 0.05f);

    // 10-30m -> 15cm
    assert(policy.getBaseResolution(10.1f) == 0.15f);
    assert(policy.getBaseResolution(25.0f) == 0.15f);

    // 30-60m -> 30cm
    assert(policy.getBaseResolution(35.0f) == 0.30f);
    assert(policy.getBaseResolution(59.9f) == 0.30f);

    // 60-100m -> 50cm
    assert(policy.getBaseResolution(65.0f) == 0.50f);
    assert(policy.getBaseResolution(90.0f) == 0.50f);

    // Test Level 2: Local refinement triggers
    ps26053::Cell cell;
    cell.bounds = {20.0f, 20.15f, 5.0f, 5.15f};
    cell.resolution = 0.15f; // inside mid-band
    cell.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
    cell.velocity_x = 2.5f; // moving vehicle
    cell.velocity_y = 0.0f;
    cell.object_id = 1;

    // Moving dynamic obstacle MUST trigger refinement
    bool refine = policy.shouldRefine(cell, 20.6f);
    assert(refine == true);

    // Static terrain without elevation gradient should NOT refine
    ps26053::Cell terrain_cell;
    terrain_cell.bounds = {25.0f, 25.15f, 0.0f, 0.15f};
    terrain_cell.resolution = 0.15f;
    terrain_cell.semantic_class = ps26053::SemanticClass::TERRAIN;
    terrain_cell.clearance = 0.02f;
    bool refine_terrain = policy.shouldRefine(terrain_cell, 25.0f);
    assert(refine_terrain == false);

    std::cout << "[Test PASS] test_resolution succeeded!\n";
    return 0;
}
