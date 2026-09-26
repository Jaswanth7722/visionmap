#include "ps26053/mapping/grid_25d.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace {

ps26053::Point3D makePt(float x, float y, float z, ps26053::SemanticClass c) {
    ps26053::Point3D p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.intensity = 0.5f;
    p.semantic_class = c;
    p.confidence = 0.9f;
    return p;
}

} // namespace

int main() {
    std::cout << "[Test] Running test_boundary_alignment ...\n";

    ps26053::ResolutionPolicy policy;

    // Exact band transition distances (the values test_resolution never probes).
    assert(policy.getBaseResolution(10.0f) == 0.05f);
    assert(policy.getBaseResolution(30.0f) == 0.15f);
    assert(policy.getBaseResolution(60.0f) == 0.30f);
    assert(policy.getBaseResolution(100.0f) == 0.50f);
    // Just outside each transition falls into the coarser band.
    assert(policy.getBaseResolution(10.01f) == 0.15f);
    assert(policy.getBaseResolution(30.01f) == 0.30f);
    assert(policy.getBaseResolution(60.01f) == 0.50f);

    // Every default band is an exact multiple of the alignment quantum.
    assert(policy.quantumMultiple(0.05f) == 1);
    assert(policy.quantumMultiple(0.15f) == 3);
    assert(policy.quantumMultiple(0.30f) == 6);
    assert(policy.quantumMultiple(0.50f) == 10);
    assert(policy.quantumMultiple(0.07f) == -1);

    // Sweep a dense line of points across every band transition, mixing
    // terrain, static and dynamic classes so refinement also fires.
    // Identically indexed cells from different bands must not alias, and no
    // two stored footprints may overlap.
    ps26053::PointCloud cloud;
    for (float x = -55.0f; x < 55.0f; x += 0.02f) {
        ps26053::SemanticClass c = ps26053::SemanticClass::TERRAIN;
        float ax = std::fabs(x);
        if (ax > 8.0f && ax < 12.0f) c = ps26053::SemanticClass::STATIC_OBSTACLE;
        if (ax > 28.0f && ax < 32.0f) c = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
        cloud.push_back(makePt(x, 0.10f, -1.5f, c));
        cloud.push_back(makePt(x, -0.10f, 0.4f, c));
    }

    ps26053::Grid25D grid;
    grid.updateWithPointCloud(cloud, 0.0);
    ps26053::BoundaryQA qa = grid.checkBoundaryAlignment();
    std::cout << "  checked cells: " << qa.checked_cells
              << ", overlapping quanta: " << qa.overlapping_quanta
              << ", misaligned cells: " << qa.misaligned_cells << "\n";
    assert(qa.checked_cells > 0);
    assert(qa.totalErrors() == 0);

    // Deterministic mapping: the same cloud into a fresh grid must produce
    // exactly the same stored cell count.
    ps26053::Grid25D grid2;
    grid2.updateWithPointCloud(cloud, 0.0);
    assert(grid2.totalActiveCells() == grid.totalActiveCells());
    assert(grid2.checkBoundaryAlignment().totalErrors() == 0);

    // C10: the live importance engine (not a hardcoded class check) drives
    // refinement. Two low-confidence dynamic points 0.5 m apart in z at
    // ~20 m give importance ~= 0.17 (proximity) + 0.25 (dynamic) + 0.20
    // (curb-range clearance, second point only) + 0.10 (uncertain): the first
    // point scores ~0.52 and must NOT refine, the second scores ~0.72 and
    // must. The refined cell records its importance on every child.
    {
        ps26053::Grid25D g3;
        ps26053::PointCloud c2;
        c2.push_back(makePt(20.0f, 0.0f, -1.0f, ps26053::SemanticClass::DYNAMIC_OBSTACLE));
        c2.push_back(makePt(20.0f, 0.0f, -0.5f, ps26053::SemanticClass::DYNAMIC_OBSTACLE));
        // Lower confidence below the 0.70 uncertainty trigger.
        c2[0].confidence = 0.5f;
        c2[1].confidence = 0.5f;
        g3.updateWithPointCloud(c2, 0.0);
        auto leaves = g3.getAllCells();
        assert(leaves.size() == 4); // one base cell refined into four children
        for (const auto& cell : leaves) {
            assert(cell.importance > 0.0f);
        }
        assert(g3.checkBoundaryAlignment().totalErrors() == 0);
    }

    std::cout << "[Test PASS] test_boundary_alignment succeeded!\n";
    return 0;
}
