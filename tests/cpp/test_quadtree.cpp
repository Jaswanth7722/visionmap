#include "ps26053/mapping/quadtree.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_quadtree ...\n";

    ps26053::BoundingBox2D bounds{0.0f, 1.0f, 0.0f, 1.0f};
    ps26053::Quadtree tree(bounds, 0.50f, 3);

    // Initial state: 1 root cell
    assert(tree.totalCells() == 1);

    // Insert points
    ps26053::Point3D p1{0.2f, 0.2f, 1.0f, 0.5f, ps26053::SemanticClass::TERRAIN, 0.9f};
    tree.insertPoint(p1, 0.0);

    auto cells_before = tree.getActiveCells();
    assert(cells_before.size() == 1);
    assert(cells_before[0]->point_count == 1);
    assert(cells_before[0]->elevation == 1.0f);

    // Trigger local refinement at (0.2, 0.2)
    tree.refineCellAt(0.2f, 0.2f);

    // Subdividing 1 cell creates 4 child cells
    assert(tree.totalCells() == 4);
    auto cells_after = tree.getActiveCells();
    assert(cells_after.size() == 4);

    // Verify boundary conservation: sum of 4 child areas == parent area
    float total_area = 0.0f;
    for (const auto* c : cells_after) {
        total_area += c->bounds.width() * c->bounds.height();
        assert(c->resolution == 0.25f); // 0.50m / 2 = 0.25m
    }
    assert(std::abs(total_area - 1.0f) < 1e-5);

    std::cout << "[Test PASS] test_quadtree succeeded!\n";
    return 0;
}
