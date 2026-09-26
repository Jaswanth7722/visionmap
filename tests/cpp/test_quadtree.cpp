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

    // C9: point_count is deterministically redistributed, never dropped.
    {
        ps26053::Cell parent;
        parent.bounds = {0.0f, 1.0f, 0.0f, 1.0f};
        parent.resolution = 1.0f;
        parent.elevation = 2.0f;
        parent.occupancy = 0.8f;
        parent.point_count = 7;
        ps26053::Cell kids[4];
        parent.subdivideTo(kids);
        uint32_t total = 0;
        for (int i = 0; i < 4; ++i) {
            total += kids[i].point_count;
            assert(kids[i].resolution == 0.5f);
            assert(kids[i].elevation == 2.0f); // inherited as prior
            assert(kids[i].occupancy == 0.8f);
        }
        assert(total == 7); // 2 + 2 + 2 + 1: exact preservation
        assert(kids[0].point_count == 2 && kids[3].point_count == 1);
    }

    // C9: a tall object in one quadrant must not leave all children identical.
    {
        ps26053::Quadtree t2({0.0f, 1.0f, 0.0f, 1.0f}, 1.0f, 3);
        ps26053::Point3D ground{0.2f, 0.2f, -1.5f, 0.5f, ps26053::SemanticClass::TERRAIN, 0.9f};
        t2.insertPoint(ground, 0.0);
        t2.refineCellAt(0.7f, 0.7f);
        ps26053::Point3D tower{0.7f, 0.7f, 2.0f, 0.8f, ps26053::SemanticClass::STATIC_OBSTACLE, 0.9f};
        t2.insertPoint(tower, 0.0);

        auto leaves = t2.getActiveCells();
        assert(leaves.size() == 4);
        bool found_tall = false;
        bool found_ground = false;
        uint32_t total_pts = 0;
        for (const auto* c : leaves) {
            total_pts += c->point_count;
            if (c->bounds.contains(0.7f, 0.7f)) {
                assert(c->elevation == 2.0f);
                found_tall = true;
            } else {
                assert(c->elevation == -1.5f);
                found_ground = true;
            }
        }
        assert(found_tall && found_ground); // children genuinely differ
        assert(total_pts == 2);             // no observation dropped
    }

    std::cout << "[Test PASS] test_quadtree succeeded!\n";
    return 0;
}
