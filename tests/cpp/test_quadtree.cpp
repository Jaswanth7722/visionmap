#include "ps26053/mapping/quadtree.hpp"
#include <cassert>
#include <cmath>
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

    // C10: refineCellAt must descend past the root to the target cell.
    {
        ps26053::Quadtree deep({0.0f, 1.0f, 0.0f, 1.0f}, 1.0f, 3);
        assert(deep.totalCells() == 1);
        deep.refineCellAt(0.7f, 0.7f);
        assert(deep.totalCells() == 4);   // root subdivided
        deep.refineCellAt(0.7f, 0.7f);
        assert(deep.totalCells() == 7);   // target child subdivided, not the root again
        deep.refineCellAt(0.7f, 0.7f);
        assert(deep.totalCells() == 10);  // third level reached (max_depth = 3)
        deep.refineCellAt(0.7f, 0.7f);
        assert(deep.totalCells() == 10);  // depth cap holds: no further subdivision

        // findLeaf locates the leaf that owns a point.
        ps26053::Cell* leaf = deep.findLeaf(0.7f, 0.7f);
        assert(leaf != nullptr);
        assert(leaf->bounds.contains(0.7f, 0.7f));
        assert(std::abs(leaf->resolution - 0.125f) < 1e-6f);
        assert(leaf->depth == 3); // Depth 3 verified
        assert(deep.findLeaf(5.0f, 5.0f) == nullptr);
    }

    // Task 9 Comprehensive Invariants:
    // 1. Child bounds remain inside parent bounds
    // 2. No child overlaps (pairwise empty interior intersections)
    // 3. No child gaps (exact area conservation)
    // 4. Correct refinement depth tracking (depth 0 -> 1 -> 2 -> 3)
    // 5. Deterministic subdivision
    {
        std::cout << "  testing Quadtree mathematical containment, no-overlap, no-gap invariants...\n";
        ps26053::BoundingBox2D root_bounds{10.0f, 10.50f, -5.0f, -4.50f};
        ps26053::Quadtree q(root_bounds, 0.50f, 3);
        
        auto initial_leaves = q.getActiveCells();
        assert(initial_leaves.size() == 1);
        assert(initial_leaves[0]->depth == 0);
        assert(initial_leaves[0]->resolution == 0.50f);

        // Subdivide Level 1
        q.refineCellAt(10.1f, -4.9f);
        auto level1_leaves = q.getActiveCells();
        assert(level1_leaves.size() == 4);

        float parent_area = root_bounds.width() * root_bounds.height();
        float sum_area = 0.0f;
        for (size_t i = 0; i < level1_leaves.size(); ++i) {
            const auto* c1 = level1_leaves[i];
            assert(c1->depth == 1);
            assert(std::abs(c1->resolution - 0.25f) < 1e-6f);
            // Inside parent bounds
            assert(c1->bounds.min_x >= root_bounds.min_x - 1e-6f);
            assert(c1->bounds.max_x <= root_bounds.max_x + 1e-6f);
            assert(c1->bounds.min_y >= root_bounds.min_y - 1e-6f);
            assert(c1->bounds.max_y <= root_bounds.max_y + 1e-6f);
            sum_area += c1->bounds.width() * c1->bounds.height();

            // Pairwise no overlap
            for (size_t j = i + 1; j < level1_leaves.size(); ++j) {
                const auto* c2 = level1_leaves[j];
                bool overlap = (c1->bounds.min_x < c2->bounds.max_x - 1e-5f &&
                                c2->bounds.min_x < c1->bounds.max_x - 1e-5f &&
                                c1->bounds.min_y < c2->bounds.max_y - 1e-5f &&
                                c2->bounds.min_y < c1->bounds.max_y - 1e-5f);
                assert(!overlap);
            }
        }
        // Zero gaps
        assert(std::abs(sum_area - parent_area) < 1e-5f);

        // Subdivide Level 2 (one quadrant refined)
        q.refineCellAt(10.1f, -4.9f);
        auto level2_leaves = q.getActiveCells();
        assert(level2_leaves.size() == 7);
        float sum_area2 = 0.0f;
        for (const auto* c : level2_leaves) {
            sum_area2 += c->bounds.width() * c->bounds.height();
            assert(c->depth >= 1 && c->depth <= 2);
        }
        assert(std::abs(sum_area2 - parent_area) < 1e-5f);
    }

    std::cout << "[Test PASS] test_quadtree succeeded!\n";
    return 0;
}
