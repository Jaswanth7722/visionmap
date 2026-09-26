#include "ps26053/mapping/quadtree.hpp"

namespace ps26053 {

QuadtreeNode::QuadtreeNode(const BoundingBox2D& bounds, float resolution, int depth)
    : bounds_(bounds), depth_(depth) {
    cell_.bounds = bounds;
    cell_.resolution = resolution;
}

int QuadtreeNode::getChildIndex(float x, float y) const {
    float mid_x = bounds_.centerX();
    float mid_y = bounds_.centerY();
    int idx = 0;
    if (x >= mid_x) idx |= 1; // right
    if (y >= mid_y) idx |= 2; // top
    return idx;
}

void QuadtreeNode::subdivide() {
    if (!isLeaf()) return;

    Cell child_cells[4];
    cell_.subdivideTo(child_cells);

    for (int i = 0; i < 4; ++i) {
        children_[i] = std::make_unique<QuadtreeNode>(child_cells[i].bounds, child_cells[i].resolution, depth_ + 1);
        children_[i]->getCell() = child_cells[i];
    }
}

void QuadtreeNode::insertPoint(const Point3D& pt, double timestamp) {
    if (!bounds_.contains(pt.x, pt.y)) return;

    if (isLeaf()) {
        cell_.addPoint(pt, timestamp);
    } else {
        int idx = getChildIndex(pt.x, pt.y);
        if (children_[idx]) {
            children_[idx]->insertPoint(pt, timestamp);
        }
    }
}

void QuadtreeNode::refineAt(float x, float y, int max_depth) {
    if (!bounds_.contains(x, y)) return;

    if (isLeaf()) {
        subdivide();
        return;
    }
    // A node at depth max_depth - 1 already holds the finest allowed cells.
    if (depth_ + 1 >= max_depth) return;
    int idx = getChildIndex(x, y);
    if (children_[idx]) {
        children_[idx]->refineAt(x, y, max_depth);
    }
}

Cell* QuadtreeNode::findLeaf(float x, float y) {
    if (!bounds_.contains(x, y)) return nullptr;

    if (isLeaf()) {
        return &cell_;
    }
    int idx = getChildIndex(x, y);
    if (children_[idx]) {
        return children_[idx]->findLeaf(x, y);
    }
    return nullptr;
}

void QuadtreeNode::getAllLeaves(std::vector<const Cell*>& out_cells) const {
    if (isLeaf()) {
        out_cells.push_back(&cell_);
    } else {
        for (int i = 0; i < 4; ++i) {
            if (children_[i]) {
                children_[i]->getAllLeaves(out_cells);
            }
        }
    }
}

void QuadtreeNode::getAllLeavesMutable(std::vector<Cell*>& out_cells) {
    if (isLeaf()) {
        out_cells.push_back(&cell_);
    } else {
        for (int i = 0; i < 4; ++i) {
            if (children_[i]) {
                children_[i]->getAllLeavesMutable(out_cells);
            }
        }
    }
}

size_t QuadtreeNode::countLeaves() const {
    if (isLeaf()) return 1;
    size_t count = 0;
    for (int i = 0; i < 4; ++i) {
        if (children_[i]) {
            count += children_[i]->countLeaves();
        }
    }
    return count;
}

// Quadtree Implementation
Quadtree::Quadtree(const BoundingBox2D& bounds, float base_resolution, int max_depth)
    : bounds_(bounds), base_resolution_(base_resolution), max_depth_(max_depth) {
    root_ = std::make_unique<QuadtreeNode>(bounds, base_resolution, 0);
}

void Quadtree::insertPoint(const Point3D& pt, double timestamp) {
    if (root_) {
        root_->insertPoint(pt, timestamp);
    }
}

void Quadtree::refineCellAt(float x, float y) {
    if (!root_ || !bounds_.contains(x, y)) return;
    root_->refineAt(x, y, max_depth_);
}

Cell* Quadtree::findLeaf(float x, float y) {
    if (!root_ || !bounds_.contains(x, y)) return nullptr;
    return root_->findLeaf(x, y);
}

std::vector<const Cell*> Quadtree::getActiveCells() const {
    std::vector<const Cell*> cells;
    if (root_) {
        root_->getAllLeaves(cells);
    }
    return cells;
}

std::vector<Cell*> Quadtree::getActiveCellsMutable() {
    std::vector<Cell*> cells;
    if (root_) {
        root_->getAllLeavesMutable(cells);
    }
    return cells;
}

size_t Quadtree::totalCells() const {
    return root_ ? root_->countLeaves() : 0;
}

} // namespace ps26053
