#pragma once

#include "ps26053/common/cell.hpp"
#include <memory>
#include <vector>

namespace ps26053 {

class QuadtreeNode {
public:
    QuadtreeNode(const BoundingBox2D& bounds, float resolution, int depth = 0);
    ~QuadtreeNode() = default;

    bool isLeaf() const { return children_[0] == nullptr; }
    void subdivide();
    void insertPoint(const Point3D& pt, double timestamp);
    void refineAt(float x, float y, int max_depth);
    Cell* findLeaf(float x, float y);

    Cell& getCell() { return cell_; }
    const Cell& getCell() const { return cell_; }

    void getAllLeaves(std::vector<const Cell*>& out_cells) const;
    void getAllLeavesMutable(std::vector<Cell*>& out_cells);

    int getDepth() const { return depth_; }
    size_t countLeaves() const;
    size_t countNodes() const;

private:
    BoundingBox2D bounds_;
    int depth_{0};
    Cell cell_;
    std::unique_ptr<QuadtreeNode> children_[4];

    int getChildIndex(float x, float y) const;
};

class Quadtree {
public:
    Quadtree(const BoundingBox2D& bounds, float base_resolution, int max_depth = 3);
    ~Quadtree() = default;

    void insertPoint(const Point3D& pt, double timestamp);
    void refineCellAt(float x, float y);
    Cell* findLeaf(float x, float y);

    std::vector<const Cell*> getActiveCells() const;
    std::vector<Cell*> getActiveCellsMutable();

    size_t totalCells() const;
    float baseResolution() const { return base_resolution_; }

    /**
     * @brief Measured memory footprint of this tree: the Quadtree object
     * itself plus every heap node plus every stored leaf Cell, all via
     * sizeof. Used by the honest benchmark memory model (C2).
     */
    size_t memoryBytes() const;

private:
    BoundingBox2D bounds_;
    float base_resolution_;
    int max_depth_;
    std::unique_ptr<QuadtreeNode> root_;
};

} // namespace ps26053
