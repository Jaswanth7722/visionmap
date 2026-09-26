#include "ps26053/mapping/grid_25d.hpp"
#include <fstream>
#include <cmath>
#include <iostream>

namespace ps26053 {

Grid25D::Grid25D(const GridConfig& config) : config_(config) {}

std::pair<int, int> Grid25D::getTileIndices(float x, float y) const {
    int ix = static_cast<int>(std::floor((x - config_.x_min) / config_.tile_size));
    int iy = static_cast<int>(std::floor((y - config_.y_min) / config_.tile_size));
    return {ix, iy};
}

void Grid25D::updateWithPointCloud(const PointCloud& cloud, double timestamp, const Eigen::Vector3f& sensor_pos) {
    for (const auto& pt : cloud) {
        if (pt.x < config_.x_min || pt.x >= config_.x_max ||
            pt.y < config_.y_min || pt.y >= config_.y_max) {
            continue;
        }

        float dist = std::hypot(pt.x - sensor_pos.x(), pt.y - sensor_pos.y());
        float base_res = policy_.getBaseResolution(dist);

        int ix = static_cast<int>(std::floor((pt.x - config_.x_min) / base_res));
        int iy = static_cast<int>(std::floor((pt.y - config_.y_min) / base_res));
        int64_t key = tileKey(ix, iy);

        auto it = tiles_.find(key);
        if (it == tiles_.end()) {
            float cell_min_x = config_.x_min + ix * base_res;
            float cell_max_x = cell_min_x + base_res;
            float cell_min_y = config_.y_min + iy * base_res;
            float cell_max_y = cell_min_y + base_res;

            BoundingBox2D cell_bounds{cell_min_x, cell_max_x, cell_min_y, cell_max_y};
            auto tree = std::make_unique<Quadtree>(cell_bounds, base_res);
            it = tiles_.emplace(key, std::move(tree)).first;
        }

        // Insert point
        it->second->insertPoint(pt, timestamp);

        // Check Level-2 refinement trigger
        if (pt.semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
            it->second->refineCellAt(pt.x, pt.y);
        }
    }
}

void Grid25D::updateTrackedObjects(const std::vector<TrackedObject>& tracks) {
    for (const auto& track : tracks) {
        if (!track.confirmed) continue;

        for (auto& [key, tree] : tiles_) {
            auto cells = tree->getActiveCellsMutable();
            for (auto* cell : cells) {
                if (track.bbox.contains(cell->bounds.centerX(), cell->bounds.centerY())) {
                    cell->object_id = track.id;
                    cell->velocity_x = track.velocity.x();
                    cell->velocity_y = track.velocity.y();
                    cell->semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
                    cell->importance = 1.0f;
                }
            }
        }
    }
}

void Grid25D::decayTemporal(double current_time, double max_staleness_sec) {
    for (auto& [key, tree] : tiles_) {
        auto cells = tree->getActiveCellsMutable();
        for (auto* cell : cells) {
            if (cell->occupancy > 0.01f && (current_time - cell->timestamp) > max_staleness_sec) {
                // Decay dynamic objects faster than terrain
                if (cell->semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
                    cell->occupancy *= 0.5f;
                } else {
                    cell->occupancy *= 0.85f;
                }
            }
        }
    }
}

std::vector<Cell> Grid25D::getAllCells() const {
    std::vector<Cell> result;
    for (const auto& [key, tree] : tiles_) {
        auto leaves = tree->getActiveCells();
        for (const auto* c : leaves) {
            result.push_back(*c);
        }
    }
    return result;
}

size_t Grid25D::totalActiveCells() const {
    size_t count = 0;
    for (const auto& [key, tree] : tiles_) {
        count += tree->totalCells();
    }
    return count;
}

bool Grid25D::exportToPLY(const std::string& filepath, bool occupied_only) const {
    std::vector<Cell> cells = getAllCells();
    std::vector<Cell> valid_cells;
    valid_cells.reserve(cells.size());

    for (const auto& c : cells) {
        if (!occupied_only || c.point_count > 0) {
            valid_cells.push_back(c);
        }
    }

    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment 2.5D World Model Map Export\n";
    out << "element vertex " << valid_cells.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "property float resolution\n";
    out << "property int class_id\n";
    out << "end_header\n";

    for (const auto& c : valid_cells) {
        float cx = c.bounds.centerX();
        float cy = c.bounds.centerY();
        float cz = c.elevation;

        uint8_t r = 90, g = 90, b = 90; // Default terrain: slate gray
        if (c.semantic_class == SemanticClass::STATIC_OBSTACLE) {
            r = 230; g = 50; b = 50; // Red
        } else if (c.semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
            r = 40; g = 220; b = 50; // Green
        }

        out << cx << " " << cy << " " << cz << " "
            << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << " "
            << c.resolution << " " << static_cast<int>(c.semantic_class) << "\n";
    }

    return true;
}

} // namespace ps26053
