#include "ps26053/mapping/grid_25d.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_set>

namespace ps26053 {

Grid25D::Grid25D(const GridConfig& config) : config_(config) {}

std::pair<int, int> Grid25D::getTileIndices(float x, float y) const {
    int ix = static_cast<int>(std::floor((x - config_.x_min) / config_.tile_size));
    int iy = static_cast<int>(std::floor((y - config_.y_min) / config_.tile_size));
    return {ix, iy};
}

void Grid25D::updateWithPointCloud(const PointCloud& cloud, double timestamp, const Eigen::Vector3f& sensor_pos) {
    constexpr float q = ResolutionPolicy::alignmentQuantum();

    for (const auto& pt : cloud) {
        if (pt.x < config_.x_min || pt.x >= config_.x_max ||
            pt.y < config_.y_min || pt.y >= config_.y_max) {
            continue;
        }

        // Deterministic point-to-cell mapping (alignment rule 3): quantize the
        // point to the shared 5 cm microcell lattice first, then select the
        // distance band from the microcell centre. Every point inside one
        // microcell therefore maps to the same band cell, and every stored
        // cell is an exact union of microcells, so stored footprints from
        // different bands are disjoint by construction.
        int mx = static_cast<int>(std::floor((pt.x - config_.x_min) / q));
        int my = static_cast<int>(std::floor((pt.y - config_.y_min) / q));
        float mcx = config_.x_min + (static_cast<float>(mx) + 0.5f) * q;
        float mcy = config_.y_min + (static_cast<float>(my) + 0.5f) * q;

        float dist = std::hypot(mcx - sensor_pos.x(), mcy - sensor_pos.y());
        float base_res = policy_.getBaseResolution(dist);
        int k = policy_.quantumMultiple(base_res);

        int64_t mkey = microKey(mx, my);
        auto owner = micro_owner_.find(mkey);
        if (owner != micro_owner_.end()) {
            // This microcell already belongs to a stored cell: route the
            // point into its owner instead of allocating an overlapping cell.
            auto ot = tiles_.find(owner->second);
            if (ot != tiles_.end()) {
                ot->second->insertPoint(pt, timestamp);
                refineByImportance(*ot->second, pt, sensor_pos);
            }
            continue;
        }

        int cx, cy;
        float cell_min_x, cell_min_y;
        if (k > 0) {
            cx = floorDiv(mx, k);
            cy = floorDiv(my, k);
            cell_min_x = config_.x_min + static_cast<float>(cx * k) * q;
            cell_min_y = config_.y_min + static_cast<float>(cy * k) * q;
        } else {
            // Custom non-multiple band: fall back to per-point indexing, but
            // keep the resolution tag in the key so bands cannot alias.
            k = 0;
            cx = static_cast<int>(std::floor((pt.x - config_.x_min) / base_res));
            cy = static_cast<int>(std::floor((pt.y - config_.y_min) / base_res));
            cell_min_x = config_.x_min + static_cast<float>(cx) * base_res;
            cell_min_y = config_.y_min + static_cast<float>(cy) * base_res;
        }

        if (k > 0) {
            // Refuse to allocate over microcells owned by another band's
            // cells. Contested microcells degrade gracefully to a single
            // quantum cell covering exactly this free microcell.
            bool contested = false;
            for (int ix = cx * k; ix < cx * k + k && !contested; ++ix) {
                for (int iy = cy * k; iy < cy * k + k; ++iy) {
                    if (micro_owner_.find(microKey(ix, iy)) != micro_owner_.end()) {
                        contested = true;
                        break;
                    }
                }
            }
            if (contested) {
                k = 1;
                cx = mx;
                cy = my;
                cell_min_x = config_.x_min + static_cast<float>(cx) * q;
                cell_min_y = config_.y_min + static_cast<float>(cy) * q;
            }
        }
        int64_t key = cellKey(k, cx, cy);

        auto it = tiles_.find(key);
        if (it == tiles_.end()) {
            float cell_max_x = cell_min_x + (k > 0 ? static_cast<float>(k) * q : base_res);
            float cell_max_y = cell_min_y + (k > 0 ? static_cast<float>(k) * q : base_res);

            BoundingBox2D cell_bounds{cell_min_x, cell_max_x, cell_min_y, cell_max_y};
            auto tree = std::make_unique<Quadtree>(
                cell_bounds, k > 0 ? static_cast<float>(k) * q : base_res, config_.max_depth);
            it = tiles_.emplace(key, std::move(tree)).first;

            if (k > 0) {
                for (int ix = cx * k; ix < cx * k + k; ++ix) {
                    for (int iy = cy * k; iy < cy * k + k; ++iy) {
                        micro_owner_.emplace(microKey(ix, iy), key);
                    }
                }
            }
        }

        // Insert point
        it->second->insertPoint(pt, timestamp);

        // Level-2 refinement driven by the real importance engine
        // (distance + motion + semantics), not a hardcoded class check.
        refineByImportance(*it->second, pt, sensor_pos);
    }
}

void Grid25D::refineByImportance(Quadtree& tree, const Point3D& pt, const Eigen::Vector3f& sensor_pos) {
    Cell* leaf = tree.findLeaf(pt.x, pt.y);
    if (!leaf) return;

    float dist = std::hypot(leaf->bounds.centerX() - sensor_pos.x(),
                            leaf->bounds.centerY() - sensor_pos.y());
    leaf->importance = policy_.computeImportance(*leaf, dist);
    if (policy_.shouldRefine(*leaf, dist)) {
        tree.refineCellAt(pt.x, pt.y);
    }
}

BoundaryQA Grid25D::checkBoundaryAlignment() const {
    constexpr float q = ResolutionPolicy::alignmentQuantum();
    constexpr float eps = 1e-4f;
    // Strict interior intersection epsilon: shared edges must NOT count as
    // overlap, so this is far above float rounding noise but far below any
    // real cell dimension.
    constexpr float overlap_eps = 1e-6f;
    BoundaryQA qa;

    auto cells = getAllCells();
    qa.checked_cells = cells.size();

    auto interiorsIntersect = [&](const Cell& a, const Cell& b) {
        return (a.bounds.min_x < b.bounds.max_x - overlap_eps &&
                b.bounds.min_x < a.bounds.max_x - overlap_eps &&
                a.bounds.min_y < b.bounds.max_y - overlap_eps &&
                b.bounds.min_y < a.bounds.max_y - overlap_eps);
    };

    // Microcell rasterization finds candidate collisions cheaply; each
    // candidate pair is then verified against the exact stored footprints so
    // that cells sharing only an edge are never miscounted.
    //
    // NOTE: micro indices are computed in double precision. In float32 the
    // quotient (coord - origin) / quantum is O(1000) with only ~7 decimal
    // digits, i.e. absolute rounding noise up to ~1e-4 — the same scale as
    // the eps margin, which made the check blind. Double precision keeps the
    // index error at ~1e-13, far below eps.
    const double qd = static_cast<double>(q);
    const double x_origin = static_cast<double>(config_.x_min);
    const double y_origin = static_cast<double>(config_.y_min);
    std::unordered_map<int64_t, size_t> owner;
    owner.reserve(cells.size() * 4);

    for (size_t ci = 0; ci < cells.size(); ++ci) {
        const auto& c = cells[ci];
        float w = c.bounds.width();
        float h = c.bounds.height();
        if (std::fabs(w - c.resolution) > eps || std::fabs(h - c.resolution) > eps) {
            ++qa.misaligned_cells;
            continue;
        }

        // Half-open microcell range covered by this footprint.
        int x0 = static_cast<int>(std::floor((static_cast<double>(c.bounds.min_x) - x_origin) / qd + eps));
        int x1 = static_cast<int>(std::ceil((static_cast<double>(c.bounds.max_x) - x_origin) / qd - eps)) - 1;
        int y0 = static_cast<int>(std::floor((static_cast<double>(c.bounds.min_y) - y_origin) / qd + eps));
        int y1 = static_cast<int>(std::ceil((static_cast<double>(c.bounds.max_y) - y_origin) / qd - eps)) - 1;

        for (int ix = x0; ix <= x1; ++ix) {
            for (int iy = y0; iy <= y1; ++iy) {
                int64_t key = (static_cast<int64_t>(ix) << 32) |
                              (static_cast<int64_t>(iy) & 0xFFFFFFFFLL);
                auto it = owner.find(key);
                if (it == owner.end()) {
                    owner.emplace(key, ci);
                } else if (it->second != ci && interiorsIntersect(cells[it->second], c)) {
                    ++qa.overlapping_quanta;
                }
            }
        }
    }

    return qa;
}

void Grid25D::updateTrackedObjects(const std::vector<TrackedObject>& tracks) {
    for (const auto& track : tracks) {
        if (!track.confirmed) continue;

        for (auto& [key, tree] : tiles_) {
            auto cells = tree->getActiveCellsMutable();
            for (auto* cell : cells) {
                if (track.bbox.contains(cell->bounds.centerX(), cell->bounds.centerY())) {
                    // Associate track identity and motion, but do NOT
                    // overwrite the network's semantic classification: a
                    // parked vehicle or wall inside a track box keeps its
                    // predicted class. Only cells the network left UNKNOWN
                    // adopt the track's dynamic label.
                    cell->object_id = track.id;
                    cell->velocity_x = track.velocity.x();
                    cell->velocity_y = track.velocity.y();
                    if (cell->semantic_class == SemanticClass::UNKNOWN) {
                        cell->semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
                    }
                    cell->importance = 1.0f;
                }
            }
        }
    }
}

void Grid25D::decayTemporal(double current_time, double max_staleness_sec) {
    // Below this occupancy a cell no longer represents anything: clearing it
    // removes departed obstacles instead of letting stale elevation and
    // semantics persist forever as ghost geometry.
    constexpr float kClearOccupancy = 0.05f;
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
                if (cell->occupancy < kClearOccupancy) {
                    BoundingBox2D bounds = cell->bounds;
                    float resolution = cell->resolution;
                    *cell = Cell();
                    cell->bounds = bounds;
                    cell->resolution = resolution;
                    cell->timestamp = current_time;
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

size_t Grid25D::estimateMemoryBytes() const {
    constexpr size_t kMapNodeBytes = 40; // approximate per-entry node cost, see header
    size_t total = sizeof(Grid25D) + tiles_.bucket_count() * sizeof(void*);
    for (const auto& [key, tree] : tiles_) {
        total += kMapNodeBytes + tree->memoryBytes();
    }
    // Microcell ownership index (the structure that guarantees disjoint
    // footprints): bucket array exactly, entries approximately.
    total += micro_owner_.bucket_count() * sizeof(void*) +
             micro_owner_.size() * kMapNodeBytes;
    return total;
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

    // Full float precision: the default 6 significant digits would quantize
    // coordinates to 1e-4 at these magnitudes and fabricate apparent
    // boundary overlaps in the artifact.
    out << std::setprecision(10);

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
