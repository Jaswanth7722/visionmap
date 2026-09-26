#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace ps26053 {

enum class SemanticClass : uint8_t {
    TERRAIN = 0,
    STATIC_OBSTACLE = 1,
    DYNAMIC_OBSTACLE = 2,
    UNKNOWN = 255
};

inline std::string semanticClassName(SemanticClass sc) {
    switch (sc) {
        case SemanticClass::TERRAIN: return "terrain";
        case SemanticClass::STATIC_OBSTACLE: return "static_obstacle";
        case SemanticClass::DYNAMIC_OBSTACLE: return "dynamic_obstacle";
        default: return "unknown";
    }
}

struct Point3D {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float intensity{0.0f};
    SemanticClass semantic_class{SemanticClass::UNKNOWN};
    float confidence{0.0f};
    int32_t cluster_id{-1};

    Eigen::Vector3f toEigen() const { return Eigen::Vector3f(x, y, z); }
};

using PointCloud = std::vector<Point3D>;

struct BoundingBox2D {
    float min_x{0.0f};
    float max_x{0.0f};
    float min_y{0.0f};
    float max_y{0.0f};

    bool contains(float x, float y) const {
        return (x >= min_x && x < max_x && y >= min_y && y < max_y);
    }

    float width() const { return max_x - min_x; }
    float height() const { return max_y - min_y; }
    float centerX() const { return 0.5f * (min_x + max_x); }
    float centerY() const { return 0.5f * (min_y + max_y); }
};

struct TrackedObject {
    int32_t id{-1};
    Eigen::Vector2f position{0.0f, 0.0f};
    Eigen::Vector2f velocity{0.0f, 0.0f};
    SemanticClass semantic_class{SemanticClass::DYNAMIC_OBSTACLE};
    float confidence{0.0f};
    BoundingBox2D bbox;
    int32_t hits{0};
    int32_t misses{0};
    bool confirmed{false};
};

} // namespace ps26053
