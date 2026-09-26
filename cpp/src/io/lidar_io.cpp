#include "ps26053/io/lidar_io.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace ps26053 {

bool LidarIO::loadBinScan(const std::string& filepath, PointCloud& out_cloud) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[LidarIO] Failed to open: " << filepath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_points = file_size / (4 * sizeof(float));
    std::vector<float> buffer(num_points * 4);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    out_cloud.clear();
    out_cloud.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        Point3D pt;
        pt.x = buffer[i * 4 + 0];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
        pt.intensity = buffer[i * 4 + 3];
        pt.semantic_class = SemanticClass::UNKNOWN;
        pt.confidence = 0.0f;
        out_cloud.push_back(pt);
    }

    return true;
}

bool LidarIO::loadLabels(const std::string& filepath, PointCloud& in_out_cloud,
                       const std::string& remap_path) {
    // Load the shared native-id -> project-class table (H7). Any problem
    // with it is fatal: guessing a label mapping silently is exactly how the
    // old hardcoded 0/1/2 table diverged from the Python one.
    std::ifstream remap_file(remap_path);
    if (!remap_file.is_open()) {
        std::cerr << "[LidarIO] Label remap table not found: " << remap_path << std::endl;
        return false;
    }
    std::unordered_map<uint16_t, int> remap;
    std::string line;
    size_t lineno = 0;
    while (std::getline(remap_file, line)) {
        ++lineno;
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        std::istringstream fields(line.substr(start));
        int raw_id = -1, class_id = -2;
        if (!(fields >> raw_id >> class_id) || raw_id < 0 || raw_id > 0xFFFF ||
            (class_id != -1 && class_id != 0 && class_id != 1 && class_id != 2)) {
            std::cerr << "[LidarIO] Malformed remap table " << remap_path
                      << ":" << lineno << std::endl;
            return false;
        }
        remap[static_cast<uint16_t>(raw_id)] = class_id;
    }
    if (remap.size() < 30) {
        std::cerr << "[LidarIO] Remap table " << remap_path << " looks truncated ("
                  << remap.size() << " entries)." << std::endl;
        return false;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_labels = file_size / sizeof(uint32_t);
    if (num_labels != in_out_cloud.size()) {
        std::cerr << "[LidarIO] Label count mismatch: " << num_labels << " vs " << in_out_cloud.size() << std::endl;
        return false;
    }

    std::vector<uint32_t> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    size_t unmapped = 0;
    for (size_t i = 0; i < num_labels; ++i) {
        uint16_t sem_label = static_cast<uint16_t>(buffer[i] & 0xFFFF);
        auto it = remap.find(sem_label);
        if (it == remap.end()) {
            // Native id absent from the shared table: admit ignorance rather
            // than guessing a class.
            in_out_cloud[i].semantic_class = SemanticClass::UNKNOWN;
            in_out_cloud[i].confidence = 0.0f;
            ++unmapped;
            continue;
        }
        switch (it->second) {
            case 0:
                in_out_cloud[i].semantic_class = SemanticClass::TERRAIN;
                in_out_cloud[i].confidence = 1.0f;
                break;
            case 1:
                in_out_cloud[i].semantic_class = SemanticClass::STATIC_OBSTACLE;
                in_out_cloud[i].confidence = 1.0f;
                break;
            case 2:
                in_out_cloud[i].semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
                in_out_cloud[i].confidence = 1.0f;
                break;
            default: // -1: ignore
                in_out_cloud[i].semantic_class = SemanticClass::UNKNOWN;
                in_out_cloud[i].confidence = 0.0f;
                break;
        }
    }
    if (unmapped > 0) {
        std::cerr << "[LidarIO] " << unmapped << " labels not present in " << remap_path
                  << "; tagged UNKNOWN." << std::endl;
    }

    return true;
}

bool LidarIO::writePLY(const std::string& filepath, const PointCloud& cloud) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    // Full float precision so exported coordinates round-trip exactly.
    out << std::setprecision(10);

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment PS26053 PointCloud PLY\n";
    out << "element vertex " << cloud.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "property int class_id\n";
    out << "end_header\n";

    for (const auto& pt : cloud) {
        uint8_t r = 120, g = 120, b = 120;
        if (pt.semantic_class == SemanticClass::TERRAIN) {
            r = 140; g = 140; b = 140; // Terrain: Gray
        } else if (pt.semantic_class == SemanticClass::STATIC_OBSTACLE) {
            r = 230; g = 50; b = 50; // Static: Red
        } else if (pt.semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
            r = 40; g = 220; b = 60; // Dynamic: Green
        }

        out << pt.x << " " << pt.y << " " << pt.z << " "
            << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << " "
            << static_cast<int>(pt.semantic_class) << "\n";
    }

    return true;
}

} // namespace ps26053
