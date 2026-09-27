#include "ps26053/io/lidar_io.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace ps26053 {

std::vector<std::string> LidarIO::listSequenceScans(const std::string& dir) {
    std::vector<std::string> scans;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return scans; // not a directory: caller treats the path as one scan
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        auto ext = entry.path().extension().string();
        for (char& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".bin" || ext == ".pcd" || ext == ".ply") {
            scans.push_back(entry.path().string());
        }
    }
    std::sort(scans.begin(), scans.end());
    return scans;
}

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
        // Locked architecture palette (M1); UNKNOWN is grey (M3).
        uint8_t r = 16, g = 185, b = 129; // TERRAIN: emerald #10B981
        if (pt.semantic_class == SemanticClass::STATIC_OBSTACLE) {
            r = 0; g = 243; b = 255; // STATIC: cyan #00F3FF
        } else if (pt.semantic_class == SemanticClass::DYNAMIC_OBSTACLE) {
            r = 245; g = 158; b = 11; // DYNAMIC: amber #F59E0B
        } else if (pt.semantic_class == SemanticClass::UNKNOWN) {
            r = 128; g = 128; b = 128; // Grey: unclassified
        }

        out << pt.x << " " << pt.y << " " << pt.z << " "
            << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << " "
            << static_cast<int>(pt.semantic_class) << "\n";
    }

    return true;
}

bool LidarIO::loadPCD(const std::string& filepath, PointCloud& out_cloud) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[LidarIO] Failed to open PCD: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::string> fields;
    std::vector<int> sizes;
    std::vector<char> types;
    int num_points = 0;
    bool is_binary = false;
    int idx_x = -1, idx_y = -1, idx_z = -1, idx_i = -1;

    while (std::getline(file, line)) {
        // Strip trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "FIELDS") {
            std::string f;
            int idx = 0;
            while (iss >> f) {
                fields.push_back(f);
                if (f == "x") idx_x = idx;
                else if (f == "y") idx_y = idx;
                else if (f == "z") idx_z = idx;
                else if (f == "intensity" || f == "i") idx_i = idx;
                idx++;
            }
        } else if (tag == "SIZE") {
            int s;
            while (iss >> s) sizes.push_back(s);
        } else if (tag == "TYPE") {
            char t;
            while (iss >> t) types.push_back(t);
        } else if (tag == "POINTS") {
            iss >> num_points;
        } else if (tag == "DATA") {
            std::string data_mode;
            iss >> data_mode;
            if (data_mode == "binary") is_binary = true;
            break; // Header ends at DATA line
        }
    }

    if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
        // Fallback default: first 3 columns are x, y, z
        idx_x = 0; idx_y = 1; idx_z = 2;
        if (fields.size() >= 4) idx_i = 3;
    }

    out_cloud.clear();
    if (num_points > 0) out_cloud.reserve(num_points);

    if (!is_binary) {
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) tokens.push_back(token);

            if (tokens.size() <= static_cast<size_t>(std::max({idx_x, idx_y, idx_z}))) continue;

            Point3D pt;
            try {
                pt.x = std::stof(tokens[idx_x]);
                pt.y = std::stof(tokens[idx_y]);
                pt.z = std::stof(tokens[idx_z]);
                pt.intensity = (idx_i >= 0 && static_cast<size_t>(idx_i) < tokens.size()) ?
                               std::stof(tokens[idx_i]) : 0.5f;
            } catch (...) {
                continue;
            }
            pt.semantic_class = SemanticClass::UNKNOWN;
            pt.confidence = 0.0f;
            out_cloud.push_back(pt);
        }
    } else {
        // Compute total point record stride in bytes
        int point_stride = 0;
        if (!sizes.empty() && sizes.size() == fields.size()) {
            for (int s : sizes) point_stride += s;
        } else {
            point_stride = static_cast<int>(fields.size() * sizeof(float));
        }
        if (point_stride < 12) point_stride = 16; // default 4x float

        std::vector<char> pt_buf(point_stride);
        while (file.read(pt_buf.data(), point_stride)) {
            Point3D pt;
            const float* fptr = reinterpret_cast<const float*>(pt_buf.data());
            pt.x = fptr[idx_x];
            pt.y = fptr[idx_y];
            pt.z = fptr[idx_z];
            pt.intensity = (idx_i >= 0) ? fptr[idx_i] : 0.5f;
            pt.semantic_class = SemanticClass::UNKNOWN;
            pt.confidence = 0.0f;
            out_cloud.push_back(pt);
        }
    }

    return !out_cloud.empty();
}

bool LidarIO::loadPLY(const std::string& filepath, PointCloud& out_cloud) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[LidarIO] Failed to open PLY: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") {
        return false;
    }

    bool is_ascii = true;
    size_t num_vertices = 0;
    std::vector<std::string> prop_names;
    int idx_x = -1, idx_y = -1, idx_z = -1, idx_class = -1, idx_r = -1, idx_g = -1, idx_b = -1;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "format") {
            std::string fmt;
            iss >> fmt;
            if (fmt != "ascii") is_ascii = false;
        } else if (tag == "element") {
            std::string elem_type;
            size_t count = 0;
            iss >> elem_type >> count;
            if (elem_type == "vertex") num_vertices = count;
        } else if (tag == "property") {
            std::string prop_type, prop_name;
            iss >> prop_type >> prop_name;
            int cur_idx = static_cast<int>(prop_names.size());
            prop_names.push_back(prop_name);
            if (prop_name == "x") idx_x = cur_idx;
            else if (prop_name == "y") idx_y = cur_idx;
            else if (prop_name == "z") idx_z = cur_idx;
            else if (prop_name == "class_id" || prop_name == "label") idx_class = cur_idx;
            else if (prop_name == "red" || prop_name == "r") idx_r = cur_idx;
            else if (prop_name == "green" || prop_name == "g") idx_g = cur_idx;
            else if (prop_name == "blue" || prop_name == "b") idx_b = cur_idx;
        } else if (tag == "end_header") {
            break;
        }
    }

    if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
        idx_x = 0; idx_y = 1; idx_z = 2;
    }

    out_cloud.clear();
    if (num_vertices > 0) out_cloud.reserve(num_vertices);

    if (is_ascii) {
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) tokens.push_back(token);

            if (tokens.size() <= static_cast<size_t>(std::max({idx_x, idx_y, idx_z}))) continue;

            Point3D pt;
            try {
                pt.x = std::stof(tokens[idx_x]);
                pt.y = std::stof(tokens[idx_y]);
                pt.z = std::stof(tokens[idx_z]);
                pt.intensity = 0.5f;

                if (idx_class >= 0 && static_cast<size_t>(idx_class) < tokens.size()) {
                    int c = std::stoi(tokens[idx_class]);
                    if (c == 0) pt.semantic_class = SemanticClass::TERRAIN;
                    else if (c == 1) pt.semantic_class = SemanticClass::STATIC_OBSTACLE;
                    else if (c == 2) pt.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
                    else pt.semantic_class = SemanticClass::UNKNOWN;
                    pt.confidence = 1.0f;
                } else if (idx_r >= 0 && idx_g >= 0 && idx_b >= 0 &&
                           static_cast<size_t>(std::max({idx_r, idx_g, idx_b})) < tokens.size()) {
                    int r = std::stoi(tokens[idx_r]);
                    int g = std::stoi(tokens[idx_g]);
                    int b = std::stoi(tokens[idx_b]);
                    if (r > 200 && g > 200 && b < 100) { // yellow
                        pt.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
                        pt.confidence = 1.0f;
                    } else if (r < 50 && g > 200 && b > 200) { // cyan
                        pt.semantic_class = SemanticClass::STATIC_OBSTACLE;
                        pt.confidence = 1.0f;
                    } else if (g > 150 && r < 50) { // green
                        pt.semantic_class = SemanticClass::TERRAIN;
                        pt.confidence = 1.0f;
                    } else {
                        pt.semantic_class = SemanticClass::UNKNOWN;
                        pt.confidence = 0.0f;
                    }
                } else {
                    pt.semantic_class = SemanticClass::UNKNOWN;
                    pt.confidence = 0.0f;
                }
            } catch (...) {
                continue;
            }
            out_cloud.push_back(pt);
        }
    } else {
        // Binary little endian PLY (basic float coordinates)
        size_t stride = prop_names.size() * sizeof(float);
        if (stride < 12) stride = 12;
        std::vector<char> buf(stride);
        while (file.read(buf.data(), stride)) {
            const float* fptr = reinterpret_cast<const float*>(buf.data());
            Point3D pt;
            pt.x = fptr[idx_x];
            pt.y = fptr[idx_y];
            pt.z = fptr[idx_z];
            pt.intensity = 0.5f;
            pt.semantic_class = SemanticClass::UNKNOWN;
            pt.confidence = 0.0f;
            out_cloud.push_back(pt);
        }
    }

    return !out_cloud.empty();
}

bool LidarIO::loadPointCloud(const std::string& filepath, PointCloud& out_cloud) {
    std::string lower = filepath;
    for (char& c : lower) c = static_cast<char>(std::tolower(c));

    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".bin") {
        return loadBinScan(filepath, out_cloud);
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".pcd") {
        return loadPCD(filepath, out_cloud);
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".ply") {
        return loadPLY(filepath, out_cloud);
    }

    // Try in order: BIN -> PCD -> PLY
    if (loadBinScan(filepath, out_cloud) && !out_cloud.empty()) return true;
    if (loadPCD(filepath, out_cloud) && !out_cloud.empty()) return true;
    if (loadPLY(filepath, out_cloud) && !out_cloud.empty()) return true;

    return false;
}

} // namespace ps26053
