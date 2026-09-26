#include "ps26053/io/lidar_io.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    std::cout << "[Test] Running test_io ...\n";

    // H7: C++ must apply the SAME shared remap table as Python
    // (config/semantickitti_remap.txt), not a hardcoded 0/1/2 guess.
    // CTest runs with CWD = build dir, hence the relative path below.
    const std::string remap_path = "../config/semantickitti_remap.txt";

    ps26053::PointCloud cloud(5); // 5 points; xyz irrelevant for label loading
    const std::vector<uint32_t> raw_labels = {
        40u,   // road -> TERRAIN
        50u,   // building -> STATIC_OBSTACLE
        10u,   // car -> DYNAMIC_OBSTACLE
        999u,  // absent from the table -> UNKNOWN
        0u     // unlabeled -> UNKNOWN (ignore)
    };
    const std::string lbl_path = "test_io_tmp.label";
    {
        std::ofstream out(lbl_path, std::ios::binary);
        assert(out.is_open());
        out.write(reinterpret_cast<const char*>(raw_labels.data()),
                  static_cast<std::streamsize>(raw_labels.size() * sizeof(uint32_t)));
    }

    bool ok = ps26053::LidarIO::loadLabels(lbl_path, cloud, remap_path);
    assert(ok);
    assert(cloud[0].semantic_class == ps26053::SemanticClass::TERRAIN);
    assert(cloud[1].semantic_class == ps26053::SemanticClass::STATIC_OBSTACLE);
    assert(cloud[2].semantic_class == ps26053::SemanticClass::DYNAMIC_OBSTACLE);
    assert(cloud[3].semantic_class == ps26053::SemanticClass::UNKNOWN);
    assert(cloud[3].confidence == 0.0f);
    assert(cloud[4].semantic_class == ps26053::SemanticClass::UNKNOWN);
    assert(cloud[0].confidence == 1.0f);
    std::remove(lbl_path.c_str());

    // Count mismatch must fail loudly, not misalign labels.
    ps26053::PointCloud short_cloud(3);
    {
        std::ofstream out(lbl_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(raw_labels.data()),
                  static_cast<std::streamsize>(raw_labels.size() * sizeof(uint32_t)));
    }
    assert(!ps26053::LidarIO::loadLabels(lbl_path, short_cloud, remap_path));
    std::remove(lbl_path.c_str());

    // Missing remap file must fail loudly, never fall back to a guess.
    {
        std::ofstream out(lbl_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(raw_labels.data()),
                  static_cast<std::streamsize>(raw_labels.size() * sizeof(uint32_t)));
    }
    ps26053::PointCloud cloud2(5);
    assert(!ps26053::LidarIO::loadLabels(lbl_path, cloud2, "nonexistent_remap.txt"));
    std::remove(lbl_path.c_str());

    // Sequence listing: sorted .bin paths, non-bin files ignored, missing
    // dir yields empty (caller falls back to single-scan mode).
    {
        const std::string seq_dir = "test_io_tmp_seq";
        std::filesystem::create_directories(seq_dir);
        for (const char* name : {"000002.bin", "000000.bin", "notes.txt", "000001.bin"}) {
            std::ofstream f(std::string(seq_dir) + "/" + name, std::ios::binary);
            float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (std::string(name).size() > 4 &&
                std::string(name).substr(std::string(name).size() - 4) == ".bin") {
                f.write(reinterpret_cast<const char*>(zero), sizeof(zero));
            } else {
                f << "not a scan";
            }
        }
        auto scans = ps26053::LidarIO::listSequenceScans(seq_dir);
        assert(scans.size() == 3);
        assert(scans[0].find("000000.bin") != std::string::npos);
        assert(scans[1].find("000001.bin") != std::string::npos);
        assert(scans[2].find("000002.bin") != std::string::npos);
        assert(ps26053::LidarIO::listSequenceScans("nonexistent_seq_dir").empty());
        std::filesystem::remove_all(seq_dir);
    }

    std::cout << "[Test PASS] test_io succeeded!\n";
    return 0;
}
