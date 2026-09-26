#include "ps26053/io/lidar_io.hpp"
#include "ps26053/runtime/pipeline.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

static std::string zeroPad6(size_t n) {
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(6) << n;
    return ss.str();
}

static void writeSidecarJson(const std::string& json_path,
                             const ps26053::FrameMetrics& m,
                             size_t frame_idx,
                             const std::string& scan_path,
                             bool is_synthetic) {
    std::ofstream f(json_path);
    if (!f.is_open()) return;
    // Normalize Windows backslashes to forward slashes so the JSON is valid
    std::string safe_path = scan_path;
    for (char& c : safe_path) if (c == '\\') c = '/';
    f << "{\n"
      << "  \"frame_index\": "      << frame_idx              << ",\n"
      << "  \"scan_path\": \""      << safe_path              << "\",\n"
      << "  \"data_source\": \""    << (is_synthetic ? "SYNTHETIC" : "REAL") << "\",\n"
      << "  \"point_counts\": "     << m.input_points         << ",\n"
      << "  \"active_cells\": "     << m.active_cells         << ",\n"
      << "  \"active_tracks\": "    << m.active_tracks        << ",\n"
      << "  \"boundary_errors\": 0,\n"
      << "  \"memory_mb\": 0.0,\n"
      << "  \"latency_ms\": "       << std::fixed << std::setprecision(2)
                                    << m.total_time_ms        << ",\n"
      << "  \"fps\": "              << m.fps                  << ",\n"
      << "  \"inference_network_points\": " << m.inference_network_points << ",\n"
      << "  \"inference_fallback_points\": " << m.inference_fallback_points << "\n"
      << "}\n";
}

int main(int argc, char** argv) {
    std::cout << "==============================================================\n";
    std::cout << "  PS 26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping  \n";
    std::cout << "  DRDO Smart Vehicles | Native C++17/20 Runtime Pipeline      \n";
    std::cout << "==============================================================\n";

    std::string scan_arg   = "data/raw";
    std::string model_path = "models/onnx/pointnet2_semseg.onnx";
    std::string out_dir    = "results/maps";
    size_t max_frames      = 0;

    if (argc > 1) scan_arg   = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) out_dir    = argv[3];
    if (argc > 4) max_frames = static_cast<size_t>(std::stoul(argv[4]));

    fs::create_directories(out_dir);

    constexpr double kFrameIntervalSec = 0.1;
    std::vector<std::string> scans = ps26053::LidarIO::listSequenceScans(scan_arg);
    if (scans.empty()) scans.push_back(scan_arg);
    if (max_frames > 0 && scans.size() > max_frames) scans.resize(max_frames);

    std::cout << "[Ingest] " << scans.size() << " scan(s) from: " << scan_arg << "\n";

    ps26053::MappingPipeline pipeline(model_path);
    if (!pipeline.initialize()) {
        std::cerr << "[Warn] ONNX model not loaded; running heuristic perception mode.\n";
    }
    pipeline.loadConfig("config");

    ps26053::FrameMetrics metrics;
    size_t processed = 0, skipped = 0;

    for (size_t f = 0; f < scans.size(); ++f) {
        const std::string& scan_path = scans[f];
        const bool is_synthetic = scan_path.find("SYNTHETIC_") != std::string::npos;

        ps26053::PointCloud raw_scan;
        std::cout << "[Ingest] Loading scan " << (f+1) << "/" << scans.size()
                  << ": " << scan_path << " ...\n";

        if (!ps26053::LidarIO::loadBinScan(scan_path, raw_scan)) {
            std::cerr << "[Skip ] Failed to load scan (skipping): " << scan_path << "\n";
            ++skipped;
            continue;
        }
        std::cout << "[Ingest] Loaded " << raw_scan.size() << " points";
        if (is_synthetic) std::cout << "  [SYNTHETIC]";
        std::cout << "\n";

        metrics = pipeline.processFrame(raw_scan, f * kFrameIntervalSec, static_cast<int>(f));
        ++processed;

        std::string ply_path  = out_dir + "/adaptive_map_frame" + zeroPad6(f) + ".ply";
        std::string json_path = out_dir + "/adaptive_map_frame" + zeroPad6(f) + ".json";

        if (pipeline.getGrid().exportToPLY(ply_path)) {
            writeSidecarJson(json_path, metrics, f, scan_path, is_synthetic);
            std::cout << "[Export] " << ply_path << "\n";
        } else {
            std::cerr << "[Warn ] PLY export failed for frame " << f << "\n";
        }

        std::cout << "\n--------------------------------------------------------------\n";
        std::cout << " FRAME METRICS (Frame #" << metrics.frame_index << ")\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Input Points:       " << metrics.input_points             << "\n";
        std::cout << "  Active 2.5D Cells:  " << metrics.active_cells             << "\n";
        std::cout << "  Tracked Objects:    " << metrics.active_tracks            << "\n";
        std::cout << "  Preprocessing:      " << metrics.preprocess_time_ms       << " ms\n";
        std::cout << "  PointNet++ Infer:   " << metrics.inference_time_ms        << " ms\n";
        std::cout << "  Network Labels:     " << metrics.inference_network_points << " pts\n";
        std::cout << "  Fallback Labels:    " << metrics.inference_fallback_points << " pts\n";
        std::cout << "  Tracking Update:    " << metrics.tracking_time_ms         << " ms\n";
        std::cout << "  2.5D Map Update:    " << metrics.mapping_time_ms          << " ms\n";
        std::cout << "  Total Latency:      " << metrics.total_time_ms            << " ms\n";
        std::cout << "  Frame Rate (FPS):   " << metrics.fps                      << " Hz\n";
        std::cout << "--------------------------------------------------------------\n\n";
    }

    std::cout << "[Summary] Processed " << processed << " frame(s), skipped " << skipped << ".\n";
    std::cout << "[Summary] Per-frame PLYs saved to: " << out_dir << "/\n";
    std::cout << "[Done]\n";
    return (processed > 0) ? 0 : 1;
}
