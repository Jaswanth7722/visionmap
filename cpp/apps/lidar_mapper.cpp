#include "ps26053/io/lidar_io.hpp"
#include "ps26053/runtime/pipeline.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "==============================================================\n";
    std::cout << "  PS 26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping  \n";
    std::cout << "  DRDO Smart Vehicles | Native C++17/20 Runtime Pipeline      \n";
    std::cout << "==============================================================\n";

    std::string scan_arg = "data/raw";
    std::string model_path = "models/onnx/pointnet2_semseg.onnx";
    std::string output_ply = "results/maps/adaptive_map_frame000000.ply";
    size_t max_frames = 0; // 0 = all frames found

    if (argc > 1) scan_arg = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) output_ply = argv[3];
    if (argc > 4) max_frames = static_cast<size_t>(std::stoul(argv[4]));

    // Single scan or sequence directory (e.g. SemanticKITTI velodyne/).
    // With only data/raw/000000.bin present this is one frame; attach a
    // sequence directory to stream many. Frame timestamps assume the nominal
    // 10 Hz from config/lidar.yaml (rate_hz) — stated, not measured.
    constexpr double kFrameIntervalSec = 0.1;
    std::vector<std::string> scans = ps26053::LidarIO::listSequenceScans(scan_arg);
    if (scans.empty()) scans.push_back(scan_arg);
    if (max_frames > 0 && scans.size() > max_frames) scans.resize(max_frames);

    std::cout << "[Ingest] " << scans.size() << " scan(s) from: " << scan_arg << "\n";

    // 2. Initialize pipeline
    ps26053::MappingPipeline pipeline(model_path);
    if (!pipeline.initialize()) {
        std::cerr << "[Warn] Proceeding with heuristic perception engine...\n";
    }
    // H3: behavior comes from config/*.yaml (compiled defaults apply loudly
    // only when the files are missing or malformed).
    pipeline.loadConfig("config");

    // 3. Process frames (grid/tracker state persists across frames)
    ps26053::FrameMetrics metrics;
    for (size_t f = 0; f < scans.size(); ++f) {
        ps26053::PointCloud raw_scan;
        std::cout << "[Ingest] Loading raw LiDAR scan: " << scans[f] << " ...\n";
        if (!ps26053::LidarIO::loadBinScan(scans[f], raw_scan)) {
            // No sample data ships with the repo: this path needs a real
            // .bin scan or sequence directory as the first argument.
            std::cerr << "[Error] Failed to load LiDAR scan!\n"
                      << "Usage: lidar_mapper <scan.bin|sequence_dir> [model.onnx] [output.ply] [max_frames]\n";
            return 1;
        }
        std::cout << "[Ingest] Loaded " << raw_scan.size() << " points.\n";

        std::cout << "[Pipeline] Running frame " << f << " end-to-end...\n";
        metrics = pipeline.processFrame(raw_scan, f * kFrameIntervalSec, static_cast<int>(f));

        // 4. Output metrics
        std::cout << "\n--------------------------------------------------------------\n";
        std::cout << " FRAME METRICS (Frame #" << metrics.frame_index << ")\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Input Points:       " << metrics.input_points << "\n";
        std::cout << "  Active 2.5D Cells:  " << metrics.active_cells << "\n";
        std::cout << "  Tracked Objects:    " << metrics.active_tracks << "\n";
        std::cout << "  Preprocessing:      " << metrics.preprocess_time_ms << " ms\n";
        std::cout << "  PointNet++ Infer:   " << metrics.inference_time_ms << " ms\n";
        std::cout << "  Network Labels:     " << metrics.inference_network_points << " pts\n";
        std::cout << "  Fallback Labels:    " << metrics.inference_fallback_points << " pts\n";
        std::cout << "  Tracking Update:    " << metrics.tracking_time_ms << " ms\n";
        std::cout << "  2.5D Map Update:    " << metrics.mapping_time_ms << " ms\n";
        std::cout << "  Total Latency:      " << metrics.total_time_ms << " ms\n";
        std::cout << "  Frame Rate (FPS):   " << metrics.fps << " Hz\n";
        std::cout << "--------------------------------------------------------------\n";
    }

    // 5. Export accumulated 2.5D map
    std::cout << "[Export] Saving 2.5D world model to: " << output_ply << " ...\n";
    if (pipeline.getGrid().exportToPLY(output_ply)) {
        std::cout << "[Success] 2.5D Map export complete.\n";
    }

    return 0;
}
