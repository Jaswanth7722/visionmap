#include "ps26053/io/lidar_io.hpp"
#include "ps26053/runtime/pipeline.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "==============================================================\n";
    std::cout << "  PS 26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping  \n";
    std::cout << "  DRDO Smart Vehicles | Native C++17/20 Runtime Pipeline      \n";
    std::cout << "==============================================================\n";

    std::string bin_path = "data/raw/000000.bin";
    std::string model_path = "models/onnx/pointnet2_semseg.onnx";
    std::string output_ply = "results/maps/adaptive_map_frame000000.ply";

    if (argc > 1) bin_path = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) output_ply = argv[3];

    // 1. Ingest raw scan
    ps26053::PointCloud raw_scan;
    std::cout << "[Ingest] Loading raw LiDAR scan: " << bin_path << " ...\n";
    if (!ps26053::LidarIO::loadBinScan(bin_path, raw_scan)) {
        std::cerr << "[Error] Failed to load LiDAR scan!\n";
        return 1;
    }
    std::cout << "[Ingest] Loaded " << raw_scan.size() << " points.\n";

    // 2. Initialize pipeline
    ps26053::MappingPipeline pipeline(model_path);
    if (!pipeline.initialize()) {
        std::cerr << "[Warn] Proceeding with heuristic perception engine...\n";
    }
    // H3: behavior comes from config/*.yaml (compiled defaults apply loudly
    // only when the files are missing or malformed).
    pipeline.loadConfig("config");

    // 3. Process frame
    std::cout << "[Pipeline] Running per-frame end-to-end processing...\n";
    double timestamp = 0.0;
    auto metrics = pipeline.processFrame(raw_scan, timestamp, 0);

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

    // 5. Export 2.5D map
    std::cout << "[Export] Saving 2.5D world model to: " << output_ply << " ...\n";
    if (pipeline.getGrid().exportToPLY(output_ply)) {
        std::cout << "[Success] 2.5D Map export complete.\n";
    }

    return 0;
}
