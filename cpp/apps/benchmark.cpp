#include "ps26053/io/lidar_io.hpp"
#include "ps26053/runtime/pipeline.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <unordered_set>

namespace {
// The verdict is derived from the measured adaptive error count, never
// hardcoded: any nonzero count is reported as a failure. The uniform column
// is n/a because Mode 1 counts occupied keys without materializing stored
// cells, so there is no stored-cell geometry to check.
std::string boundaryVerdict(size_t adaptive_errors) {
    return (adaptive_errors == 0) ? "0 Errors (PASS)" : "NONZERO (FAIL)";
}
} // namespace

int main(int argc, char** argv) {
    std::cout << "==============================================================\n";
    std::cout << "  PS 26053: UNIFORM vs ADAPTIVE MAPPING BENCHMARK              \n";
    std::cout << "  Empirical Proof-of-Value Verification Execution             \n";
    std::cout << "==============================================================\n";

    std::string bin_path = "data/raw/000000.bin";
    std::string model_path = "models/onnx/pointnet2_semseg.onnx";

    if (argc > 1) bin_path = argv[1];
    if (argc > 2) model_path = argv[2];

    ps26053::PointCloud raw_scan;
    if (!ps26053::LidarIO::loadBinScan(bin_path, raw_scan)) {
        std::cerr << "[Benchmark Error] Failed to load scan: " << bin_path << std::endl;
        return 1;
    }

    std::cout << "[Benchmark] Input scan: " << bin_path << " (" << raw_scan.size() << " points)\n\n";

    // --- MODE 2 FIRST: the adaptive pipeline defines the shared input. ---
    // Mode 1 then counts uniform cells over the identical post-ROI,
    // post-voxel cloud the mapping stage actually receives — not the raw scan.
    std::cout << "--- Running Mode 2: Adaptive Variable-Resolution (Dual-Level Architecture) ---\n";
    ps26053::MappingPipeline adaptive_pipeline(model_path);
    adaptive_pipeline.initialize();

    auto metrics = adaptive_pipeline.processFrame(raw_scan, 0.0, 0);
    size_t adaptive_cell_count = metrics.active_cells;
    const ps26053::PointCloud& shared_cloud = adaptive_pipeline.getProcessedCloud();
    // Honest memory model: measured live-grid footprint (Quadtree objects +
    // heap nodes + leaf Cells + container overhead), not sizeof(Cell) alone.
    double adaptive_memory_mb = adaptive_pipeline.getGrid().estimateMemoryBytes() / (1024.0 * 1024.0);

    // --- MODE 1: UNIFORM HIGH-RESOLUTION (5cm Fixed Grid, identical input) ---
    std::cout << "--- Running Mode 1: Uniform High-Resolution Baseline (Fixed 5 cm, identical input) ---\n";
    auto t_u_start = std::chrono::high_resolution_clock::now();

    // A uniform 5cm grid across 100m x 100m would theoretically require (100 / 0.05)^2 = 4,000,000 cells!
    // We count populated active uniform cells over the shared cloud:
    size_t uniform_cell_count = 0;
    {
        std::unordered_set<int64_t> occupied_5cm_cells;
        for (const auto& pt : shared_cloud) {
            int64_t cx = static_cast<int64_t>(std::floor(pt.x / 0.05f));
            int64_t cy = static_cast<int64_t>(std::floor(pt.y / 0.05f));
            int64_t k = (cx << 32) | (cy & 0xFFFFFFFFLL);
            occupied_5cm_cells.insert(k);
        }
        uniform_cell_count = occupied_5cm_cells.size();
    }
    auto t_u_end = std::chrono::high_resolution_clock::now();
    double uniform_latency_ms = std::chrono::duration<double, std::milli>(t_u_end - t_u_start).count();
    // A flat uniform map stores one Cell per occupied key: exact, no overhead.
    double uniform_memory_mb = (uniform_cell_count * sizeof(ps26053::Cell)) / (1024.0 * 1024.0);

    // --- BOUNDARY-ALIGNMENT QA VERIFICATION ---
    // Computed over every stored adaptive cell: overlapping 5 cm microcells
    // plus cells whose stored width disagrees with their stored resolution.
    ps26053::BoundaryQA boundary_qa = adaptive_pipeline.getGrid().checkBoundaryAlignment();
    size_t adaptive_boundary_errors = boundary_qa.totalErrors();

    // --- COMPARATIVE REPORT ---
    // Signed percentages throughout: a negative value is a regression and is
    // reported as such — never clamped, never relabeled.
    double cell_reduction_pct = 100.0 * (1.0 - double(adaptive_cell_count) / double(uniform_cell_count));
    double memory_delta_pct = 100.0 * (1.0 - adaptive_memory_mb / uniform_memory_mb);

    std::cout << "\n==============================================================\n";
    std::cout << "       DIRECT UNIFORM vs ADAPTIVE COMPARATIVE BENCHMARK       \n";
    std::cout << "==============================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " METRIC                     | UNIFORM BASELINE  | ADAPTIVE (OURS)   | GAIN\n";
    std::cout << "----------------------------+-------------------+-------------------+-----------\n";
    std::cout << " Shared Input Points        | " << std::setw(17) << shared_cloud.size()
              << " | " << std::setw(17) << shared_cloud.size() << " | identical\n";
    std::cout << " Active Stored Cells        | " << std::setw(17) << uniform_cell_count
              << " | " << std::setw(17) << adaptive_cell_count
              << " | " << cell_reduction_pct << "% fewer\n";
    std::cout << " Memory Footprint (MB)      | " << std::setw(14) << uniform_memory_mb << " MB"
              << " | " << std::setw(14) << adaptive_memory_mb << " MB"
              << " | " << memory_delta_pct << "%\n";
    std::cout << "  (uniform: flat Cells; adaptive: measured grid incl. nodes/overhead;\n";
    std::cout << "   negative % means adaptive currently uses MORE memory — design issue, not a win)\n";
    std::cout << " Processing Latency (ms)    | " << std::setw(14) << uniform_latency_ms << " ms"
              << " | " << std::setw(14) << metrics.total_time_ms << " ms"
              << " | " << (metrics.fps) << " FPS\n";
    std::cout << "  (uniform: key-counting only, not a full pipeline)\n";
    {
        size_t net = metrics.inference_network_points;
        size_t fb = metrics.inference_fallback_points;
        double total = static_cast<double>(net + fb);
        double coverage = (total > 0.0) ? (100.0 * net / total) : 0.0;
        std::cout << " Network-Labeled Points     | " << std::setw(17) << "n/a"
                  << " | " << std::setw(11) << net << " (" << coverage << "%)\n";
        std::cout << " Fallback-Labeled Points    | " << std::setw(17) << "n/a"
                  << " | " << std::setw(17) << fb << "\n";
    }
    std::cout << " Boundary Alignment Errors  | " << std::setw(17) << "n/a"
              << " | " << std::setw(17) << adaptive_boundary_errors
              << " | " << boundaryVerdict(adaptive_boundary_errors) << "\n";
    std::cout << " QA Cells Checked           | " << std::setw(17) << uniform_cell_count
              << " | " << std::setw(17) << boundary_qa.checked_cells << "\n";
    std::cout << "  (of which overlapping)    | " << std::setw(17) << "n/a"
              << " | " << std::setw(17) << boundary_qa.overlapping_quanta << "\n";
    std::cout << "  (of which misaligned)     | " << std::setw(17) << "n/a"
              << " | " << std::setw(17) << boundary_qa.misaligned_cells << "\n";
    std::cout << "  (n/a: uniform mode counts occupied keys; it stores no cells to check)\n";
    std::cout << "==============================================================\n";

    return 0;
}
