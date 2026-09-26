#pragma once

#include "ps26053/common/types.hpp"
#include "ps26053/preprocessing/point_cloud_ops.hpp"
#include "ps26053/inference/onnx_engine.hpp"
#include "ps26053/tracking/kalman_tracker.hpp"
#include "ps26053/mapping/grid_25d.hpp"
#include <chrono>

namespace ps26053 {

struct FrameMetrics {
    int frame_index{0};
    double total_time_ms{0.0};
    double preprocess_time_ms{0.0};
    double inference_time_ms{0.0};
    double tracking_time_ms{0.0};
    double mapping_time_ms{0.0};
    size_t input_points{0};
    size_t active_cells{0};
    size_t active_tracks{0};
    double fps{0.0};
};

class MappingPipeline {
public:
    MappingPipeline(const std::string& model_path = "models/onnx/pointnet2_semseg.onnx");
    ~MappingPipeline() = default;

    bool initialize();

    /**
     * @brief Process one complete LiDAR scan through the end-to-end C++ pipeline
     */
    FrameMetrics processFrame(const PointCloud& raw_scan, double timestamp, int frame_idx = 0);

    const Grid25D& getGrid() const { return grid_; }
    Grid25D& getGrid() { return grid_; }

    const std::vector<TrackedObject>& getTracks() const { return tracker_.getActiveTracks(); }
    const PointCloud& getProcessedCloud() const { return processed_cloud_; }

private:
    PreprocessingConfig prep_config_;
    OnnxEngine inference_engine_;
    KalmanTracker tracker_;
    Grid25D grid_;
    PointCloud processed_cloud_;
    bool initialized_{false};
};

} // namespace ps26053
