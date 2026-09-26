#include "ps26053/runtime/pipeline.hpp"
#include <iostream>

namespace ps26053 {

MappingPipeline::MappingPipeline(const std::string& model_path)
    : inference_engine_([&model_path]() {
        InferenceConfig cfg;
        cfg.model_path = model_path;
        cfg.num_points = 4096;
        cfg.num_threads = 4;
        return OnnxEngine(cfg);
    }()) {
}

bool MappingPipeline::initialize() {
    initialized_ = inference_engine_.initialize();
    return initialized_;
}

FrameMetrics MappingPipeline::processFrame(const PointCloud& raw_scan, double timestamp, int frame_idx) {
    FrameMetrics metrics;
    metrics.frame_index = frame_idx;
    metrics.input_points = raw_scan.size();

    auto total_start = std::chrono::high_resolution_clock::now();

    // Stage 0: Coordinate transform (sensor frame -> world frame). The pose
    // defaults to identity; the sensor world position flows into resolution
    // banding downstream instead of a hardcoded origin.
    PointCloud world_scan = raw_scan;
    coord_transform_.toWorld(world_scan);
    const Eigen::Vector3f sensor_pos = coord_transform_.sensorPosition();

    // Stage 1: Preprocessing (ROI Crop & Voxel Downsampling)
    auto t0 = std::chrono::high_resolution_clock::now();
    PointCloud roi_cloud = PointCloudOps::filterROI(world_scan, prep_config_);
    processed_cloud_ = PointCloudOps::voxelGridDownsample(roi_cloud, prep_config_.voxel_leaf_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics.preprocess_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Stage 2: Semantic Perception (PointNet++ ONNX Runtime C++)
    metrics.inference_time_ms = inference_engine_.infer(processed_cloud_);

    // Stage 3: Dynamic Object Extraction & Kalman Tracking
    auto t2 = std::chrono::high_resolution_clock::now();
    tracker_.update(processed_cloud_, timestamp);
    auto t3 = std::chrono::high_resolution_clock::now();
    metrics.tracking_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    metrics.active_tracks = tracker_.getActiveTracks().size();

    // Stage 4: Adaptive 2.5D World Model Mapping
    auto t4 = std::chrono::high_resolution_clock::now();
    grid_.updateWithPointCloud(processed_cloud_, timestamp, sensor_pos);
    grid_.updateTrackedObjects(tracker_.getActiveTracks());
    grid_.decayTemporal(timestamp, 2.0);
    auto t5 = std::chrono::high_resolution_clock::now();
    metrics.mapping_time_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    metrics.active_cells = grid_.totalActiveCells();

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    metrics.fps = (metrics.total_time_ms > 0.0) ? (1000.0 / metrics.total_time_ms) : 0.0;

    return metrics;
}

} // namespace ps26053
