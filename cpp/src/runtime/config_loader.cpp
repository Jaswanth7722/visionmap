#include "ps26053/runtime/config_loader.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>

namespace ps26053 {

namespace {

template <typename T>
T require(const YAML::Node& node, const std::string& file, const std::string& key) {
    if (!node[key]) {
        throw std::runtime_error("missing required key '" + key + "'");
    }
    try {
        return node[key].as<T>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("bad value for key '" + key + "': " + e.what());
    }
}

} // namespace

bool loadAppConfig(const std::string& config_dir, AppConfig& out) {
    try {
        // --- config/lidar.yaml -> PreprocessingConfig ---
        {
            YAML::Node root = YAML::LoadFile(config_dir + "/lidar.yaml");
            const YAML::Node roi = require<YAML::Node>(root, "lidar.yaml", "roi_crop");
            // A disabled ROI means "no cropping": widen the bounds so every
            // point passes instead of pretending to crop.
            bool roi_enabled = roi["enabled"] ? roi["enabled"].as<bool>() : true;
            if (roi_enabled) {
                out.preprocessing.x_min = require<float>(roi, "lidar.yaml", "x_min");
                out.preprocessing.x_max = require<float>(roi, "lidar.yaml", "x_max");
                out.preprocessing.y_min = require<float>(roi, "lidar.yaml", "y_min");
                out.preprocessing.y_max = require<float>(roi, "lidar.yaml", "y_max");
                out.preprocessing.z_min = require<float>(roi, "lidar.yaml", "z_min");
                out.preprocessing.z_max = require<float>(roi, "lidar.yaml", "z_max");
            } else {
                out.preprocessing.x_min = out.preprocessing.y_min = -1.0e6f;
                out.preprocessing.x_max = out.preprocessing.y_max = 1.0e6f;
                out.preprocessing.z_min = -1.0e6f;
                out.preprocessing.z_max = 1.0e6f;
            }
            const YAML::Node voxel = require<YAML::Node>(root, "lidar.yaml", "voxel_downsample");
            out.preprocessing.enable_voxel = require<bool>(voxel, "lidar.yaml", "enabled");
            out.preprocessing.voxel_leaf_size = require<float>(voxel, "lidar.yaml", "leaf_size");
            if (out.preprocessing.voxel_leaf_size <= 0.0f) {
                throw std::runtime_error("voxel_downsample.leaf_size must be positive");
            }
        }

        // --- config/resolution.yaml -> bands, thresholds, grid extent ---
        {
            YAML::Node root = YAML::LoadFile(config_dir + "/resolution.yaml");
            const YAML::Node bands = require<YAML::Node>(root, "resolution.yaml", "level1_distance_bands");
            if (!bands.IsSequence() || bands.size() == 0) {
                throw std::runtime_error("level1_distance_bands must be a non-empty sequence");
            }
            out.bands.clear();
            float prev_max = -1.0f;
            for (const auto& b : bands) {
                DistanceBand band;
                band.max_range = require<float>(b, "resolution.yaml", "max_range");
                band.cell_size = require<float>(b, "resolution.yaml", "cell_size");
                band.name = b["name"] ? b["name"].as<std::string>() : "";
                if (band.max_range <= prev_max || band.cell_size <= 0.0f) {
                    throw std::runtime_error("distance bands must have ascending max_range and positive cell_size");
                }
                prev_max = band.max_range;
                out.bands.push_back(band);
            }
            const YAML::Node level2 = require<YAML::Node>(root, "resolution.yaml", "level2_local_refinement");
            out.grid.max_depth = require<int>(level2, "resolution.yaml", "max_subdivision_depth");
            if (out.grid.max_depth < 0 || out.grid.max_depth > 6) {
                throw std::runtime_error("max_subdivision_depth out of supported range 0..6");
            }
            const YAML::Node triggers = require<YAML::Node>(level2, "resolution.yaml", "triggers");
            const YAML::Node motion = require<YAML::Node>(triggers, "resolution.yaml", "dynamic_object_motion");
            out.motion_velocity_threshold = require<float>(motion, "resolution.yaml", "min_velocity_threshold");
            const YAML::Node elev = require<YAML::Node>(triggers, "resolution.yaml", "elevation_gradient");
            out.curb_threshold = require<float>(elev, "resolution.yaml", "curb_height_threshold");

            const YAML::Node extent = require<YAML::Node>(root, "resolution.yaml", "world_grid_extent");
            out.grid.x_min = require<float>(extent, "resolution.yaml", "x_min");
            out.grid.x_max = require<float>(extent, "resolution.yaml", "x_max");
            out.grid.y_min = require<float>(extent, "resolution.yaml", "y_min");
            out.grid.y_max = require<float>(extent, "resolution.yaml", "y_max");
            if (extent["base_tile_size"]) {
                out.grid.tile_size = extent["base_tile_size"].as<float>();
            }
        }

        // --- config/tracking.yaml -> TrackerConfig ---
        {
            YAML::Node root = YAML::LoadFile(config_dir + "/tracking.yaml");
            require<YAML::Node>(root, "tracking.yaml", "tracking"); // section must exist
            const YAML::Node assoc = require<YAML::Node>(root, "tracking.yaml", "association");
            out.tracker.association_distance = require<float>(assoc, "tracking.yaml", "max_euclidean_distance");
            out.tracker.max_missed_frames = require<int>(assoc, "tracking.yaml", "max_missed_frames");
            out.tracker.min_hits_to_confirm = require<int>(assoc, "tracking.yaml", "min_hits_to_confirm");
            const YAML::Node kf = require<YAML::Node>(root, "tracking.yaml", "kalman_filter");
            out.tracker.process_noise_pos = require<float>(kf, "tracking.yaml", "process_noise_pos");
            out.tracker.process_noise_vel = require<float>(kf, "tracking.yaml", "process_noise_vel");
            out.tracker.measurement_noise_pos = require<float>(kf, "tracking.yaml", "measurement_noise_pos");
            out.tracker.nominal_dt = require<float>(kf, "tracking.yaml", "time_step_dt");
            const YAML::Node cl = require<YAML::Node>(root, "tracking.yaml", "cluster_extraction");
            out.tracker.min_cluster_size = require<size_t>(cl, "tracking.yaml", "min_cluster_size");
            out.tracker.cluster_radius = require<float>(cl, "tracking.yaml", "cluster_tolerance");
            out.tracker.max_cluster_width = require<float>(cl, "tracking.yaml", "max_cluster_width");
        }
    } catch (const std::exception& e) {
        std::cerr << "[Config] Failed to load runtime config from '" << config_dir
                  << "': " << e.what() << std::endl;
        return false;
    }
    return true;
}

} // namespace ps26053
