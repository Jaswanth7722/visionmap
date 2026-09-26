#pragma once

#include "ps26053/preprocessing/point_cloud_ops.hpp"
#include "ps26053/resolution/resolution_policy.hpp"
#include "ps26053/mapping/grid_25d.hpp"
#include "ps26053/tracking/kalman_tracker.hpp"
#include <string>
#include <vector>

namespace ps26053 {

/**
 * @brief Runtime configuration aggregated from the YAML files (H3).
 *
 * loadAppConfig() reads config/lidar.yaml, config/resolution.yaml and
 * config/tracking.yaml from the given directory. Every value it fills is
 * actually consumed by the pipeline via MappingPipeline::loadConfig — keys
 * describing unimplemented features were removed from the YAML files instead
 * of being silently ignored here.
 */
struct AppConfig {
    PreprocessingConfig preprocessing;
    std::vector<DistanceBand> bands;
    float motion_velocity_threshold{0.25f};
    float curb_threshold{0.12f};
    GridConfig grid;
    TrackerConfig tracker;
};

/**
 * @brief Load all runtime YAML files.
 * @return true on success; false (with a stderr diagnostic) when any file is
 * missing or malformed. Callers fall back to compiled defaults LOUDLY.
 */
bool loadAppConfig(const std::string& config_dir, AppConfig& out);

} // namespace ps26053
