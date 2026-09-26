#include "ps26053/runtime/config_loader.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_config ...\n";

    // H3: the shipped YAML files must load and drive behavior. CTest runs
    // with CWD = build dir, hence the relative config path.
    ps26053::AppConfig cfg;
    assert(ps26053::loadAppConfig("../config", cfg));

    // lidar.yaml -> preprocessing
    assert(cfg.preprocessing.x_min == -50.0f);
    assert(cfg.preprocessing.x_max == 50.0f);
    assert(cfg.preprocessing.z_min == -4.0f);
    assert(cfg.preprocessing.z_max == 6.0f);
    assert(cfg.preprocessing.enable_voxel);
    assert(cfg.preprocessing.voxel_leaf_size == 0.10f);

    // resolution.yaml -> bands (must match the locked spec values)
    assert(cfg.bands.size() == 4);
    assert(cfg.bands[0].max_range == 10.0f && cfg.bands[0].cell_size == 0.05f);
    assert(cfg.bands[1].max_range == 30.0f && cfg.bands[1].cell_size == 0.15f);
    assert(cfg.bands[2].max_range == 60.0f && cfg.bands[2].cell_size == 0.30f);
    assert(cfg.bands[3].max_range == 100.0f && cfg.bands[3].cell_size == 0.50f);
    assert(std::fabs(cfg.motion_velocity_threshold - 0.25f) < 1e-6f);
    assert(std::fabs(cfg.curb_threshold - 0.12f) < 1e-6f);
    assert(cfg.grid.max_depth == 2);
    assert(cfg.grid.x_min == -60.0f && cfg.grid.x_max == 60.0f);

    // Every shipped band must be an exact multiple of the alignment quantum,
    // or the C1 disjointness guarantee does not hold for file-driven bands.
    {
        ps26053::ResolutionPolicy policy;
        policy.setBands(cfg.bands);
        assert(policy.quantumMultiple(0.05f) == 1);
        assert(policy.quantumMultiple(0.15f) == 3);
        assert(policy.quantumMultiple(0.30f) == 6);
        assert(policy.quantumMultiple(0.50f) == 10);
        assert(policy.getBaseResolution(25.0f) == 0.15f);
    }

    // tracking.yaml -> tracker
    assert(std::fabs(cfg.tracker.association_distance - 2.5f) < 1e-6f);
    assert(cfg.tracker.max_missed_frames == 5);
    assert(cfg.tracker.min_hits_to_confirm == 3);
    assert(cfg.tracker.min_cluster_size == 15);
    assert(std::fabs(cfg.tracker.cluster_radius - 0.65f) < 1e-6f);
    assert(std::fabs(cfg.tracker.max_cluster_width - 6.0f) < 1e-6f);

    // Missing directory must fail loudly, never silently use defaults.
    ps26053::AppConfig bad;
    assert(!ps26053::loadAppConfig("nonexistent_config_dir", bad));

    std::cout << "[Test PASS] test_config succeeded!\n";
    return 0;
}
