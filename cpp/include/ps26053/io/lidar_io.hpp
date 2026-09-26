#pragma once

#include "ps26053/common/types.hpp"
#include <string>

namespace ps26053 {

class LidarIO {
public:
    /**
     * @brief Load raw Velodyne HDL64E binary file (4 float32 per point: x, y, z, intensity)
     */
    static bool loadBinScan(const std::string& filepath, PointCloud& out_cloud);

    /**
     * @brief Load SemanticKITTI binary label file (uint32_t per point, lower 16 bits = label)
     * @param remap_path Native-id to project-class table. Defaults to the
     * single shared source of truth (config/semantickitti_remap.txt), which
     * python/training/dataset.py also loads — the mapping lives in exactly
     * one file so the two languages cannot silently disagree (H7).
     */
    static bool loadLabels(const std::string& filepath, PointCloud& in_out_cloud,
                           const std::string& remap_path = "config/semantickitti_remap.txt");

    /**
     * @brief Write point cloud to PLY file with RGB semantic coloring
     */
    static bool writePLY(const std::string& filepath, const PointCloud& cloud);
};

} // namespace ps26053
