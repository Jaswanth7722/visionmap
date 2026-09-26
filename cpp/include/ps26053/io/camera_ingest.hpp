#pragma once

#include "ps26053/common/types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ps26053 {

// Decoded 8-bit RGB image (row-major, 3 bytes per pixel).
struct RgbImage {
    int width{0};
    int height{0};
    std::vector<uint8_t> pixels; // size = width * height * 3

    bool empty() const { return width <= 0 || height <= 0 || pixels.empty(); }
};

class CameraIngest {
public:
    /**
     * @brief Decode an in-memory encoded image (JPEG/PNG) to RGB.
     * @return true on success; false (with stderr diagnostic) otherwise.
     * No fallback pixels are ever synthesized: failure is reported.
     */
    static bool decodeImage(const uint8_t* data, size_t size, RgbImage& out);

    /**
     * @brief Project image pixels into a 3D point cloud with the documented
     * monocular depth model (pixels lower in frame = closer):
     *   vNorm = clip((v - 0.35) / 0.65, 0.05, 1.0)
     *   depth = 5 + 45 * (1 - vNorm)            // metres forward (KITTI X)
     *   lateral = u * depth * tan(30 deg)        // KITTI Y
     *   height = -1.6 + (1 - vNorm) * 4.0        // KITTI Z
     *   intensity = pixel luma
     *
     * Geometry is ESTIMATED, never metric truth: callers must label any
     * downstream product accordingly. Sampling is a deterministic stride
     * capped at max_points so frame cost is bounded and reproducible.
     */
    static PointCloud imageToPointCloud(const RgbImage& img, size_t max_points = 4096);
};

} // namespace ps26053
