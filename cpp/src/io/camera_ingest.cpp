// stb_image is public domain (http://nothings.org/stb). Vendored under
// third_party/stb/ so JPEG upload frames decode with no extra dependency.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "ps26053/io/camera_ingest.hpp"
#include <cmath>
#include <iostream>

namespace ps26053 {

bool CameraIngest::decodeImage(const uint8_t* data, size_t size, RgbImage& out) {
    out = RgbImage();
    if (!data || size == 0) {
        std::cerr << "[CameraIngest] Empty image buffer: nothing to decode.\n";
        return false;
    }
    int w = 0, h = 0, channels = 0;
    unsigned char* rgb = stbi_load_from_memory(
        data, static_cast<int>(size), &w, &h, &channels, 3);
    if (!rgb) {
        std::cerr << "[CameraIngest] Image decode failed: "
                  << stbi_failure_reason() << "\n";
        return false;
    }
    out.width = w;
    out.height = h;
    out.pixels.assign(rgb, rgb + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(rgb);
    return true;
}

PointCloud CameraIngest::imageToPointCloud(const RgbImage& img, size_t max_points) {
    PointCloud cloud;
    if (img.empty() || max_points == 0) return cloud;

    const int W = img.width;
    const int H = img.height;
    const float total = static_cast<float>(W) * static_cast<float>(H);
    // Stride chosen so the output never exceeds max_points.
    int stride = 1;
    while (total / (static_cast<float>(stride) * static_cast<float>(stride)) >
           static_cast<float>(max_points)) {
        ++stride;
    }

    constexpr float kTan30 = 0.5773503f; // tan(30 deg), ~60 deg horizontal FOV
    cloud.reserve(max_points);
    for (int y = 0; y < H && cloud.size() < max_points; y += stride) {
        float v = static_cast<float>(y) / static_cast<float>(H - 1);
        float v_norm = (v - 0.35f) / 0.65f;
        v_norm = v_norm < 0.05f ? 0.05f : (v_norm > 1.0f ? 1.0f : v_norm);
        float depth = 5.0f + 45.0f * (1.0f - v_norm);
        float height = -1.6f + (1.0f - v_norm) * 4.0f;
        for (int x = 0; x < W && cloud.size() < max_points; x += stride) {
            size_t k = (static_cast<size_t>(y) * static_cast<size_t>(W) +
                        static_cast<size_t>(x)) * 3;
            float luma = (0.299f * img.pixels[k] + 0.587f * img.pixels[k + 1] +
                          0.114f * img.pixels[k + 2]) / 255.0f;
            float u = (static_cast<float>(x) / static_cast<float>(W - 1)) * 2.0f - 1.0f;
            Point3D pt;
            pt.x = depth;                    // forward
            pt.y = u * depth * kTan30;       // lateral
            pt.z = height;                   // vertical
            pt.intensity = luma;
            pt.semantic_class = SemanticClass::UNKNOWN;
            pt.confidence = 0.0f;
            cloud.push_back(pt);
        }
    }
    return cloud;
}

} // namespace ps26053
