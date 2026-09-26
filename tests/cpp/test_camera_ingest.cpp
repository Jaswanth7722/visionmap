#include "ps26053/io/camera_ingest.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

ps26053::RgbImage makeTestImage(int w, int h) {
    ps26053::RgbImage img;
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t k = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3;
            img.pixels[k] = static_cast<uint8_t>((x * 255) / w);
            img.pixels[k + 1] = static_cast<uint8_t>((y * 255) / h);
            img.pixels[k + 2] = 128;
        }
    }
    return img;
}

} // namespace

int main() {
    std::cout << "[Test] Running test_camera_ingest ...\n";

    // Decode failure paths report failure instead of synthesizing pixels.
    {
        ps26053::RgbImage out;
        assert(!ps26053::CameraIngest::decodeImage(nullptr, 0, out));
        assert(out.empty());
        const uint8_t garbage[16] = {0};
        assert(!ps26053::CameraIngest::decodeImage(garbage, sizeof(garbage), out));
        assert(out.empty());
    }

    // Projection: bounded, deterministic, formula-exact.
    {
        auto img = makeTestImage(160, 90);
        auto c1 = ps26053::CameraIngest::imageToPointCloud(img, 4096);
        auto c2 = ps26053::CameraIngest::imageToPointCloud(img, 4096);
        assert(!c1.empty());
        assert(c1.size() <= 4096);
        assert(c1.size() == c2.size());
        for (size_t i = 0; i < c1.size(); ++i) {
            assert(c1[i].x == c2[i].x && c1[i].y == c2[i].y && c1[i].z == c2[i].z);
            assert(c1[i].semantic_class == ps26053::SemanticClass::UNKNOWN);
        }

        // Depth/height stay inside the documented model range; the far row
        // (v = 0 -> vNorm clamped to 0.05) sits at 47.75 m.
        for (const auto& pt : c1) {
            assert(pt.x >= 5.0f && pt.x <= 47.76f);
            assert(pt.z >= -1.61f && pt.z <= 2.21f);
        }
        bool found_far = false;
        for (const auto& pt : c1) {
            if (std::fabs(pt.x - 47.75f) < 0.1f) {
                found_far = true;
                break;
            }
        }
        assert(found_far);

        // Intensity equals pixel luma for a known pixel (x=0,y=0: r=0,g=0,b=128).
        assert(std::fabs(c1[0].intensity - (0.114f * 128.0f / 255.0f)) < 1e-5f);
    }

    // Empty image and zero budget yield empty clouds, never crashes.
    {
        ps26053::RgbImage empty;
        assert(ps26053::CameraIngest::imageToPointCloud(empty, 4096).empty());
        auto img = makeTestImage(32, 32);
        assert(ps26053::CameraIngest::imageToPointCloud(img, 0).empty());
    }

    std::cout << "[Test PASS] test_camera_ingest succeeded!\n";
    return 0;
}
