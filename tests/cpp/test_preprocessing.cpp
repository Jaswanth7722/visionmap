#include "ps26053/preprocessing/point_cloud_ops.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <tuple>

namespace {

ps26053::Point3D makePt(float x, float y, float z) {
    ps26053::Point3D p;
    p.x = x; p.y = y; p.z = z;
    p.intensity = 0.5f;
    p.semantic_class = ps26053::SemanticClass::TERRAIN;
    p.confidence = 0.9f;
    return p;
}

} // namespace

int main() {
    std::cout << "[Test] Running test_preprocessing ...\n";

    // C3: distant points must never merge. The old 32-bit XOR hash fused
    // voxels up to ~25 m apart; exact sort-based voxelization keeps them.
    {
        ps26053::PointCloud cloud;
        cloud.push_back(makePt(0.0f, 0.0f, 0.0f));
        cloud.push_back(makePt(24.9f, 0.0f, 0.0f));
        auto out = ps26053::PointCloudOps::voxelGridDownsample(cloud, 0.10f);
        assert(out.size() == 2);
    }

    // C3 bulk invariant: output size equals the independently counted number
    // of distinct voxels (any hash collision would shrink the output), and
    // the centroid of a shared voxel is the exact mean.
    {
        std::mt19937 rng(26053);
        std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
        ps26053::PointCloud cloud;
        for (int i = 0; i < 20000; ++i) {
            cloud.push_back(makePt(dist(rng), dist(rng), dist(rng)));
        }
        // Two points deliberately sharing one voxel.
        cloud.push_back(makePt(1.001f, 2.001f, 3.001f));
        cloud.push_back(makePt(1.009f, 2.009f, 3.009f));

        std::set<std::tuple<int64_t, int64_t, int64_t>> voxels;
        for (const auto& pt : cloud) {
            voxels.emplace(static_cast<int64_t>(std::floor(pt.x * 10.0f)),
                           static_cast<int64_t>(std::floor(pt.y * 10.0f)),
                           static_cast<int64_t>(std::floor(pt.z * 10.0f)));
        }

        auto out = ps26053::PointCloudOps::voxelGridDownsample(cloud, 0.10f);
        assert(out.size() == voxels.size());

        bool found_pair = false;
        for (const auto& pt : out) {
            if (std::fabs(pt.x - 1.005f) < 1e-5f) {
                assert(std::fabs(pt.y - 2.005f) < 1e-5f);
                assert(std::fabs(pt.z - 3.005f) < 1e-5f);
                found_pair = true;
            }
        }
        assert(found_pair);

        // Determinism: identical input reproduces bit-identical output.
        auto out2 = ps26053::PointCloudOps::voxelGridDownsample(cloud, 0.10f);
        assert(out2.size() == out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            assert(out2[i].x == out[i].x && out2[i].y == out[i].y && out2[i].z == out[i].z);
        }
    }

    std::cout << "[Test PASS] test_preprocessing succeeded!\n";
    return 0;
}
