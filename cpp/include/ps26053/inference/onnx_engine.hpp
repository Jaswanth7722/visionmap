#pragma once

#include "ps26053/common/types.hpp"
#include <string>
#include <memory>
#include <vector>

namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
}

namespace ps26053 {

struct InferenceConfig {
    std::string model_path{"models/onnx/pointnet2_semseg.onnx"};
    size_t num_points{4096}; // per-chunk point count; the full cloud is
                             // processed in chunks so every point is classified
    int num_threads{4};
};

class OnnxEngine {
public:
    explicit OnnxEngine(const InferenceConfig& config = InferenceConfig());
    ~OnnxEngine();

    OnnxEngine(OnnxEngine&&) noexcept = default;
    OnnxEngine& operator=(OnnxEngine&&) noexcept = default;
    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    bool initialize();
    bool isLoaded() const { return loaded_; }

    /**
     * @brief Run PointNet++ semantic segmentation inference natively in C++
     * @param cloud Input point cloud (will be tagged with semantic_class and confidence)
     * @return Inference duration in milliseconds (always measured, never 0)
     *
     * The full cloud is classified by the network in chunks. Points that
     * could not go through the network (model not loaded, chunk failure)
     * receive geometric fallback labels and are counted in fallbackPoints().
     * No point is left UNKNOWN and no silent degradation occurs: the counts
     * below report exactly what happened.
     */
    double infer(PointCloud& cloud);

    size_t networkPoints() const { return network_points_; }
    size_t fallbackPoints() const { return fallback_points_; }

private:
    InferenceConfig config_;
    bool loaded_{false};
    size_t network_points_{0};
    size_t fallback_points_{0};

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace ps26053
