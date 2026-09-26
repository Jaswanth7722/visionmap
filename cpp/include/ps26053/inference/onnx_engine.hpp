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
    size_t num_points{4096};
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
     * @return Inference duration in milliseconds
     */
    double infer(PointCloud& cloud);

private:
    InferenceConfig config_;
    bool loaded_{false};

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace ps26053
