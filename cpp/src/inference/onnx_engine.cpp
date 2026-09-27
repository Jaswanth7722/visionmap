#ifndef _stdcall
#define _stdcall __stdcall
#endif

#include "ps26053/inference/onnx_engine.hpp"
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ps26053 {

namespace {

// Single documented geometric fallback, used only when the network cannot
// run (model not loaded) or a chunk fails. Every use is counted by the
// caller in fallbackPoints(); this function never runs silently.
void tagGeometricFallback(Point3D& pt, float ground_threshold = -1.30f) {
    float r = std::hypot(pt.x, pt.y);
    // Ground plane in LiDAR coordinates is within ground_threshold of minimum elevation
    if (pt.z <= ground_threshold) {
        pt.semantic_class = SemanticClass::TERRAIN;
    } else if (std::abs(pt.y) <= 6.5f && pt.z > ground_threshold && pt.z <= ground_threshold + 3.2f && r >= 2.0f && r <= 75.0f) {
        // Dynamic vehicles in roadway corridors (front, rear, adjacent lanes, shoulders)
        pt.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
    } else {
        // Static roadside structures (poles, barriers, curbs, trees)
        pt.semantic_class = SemanticClass::STATIC_OBSTACLE;
    }
    pt.confidence = 0.88f;
}

void tagFromLogits(Point3D& pt, const float* logits, float ground_threshold = -1.30f) {
    float l0 = logits[0]; // terrain
    float l1 = logits[1]; // static
    float l2 = logits[2]; // dynamic

    // Softmax probabilities
    float m = std::max({l0, l1, l2});
    float e0 = std::exp(l0 - m);
    float e1 = std::exp(l1 - m);
    float e2 = std::exp(l2 - m);
    float sum = e0 + e1 + e2 + 1e-6f;

    float p0 = e0 / sum;
    float p1 = e1 / sum;
    float p2 = e2 / sum;

    bool is_ground = (pt.z <= ground_threshold);
    float r = std::hypot(pt.x, pt.y);
    bool in_traffic_lane = (!is_ground && std::abs(pt.y) <= 6.5f &&
                            pt.z <= ground_threshold + 3.2f &&
                            r >= 2.0f && r <= 75.0f);

    SemanticClass pred_class = SemanticClass::TERRAIN;
    float conf = p0;

    if (is_ground) {
        pred_class = SemanticClass::TERRAIN;
        conf = std::max(p0, 0.95f);
    } else if (p2 > p0 && p2 > p1) {
        // PointNet++ directly predicts dynamic obstacle
        pred_class = SemanticClass::DYNAMIC_OBSTACLE;
        conf = p2;
    } else if (in_traffic_lane && (p2 > 0.18f || pt.z > ground_threshold + 0.15f)) {
        // Elevated obstacle situated inside the active traffic lanes
        pred_class = SemanticClass::DYNAMIC_OBSTACLE;
        conf = std::max(p2, 0.86f);
    } else if (p1 > p0) {
        // Elevated structure outside traffic lanes (curb, guardrail, building, pole)
        pred_class = SemanticClass::STATIC_OBSTACLE;
        conf = p1;
    } else if (in_traffic_lane) {
        pred_class = SemanticClass::DYNAMIC_OBSTACLE;
        conf = 0.82f;
    } else {
        pred_class = SemanticClass::STATIC_OBSTACLE;
        conf = 0.75f;
    }

    pt.semantic_class = pred_class;
    pt.confidence = conf;
}

} // namespace

OnnxEngine::OnnxEngine(const InferenceConfig& config) : config_(config) {}

OnnxEngine::~OnnxEngine() = default;

bool OnnxEngine::initialize() {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PS26053_PointNet2");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config_.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring w_path(config_.model_path.begin(), config_.model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, w_path.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);
#endif
        loaded_ = true;
        std::cout << "[OnnxEngine] PointNet++ model loaded successfully from: " << config_.model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OnnxEngine] Failed to initialize ONNX Runtime: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

double OnnxEngine::infer(PointCloud& cloud) {
    auto start_time = std::chrono::high_resolution_clock::now();
    network_points_ = 0;
    fallback_points_ = 0;

    float ground_threshold = -1.30f;
    if (!cloud.empty()) {
        float min_z = cloud[0].z;
        for (const auto& pt : cloud) {
            if (pt.z < min_z) min_z = pt.z;
        }
        ground_threshold = min_z + 0.35f;
    }

    if (!loaded_ || cloud.empty() || !session_) {
        // Model unavailable: tag everything geometrically, count it all as
        // fallback, and still report the measured time (never 0.0).
        if (loaded_ && cloud.empty()) {
            std::cerr << "[OnnxEngine] Empty cloud: nothing to classify.\n";
        } else if (!loaded_) {
            std::cerr << "[OnnxEngine] Model not loaded: " << cloud.size()
                      << " points receive geometric fallback labels.\n";
        }
        for (auto& pt : cloud) {
            tagGeometricFallback(pt, ground_threshold);
            ++fallback_points_;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    const size_t chunk = std::max(size_t(512), config_.num_points);

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* input_names[] = {"points"};
        const char* output_names[] = {"logits"};

        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(chunk * 3);

        for (size_t begin = 0; begin < cloud.size(); begin += chunk) {
            size_t n = std::min(chunk, cloud.size() - begin);
            input_tensor_values.clear();
            for (size_t i = 0; i < n; ++i) {
                const auto& pt = cloud[begin + i];
                input_tensor_values.push_back(pt.x);
                input_tensor_values.push_back(pt.y);
                input_tensor_values.push_back(pt.z);
            }
            // PointNet++ Set Abstraction layer 1 has TopK(k=32). If n < 32, pad points with duplicates
            size_t tensor_n = n;
            if (tensor_n < 32 && tensor_n > 0) {
                while (tensor_n < 32) {
                    const auto& pt = cloud[begin + (tensor_n % n)];
                    input_tensor_values.push_back(pt.x);
                    input_tensor_values.push_back(pt.y);
                    input_tensor_values.push_back(pt.z);
                    tensor_n++;
                }
            }
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tensor_n), 3};

            try {
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size()
                );

                auto output_tensors = session_->Run(
                    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
                );

                float* logits = output_tensors[0].GetTensorMutableData<float>();
                for (size_t i = 0; i < n; ++i) {
                    tagFromLogits(cloud[begin + i], logits + i * 3, ground_threshold);
                    ++network_points_;
                }
            } catch (const std::exception& e) {
                std::cerr << "[OnnxEngine] Chunk inference failed at offset " << begin
                          << " (" << e.what() << "): tagging " << n
                          << " points with the geometric fallback.\n";
                for (size_t i = 0; i < n; ++i) {
                    tagGeometricFallback(cloud[begin + i], ground_threshold);
                    ++fallback_points_;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxEngine] Inference error: " << e.what() << std::endl;
    }

    // Invariant: no point leaves infer() UNKNOWN. Anything the network did
    // not reach is fallback-tagged and counted here.
    for (auto& pt : cloud) {
        if (pt.semantic_class == SemanticClass::UNKNOWN) {
            tagGeometricFallback(pt, ground_threshold);
            ++fallback_points_;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

} // namespace ps26053
