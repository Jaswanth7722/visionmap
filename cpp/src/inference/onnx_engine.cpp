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
    if (!loaded_ || cloud.empty() || !session_) {
        // Fallback: fast geometric tagging if model is not loaded
        for (auto& pt : cloud) {
            if (pt.z < -1.2f) {
                pt.semantic_class = SemanticClass::TERRAIN;
            } else if (std::hypot(pt.x, pt.y) < 25.0f && pt.z > -0.8f && pt.z < 1.8f) {
                pt.semantic_class = SemanticClass::DYNAMIC_OBSTACLE;
            } else {
                pt.semantic_class = SemanticClass::STATIC_OBSTACLE;
            }
            pt.confidence = 0.85f;
        }
        return 0.0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t N = config_.num_points;
    std::vector<float> input_tensor_values(N * 3);

    // Sample or pad points for inference batch
    size_t step = std::max(size_t(1), cloud.size() / N);
    std::vector<size_t> sampled_indices;
    sampled_indices.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        size_t idx = (i * step) % cloud.size();
        sampled_indices.push_back(idx);
        const auto& pt = cloud[idx];
        input_tensor_values[i * 3 + 0] = pt.x;
        input_tensor_values[i * 3 + 1] = pt.y;
        input_tensor_values[i * 3 + 2] = pt.z;
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(N), 3};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        const char* input_names[] = {"points"};
        const char* output_names[] = {"logits"};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        float* logits = output_tensors[0].GetTensorMutableData<float>();

        // Tag sampled points
        for (size_t i = 0; i < N; ++i) {
            float l0 = logits[i * 3 + 0]; // terrain
            float l1 = logits[i * 3 + 1]; // static
            float l2 = logits[i * 3 + 2]; // dynamic

            // Softmax
            float m = std::max({l0, l1, l2});
            float e0 = std::exp(l0 - m);
            float e1 = std::exp(l1 - m);
            float e2 = std::exp(l2 - m);
            float sum = e0 + e1 + e2;

            SemanticClass pred_class = SemanticClass::TERRAIN;
            float conf = e0 / sum;

            if (l1 > l0 && l1 > l2) {
                pred_class = SemanticClass::STATIC_OBSTACLE;
                conf = e1 / sum;
            } else if (l2 > l0 && l2 > l1) {
                pred_class = SemanticClass::DYNAMIC_OBSTACLE;
                conf = e2 / sum;
            }

            size_t orig_idx = sampled_indices[i];
            cloud[orig_idx].semantic_class = pred_class;
            cloud[orig_idx].confidence = conf;
        }

        // Tag non-sampled points using fast nearest neighbor / height heuristics
        for (auto& pt : cloud) {
            if (pt.semantic_class == SemanticClass::UNKNOWN) {
                if (pt.z < -1.25f) {
                    pt.semantic_class = SemanticClass::TERRAIN;
                    pt.confidence = 0.90f;
                } else {
                    pt.semantic_class = SemanticClass::STATIC_OBSTACLE;
                    pt.confidence = 0.75f;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxEngine] Inference error: " << e.what() << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

} // namespace ps26053
