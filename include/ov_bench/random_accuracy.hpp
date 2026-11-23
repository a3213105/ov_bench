#pragma once

#include "ov_bench/application.hpp"

#include <cstdint>
#include <optional>
#include <nlohmann/json.hpp>

namespace ovbench {

struct RandomAccuracyOptions {
    std::filesystem::path model_root, output, reference_root, reference_config, shapes;
    std::string python, worker;
    std::vector<std::string> precisions;
    size_t iterations = 10;
    uint64_t seed = 43;
    int threads = 2;
    unsigned timeout_seconds = 300;
    double threshold = 0.99;
    std::map<std::string, double> atol, rtol;
    std::optional<bool> cpu_pinning;
    bool profiling = false, sdpa_bf16 = true;
    bool tolerance_search = false;
    bool resample_common_cosine_failures = false;
};

int run_random_accuracy(const std::vector<ModelSpec>& specs, const RandomAccuracyOptions& options);

}  // namespace ovbench
