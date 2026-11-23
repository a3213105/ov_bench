#pragma once

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ovbench {

// Usage-class exception (exit code 2), mirroring the Python backend's
// OVBError (a ValueError subclass). Runtime failures remain
// std::runtime_error and exit with code 1, identical to the Python
// OVRuntimeError path.
class OVBError : public std::exception {
public:
    explicit OVBError(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

struct Stats {
    double mean_ms, p50_ms, p90_ms, p99_ms, throughput_fps;
    std::size_t samples;
};

Stats statistics(const std::vector<double>& latencies_ms, std::size_t batch_size = 1);

struct Difference {
    bool finite;
    double cosine, normalized_rmse, mean_abs, max_abs, abs_error_variance;
};

// Shapes must agree; supported numeric dtypes may differ. Metrics aggregate all
// outputs. NRMSE is ||actual-expected|| / ||expected||. Zero norms are errors;
// nonfinite data returns finite=false and zero-valued (JSON-safe) metrics.
// abs_error_variance is the population variance of |actual-expected|.
Difference compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual);
// Per-output metrics, including defined zero-norm handling and exact integer comparison.
struct Tolerance {
    double atol, rtol;
};
Tolerance default_tolerance(const std::string& precision);
nlohmann::json compare_output(const ov::Tensor& expected, const ov::Tensor& actual, double threshold,
                              Tolerance tolerance = {1e-5, 1e-4});
std::vector<ov::Tensor> copy_outputs(ov::InferRequest& request);

// Streaming, lowercase MD5 for file provenance (not authentication).
std::string md5_file(const std::filesystem::path& path);

// Load a model supported by OpenVINO, including ONNX and XML IR; SDK errors propagate.
std::shared_ptr<ov::Model> read_model(const std::filesystem::path& path);

// Binary files are contiguous native-endian payloads, without a header.
// save_tensor creates exclusively, verifies bytes read back, and omits "file".
nlohmann::json save_tensor(const ov::Tensor& tensor, const std::filesystem::path& path);
nlohmann::json tensor_info(const ov::Tensor& tensor);
ov::Tensor load_tensor(const nlohmann::json& descriptor, const std::filesystem::path& base);
nlohmann::json runtime_info(const ov::CompiledModel& compiled);

class Session {
public:
    Session(const std::shared_ptr<ov::Model>& model, const std::string& precision, int threads,
            std::optional<bool> cpu_pinning = std::nullopt, bool profiling = true,
            std::optional<bool> hyper_threading = std::nullopt);
    ov::InferRequest& request();
    const ov::CompiledModel& compiled() const;

private:
    ov::CompiledModel compiled_;
    ov::InferRequest request_;
};

}  // namespace ovbench
