#include "ovbench/core.hpp"

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/runtime/properties.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace ovbench {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw OVBError(message);
}

struct AbsoluteErrorMoments {
    std::size_t count = 0;
    long double mean = 0, m2 = 0;

    void add(long double error) {
        const auto delta = error - mean;
        mean += delta / ++count;
        m2 += delta * (error - mean);
    }

    long double variance() const {
        return std::max(0.L, m2 / count);
    }
};

bool supported(ov::element::Type type) {
    switch (static_cast<ov::element::Type_t>(type)) {
    case ov::element::Type_t::boolean:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f64:
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
        return true;
    default:
        return false;
    }
}

std::size_t tensor_bytes(ov::element::Type type, const ov::Shape& shape) {
    require(supported(type), "Unsupported tensor dtype: " + type.get_type_name());
    std::size_t bytes = type.size();
    for (auto dimension : shape) {
        require(dimension != 0, "Tensor shape contains a zero dimension");
        require(bytes <= std::numeric_limits<std::size_t>::max() / dimension, "Tensor byte size overflow");
        bytes *= dimension;
    }
    return bytes;
}

ov::Tensor owned(const ov::Tensor& tensor) {
    tensor_bytes(tensor.get_element_type(), tensor.get_shape());
    ov::Tensor result(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(result);
    return result;
}

template <typename T>
long double value(ov::Tensor& tensor, std::size_t i) {
    return static_cast<long double>(tensor.data<T>()[i]);
}

long double value_at(ov::Tensor& tensor, std::size_t i) {
    switch (static_cast<ov::element::Type_t>(tensor.get_element_type())) {
    case ov::element::Type_t::boolean: return static_cast<const std::uint8_t*>(tensor.data())[i];
    case ov::element::Type_t::f16: return value<ov::float16>(tensor, i);
    case ov::element::Type_t::bf16: return value<ov::bfloat16>(tensor, i);
    case ov::element::Type_t::f32: return value<float>(tensor, i);
    case ov::element::Type_t::f64: return value<double>(tensor, i);
    case ov::element::Type_t::i8: return value<std::int8_t>(tensor, i);
    case ov::element::Type_t::i16: return value<std::int16_t>(tensor, i);
    case ov::element::Type_t::i32: return value<std::int32_t>(tensor, i);
    case ov::element::Type_t::i64: return value<std::int64_t>(tensor, i);
    case ov::element::Type_t::u8: return value<std::uint8_t>(tensor, i);
    case ov::element::Type_t::u16: return value<std::uint16_t>(tensor, i);
    case ov::element::Type_t::u32: return value<std::uint32_t>(tensor, i);
    case ov::element::Type_t::u64: return value<std::uint64_t>(tensor, i);
    default: throw OVBError("Unsupported tensor dtype");
    }
}

bool finite_tensor(ov::Tensor tensor) {
    for (std::size_t i = 0; i < tensor.get_size(); ++i)
        if (!std::isfinite(value_at(tensor, i))) return false;
    return true;
}

class MD5 {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(), EVP_MD_CTX_free};
public:
    MD5() {
        if (!context || EVP_DigestInit_ex(context.get(), EVP_md5(), nullptr) != 1)
            throw std::runtime_error("OpenSSL MD5 initialization failed");
    }
    void update(const void* data, std::size_t size) {
        if (EVP_DigestUpdate(context.get(), data, size) != 1)
            throw std::runtime_error("OpenSSL MD5 update failed");
    }
    std::string finish() {
        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int length = 0;
        if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1)
            throw std::runtime_error("OpenSSL MD5 finalization failed");
        std::ostringstream text;
        text << std::hex << std::setfill('0');
        for (unsigned int i = 0; i < length; ++i) text << std::setw(2) << static_cast<unsigned int>(digest[i]);
        return text.str();
    }
};

std::string md5(const void* data, std::size_t size) {
    MD5 digest;
    digest.update(data, size);
    return digest.finish();
}

void read_exact(const std::filesystem::path& path, void* data, std::size_t bytes) {
    require(bytes <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
            "Tensor is too large for binary I/O");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open tensor file: " + path.string());
    const auto length = input.tellg();
    if (length < 0 || static_cast<std::uintmax_t>(length) != bytes)
        throw std::runtime_error("Tensor file byte size mismatch: " + path.string());
    input.seekg(0);
    if (!input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Cannot read tensor file: " + path.string());
}

double finite_double(long double number, const char* metric) {
    const double result = static_cast<double>(number);
    if (!std::isfinite(result)) throw std::overflow_error(std::string(metric) + " is not representable");
    return result;
}

std::size_t positive_size(const nlohmann::json& number, const char* label) {
    require(number.is_number_integer(), std::string(label) + " must be a positive integer");
    if (number.is_number_unsigned()) {
        const auto n = number.get<std::uint64_t>();
        require(n > 0 && n <= std::numeric_limits<std::size_t>::max(), std::string(label) + " out of range");
        return static_cast<std::size_t>(n);
    }
    const auto n = number.get<std::int64_t>();
    require(n > 0 && static_cast<std::uint64_t>(n) <= std::numeric_limits<std::size_t>::max(),
            std::string(label) + " out of range");
    return static_cast<std::size_t>(n);
}

}  // namespace

std::string md5_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file for MD5: " + path.string());
    MD5 digest;
    std::array<char, 64 * 1024> buffer;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        digest.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("Cannot read file for MD5: " + path.string());
    return digest.finish();
}

std::shared_ptr<ov::Model> read_model(const std::filesystem::path& path) {
    return ov::Core().read_model(path.string());
}

Stats statistics(const std::vector<double>& latencies_ms, std::size_t batch_size) {
    require(!latencies_ms.empty(), "Latency samples must not be empty");
    require(batch_size > 0, "Batch size must be positive");
    long double sum = 0;
    for (double latency : latencies_ms) {
        require(std::isfinite(latency) && latency > 0, "Latency samples must be positive and finite");
        sum += latency;
    }
    auto sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](long double fraction) {
        const auto position = fraction * (sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(std::floor(position));
        const auto upper = static_cast<std::size_t>(std::ceil(position));
        return finite_double(static_cast<long double>(sorted[lower]) +
                                 (position - lower) * (static_cast<long double>(sorted[upper]) - sorted[lower]),
                             "Percentile");
    };
    const auto mean = sum / sorted.size();
    return {finite_double(mean, "Mean"), percentile(0.5L), percentile(0.9L), percentile(0.99L),
            finite_double(1000.L * batch_size / mean, "Throughput"), sorted.size()};
}

Difference compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    require(!expected.empty(), "Comparison outputs must not be empty");
    require(expected.size() == actual.size(), "Output count mismatch");
    for (std::size_t p = 0; p < expected.size(); ++p) {
        require(expected[p].get_shape() == actual[p].get_shape(), "Output shape mismatch at port " + std::to_string(p));
        tensor_bytes(expected[p].get_element_type(), expected[p].get_shape());
        tensor_bytes(actual[p].get_element_type(), actual[p].get_shape());
    }
    long double dot = 0, reference_norm = 0, actual_norm = 0, squared_error = 0, absolute_error = 0, maximum = 0;
    std::size_t count = 0;
    bool finite = true;
    AbsoluteErrorMoments moments;
    for (std::size_t p = 0; p < expected.size(); ++p) {
        auto reference = owned(expected[p]);
        auto output = owned(actual[p]);
        require(count <= std::numeric_limits<std::size_t>::max() - reference.get_size(), "Output element count overflow");
        count += reference.get_size();
        for (std::size_t i = 0; i < reference.get_size(); ++i) {
            const auto x = value_at(reference, i), y = value_at(output, i);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                finite = false;
                continue;
            }
            const auto error = std::abs(y - x);
            moments.add(error);
            dot += x * y;
            reference_norm += x * x;
            actual_norm += y * y;
            squared_error += error * error;
            absolute_error += error;
            maximum = std::max(maximum, error);
        }
    }
    if (!finite) return {false, 0, 0, 0, 0, 0};
    require(reference_norm > 0 && actual_norm > 0, "Cosine similarity is undefined for zero-norm outputs");
    const auto cosine = dot / std::sqrt(reference_norm) / std::sqrt(actual_norm);
    return {true, finite_double(std::clamp(cosine, -1.L, 1.L), "Cosine"),
            finite_double(std::sqrt(squared_error) / std::sqrt(reference_norm), "NRMSE"),
            finite_double(absolute_error / count, "Mean absolute error"), finite_double(maximum, "Max absolute error"),
            finite_double(moments.variance(), "Absolute error variance")};
}

std::vector<ov::Tensor> copy_outputs(ov::InferRequest& request) {
    std::vector<ov::Tensor> outputs;
    const auto count = request.get_compiled_model().outputs().size();
    outputs.reserve(count);
    for (std::size_t i = 0; i < count; ++i) outputs.push_back(owned(request.get_output_tensor(i)));
    return outputs;
}

Tolerance default_tolerance(const std::string& precision) {
    if (precision == "FP32") return {1e-5, 1e-4};
    if (precision == "BF16") return {1e-3, 1e-2};
    if (precision == "FP16") return {1e-4, 1e-3};
    throw OVBError("Unsupported tolerance precision: " + precision);
}

nlohmann::json compare_output(const ov::Tensor& expected, const ov::Tensor& actual, double threshold,
                              Tolerance tolerance) {
    require(std::isfinite(threshold) && threshold >= -1 && threshold < 1, "Invalid cosine threshold");
    require(std::isfinite(tolerance.atol) && tolerance.atol >= 0 &&
            std::isfinite(tolerance.rtol) && tolerance.rtol >= 0, "Tolerances must be finite and nonnegative");
    require(expected.get_shape() == actual.get_shape(), "Output shape mismatch");
    require(expected.get_element_type().is_real() == actual.get_element_type().is_real(),
            "Output real/integer category mismatch");
    auto reference = owned(expected), output = owned(actual);
    long double dot = 0, xnorm = 0, ynorm = 0, squared = 0, absolute = 0, maximum = 0, required_scale = 0;
    bool scalable = true;
    size_t mismatches = 0, nonfinite = 0, outside = 0;
    AbsoluteErrorMoments moments;
    const bool real = expected.get_element_type().is_real();
    for (size_t i = 0; i < reference.get_size(); ++i) {
        const auto x = value_at(reference, i), y = value_at(output, i);
        if (!std::isfinite(x) || !std::isfinite(y)) { ++nonfinite; ++outside; continue; }
        mismatches += x != y;
        const auto error = std::abs(y - x);
        moments.add(error);
        if (real && error > 0) {
            const auto allowance = static_cast<long double>(tolerance.atol) +
                                   static_cast<long double>(tolerance.rtol) * std::abs(x);
            if (allowance == 0) scalable = false;
            else required_scale = std::max(required_scale, error / allowance);
        }
        outside += real ? error > static_cast<long double>(tolerance.atol) +
                                      static_cast<long double>(tolerance.rtol) * std::abs(x)
                        : x != y;
        dot += x * y;
        xnorm += x * x;
        ynorm += y * y;
        squared += error * error;
        absolute += error;
        maximum = std::max(maximum, error);
    }
    nlohmann::json row{
        {"reference_dtype", expected.get_element_type().get_type_name()},
        {"dtype", actual.get_element_type().get_type_name()}, {"shape", actual.get_shape()},
        {"elements", reference.get_size()}, {"finite", nonfinite == 0},
        {"nonfinite_elements", nonfinite}, {"mismatched_elements", mismatches},
        {"atol", real ? tolerance.atol : 0.0}, {"rtol", real ? tolerance.rtol : 0.0},
        {"out_of_tolerance_elements", outside},
        {"out_of_tolerance_fraction", static_cast<double>(outside) / reference.get_size()},
        {"tolerance_passed", outside == 0}, {"cosine_passed", false},
        {"exact_equal", nonfinite == 0 && mismatches == 0},
        {"cosine_similarity", nullptr}, {"normalized_rmse", nullptr},
        {"mean_abs", nullptr}, {"max_abs", nullptr}, {"abs_error_variance", nullptr},
        {"required_tolerance_scale", nullptr},
        {"status", "FAIL"}};
    if (nonfinite) return row;
    if (scalable) {
        auto scale = finite_double(required_scale, "Required tolerance scale");
        if (static_cast<long double>(scale) < required_scale)
            scale = std::nextafter(scale, std::numeric_limits<double>::infinity());
        row["required_tolerance_scale"] = scale;
    }
    row["mean_abs"] = finite_double(absolute / reference.get_size(), "Mean absolute error");
    row["max_abs"] = finite_double(maximum, "Max absolute error");
    row["abs_error_variance"] = finite_double(moments.variance(), "Absolute error variance");
    row["normalized_rmse"] = xnorm > 0
        ? nlohmann::json(finite_double(std::sqrt(squared / xnorm), "NRMSE"))
        : (squared == 0 ? nlohmann::json(0.0) : nlohmann::json(nullptr));
    const bool zero = xnorm == 0 || ynorm == 0;
    row["zero_norm"] = zero;
    if (!zero)
        row["cosine_similarity"] = finite_double(std::clamp(dot / std::sqrt(xnorm) / std::sqrt(ynorm),
                                                           -1.L, 1.L), "Cosine");
    const bool cosine_passed = zero ? mismatches == 0 : row.at("cosine_similarity").get<double>() > threshold;
    row["cosine_passed"] = cosine_passed;
    const bool passed = real ? cosine_passed && outside == 0 : mismatches == 0;
    row["status"] = passed ? "PASS" : "FAIL";
    return row;
}

nlohmann::json tensor_info(const ov::Tensor& tensor) {
    auto view = tensor;
    tensor_bytes(tensor.get_element_type(), tensor.get_shape());
    require(tensor.is_continuous(), "Tensor inspection requires contiguous data");
    return {{"dtype", tensor.get_element_type().get_type_name()}, {"shape", tensor.get_shape()},
            {"bytes", tensor.get_byte_size()}, {"md5", md5(view.data(), tensor.get_byte_size())},
            {"finite", finite_tensor(tensor)}};
}

nlohmann::json save_tensor(const ov::Tensor& tensor, const std::filesystem::path& path) {
    auto copy = owned(tensor);
    const auto bytes = copy.get_byte_size();
    const auto metadata = tensor_info(copy);
    const auto checksum = metadata.at("md5").get<std::string>();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Cannot create tensor file " + path.string());
    std::size_t offset = 0;
    while (offset < bytes) {
        const auto chunk = std::min(bytes - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto written = ::write(fd, static_cast<const char*>(copy.data()) + offset, chunk);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const int error = written < 0 ? errno : EIO;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "Cannot write tensor file " + path.string());
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(fd) != 0) throw std::system_error(errno, std::generic_category(), "Cannot close tensor file " + path.string());
    if (std::filesystem::file_size(path) != bytes || md5_file(path) != checksum)
        throw std::runtime_error("Saved tensor MD5 readback mismatch: " + path.string());
    return metadata;
}

ov::Tensor load_tensor(const nlohmann::json& descriptor, const std::filesystem::path& base) {
    require(descriptor.is_object(), "Tensor descriptor must be an object");
    const auto dtype = descriptor.at("dtype").get<std::string>();
    const ov::element::Type type(dtype);
    require(supported(type), "Unsupported tensor dtype: " + dtype);
    const auto& dimensions = descriptor.at("shape");
    require(dimensions.is_array(), "Tensor shape must be an array");
    ov::Shape shape;
    for (const auto& dimension : dimensions) shape.push_back(positive_size(dimension, "Tensor dimension"));
    const auto bytes = tensor_bytes(type, shape);
    if (descriptor.contains("bytes"))
        require(positive_size(descriptor.at("bytes"), "Tensor bytes") == bytes, "Descriptor byte size mismatch");
    const auto filename = descriptor.at("file").get<std::string>();
    require(!filename.empty(), "Tensor file path must not be empty");
    const auto path = base / filename;
    // Validate file size before allocating memory from untrusted metadata.
    if (std::filesystem::file_size(path) != bytes) throw std::runtime_error("Tensor file byte size mismatch: " + path.string());
    ov::Tensor tensor(type, shape);
    read_exact(path, tensor.data(), bytes);
    if (descriptor.contains("md5")) {
        auto checksum = descriptor.at("md5").get<std::string>();
        require(checksum.size() == 32 && checksum.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos,
                "Descriptor MD5 must contain 32 hexadecimal characters");
        std::transform(checksum.begin(), checksum.end(), checksum.begin(), [](unsigned char c) {
            return c >= 'A' && c <= 'F' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
        });
        if (checksum != md5_file(path) || checksum != md5(tensor.data(), bytes))
            throw std::runtime_error("Tensor MD5 mismatch: " + path.string());
    }
    if (descriptor.contains("finite"))
        require(descriptor.at("finite").get<bool>() == finite_tensor(tensor), "Tensor finite metadata mismatch");
    return tensor;
}

nlohmann::json runtime_info(const ov::CompiledModel& compiled) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        nlohmann::json metadata = nlohmann::json::object();
        for (const auto& entry : node->get_rt_info()) {
            std::ostringstream text;
            entry.second.print(text);
            metadata[entry.first] = text.str();
        }
        nlohmann::json output_types = nlohmann::json::array();
        for (const auto& output : node->outputs()) output_types.push_back(output.get_element_type().get_type_name());
        nodes.push_back({{"name", node->get_friendly_name()}, {"type", node->get_type_name()},
                         {"output_types", output_types}, {"rt_info", metadata}});
    }
    return {{"nodes", nodes}};
}

Session::Session(const std::shared_ptr<ov::Model>& model, const std::string& precision, int threads,
                 std::optional<bool> cpu_pinning, bool profiling,
                 std::optional<bool> hyper_threading) {
    require(model != nullptr, "Session model must not be null");
    require(threads > 0, "Inference threads must be positive");
    require(precision == "FP32" || precision == "BF16" || precision == "FP16",
            "Precision must be FP32, BF16, or FP16");
    const auto type = precision == "FP32" ? ov::element::f32 :
                      precision == "BF16" ? ov::element::bf16 : ov::element::f16;
    ov::Core core;
    ov::AnyMap config{
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::num_streams(1), ov::inference_num_threads(threads),
        ov::hint::inference_precision(type), ov::enable_profiling(profiling)};
    if (cpu_pinning.has_value())
        config[ov::hint::enable_cpu_pinning.name()] = *cpu_pinning;
    if (hyper_threading.has_value())
        config[ov::hint::enable_hyper_threading.name()] = *hyper_threading;
    compiled_ = core.compile_model(model, "CPU", config);
    request_ = compiled_.create_infer_request();
}

ov::InferRequest& Session::request() { return request_; }
const ov::CompiledModel& Session::compiled() const { return compiled_; }

}  // namespace ovbench
