#include "ovbench/application.hpp"
#include "ovbench/core.hpp"
#include "ovbench/image.hpp"
#include "ovbench/stability.hpp"
#include "ovbench/random_accuracy.hpp"
#include "ovbench/self_test.hpp"

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

enum class TimedStage : sig_atomic_t {
    None = 0,
    ReadModel,
    LoadInputs,
    CompileModel,
    BindInputs,
    Inference,
    CopyOutputs,
    InspectOutputs,
};

volatile sig_atomic_t active_timed_stage = static_cast<sig_atomic_t>(TimedStage::None);
char timeout_marker_path[4096] = {};
char timeout_record[256] = {};
volatile sig_atomic_t timeout_record_length = 0;

void set_timeout_marker_path(const fs::path& path) {
    const auto value = path.string();
    if (value.size() >= sizeof(timeout_marker_path))
        throw std::runtime_error("Single-mode timeout marker path is too long");
    std::copy(value.begin(), value.end(), timeout_marker_path);
    timeout_marker_path[value.size()] = '\0';
}

void write_timeout_message(int descriptor, const char* message, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        const auto written = ::write(descriptor, message + offset, length - offset);
        if (written > 0) offset += static_cast<size_t>(written);
        else if (written < 0 && errno == EINTR) continue;
        else break;
    }
}

void timeout_handler(int) {
    const int marker = ::open(timeout_marker_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (marker >= 0) {
        const auto length = static_cast<size_t>(timeout_record_length);
        write_timeout_message(marker, timeout_record, length);
        (void)::close(marker);
    }
    switch (static_cast<TimedStage>(active_timed_stage)) {
    case TimedStage::ReadModel:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: read_model exceeded its configured deadline\n",
                      sizeof("TIMEOUT: read_model exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::LoadInputs:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: load_inputs exceeded its configured deadline\n",
                      sizeof("TIMEOUT: load_inputs exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::CompileModel:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: compile_model exceeded its configured deadline\n",
                      sizeof("TIMEOUT: compile_model exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::BindInputs:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: bind_inputs exceeded its configured deadline\n",
                      sizeof("TIMEOUT: bind_inputs exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::Inference:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: inference exceeded its configured deadline\n",
                      sizeof("TIMEOUT: inference exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::CopyOutputs:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: copy_outputs exceeded its configured deadline\n",
                      sizeof("TIMEOUT: copy_outputs exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::InspectOutputs:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: inspect_outputs exceeded its configured deadline\n",
                      sizeof("TIMEOUT: inspect_outputs exceeded its configured deadline\n") - 1);
        ::_exit(124);
    case TimedStage::None:
        write_timeout_message(STDERR_FILENO, "TIMEOUT: single-inference stage timed out\n",
                      sizeof("TIMEOUT: single-inference stage timed out\n") - 1);
        ::_exit(124);
    }
    ::_exit(124);
}

template <typename Function>
auto timed_stage(json& timings, const char* name, unsigned int timeout_seconds,
                 TimedStage stage, Function&& function) -> std::invoke_result_t<Function> {
    using Result = std::invoke_result_t<Function>;
    const int record_length = std::snprintf(
        timeout_record, sizeof(timeout_record),
        "{\"status\":\"TIMEOUT\",\"stage\":\"%s\",\"timeout_seconds\":%u}\n",
        name, timeout_seconds);
    if (record_length < 0 || static_cast<size_t>(record_length) >= sizeof(timeout_record))
        throw std::runtime_error("Cannot prepare the single-mode timeout record");
    timeout_record_length = static_cast<sig_atomic_t>(record_length);
    const auto start = Clock::now();
    active_timed_stage = static_cast<sig_atomic_t>(stage);
    std::cerr << "[single] " << name << ".begin timeout_seconds=" << timeout_seconds << std::endl;
    ::alarm(timeout_seconds);
    try {
        if constexpr (std::is_void_v<Result>) {
            std::forward<Function>(function)();
            ::alarm(0);
            active_timed_stage = static_cast<sig_atomic_t>(TimedStage::None);
            const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            timings[name] = elapsed;
            std::cerr << "[single] " << name << ".end elapsed_ms=" << elapsed << std::endl;
        } else {
            auto result = std::forward<Function>(function)();
            ::alarm(0);
            active_timed_stage = static_cast<sig_atomic_t>(TimedStage::None);
            const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            timings[name] = elapsed;
            std::cerr << "[single] " << name << ".end elapsed_ms=" << elapsed << std::endl;
            return result;
        }
    } catch (...) {
        ::alarm(0);
        active_timed_stage = static_cast<sig_atomic_t>(TimedStage::None);
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        std::cerr << "[single] " << name << ".error elapsed_ms=" << elapsed << std::endl;
        throw;
    }
}

struct Options {
    fs::path model_root, images, manifest, output, start_signal;
    fs::path model_file;
    std::map<std::string, ov::Shape> input_shapes;
    size_t random_samples = 10;
    size_t input_count = 10;
    bool input_count_explicit = false;
    fs::path reference_root, reference_config, random_shapes;
    std::string reference_python = "python3";
    uint64_t seed = 43;
    std::string model = "all", precision = "FP32,BF16,FP16", mode = "performance";
    int batch = 0;
    std::vector<int> batches{0};
    bool batch_explicit = false;
    size_t warmup = 3, iterations = 200, rounds = 10;
    bool iterations_explicit = false;
    std::optional<double> duration_seconds;
    unsigned int step_timeout_seconds = 300;
    int threads = 6;
    double threshold = 0.99;
    std::map<std::string, double> atol, rtol;
    std::vector<fs::path> stability_dirs;
    std::string cpu_plugin;
    bool prepared = false, sdpa_bf16 = true;
    bool save_performance_outputs = true;
    bool resample_common_cosine_failures = false;
    std::optional<bool> cpu_pinning;
    std::optional<bool> hyper_threading;
    bool profiling = true;
};

size_t positive_integer(const std::string& value, bool zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected nonnegative integer, got: " + value);
    const auto n = std::stoull(value);
    if ((!zero && !n) || n > 1000000000)
        throw std::invalid_argument("Integer out of range: " + value);
    return static_cast<size_t>(n);
}

std::vector<std::string> split(const std::string& value) {
    std::vector<std::string> items;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty() || std::find(items.begin(), items.end(), item) != items.end())
            throw std::invalid_argument("Empty or duplicate CSV item: " + value);
        items.push_back(item);
    }
    if (items.empty() || value.back() == ',') throw std::invalid_argument("Invalid CSV: " + value);
    return items;
}

std::string batch_directory_name(int batch) {
    return batch == 0 ? "batch_dynamic" : "batch_" + std::to_string(batch);
}

void parse_tolerances(const std::string& key, const std::string& value,
                      std::map<std::string, double>& tolerances) {
    for (const auto& item : split(value)) {
        const auto separator = item.find('=');
        if (separator == std::string::npos || separator == 0 || separator + 1 == item.size())
            throw std::invalid_argument(key + " requires PRECISION=VALUE[,PRECISION=VALUE]");
        const auto precision = item.substr(0, separator);
        if (precision != "FP32" && precision != "BF16" && precision != "FP16")
            throw std::invalid_argument("Unsupported tolerance precision: " + precision);
        const auto number = item.substr(separator + 1);
        size_t parsed = 0;
        const auto tolerance = std::stod(number, &parsed);
        if (parsed != number.size() || !std::isfinite(tolerance) || tolerance < 0)
            throw std::invalid_argument(key + " values must be finite and nonnegative");
        if (!tolerances.emplace(precision, tolerance).second)
            throw std::invalid_argument("Duplicate " + key + " precision: " + precision);
    }
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout <<
                "ov_benchmark --model-root DIR --output NEW_DIR [options]\n"
                "  --model all|MODEL_ID[,MODEL_ID]  --precisions FP32,BF16,FP16\n"
                "  --model-file FILE               Test an unregistered ONNX/XML model with native random inputs\n"
                "  --input-shape NAME=DIM[,DIM...]  Override one input shape; repeat per input\n"
                "  --samples 10 --seed 43          Generic random dataset size and seed\n"
                "  --input-count 10                Prediction random input tensor count\n"
                "  --images FILE_OR_DIR            Native JPEG/PNG preprocessing\n"
                "  --manifest FILE --prepared-inputs  Raw input/FP32-reference dataset\n"
                "  --mode performance|accuracy|random-accuracy|stability|all|preprocess|single\n"
                "  --batch 0|N[,0|N...]       Dynamic batch (0) or static batch N; default: 0\n"
                "    Example: --batch 0,1,2 tests dynamic, batch 1, and batch 2\n"
                "    Legacy single value -1 is accepted as an alias for 0\n"
                "  --mode self-test                 Run native regression tests; no model/output required\n"
                "  --mode gather-guard [--cpu-plugin FILE]  Verify Gather BF16 output and memory guards\n"
                "  --mode tolerance-search --precisions FP32  Search fixed-ratio FP32 tolerances, then verify\n"
                "  --mode compare-stability --stability-dir DIR [--stability-dir DIR ...] --output NEW.json\n"
                "  --threads 6 --warmup 3 --iterations 200 --rounds 10\n"
                "  --duration SECONDS             Timed inference duration instead of --iterations (exclusive)\n"
                "    warmup/iterations count individual inferences; cooldown equals warmup\n"
                "    Inputs cycle sequentially across warmup, timed inference and cooldown\n"
                "  --step-timeout-seconds 300       Per-stage timeout in single mode\n"
                "  --threshold 0.99               Cosine threshold; ViT FP16 SDPA BF16 rewrite is ON by default\n"
                "  --no-sdpa-bf16                 Disable rewrite for diagnostic comparison\n"
                "  --start-signal FILE            Wait after warmup for multi-process release\n"
                "  --no-performance-output-files  Keep output checks/hashes without saving check_*.bin\n"
                "  --cpu-pinning YES|NO           Override OpenVINO internal pinning\n"
                "  --hyper-threading YES|NO       Override ENABLE_HYPER_THREADING; --threads is the explicit thread count\n"
                "  --perf-count YES|NO            Enable or disable OpenVINO profiling\n";
            std::cout <<
                "Random accuracy (no image/reference dataset required):\n"
                "  --iterations 200 --seed 43        Fresh shared random inputs; default seed 43\n"
                "  --reference-root DIR             Native ONNX/PyTorch assets (default: sibling models/)\n"
                "  --reference-config FILE          Per-model framework adapter configuration\n"
                "  --reference-python EXECUTABLE    Python with ONNX Runtime/PyTorch dependencies\n"
                "  --random-shapes FILE             Per-model named input shape overrides\n";
            std::cout <<
                "  --resample-common-cosine-failures  Count inputs only if BF16 or FP16 passes all output cosine checks\n"
                "    Requires both BF16 and FP16; rejected inputs and metrics are saved separately; retries are unlimited\n";
            std::cout <<
                "  --atol FP32=1e-5,BF16=1e-3,FP16=1e-4\n"
                "  --rtol FP32=1e-4,BF16=1e-2,FP16=1e-3\n"
                "    Override by precision; omitted precisions keep defaults; options may repeat\n"
                "    Defaults (atol/rtol): FP32 1e-5/1e-4, BF16 1e-3/1e-2, FP16 1e-4/1e-3\n";
            std::exit(0);
        }
        if (key == "--prepared-inputs") { o.prepared = true; continue; }
        if (key == "--no-performance-output-files") { o.save_performance_outputs = false; continue; }
        if (key == "--no-sdpa-bf16") { o.sdpa_bf16 = false; continue; }
        if (key == "--resample-common-cosine-failures") { o.resample_common_cosine_failures = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model-root") o.model_root = value;
        else if (key == "--model-file") o.model_file = value;
        else if (key == "--samples") o.random_samples = positive_integer(value);
        else if (key == "--input-count") {
            o.input_count = positive_integer(value);
            o.input_count_explicit = true;
        }
        else if (key == "--input-shape") {
            const auto separator = value.find('=');
            if (separator == std::string::npos || separator == 0)
                throw std::invalid_argument("--input-shape requires NAME=DIM[,DIM...]");
            ov::Shape shape;
            const auto dimensions = value.substr(separator + 1);
            if (!dimensions.empty()) {
                std::istringstream stream(dimensions);
                std::string dimension;
                while (std::getline(stream, dimension, ','))
                    shape.push_back(positive_integer(dimension));
                if (dimensions.back() == ',') throw std::invalid_argument("Invalid --input-shape dimension list");
            }
            if (!o.input_shapes.emplace(value.substr(0, separator), shape).second)
                throw std::invalid_argument("Duplicate --input-shape input name");
        }
        else if (key == "--output") o.output = value;
        else if (key == "-batch" || key == "--batch") {
            std::vector<std::string> values = value == "-1" ? std::vector<std::string>{"0"} : split(value);
            std::set<int> unique_batches;
            o.batches.clear();
            for (const auto& item : values) {
                const auto batch = static_cast<int>(positive_integer(item, true));
                if (!unique_batches.insert(batch).second)
                    throw std::invalid_argument("Duplicate batch value: " + item);
                o.batches.push_back(batch);
            }
            o.batch = o.batches.front();
            o.batch_explicit = true;
        }
        else if (key == "--images") o.images = value;
        else if (key == "--manifest") o.manifest = value;
        else if (key == "--start-signal") o.start_signal = value;
        else if (key == "--reference-root") o.reference_root = value;
        else if (key == "--reference-config") o.reference_config = value;
        else if (key == "--reference-python") o.reference_python = value;
        else if (key == "--random-shapes") o.random_shapes = value;
        else if (key == "--stability-dir") o.stability_dirs.emplace_back(value);
        else if (key == "--cpu-plugin") o.cpu_plugin = value;
        else if (key == "--seed") o.seed = positive_integer(value, true);
        else if (key == "--hyper-threading") {
            if (value != "YES" && value != "NO")
                throw std::invalid_argument("--hyper-threading must be YES or NO");
            o.hyper_threading = value == "YES";
        }
        else if (key == "--cpu-pinning") {
            if (value != "YES" && value != "NO")
                throw std::invalid_argument("--cpu-pinning must be YES or NO");
            o.cpu_pinning = value == "YES";
        }
        else if (key == "--perf-count") {
            if (value != "YES" && value != "NO")
                throw std::invalid_argument("--perf-count must be YES or NO");
            o.profiling = value == "YES";
        }
        else if (key == "--model") o.model = value;
        else if (key == "--precisions") o.precision = value;
        else if (key == "--mode") o.mode = value;
        else if (key == "--warmup") o.warmup = positive_integer(value, true);
        else if (key == "--iterations") {
            o.iterations = positive_integer(value);
            o.iterations_explicit = true;
        }
        else if (key == "--duration") {
            size_t parsed = 0;
            const auto duration = std::stod(value, &parsed);
            if (parsed != value.size() || !std::isfinite(duration) || duration <= 0)
                throw std::invalid_argument("--duration must be finite and positive seconds");
            o.duration_seconds = duration;
        }
        else if (key == "--rounds") o.rounds = positive_integer(value);
        else if (key == "--threads") o.threads = static_cast<int>(positive_integer(value));
        else if (key == "--step-timeout-seconds")
            o.step_timeout_seconds = static_cast<unsigned int>(positive_integer(value));
        else if (key == "--atol" || key == "--rtol") {
            parse_tolerances(key, value, key == "--atol" ? o.atol : o.rtol);
        }
        else if (key == "--threshold") {
            size_t parsed = 0;
            o.threshold = std::stod(value, &parsed);
            if (parsed != value.size() || !std::isfinite(o.threshold) || o.threshold < -1 || o.threshold >= 1)
                throw std::invalid_argument("threshold must be finite in [-1,1)");
        } else throw std::invalid_argument("Unknown option: " + key);
    }
    const bool batch_supported = o.mode == "performance" || o.mode == "accuracy" ||
        o.mode == "stability" || o.mode == "all" || o.mode == "preprocess" || o.mode == "single";
    if (!batch_supported &&
        (o.batches.size() > 1 || std::any_of(o.batches.begin(), o.batches.end(),
                                             [](int batch) { return batch != 0; })))
        throw std::invalid_argument("Batch lists and static batch values require a model execution mode");
    if (o.duration_seconds.has_value()) {
        if (o.iterations_explicit)
            throw std::invalid_argument("--iterations and --duration are mutually exclusive");
        if (o.mode != "performance" && o.mode != "all")
            throw std::invalid_argument("--duration requires --mode performance or all");
    }
    if (o.input_count_explicit && o.mode != "performance" && o.mode != "all")
        throw std::invalid_argument("--input-count requires --mode performance or all");
    if (o.input_count_explicit && (o.prepared || !o.manifest.empty() || !o.model_file.empty()))
        throw std::invalid_argument("--input-count requires generated prediction inputs, not a manifest or --model-file");
    if ((!o.atol.empty() || !o.rtol.empty()) && o.mode != "random-accuracy")
        throw std::invalid_argument("--atol and --rtol require --mode random-accuracy");
    if (o.resample_common_cosine_failures && o.mode != "random-accuracy")
        throw std::invalid_argument("--resample-common-cosine-failures requires --mode random-accuracy");
    if (o.mode == "tolerance-search" && o.precision != "FP32")
        throw std::invalid_argument("tolerance-search requires --precisions FP32");
    if (!o.stability_dirs.empty() && o.mode != "compare-stability")
        throw std::invalid_argument("--stability-dir requires --mode compare-stability");
    if (!o.cpu_plugin.empty() && o.mode != "gather-guard")
        throw std::invalid_argument("--cpu-plugin requires --mode gather-guard");
    if (o.mode == "self-test" || o.mode == "gather-guard") return o;
    if (o.mode == "compare-stability") {
        if (o.output.empty() || fs::exists(o.output) || o.stability_dirs.empty())
            throw std::invalid_argument("compare-stability requires --stability-dir and --output naming a new JSON file");
        return o;
    }
    if (!o.model_file.empty()) {
        if (!fs::is_regular_file(o.model_file))
            throw std::invalid_argument("Model file not found: " + o.model_file.string());
        if (!o.model_root.empty() || o.model != "all" || o.prepared ||
            !o.manifest.empty() || !o.images.empty())
            throw std::invalid_argument("--model-file generates random inputs; do not specify model-root, model, images or manifest");
        if (o.mode != "performance" && o.mode != "accuracy" && o.mode != "stability" &&
            o.mode != "all" && o.mode != "preprocess")
            throw std::invalid_argument("--model-file supports performance, accuracy, stability, all and preprocess");
        o.model_file = fs::absolute(o.model_file);
        o.model_root = o.model_file.parent_path();
    } else if (!o.input_shapes.empty() || o.random_samples != 10) {
        throw std::invalid_argument("--input-shape and --samples require --model-file");
    }
    if (o.output.empty() || o.model_root.empty())
        throw std::invalid_argument("--output and --model-root are required");
    if (o.hyper_threading.has_value() &&
        (o.mode == "random-accuracy" || o.mode == "tolerance-search"))
        throw std::invalid_argument("--hyper-threading is not supported by random-accuracy or tolerance-search");
    if (o.mode != "performance" && o.mode != "accuracy" && o.mode != "stability" &&
        o.mode != "all" && o.mode != "preprocess" && o.mode != "single" &&
        o.mode != "random-accuracy" && o.mode != "tolerance-search")
        throw std::invalid_argument("Unsupported mode: " + o.mode);
    if ((o.mode == "all" || o.mode == "stability") && o.rounds < 2)
        throw std::invalid_argument("Stability requires at least two rounds");
    if ((o.prepared || o.mode == "single") && o.manifest.empty())
        throw std::invalid_argument("Prepared inputs and accuracy require --manifest");
    if (o.mode == "single" &&
        (!o.prepared || o.model == "all" || split(o.model).size() != 1 ||
         split(o.precision).size() != 1 || o.step_timeout_seconds == 0))
        throw std::invalid_argument(
            "Single mode requires --prepared-inputs, one model, one precision, and a positive --step-timeout-seconds");
    if (!o.prepared && o.model_file.empty() && o.images.empty() && o.mode != "random-accuracy" && o.mode != "tolerance-search")
        throw std::invalid_argument("Native image input requires --images");
    const auto precisions = split(o.precision);
    if (o.resample_common_cosine_failures &&
        (std::find(precisions.begin(), precisions.end(), "BF16") == precisions.end() ||
         std::find(precisions.begin(), precisions.end(), "FP16") == precisions.end()))
        throw std::invalid_argument("--resample-common-cosine-failures requires both BF16 and FP16");
    for (const auto& precision : precisions)
        if (precision != "FP32" && precision != "BF16" && precision != "FP16")
            throw std::invalid_argument("Unsupported precision: " + precision);
    if ((o.mode == "random-accuracy" || o.mode == "tolerance-search") &&
        (o.prepared || !o.manifest.empty() || !o.images.empty() || !o.start_signal.empty()))
        throw std::invalid_argument("Random accuracy generates fresh inputs; do not specify a dataset, images, or start signal");
    if (o.mode != "random-accuracy" && o.mode != "tolerance-search" &&
        (!o.reference_root.empty() || !o.reference_config.empty() || !o.random_shapes.empty() ||
         o.reference_python != "python3" || (o.seed != 43 && o.model_file.empty())))
        throw std::invalid_argument("Random input/reference options require --mode random-accuracy");
    if (o.reference_python.empty() || o.step_timeout_seconds == 0)
        throw std::invalid_argument("Reference Python and timeout must be nonempty and positive");
    return o;
}

void write_json(const fs::path& path, const json& value) {
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << value.dump(2) << '\n';
}

json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open JSON: " + path.string());
    return json::parse(input);
}

std::string tsv_escape(const std::string& value) {
    if (value.find_first_of("\t\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '"') escaped += "\"\"";
        else escaped += character;
    }
    escaped += '"';
    return escaped;
}

void write_tsv_row(std::ostream& output, const std::vector<std::string>& fields) {
    for (size_t index = 0; index < fields.size(); ++index) {
        if (index) output << '\t';
        output << tsv_escape(fields[index]);
    }
    output << '\n';
}

std::string json_tsv_value(const json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

void write_summary(const fs::path& path, const json& cases, bool include_batch = false) {
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    std::vector<std::string> header = {"model", "precision", "sdpa_bf16_nodes", "images", "mean_ms", "p99_ms",
                                       "throughput_fps", "accuracy_passed", "md5_consistent", "stable_outputs_finite",
                                       "status", "timed_inferences", "elapsed_seconds", "effective_threads"};
    if (include_batch) header.insert(header.begin() + 2, "batch");
    write_tsv_row(output, header);
    for (const auto& row : cases) {
        std::vector<std::string> fields = {row.at("model").get<std::string>(),
                                           row.at("precision").get<std::string>(),
                                           json_tsv_value(row.at("sdpa_bf16_nodes")),
                                           json_tsv_value(row.at("image_count"))};
        if (include_batch) fields.insert(fields.begin() + 2, json_tsv_value(row.at("batch")));
        for (const auto& metric : {"mean_ms", "p99_ms", "throughput_fps"}) {
            fields.push_back(row.contains("performance")
                                 ? json_tsv_value(row.at("performance").at(metric))
                                 : std::string{});
        }
        fields.push_back(row.contains("accuracy") ? json_tsv_value(row.at("accuracy").at("passed_images")) : "");
        fields.push_back(row.contains("stability") ? json_tsv_value(row.at("stability").at("md5_consistent")) : "");
        fields.push_back(row.contains("stability") ? json_tsv_value(row.at("stability").at("all_outputs_finite")) : "");
        fields.push_back(row.at("status").get<std::string>());
        fields.push_back(row.contains("performance") ? json_tsv_value(row.at("performance").at("samples")) : "");
        fields.push_back(row.contains("performance")
                             ? json_tsv_value(row.at("performance").at("elapsed_seconds"))
                             : std::string{});
        fields.push_back(row.contains("effective_config") &&
                                 row.at("effective_config").contains("INFERENCE_NUM_THREADS")
                             ? json_tsv_value(row.at("effective_config").at("INFERENCE_NUM_THREADS"))
                             : std::string{});
        write_tsv_row(output, fields);
    }
}

std::string safe_id(std::string id) {
    std::replace(id.begin(), id.end(), ':', '_');
    return id;
}

struct Sample {
    std::string image;
    std::vector<std::pair<std::string, ov::Tensor>> inputs;
    json references;
};

std::vector<fs::path> image_files(const fs::path& source) {
    std::vector<fs::path> files;
    const auto supported = [](const fs::path& path) {
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
    };
    if (fs::is_regular_file(source)) {
        if (!supported(source)) throw std::invalid_argument("Supported image formats: JPEG, PNG");
        files.push_back(source);
    } else {
        for (const auto& entry : fs::directory_iterator(source))
            if (entry.is_regular_file() && supported(entry.path())) files.push_back(entry.path());
        std::sort(files.begin(), files.end());
    }
    if (files.empty()) throw std::runtime_error("No JPEG/PNG images found");
    return files;
}

bool zero_tensor(ov::Tensor tensor) {
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        double value;
        if (tensor.get_element_type() == ov::element::f32) value = tensor.data<float>()[i];
        else if (tensor.get_element_type() == ov::element::f64) value = tensor.data<double>()[i];
        else if (tensor.get_element_type() == ov::element::f16)
            value = static_cast<float>(tensor.data<ov::float16>()[i]);
        else if (tensor.get_element_type() == ov::element::bf16)
            value = static_cast<float>(tensor.data<ov::bfloat16>()[i]);
        else throw std::runtime_error("Unsupported real output dtype");
        if (value != 0) return false;
    }
    return true;
}

json compare_prediction_outputs(const std::vector<ov::Tensor>& reference,
                                const std::vector<ov::Tensor>& actual,
                                const std::vector<std::string>& output_names, double threshold) {
    if (reference.size() != actual.size())
        throw std::runtime_error("Prediction output count mismatch");
    bool finite = true, exact_outputs = true;
    std::vector<ov::Tensor> real_reference, real_actual;
    json exact_mismatches = json::array();
    size_t exact_output_count = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (reference[i].get_shape() != actual[i].get_shape())
            throw std::runtime_error("Prediction output shape mismatch at port " + std::to_string(i));
        const auto reference_info = ovbench::tensor_info(reference[i]);
        const auto actual_info = ovbench::tensor_info(actual[i]);
        finite = finite && reference_info.at("finite").get<bool>() &&
                 actual_info.at("finite").get<bool>();
        if (reference[i].get_element_type().is_real()) {
            real_reference.push_back(reference[i]);
            real_actual.push_back(actual[i]);
        } else {
            ++exact_output_count;
            const bool equal =
                reference[i].get_element_type() == actual[i].get_element_type() &&
                reference_info.at("md5") == actual_info.at("md5");
            if (!equal) {
                exact_outputs = false;
                exact_mismatches.push_back(i < output_names.size()
                                               ? output_names[i]
                                               : std::to_string(i));
            }
        }
    }

    json row{{"finite", finite}, {"exact_outputs_equal", exact_outputs},
             {"exact_output_count", exact_output_count},
             {"exact_output_mismatches", exact_mismatches}};
    if (!finite) {
        row["cosine_similarity"] = nullptr;
        row["normalized_rmse"] = nullptr;
        row["mean_abs"] = nullptr;
        row["max_abs"] = nullptr;
        row["abs_error_variance"] = nullptr;
        row["status"] = "FAIL";
        return row;
    }
    bool real_outputs_equal = true;
    if (real_reference.empty()) {
        row["cosine_similarity"] = nullptr;
        row["normalized_rmse"] = nullptr;
        row["mean_abs"] = nullptr;
        row["max_abs"] = nullptr;
        row["abs_error_variance"] = nullptr;
    } else {
        const bool reference_zero = std::all_of(real_reference.begin(), real_reference.end(), zero_tensor);
        const bool actual_zero = std::all_of(real_actual.begin(), real_actual.end(), zero_tensor);
        if (reference_zero || actual_zero) {
            real_outputs_equal = reference_zero && actual_zero;
            row["cosine_similarity"] = nullptr;
            row["normalized_rmse"] = nullptr;
            row["mean_abs"] = nullptr;
            row["max_abs"] = nullptr;
            row["abs_error_variance"] = nullptr;
            row["zero_norm"] = true;
        } else {
            const auto diff = ovbench::compare(real_reference, real_actual);
            row["cosine_similarity"] = diff.cosine;
            row["normalized_rmse"] = diff.normalized_rmse;
            row["mean_abs"] = diff.mean_abs;
            row["max_abs"] = diff.max_abs;
            row["abs_error_variance"] = diff.abs_error_variance;
            real_outputs_equal = diff.cosine > threshold;
        }
    }
    row["status"] = exact_outputs && real_outputs_equal ? "PASS" : "FAIL";
    return row;
}

void bind_inputs(ov::InferRequest& request, const Sample& sample) {
    for (const auto& input : sample.inputs) {
        if (input.first.empty()) request.set_input_tensor(input.second);
        else request.set_tensor(input.first, input.second);
    }
}

std::vector<Sample> samples_for(const ovbench::ModelSpec& spec, const ov::Model& model,
                                const Options& o, const json& manifest) {
    std::map<std::string, json> entries;
    bool found = false;
    if (!o.manifest.empty()) {
        if (manifest.at("schema_version") != 1) throw std::runtime_error("Unsupported dataset schema");
        std::set<std::string> models;
        for (const auto& model : manifest.at("models")) {
            const auto id = model.at("model").get<std::string>();
            if (!models.insert(id).second) throw std::runtime_error("Duplicate model in dataset: " + id);
            if (id != spec.id) continue;
            found = true;
            for (const auto& sample : model.at("samples")) {
                if (!entries.emplace(sample.at("image").get<std::string>(), sample).second)
                    throw std::runtime_error("Duplicate dataset image");
            }
        }
        if (!found) throw std::runtime_error("Model missing from manifest: " + spec.id);
    }
    std::vector<Sample> samples;
    if (o.prepared) {
        for (const auto& [image, entry] : entries) {
            std::vector<std::pair<std::string, ov::Tensor>> inputs;
            if (entry.contains("inputs")) {
                for (const auto& descriptor : entry.at("inputs"))
                    inputs.emplace_back(descriptor.at("name").get<std::string>(),
                                        ovbench::load_tensor(descriptor, o.manifest.parent_path()));
            } else {
                inputs.emplace_back("", ovbench::load_tensor(entry.at("input"), o.manifest.parent_path()));
            }
            samples.push_back({image, std::move(inputs), entry.value("outputs", json::array())});
            if (o.mode == "single") break;
        }
    } else {
        if (spec.family == "prediction" || spec.family == "generic") {
            std::vector<uint32_t> seed_words{
                static_cast<uint32_t>(o.seed), static_cast<uint32_t>(o.seed >> 32)};
            for (const unsigned char character : spec.id) seed_words.push_back(character);
            std::seed_seq seed(seed_words.begin(), seed_words.end());
            std::mt19937_64 generator(seed);
            const auto sample_count = spec.family == "generic" ? o.random_samples : o.input_count;
            for (size_t sample = 0; sample < sample_count; ++sample) {
                std::ostringstream name;
                name << "synthetic_" << std::setfill('0') << std::setw(2) << sample;
                samples.push_back({name.str(), ovbench::random_inputs(spec, model, generator, o.input_shapes), json::array()});
            }
        } else {
            for (const auto& image : image_files(o.images)) {
                const auto name = image.filename().string();
                if (!o.manifest.empty() && !entries.count(name))
                    throw std::runtime_error("Missing image reference: " + name);
                samples.push_back({
                    name,
                    {{"", ovbench::preprocess_image(image, spec.family, spec.input_size, spec.preprocess_scale)}},
                    entries.count(name) ? entries.at(name).value("outputs", json::array()) : json::array()
                });
            }
        }
    }
    if (samples.empty()) throw std::runtime_error("Empty input dataset: " + spec.id);
    if (spec.family == "prediction" || spec.family == "generic") {
        for (const auto& sample : samples) ovbench::validate_inputs(model, sample.inputs);
    } else {
        for (const auto& sample : samples) {
            if (sample.inputs.size() != 1)
                throw std::runtime_error("Image models require exactly one input tensor");
            ovbench::validate_input(model, sample.inputs.front().second);
        }
    }
    return samples;
}

void reshape_model_batch(const std::shared_ptr<ov::Model>& model, int batch) {
    if (batch <= 0) return;
    std::map<std::string, ov::PartialShape> input_shapes;
    for (const auto& input : model->inputs()) {
        auto shape = input.get_partial_shape();
        if (shape.rank().is_dynamic() || shape.rank().get_length() == 0)
            throw std::invalid_argument("Static batch reshape requires ranked, non-scalar model inputs");
        if (shape[0].is_static() && shape[0].get_length() != batch)
            throw std::invalid_argument("Requested batch " + std::to_string(batch) +
                                        " conflicts with static input batch " +
                                        std::to_string(shape[0].get_length()) + " for " + input.get_any_name());
        shape[0] = batch;
        input_shapes.emplace(input.get_any_name(), shape);
    }
    model->reshape(input_shapes);
}

json output_ports(const ov::CompiledModel& compiled) {
    json ports = json::array();
    for (const auto& port : compiled.outputs()) {
        json shape = json::array();
        for (const auto& dimension : port.get_partial_shape())
            shape.push_back(dimension.is_static() ? json(dimension.get_length()) : json(nullptr));
        ports.push_back({{"names", port.get_names()}, {"shape", shape},
                         {"dtype", port.get_element_type().get_type_name()}});
    }
    return ports;
}

json input_ports(const ov::Model& model) {
    json inputs = json::array();
    for (size_t index = 0; index < model.inputs().size(); ++index) {
        const auto input = model.input(index);
        std::ostringstream shape;
        shape << input.get_partial_shape();
        inputs.push_back({{"index", index}, {"names", input.get_names()},
                          {"dtype", input.get_element_type().get_type_name()},
                          {"partial_shape", shape.str()}});
    }
    return inputs;
}

int run(const Options& o) {
    if (o.batches.size() > 1 && !o.start_signal.empty())
        throw std::invalid_argument("Multiple --batch values cannot be combined with --start-signal; run batch variants separately");
    if (o.mode == "gather-guard") return ovbench::run_gather_guard_test(o.cpu_plugin);
    if (o.mode == "self-test") return ovbench::run_random_accuracy_core_tests();
    if (o.mode == "compare-stability") {
        const auto report = ovbench::compare_stability_directories(o.stability_dirs);
        write_json(o.output, report);
        std::cout << report.at("status").get<std::string>() << ": "
                  << report.at("groups").size() << " groups, " << report.at("binary_files") << " binaries\n";
        return report.at("status") == "PASS" ? 0 : 1;
    }
    if (o.batches.size() > 1) {
        if (fs::exists(o.output)) throw std::runtime_error("Output directory already exists: " + o.output.string());
        fs::create_directories(o.output);
        json combined{{"schema_version", 1}, {"status", "RUNNING"}, {"mode", o.mode},
                      {"batch_variants", json::array()}, {"cases", json::array()}};
        bool passed = true;
        for (const auto batch : o.batches) {
            const auto directory = batch_directory_name(batch);
            auto options = o;
            options.batch = batch;
            options.batches = {batch};
            options.batch_explicit = false;
            options.output = o.output / directory;
            const auto code = run(options);
            const auto report = read_json(options.output / "results.json");
            combined["batch_variants"].push_back({{"batch", batch},
                                                   {"directory", directory},
                                                   {"status", report.value("status", "ERROR")}});
            for (auto item : report.at("cases")) {
                item["batch"] = batch;
                item["batch_directory"] = directory;
                combined["cases"].push_back(std::move(item));
            }
            passed = passed && code == 0 && report.value("status", "ERROR") == "PASS";
        }
        combined["status"] = passed ? "PASS" : "FAIL";
        write_json(o.output / "results.json", combined);
        write_summary(o.output / "summary.tsv", combined.at("cases"), true);
        return passed ? 0 : 1;
    }
    const auto ids = o.model == "all" ? std::vector<std::string>{} : split(o.model);
    std::vector<ovbench::ModelSpec> specs;
    if (!o.model_file.empty()) {
        auto id = o.model_file.stem().string();
        if (id.empty() || id == "." || id == "..") id = "custom_model";
        specs.push_back({id, "generic", o.model_file.string(), 0, 0, {}, {}});
    } else if (ids.empty()) specs = ovbench::model_specs();
    else for (const auto& id : ids) specs.push_back(ovbench::find_model(id));
    if (o.mode == "random-accuracy" || o.mode == "tolerance-search") {
        ovbench::RandomAccuracyOptions random;
        random.model_root = o.model_root;
        random.output = o.output;
        random.reference_root = o.reference_root;
        if (random.reference_root.empty()) {
            const auto sibling = fs::absolute(o.model_root).parent_path() / "models";
            random.reference_root = fs::is_directory(sibling) ? sibling : o.model_root;
        }
        random.reference_config = o.reference_config;
        random.shapes = o.random_shapes;
        random.python = o.reference_python;
        random.worker = OVBENCH_REFERENCE_WORKER;
        random.precisions = split(o.precision);
        random.iterations = o.iterations;
        random.seed = o.seed;
        random.threads = o.threads;
        random.timeout_seconds = o.step_timeout_seconds;
        random.threshold = o.threshold;
        random.tolerance_search = o.mode == "tolerance-search";
        random.resample_common_cosine_failures = o.resample_common_cosine_failures;
        random.atol = o.atol;
        random.rtol = o.rtol;
        random.cpu_pinning = o.cpu_pinning;
        random.profiling = o.profiling;
        random.sdpa_bf16 = o.sdpa_bf16;
        return ovbench::run_random_accuracy(specs, random);
    }
    if (o.mode == "single" &&
        (specs.size() != 1 || split(o.precision).size() != 1 || !o.prepared))
        throw std::invalid_argument("Single mode requires one model, one precision, and prepared inputs");
    if (o.mode == "single") {
        struct sigaction action {};
        action.sa_handler = timeout_handler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (::sigaction(SIGALRM, &action, nullptr) != 0)
            throw std::runtime_error("Cannot install single-mode stage timeout handler");
    }
    if (!o.start_signal.empty() &&
        (o.mode != "performance" || specs.size() != 1 || split(o.precision).size() != 1 ||
         !fs::is_directory(o.start_signal.parent_path()) || fs::exists(o.start_signal)))
        throw std::invalid_argument("--start-signal requires one performance case and a new signal in an existing directory");
    const json manifest = o.manifest.empty() ? json::object() : read_json(o.manifest);
    if (fs::exists(o.output)) throw std::runtime_error("Output directory already exists: " + o.output.string());
    fs::create_directories(o.output);
    const auto ov_version = ov::get_openvino_version();
    json report = {
        {"schema_version", 1}, {"status", "RUNNING"},
        {"openvino", {{"build", ov_version.buildNumber}, {"description", ov_version.description}}},
        {"mode", o.mode}, {"threads", o.threads}, {"batch", o.batch}, {"warmup", o.warmup},
        {"iterations", o.duration_seconds ? json(nullptr) : json(o.iterations)},
        {"duration_seconds", o.duration_seconds ? json(*o.duration_seconds) : json(nullptr)},
        {"rounds", o.rounds}, {"threshold", o.threshold},
        {"step_timeout_seconds", o.step_timeout_seconds},
        {"prepared_inputs", o.prepared}, {"sdpa_bf16", o.sdpa_bf16},
        {"random_input_count", o.input_count},
        {"model_root", fs::absolute(o.model_root).string()},
        {"images", o.images.string()}, {"manifest", o.manifest.string()},
        {"manifest_md5", o.manifest.empty() ? std::string{} : ovbench::md5_file(o.manifest)},
           {"reference_backend", manifest.value(
               "reference_backend",
               o.manifest.empty() ? std::string("OpenVINO FP32 self-reference") : std::string{})},
        {"cases", json::array()},
    };
    write_json(o.output / "results.json", report);
    if (o.mode == "single") set_timeout_marker_path(o.output / "timeout.json");
    bool all_passed = true;
    const auto process = ovbench::process_identity();
    report["process_identity"] = process;
    try {
    for (const auto& spec : specs) {
        json stage_timings = json::object();
        const auto source_model = ovbench::model_path(spec, o.model_root);
        json provenance;
        if (source_model.extension() == ".xml") {
            auto bin = source_model;
            bin.replace_extension(".bin");
            provenance = {
                {"xml", fs::absolute(source_model).string()},
                {"xml_md5", ovbench::md5_file(source_model)},
                {"bin_md5", ovbench::md5_file(bin)},
            };
        } else {
            provenance = {
                {"onnx", fs::absolute(source_model).string()},
                {"onnx_md5", ovbench::md5_file(source_model)},
            };
        }
        auto model = o.mode == "single"
            ? timed_stage(stage_timings, "read_model", o.step_timeout_seconds,
                          TimedStage::ReadModel, [&] {
                              auto loaded = ovbench::load_model(spec, o.model_root);
                              reshape_model_batch(loaded, o.batch);
                              return loaded;
                          })
            : ovbench::load_model(spec, o.model_root);
        if (o.mode != "single") reshape_model_batch(model, o.batch);
        const auto model_dir = o.output / safe_id(spec.id);
        fs::create_directory(model_dir);
        if (spec.family == "generic") {
            report["input_analysis"] = {{"model_file", o.model_file.string()},
                                         {"seed", o.seed}, {"inputs", input_ports(*model)}};
            write_json(o.output / "results.json", report);
            write_json(model_dir / "input_analysis.json", report["input_analysis"]);
            std::cout << "MODEL_INPUTS " << report["input_analysis"].dump() << std::endl;
        }
        auto samples = o.mode == "single"
            ? timed_stage(stage_timings, "load_inputs", o.step_timeout_seconds,
                          TimedStage::LoadInputs, [&] {
                              return samples_for(spec, *model, o, manifest);
                          })
            : samples_for(spec, *model, o, manifest);
        if (spec.family == "generic") {
            json inputs = input_ports(*model);
            for (size_t index = 0; index < model->inputs().size(); ++index) {
                json generated = json::array();
                for (const auto& sample : samples)
                    generated.push_back({{"sample", sample.image},
                                         {"tensor", ovbench::tensor_info(sample.inputs[index].second)}});
                inputs[index]["samples"] = generated;
            }
            const json analysis = {
                {"model_file", o.model_file.string()}, {"seed", o.seed},
                {"samples", samples.size()}, {"inputs", inputs},
                {"generation", "Floating: normal(0,0.1); integer: uniform {0,1}; boolean: Bernoulli(0.75)"},
                {"scope", "Synthetic inputs do not infer index bounds, shape tensor values, or application semantics."},
            };
            write_json(model_dir / "input_analysis.json", analysis);
            report["input_analysis"] = analysis;
            write_json(o.output / "results.json", report);
            std::cout << "INPUT_ANALYSIS " << analysis.dump() << std::endl;
        }
        if (o.mode == "preprocess") {
            json rows = json::array();
            for (size_t i = 0; i < samples.size(); ++i) {
                json input_descriptors = json::array();
                for (size_t input = 0; input < samples[i].inputs.size(); ++input) {
                    const auto name = "input_" + std::to_string(i) + "_" +
                                      std::to_string(input) + ".bin";
                    auto descriptor = ovbench::save_tensor(samples[i].inputs[input].second,
                                                           model_dir / name);
                    descriptor["file"] = name;
                    if (!samples[i].inputs[input].first.empty())
                        descriptor["name"] = samples[i].inputs[input].first;
                    input_descriptors.push_back(std::move(descriptor));
                }
                if (spec.family == "prediction" || spec.family == "generic")
                    rows.push_back({{"image", samples[i].image}, {"inputs", input_descriptors}});
                else
                    rows.push_back({{"image", samples[i].image},
                                    {"input", input_descriptors.front()}});
            }
            write_json(model_dir / "inputs.json", {{"model", spec.id}, {"samples", rows}});
            continue;
        }
        for (const auto& precision : split(o.precision)) {
            const auto dir = model_dir / precision;
            fs::create_directory(dir);
            auto graph = model->clone();
            const auto patched = ovbench::apply_default_sdpa_bf16(spec, precision, graph, o.sdpa_bf16);
            const auto setup_start = Clock::now();
            std::unique_ptr<ovbench::Session> session_owner;
            if (o.mode == "single") {
                timed_stage(stage_timings, "compile_model", o.step_timeout_seconds,
                            TimedStage::CompileModel, [&] {
                                session_owner = std::make_unique<ovbench::Session>(
                                    graph, precision, o.threads, o.cpu_pinning, o.profiling, o.hyper_threading);
                            });
            } else {
                session_owner = std::make_unique<ovbench::Session>(
                    graph, precision, o.threads, o.cpu_pinning, o.profiling, o.hyper_threading);
            }
            auto& session = *session_owner;
            auto& request = session.request();
            json result = {{"model", spec.id}, {"family", spec.family}, {"precision", precision},
                           {"batch", o.batch},
                           {"sdpa_bf16_nodes", patched}, {"image_count", samples.size()},
                           {"ir", provenance},
                           {"setup_seconds", std::chrono::duration<double>(Clock::now() - setup_start).count()},
                           {"outputs", output_ports(session.compiled())}, {"status", "PASS"}};
            for (const auto* key : {"INFERENCE_NUM_THREADS", "NUM_STREAMS", "PERFORMANCE_HINT",
                                    "INFERENCE_PRECISION_HINT", "PERF_COUNT", "ENABLE_CPU_PINNING",
                                    "ENABLE_HYPER_THREADING"})
                result["effective_config"][key] = session.compiled().get_property(key).as<std::string>();
            if (o.mode == "single") {
                const auto& sample = samples.front();
                timed_stage(stage_timings, "bind_inputs", o.step_timeout_seconds,
                            TimedStage::BindInputs, [&] {
                                bind_inputs(request, sample);
                            });
                timed_stage(stage_timings, "inference", o.step_timeout_seconds,
                            TimedStage::Inference, [&] {
                                request.infer();
                            });
                auto outputs = timed_stage(stage_timings, "copy_outputs", o.step_timeout_seconds,
                                           TimedStage::CopyOutputs, [&] {
                                               return ovbench::copy_outputs(request);
                                           });
                json output_rows = json::array();
                bool finite = true;
                timed_stage(stage_timings, "inspect_outputs", o.step_timeout_seconds,
                            TimedStage::InspectOutputs, [&] {
                                for (size_t port = 0; port < outputs.size(); ++port) {
                                    const auto info = ovbench::tensor_info(outputs[port]);
                                    finite = finite && info.at("finite").get<bool>();
                                    output_rows.push_back({
                                        {"port", port}, {"names", session.compiled().output(port).get_names()},
                                        {"dtype", info.at("dtype")}, {"shape", info.at("shape")},
                                        {"finite", info.at("finite")}, {"md5", info.at("md5")}});
                                }
                            });
                result["step_timings_ms"] = stage_timings;
                result["single_inference"] = {
                    {"sample", sample.image}, {"completed", true}, {"finite_outputs", finite},
                    {"output_count", outputs.size()}, {"outputs", output_rows},
                    {"inference_ms", stage_timings.at("inference")},
                };
                if (!finite) result["status"] = "FAIL";
                std::cerr << "[single] completed sample=" << sample.image
                          << " finite_outputs=" << (finite ? "true" : "false") << std::endl;
            }
            const auto infer = [&](const Sample& sample) {
                bind_inputs(request, sample);
                request.infer();
            };
            if (o.mode == "performance" || o.mode == "all") {
                size_t sample_index = 0;
                const bool reuse_single_sample = samples.size() == 1;
                const auto next_sample = [&]() -> const Sample& {
                    const auto& sample = samples[sample_index];
                    sample_index = (sample_index + 1) % samples.size();
                    return sample;
                };
                if (reuse_single_sample) bind_inputs(request, samples.front());
                for (size_t iteration = 0; iteration < o.warmup; ++iteration) {
                    if (reuse_single_sample) request.infer();
                    else infer(next_sample());
                }
                if (!o.start_signal.empty()) {
                    write_json(o.output / "ready.json", {{"status", "READY"}});
                    const auto deadline = Clock::now() + std::chrono::minutes(10);
                    while (!fs::exists(o.start_signal)) {
                        if (fs::exists(o.start_signal.parent_path() / "abort.signal"))
                            throw std::runtime_error("Multi-process wave aborted");
                        if (Clock::now() > deadline)
                            throw std::runtime_error("Timed out waiting for start signal");
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                std::vector<double> latencies;
                std::vector<size_t> sample_indices;
                json validity = json::array();
                const auto measurement_start = Clock::now();
                const auto measurement_active = [&]() {
                    if (o.duration_seconds)
                        return latencies.empty() ||
                            std::chrono::duration<double>(Clock::now() - measurement_start).count() < *o.duration_seconds;
                    return latencies.size() < o.iterations;
                };
                while (measurement_active()) {
                    sample_indices.push_back(sample_index);
                    const auto& sample = next_sample();
                    if (!reuse_single_sample) bind_inputs(request, sample);
                    const auto start = Clock::now();
                    request.infer();
                    latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                }
                const auto elapsed_seconds = std::chrono::duration<double>(Clock::now() - measurement_start).count();
                for (size_t iteration = 0; iteration < o.warmup; ++iteration) {
                    if (reuse_single_sample) request.infer();
                    else infer(next_sample());
                }
                const auto batch_size = samples.front().inputs.front().second.get_shape().empty()
                                            ? size_t{1}
                                            : samples.front().inputs.front().second.get_shape().front();
                const auto stats = ovbench::statistics(latencies, batch_size);
                result["performance"] = {
                    {"mean_ms", stats.mean_ms}, {"p50_ms", stats.p50_ms}, {"p90_ms", stats.p90_ms},
                    {"p99_ms", stats.p99_ms},
                    {"throughput_fps", stats.samples * batch_size / elapsed_seconds},
                    {"inference_throughput_fps", stats.throughput_fps},
                    {"samples", stats.samples}, {"iterations", stats.samples},
                    {"measurement_mode", o.duration_seconds ? "duration" : "iterations"},
                    {"requested_iterations", o.duration_seconds ? json(nullptr) : json(o.iterations)},
                    {"requested_duration_seconds", o.duration_seconds ? json(*o.duration_seconds) : json(nullptr)},
                    {"elapsed_seconds", elapsed_seconds},
                    {"warmup_iterations", o.warmup}, {"cooldown_iterations", o.warmup},
                    {"output_files_saved", o.save_performance_outputs},
                    {"input_sequence", "cyclic; continues across warmup, performance and cooldown"},
                    {"latency_unit", "single_input_batch"}, {"batch_size", batch_size},
                };
                write_json(dir / "latencies_ms.json", latencies);
                write_json(dir / "sample_indices.json", sample_indices);
                for (size_t i = 0; i < samples.size(); ++i) {
                    infer(samples[i]);
                    const auto outputs = ovbench::copy_outputs(request);
                    bool finite = true;
                    json output_metadata = json::array();
                    for (size_t p = 0; p < outputs.size(); ++p) {
                        auto descriptor = o.save_performance_outputs
                            ? ovbench::save_tensor(outputs[p], dir / ("check_" + std::to_string(i) + "_" + std::to_string(p) + ".bin"))
                            : ovbench::tensor_info(outputs[p]);
                        finite = finite && descriptor.at("finite").get<bool>();
                        descriptor["output_port"] = p;
                        output_metadata.push_back(std::move(descriptor));
                    }
                    validity.push_back({{"image", samples[i].image}, {"finite", finite},
                                        {"outputs", output_metadata}});
                    if (!finite) result["status"] = "FAIL";
                }
                write_json(dir / "output_validity.json", validity);
                result["output_validity"] = "output_validity.json";
            }
            if (o.mode == "accuracy" || o.mode == "all") {
                json comparisons = json::array();
                size_t passed = 0;
                const bool native_self_reference = std::all_of(
                    samples.begin(), samples.end(), [](const Sample& sample) {
                        return sample.references.empty();
                    });
                std::unique_ptr<ovbench::Session> native_reference_session;
                if (native_self_reference && precision != "FP32") {
                    native_reference_session = std::make_unique<ovbench::Session>(
                        model->clone(), "FP32", o.threads, o.cpu_pinning, o.profiling, o.hyper_threading);
                }
                for (const auto& sample : samples) {
                    if (!native_self_reference &&
                        sample.references.size() != session.compiled().outputs().size())
                        throw std::runtime_error("Reference output count mismatch: " + sample.image);
                    std::vector<ov::Tensor> reference;
                    if (native_self_reference) {
                        if (precision == "FP32") {
                            infer(sample);
                            reference = ovbench::copy_outputs(request);
                        } else {
                            auto& reference_request = native_reference_session->request();
                            bind_inputs(reference_request, sample);
                            reference_request.infer();
                            reference = ovbench::copy_outputs(reference_request);
                        }
                    } else {
                        size_t port_index = 0;
                        for (const auto& desc : sample.references) {
                            if (desc.contains("name") && !session.compiled().output(port_index).get_names().count(desc.at("name").get<std::string>()))
                                throw std::runtime_error("Reference output name mismatch: " + sample.image);
                            reference.push_back(ovbench::load_tensor(desc, o.manifest.parent_path()));
                            ++port_index;
                        }
                    }
                    infer(sample);
                    json row;
                    if (spec.family == "prediction" || spec.family == "generic") {
                        row = compare_prediction_outputs(reference, ovbench::copy_outputs(request),
                                                         spec.output_names, o.threshold);
                    } else {
                        const auto diff = ovbench::compare(reference, ovbench::copy_outputs(request));
                        const bool pass = diff.finite && diff.cosine > o.threshold;
                        row = {{"finite", diff.finite}, {"status", pass ? "PASS" : "FAIL"}};
                        row["cosine_similarity"] = diff.finite ? json(diff.cosine) : json(nullptr);
                        row["normalized_rmse"] = diff.finite ? json(diff.normalized_rmse) : json(nullptr);
                        row["mean_abs"] = diff.finite ? json(diff.mean_abs) : json(nullptr);
                        row["max_abs"] = diff.finite ? json(diff.max_abs) : json(nullptr);
                        row["abs_error_variance"] = diff.finite ? json(diff.abs_error_variance) : json(nullptr);
                    }
                    const bool pass = row.at("status") == "PASS";
                    if (pass) ++passed;
                    row["image"] = sample.image;
                    comparisons.push_back(row);
                }
                result["accuracy"] = {{"passed_images", passed}, {"images", samples.size()}, {"records", comparisons}};
                if (passed != samples.size()) result["status"] = "FAIL";
            }
            if (o.mode == "stability" || o.mode == "all") {
                std::vector<json> first(samples.size());
                json files = json::array();
                bool consistent = true, finite = true;
                for (size_t round = 0; round < o.rounds; ++round) {
                    const auto folder = dir / ("round_" + std::to_string(round + 1));
                    fs::create_directory(folder);
                    for (size_t i = 0; i < samples.size(); ++i) {
                        infer(samples[i]);
                        const auto outputs = ovbench::copy_outputs(request);
                        if (round == 0) first[i] = json::array();
                        for (size_t p = 0; p < outputs.size(); ++p) {
                            const auto configuration = patched ? "sdpa_bf16" : "original";
                            const auto path = folder / ovbench::stability_filename({
                                samples[i].image, spec.id, precision, configuration, process, round + 1, p});
                            auto descriptor = ovbench::save_tensor(outputs[p], path);
                            if (round == 0) first[i].push_back(descriptor);
                            const auto& base = first[i].at(p);
                            bool equal = true;
                            for (const auto& key : {"md5", "dtype", "shape", "bytes"})
                                equal = equal && descriptor.at(key) == base.at(key);
                            consistent = consistent && equal;
                            finite = finite && descriptor.at("finite").get<bool>();
                            descriptor["file"] = fs::relative(path, o.output).generic_string();
                            descriptor["round"] = round + 1;
                            descriptor["image"] = samples[i].image;
                            descriptor["output_port"] = p;
                            descriptor["process"] = process;
                            descriptor["configuration"] = configuration;
                            descriptor["matches_round_1"] = equal;
                            files.push_back(descriptor);
                        }
                    }
                }
                write_json(dir / "files.json", files);
                result["stability"] = {{"rounds", o.rounds}, {"binary_files", files.size()},
                                       {"md5_consistent", consistent}, {"all_outputs_finite", finite}};
                if (!consistent || !finite) result["status"] = "FAIL";
            }
            if (o.mode != "single") {
                write_json(dir / "runtime_graph.json", ovbench::runtime_info(session.compiled()));
                result["runtime_graph"] = "runtime_graph.json";
            }
            all_passed = all_passed && result["status"] == "PASS";
            report["cases"].push_back(result);
            write_json(dir / "result.json", result);
            write_json(o.output / "results.json", report);
            write_summary(o.output / "summary.tsv", report.at("cases"), o.batch_explicit);
            std::cout << spec.id << ' ' << precision << ' ' << result.at("status") << std::endl;
        }
    }
    report["status"] = all_passed ? "PASS" : "FAIL";
    write_json(o.output / "results.json", report);
    return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        report["status"] = "ERROR";
        report["error"] = error.what();
        write_json(o.output / "results.json", report);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse(argc, argv);
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
