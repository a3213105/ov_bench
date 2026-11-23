#include "ov_bench/application.hpp"
#include "ov_bench/core.hpp"

#include <onnxruntime_cxx_api.h>
#include <openvino/runtime/properties.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#if __GLIBC_PREREQ(2, 33)
#define OVBENCH_HAS_MALLINFO2 1
#endif
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Inputs = std::vector<std::pair<std::string, ov::Tensor>>;

namespace {
json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open JSON: " + path.string());
    return json::parse(stream);
}

void write_json(const fs::path& path, const json& value) {
    std::ofstream stream(path);
    stream << value.dump(2) << '\n';
    stream.flush();
    if (!stream) throw std::runtime_error("Cannot write JSON: " + path.string());
}

json memory() {
    json result = json::object();
    for (const auto* name : {"smaps_rollup", "status"}) {
        std::ifstream stream(std::string("/proc/self/") + name);
        if (!stream) throw std::runtime_error(std::string("Cannot read /proc/self/") + name);
        std::string line;
        while (std::getline(stream, line)) {
            std::istringstream fields(line);
            std::string key, unit;
            uint64_t value = 0;
            fields >> key;
            if (key == "Rss:" || key == "Pss:" || key == "Private_Clean:" ||
                key == "Private_Dirty:" || key == "Anonymous:" || key == "VmHWM:" ||
                key == "VmSize:") {
                if (!(fields >> value >> unit) || unit != "kB")
                    throw std::runtime_error("Invalid proc memory field: " + line);
                key.pop_back();
                result[key + "_kib"] = value;
            } else if (key == "Threads:") {
                if (!(fields >> value)) throw std::runtime_error("Invalid proc Threads field");
                result["threads"] = value;
            } else if (key == "Cpus_allowed_list:") {
                fields >> unit;
                result["affinity"] = unit;
            }
        }
        if (stream.bad()) throw std::runtime_error("Cannot finish reading proc memory");
    }
    for (const auto* key : {"Rss_kib", "Pss_kib", "Private_Clean_kib", "Private_Dirty_kib",
                            "Anonymous_kib", "VmHWM_kib", "VmSize_kib", "threads", "affinity"})
        if (!result.contains(key)) throw std::runtime_error(std::string("Missing proc field: ") + key);
    result["private_kib"] = result.at("Private_Clean_kib").get<uint64_t>() +
                            result.at("Private_Dirty_kib").get<uint64_t>();
    result["uss_kib"] = result.at("private_kib");
#if defined(OVBENCH_HAS_MALLINFO2)
    const auto allocator = mallinfo2();
    result["allocator_uordblks_bytes"] = allocator.uordblks;
    result["allocator_fordblks_bytes"] = allocator.fordblks;
    result["allocator_hblkhd_bytes"] = allocator.hblkhd;
#endif
    return result;
}

class Recorder {
    std::ofstream stream_;
    Clock::time_point start_ = Clock::now();
public:
    explicit Recorder(const fs::path& path) : stream_(path) {
        if (!stream_) throw std::runtime_error("Cannot open memory stage log");
    }
    void stage(const std::string& name, size_t iteration = 0, double infer_ms = 0) {
        auto row = memory();
        row["stage"] = name;
        row["iteration"] = iteration;
        row["elapsed_seconds"] = std::chrono::duration<double>(Clock::now() - start_).count();
        row["last_infer_ms"] = infer_ms;
        stream_ << row.dump() << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("Cannot write memory stage log");
        std::cout << name << " iteration=" << iteration << " rss_mib="
                  << row.at("Rss_kib").get<double>() / 1024
                  << " pss_mib=" << row.at("Pss_kib").get<double>() / 1024
                  << " uss_mib=" << row.at("uss_kib").get<double>() / 1024 << std::endl;
    }
};

size_t positive(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw ovbench::OVBError("Expected positive integer: " + text);
    const auto n = std::stoull(text);
    if (!n || n > 1000000000) throw ovbench::OVBError("Integer out of range: " + text);
    return static_cast<size_t>(n);
}

void prepare(const fs::path& root, const fs::path& output) {
    json manifest = json::object();
    for (const auto& spec : ovbench::model_specs()) {
        if (spec.family != "prediction") continue;
        const auto model = ovbench::load_model(spec, root);
        std::vector<uint32_t> words{43, 0};
        for (const unsigned char c : spec.id) words.push_back(c);
        std::seed_seq seed(words.begin(), words.end());
        std::mt19937_64 generator(seed);
        json samples = json::array();
        for (size_t i = 0; i < 10; ++i) {
            const auto directory = output / spec.id / std::to_string(i);
            fs::create_directories(directory);
            json sample = json::array();
            for (const auto& [name, tensor] : ovbench::random_inputs(spec, *model, generator)) {
                const auto file = directory / (name + ".bin");
                auto descriptor = ovbench::save_tensor(tensor, file);
                descriptor["name"] = name;
                descriptor["file"] = fs::relative(file, output).string();
                sample.push_back(descriptor);
            }
            samples.push_back(sample);
        }
        manifest[spec.id] = samples;
    }
    write_json(output / "inputs.json", manifest);
}

std::vector<Inputs> load_inputs(const fs::path& manifest, const std::string& model) {
    const auto records = read_json(manifest).at(model);
    if (!records.is_array() || records.empty()) throw std::runtime_error("Empty input samples");
    std::vector<Inputs> samples;
    for (const auto& sample : records) {
        Inputs inputs;
        for (const auto& entry : sample)
            inputs.emplace_back(entry.at("name").get<std::string>(),
                                ovbench::load_tensor(entry, manifest.parent_path()));
        samples.push_back(std::move(inputs));
    }
    return samples;
}

void prepare_three_batches(std::vector<Inputs>& samples) {
    if (samples.size() < 3) throw std::runtime_error("Three input samples are required");
    for (size_t sample = 0; sample < 3; ++sample) {
        const size_t batch = sample + 1;
        for (auto& [name, tensor] : samples[sample]) {
            auto shape = tensor.get_shape();
            if (shape.empty() || shape[0] != 1)
                throw std::runtime_error("Expected a batch-one input tensor: " + name);
            shape[0] = batch;
            ov::Tensor expanded(tensor.get_element_type(), shape);
            const auto bytes = tensor.get_byte_size();
            if (expanded.get_byte_size() != batch * bytes)
                throw std::runtime_error("Unexpected batched tensor size: " + name);
            auto* dst = static_cast<unsigned char*>(expanded.data());
            for (size_t i = 0; i < batch; ++i)
                std::memcpy(dst + i * bytes, tensor.data(), bytes);
            tensor = std::move(expanded);
        }
    }
}

void check_ov(ov::InferRequest& request, size_t count) {
    for (size_t p = 0; p < count; ++p) {
        auto tensor = request.get_output_tensor(p);
        if (tensor.get_element_type() == ov::element::f32) {
            const auto* values = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); ++i)
                if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite OV output");
        } else if (tensor.get_element_type() != ov::element::i32) {
            throw std::runtime_error("Unexpected OV output dtype");
        }
    }
}

bool checkpoint(size_t i, size_t total) {
    return i == 1 || i % 10 == 0 || i == total;
}

void archive_maps(const fs::path& output) {
    std::ifstream maps("/proc/self/maps");
    std::ofstream archive(output / "loaded_maps.txt");
    archive << maps.rdbuf();
    archive.flush();
    if (!maps || !archive) throw std::runtime_error("Cannot archive loaded runtime mappings");
}

void verify_stability(size_t sample, const std::vector<std::string>& hashes,
                      std::vector<std::vector<std::string>>& expected) {
    if (expected.at(sample).empty()) expected[sample] = hashes;
    else if (expected[sample] != hashes) throw std::runtime_error("Same-input output hash changed");
}

void run_ov(const ovbench::ModelSpec& spec, const fs::path& root, const std::string& precision,
            int threads, size_t iterations, const std::vector<Inputs>& samples,
            const fs::path& output, Recorder& recorder, json& result) {
    const auto* plugin_config = std::getenv("OVBENCH_CPU_PLUGIN_CONFIG");
    ov::Core core(plugin_config ? plugin_config : "");
    if (const auto* plugin = std::getenv("OVBENCH_CPU_PLUGIN")) {
        result["cpu_plugin"] = plugin;
    }
    core.get_versions("CPU");
    recorder.stage("plugin_loaded");
    auto model = ovbench::load_model(spec, root);
    const auto forced = ovbench::apply_default_sdpa_bf16(spec, precision, model);
    recorder.stage("model_loaded");
    for (const auto& sample : samples) ovbench::validate_inputs(*model, sample);
    const auto type = precision == "FP32" ? ov::element::f32 :
                      precision == "BF16" ? ov::element::bf16 : ov::element::f16;
    const ov::AnyMap config{
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::num_streams(1), ov::inference_num_threads(threads),
        ov::hint::inference_precision(type), ov::enable_profiling(false),
        ov::hint::enable_cpu_pinning(false)};
    const auto compile_begin = Clock::now();
    auto compiled = core.compile_model(model, "CPU", config);
    result["compile_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - compile_begin).count();
    recorder.stage("model_compiled");
    model.reset();
    recorder.stage("source_model_released");
    auto request = compiled.create_infer_request();
    recorder.stage("request_created");
    for (const auto& [name, tensor] : samples.front()) request.set_tensor(name, tensor);
    recorder.stage("inputs_bound");
    const bool stability = std::getenv("OVBENCH_VERIFY_STABILITY") != nullptr;
    const bool three_batches = std::getenv("OVBENCH_THREE_BATCHES") != nullptr;
    std::vector<std::vector<std::string>> hashes(samples.size());
    if (three_batches) result["inference_shapes"] = json::array();
    double total_ms = 0;
    for (size_t i = 1; i <= iterations; ++i) {
        const auto& sample = samples[(i - 1) % samples.size()];
        if (three_batches)
            std::cout << "inference_start iteration=" << i << " batch="
                      << sample.front().second.get_shape().at(0) << std::endl;
        for (const auto& [name, tensor] : sample)
            request.set_tensor(name, tensor);
        const auto begin = Clock::now();
        request.infer();
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        total_ms += ms;
        if (three_batches || checkpoint(i, iterations)) recorder.stage("inference", i, ms);
        check_ov(request, spec.output_count);
        if (three_batches) {
            json inputs = json::object();
            for (const auto& [name, tensor] : sample) inputs[name] = tensor.get_shape();
            json outputs = json::array();
            for (size_t port = 0; port < spec.output_count; ++port)
                outputs.push_back(request.get_output_tensor(port).get_shape());
            result["inference_shapes"].push_back({{"iteration", i}, {"inputs", inputs}, {"outputs", outputs}});
        }
        if (stability) {
            std::vector<std::string> current;
            for (size_t port = 0; port < spec.output_count; ++port)
                current.push_back(ovbench::tensor_info(request.get_output_tensor(port)).at("md5"));
            verify_stability((i - 1) % samples.size(), current, hashes);
        }
    }
    recorder.stage("steady", iterations);
    result["same_input_stability_checked"] = stability && !three_batches;
    if (three_batches) result["output_hashes_collected"] = stability;
    if (stability) result["sample_output_md5"] = hashes;
    archive_maps(output);
    result["mean_infer_ms"] = total_ms / iterations;
    result["openvino_version"] = ov::get_openvino_version().buildNumber;
    result["sdpa_bf16_forced"] = forced;
    json effective = json::object();
    for (const auto& [key, ignored] : config) {
        (void)ignored;
        std::ostringstream text;
        compiled.get_property(key).print(text);
        effective[key] = text.str();
    }
    result["effective_config"] = effective;
    const auto runtime = ovbench::runtime_info(compiled);
    write_json(output / "runtime_graph.json", runtime);
    json counts = json::object();
    for (const auto& node : runtime.at("nodes")) {
        const auto key = node.at("rt_info").value("runtimePrecision", "unspecified");
        counts[key] = counts.value(key, 0) + 1;
    }
    result["runtime_precision_counts"] = counts;
    // Graph inspection happens after all primary memory measurements.
    recorder.stage("after_runtime_inspection", iterations);
}

void run_ort(const ovbench::ModelSpec& spec, const fs::path& root, int threads, size_t iterations,
             const std::vector<Inputs>& samples, const fs::path& output, Recorder& recorder, json& result) {
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "pred_memory");
    recorder.stage("ort_env_created");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const auto compile_begin = Clock::now();
    Ort::Session session(environment, ovbench::model_path(spec, root).c_str(), options);
    result["compile_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - compile_begin).count();
    recorder.stage("session_created_load_and_optimize");
    if (session.GetInputCount() != spec.input_names.size() ||
        session.GetOutputCount() != spec.output_count)
        throw std::runtime_error("Unexpected ORT model ports");
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::vector<Ort::Value>> ort_samples;
    std::vector<const char*> names;
    for (const auto& [name, tensor] : samples.front()) {
        (void)tensor;
        names.push_back(name.c_str());
    }
    for (const auto& sample : samples) {
        std::vector<Ort::Value> tensors;
        for (const auto& [name, source_tensor] : sample) {
            auto tensor = source_tensor;
            if (name != names.at(tensors.size())) throw std::runtime_error("Inconsistent input order");
            const auto shape = tensor.get_shape();
            const std::vector<int64_t> dimensions(shape.begin(), shape.end());
            const auto dtype = tensor.get_element_type() == ov::element::f32
                ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            if (tensor.get_element_type() != ov::element::f32 &&
                tensor.get_element_type() != ov::element::boolean)
                throw std::runtime_error("Unexpected ORT input dtype");
            tensors.push_back(Ort::Value::CreateTensor(memory_info, tensor.data(), tensor.get_byte_size(),
                              dimensions.data(), dimensions.size(), dtype));
        }
        ort_samples.push_back(std::move(tensors));
    }
    std::vector<const char*> outputs;
    for (const auto& name : spec.output_names) outputs.push_back(name.c_str());
    recorder.stage("inputs_bound");
    double total_ms = 0;
    std::vector<Ort::Value> values;
    for (size_t i = 1; i <= iterations; ++i) {
        // Release previous result before Run: one live output set, as in OV's request.
        values.clear();
        const auto& tensors = ort_samples[(i - 1) % samples.size()];
        const auto begin = Clock::now();
        values = session.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(), tensors.size(),
                             outputs.data(), outputs.size());
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        total_ms += ms;
        if (checkpoint(i, iterations)) recorder.stage("inference", i, ms);
        for (const auto& value : values) {
            const auto info = value.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const auto* data = value.GetTensorData<float>();
                for (size_t j = 0; j < info.GetElementCount(); ++j)
                    if (!std::isfinite(data[j])) throw std::runtime_error("Nonfinite ORT output");
            } else if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                throw std::runtime_error("Unexpected ORT output dtype");
            }
        }
    }
    recorder.stage("steady", iterations);
    archive_maps(output);
    result["mean_infer_ms"] = total_ms / iterations;
    result["onnxruntime_version"] = Ort::GetVersionString();
    result["effective_config"] = {{"intra_op_threads", threads}, {"inter_op_threads", 1},
        {"execution_mode", "ORT_SEQUENTIAL"}, {"graph_optimization", "ORT_ENABLE_ALL"},
        {"cpu_mem_arena", true}, {"memory_pattern", true}, {"provider", "CPUExecutionProvider"}};
}
}

int main(int argc, char** argv) {
    fs::path output;
    json result = {{"status", "RUNNING"}};
    try {
        if (argc == 4 && std::string(argv[1]) == "--prepare") {
            output = argv[3];
            if (!fs::create_directory(output)) throw std::runtime_error("Output already exists");
            prepare(argv[2], output);
            return 0;
        }
        if (argc != 9)
            throw ovbench::OVBError(
                "Usage: pred_memory MODEL_ROOT MANIFEST MODEL BACKEND PRECISION THREADS ITERATIONS OUTPUT"
                " or pred_memory --prepare MODEL_ROOT OUTPUT");
        output = argv[8];
        if (!fs::create_directory(output)) throw std::runtime_error("Output already exists");
        const auto& spec = ovbench::find_model(argv[3]);
        const std::string backend = argv[4], precision = argv[5];
        const int threads = static_cast<int>(positive(argv[6]));
        const auto iterations = positive(argv[7]);
        if (spec.family != "prediction" || (backend != "ov" && backend != "ort") ||
            (precision != "FP32" && precision != "BF16" && precision != "FP16") ||
            (backend == "ort" && precision != "FP32"))
            throw ovbench::OVBError("Invalid model/backend/precision combination");
        result.update({{"model", spec.id}, {"backend", backend}, {"precision", precision},
            {"inference_threads", threads}, {"iterations", iterations}, {"pid", ::getpid()},
            {"source_model_policy", backend == "ov" ? "released immediately after compile" : "ORT owned"},
            {"model_path", ovbench::model_path(spec, argv[1]).string()}});
        Recorder recorder(output / "stages.jsonl");
        recorder.stage("process_baseline");
        auto samples = load_inputs(argv[2], spec.id);
        if (const auto* three_batches = std::getenv("OVBENCH_THREE_BATCHES")) {
            if (std::string(three_batches) != "1" || backend != "ov" || iterations != 3)
                throw ovbench::OVBError("OVBENCH_THREE_BATCHES=1 requires OV and exactly 3 iterations");
            prepare_three_batches(samples);
        }
        recorder.stage("inputs_loaded");
        result["samples"] = samples.size();
        if (backend == "ov")
            run_ov(spec, argv[1], precision, threads, iterations, samples, output, recorder, result);
        else
            run_ort(spec, argv[1], threads, iterations, samples, output, recorder, result);
        recorder.stage("backend_released", iterations);
        result["status"] = "PASS";
        result["finite_outputs_checked_each_iteration"] = true;
        write_json(output / "result.json", result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        result.update({{"status", "ERROR"}, {"error", error.what()}});
        if (!output.empty() && fs::is_directory(output)) write_json(output / "error.json", result);
        return 1;
    }
}
