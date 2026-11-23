#include "ov_bench/random_accuracy.hpp"
#include "ov_bench/core.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <limits>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace ovbench {
namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

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

void write_tsv_row(std::ostream& stream, const std::vector<std::string>& fields) {
    for (size_t index = 0; index < fields.size(); ++index) {
        if (index) stream << '\t';
        stream << tsv_escape(fields[index]);
    }
    stream << '\n';
}

std::string tsv_json_value(const json& value) {
    if (value.is_null()) return {};
    return value.is_string() ? value.get<std::string>() : value.dump();
}

void write_json(const fs::path& path, const json& value) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << value.dump(2) << '\n';
}

json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open JSON: " + path.string());
    return json::parse(stream);
}

std::string safe_id(std::string id) {
    std::replace(id.begin(), id.end(), ':', '_');
    return id;
}

class Scratch {
public:
    fs::path path;
    Scratch() {
        const auto root = fs::temp_directory_path() / ("ovbench-" + std::to_string(::getuid()));
        fs::create_directories(root);
        auto pattern = (root / "random-accuracy-XXXXXXXX").string();
        if (!::mkdtemp(pattern.data())) throw std::runtime_error("Cannot create random accuracy scratch directory");
        path = pattern;
    }
    ~Scratch() {
        std::error_code error;
        fs::remove_all(path, error);
        if (error) std::cerr << "Scratch cleanup failed: " << path << ": " << error.message() << '\n';
    }
};

class ReferenceWorker {
    int fd_ = -1;
    pid_t pid_ = -1;
    unsigned timeout_;
    std::string buffered_;

    void terminate() noexcept {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (pid_ <= 0) return;
        int status = 0;
        for (int i = 0; i < 100; ++i) {
            const auto result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_ || (result < 0 && errno == ECHILD)) { pid_ = -1; return; }
            if (i == 0) ::kill(pid_, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ::kill(pid_, SIGKILL);
        while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
        pid_ = -1;
    }

public:
    ReferenceWorker(const RandomAccuracyOptions& o, const fs::path& config,
                    const fs::path& directory, const fs::path& log) : timeout_(o.timeout_seconds) {
        int sockets[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets))
            throw std::runtime_error("Cannot create reference worker socket");
        posix_spawn_file_actions_t actions;
        int error = ::posix_spawn_file_actions_init(&actions);
        if (error) {
            ::close(sockets[0]); ::close(sockets[1]);
            throw std::runtime_error("Cannot initialize reference worker spawn actions");
        }
        error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
        if (!error) error = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
        if (!error) error = ::posix_spawn_file_actions_addopen(
            &actions, STDERR_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (!error) error = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
        std::vector<std::string> argv{o.python, o.worker, "--config", config.string(),
                                      "--directory", directory.string()};
        std::vector<char*> raw;
        for (auto& arg : argv) raw.push_back(arg.data());
        raw.push_back(nullptr);
        // On large hosts the reference worker's OpenVINO/OpenMP runtimes spin up
        // 2*core-count global arenas and can stall in futex queues during
        // dlopen/init, so READY may never reach us within the step timeout.
        // Cap the worker's thread pools (its inference runs with o.threads).
        std::vector<std::string> env_copy;
        for (char** env_ptr = environ; *env_ptr; ++env_ptr) env_copy.emplace_back(*env_ptr);
        env_copy.push_back("TBB_MAX_THREAD_NUM=" + std::to_string(std::max(4u, 2u * o.threads)));
        env_copy.push_back("OMP_NUM_THREADS=" + std::to_string(std::max(4u, 2u * o.threads)));
        std::vector<char*> env_raw;
        for (auto& entry : env_copy) env_raw.push_back(entry.data());
        env_raw.push_back(nullptr);
        if (!error) error = ::posix_spawnp(&pid_, o.python.c_str(), &actions, nullptr, raw.data(),
                                           env_raw.data());
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(sockets[1]);
        if (error) {
            ::close(sockets[0]);
            throw std::runtime_error("Cannot start reference worker: " + std::string(std::strerror(error)));
        }
        fd_ = sockets[0];
    }
    ~ReferenceWorker() { terminate(); }
    ReferenceWorker(const ReferenceWorker&) = delete;
    ReferenceWorker& operator=(const ReferenceWorker&) = delete;

    void send(const json& message) {
        const auto data = message.dump() + '\n';
        size_t offset = 0;
        while (offset < data.size()) {
            const auto sent = ::send(fd_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            if (sent < 0 && errno == EINTR) continue;
            if (sent <= 0) throw std::runtime_error("Reference worker send failed; see reference_worker.log");
            offset += static_cast<size_t>(sent);
        }
    }
    json receive() {
        const auto deadline = Clock::now() + std::chrono::seconds(timeout_);
        while (true) {
            const auto newline = buffered_.find('\n');
            if (newline != std::string::npos) {
                auto result = json::parse(buffered_.substr(0, newline));
                buffered_.erase(0, newline + 1);
                if (result.value("status", "") == "error")
                    throw std::runtime_error("Reference worker: " + result.at("error").get<std::string>());
                return result;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("Reference worker timed out; see reference_worker.log");
            pollfd descriptor{fd_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, static_cast<int>(std::min<int64_t>(remaining, 1000)));
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::runtime_error("Reference worker poll failed");
            if (!ready) continue;
            char chunk[4096];
            const auto count = ::read(fd_, chunk, sizeof(chunk));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("Reference worker exited without a response; see reference_worker.log");
            buffered_.append(chunk, static_cast<size_t>(count));
            if (buffered_.size() > 16 * 1024 * 1024)
                throw std::runtime_error("Reference worker response exceeds 16 MiB");
        }
    }
    void finish() {
        send({{"command", "close"}});
        ::shutdown(fd_, SHUT_WR);
        const auto deadline = Clock::now() + std::chrono::seconds(timeout_);
        int status;
        while (true) {
            const auto result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_) break;
            if (result < 0 && errno != EINTR) throw std::runtime_error("Cannot reap reference worker");
            if (Clock::now() >= deadline) throw std::runtime_error("Reference worker close timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        pid_ = -1;
        if (!WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("Reference worker exited unsuccessfully; see reference_worker.log");
    }
};

void summary(const fs::path& path, const json& cases) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    write_tsv_row(stream, {"model", "precision", "sdpa_bf16_nodes", "images", "mean_ms", "p99_ms",
                           "throughput_fps", "accuracy_passed", "md5_consistent", "stable_outputs_finite",
                           "status"});
    for (const auto& c : cases)
        write_tsv_row(stream, {c.at("model").get<std::string>(), c.at("precision").get<std::string>(),
                       tsv_json_value(c.at("sdpa_bf16_nodes")), tsv_json_value(c.at("image_count")),
                       "", "", "", tsv_json_value(c.at("accuracy").at("passed_images")),
                               "", "", c.at("status").get<std::string>()});
}

void output_summary(const fs::path& path, const json& cases) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17);
    write_tsv_row(stream, {"model", "precision", "output", "iterations", "passed", "mean_cosine",
                           "min_cosine", "max_abs", "atol", "rtol", "out_of_tolerance_elements",
                           "elements", "out_of_tolerance_fraction", "status", "abs_error_variance"});
    for (const auto& c : cases) {
        struct Totals {
            size_t count = 0, passed = 0, cosine_count = 0, error_count = 0;
            size_t outside = 0, elements = 0;
            double atol = 0, rtol = 0;
            double sum = 0, minimum = 1, maximum = 0;
            size_t error_elements = 0;
            long double error_mean = 0, error_m2 = 0;
        };
        std::map<std::string, Totals> totals;
        for (const auto& iteration : c.at("accuracy").at("records")) {
            for (const auto& output : iteration.at("outputs")) {
                auto& t = totals[output.at("name").get<std::string>()];
                ++t.count;
                t.passed += output.at("status") == "PASS";
                t.outside += output.at("out_of_tolerance_elements").get<size_t>();
                t.elements += output.at("elements").get<size_t>();
                t.atol = output.at("atol").get<double>();
                t.rtol = output.at("rtol").get<double>();
                if (!output.at("cosine_similarity").is_null()) {
                    const auto cosine = output.at("cosine_similarity").get<double>();
                    ++t.cosine_count;
                    t.sum += cosine;
                    t.minimum = std::min(t.minimum, cosine);
                }
                if (!output.at("max_abs").is_null()) {
                    ++t.error_count;
                    t.maximum = std::max(t.maximum, output.at("max_abs").get<double>());
                    const auto n = output.at("elements").get<size_t>();
                    const auto mean = output.at("mean_abs").get<long double>();
                    const auto variance = output.at("abs_error_variance").get<long double>();
                    const auto delta = mean - t.error_mean;
                    const auto combined = t.error_elements + n;
                    t.error_m2 += variance * n +
                        delta * delta * t.error_elements * n / combined;
                    t.error_mean += delta * n / combined;
                    t.error_elements = combined;
                }
            }
        }
        for (const auto& entry : totals) {
            const auto& t = entry.second;
                write_tsv_row(stream, {c.at("model").get<std::string>(), c.at("precision").get<std::string>(),
                            entry.first, std::to_string(t.count), std::to_string(t.passed),
                                   t.cosine_count ? json(t.sum / t.cosine_count).dump() : "",
                                   t.cosine_count ? json(t.minimum).dump() : "",
                                   t.error_count ? json(t.maximum).dump() : "", json(t.atol).dump(),
                                   json(t.rtol).dump(), std::to_string(t.outside), std::to_string(t.elements),
                                   json(static_cast<double>(t.outside) / t.elements).dump(),
                            t.passed == t.count ? "PASS" : "FAIL",
                            t.error_count == t.count && t.error_elements
                                ? json(std::max(0.L, t.error_m2 / t.error_elements)).dump()
                                : ""});
        }
    }
}

void precision_metrics(const fs::path& path, const json& cases) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    write_tsv_row(stream, {"model", "precision", "iteration", "output", "cosine", "max_abs",
                           "normalized_rmse", "mismatched_elements", "status"});
    for (const auto& c : cases) {
        for (const auto& iteration : c.at("accuracy").at("records")) {
            for (const auto& output : iteration.at("outputs")) {
                write_tsv_row(stream, {c.at("model").get<std::string>(), c.at("precision").get<std::string>(),
                                       tsv_json_value(iteration.at("iteration")),
                                output.at("name").get<std::string>(),
                                       tsv_json_value(output.at("cosine_similarity")),
                                       tsv_json_value(output.at("max_abs")),
                                       tsv_json_value(output.at("normalized_rmse")),
                                       tsv_json_value(output.at("mismatched_elements")),
                                output.at("status").get<std::string>()});
            }
        }
    }
}

std::string port_name(const ov::Output<ov::Node>& port, size_t index, const std::string& prefix) {
    const auto names = port.get_names();
    return names.empty() ? prefix + std::to_string(index) : *std::min_element(names.begin(), names.end());
}

}  // namespace

int run_random_accuracy(const std::vector<ModelSpec>& specs, const RandomAccuracyOptions& o) {
    if (o.resample_common_cosine_failures &&
        (o.tolerance_search ||
         std::find(o.precisions.begin(), o.precisions.end(), "BF16") == o.precisions.end() ||
         std::find(o.precisions.begin(), o.precisions.end(), "FP16") == o.precisions.end()))
        throw OVBError("Resampling requires random-accuracy with both BF16 and FP16");
    if (o.tolerance_search && (o.precisions != std::vector<std::string>{"FP32"} ||
                               !o.atol.empty() || !o.rtol.empty()))
        throw OVBError("Tolerance search requires FP32 only with default base tolerances");
    if (fs::exists(o.output)) throw std::runtime_error("Output directory already exists: " + o.output.string());
    fs::create_directories(o.output);
    json report{{"schema_version", 2}, {"status", "RUNNING"}, {"mode", "random-accuracy"},
                {"seed", o.seed}, {"iterations", o.iterations}, {"threads", o.threads},
                {"threshold", o.threshold}, {"cases", json::array()},
                {"openvino", {{"build", ov::get_openvino_version().buildNumber}}},
                {"input_distribution", "FP32 normal(0,0.1); boolean Bernoulli(0.75); prediction mask groups nondegenerate"},
                {"execution", "All selected InferRequests coexist; fresh shared input each iteration; sequential inference"},
                {"pass_criterion", "Every output must pass: all real elements abs(actual-reference) <= atol + rtol*abs(reference) AND cosine > threshold (zero norm requires exact equality); integer/boolean exact equality; nonfinite fails"}};
    if (o.resample_common_cosine_failures)
        report["sampling"] = {
            {"enabled", true}, {"models", json::object()},
            {"criterion", "Accept if all outputs pass cosine checks in BF16 OR all outputs pass cosine checks in FP16"},
            {"selection_bias", "Conditional random sample; not an unfiltered accuracy result"},
            {"max_attempts", nullptr}};
    write_json(o.output / "results.json", report);
    if (o.tolerance_search) {
        report["mode"] = "tolerance-search";
        report["pass_criterion"] = "FP32 fixed-ratio search: base atol=1e-5, rtol=1e-4; scale>=1; cosine failure stops model; final tolerances verified by fresh inference on the same input sequence";
    }
    bool all_passed = true;
    try {
        const auto configs = o.reference_config.empty() ? json::object() : read_json(o.reference_config);
        const auto shape_configs = o.shapes.empty() ? json::object() : read_json(o.shapes);
        if (!configs.is_object() || !shape_configs.is_object())
            throw OVBError("Reference config and random shapes must be objects keyed by model ID");
        json cpu_versions = json::object();
        // OpenVINO 2025.4.2 _gather build corrupts the returned version map when
        // get_versions() is invoked on a prvalue Core() temporary within the same
        // expression (heap nodes are reused by the plugin before the map is read).
        // Keep a named Core and copy the versions out so the loop iterates a stable
        // local map.
        const ov::Core core;
        for (const auto& entry : core.get_versions("CPU"))
            cpu_versions[entry.first] = {{"build", entry.second.buildNumber},
                                         {"description", entry.second.description}};
        report["cpu_plugin"] = cpu_versions;
        for (const auto& spec : specs) {
            auto native_root = o.model_root;
            if (spec.family == "prediction" && !fs::exists(model_path(spec, native_root)))
                native_root = o.reference_root;
            auto model = load_model(spec, native_root);
            const auto folder = o.output / safe_id(spec.id);
            fs::create_directory(folder);
            std::ofstream rejected;
            if (o.resample_common_cosine_failures) {
                rejected.exceptions(std::ios::failbit | std::ios::badbit);
                rejected.open(folder / "rejected_inputs.jsonl");
                fs::create_directory(folder / "rejected_inputs");
                report["sampling"]["models"][spec.id] = {
                    {"attempted", 0}, {"accepted", 0}, {"rejected", 0},
                    {"rejected_log", (fs::path(safe_id(spec.id)) / "rejected_inputs.jsonl").generic_string()}};
            }
            Scratch scratch;
            std::vector<std::string> input_names, output_names;
            for (size_t i = 0; i < model->inputs().size(); ++i)
                input_names.push_back(port_name(model->input(i), i, "input_"));
            for (size_t i = 0; i < model->outputs().size(); ++i)
                output_names.push_back(spec.output_names.empty() ? "output_" + std::to_string(i) : spec.output_names.at(i));
            const json config{{"model", spec.id}, {"family", spec.family}, {"artifact", spec.artifact},
                              {"model_root", fs::absolute(native_root).string()},
                              {"reference_root", fs::absolute(o.reference_root).string()},
                              {"threads", o.threads}, {"input_names", input_names}, {"output_names", output_names},
                              {"reference_config", configs.value(spec.id, json::object())}};
            write_json(folder / "reference_config.json", config);
            ReferenceWorker reference(o, fs::absolute(folder / "reference_config.json"),
                                      scratch.path, fs::absolute(folder / "reference_worker.log"));
            const auto ready = reference.receive();
            if (ready.value("status", "") != "ready")
                throw std::runtime_error("Reference worker did not report ready");
            const bool ov_reference = ready.at("backend") == "OpenVINO FP32";
            std::map<std::string, ov::Shape> shape_overrides;
            if (shape_configs.contains(spec.id))
                shape_overrides = shape_configs.at(spec.id).get<std::map<std::string, ov::Shape>>();
            // Salt the generator with a stable model ID, independent of model selection order.
            std::vector<uint32_t> seed_words{static_cast<uint32_t>(o.seed), static_cast<uint32_t>(o.seed >> 32)};
            for (unsigned char c : spec.id) seed_words.push_back(c);
            std::seed_seq seed(seed_words.begin(), seed_words.end());
            std::mt19937_64 generator(seed);
            std::map<std::string, std::unique_ptr<Session>> sessions;
            std::map<std::string, size_t> patched;
            auto compile = [&](const std::string& precision) {
                auto graph = model->clone();
                patched[precision] = apply_default_sdpa_bf16(spec, precision, graph, o.sdpa_bf16);
                sessions[precision] = std::make_unique<Session>(
                    graph, precision, o.threads, o.cpu_pinning, o.profiling);
            };
            for (const auto& precision : o.precisions) compile(precision);
            if (ov_reference && !sessions.count("FP32")) compile("FP32");
            auto resident_precisions = o.precisions;
            if (ov_reference && std::find(resident_precisions.begin(), resident_precisions.end(), "FP32") ==
                                    resident_precisions.end())
                resident_precisions.push_back("FP32");
            std::map<std::string, size_t> case_indices;
            for (const auto& precision : o.precisions) {
                auto tolerance = default_tolerance(precision);
                if (o.atol.count(precision)) tolerance.atol = o.atol.at(precision);
                if (o.rtol.count(precision)) tolerance.rtol = o.rtol.at(precision);
                case_indices[precision] = report["cases"].size();
                json effective = json::object();
                for (const auto* key : {"INFERENCE_NUM_THREADS", "NUM_STREAMS", "INFERENCE_PRECISION_HINT", "ENABLE_CPU_PINNING"})
                    effective[key] = sessions.at(precision)->compiled().get_property(key).as<std::string>();
                json ports = json::array();
                for (size_t i = 0; i < output_names.size(); ++i)
                    ports.push_back({{"name", output_names[i]}, {"port", i},
                                     {"names", sessions.at(precision)->compiled().output(i).get_names()}});
                report["cases"].push_back({
                    {"model", spec.id}, {"family", spec.family}, {"precision", precision},
                    {"sdpa_bf16_nodes", patched[precision]}, {"image_count", o.iterations},
                    {"reference", ready}, {"seed", o.seed}, {"effective_config", effective},
                    {"source", {{"path", fs::absolute(model_path(spec, native_root)).string()},
                                {"md5", md5_file(model_path(spec, native_root))}}},
                    {"resident_precisions", resident_precisions},
                    {"outputs", ports}, {"status", "RUNNING"},
                    {"accuracy", {{"images", o.iterations}, {"passed_images", 0},
                                  {"atol", tolerance.atol}, {"rtol", tolerance.rtol},
                                  {"records", json::array()}}}});
                fs::create_directory(folder / precision);
            }
            write_json(o.output / "results.json", report);
            bool search_blocked = false;
            double required_scale = 0;
            json limiting_output = nullptr;
            size_t accepted = 0;
            for (size_t iteration = 0; accepted < o.iterations; ++iteration) {
                auto inputs = random_inputs(spec, *model, generator, shape_overrides);
                json input_descriptors = json::array();
                std::vector<fs::path> temporary;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const auto path = scratch.path / ("input_" + std::to_string(iteration) + "_" + std::to_string(i) + ".bin");
                    auto desc = save_tensor(inputs[i].second, path);
                    desc["name"] = inputs[i].first;
                    desc["file"] = path.string();
                    input_descriptors.push_back(desc);
                    temporary.push_back(path);
                }
                auto infer = [&](Session& session) {
                    for (const auto& input : inputs) session.request().set_tensor(input.first, input.second);
                    session.request().infer();
                    return copy_outputs(session.request());
                };
                std::vector<ov::Tensor> baseline;
                if (ov_reference) baseline = infer(*sessions.at("FP32"));
                else {
                    reference.send({{"command", "infer"}, {"iteration", iteration}, {"inputs", input_descriptors}});
                    const auto response = reference.receive();
                    if (response.value("status", "") != "ok" || response.at("outputs").size() != output_names.size())
                        throw std::runtime_error("Invalid reference output response");
                    for (size_t i = 0; i < output_names.size(); ++i) {
                        const auto& desc = response.at("outputs").at(i);
                        if (desc.at("name") != output_names[i])
                            throw std::runtime_error("Reference output name mismatch");
                        const fs::path path = desc.at("file").get<std::string>();
                        if (path.parent_path() != scratch.path)
                            throw std::runtime_error("Reference output must be inside worker scratch directory");
                        baseline.push_back(load_tensor(desc, scratch.path));
                        temporary.push_back(path);
                    }
                }
                std::map<std::string, json> candidate_records;
                bool low_precision_cosine_passed = false;
                for (const auto& precision : o.precisions) {
                    const auto& accuracy_config = report["cases"].at(case_indices.at(precision)).at("accuracy");
                    const Tolerance tolerance{accuracy_config.at("atol").get<double>(),
                                              accuracy_config.at("rtol").get<double>()};
                    const auto actual = ov_reference && precision == "FP32"
                        ? baseline : infer(*sessions.at(precision));
                    if (actual.size() != baseline.size() || actual.size() != output_names.size())
                        throw std::runtime_error("Output count mismatch");
                    json rows = json::array();
                    bool passed = true;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        auto row = compare_output(baseline[i], actual[i], o.threshold, tolerance);
                        row["name"] = output_names[i];
                        row["port"] = i;
                        row["reference_md5"] = tensor_info(baseline[i]).at("md5");
                        row["md5"] = tensor_info(actual[i]).at("md5");
                        passed &= row.at("status") == "PASS";
                        if (o.tolerance_search) {
                            const bool real = baseline[i].get_element_type().is_real();
                            if (!row.at("finite").get<bool>() ||
                                (real ? !row.at("cosine_passed").get<bool>() :
                                        !row.at("exact_equal").get<bool>())) {
                                if (!search_blocked)
                                    limiting_output = {{"iteration", iteration}, {"name", output_names[i]},
                                                       {"reason", "Nonfinite, cosine, or exact integer check failed"}};
                                search_blocked = true;
                            } else if (row.at("required_tolerance_scale").is_null()) {
                                throw std::runtime_error("No finite common tolerance scale for output");
                            } else if (row.at("required_tolerance_scale").get<double>() > required_scale) {
                                required_scale = row.at("required_tolerance_scale").get<double>();
                                if (!search_blocked)
                                    limiting_output = {{"iteration", iteration}, {"name", output_names[i]}};
                            }
                        }
                        rows.push_back(row);
                    }
                    for (size_t i = 0; i < inputs.size(); ++i)
                        if (tensor_info(inputs[i].second).at("md5") != input_descriptors.at(i).at("md5"))
                            throw std::runtime_error("Input modified during inference");
                    auto recorded_inputs = input_descriptors;
                    for (auto& d : recorded_inputs) d.erase("file");
                    if (precision == "BF16" || precision == "FP16")
                        low_precision_cosine_passed |= std::all_of(rows.begin(), rows.end(), [](const json& row) {
                            return row.at("cosine_passed").get<bool>();
                        });
                    candidate_records[precision] = {
                        {"iteration", iteration}, {"image", "random_" + std::to_string(iteration)},
                        {"inputs", recorded_inputs}, {"outputs", rows}, {"status", passed ? "PASS" : "FAIL"}};
                }
                const bool valid = !o.resample_common_cosine_failures || low_precision_cosine_passed;
                if (!valid) {
                    const auto rejected_folder = folder / "rejected_inputs" / ("random_" + std::to_string(iteration));
                    fs::create_directory(rejected_folder);
                    auto saved_inputs = input_descriptors;
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        const auto path = rejected_folder / ("input_" + std::to_string(i) + ".bin");
                        fs::copy_file(temporary.at(i), path);
                        saved_inputs.at(i)["file"] = fs::relative(path, folder).generic_string();
                    }
                    rejected << json({{"model", spec.id}, {"iteration", iteration}, {"seed", o.seed},
                                      {"threshold", o.threshold}, {"reference", ready}, {"inputs", saved_inputs},
                                      {"reason", "Both BF16 and FP16 have at least one failing output cosine check"},
                                      {"precisions", candidate_records}}).dump() << '\n';
                    rejected.flush();
                    std::cerr << spec.id << " rejected random_" << iteration
                              << ": both low precisions failed cosine; accepted " << accepted
                              << '/' << o.iterations << std::endl;
                } else {
                    for (const auto& precision : o.precisions) {
                        auto& record = candidate_records.at(precision);
                        if (o.resample_common_cosine_failures) record["accepted_index"] = accepted;
                        const bool passed = record.at("status") == "PASS";
                        auto& c = report["cases"].at(case_indices.at(precision));
                        auto& accuracy = c["accuracy"];
                        if (passed) accuracy["passed_images"] = accuracy.at("passed_images").get<size_t>() + 1;
                        accuracy["records"].push_back(record);
                        c["status"] = accuracy.at("passed_images").get<size_t>() == accepted + 1 ? "PASS" : "FAIL";
                        if (!o.tolerance_search) all_passed &= passed;
                    }
                    ++accepted;
                }
                if (o.resample_common_cosine_failures) {
                    auto& sampling = report["sampling"]["models"][spec.id];
                    sampling["attempted"] = iteration + 1;
                    sampling["accepted"] = accepted;
                    sampling["rejected"] = iteration + 1 - accepted;
                    for (const auto& precision : o.precisions) {
                        auto& c = report["cases"].at(case_indices.at(precision));
                        c["sampling"] = sampling;
                        c["sampling"]["criterion"] = report.at("sampling").at("criterion");
                        c["sampling"]["selection_bias"] = report.at("sampling").at("selection_bias");
                    }
                }
                for (const auto& precision : o.precisions)
                    write_json(folder / precision / "result.json", report["cases"].at(case_indices.at(precision)));
                for (const auto& path : temporary) fs::remove(path);
                write_json(o.output / "results.json", report);
                summary(o.output / "summary.tsv", report.at("cases"));
                output_summary(o.output / "outputs.tsv", report.at("cases"));
                precision_metrics(o.output / "precision_metrics.tsv", report.at("cases"));
                if (search_blocked) break;
            }
            reference.finish();
            if (o.tolerance_search) {
                auto& c = report["cases"].at(case_indices.at("FP32"));
                c["tolerance_search"] = {{"status", "FAIL"}, {"base_atol", 1e-5}, {"base_rtol", 1e-4},
                    {"minimum_scale_allowed", 1}, {"required_scale", required_scale},
                    {"limiting_output", limiting_output}, {"scale", nullptr}, {"atol", nullptr}, {"rtol", nullptr}};
                auto& search = c["tolerance_search"];
                if (search_blocked) {
                    search["reason"] = "Cosine/nonfinite/integer failure; tolerance relaxation cannot fix this";
                    c["status"] = "FAIL";
                } else {
                    double low = 1, high = 1;
                    json trials = json::array();
                    while (high < required_scale) {
                        trials.push_back({{"scale", high}, {"passed", false}, {"stage", "expand"}});
                        low = high;
                        if (high > std::numeric_limits<double>::max() / 2)
                            throw std::runtime_error("Tolerance search scale overflow");
                        high *= 2;
                    }
                    trials.push_back({{"scale", high}, {"passed", true}, {"stage", "bracket"}});
                    while (high - low > 1e-8 * std::max(1.0, high)) {
                        const auto mid = low + (high - low) / 2;
                        const bool passed = mid >= required_scale;
                        trials.push_back({{"scale", mid}, {"passed", passed}, {"stage", "bisect"}});
                        if (passed) high = mid;
                        else low = mid;
                    }
                    const auto outward_product = [](double base, double scale) {
                        const auto exact = static_cast<long double>(base) * scale;
                        auto value = static_cast<double>(exact);
                        if (static_cast<long double>(value) < exact)
                            value = std::nextafter(value, std::numeric_limits<double>::infinity());
                        return value;
                    };
                    const auto atol = outward_product(1e-5, high), rtol = outward_product(1e-4, high);
                    search["scale"] = high;
                    search["atol"] = atol;
                    search["rtol"] = rtol;
                    search["lower_scale"] = low;
                    search["relative_search_resolution"] = 1e-8;
                    search["trials"] = trials;
                    search["method"] = "Exact per-element max(error/(base_atol+base_rtol*abs(reference))); expansion/bisection of this monotone predicate; fresh full verification";
                    c["calibration_accuracy"] = c.at("accuracy");
                    write_json(o.output / "results.json", report);
                    auto verification = o;
                    verification.tolerance_search = false;
                    verification.output = folder / "verification";
                    verification.atol = {{"FP32", atol}};
                    verification.rtol = {{"FP32", rtol}};
                    const auto code = run_random_accuracy({spec}, verification);
                    const auto verified = read_json(verification.output / "results.json");
                    const auto& verified_case = verified.at("cases").at(0);
                    const auto& original_records = c.at("calibration_accuracy").at("records");
                    const auto& verified_records = verified_case.at("accuracy").at("records");
                    if (original_records.size() != verified_records.size())
                        throw std::runtime_error("Verification input count mismatch");
                    for (size_t i = 0; i < original_records.size(); ++i)
                        if (original_records.at(i).at("inputs") != verified_records.at(i).at("inputs"))
                            throw std::runtime_error("Verification random input hashes differ from calibration");
                    c["accuracy"] = verified_case.at("accuracy");
                    c["status"] = code == 0 ? "PASS" : "FAIL";
                    search["status"] = c.at("status");
                    search["verification_report"] = (verification.output / "results.json").string();
                    search["verification_input_hashes_equal"] = true;
                    if (code != 0)
                        search["reason"] = "Fresh verification failed; no verified minimum claimed";
                }
                all_passed &= c.at("status") == "PASS";
                write_json(folder / "FP32" / "result.json", c);
                write_json(o.output / "results.json", report);
                summary(o.output / "summary.tsv", report.at("cases"));
                output_summary(o.output / "outputs.tsv", report.at("cases"));
                precision_metrics(o.output / "precision_metrics.tsv", report.at("cases"));
            }
            for (const auto& precision : o.precisions)
                std::cout << spec.id << ' ' << precision << ' '
                          << report["cases"].at(case_indices.at(precision)).at("status") << std::endl;
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

}  // namespace ovbench
