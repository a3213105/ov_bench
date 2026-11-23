#include "ovbench/stability.hpp"
#include "ovbench/core.hpp"

#include <openssl/rand.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ovbench {
namespace {
std::string hex(const std::string& value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (unsigned char c : value) {
        out += digits[c >> 4];
        out += digits[c & 15];
    }
    return out;
}

std::string unhex(const std::string& value) {
    if (value.empty() || value.size() % 2) throw std::invalid_argument("Invalid encoded filename field");
    const auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    std::string out;
    for (size_t i = 0; i < value.size(); i += 2)
        out += static_cast<char>((digit(value[i]) << 4) | digit(value[i + 1]));
    return out;
}

std::string encode(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '-')
            out += static_cast<char>(c);
        else out += "%" + hex(std::string(1, static_cast<char>(c)));
    }
    return out;
}

std::string decode(const std::string& value) {
    std::string out;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%') {
            if (i + 2 >= value.size()) throw std::invalid_argument("Invalid percent escape");
            out += unhex(value.substr(i + 1, 2));
            i += 2;
        } else out += value[i];
    }
    return out;
}

std::string random_id() {
    std::array<unsigned char, 8> bytes{};
    if (RAND_bytes(bytes.data(), bytes.size()) != 1)
        throw std::runtime_error("Cannot generate stability filename randomness");
    return hex(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

size_t number(const std::string& value, bool allow_zero) {
    size_t used = 0;
    const auto n = std::stoull(value, &used);
    if (used != value.size() || (!allow_zero && n == 0) || n > SIZE_MAX)
        throw std::invalid_argument("Invalid stability round/port");
    return static_cast<size_t>(n);
}
}

std::string process_identity() {
    return std::to_string(::getpid()) + "-" + random_id();
}

std::string stability_filename(const OutputIdentity& id) {
    if (id.image.empty() || id.model.empty() || id.configuration.empty() || id.round == 0 ||
        !std::regex_match(id.process, std::regex("[0-9]+-[0-9a-f]{16}")) ||
        (id.precision != "FP32" && id.precision != "BF16" && id.precision != "FP16"))
        throw std::invalid_argument("Invalid stability output identity");
    const auto name = "stb2__" + encode(id.image) + "__" + id.precision + "__" + id.process +
           "__" + std::to_string(id.round) + "__" + encode(id.model) +
           "__" + std::to_string(id.port) + "__" + encode(id.configuration) +
           "__" + random_id() + ".bin";
    if (name.size() > 255) throw std::invalid_argument("Encoded stability filename exceeds 255 bytes");
    return name;
}

OutputIdentity parse_stability_filename(const std::string& filename) {
    static const std::regex format(
        R"(^stb2__((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__(FP32|BF16|FP16)__([0-9]+-[0-9a-f]{16})__([0-9]+)__((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__([0-9]+)__((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__[0-9a-f]{16}\.bin$)");
    std::smatch match;
    if (!std::regex_match(filename, match, format))
        throw std::invalid_argument("Malformed stability filename: " + filename);
    return {decode(match[1]), decode(match[5]), match[2], decode(match[7]), match[3],
            number(match[4], false), number(match[6], true)};
}

nlohmann::json compare_stability_directories(const std::vector<std::filesystem::path>& directories) {
    if (directories.empty()) throw std::invalid_argument("At least one result directory is required");
    using Key = std::tuple<std::string, std::string, std::string, size_t, std::string>;
    using json = nlohmann::json;
    std::map<Key, std::vector<json>> groups;
    std::set<std::filesystem::path> seen;
    std::set<std::string> processes;
    size_t ignored = 0;
    for (const auto& directory : directories) {
        if (!std::filesystem::is_directory(directory))
            throw std::invalid_argument("Not a directory: " + directory.string());
        size_t found = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".bin") continue;
            const auto name = entry.path().filename().string();
            if (name.rfind("stb1__", 0) == 0)
                throw std::invalid_argument("Unsupported legacy stability filename (expected stb2): " + name);
            if (name.rfind("stb2__", 0) != 0) { ++ignored; continue; }
            ++found;
            const auto file = std::filesystem::canonical(entry.path());
            if (!seen.insert(file).second) continue;
            const auto id = parse_stability_filename(name);
            processes.insert(id.process);
            groups[{id.model, id.image, id.precision, id.port, id.configuration}].push_back({
                {"file", file.string()}, {"process", id.process}, {"round", id.round},
                {"md5", md5_file(file)}, {"bytes", std::filesystem::file_size(file)},
            });
        }
        if (!found) throw std::runtime_error("No named stability binaries found in: " + directory.string());
    }
    if (processes.size() < 2)
        throw std::runtime_error("Cross-process comparison requires at least two distinct process identities");
    json results = json::array();
    bool passed = true;
    for (const auto& [key, files] : groups) {
        const auto& [model, image, precision, port, configuration] = key;
        std::map<std::string, std::set<size_t>> rounds;
        bool duplicate = false, consistent = true;
        const auto md5 = files.front().at("md5");
        const auto bytes = files.front().at("bytes");
        for (const auto& row : files) {
            duplicate = !rounds[row.at("process").get<std::string>()].insert(row.at("round").get<size_t>()).second || duplicate;
            consistent = consistent && row.at("md5") == md5 && row.at("bytes") == bytes;
        }
        json missing = json::array();
        for (const auto& process : processes)
            if (!rounds.count(process)) missing.push_back(process);
        bool complete = missing.empty();
        const auto expected_rounds = rounds.begin()->second;
        if (expected_rounds.empty() || *expected_rounds.begin() != 1 ||
            *expected_rounds.rbegin() != expected_rounds.size()) complete = false;
        for (const auto& [process, values] : rounds)
            if (values != expected_rounds) complete = false;
        const bool ok = consistent && complete && !duplicate;
        passed = passed && ok;
        results.push_back({
            {"model", model}, {"image", image}, {"precision", precision},
            {"output_port", port}, {"configuration", configuration},
            {"processes", rounds.size()}, {"files", files}, {"md5_consistent", consistent},
            {"missing_processes", missing}, {"rounds_complete", complete},
            {"duplicate_process_round", duplicate}, {"status", ok ? "PASS" : "FAIL"},
        });
    }
    return {{"status", passed ? "PASS" : "FAIL"}, {"processes", processes},
            {"binary_files", seen.size()}, {"ignored_binary_files", ignored}, {"groups", results},
            {"scope", "Filename-based cross-process byte consistency; not a numerical accuracy or finite-value check."}};
}
}
