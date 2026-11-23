#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ovbench {

struct OutputIdentity {
    std::string image, model, precision, configuration, process;
    size_t round, port;
};

std::string process_identity();
std::string stability_filename(const OutputIdentity& identity);
OutputIdentity parse_stability_filename(const std::string& filename);
nlohmann::json compare_stability_directories(const std::vector<std::filesystem::path>& directories);

}
