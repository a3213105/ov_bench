#pragma once

#include <filesystem>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <random>
#include <openvino/openvino.hpp>

namespace ovbench {

struct ModelSpec {
    std::string id;
    std::string family;
    std::string artifact;
    size_t input_size;
    size_t output_count;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string artifact_dir{};
    std::string artifact_file{};
    double preprocess_scale = 1.0;
};

const std::vector<ModelSpec>& model_specs();
const ModelSpec& find_model(const std::string& id);
std::filesystem::path model_path(const ModelSpec& spec, const std::filesystem::path& root);
std::shared_ptr<ov::Model> load_model(const ModelSpec& spec, const std::filesystem::path& root);
void validate_input(const ov::Model& model, const ov::Tensor& tensor);
void validate_inputs(const ov::Model& model,
                     const std::vector<std::pair<std::string, ov::Tensor>>& tensors);
size_t force_sdpa_bf16(const std::shared_ptr<ov::Model>& model);
size_t apply_default_sdpa_bf16(const ModelSpec& spec, const std::string& precision,
                             const std::shared_ptr<ov::Model>& model, bool enabled = true);
std::vector<std::pair<std::string, ov::Tensor>> random_inputs(
    const ModelSpec& spec, const ov::Model& model, std::mt19937_64& generator,
    const std::map<std::string, ov::Shape>& shapes = {});

}  // namespace ovbench
