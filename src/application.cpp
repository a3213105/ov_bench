#include "ovbench/application.hpp"
#include "ovbench/core.hpp"

#include <set>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/topk.hpp>

namespace ovbench {
namespace {

template <typename Element>
void fill_random_real(ov::Tensor& tensor, std::mt19937_64& generator) {
    std::normal_distribution<float> distribution(0.f, 0.1f);
    auto* data = tensor.data<Element>();
    for (size_t index = 0; index < tensor.get_size(); ++index)
        data[index] = static_cast<Element>(distribution(generator));
}

template <typename Element>
void fill_random_integer(ov::Tensor& tensor, std::mt19937_64& generator) {
    std::uniform_int_distribution<int> distribution(0, 1);
    auto* data = tensor.data<Element>();
    for (size_t index = 0; index < tensor.get_size(); ++index)
        data[index] = static_cast<Element>(distribution(generator));
}

void fill_generic_random(ov::Tensor& tensor, std::mt19937_64& generator) {
    switch (static_cast<ov::element::Type_t>(tensor.get_element_type())) {
    case ov::element::Type_t::f16: fill_random_real<ov::float16>(tensor, generator); break;
    case ov::element::Type_t::bf16: fill_random_real<ov::bfloat16>(tensor, generator); break;
    case ov::element::Type_t::f64: fill_random_real<double>(tensor, generator); break;
    case ov::element::Type_t::i8: fill_random_integer<int8_t>(tensor, generator); break;
    case ov::element::Type_t::i16: fill_random_integer<int16_t>(tensor, generator); break;
    case ov::element::Type_t::i32: fill_random_integer<int32_t>(tensor, generator); break;
    case ov::element::Type_t::i64: fill_random_integer<int64_t>(tensor, generator); break;
    case ov::element::Type_t::u8: fill_random_integer<uint8_t>(tensor, generator); break;
    case ov::element::Type_t::u16: fill_random_integer<uint16_t>(tensor, generator); break;
    case ov::element::Type_t::u32: fill_random_integer<uint32_t>(tensor, generator); break;
    case ov::element::Type_t::u64: fill_random_integer<uint64_t>(tensor, generator); break;
    default:
        throw std::invalid_argument("Unsupported generic random input dtype: " + tensor.get_element_type().get_type_name());
    }
}

void enable_stable_topk_order(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(node);
        if (!topk) continue;
        topk->set_stable(true);
    }
}

}  // namespace

const std::vector<ModelSpec>& model_specs() {
    static const std::vector<ModelSpec> specs = {
        {"dinov3-convnext-tiny-pretrain-lvd1689m", "dinov3", "dinov3-convnext-tiny-pretrain-lvd1689m", 224, 1, {}, {}},
        {"dinov3-vits16-pretrain-lvd1689m", "dinov3", "dinov3-vits16-pretrain-lvd1689m", 224, 1, {}, {}},
        {"dinov3-vits16plus-pretrain-lvd1689m", "dinov3", "dinov3-vits16plus-pretrain-lvd1689m", 224, 1, {}, {}},
        {"v2.1:640", "yolo-world", "yolo_world_v2_1_m_640", 640, 3, {}, {}},
        {"v2.1:1280", "yolo-world", "yolo_world_v2_1_m_1280", 1280, 3, {}, {}},
        {"v2:640", "yolo-world", "yolo_world_v2_m_640", 640, 3, {}, {}},
        {"v2:1280", "yolo-world", "yolo_world_v2_m_1280", 1280, 3, {}, {}},
        {"unet_carvana_scale0.5_epoch2", "unet", "unet_carvana_scale0.5_epoch2", 0, 1, {}, {},
         "unet_carvana_scale0_5_epoch2", "unet_carvana_scale0.5_epoch2", 0.5},
        {"unet_carvana_scale1.0_epoch2", "unet", "unet_carvana_scale1.0_epoch2", 0, 1, {}, {},
         "unet_carvana_scale1_0_epoch2", "unet_carvana_scale1.0_epoch2", 1.0},
        {"26Q2_round13_run2_e40_no_map_attn", "prediction", "26Q2_round13_run2_e40_no_map_attn", 0, 11,
         {"agents_feature", "agents_mask", "map_objects_feature", "map_objects_mask", "route_feature",
          "route_mask", "lidar_patch_feature", "lidar_patch_mask", "patch_cuboid_feature",
          "patch_cuboid_mask", "lidar_context_feature", "lidar_context_mask"},
         {"trajs", "traj_probs", "yield_prob", "covariance", "traj_box_headings", "traj_speeds",
          "traj_accels", "traj_steerings", "modals_index", "ignore_ego_intention_prob", "fallen_behavior_prob"}},
        {"ped_26Q1_history_data_gen4_800v2_issue2_e23_refine", "prediction",
         "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine", 0, 5,
         {"agents_feature", "agents_mask", "map_objects_feature", "map_objects_mask", "skeleton_feature",
          "skeleton_mask", "route_feature", "route_mask", "lidar_patch_feature", "lidar_patch_mask",
          "patch_cuboid_feature", "patch_cuboid_mask"},
         {"trajs", "traj_probs", "yield_prob", "covariance", "ignore_ego_intention_prob"}},
        {"vehicle_q2_release_v0_w_leading_25", "prediction", "vehicle_q2_release_v0_w_leading_25", 0, 15,
         {"agents_feature", "agents_mask", "map_objects_feature", "map_objects_mask", "lidar_patch_feature",
          "lidar_patch_mask", "patch_cuboid_feature", "patch_cuboid_mask"},
         {"trajs", "traj_probs", "yield_prob", "covariance", "start_to_move_probs", "start_to_move_means",
          "start_to_move_ln_sd", "traj_box_headings", "modals_index", "traj_speeds", "traj_accels",
          "traj_steerings", "vae_scores", "anomaly_scores", "ignore_ego_intention_prob"}},
    };
    return specs;
}

const ModelSpec& find_model(const std::string& id) {
    for (const auto& spec : model_specs())
        if (spec.id == id) return spec;
    throw std::invalid_argument("Unsupported model: " + id);
}

std::filesystem::path model_path(const ModelSpec& spec, const std::filesystem::path& root) {
    if (spec.family == "generic") return spec.artifact;
    const auto artifact_dir = spec.artifact_dir.empty() ? spec.artifact : spec.artifact_dir;
    const auto artifact_file = spec.artifact_file.empty() ? spec.artifact : spec.artifact_file;
    const auto filename = spec.family == "prediction" ? artifact_file + ".onnx" : artifact_file + ".xml";
    if (spec.family == "prediction" && !std::filesystem::exists(root / artifact_dir / filename)) {
        if (std::filesystem::exists(root / filename)) return root / filename;
        if (std::filesystem::exists(root / "pred_model" / filename)) return root / "pred_model" / filename;
    }
    return root / artifact_dir / filename;
}

std::vector<std::pair<std::string, ov::Tensor>> random_inputs(
    const ModelSpec& spec, const ov::Model& model, std::mt19937_64& generator,
    const std::map<std::string, ov::Shape>& shapes) {
    for (const auto& entry : shapes) (void)model.input(entry.first);
    std::normal_distribution<float> normal(0.f, 0.1f);
    std::bernoulli_distribution valid(0.75);
    std::vector<std::pair<std::string, ov::Tensor>> inputs;
    for (const auto& port : model.inputs()) {
        const auto names = port.get_names();
        if (names.empty()) throw std::runtime_error("Random input requires named model inputs");
        const auto name = *std::min_element(names.begin(), names.end());
        ov::Shape shape;
        const auto override = std::find_if(shapes.begin(), shapes.end(), [&](const auto& entry) {
            return names.count(entry.first) != 0;
        });
        if (std::count_if(shapes.begin(), shapes.end(), [&](const auto& entry) {
                return names.count(entry.first) != 0;
            }) > 1)
            throw std::invalid_argument("Multiple random shape overrides for aliases of input: " + name);
        const auto partial = port.get_partial_shape();
        if (override != shapes.end()) {
            shape = override->second;
            if (!partial.compatible(ov::PartialShape(shape)))
                throw std::invalid_argument("Random shape incompatible with input: " + name);
        } else {
            if (partial.rank().is_dynamic())
                throw std::invalid_argument("Random input requires explicit shape for dynamic rank: " + name);
            for (size_t axis = 0; axis < static_cast<size_t>(partial.rank().get_length()); ++axis) {
                if (partial[axis].is_static()) shape.push_back(partial[axis].get_length());
                else if (axis == 0 && partial[axis].compatible(1)) shape.push_back(1);
                else throw std::invalid_argument("Unresolved random input dimension: " + name +
                                                 "; specify --input-shape NAME=DIM[,DIM...] for --model-file");
            }
        }
        if (std::find(shape.begin(), shape.end(), 0) != shape.end())
            throw std::invalid_argument("Random input has empty dimension: " + name);
        ov::Tensor tensor(port.get_element_type(), shape);
        if (tensor.get_element_type() == ov::element::f32) {
            for (size_t i = 0; i < tensor.get_size(); ++i) tensor.data<float>()[i] = normal(generator);
        } else if (tensor.get_element_type() == ov::element::boolean) {
            auto* values = static_cast<uint8_t*>(tensor.data());
            for (size_t i = 0; i < tensor.get_size(); ++i) values[i] = valid(generator);
            if (spec.family == "prediction") {
                const auto group = shape.empty() ? 1 : shape.back();
                for (size_t i = 0; i < tensor.get_size(); i += group) {
                    values[i] = 1;
                    if (group > 1) values[i + group - 1] = 0;
                }
            }
        } else if (spec.family == "generic") {
            fill_generic_random(tensor, generator);
        } else throw std::invalid_argument("Unsupported random input dtype: " + name);
        inputs.emplace_back(name, std::move(tensor));
    }
    validate_inputs(model, inputs);
    return inputs;
}

std::shared_ptr<ov::Model> load_model(const ModelSpec& spec, const std::filesystem::path& root) {
    auto model = read_model(model_path(spec, root));
    if (spec.family == "generic") {
        if (model->inputs().empty() || model->outputs().empty())
            throw std::invalid_argument("Generic testing requires model inputs and outputs");
        std::set<std::string> input_names;
        for (const auto& input : model->inputs())
            for (const auto& name : input.get_names()) input_names.insert(name);
        for (size_t index = 0; index < model->inputs().size(); ++index) {
            auto input = model->input(index);
            if (!input.get_names().empty()) continue;
            auto name = "input_" + std::to_string(index);
            while (input_names.count(name)) name += "_";
            input.get_tensor().set_names({name});
            input_names.insert(name);
        }
        return model;
    }
    enable_stable_topk_order(model);
    if (spec.family == "prediction") {
        if (spec.input_names.size() != model->inputs().size() ||
            model->outputs().size() != spec.output_count ||
            spec.output_names.size() != spec.output_count)
            throw std::runtime_error("Unexpected input/output count for " + spec.id);
        for (const auto& name : spec.input_names) {
            const auto input = model->input(name);
            if (input.get_element_type() != ov::element::f32 &&
                input.get_element_type() != ov::element::boolean)
                throw std::runtime_error("Unsupported prediction input type for " + spec.id + ": " + name);
        }
        for (const auto& name : spec.output_names) {
            const auto output = model->output(name);
            if (output.get_element_type() != ov::element::f32 &&
                output.get_element_type() != ov::element::i32)
                throw std::runtime_error("Unsupported prediction output type for " + spec.id + ": " + name);
        }
        return model;
    }
    if (model->inputs().size() != 1 || model->outputs().size() != spec.output_count)
        throw std::runtime_error("Unexpected input/output count for " + spec.id);
    if (model->input().get_element_type() != ov::element::f32)
        throw std::runtime_error("Expected f32 image input for " + spec.id);

    if (spec.family == "unet") {
        const auto input_shape = model->input().get_partial_shape();
        if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 4)
            throw std::runtime_error("UNet expects 4D NCHW input for " + spec.id);
        if (input_shape[1].is_static() && input_shape[1].get_length() != 3)
            throw std::runtime_error("UNet expects 3-channel input for " + spec.id);
        if (input_shape[0].is_static() && input_shape[0].get_length() != 1)
            throw std::runtime_error("UNet expects batch=1 input for " + spec.id);
        const auto output_shape = model->output().get_partial_shape();
        if (output_shape.rank().is_dynamic() || output_shape.rank().get_length() != 4)
            throw std::runtime_error("UNet expects 4D output for " + spec.id);
        if (model->output().get_element_type() != ov::element::f32)
            throw std::runtime_error("UNet expects f32 output for " + spec.id);
        if (output_shape[1].is_static() && output_shape[1].get_length() != 2)
            throw std::runtime_error("UNet expects 2-class logits output for " + spec.id);
        return model;
    }

    const ov::PartialShape expected{1, 3, static_cast<int64_t>(spec.input_size), static_cast<int64_t>(spec.input_size)};
    if (!model->input().get_partial_shape().compatible(expected))
        throw std::runtime_error("Incompatible NCHW image shape for " + spec.id);
    model->reshape(expected);
    if (spec.family == "yolo-world") {
        for (size_t port = 0; port < spec.output_count; ++port) {
            const auto spatial = static_cast<int64_t>(spec.input_size / (8u << port));
            const ov::PartialShape head{1, 84, spatial, spatial};
            if (!model->output(port).get_partial_shape().compatible(head))
                throw std::runtime_error("YOLO-World raw head shape/order mismatch for " + spec.id);
        }
    } else {
        const auto hidden = spec.id.find("convnext") != std::string::npos ? 768 : 384;
        if (!model->output().get_partial_shape().compatible(ov::PartialShape{1, -1, hidden}))
            throw std::runtime_error("DINOv3 hidden-state shape mismatch for " + spec.id);
    }
    return model;
}

void validate_input(const ov::Model& model, const ov::Tensor& tensor) {
    if (model.input().get_element_type() != tensor.get_element_type() ||
    !model.input().get_partial_shape().compatible(ov::PartialShape(tensor.get_shape())))
        throw std::runtime_error("Input tensor type/shape does not match model input");
}

void validate_inputs(const ov::Model& model,
                     const std::vector<std::pair<std::string, ov::Tensor>>& tensors) {
    if (tensors.size() != model.inputs().size())
        throw std::runtime_error("Input count does not match model input count");
    std::set<std::string> names;
    for (const auto& named_tensor : tensors) {
        if (!names.insert(named_tensor.first).second)
            throw std::runtime_error("Duplicate input name: " + named_tensor.first);
        const auto port = model.input(named_tensor.first);
        if (!port.get_partial_shape().compatible(ov::PartialShape(named_tensor.second.get_shape())) ||
            port.get_element_type() != named_tensor.second.get_element_type())
            throw std::runtime_error("Input tensor type/shape does not match model input: " +
                                     named_tensor.first);
    }
    for (const auto& port : model.inputs()) {
        bool found = false;
        for (const auto& name : port.get_names())
            if (names.count(name)) found = true;
        if (!found)
            throw std::runtime_error("Missing named model input: " + port.get_any_name());
    }
}

size_t apply_default_sdpa_bf16(const ModelSpec& spec, const std::string& precision,
                             const std::shared_ptr<ov::Model>& model, bool enabled) {
    if (!enabled || precision != "FP16" || spec.family != "dinov3" ||
        spec.id.find("dinov3-vits16") != 0)
        return 0;
    const auto count = force_sdpa_bf16(model);
    if (count != 12)
        throw std::runtime_error("Expected 12 SDPA nodes, found " + std::to_string(count));
    return count;
}

size_t force_sdpa_bf16(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa) continue;
        ov::OutputVector inputs;
        for (size_t i = 0; i < sdpa->get_input_size(); ++i) {
            if (i == 3 && sdpa->input_value(i).get_element_type() == ov::element::boolean)
                inputs.emplace_back(sdpa->input_value(i));
            else
                inputs.emplace_back(std::make_shared<ov::op::v0::Convert>(sdpa->input_value(i), ov::element::bf16));
        }
        auto replacement = sdpa->clone_with_new_inputs(inputs);
        auto restored = std::make_shared<ov::op::v0::Convert>(replacement, sdpa->get_output_element_type(0));
        restored->set_friendly_name(sdpa->get_friendly_name() + "/sdpa_bf16");
        restored->output(0).set_names(sdpa->output(0).get_names());
        sdpa->output(0).replace(restored->output(0));
        ++count;
    }
    if (count) model->validate_nodes_and_infer_types();
    return count;
}

}  // namespace ovbench
