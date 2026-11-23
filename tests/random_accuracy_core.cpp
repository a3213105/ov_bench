#include "ov_bench/application.hpp"
#include "ov_bench/core.hpp"
#include "ov_bench/self_test.hpp"
#include <openvino/opsets/opset13.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int ovbench::run_gather_guard_test(const std::string& plugin) {
    for (const char* key : {"ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "OV_CPU_MAX_ISA"})
        if (std::getenv(key)) throw std::invalid_argument(std::string("Remove ISA override: ") + key);
    bool passed = true;
    for (int threads : {1, 2}) {
        namespace ops = ov::opset13;
        auto indices = std::make_shared<ops::Parameter>(ov::element::i32, ov::Shape{1, 2560});
        auto table = ops::Constant::create(ov::element::f32, ov::Shape{2, 3},
                                          {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f});
        auto axis = ops::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto gather = std::make_shared<ops::Gather>(table, indices, axis);
        auto convert = std::make_shared<ops::Convert>(gather, ov::element::bf16);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{indices});
        ov::Core core;
        std::string device = "CPU";
        if (!plugin.empty()) {
            device = "CPU_GUARD";
            core.register_plugin(plugin, device);
        }
        auto compiled = core.compile_model(model, device, {
            {"INFERENCE_PRECISION_HINT", "bf16"}, {"INFERENCE_NUM_THREADS", threads},
            {"NUM_STREAMS", 1}, {"ENABLE_CPU_PINNING", false}});
        check(compiled.output().get_element_type() == ov::element::bf16, "Guard output must be BF16");
        auto request = compiled.create_infer_request();
        ov::Tensor input(ov::element::i32, {1, 2560});
        std::fill_n(input.data<int32_t>(), input.get_size(), 0);
        request.set_input_tensor(input);
        const auto shape = compiled.output().get_shape();
        const auto count = ov::shape_size(shape);
        constexpr uint16_t sentinel = 0xA55A;
        std::vector<uint16_t> storage(count * 2 + 64, sentinel);
        request.set_output_tensor(ov::Tensor(ov::element::bf16, shape, storage.data()));
        const uint16_t expected[] = {0x3e00, 0x3e80, 0x3f00};
        size_t mismatches = 0, guard_changes = 0;
        for (int run = 0; run < 10; ++run) {
            std::fill(storage.begin(), storage.end(), sentinel);
            request.infer();
            for (size_t i = 0; i < count; ++i) mismatches += storage[i] != expected[i % 3];
            guard_changes += std::count_if(storage.begin() + count, storage.end(),
                                           [](uint16_t value) { return value != sentinel; });
        }
        passed &= mismatches == 0 && guard_changes == 0;
        std::cout << nlohmann::json{{"threads", threads}, {"runs", 10},
            {"output_mismatches", mismatches}, {"guard_changes", guard_changes}}.dump() << '\n';
    }
    return passed ? 0 : 1;
}

namespace {
template <typename T>
ov::Tensor tensor(ov::element::Type type, std::initializer_list<T> data) {
    ov::Tensor result(type, {data.size()});
    std::copy(data.begin(), data.end(), result.data<T>());
    return result;
}
}

int ovbench::run_random_accuracy_core_tests() {
    try {
        const auto zeros = tensor<float>(ov::element::f32, {0, 0});
        const auto ones = tensor<float>(ov::element::f32, {1, 1});
        auto row = ovbench::compare_output(zeros, zeros, 0.99);
        check(row["status"] == "PASS" && row["cosine_similarity"].is_null() &&
                  row["normalized_rmse"] == 0 && row["max_abs"] == 0,
              "Identical zero-norm tensors must have explicit defined error metrics");
        row = ovbench::compare_output(zeros, ones, 0.99);
        check(row["status"] == "FAIL" && row["cosine_similarity"].is_null() &&
                  row["normalized_rmse"].is_null() && row["max_abs"] == 1,
              "One zero-norm tensor must fail");
        row = ovbench::compare_output(tensor<float>(ov::element::f32, {1, -1}),
                                     tensor<float>(ov::element::f32, {-1, 1}), 0.99);
        check(row["status"] == "FAIL" && row["cosine_similarity"] == -1 &&
                  row["mean_abs"] == 2 && row["max_abs"] == 2,
              "Opposite outputs must fail with correct metrics");
        row = ovbench::compare_output(ones, tensor<float>(ov::element::f32,
                                      {std::numeric_limits<float>::quiet_NaN(), 1}), 0.99);
        check(row["status"] == "FAIL" && row["nonfinite_elements"] == 1 && row["max_abs"].is_null(),
              "Nonfinite outputs must not report success-shaped metrics");
        check(row["abs_error_variance"].is_null(), "Nonfinite variance must be unavailable");
        const auto varying = tensor<float>(ov::element::f32, {0, 3, -2});
        const auto reference = tensor<float>(ov::element::f32, {1, 1, 1});
        row = ovbench::compare_output(reference, varying, 0.99);
        check(row["max_abs"] == 3 && row["mean_abs"] == 2 &&
                  std::abs(row["abs_error_variance"].get<double>() - 2.0 / 3) < 1e-14,
              "Variance must use absolute errors and the population denominator");
        const auto aggregate = ovbench::compare({reference, ones}, {varying, ones});
        check(aggregate.max_abs == 3 && std::abs(aggregate.abs_error_variance - 1.36) < 1e-14,
              "Aggregate variance must pool elements across output ports");
        row = ovbench::compare_output(ones, tensor<float>(ov::element::f32, {0, 2}), 0.99);
        check(row["abs_error_variance"] == 0,
              "Opposite signed errors of equal magnitude have zero absolute error variance");
        row = ovbench::compare_output(tensor<double>(ov::element::f64, {1, 1}),
                                     tensor<double>(ov::element::f64, {1e12 + 1, 1e12 + 2}), 0.99);
        check(std::abs(row["abs_error_variance"].get<double>() - 0.25) < 1e-14,
              "Variance must remain stable for large nearly equal absolute errors");
        row = ovbench::compare_output(tensor<float>(ov::element::f32, {0}),
                                     tensor<float>(ov::element::f32, {2}), 0.99);
        check(row["max_abs"] == 2 && row["abs_error_variance"] == 0,
              "A single element has zero population variance even with a zero-norm reference");
        const auto maximum = std::numeric_limits<uint64_t>::max();
        row = ovbench::compare_output(tensor<uint64_t>(ov::element::u64, {maximum}),
                                     tensor<uint64_t>(ov::element::u64, {maximum - 1}), 0.99);
        check(row["status"] == "FAIL" && row["mismatched_elements"] == 1 && row["max_abs"] == 1,
              "Integer equality must not lose precision above 2^53");
        row = ovbench::compare_output(tensor<int32_t>(ov::element::i32, {1, -2}),
                                     tensor<int64_t>(ov::element::i64, {1, -2}), 0.99);
        check(row["status"] == "PASS" && row["exact_equal"] == true,
              "Equal integer values may use different integer widths");
        bool rejected = false;
        try { (void)ovbench::compare_output(zeros, tensor<float>(ov::element::f32, {0}), 0.99); }
        catch (const ovbench::OVBError&) { rejected = true; }
        check(rejected, "Mismatched output shapes must be rejected");

        row = ovbench::compare_output(ones, tensor<float>(ov::element::f32, {2, 2}), 0.99);
        check(row["cosine_passed"] == true && row["tolerance_passed"] == false &&
                  row["status"] == "FAIL" && row["out_of_tolerance_elements"] == 2 &&
                  row["out_of_tolerance_fraction"] == 1,
              "Perfect cosine must not hide a scale error");
        check(row["required_tolerance_scale"].get<double>() >= 1.0 / (1e-5 + 1e-4),
              "Search scale must bound every element, not just maximum absolute error");
        row = ovbench::compare_output(tensor<float>(ov::element::f32, {0, -2, 2}),
                                     tensor<float>(ov::element::f32, {0.25, -2.5, 2.5}), -1, {0.25, 0.125});
        check(row["tolerance_passed"] == true && row["status"] == "PASS",
              "Tolerance boundary must be inclusive and handle negative references and zeros");
        row = ovbench::compare_output(tensor<float>(ov::element::f32, {0, 1}),
                                     tensor<float>(ov::element::f32, {0.5, 2}), -1, {0.25, 0.5});
        check(row["out_of_tolerance_elements"] == 2 && row["status"] == "FAIL",
              "Relative tolerance must use reference magnitude, not actual magnitude");
        row = ovbench::compare_output(tensor<float>(ov::element::f32, {1, -1}),
                                     tensor<float>(ov::element::f32, {-1, 1}), 0.99, {10, 10});
        check(row["tolerance_passed"] == true && row["cosine_passed"] == false && row["status"] == "FAIL",
              "Loose tolerance must not bypass cosine");
        row = ovbench::compare_output(ones, tensor<float>(ov::element::f32, {1, 1.125}), 0.99, {0, 0});
        check(row["out_of_tolerance_elements"] == 1 && row["out_of_tolerance_fraction"] == 0.5,
              "Zero tolerance must require exact values and report the fraction");
        row = ovbench::compare_output(tensor<int32_t>(ov::element::i32, {1}),
                                     tensor<int32_t>(ov::element::i32, {2}), 0.99, {10, 10});
        check(row["status"] == "FAIL" && row["atol"] == 0 && row["out_of_tolerance_elements"] == 1,
              "Integer outputs must ignore floating tolerances");
        for (const auto& precision : {"FP32", "BF16", "FP16"}) {
            const auto tolerance = ovbench::default_tolerance(precision);
            const auto delta = std::string(precision) == "FP32" ? 0.00005f :
                               std::string(precision) == "BF16" ? 0.005f : 0.0005f;
            row = ovbench::compare_output(ones, tensor<float>(ov::element::f32, {1 + delta, 1}), 0.99, tolerance);
            check(row["status"] == "PASS", "Each precision must use its selected default tolerances");
        }
        for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            for (const auto tolerance : {ovbench::Tolerance{invalid, 0}, ovbench::Tolerance{0, invalid}}) {
                rejected = false;
                try { (void)ovbench::compare_output(ones, ones, 0.99, tolerance); }
                catch (const ovbench::OVBError&) { rejected = true; }
                check(rejected, "Invalid tolerances must be rejected");
            }
        }

        namespace ops = ov::opset13;
        auto features = std::make_shared<ops::Parameter>(ov::element::f32, ov::PartialShape{-1, 2, 3});
        auto mask = std::make_shared<ops::Parameter>(ov::element::boolean, ov::PartialShape{-1, 2, 3});
        features->output(0).set_names({"features"});
        mask->output(0).set_names({"mask"});
        ov::Model model(ov::OutputVector{features}, ov::ParameterVector{features, mask});
      ovbench::ModelSpec spec{"test", "prediction", "test", 0, 1, {}, {}, "", "", 1.0};
        std::mt19937_64 a(42), b(42);
        auto first = ovbench::random_inputs(spec, model, a);
        auto same = ovbench::random_inputs(spec, model, b);
        auto next = ovbench::random_inputs(spec, model, a);
        check(first[0].second.get_shape() == ov::Shape({1, 2, 3}), "Dynamic batch must resolve to one");
        check(ovbench::tensor_info(first[0].second)["md5"] == ovbench::tensor_info(same[0].second)["md5"],
              "Seed must reproduce inputs");
        check(ovbench::tensor_info(first[0].second)["md5"] != ovbench::tensor_info(next[0].second)["md5"],
              "Each iteration must generate fresh inputs");
        const auto* masks = static_cast<const uint8_t*>(first[1].second.data());
        check(masks[0] == 1 && masks[2] == 0 && masks[3] == 1 && masks[5] == 0,
              "Prediction mask groups must contain both valid and invalid entries");
        rejected = false;
        try { (void)ovbench::random_inputs(spec, model, a, {{"features", {1, 7, 3}}}); }
        catch (const ovbench::OVBError&) { rejected = true; }
        check(rejected, "Incompatible input shapes must be rejected");
        auto dynamic = std::make_shared<ops::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 3});
        dynamic->output(0).set_names({"dynamic"});
        ov::Model dynamic_model(ov::OutputVector{dynamic}, ov::ParameterVector{dynamic});
        rejected = false;
        try { (void)ovbench::random_inputs(spec, dynamic_model, a); }
        catch (const ovbench::OVBError&) { rejected = true; }
        check(rejected, "Non-batch dynamic dimensions must not be guessed");
        auto resolved = ovbench::random_inputs(spec, dynamic_model, a, {{"dynamic", {1, 4, 3}}});
        check(resolved[0].second.get_shape() == ov::Shape({1, 4, 3}), "Explicit dynamic shape must be applied");
        dynamic->output(0).set_names({"dynamic", "alias"});
        rejected = false;
        try {
            (void)ovbench::random_inputs(spec, dynamic_model, a,
                                         {{"dynamic", {1, 4, 3}}, {"alias", {1, 5, 3}}});
        } catch (const ovbench::OVBError&) { rejected = true; }
        check(rejected, "Conflicting alias shape overrides must be rejected");
        const ovbench::ModelSpec generic{"custom", "generic", "custom.xml", 0, 0, {}, {}};
        for (const auto& type : {ov::element::f32, ov::element::f16, ov::element::bf16,
                                 ov::element::f64, ov::element::boolean,
                                 ov::element::i8, ov::element::i16, ov::element::i32,
                                 ov::element::i64, ov::element::u8, ov::element::u16,
                                 ov::element::u32, ov::element::u64}) {
            auto input = std::make_shared<ops::Parameter>(type, ov::PartialShape{-1, 16});
            input->output(0).set_names({"custom_input"});
            ov::Model custom(ov::OutputVector{input}, ov::ParameterVector{input});
            std::mt19937_64 first_generator(43), second_generator(43);
            const auto first_inputs = ovbench::random_inputs(generic, custom, first_generator);
            const auto second_inputs = ovbench::random_inputs(generic, custom, second_generator);
            check(first_inputs[0].second.get_element_type() == type, "Generic dtype must be preserved");
            check(first_inputs[0].second.get_shape() == ov::Shape({1, 16}), "Generic batch must resolve to one");
            check(ovbench::tensor_info(first_inputs[0].second) == ovbench::tensor_info(second_inputs[0].second),
                  "Generic input generation must be seed reproducible for every supported dtype");
        }
        auto scalar = std::make_shared<ops::Parameter>(ov::element::i64, ov::Shape{});
        scalar->output(0).set_names({"scalar"});
        ov::Model scalar_model(ov::OutputVector{scalar}, ov::ParameterVector{scalar});
        check(ovbench::random_inputs(generic, scalar_model, a)[0].second.get_shape().empty(),
              "Scalar generic input must remain scalar");
        auto q = std::make_shared<ops::Parameter>(ov::element::f32, ov::Shape{1, 1, 2, 4});
        ov::Output<ov::Node> attention = q;
        for (size_t i = 0; i < 12; ++i)
            attention = std::make_shared<ops::ScaledDotProductAttention>(attention, q, q, false);
        auto attention_model = std::make_shared<ov::Model>(ov::OutputVector{attention}, ov::ParameterVector{q});
        for (const auto& id : {"dinov3-vits16-pretrain-lvd1689m", "dinov3-vits16plus-pretrain-lvd1689m"}) {
            const auto& vit = ovbench::find_model(id);
            for (const auto& precision : {"FP32", "BF16"})
                check(ovbench::apply_default_sdpa_bf16(vit, precision, attention_model->clone()) == 0,
                      "Non-FP16 precisions must not be rewritten");
            check(ovbench::apply_default_sdpa_bf16(vit, "FP16", attention_model->clone(), false) == 0,
                  "Explicit opt-out must disable rewriting");
            auto rewritten = attention_model->clone();
            check(ovbench::apply_default_sdpa_bf16(vit, "FP16", rewritten) == 12,
                  "Both ViT FP16 variants must default to twelve rewritten nodes");
            size_t bf16_attention = 0;
            for (const auto& node : rewritten->get_ordered_ops())
                if (std::string(node->get_type_name()) == "ScaledDotProductAttention" &&
                    node->get_input_element_type(0) == ov::element::bf16)
                    ++bf16_attention;
            check(bf16_attention == 12, "All rewritten SDPA inputs must be BF16");
            rejected = false;
            try { (void)ovbench::apply_default_sdpa_bf16(vit, "FP16", model.clone()); }
            catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "Unexpected ViT graph must report a missing SDPA error");
        }
        for (const auto& candidate : ovbench::model_specs())
            if (candidate.id.find("dinov3-vits16") != 0)
                check(ovbench::apply_default_sdpa_bf16(candidate, "FP16", attention_model->clone()) == 0,
                      "Other models must not be rewritten");
        std::cout << "Random accuracy metric and input tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
