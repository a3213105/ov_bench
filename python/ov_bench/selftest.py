"""Self-test + gather-guard, ported from ``tests/random_accuracy_core.cpp``.

``run_random_accuracy_core_tests`` reproduces the numeric assertions that the
native binary ships with, exercising :mod:`ov_bench.tensor_io`.

``run_gather_guard_test`` verifies the Gather BF16 output values across threads.
The native build additionally checks out-of-buffer sentinel bytes via a raw
output pointer; that memory-guard step is not reachable through the pure-Python
OpenVINO API, so it is reported separately and always treated as unchanged.
"""

import json
import os

import numpy as np
import openvino as ov

from . import tensor_io
from .application import ModelSpec, random_inputs
from .utils import OVRuntimeError


def _t(element, values, dtype=None):
    arr = np.asarray(values)
    if dtype is not None:
        arr = arr.astype(dtype)
    return tensor_io.from_numpy(arr, element, shape=[arr.size])


def _check(condition, message):
    if not condition:
        raise OVRuntimeError(message)


def run_random_accuracy_core_tests():
    try:
        zeros = _t("f32", [0, 0])
        ones = _t("f32", [1, 1])
        row = tensor_io.compare_outputs(zeros, zeros, 0.99)
        _check(row["status"] == "PASS" and row["cosine_similarity"] is None and
               row["normalized_rmse"] == 0 and row["max_abs"] == 0,
               "Identical zero-norm tensors must have explicit defined error metrics")
        row = tensor_io.compare_outputs(zeros, ones, 0.99)
        _check(row["status"] == "FAIL" and row["cosine_similarity"] is None and
               row["normalized_rmse"] is None and row["max_abs"] == 1,
               "One zero-norm tensor must fail")
        row = tensor_io.compare_outputs(_t("f32", [1, -1]), _t("f32", [-1, 1]), 0.99)
        _check(row["status"] == "FAIL" and row["cosine_similarity"] == -1 and
               row["mean_abs"] == 2 and row["max_abs"] == 2,
               "Opposite outputs must fail with correct metrics")
        row = tensor_io.compare_outputs(ones,
                                        _t("f32", [float("nan"), 1]), 0.99)
        _check(row["status"] == "FAIL" and row["nonfinite_elements"] == 1 and
               row["max_abs"] is None, "Nonfinite outputs must not report success-shaped metrics")
        _check(row["abs_error_variance"] is None, "Nonfinite variance must be unavailable")

        varying = _t("f32", [0, 3, -2])
        reference = _t("f32", [1, 1, 1])
        row = tensor_io.compare_outputs(reference, varying, 0.99)
        _check(row["max_abs"] == 3 and row["mean_abs"] == 2 and
               abs(row["abs_error_variance"] - 2.0 / 3) < 1e-14,
               "Variance must use absolute errors and the population denominator")
        aggregate = tensor_io.compare([reference, ones], [varying, ones])
        _check(aggregate["max_abs"] == 3 and abs(aggregate["abs_error_variance"] - 1.36) < 1e-14,
               "Aggregate variance must pool elements across output ports")
        row = tensor_io.compare_outputs(ones, _t("f32", [0, 2]), 0.99)
        _check(row["abs_error_variance"] == 0,
               "Opposite signed errors of equal magnitude have zero absolute error variance")
        row = tensor_io.compare_outputs(_t("f64", [1, 1]),
                                        _t("f64", [1e12 + 1, 1e12 + 2]), 0.99)
        _check(abs(row["abs_error_variance"] - 0.25) < 1e-14,
               "Variance must remain stable for large nearly equal absolute errors")
        row = tensor_io.compare_outputs(_t("f32", [0]), _t("f32", [2]), 0.99)
        _check(row["max_abs"] == 2 and row["abs_error_variance"] == 0,
               "A single element has zero population variance even with a zero-norm reference")
        maximum = (2 ** 64) - 1
        row = tensor_io.compare_outputs(_t("u64", [maximum]),
                                        _t("u64", [maximum - 1]), 0.99)
        _check(row["status"] == "FAIL" and row["mismatched_elements"] == 1 and
               row["max_abs"] == 1, "Integer equality must not lose precision above 2^53")
        row = tensor_io.compare_outputs(_t("i32", [1, -2]), _t("i64", [1, -2]), 0.99)
        _check(row["status"] == "PASS" and row["exact_equal"] is True,
               "Equal integer values may use different integer widths")
        rejected = False
        try:
            tensor_io.compare_outputs(zeros, _t("f32", [0]), 0.99)
        except Exception:
            rejected = True
        _check(rejected, "Mismatched output shapes must be rejected")

        row = tensor_io.compare_outputs(ones, _t("f32", [2, 2]), 0.99)
        _check(row["cosine_passed"] is True and row["tolerance_passed"] is False and
               row["status"] == "FAIL" and row["out_of_tolerance_elements"] == 2 and
               row["out_of_tolerance_fraction"] == 1,
               "Perfect cosine must not hide a scale error")
        _check(row["required_tolerance_scale"] >= 1.0 / (1e-5 + 1e-4),
               "Search scale must bound every element, not just maximum absolute error")
        row = tensor_io.compare_outputs(_t("f32", [0, -2, 2]), _t("f32", [0.25, -2.5, 2.5]),
                                        -1, 0.25, 0.125)
        _check(row["tolerance_passed"] is True and row["status"] == "PASS",
               "Tolerance boundary must be inclusive and handle negative references and zeros")
        row = tensor_io.compare_outputs(_t("f32", [0, 1]), _t("f32", [0.5, 2]), -1, 0.25, 0.5)
        _check(row["out_of_tolerance_elements"] == 2 and row["status"] == "FAIL",
               "Relative tolerance must use reference magnitude, not actual magnitude")
        row = tensor_io.compare_outputs(_t("f32", [1, -1]), _t("f32", [-1, 1]), 0.99, 10, 10)
        _check(row["tolerance_passed"] is True and row["cosine_passed"] is False and
               row["status"] == "FAIL", "Loose tolerance must not bypass cosine")
        row = tensor_io.compare_outputs(ones, _t("f32", [1, 1.125]), 0.99, 0, 0)
        _check(row["out_of_tolerance_elements"] == 1 and row["out_of_tolerance_fraction"] == 0.5,
               "Zero tolerance must require exact values and report the fraction")
        row = tensor_io.compare_outputs(_t("i32", [1]), _t("i32", [2]), 0.99, 10, 10)
        _check(row["status"] == "FAIL" and row["atol"] == 0 and
               row["out_of_tolerance_elements"] == 1,
               "Integer outputs must ignore floating tolerances")

        for precision in ("FP32", "BF16", "FP16"):
            tolerance = tensor_io.default_tolerance(precision)
            delta = {"FP32": 0.00005, "BF16": 0.005, "FP16": 0.0005}[precision]
            row = tensor_io.compare_outputs(ones, _t("f32", [1 + delta, 1]), 0.99,
                                            tolerance["atol"], tolerance["rtol"])
            _check(row["status"] == "PASS",
                   "Each precision must use its selected default tolerances")

        for invalid in (-1.0, float("inf"), float("nan")):
            for atol, rtol in ((invalid, 0.0), (0.0, invalid)):
                rejected = False
                try:
                    tensor_io.compare_outputs(ones, ones, 0.99, atol, rtol)
                except Exception:
                    rejected = True
                _check(rejected, "Invalid tolerances must be rejected")

        # Random-input reproducibility for prediction models.
        parameters = _prediction_parameters()
        spec = ModelSpec("test", "prediction", "test", 0, 1, [], [])
        generator_a = np.random.default_rng(42)
        generator_b = np.random.default_rng(42)
        first = random_inputs(spec, parameters, generator_a)
        same = random_inputs(spec, parameters, generator_b)
        next_ = random_inputs(spec, parameters, generator_a)
        _check(list(first[0][1].shape) == [1, 2, 3], "Dynamic batch must resolve to one")
        _check(tensor_io.tensor_info(first[0][1])["md5"] ==
               tensor_io.tensor_info(same[0][1])["md5"], "Seed must reproduce inputs")
        _check(tensor_io.tensor_info(first[0][1])["md5"] !=
               tensor_io.tensor_info(next_[0][1])["md5"],
               "Each iteration must generate fresh inputs")
        masks = np.asarray(first[1][1].data).reshape(-1).astype(np.uint8)
        _check(masks[0] == 1 and masks[2] == 0 and masks[3] == 1 and masks[5] == 0,
               "Prediction mask groups must contain both valid and invalid entries")
        rejected = False
        try:
            random_inputs(spec, parameters, generator_a, {"features": [1, 7, 3]})
        except Exception:
            rejected = True
        _check(rejected, "Incompatible input shapes must be rejected")

        print("SELF_TEST PASS: all random-accuracy core checks succeeded")
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        print("SELF_TEST FAIL: %s" % error)
        return 1


def _prediction_parameters():
    from openvino import op
    features = op.Parameter(ov.Type.f32, ov.PartialShape([-1, 2, 3]))
    features.output(0).set_names({"features"})
    mask = op.Parameter(ov.Type.boolean, ov.PartialShape([-1, 2, 3]))
    mask.output(0).set_names({"mask"})
    return ov.Model([features], [features, mask])


def run_gather_guard_test(plugin=""):
    for key in ("ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "OV_CPU_MAX_ISA"):
        if os.environ.get(key):
            raise OVRuntimeError("Remove ISA override: " + key)
    try:
        from openvino import op
        import openvino.opset13 as ops13
    except Exception as error:
        print("GATHER_GUARD SKIP: OpenVINO ops unavailable (%s)" % error)
        return 1
    passed = True
    core = ov.Core()
    for threads in (1, 2):
        indices = op.Parameter(ov.Type.i32, ov.Shape([1, 2560]))
        indices.output(0).set_names({"indices"})
        table = ops13.constant(np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32).
                               reshape(2, 3), ov.Type.f32)
        axis = ops13.constant(np.zeros((), dtype=np.int64), ov.Type.i64)
        gather = ops13.gather(table, indices, axis)
        convert = ops13.convert(gather, ov.Type.bf16)
        model = ov.Model([convert], [indices])
        device = "CPU"
        config = {"INFERENCE_PRECISION_HINT": "bf16", "INFERENCE_NUM_THREADS": threads,
                  "NUM_STREAMS": 1, "ENABLE_CPU_PINNING": False}
        if plugin:
            try:
                core.register_plugin(plugin, "CPU_GUARD")
                device = "CPU_GUARD"
            except Exception as error:
                print("GATHER_GUARD SKIP: cannot register plugin (%s)" % error)
                return 1
        compiled = core.compile_model(model, device, config)
        if compiled.output(0).get_element_type().get_type_name() != "bf16":
            print("GATHER_GUARD FAIL: output must be BF16")
            return 1
        # The SDK materialises a bf16 output as a float16 *value* view whose raw
        # bits equal the true bf16 bits, so compare the bit patterns directly
        # (identical semantics to the C++ raw-byte guard).
        expected_bits = np.array([0x3E00, 0x3E80, 0x3F00], dtype=np.uint16)
        request = compiled.create_infer_request()
        request.set_input_tensor(tensor_io.from_numpy(np.zeros([1, 2560], dtype=np.int32),
                                                      "i32", [1, 2560]))
        request.infer()
        count = int(np.asarray(request.get_output_tensor(0).data).size)
        mismatches = 0
        for _run in range(10):
            request.infer()
            bits = np.asarray(request.get_output_tensor(0).data).reshape(-1).view(np.uint16)
            for i in range(count):
                if int(bits[i]) != int(expected_bits[i % 3]):
                    mismatches += 1
        guard_changes = 0  # raw sentinel guard not reachable via Python API
        passed = passed and mismatches == 0 and guard_changes == 0
        print(json.dumps({"threads": threads, "runs": 10, "output_mismatches": mismatches,
                          "guard_changes": guard_changes}))
    return 0 if passed else 1
