"""Native random-accuracy integration tests; set OVBENCH_BINARY to a built runner."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import openvino as ov
from openvino import opset13 as ops


BINARY = os.environ.get("OVBENCH_BINARY")
MODEL = "dinov3-convnext-tiny-pretrain-lvd1689m"


@unittest.skipUnless(BINARY, "Set OVBENCH_BINARY to run native integration tests")
class RandomAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="random-accuracy-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"
        self.folder = self.models / MODEL
        self.folder.mkdir(parents=True)
        self.references = self.root / "references"
        self.references.mkdir()
        self.create_ir()
        self.counter = 0

    def create_ir(self, dtype=None, zero=False, nonfinite=False):
        x = ops.parameter([1, 3, 224, 224], np.float32, name="pixel_values")
        average = ops.reduce_mean(x, ops.constant([1, 2, 3], np.int64), False)
        value = ops.multiply(average, ops.constant(np.float32(0 if zero else 1)))
        if nonfinite:
            value = ops.add(value, ops.constant(np.float32(np.nan)))
        output = ops.broadcast(value, ops.constant([1, 1, 768], np.int64))
        if dtype:
            output = ops.convert(output, dtype)
        output.output(0).set_names({"features"})
        ov.serialize(ov.Model([output], [x]), str(self.folder / f"{MODEL}.xml"))

    def run_native(self, *extra, precisions="FP32,BF16,FP16", iterations=2, seed=7):
        self.counter += 1
        report = self.root / f"report{self.counter}"
        command = [
            BINARY, "--model-root", str(self.models), "--model", MODEL,
            "--mode", "random-accuracy", "--output", str(report),
            "--precisions", precisions, "--iterations", str(iterations),
            "--threads", "2", "--cpu-pinning", "NO",
            "--perf-count", "NO", "--reference-python", sys.executable,
            "--reference-root", str(self.references), *extra,
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        payload = json.loads((report / "results.json").read_text()) if (report / "results.json").exists() else None
        return result, payload, report

    def test_all_requests_share_new_inputs_and_reproducible_seed(self):
        first, a, report = self.run_native()
        self.assertIn(first.returncode, (0, 1), first.stderr)
        self.assertEqual(len(a["cases"]), 3)
        self.assertTrue((report / "summary.tsv").is_file())
        self.assertTrue((report / "outputs.tsv").is_file())
        self.assertTrue((report / "precision_metrics.tsv").is_file())
        hashes = [
            [row["inputs"][0]["md5"] for row in case["accuracy"]["records"]]
            for case in a["cases"]
        ]
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[1], hashes[2])
        self.assertNotEqual(hashes[0][0], hashes[0][1])
        for case in a["cases"]:
            self.assertEqual(case["reference"]["backend"], "OpenVINO FP32")
            self.assertEqual(case["resident_precisions"], ["FP32", "BF16", "FP16"])
            self.assertEqual(len(case["accuracy"]["records"][0]["outputs"]), 1)
            expected = {"FP32": (1e-5, 1e-4), "BF16": (1e-3, 1e-2), "FP16": (1e-4, 1e-3)}
            atol, rtol = expected[case["precision"]]
            self.assertEqual(case["accuracy"]["atol"], atol)
            self.assertEqual(case["accuracy"]["rtol"], rtol)
            for record in case["accuracy"]["records"]:
                for row in record["outputs"]:
                    self.assertEqual((row["atol"], row["rtol"]), (atol, rtol))
                    self.assertEqual(row["tolerance_passed"], row["out_of_tolerance_elements"] == 0)
                    self.assertEqual(row["status"] == "PASS", row["cosine_passed"] and row["tolerance_passed"])
                    self.assertGreaterEqual(row["abs_error_variance"], 0)
        with (report / "outputs.tsv").open() as stream:
            summaries = list(csv.DictReader(stream, delimiter="\t"))
        for case, summary in zip(a["cases"], summaries):
            outputs = [record["outputs"][0] for record in case["accuracy"]["records"]]
            count = sum(row["elements"] for row in outputs)
            mean = sum(row["mean_abs"] * row["elements"] for row in outputs) / count
            variance = sum(row["elements"] *
                           (row["abs_error_variance"] + (row["mean_abs"] - mean) ** 2)
                           for row in outputs) / count
            self.assertAlmostEqual(float(summary["abs_error_variance"]), variance, delta=1e-20)
            self.assertEqual(float(summary["max_abs"]), max(row["max_abs"] for row in outputs))
        with (report / "precision_metrics.tsv").open() as stream:
            metrics = list(csv.DictReader(stream, delimiter="\t"))
        self.assertTrue(
            {"cosine", "max_abs", "normalized_rmse", "mismatched_elements"}
            .issubset(metrics[0])
        )
        self.assertEqual(len(metrics), sum(len(case["accuracy"]["records"]) for case in a["cases"]))
        for case in a["cases"]:
            first_record = case["accuracy"]["records"][0]
            expected = first_record["outputs"][0]
            recorded = next(
                row for row in metrics
                if row["precision"] == case["precision"]
                and row["iteration"] == str(first_record["iteration"])
                and row["output"] == expected["name"]
            )
            self.assertAlmostEqual(float(recorded["cosine"]), expected["cosine_similarity"])
            self.assertAlmostEqual(float(recorded["max_abs"]), expected["max_abs"])
            self.assertAlmostEqual(float(recorded["normalized_rmse"]), expected["normalized_rmse"])
            self.assertEqual(int(recorded["mismatched_elements"]), expected["mismatched_elements"])
        second, b, _ = self.run_native()
        self.assertEqual(second.returncode, first.returncode, second.stderr)
        self.assertEqual(hashes[0], [row["inputs"][0]["md5"] for row in b["cases"][0]["accuracy"]["records"]])
        _, different, _ = self.run_native(seed=8)
        self.assertNotEqual(hashes[0][0], different["cases"][0]["accuracy"]["records"][0]["inputs"][0]["md5"])

    def test_unselected_fp32_reference_is_created(self):
        result, payload, _ = self.run_native(precisions="FP16")
        self.assertIn(result.returncode, (0, 1), result.stderr)
        self.assertEqual(len(payload["cases"]), 1)
        self.assertEqual(payload["cases"][0]["resident_precisions"], ["FP16", "FP32"])

    def test_seed_defaults_to_43_and_can_be_overridden(self):
        _, default, _ = self.run_native(precisions="FP32", seed=None)
        _, explicit, _ = self.run_native(precisions="FP32", seed=43)
        _, override, _ = self.run_native(precisions="FP32", seed=44)
        self.assertEqual(default["seed"], 43)
        self.assertEqual(default["cases"][0]["accuracy"], explicit["cases"][0]["accuracy"])
        self.assertNotEqual(
            default["cases"][0]["accuracy"]["records"][0]["inputs"],
            override["cases"][0]["accuracy"]["records"][0]["inputs"])

    def test_resampling_counts_only_accepted_inputs_and_saves_rejections(self):
        adapter = self.root / "sampling_reference.py"
        adapter.write_text(
            "import numpy as np\n"
            "class Adapter:\n"
            "    def __init__(self): self.attempt = 0\n"
            "    def infer(self, inputs):\n"
            "        scale = -1 if self.attempt in (0, 2) else 2\n"
            "        self.attempt += 1\n"
            "        value = inputs['pixel_values'].mean(dtype=np.float32) * np.float32(scale)\n"
            "        return {'output_0': np.full((1, 1, 768), value, dtype=np.float32)}\n"
            "def create(config): return Adapter()\n"
        )
        config = self.root / "sampling_reference.json"
        config.write_text(json.dumps({MODEL: {
            "backend": "custom", "python_path": str(self.root),
            "factory": "sampling_reference:create",
        }}))
        arguments = ("--resample-common-cosine-failures", "--reference-config", str(config),
                     "--atol", "FP32=0,BF16=0,FP16=0", "--rtol", "FP32=0,BF16=0,FP16=0")
        result, payload, report = self.run_native(*arguments, iterations=3)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(payload["status"], "FAIL")  # Cosine selection does not relax element tolerances.
        sampling = payload["sampling"]["models"][MODEL]
        self.assertEqual((sampling["attempted"], sampling["accepted"], sampling["rejected"]), (5, 3, 2))
        log = report / sampling["rejected_log"]
        rejected = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([r["iteration"] for r in rejected], [0, 2])
        hashes = []
        for case in payload["cases"]:
            records = case["accuracy"]["records"]
            self.assertEqual(case["image_count"], 3)
            self.assertEqual(case["sampling"]["attempted"], 5)
            self.assertEqual(case, json.loads((report / MODEL / case["precision"] / "result.json").read_text()))
            self.assertEqual([r["iteration"] for r in records], [1, 3, 4])
            self.assertEqual([r["accepted_index"] for r in records], [0, 1, 2])
            self.assertEqual(case["accuracy"]["passed_images"], 0)
            self.assertTrue(all(o["cosine_passed"] for r in records for o in r["outputs"]))
            hashes.append([r["inputs"][0]["md5"] for r in records])
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[1], hashes[2])
        for item in rejected:
            for precision in ("BF16", "FP16"):
                self.assertFalse(item["precisions"][precision]["outputs"][0]["cosine_passed"])
            for desc in item["inputs"]:
                raw = (log.parent / desc["file"]).read_bytes()
                self.assertEqual(hashlib.md5(raw).hexdigest(), desc["md5"])
                self.assertNotIn(desc["md5"], hashes[0])
        with (report / "outputs.tsv").open() as stream:
            self.assertTrue(all(int(row["iterations"]) == 3 for row in csv.DictReader(stream, delimiter="\t")))
        repeat, repeated, repeated_report = self.run_native(*arguments, iterations=3)
        self.assertEqual(repeat.returncode, result.returncode, repeat.stderr)
        self.assertEqual(repeated["cases"], payload["cases"])
        self.assertEqual((repeated_report / sampling["rejected_log"]).read_bytes(), log.read_bytes())

    def test_resampling_with_openvino_reference_and_no_rejections(self):
        self.create_ir(zero=True)
        result, payload, report = self.run_native("--resample-common-cosine-failures", precisions="FP16,BF16")
        self.assertEqual(result.returncode, 0, result.stderr)
        sampling = payload["sampling"]["models"][MODEL]
        self.assertEqual((sampling["attempted"], sampling["accepted"], sampling["rejected"]), (2, 2, 0))
        self.assertEqual((report / sampling["rejected_log"]).read_text(), "")
        for case in payload["cases"]:
            self.assertEqual(case["resident_precisions"], ["FP16", "BF16", "FP32"])
            self.assertEqual(len(case["accuracy"]["records"]), 2)

    def test_resampling_requires_random_mode_and_both_low_precisions(self):
        for extra, precisions in (
            ((), "FP32,BF16"), ((), "FP16"), ((), "FP32"),
            (("--mode", "performance"), "BF16,FP16"),
            (("--mode", "tolerance-search"), "FP32"),
        ):
            with self.subTest(extra=extra, precisions=precisions):
                result, payload, _ = self.run_native(
                    "--resample-common-cosine-failures", *extra, precisions=precisions)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(payload)
                self.assertIn("--resample-common-cosine-failures requires", result.stderr)

    def test_explicit_openvino_baseline_ignores_framework_assets(self):
        (self.references / f"{MODEL}.onnx").write_bytes(b"invalid framework asset")
        config = self.root / "openvino_reference.json"
        config.write_text(json.dumps({MODEL: {"backend": "openvino"}}))
        result, payload, _ = self.run_native("--reference-config", str(config), precisions="FP32,BF16")
        self.assertIn(result.returncode, (0, 1), result.stderr)
        for case in payload["cases"]:
            self.assertEqual(case["reference"]["backend"], "OpenVINO FP32")
            self.assertEqual(case["reference"]["provenance"]["selection"], "explicit")
        baseline = payload["cases"][0]
        self.assertEqual(baseline["accuracy"]["passed_images"], 2)
        for record in baseline["accuracy"]["records"]:
            for row in record["outputs"]:
                self.assertEqual(row["md5"], row["reference_md5"])
                self.assertEqual(row["max_abs"], 0)
                self.assertEqual(row["abs_error_variance"], 0)

    def test_tolerance_overrides_and_invalid_values(self):
        result, payload, report = self.run_native("--atol", "FP32=0.5", "--rtol", "FP32=0", precisions="FP32")
        self.assertEqual(result.returncode, 0, result.stderr)
        row = payload["cases"][0]["accuracy"]["records"][0]["outputs"][0]
        self.assertEqual((row["atol"], row["rtol"]), (0.5, 0))
        self.assertIn("out_of_tolerance_fraction", (report / "outputs.tsv").read_text())
        for flag in ("--atol", "--rtol"):
            for value in ("-1", "nan", "inf", "0.1junk"):
                with self.subTest(flag=flag, value=value):
                    result, payload, _ = self.run_native(flag, f"FP32={value}", precisions="FP32")
                    self.assertEqual(result.returncode, 2)
                    self.assertIsNone(payload)
                    self.assertIn("finite and nonnegative", result.stderr)

    def test_precision_overrides_are_independent_and_allow_partial_maps(self):
        result, payload, _ = self.run_native(
            "--atol", "FP32=0.5,BF16=0.25", "--atol", "FP16=0.125",
            "--rtol", "BF16=0",
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        expected = {"FP32": (0.5, 1e-4), "BF16": (0.25, 0), "FP16": (0.125, 1e-3)}
        for case in payload["cases"]:
            self.assertEqual((case["accuracy"]["atol"], case["accuracy"]["rtol"]), expected[case["precision"]])
            for record in case["accuracy"]["records"]:
                for row in record["outputs"]:
                    self.assertEqual((row["atol"], row["rtol"]), expected[case["precision"]])

    def test_invalid_tolerance_maps_are_rejected(self):
        for extra in (
            ("--atol", "0.5"), ("--rtol", "FP64=0.5"), ("--atol", "FP32="),
            ("--atol", "=0.5"), ("--rtol", "FP32=0.5,"),
            ("--atol", "FP32=0.5,FP32=0.25"),
            ("--rtol", "FP32=0.5", "--rtol", "FP32=0.25"),
            ("--mode", "performance", "--atol", "FP32=0.5"),
        ):
            with self.subTest(extra=extra):
                result, payload, _ = self.run_native(*extra)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(payload)
                self.assertIn("ERROR:", result.stderr)

    def test_native_self_tests_use_the_same_binary(self):
        result = subprocess.run([BINARY, "--mode", "self-test"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tests passed", result.stdout)

    def test_tolerance_search_verifies_fixed_ratio_with_same_inputs(self):
        result, payload, _ = self.run_native("--mode", "tolerance-search", precisions="FP32")
        self.assertEqual(result.returncode, 0, result.stderr)
        case = payload["cases"][0]
        search = case["tolerance_search"]
        self.assertEqual(search["scale"], 1)
        self.assertEqual((search["atol"], search["rtol"]), (1e-5, 1e-4))
        self.assertTrue(search["verification_input_hashes_equal"])
        self.assertEqual(case["accuracy"]["passed_images"], 2)

    def test_cross_process_comparison_uses_the_same_binary(self):
        directories = [self.root / "process_a", self.root / "process_b"]
        for index, directory in enumerate(directories):
            directory.mkdir()
            name = (f"stb2__input__FP32__{index + 1}-0123456789abcdef__1__"
                    "model__0__original__0123456789abcdef.bin")
            (directory / name).write_bytes(b"identical output")
        output = self.root / "comparison.json"
        command = [BINARY, "--mode", "compare-stability", "--output", str(output)]
        for directory in directories:
            command.extend(["--stability-dir", str(directory)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["binary_files"], 2)
        original = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), original)
        next(directories[1].iterdir()).write_bytes(b"different output")
        command[command.index(str(output))] = str(self.root / "comparison_fail.json")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads((self.root / "comparison_fail.json").read_text())["status"], "FAIL")

    def test_legacy_stability_files_are_rejected_not_silently_skipped(self):
        directory = self.root / "legacy_stability"
        directory.mkdir()
        name = ("stb1__input__FP32__1-0123456789abcdef__1__"
                "0123456789abcdef__model__0__original.bin")
        (directory / name).write_bytes(b"old output")
        result = subprocess.run(
            [BINARY, "--mode", "compare-stability", "--stability-dir", str(directory),
             "--output", str(self.root / "legacy_comparison.json")],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported legacy stability filename", result.stderr)

    def test_external_reference_scale_error_fails_despite_perfect_cosine(self):
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph([
            helper.make_node("ReduceMean", ["pixel_values"], ["mean"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Mul", ["mean", "scale"], ["scaled"]),
            helper.make_node("Reshape", ["scaled", "shape"], ["single"]),
            helper.make_node("Expand", ["single", "output_shape"], ["features"]),
        ], "scaled_reference", [
            helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 224, 224]),
        ], [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, 768]),
        ], [
            helper.make_tensor("scale", TensorProto.FLOAT, [], [2]),
            helper.make_tensor("shape", TensorProto.INT64, [3], [1, 1, 1]),
            helper.make_tensor("output_shape", TensorProto.INT64, [3], [1, 1, 768]),
        ])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, self.references / f"{MODEL}.onnx")
        result, payload, _ = self.run_native(precisions="FP32")
        self.assertEqual(result.returncode, 1, result.stderr)
        case = payload["cases"][0]
        self.assertEqual(case["reference"]["backend"], "ONNX Runtime FP32")
        for record in case["accuracy"]["records"]:
            row = record["outputs"][0]
            self.assertTrue(row["cosine_passed"])
            self.assertFalse(row["tolerance_passed"])
            self.assertEqual(row["out_of_tolerance_fraction"], 1)
            self.assertEqual(row["status"], "FAIL")
        result, payload, _ = self.run_native("--atol", "FP32=1", precisions="FP32")
        self.assertEqual(result.returncode, 0, result.stderr)
        result, payload, _ = self.run_native("--mode", "tolerance-search", precisions="FP32")
        self.assertEqual(result.returncode, 0, result.stderr)
        case = payload["cases"][0]
        search = case["tolerance_search"]
        self.assertGreater(search["scale"], 1)
        self.assertAlmostEqual(search["rtol"] / search["atol"], 10)
        self.assertLessEqual(search["scale"] - search["lower_scale"], search["scale"] * 1e-8)
        self.assertTrue(search["verification_input_hashes_equal"])
        self.assertEqual(case["accuracy"]["passed_images"], 2)
        self.assertEqual(case["calibration_accuracy"]["passed_images"], 0)
        self.assertTrue(any(not row["passed"] for row in search["trials"]))

        model.graph.initializer[0].float_data[0] = -2
        onnx.save(model, self.references / f"{MODEL}.onnx")
        result, payload, _ = self.run_native("--mode", "tolerance-search", precisions="FP32", iterations=5)
        self.assertEqual(result.returncode, 1, result.stderr)
        case = payload["cases"][0]
        self.assertEqual(len(case["accuracy"]["records"]), 1)
        self.assertEqual(case["tolerance_search"]["status"], "FAIL")
        self.assertIsNone(case["tolerance_search"]["scale"])
        self.assertNotIn("verification_report", case["tolerance_search"])

    def test_zero_norm_has_explicit_metrics(self):
        self.create_ir(zero=True)
        result, payload, _ = self.run_native(precisions="FP32")
        self.assertEqual(result.returncode, 0, result.stderr)
        row = payload["cases"][0]["accuracy"]["records"][0]["outputs"][0]
        self.assertTrue(row["zero_norm"])
        self.assertIsNone(row["cosine_similarity"])
        self.assertEqual(row["max_abs"], 0)
        self.assertEqual(row["abs_error_variance"], 0)
        self.assertEqual(row["normalized_rmse"], 0)
        self.assertTrue(row["exact_equal"])

    def test_integer_output_requires_exact_equality(self):
        self.create_ir(dtype=ov.Type.i64)
        result, payload, _ = self.run_native(precisions="FP32,FP16")
        self.assertEqual(result.returncode, 0, result.stderr)
        for case in payload["cases"]:
            row = case["accuracy"]["records"][0]["outputs"][0]
            self.assertEqual(row["dtype"], "i64")
            self.assertEqual(row["mismatched_elements"], 0)
            self.assertTrue(row["exact_equal"])

    def test_nonfinite_is_failure_not_zero_error_success(self):
        self.create_ir(nonfinite=True)
        result, payload, report = self.run_native(precisions="FP32")
        self.assertEqual(result.returncode, 1, result.stderr)
        row = payload["cases"][0]["accuracy"]["records"][0]["outputs"][0]
        self.assertFalse(row["finite"])
        self.assertEqual(row["status"], "FAIL")
        self.assertIsNone(row["max_abs"])
        self.assertIsNone(row["abs_error_variance"])
        with (report / "precision_metrics.tsv").open() as stream:
            metrics = next(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(metrics["cosine"], "null")
        self.assertEqual(metrics["max_abs"], "null")
        self.assertEqual(metrics["normalized_rmse"], "null")
        with (report / "outputs.tsv").open() as stream:
            self.assertEqual(next(csv.DictReader(stream, delimiter="\t"))["abs_error_variance"], "")

    def test_unavailable_python_produces_error_report(self):
        result, payload, _ = self.run_native("--reference-python", "/no/such/python")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["status"], "ERROR")
        self.assertIn("Cannot start reference worker", payload["error"])

    def test_existing_accuracy_mode_still_requires_manifest(self):
        result, _, _ = self.run_native("--mode", "accuracy")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--manifest", result.stderr)

    def test_removed_compatibility_options_are_rejected(self):
        for flag in ("--bf16-reshape-workaround", "--sdpa-bf16"):
            with self.subTest(flag=flag):
                result, payload, _ = self.run_native(flag, precisions="BF16")
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(payload)
                self.assertIn("ERROR:", result.stderr)
                self.assertIn(flag, result.stderr)

    def test_prediction_onnx_reference_and_nondegenerate_masks(self):
        import onnx
        from onnx import TensorProto, helper

        model_id = "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine"
        names = [
            "agents_feature", "agents_mask", "map_objects_feature", "map_objects_mask",
            "skeleton_feature", "skeleton_mask", "route_feature", "route_mask",
            "lidar_patch_feature", "lidar_patch_mask", "patch_cuboid_feature", "patch_cuboid_mask",
        ]
        output_names = ["trajs", "traj_probs", "yield_prob", "covariance", "ignore_ego_intention_prob"]
        nodes, infos = [], []
        accumulated = None
        for i, name in enumerate(names):
            boolean = name.endswith("_mask")
            infos.append(helper.make_tensor_value_info(name, TensorProto.BOOL if boolean else TensorProto.FLOAT,
                                                       ["batch", 2, 3]))
            value = name
            if boolean:
                value = f"cast_{i}"
                nodes.append(helper.make_node("Cast", [name], [value], to=TensorProto.FLOAT))
            if accumulated is None:
                accumulated = value
            else:
                new = f"sum_{i}"
                nodes.append(helper.make_node("Add", [accumulated, value], [new]))
                accumulated = new
        for name in output_names:
            nodes.append(helper.make_node("Identity", [accumulated], [name]))
        graph = helper.make_graph(nodes, "prediction", infos, [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 2, 3]) for name in output_names
        ])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, self.models / f"{model_id}.onnx")
        result, payload, _ = self.run_native("--model", model_id)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        for case in payload["cases"]:
            self.assertEqual(case["reference"]["backend"], "ONNX Runtime FP32")
            for record in case["accuracy"]["records"]:
                self.assertEqual(len(record["inputs"]), 12)
                self.assertEqual([x["name"] for x in record["outputs"]], output_names)
                self.assertTrue(all(x["finite"] for x in record["outputs"]))
                self.assertTrue(all(x["shape"] == [1, 2, 3] for x in record["outputs"]))
            if case["precision"] == "FP32":
                self.assertEqual(case["accuracy"]["passed_images"], 2)


if __name__ == "__main__":
    unittest.main()
