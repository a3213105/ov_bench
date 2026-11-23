"""Dependency-light unit tests; run with unittest (no ONNX runtime required)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "prepare_pred_models.py"
SPEC = importlib.util.spec_from_file_location("prepare_pred_models", SCRIPT)
prep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prep
SPEC.loader.exec_module(prep)


class PredInputsTests(unittest.TestCase):
    def test_only_dynamic_batch_resolved(self):
        self.assertEqual(prep.resolve_input_shape(["batch", 33, 10]), (1, 33, 10))
        self.assertEqual(prep.resolve_input_shape([None, 4]), (1, 4))
        self.assertEqual(prep.resolve_input_shape([2, 4]), (2, 4))
        self.assertEqual(prep.resolve_input_shape([]), ())
        for dimensions in ([1, "sequence"], [None, None], [0, 3], [1, -1]):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                prep.resolve_input_shape(dimensions)

    def test_reproducibility_and_non_degenerate_groups(self):
        specs = [
            prep.InputSpec("features", "f32", (1, 4, 12, 3)),
            prep.InputSpec("other_features", "f32", (1, 4, 12, 3)),
            prep.InputSpec("mask", "boolean", (1, 4, 12)),
            prep.InputSpec("single_mask", "boolean", (1, 4, 1)),
        ]
        first = prep.generate_inputs(specs, prep.DEFAULT_SEED, "model", 0)
        second = prep.generate_inputs(list(reversed(specs)), prep.DEFAULT_SEED, "model", 0)
        for name, value in first.items():
            np.testing.assert_array_equal(value, second[name])
            self.assertTrue(value.flags.c_contiguous)
        self.assertEqual(first["features"].dtype, np.float32)
        self.assertEqual(first["mask"].dtype, np.bool_)
        self.assertTrue(np.isfinite(first["features"]).all())
        self.assertGreater(float(first["features"].std()), 0.05)
        self.assertLess(float(first["features"].std()), 0.15)
        self.assertTrue(first["mask"].any(axis=-1).all())
        self.assertTrue((~first["mask"]).any(axis=-1).all())
        self.assertTrue(first["single_mask"].all())
        self.assertFalse(np.array_equal(first["features"], first["other_features"]))
        for seed, model, sample in (
            (prep.DEFAULT_SEED + 1, "model", 0),
            (prep.DEFAULT_SEED, "other_model", 0),
            (prep.DEFAULT_SEED, "model", 1),
        ):
            changed = prep.generate_inputs(specs, seed, model, sample)
            self.assertFalse(np.array_equal(first["features"], changed["features"]))

    def test_ten_distinct_reproducible_cases(self):
        specs = [prep.InputSpec("feature", "f32", (1, 10))]
        runs = [
            [prep.generate_inputs(specs, 20261030, "model", i)["feature"].tobytes()
             for i in range(10)]
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(set(runs[0])), 10)

    def test_invalid_generator_type_and_seed(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            prep.generate_inputs([prep.InputSpec("ids", "i32", (1, 2))], 1, "m", 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            prep.generate_inputs([], -1, "m", 0)

    def test_output_validation_preserves_types(self):
        for dtype in (np.float32, np.int32):
            prep.validate_output(np.array([1, 2], dtype=dtype), "m", "s", "out")
        for dtype in (np.float16, np.float64, np.int64, np.bool_):
            with self.assertRaisesRegex(ValueError, "unsupported dtype"):
                prep.validate_output(np.array([1], dtype=dtype), "m", "s", "out")
        with self.assertRaisesRegex(ValueError, "2/3 nonfinite"):
            prep.validate_output(np.array([np.nan, np.inf, 1], np.float32), "m", "s", "out")

    def test_raw_descriptors_and_scalar_shape(self):
        root = Path.cwd() / f".pred-input-test-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        for i, (dtype, tag) in enumerate(
            ((np.float32, "f32"), (np.bool_, "boolean"), (np.int32, "i32"), (np.int64, "i64"))
        ):
            value = np.arange(12).reshape(3, 4).astype(dtype)[:, ::2]
            descriptor = prep.write_tensor(root / f"{i}.bin", value, "input", root)
            raw = (root / descriptor["file"]).read_bytes()
            self.assertEqual(descriptor["dtype"], tag)
            self.assertEqual(descriptor["shape"], [3, 2])
            self.assertEqual(descriptor["bytes"], value.nbytes)
            self.assertEqual(descriptor["md5"], hashlib.md5(raw).hexdigest())
            np.testing.assert_array_equal(
                np.frombuffer(raw, dtype=value.dtype).reshape(value.shape), value
            )
        descriptor = prep.write_tensor(root / "scalar.bin", np.array(2, np.int32), "out", root)
        self.assertEqual(descriptor["shape"], [])
        self.assertEqual(descriptor["bytes"], 4)
        self.assertEqual(prep.sha256_file(root / "scalar.bin"),
                         hashlib.sha256((root / "scalar.bin").read_bytes()).hexdigest())

    def test_manifest_and_fresh_output_contract(self):
        root = Path.cwd() / f".pred-manifest-test-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        model_names = [
            "26Q2_round13_run2_e40_no_map_attn",
            "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine",
            "vehicle_q2_release_v0_w_leading_25",
        ]
        for name in model_names:
            (root / f"{name}.onnx").write_bytes(b"fixture")
        output = root / "generated"
        with patch.object(
            prep, "prepare_model", side_effect=lambda path, *_: {"model": path.stem}
        ) as mocked:
            manifest_path = prep.prepare(root, output)
        self.assertEqual(
            [call.args[0].stem for call in mocked.call_args_list], sorted(model_names)
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["reference_backend"], "ONNX Runtime FP32 CPU")
        self.assertEqual(manifest["seed"], 43)
        self.assertIn("business semantics unknown", manifest["synthetic_data"]["mask_semantics"])
        self.assertEqual(
            manifest["models"], [{"model": name} for name in sorted(model_names)]
        )
        with self.assertRaises(FileExistsError):
            prep.prepare(root, output)
        with self.assertRaisesRegex(ValueError, "positive"):
            prep.prepare(root, root / "invalid", samples=0)


if __name__ == "__main__":
    unittest.main()
