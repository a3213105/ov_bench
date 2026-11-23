"""Dependency-light tests for the persistent random-accuracy reference worker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


WORKER_PATH = Path(__file__).resolve().parents[1] / "tools" / "random_reference_worker.py"
SPEC = importlib.util.spec_from_file_location("random_reference_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def base_config(root: Path, **overrides):
    config = {
        "model": "sample_model",
        "family": "prediction",
        "artifact": "sample_model",
        "model_root": str(root / "ov"),
        "reference_root": str(root / "reference"),
        "threads": 2,
        "input_names": ["input_a", "input_b"],
        "output_names": ["output_a", "output_b"],
        "reference_config": {},
    }
    config.update(overrides)
    return config


class FakeOnnxSession:
    def __init__(self):
        self.call = None

    def run(self, names, feed):
        self.call = (names, feed)
        return [
            np.asarray([10], dtype=np.float32) if name == "onnx_out_b"
            else np.asarray([20], dtype=np.float32)
            for name in names
        ]


class RandomReferenceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "ov").mkdir()
        (self.root / "reference").mkdir()
        self.config_path = self.root / "worker_config.json"
        self.config_path.write_text("{}", encoding="utf-8")
        self.directory = self.root / "scratch"

    def test_prediction_discovery_priority(self):
        config = base_config(self.root)
        root = Path(config["model_root"])
        ref = Path(config["reference_root"])
        candidates = [
            root / "sample_model" / "sample_model.onnx",
            root / "sample_model.onnx",
            ref / "pred_model" / "sample_model.onnx",
        ]
        for path in candidates:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.assertEqual(worker.discover_onnx(config), candidates[0].resolve())
        candidates[0].unlink()
        self.assertEqual(worker.discover_onnx(config), candidates[1].resolve())
        candidates[1].unlink()
        self.assertEqual(worker.discover_onnx(config), candidates[2].resolve())

    def test_broken_discovered_onnx_is_fatal_without_pytorch_fallback(self):
        config = base_config(self.root)
        onnx_path = Path(config["model_root"]) / "sample_model" / "sample_model.onnx"
        onnx_path.parent.mkdir(parents=True)
        onnx_path.write_bytes(b"not-a-model")
        with patch.object(worker, "_load_onnx", side_effect=RuntimeError("broken ONNX")):
            with patch.object(worker, "_load_pytorch") as pytorch:
                with self.assertRaisesRegex(RuntimeError, "broken ONNX"):
                    worker.ReferenceWorker(config, self.directory, self.config_path)
                pytorch.assert_not_called()

    def test_relocated_worker_uses_explicit_yolo_project_root(self):
        project = self.root / "project" / "yolo-world"
        project.mkdir(parents=True)
        (project / "yolo_world_benchmark.py").write_text(
            "CONFIG_FILES = {640: 'configs/test.py'}\n", encoding="utf-8"
        )
        reference = self.root / "reference"
        (reference / "yolo-world").mkdir()
        checkpoint = reference / "yolo-world" / worker._YOLO_FILES["v2:640"]
        checkpoint.write_bytes(b"fixture")
        config = base_config(self.root, model="v2:640", family="yolo-world")
        with patch.dict(worker.os.environ, {"OVBENCH_REFERENCE_PROJECT_ROOT": str(project.parent)}):
            assets = worker._native_pytorch_assets(config)
        self.assertEqual(assets["project"], project)
        self.assertEqual(assets["checkpoint"], checkpoint)
        self.assertEqual(assets["config"], project / "third_party/YOLO-World/configs/test.py")

    def _raw_input(self, name, value, dtype_tag):
        path = self.root / f"{name}.bin"
        array = np.asarray(value, dtype=worker._DTYPES[dtype_tag])
        raw = array.tobytes(order="C")
        path.write_bytes(raw)
        return {
            "name": name,
            "file": str(path),
            "dtype": dtype_tag,
            "shape": list(array.shape),
            "bytes": len(raw),
        }

    def test_onnx_mapping_order_and_protocol_descriptors_are_deterministic(self):
        config = base_config(
            self.root,
            reference_config={
                "backend": "onnx",
                "path": str(self.root / "chosen.onnx"),
                "input_names": {"input_a": "onnx_input_b", "input_b": "onnx_input_a"},
                "output_names": {"output_a": "onnx_out_b", "output_b": "onnx_out_a"},
            },
        )
        path = Path(config["reference_config"]["path"])
        path.write_bytes(b"fixture")
        session = FakeOnnxSession()
        ports = {
            "inputs": ["onnx_input_a", "onnx_input_b"],
            "outputs": ["onnx_out_a", "onnx_out_b"],
            "providers": ["CPUExecutionProvider"],
        }
        with patch.object(worker, "_load_onnx", return_value=(session, ports)):
            instance = worker.ReferenceWorker(config, self.directory, self.config_path)
        self.assertEqual(instance.ready["backend"], "ONNX Runtime FP32")
        self.assertEqual(
            instance.ready["mappings"]["inputs"],
            {"input_a": "onnx_input_b", "input_b": "onnx_input_a"},
        )
        descriptors = [
            self._raw_input("input_a", [1, 2], "f32"),
            self._raw_input("input_b", [False, True], "boolean"),
        ]
        response = instance.infer({"iteration": 4, "inputs": descriptors})
        self.assertEqual(session.call[0], ["onnx_out_b", "onnx_out_a"])
        np.testing.assert_array_equal(session.call[1]["onnx_input_b"], [1, 2])
        np.testing.assert_array_equal(session.call[1]["onnx_input_a"], [False, True])
        self.assertEqual([item["name"] for item in response["outputs"]], ["output_a", "output_b"])
        self.assertEqual([item["file"] for item in response["outputs"]], [
            str(self.directory.resolve() / "reference_4_0.bin"),
            str(self.directory.resolve() / "reference_4_1.bin"),
        ])
        self.assertEqual(
            [item["md5"] for item in response["outputs"]],
            [hashlib.md5(Path(item["file"]).read_bytes()).hexdigest() for item in response["outputs"]],
        )
        with self.assertRaisesRegex(worker.WorkerError, "Refusing to overwrite"):
            instance.infer({"iteration": 4, "inputs": descriptors})

    def test_custom_adapter_named_mapping_scalar_boolean_and_i64_raw_io(self):
        module_path = self.root / "custom_adapter.py"
        module_path.write_text(
            "import numpy as np\n"
            "class Adapter:\n"
            "    def infer(self, inputs):\n"
            "        assert list(inputs) == ['framework_input']\n"
            "        assert inputs['framework_input'].shape == ()\n"
            "        return {'framework_flag': np.asarray(True, dtype=np.bool_), "
            "'framework_integer': np.asarray([2**40], dtype=np.int64)}\n"
            "def create(config):\n"
            "    return Adapter(), {'factory_loaded': True}\n",
            encoding="utf-8",
        )
        config = base_config(
            self.root,
            input_names=["canonical_input"],
            output_names=["flag", "integer"],
            reference_config={
                "backend": "custom",
                "python_path": str(self.root),
                "factory": "custom_adapter:create",
                "available_input_names": ["framework_input"],
                "available_output_names": ["framework_flag", "framework_integer"],
                "input_names": {"canonical_input": "framework_input"},
                "output_names": {
                    "flag": "framework_flag",
                    "integer": "framework_integer",
                },
            },
        )
        instance = worker.ReferenceWorker(config, self.directory, self.config_path)
        self.assertEqual(instance.ready["backend"], "Custom FP32")
        self.assertTrue(instance.ready["provenance"]["factory_loaded"])
        descriptor = self._raw_input("canonical_input", np.asarray(True), "boolean")
        response = instance.infer({"iteration": 0, "inputs": [descriptor]})
        boolean, integer = response["outputs"]
        self.assertEqual((boolean["dtype"], boolean["shape"], boolean["bytes"]), ("boolean", [], 1))
        self.assertEqual((integer["dtype"], integer["shape"], integer["bytes"]), ("i64", [1], 8))
        self.assertEqual(Path(boolean["file"]).read_bytes(), b"\x01")
        self.assertEqual(
            np.frombuffer(Path(integer["file"]).read_bytes(), dtype="<i8").tolist(),
            [2**40],
        )

    def test_input_byte_and_shape_validation(self):
        config = base_config(self.root, reference_config={"backend": "openvino"})
        instance = worker.ReferenceWorker(config, self.directory, self.config_path)
        malformed = self._raw_input("input_a", [1], "f32")
        malformed["bytes"] = 0
        with self.assertRaisesRegex(worker.WorkerError, "byte count mismatch"):
            instance._read_inputs([malformed, self._raw_input("input_b", [True], "boolean")])
        malformed = self._raw_input("input_a", [1], "f32")
        malformed["shape"] = [0]
        with self.assertRaisesRegex(worker.WorkerError, "shape"):
            instance._read_inputs([malformed, self._raw_input("input_b", [True], "boolean")])


if __name__ == "__main__":
    unittest.main()
