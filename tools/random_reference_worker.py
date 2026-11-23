#!/usr/bin/env python3
"""Persistent FP32 reference worker for C++ random-accuracy runs.

The worker accepts one JSON object per stdin line and writes exactly one JSON
response per command to stdout. Framework output is redirected to stderr.

``reference_config`` is a per-model object. ``backend`` may be ``onnx``,
``pytorch``, ``openvino``, or ``custom`` and disables automatic backend
selection. ``path`` selects an ONNX or TorchScript file (relative paths are
resolved against the outer worker config's directory). ``factory`` names a
custom adapter as ``module:function``; the function receives the inner config
and may return an adapter, or ``(adapter, metadata)``. An adapter implements
``infer(dict[str, numpy.ndarray]) -> dict[str, array-like]``. Optional
``python_path`` (a path or list of paths) is prepended to ``sys.path``.

For framework name mapping, ``input_names`` and ``output_names`` may each be
a positional list in C++ port order, or a mapping from C++ canonical name to
framework tensor name. Without overrides, exact names are preferred, with
positional mapping as a fallback only when the port counts match. Inputs and
outputs are always returned in the C++-declared order.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


_PROTOCOL_OUT = sys.stdout
_DTYPES: dict[str, np.dtype[Any]] = {
    "f32": np.dtype("<f4"),
    "boolean": np.dtype("?"),
    "i8": np.dtype("i1"),
    "i16": np.dtype("<i2"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "u32": np.dtype("<u4"),
    "u64": np.dtype("<u8"),
}
_DTYPE_TAGS = {dtype: tag for tag, dtype in _DTYPES.items()}
_REAL_DTYPES = {np.dtype("float32")}
_INTEGER_DTYPES = {
    np.dtype("int8"), np.dtype("int16"), np.dtype("int32"), np.dtype("int64"),
    np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64"),
}
_BOOLEAN_DTYPE = np.dtype("bool")
_FAMILY_BACKEND_NAMES = {"prediction", "dinov3", "yolo-world"}
_YOLO_FILES = {
    "v2.1:640": "yolo_world_v2_1_m_640_stage1-7e1e5299.pth",
    "v2.1:1280": "yolo_world_v2_1_m_1280_stage2-9987dcb1.pth",
    "v2:640": "yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth",
    "v2:1280": "yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth",
}


class WorkerError(RuntimeError):
    """An invalid configuration, protocol message, or model operation."""


def _absolute(path: str | os.PathLike[str], base: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and base is not None:
        value = base / value
    return value.resolve()


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError(f"config.{key} must be a non-empty string")
    return value


def _port_names(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise WorkerError(f"config.{key} must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise WorkerError(f"config.{key} contains duplicate names")
    return value


def _path_candidates(config: dict[str, Any], family: str, artifact: str) -> list[Path]:
    model_root = Path(config["model_root"])
    reference_root = Path(config["reference_root"])
    candidates: list[Path] = []
    if family == "prediction":
        candidates.extend((
            model_root / artifact / f"{artifact}.onnx",
            model_root / f"{artifact}.onnx",
            reference_root / "pred_model" / f"{artifact}.onnx",
        ))
    else:
        roots = (model_root, reference_root)
        for root in roots:
            candidates.extend((root / artifact / f"{artifact}.onnx", root / f"{artifact}.onnx"))
        if family == "dinov3":
            candidates.append(reference_root / "dinov3" / artifact / f"{artifact}.onnx")
        elif family == "yolo-world":
            candidates.append(reference_root / "yolo-world" / artifact / f"{artifact}.onnx")
    # Keep discovery deterministic while avoiding duplicate probes.
    return list(dict.fromkeys(path.resolve() for path in candidates))


def discover_onnx(config: dict[str, Any]) -> Path | None:
    """Return the first associated ONNX file according to the family priority."""
    for candidate in _path_candidates(config, config["family"], config["artifact"]):
        if candidate.is_file():
            return candidate
    return None


def _native_pytorch_assets(config: dict[str, Any]) -> dict[str, Path] | None:
    family, model = config["family"], config["model"]
    root = Path(config["reference_root"])
    if family == "dinov3":
        model_dir = root / "dinov3" / model
        if (model_dir / "config.json").is_file() and (model_dir / "model.safetensors").is_file():
            return {"model_dir": model_dir}
        return None
    if family == "yolo-world":
        checkpoint_name = _YOLO_FILES.get(model)
        if checkpoint_name:
            checkpoint = root / "yolo-world" / checkpoint_name
            project_root = Path(os.environ.get(
                "OVBENCH_REFERENCE_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])
            )).resolve()
            project = project_root / "yolo-world"
            upstream = project / "third_party" / "YOLO-World"
            if checkpoint.is_file():
                benchmark = _import_module_from_file(
                    "_random_reference_yolo_discovery", project / "yolo_world_benchmark.py"
                )
                config_path = upstream / benchmark.CONFIG_FILES[int(model.split(":")[1])]
                return {
                    "checkpoint": checkpoint,
                    "config": config_path,
                    "upstream": upstream,
                    "project": project,
                }
    return None


def _mapping(
    canonical_names: list[str],
    available_names: list[str],
    override: Any,
    label: str,
    allow_positional: bool,
) -> list[str]:
    if override is not None:
        if isinstance(override, list):
            if len(override) != len(canonical_names) or any(
                not isinstance(item, str) or not item for item in override
            ):
                raise WorkerError(f"reference_config.{label} list must match declared port count")
            result = list(override)
        elif isinstance(override, dict):
            unknown = set(override) - set(canonical_names)
            missing = set(canonical_names) - set(override)
            if unknown or missing:
                raise WorkerError(
                    f"reference_config.{label} mapping keys mismatch; missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}"
                )
            result = [override[name] for name in canonical_names]
            if any(not isinstance(item, str) or not item for item in result):
                raise WorkerError(f"reference_config.{label} values must be non-empty strings")
        else:
            raise WorkerError(f"reference_config.{label} must be a list or object")
    elif all(name in available_names for name in canonical_names):
        result = list(canonical_names)
    elif allow_positional and len(canonical_names) == len(available_names):
        result = list(available_names)
    else:
        raise WorkerError(
            f"Cannot map declared {label} {canonical_names!r} to framework names {available_names!r}"
        )
    unknown_names = set(result) - set(available_names)
    if unknown_names:
        raise WorkerError(f"Mapped {label} names not found in framework: {sorted(unknown_names)}")
    if len(set(result)) != len(result):
        raise WorkerError(f"Mapped {label} names are not unique")
    return result


def _configure_torch(threads: int) -> Any:
    try:
        import torch
    except Exception as error:
        raise WorkerError(f"Cannot import PyTorch: {type(error).__name__}: {error}") from error
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "allow_tf32"):
        torch.backends.mkldnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def _load_onnx(path: Path, threads: int) -> tuple[Any, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as error:
        raise WorkerError(f"Cannot import ONNX Runtime: {type(error).__name__}: {error}") from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = max(1, min(threads, 4))
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise WorkerError(f"ONNX Runtime did not select CPU-only execution: {providers!r}")
    return session, {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
        "providers": providers,
    }


def _import_module_from_file(module_name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkerError(f"Cannot load repository module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_hf_dino(config: dict[str, Any], assets: dict[str, Path], torch: Any) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModel

    # No device_map: CPU-only loading does not require the optional accelerate package.
    model = AutoModel.from_pretrained(
        str(assets["model_dir"]), dtype=torch.float32, local_files_only=True
    ).to(device="cpu", dtype=torch.float32).eval()

    class DinoAdapter:
        def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
            if len(inputs) != 1:
                raise WorkerError("DINOv3 PyTorch reference expects one named input")
            tensor = torch.from_numpy(next(iter(inputs.values()))).to(dtype=torch.float32, device="cpu")
            return model(pixel_values=tensor).last_hidden_state

    return DinoAdapter(), {
        "model_dir": str(assets["model_dir"]), "loader": "transformers.AutoModel.from_pretrained",
        "local_files_only": True, "torch_version": torch.__version__,
    }


def _load_yolo(config: dict[str, Any], assets: dict[str, Path], torch: Any) -> tuple[Any, dict[str, Any]]:
    project = assets["project"]
    benchmark = _import_module_from_file("_random_reference_yolo_benchmark", project / "yolo_world_benchmark.py")
    version, size_text = config["model"].split(":", 1)
    size = int(size_text)
    variant = benchmark.MODEL_VARIANTS[version]
    spec = benchmark.ModelSpec(
        version=version,
        size=size,
        artifact_prefix=variant["artifact_prefix"],
        checkpoint=assets["checkpoint"],
        config=assets["config"],
        upstream_dir=assets["upstream"],
        onnx_root=project / "artifacts",
        openvino_root=Path(config["model_root"]),
    )
    # The existing loader catches and prints build errors; convert its
    # success-shaped None result into an explicit worker failure.
    with contextlib.redirect_stdout(sys.stderr):
        model, _ = benchmark.build_pytorch_model(spec, argparse.Namespace())
    if model is None:
        raise WorkerError("Repository YOLO-World PyTorch model builder failed")
    model.eval()

    class YoloAdapter:
        def infer(self, inputs: dict[str, np.ndarray]) -> Any:
            if len(inputs) != 1:
                raise WorkerError("YOLO-World PyTorch reference expects one named input")
            array = next(iter(inputs.values()))
            tensor = torch.from_numpy(array).to(device="cpu")
            return model(tensor)

    return YoloAdapter(), {
        "checkpoint": str(assets["checkpoint"]),
        "config": str(assets["config"]),
        "upstream": str(assets["upstream"]),
        "loader": str(project / "yolo_world_benchmark.py"),
    }


def _load_custom(config: dict[str, Any], reference: dict[str, Any], base: Path) -> tuple[Any, dict[str, Any]]:
    python_path = reference.get("python_path", [])
    if isinstance(python_path, str):
        python_path = [python_path]
    if not isinstance(python_path, list) or any(not isinstance(item, str) for item in python_path):
        raise WorkerError("reference_config.python_path must be a path or list of paths")
    for item in reversed(python_path):
        sys.path.insert(0, str(_absolute(item, base)))
    factory_path = reference.get("factory")
    if not isinstance(factory_path, str) or ":" not in factory_path:
        raise WorkerError("reference_config.factory must be 'module:function'")
    module_name, function_name = factory_path.split(":", 1)
    if not module_name or not function_name:
        raise WorkerError("reference_config.factory must be 'module:function'")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise WorkerError(f"Custom factory is not callable: {factory_path}")
    result = factory(config)
    metadata: dict[str, Any] = {}
    if isinstance(result, tuple) and len(result) == 2:
        adapter, metadata_value = result
        if metadata_value is not None:
            if not isinstance(metadata_value, dict):
                raise WorkerError("Custom factory metadata must be an object")
            metadata = metadata_value
    else:
        adapter = result
    if not callable(getattr(adapter, "infer", None)):
        raise WorkerError("Custom adapter must implement infer(dict[str, ndarray])")
    return adapter, metadata


def _load_pytorch(
    config: dict[str, Any], reference: dict[str, Any], base: Path, threads: int
) -> tuple[Any, dict[str, Any], Any | None]:
    torch = _configure_torch(threads)
    if reference.get("path"):
        path = _absolute(reference["path"], base)
        if not path.is_file():
            raise WorkerError(f"Configured PyTorch reference does not exist: {path}")
        model = torch.jit.load(str(path), map_location="cpu").float().eval()

        class TorchScriptAdapter:
            def infer(self, inputs: dict[str, np.ndarray]) -> Any:
                tensors = [torch.from_numpy(value) for value in inputs.values()]
                with torch.inference_mode():
                    return model(*tensors)

        return TorchScriptAdapter(), {"path": str(path), "format": "TorchScript"}, torch
    assets = _native_pytorch_assets(config)
    if assets is None:
        raise WorkerError(f"No supported native PyTorch assets found for {config['model']}")
    if config["family"] == "dinov3":
        adapter, metadata = _load_hf_dino(config, assets, torch)
    elif config["family"] == "yolo-world":
        adapter, metadata = _load_yolo(config, assets, torch)
    else:
        raise WorkerError(f"No native PyTorch loader for family {config['family']!r}")
    return adapter, metadata, torch


class ReferenceWorker:
    def __init__(self, config: dict[str, Any], directory: Path, config_path: Path):
        self.config = config
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.reference = config.get("reference_config", {})
        if not isinstance(self.reference, dict):
            raise WorkerError("config.reference_config must be an object")
        self.family = _required_string(config, "family")
        if self.family not in _FAMILY_BACKEND_NAMES:
            raise WorkerError(f"Unsupported model family: {self.family!r}")
        self.model = _required_string(config, "model")
        self.artifact = _required_string(config, "artifact")
        self.model_root = _absolute(_required_string(config, "model_root"))
        self.reference_root = _absolute(
            config.get("reference_root", "/home/sgui/2026/models")
        )
        self.threads = config.get("threads", 1)
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise WorkerError("config.threads must be a positive integer")
        self.input_names = _port_names(config, "input_names")
        self.output_names = _port_names(config, "output_names")
        self.config = dict(config)
        self.config.update(model_root=str(self.model_root), reference_root=str(self.reference_root))
        self.base = config_path.resolve().parent
        with contextlib.redirect_stdout(sys.stderr):
            (
                self.backend, self.model_object, self.input_mapping,
                self.output_mapping, metadata,
            ) = self._initialize()
        self.ready = {
            "status": "ready",
            "backend": self.backend,
            "model": self.model,
            "family": self.family,
            "artifact": self.artifact,
            "model_root": str(self.model_root),
            "reference_root": str(self.reference_root),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "mappings": {
                "inputs": dict(zip(self.input_names, self.input_mapping)),
                "outputs": dict(zip(self.output_names, self.output_mapping)),
            },
            "provenance": metadata,
        }

    def _initialize(self) -> tuple[str, Any, list[str], list[str], dict[str, Any]]:
        reference = self.reference
        selected = reference.get("backend")
        if selected is not None and selected not in ("onnx", "pytorch", "openvino", "custom"):
            raise WorkerError("reference_config.backend must be onnx, pytorch, openvino, or custom")

        if selected == "openvino":
            return "OpenVINO FP32", None, [], [], {"selection": "explicit"}
        if selected == "onnx":
            path = (
                _absolute(reference["path"], self.base)
                if reference.get("path")
                else discover_onnx(self.config)
            )
            if path is None or not path.is_file():
                raise WorkerError(f"Configured ONNX reference does not exist: {path}")
            return self._initialize_onnx(path, "explicit")
        if selected == "pytorch":
            adapter, metadata, torch = _load_pytorch(
                self.config, reference, self.base, self.threads
            )
            return self._initialize_adapter(
                "PyTorch FP32", adapter, metadata, torch, "explicit"
            )
        if selected == "custom":
            adapter, metadata = _load_custom(self.config, reference, self.base)
            return self._initialize_adapter(
                "Custom FP32", adapter, metadata, None, "custom"
            )

        onnx_path = discover_onnx(self.config)
        if onnx_path is not None:
            return self._initialize_onnx(onnx_path, "discovered")
        assets = _native_pytorch_assets(self.config)
        if assets is not None:
            # A present native model is considered available. Loader failures
            # are fatal and must not silently change the reference backend.
            adapter, metadata, torch = _load_pytorch(
                self.config, reference, self.base, self.threads
            )
            return self._initialize_adapter(
                "PyTorch FP32", adapter, metadata, torch, "discovered"
            )
        return "OpenVINO FP32", None, [], [], {"selection": "no external reference assets"}

    def _initialize_onnx(self, path: Path, selection: str):
        session, ports = _load_onnx(path, self.threads)
        input_mapping = _mapping(
            self.input_names, ports["inputs"], self.reference.get("input_names"),
            "input_names", allow_positional=self.family != "prediction",
        )
        output_mapping = _mapping(
            self.output_names, ports["outputs"], self.reference.get("output_names"),
            "output_names", allow_positional=True,
        )
        return (
            "ONNX Runtime FP32", session, input_mapping, output_mapping,
            {"selection": selection, "path": str(path), "providers": ports["providers"],
             "framework_inputs": ports["inputs"], "framework_outputs": ports["outputs"]},
        )

    def _initialize_adapter(
        self, backend: str, adapter: Any, metadata: dict[str, Any],
        torch: Any | None, selection: str,
    ):
        if torch is None:
            # Custom adapters own their execution stack; torch is intentionally
            # optional. For custom interface mappings, choose explicit names
            # when supplied, otherwise canonical positional names.
            in_available = self.reference.get("available_input_names", self.input_names)
            out_available = self.reference.get("available_output_names", self.output_names)
        else:
            in_available = self.reference.get("available_input_names", self.input_names)
            out_available = self.reference.get("available_output_names", self.output_names)
        if not isinstance(in_available, list) or not isinstance(out_available, list):
            raise WorkerError("available_input_names/output_names must be lists")
        in_mapping = _mapping(
            self.input_names, in_available, self.reference.get("input_names"),
            "input_names", allow_positional=True,
        )
        out_mapping = _mapping(
            self.output_names, out_available, self.reference.get("output_names"),
            "output_names", allow_positional=True,
        )
        info = {"selection": selection}
        info.update(metadata)
        if torch is not None:
            info["torch_version"] = torch.__version__
        return backend, adapter, in_mapping, out_mapping, info

    def _read_inputs(self, descriptors: Any) -> dict[str, np.ndarray]:
        if not isinstance(descriptors, list) or len(descriptors) != len(self.input_names):
            raise WorkerError("infer.inputs count does not match declared input names")
        values: dict[str, np.ndarray] = {}
        for index, (descriptor, expected_name) in enumerate(zip(descriptors, self.input_names)):
            if not isinstance(descriptor, dict):
                raise WorkerError(f"inputs[{index}] must be an object")
            if descriptor.get("name") != expected_name:
                raise WorkerError(f"inputs[{index}] name mismatch: expected {expected_name!r}")
            tag = descriptor.get("dtype")
            if tag not in _DTYPES:
                raise WorkerError(f"inputs[{index}] has unsupported dtype {tag!r}")
            shape = descriptor.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
            ):
                if shape != []:
                    raise WorkerError(f"inputs[{index}].shape must contain positive integers")
            byte_count = descriptor.get("bytes")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
                raise WorkerError(f"inputs[{index}].bytes must be a nonnegative integer")
            elements = math.prod(shape)
            expected_bytes = elements * _DTYPES[tag].itemsize
            if byte_count != expected_bytes:
                raise WorkerError(
                    f"inputs[{index}] byte count mismatch: declared {byte_count}, expected {expected_bytes}"
                )
            file_value = descriptor.get("file")
            if not isinstance(file_value, str) or not file_value:
                raise WorkerError(f"inputs[{index}].file must be a non-empty path")
            path = Path(file_value)
            if not path.is_absolute():
                raise WorkerError(f"inputs[{index}].file must be absolute")
            raw = path.read_bytes()
            if len(raw) != byte_count:
                raise WorkerError(
                    f"inputs[{index}] file size mismatch: declared {byte_count}, got {len(raw)}"
                )
            if "md5" in descriptor and hashlib.md5(raw).hexdigest() != descriptor["md5"]:
                raise WorkerError(f"inputs[{index}] MD5 mismatch")
            array = np.frombuffer(raw, dtype=_DTYPES[tag]).reshape(tuple(shape)).copy()
            if expected_name in values:
                raise WorkerError(f"Duplicate input name: {expected_name}")
            values[expected_name] = array
        return values

    @staticmethod
    def _numpy_output(value: Any, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").numpy()
        array = np.asarray(value)
        dtype = array.dtype
        if dtype not in _REAL_DTYPES and dtype not in _INTEGER_DTYPES and dtype != _BOOLEAN_DTYPE:
            raise WorkerError(
                f"output {name!r} has unsupported dtype {dtype}; expected float32, integer, or boolean"
            )
        if dtype == _BOOLEAN_DTYPE:
            tag = "boolean"
        elif dtype == np.dtype("float32"):
            tag = "f32"
        else:
            normalized = dtype.newbyteorder("=")
            if normalized not in _DTYPE_TAGS:
                raise WorkerError(f"output {name!r} has unsupported integer dtype {dtype}")
            tag = _DTYPE_TAGS[normalized]
        little_dtype = _DTYPES[tag]
        return array.astype(little_dtype, copy=False).copy(order="C")

    def _run_inference(self, values: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.backend == "ONNX Runtime FP32":
            feed = {
                framework_name: values[canonical_name]
                for canonical_name, framework_name in zip(self.input_names, self.input_mapping)
            }
            outputs = self.model_object.run(self.output_mapping, feed)
        else:
            adapter = self.model_object
            mapped = {
                framework_name: values[canonical_name]
                for canonical_name, framework_name in zip(self.input_names, self.input_mapping)
            }
            torch = sys.modules.get("torch")
            if torch is None:
                result = adapter.infer(mapped)
            else:
                with torch.inference_mode():
                    result = adapter.infer(mapped)
            if isinstance(result, dict):
                missing = set(self.output_mapping) - set(result)
                if missing:
                    raise WorkerError(f"Reference adapter omitted outputs: {sorted(missing)}")
                outputs = [result[name] for name in self.output_mapping]
            elif isinstance(result, (tuple, list)):
                outputs = list(result)
            elif len(self.output_mapping) == 1:
                outputs = [result]
            else:
                raise WorkerError("Reference adapter must return a dict or ordered outputs")
        if len(outputs) != len(self.output_names):
            raise WorkerError(
                f"Reference returned {len(outputs)} outputs; expected {len(self.output_names)}"
            )
        return [
            self._numpy_output(value, name)
            for name, value in zip(self.output_names, outputs)
        ]

    def infer(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.backend == "OpenVINO FP32":
            raise WorkerError("infer command is invalid for OpenVINO FP32 reference")
        iteration = message.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise WorkerError("infer.iteration must be a nonnegative integer")
        values = self._read_inputs(message.get("inputs"))
        with contextlib.redirect_stdout(sys.stderr):
            outputs = self._run_inference(values)
        descriptors = []
        for index, (name, value) in enumerate(zip(self.output_names, outputs)):
            tag = "boolean" if value.dtype == np.dtype("bool") else _DTYPE_TAGS[value.dtype.newbyteorder("=")]
            raw = value.tobytes(order="C")
            path = self.directory / f"reference_{iteration}_{index}.bin"
            try:
                with path.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError as error:
                raise WorkerError(f"Refusing to overwrite existing output: {path}") from error
            descriptors.append({
                "name": name,
                "file": str(path),
                "dtype": tag,
                "shape": list(value.shape),
                "bytes": len(raw),
                "md5": hashlib.md5(raw).hexdigest(),
            })
        return {"status": "ok", "outputs": descriptors}


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerError(f"Cannot read worker config {path}: {error}") from error
    if not isinstance(value, dict):
        raise WorkerError("Worker config must be a JSON object")
    return value


def serve(config_path: Path, directory: Path) -> int:
    try:
        worker = ReferenceWorker(_load_config(config_path), directory, config_path)
        _PROTOCOL_OUT.write(json.dumps(worker.ready, separators=(",", ":"), allow_nan=False) + "\n")
        _PROTOCOL_OUT.flush()
        for line_number, line in enumerate(sys.stdin, 1):
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise WorkerError("Protocol command must be a JSON object")
                command = message.get("command")
                if command == "close":
                    return 0
                if command != "infer":
                    raise WorkerError(f"Unknown command: {command!r}")
                with contextlib.redirect_stdout(sys.stderr):
                    response = worker.infer(message)
                _PROTOCOL_OUT.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
                _PROTOCOL_OUT.flush()
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                _PROTOCOL_OUT.write(json.dumps(response, separators=(",", ":")) + "\n")
                _PROTOCOL_OUT.flush()
                return 1
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        _PROTOCOL_OUT.write(json.dumps(
            {"status": "error", "error": f"{type(error).__name__}: {error}"},
            separators=(",", ":"),
        ) + "\n")
        _PROTOCOL_OUT.flush()
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    return serve(args.config, args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
