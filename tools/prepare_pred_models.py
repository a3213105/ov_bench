#!/usr/bin/env python3
"""Prepare deterministic synthetic prediction-model inputs and FP32 references.

These are numerical regression inputs, not business-accuracy ground truth. Boolean
True is assumed to mean valid; the models' actual business mask semantics are
unknown. Every last-axis mask group has a valid entry, and groups of size > 1
also have an invalid entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


DEFAULT_MODEL_ROOT = Path("/home/sgui/2026/models/pred_model")
DEFAULT_SEED = 43
DTYPES = {
    np.dtype("float32"): "f32",
    np.dtype("bool"): "boolean",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
}


@dataclass(frozen=True)
class InputSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def metadata(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


def resolve_input_shape(dimensions: list[int | str | None]) -> tuple[int, ...]:
    """Resolve only a dynamic leading batch axis; never guess other dimensions."""
    shape = []
    for axis, dimension in enumerate(dimensions):
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            if dimension <= 0:
                raise ValueError(f"Input axis {axis} has invalid dimension {dimension}")
            shape.append(dimension)
        elif axis == 0:
            shape.append(1)
        else:
            raise ValueError(
                f"Unresolved non-batch input dimension at axis {axis}: {dimension!r}"
            )
    return tuple(shape)


def inspect_inputs(path: Path) -> list[InputSpec]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    specs = []
    input_types = {onnx.TensorProto.FLOAT: "f32", onnx.TensorProto.BOOL: "boolean"}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        if not value.type.HasField("tensor_type"):
            raise ValueError(f"{path.name}: non-tensor input {value.name!r}")
        tensor = value.type.tensor_type
        if tensor.elem_type not in input_types:
            raise ValueError(
                f"{path.name}: unsupported input type {tensor.elem_type} for {value.name!r}"
            )
        if not tensor.HasField("shape"):
            raise ValueError(f"{path.name}: missing input rank for {value.name!r}")
        dimensions = [
            d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in tensor.shape.dim
        ]
        specs.append(
            InputSpec(value.name, input_types[tensor.elem_type], resolve_input_shape(dimensions))
        )
    if not specs:
        raise ValueError(f"{path.name}: no runtime inputs found")
    return specs


def generate_inputs(
    specs: list[InputSpec], seed: int, model_name: str, sample_index: int
) -> dict[str, np.ndarray]:
    if seed < 0 or sample_index < 0:
        raise ValueError("Seed and sample index must be nonnegative")
    result = {}
    for spec in specs:
        digest = hashlib.sha256(f"{model_name}\0{spec.name}".encode()).digest()
        entropy = [seed, sample_index, *np.frombuffer(digest, dtype="<u4").tolist()]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        if spec.dtype == "f32":
            value = rng.normal(0.0, 0.1, size=spec.shape).astype(np.float32)
        elif spec.dtype == "boolean":
            value = np.asarray(rng.random(spec.shape) < 0.75, dtype=np.bool_)
            if value.ndim == 0:
                value[...] = True
            else:
                if value.shape[-1] == 0:
                    raise ValueError(f"{spec.name}: empty mask group")
                value[..., 0] = True
                if value.shape[-1] > 1:
                    value[..., -1] = False
        else:
            raise ValueError(f"{spec.name}: unsupported synthetic input dtype {spec.dtype}")
        result[spec.name] = np.ascontiguousarray(value) if value.ndim else value
    return result


def validate_output(value: np.ndarray, model: str, sample: str, name: str) -> None:
    if value.dtype not in (np.dtype("float32"), np.dtype("int32")):
        raise ValueError(
            f"{model}/{sample}: output {name!r} has unsupported dtype {value.dtype}; "
            "expected float32 or int32 (no implicit conversion)"
        )
    if not np.isfinite(value).all():
        count = int(np.count_nonzero(~np.isfinite(value)))
        raise ValueError(
            f"{model}/{sample}: output {name!r} has {count}/{value.size} nonfinite values"
        )


def write_tensor(path: Path, value: np.ndarray, name: str, root: Path) -> dict:
    array = np.asarray(value)
    if array.dtype not in DTYPES:
        raise ValueError(f"{name}: unsupported raw tensor dtype {array.dtype}")
    # The raw interchange format is explicitly little-endian and C-contiguous.
    raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "name": name,
        "file": path.relative_to(root).as_posix(),
        "dtype": DTYPES[array.dtype],
        "shape": list(array.shape),
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model(path: Path, root: Path, samples: int, seed: int) -> dict:
    import onnxruntime as ort

    specs = inspect_inputs(path)
    model_dir = root / path.stem
    model_dir.mkdir()
    onnx = model_dir / f"{path.stem}.onnx"
    shutil.copyfile(path, onnx)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"{path.name}: expected CPU-only ONNX Runtime provider")
    output_names = [output.name for output in session.get_outputs()]
    record = {
        "model": path.stem,
        "onnx": onnx.relative_to(root).as_posix(),
        "source_sha256": sha256_file(path),
        "inputs": [spec.metadata() for spec in specs],
        "samples": [],
    }
    for sample_index in range(samples):
        image = f"synthetic_{sample_index:02d}"
        values = generate_inputs(specs, seed, path.stem, sample_index)
        outputs = session.run(output_names, values)
        for name, value in zip(output_names, outputs):
            validate_output(value, path.stem, image, name)
        sample_dir = model_dir / image
        record["samples"].append(
            {
                "image": image,
                "inputs": [
                    write_tensor(sample_dir / f"input_{i:02d}.bin", values[spec.name], spec.name, root)
                    for i, spec in enumerate(specs)
                ],
                "outputs": [
                    write_tensor(sample_dir / f"output_{i:02d}.bin", value, name, root)
                    for i, (name, value) in enumerate(zip(output_names, outputs))
                ],
            }
        )
        print(f"{path.stem}/{image}: {len(outputs)} outputs finite", flush=True)
    return record


def prepare(model_root: Path, output: Path, samples: int = 10, seed: int = DEFAULT_SEED) -> Path:
    if samples <= 0 or seed < 0:
        raise ValueError("--samples must be positive and --seed must be nonnegative")
    model_root = model_root.resolve()
    output = output.resolve()
    if not model_root.is_dir():
        raise ValueError(f"Model root is not a directory: {model_root}")
    paths = sorted(model_root.glob("*.onnx"))
    if not paths:
        raise ValueError(f"No ONNX models found in {model_root}")
    # A fresh directory prevents accidental mixing with earlier benchmark data.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "reference_backend": "ONNX Runtime FP32 CPU",
        "seed": seed,
        "synthetic_data": {
            "purpose": "numerical consistency reference, not business accuracy",
            "float_distribution": "normal(mean=0, stddev=0.1), float32",
            "mask_semantics": "ASSUMPTION: True means valid; business semantics unknown",
            "mask_groups": "last axis; at least one valid, and one invalid when size > 1",
            "dynamic_dimensions": "original ONNX is loaded directly; synthetic inputs use batch 1",
            "raw_byte_order": "little-endian",
        },
        "models": [],
    }
    for path in paths:
        manifest["models"].append(prepare_model(path, output, samples, seed))
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="new, unique scratch directory")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        destination = prepare(args.model_root, args.output, args.samples, args.seed)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Manifest: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
