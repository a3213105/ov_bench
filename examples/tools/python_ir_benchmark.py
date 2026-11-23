#!/usr/bin/env python3
"""Python OpenVINO counterpart to the native prepared-input performance mode."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import openvino as ov

from benchmark_common import MODELS, require_runtime, save


ARTIFACTS = dict(zip(MODELS, [
    MODELS[0], MODELS[1], MODELS[2],
    "yolo_world_v2_1_m_640", "yolo_world_v2_1_m_1280",
    "yolo_world_v2_m_640", "yolo_world_v2_m_1280",
]))


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def force_sdpa_bf16(model):
    patched = 0
    for node in list(model.get_ordered_ops()):
        if node.get_type_name() != "ScaledDotProductAttention":
            continue
        inputs = [
            node.input_value(index) if index == 3 and node.input_value(index).get_element_type() == ov.Type.boolean
            else ov.opset13.convert(node.input_value(index), ov.Type.bf16).output(0)
            for index in range(node.get_input_size())
        ]
        replacement = ov.opset13.scaled_dot_product_attention(
            *inputs, causal=node.get_attributes()["causal"])
        restored = ov.opset13.convert(replacement, node.get_output_element_type(0))
        restored.set_friendly_name(node.get_friendly_name() + "/sdpa_bf16")
        restored.output(0).set_names(node.output(0).get_names())
        node.output(0).replace(restored.output(0))
        patched += 1
    if patched:
        model.validate_nodes_and_infer_types()
    return patched


def load_inputs(manifest_path, model):
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["models"] if entry["model"] == model]
    if len(entries) != 1:
        raise ValueError(f"Expected one manifest entry for {model}")
    result = []
    for sample in sorted(entries[0]["samples"], key=lambda sample: sample["image"]):
        descriptor = sample["input"]
        path = manifest_path.parent / descriptor["file"]
        if descriptor["dtype"] != "f32" or md5_file(path) != descriptor["md5"]:
            raise ValueError(f"Input dtype/checksum mismatch: {path}")
        array = np.fromfile(path, dtype="<f4")
        if array.size != int(np.prod(descriptor["shape"])) or not np.isfinite(array).all():
            raise ValueError(f"Invalid input data: {path}")
        array = array.reshape(descriptor["shape"])
        result.append((sample["image"], ov.Tensor(array)))
    images = [image for image, _ in result]
    if len(images) != 10 or len(set(images)) != 10:
        raise ValueError("Expected ten distinct input images")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--precision", choices=("FP32", "BF16", "FP16"), required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-signal", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("Invalid threads, warmup or iterations")
    require_runtime(ov.get_version())
    if args.start_signal and (args.start_signal.exists() or not args.start_signal.parent.is_dir()):
        raise ValueError("Start signal must be new and its parent must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING", "cases": [], "openvino": {"build": ov.get_version()},
        "mode": "performance", "threads": args.threads, "warmup": args.warmup,
        "iterations": args.iterations, "prepared_inputs": True,
        "manifest_md5": md5_file(args.manifest),
        "affinity": sorted(os.sched_getaffinity(0)),
        "python_infer_options": {"share_outputs": True},
    }
    save(args.output / "results.json", report)
    try:
        artifact = ARTIFACTS[args.model]
        xml = args.model_root.resolve() / artifact / f"{artifact}.xml"
        size = 224 if args.model.startswith("dinov3") else int(args.model.split(":")[1])
        core = ov.Core()
        model = core.read_model(str(xml))
        if len(model.inputs) != 1 or model.input().get_element_type() != ov.Type.f32:
            raise ValueError("Expected a single f32 input")
        model.reshape([1, 3, size, size])
        samples = load_inputs(args.manifest, args.model)
        if any(list(tensor.shape) != list(model.input().shape) for _, tensor in samples):
            raise ValueError("Prepared input shape mismatch")
        patched = force_sdpa_bf16(model) if args.precision == "FP16" and args.model.startswith("dinov3") else 0
        expected = 12 if args.precision == "FP16" and "vits" in args.model else 0
        if patched != expected:
            raise RuntimeError(f"Expected {expected} patched nodes, got {patched}")
        config = {
            "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
            "INFERENCE_NUM_THREADS": args.threads,
            "INFERENCE_PRECISION_HINT": {"FP32": "f32", "BF16": "bf16", "FP16": "f16"}[args.precision],
            "PERF_COUNT": True,
        }
        compiled = core.compile_model(model, "CPU", config)
        request = compiled.create_infer_request()
        effective = {key: str(compiled.get_property(key)) for key in config}
        outputs = [
            {"dtype": port.get_element_type().get_type_name(), "shape": list(port.shape),
             "names": sorted(port.get_names())}
            for port in compiled.outputs
        ]
        output_dir = args.output / args.model.replace(":", "_") / args.precision
        output_dir.mkdir(parents=True)
        for _ in range(args.warmup):
            for _, tensor in samples:
                request.set_input_tensor(tensor)
                request.infer(share_outputs=True)
        if args.start_signal:
            save(args.output / "ready.json", {"status": "READY"})
            deadline = time.monotonic() + 600
            while not args.start_signal.exists():
                if (args.start_signal.parent / "abort.signal").exists():
                    raise RuntimeError("Multi-process wave aborted")
                if time.monotonic() > deadline:
                    raise TimeoutError("Timed out waiting for start signal")
                time.sleep(0.01)
        latencies = []
        for _ in range(args.iterations):
            for _, tensor in samples:
                request.set_input_tensor(tensor)
                start = time.perf_counter_ns()
                request.infer(share_outputs=True)
                latencies.append((time.perf_counter_ns() - start) / 1e6)
        save(output_dir / "latencies_ms.json", latencies)
        validity = []
        for index, (image, tensor) in enumerate(samples):
            request.set_input_tensor(tensor)
            request.infer(share_outputs=True)
            finite = True
            for port_index in range(len(outputs)):
                data = np.array(request.get_output_tensor(port_index).data, copy=True)
                finite = finite and bool(np.isfinite(data).all())
                with (output_dir / f"check_{index}_{port_index}.bin").open("xb") as stream:
                    stream.write(data.tobytes(order="C"))
            validity.append({"image": image, "finite": finite})
        case = {
            "model": args.model, "precision": args.precision,
            "image_count": len(samples), "sdpa_bf16_nodes": patched,
            "ir": {"xml": str(xml), "xml_md5": md5_file(xml), "bin_md5": md5_file(xml.with_suffix(".bin"))},
            "outputs": outputs, "output_validity": validity, "effective_config": effective,
            "performance": {
                "samples": len(latencies), "dataset_rounds": args.iterations,
                "batch_size": 1, "latency_unit": "single_input_batch",
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50, method="linear")),
                "p90_ms": float(np.percentile(latencies, 90, method="linear")),
                "p99_ms": float(np.percentile(latencies, 99, method="linear")),
                "throughput_fps": 1000 * len(latencies) / sum(latencies),
            },
            "status": "PASS" if all(row["finite"] for row in validity) else "FAIL",
        }
        report["cases"] = [case]
        report["status"] = case["status"]
        save(args.output / "results.json", report)
        print(f"{args.model} {args.precision} {case['status']}", flush=True)
        return 0 if case["status"] == "PASS" else 1
    except Exception as error:
        report.update(status="ERROR", error=str(error))
        save(args.output / "results.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
