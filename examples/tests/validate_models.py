#!/usr/bin/env python3
"""Integration validation against existing Python preprocessing and FP32 references."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# examples/tests/ -> project root is parents[1]; external model scripts are two
# more levels up under /home/sgui/2026, as before the move.
ROOT = Path(__file__).resolve().parents[3]
MODELS = [
    ("dinov3-convnext-tiny-pretrain-lvd1689m", 224),
    ("dinov3-vits16-pretrain-lvd1689m", 224),
    ("dinov3-vits16plus-pretrain-lvd1689m", 224),
    ("v2.1:640", 640), ("v2.1:1280", 1280),
    ("v2:640", 640), ("v2:1280", 1280),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dino = load_module("cpp_validation_dino", ROOT / "dinov3/dinov3_benchmark.py")
    yolo = load_module("cpp_validation_yolo", ROOT / "yolo-world/yolo_world_benchmark.py")
    images = sorted(p for p in (ROOT / "test_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    common = SimpleNamespace(random_input=False, input_images=images, batch_size=1, input_size=224)
    command = [str(args.binary.resolve()), "--model-root", str(ROOT / "models_ov")]
    subprocess.run(command + [
        "--mode", "preprocess", "--images", str(ROOT / "test_images"),
        "--output", str(args.output / "preprocess"),
    ], check=True)
    preprocessing = []
    for model, size in MODELS:
        folder = args.output / "preprocess" / model.replace(":", "_")
        payload = json.loads((folder / "inputs.json").read_text())
        expected = dino.make_inputs(common) if model.startswith("dinov3") else yolo.make_inputs(SimpleNamespace(size=size), common)
        if len(payload["samples"]) != len(images):
            raise RuntimeError(f"Incomplete preprocessing coverage: {model}")
        for sample, reference in zip(payload["samples"], expected):
            desc = sample["input"]
            actual = np.fromfile(folder / desc["file"], dtype="<f4").reshape(desc["shape"])
            reference = reference.numpy()
            delta = np.abs(actual - reference)
            row = {
                "model": model, "image": sample["image"],
                "bit_exact": actual.tobytes() == reference.tobytes(),
                "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            }
            preprocessing.append(row)
            if not row["bit_exact"]:
                (args.output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
                raise RuntimeError(f"Native/Python preprocessing differs: {row}")
    (args.output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
    subprocess.run(command + [
        "--mode", "all", "--manifest", str(args.manifest.resolve()), "--prepared-inputs",
        "--warmup", "1", "--iterations", "2", "--rounds", "3",
        "--threshold", "0.99", "--output", str(args.output / "all"),
    ], check=True)
    results = json.loads((args.output / "all/results.json").read_text())
    if results["status"] != "PASS" or len(results["cases"]) != 21:
        raise RuntimeError("Not all 21 cases passed")
    files_checked = 0
    for case in results["cases"]:
        samples = case["image_count"]
        if case["performance"]["samples"] != 2 * samples:
            raise RuntimeError(f"Not per-image timing: {case['model']}")
        if case["accuracy"]["passed_images"] != samples:
            raise RuntimeError(f"Accuracy failed: {case['model']}")
        if not case["stability"]["md5_consistent"] or not case["stability"]["all_outputs_finite"]:
            raise RuntimeError(f"Stability failed: {case['model']}")
        folder = args.output / "all" / case["model"].replace(":", "_") / case["precision"]
        files = json.loads((folder / "files.json").read_text())
        if len(files) != 3 * samples * len(case["outputs"]):
            raise RuntimeError(f"Incomplete binary outputs: {folder}")
        for desc in files:
            file = args.output / "all" / desc["file"]
            if hashlib.md5(file.read_bytes()).hexdigest() != desc["md5"]:
                raise RuntimeError(f"Binary MD5 mismatch: {file}")
            files_checked += 1
    (args.output / "validation.json").write_text(json.dumps({
        "status": "PASS", "preprocessed_inputs": len(preprocessing),
        "all_native_inputs_bit_exact": True, "cases": 21,
        "stability_files_rehashed": files_checked,
    }, indent=2) + "\n")
    print(f"PASS: {len(preprocessing)} bit-exact inputs, 21 real-model cases, {files_checked} rehashed files")


if __name__ == "__main__":
    main()
