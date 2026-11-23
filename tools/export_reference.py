#!/usr/bin/env python3
"""Convert an existing eager FP32 reference dataset to the C++ raw tensor format."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_tensor(array, path, root):
    array = np.ascontiguousarray(array)
    if array.dtype != np.dtype("float32") or not np.isfinite(array).all():
        raise ValueError(f"Expected finite float32 reference tensor: {path}")
    data = array.astype("<f4", copy=False).tobytes()
    with path.open("xb") as stream:
        stream.write(data)
    return {
        "file": str(path.relative_to(root)), "dtype": "f32",
        "shape": list(array.shape), "bytes": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", help="Export only this exact image filename.")
    args = parser.parse_args()
    source_manifest = args.source / "manifest.json"
    original = json.loads(source_manifest.read_text())
    if original["reference_backend"] != "eager" or original["reference_precision"] != "FP32":
        raise ValueError("Expected eager FP32 reference dataset")
    if args.image and args.image not in {row["name"] for row in original["images"]}:
        parser.error(f"Image not in dataset: {args.image}")
    args.output.mkdir(parents=True, exist_ok=False)
    converted = {
        "schema_version": 1, "reference_backend": "PyTorch eager FP32",
        "torch_version": original["torch_version"],
        "source_manifest_sha256": sha256(source_manifest),
        "byte_order": "little", "models": [],
    }
    for model in original["models"]:
        folder = args.output / model["model"].replace(":", "_")
        folder.mkdir()
        samples = []
        for index, sample in enumerate(model["samples"]):
            if args.image and sample["image"] != args.image:
                continue
            input_path, output_path = (args.source / sample[key] for key in ("input", "output"))
            if sha256(input_path) != sample["input_sha256"] or sha256(output_path) != sample["output_sha256"]:
                raise ValueError(f"Reference checksum mismatch: {model['model']} {sample['image']}")
            tensor = np.load(input_path, allow_pickle=False)
            descriptors = []
            with np.load(output_path, allow_pickle=False) as outputs:
                for port in range(len(outputs.files)):
                    descriptors.append(raw_tensor(
                        outputs[f"output_{port}"], folder / f"{index}_output_{port}.bin", args.output,
                    ))
            samples.append({
                "image": sample["image"],
                "input": raw_tensor(tensor, folder / f"{index}_input.bin", args.output),
                "outputs": descriptors,
            })
        converted["models"].append({"model": model["model"], "samples": samples})
    (args.output / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")


if __name__ == "__main__":
    main()
