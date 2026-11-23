#!/usr/bin/env python3
"""Orchestrate CPU-bound C++ processes; inference and metrics remain native C++."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


from benchmark_common import MODELS, save


def assignments():
    rows = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    allowed = os.sched_getaffinity(0)
    nodes = {}
    for line in rows.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, node = map(int, line.split(","))
        if cpu in allowed:
            nodes.setdefault(node, {}).setdefault((socket, core), []).append(cpu)
    result = []
    for node, cores in sorted(nodes.items()):
        physical = [sorted(cpus) for _, cpus in sorted(cores.items()) if len(cpus) == 2]
        for offset in range(0, min(len(physical), 30), 6):
            group = physical[offset:offset + 6]
            if len(group) == 6:
                result.append({"numa": node, "cpus": sorted(cpu for cpus in group for cpu in cpus)})
        if len(result) >= 15:
            break
    if len(result) != 15:
        raise RuntimeError(f"Expected 15 groups of 6 physical cores / 12 logical CPUs, got {len(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("performance", "stability"), required=True)
    args = parser.parse_args()
    binding = assignments()
    if args.mode == "stability" and shutil.disk_usage(args.output.parent).free < 150_000_000_000:
        raise RuntimeError("Stability run requires at least 150 GB free")
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    save(args.output / "configuration.json", {
        "mode": args.mode, "bindings": binding, "models": MODELS,
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "precisions": ["FP32", "BF16", "FP16"], "threads": 6,
        "warmup": 3, "iterations": 200, "stability_rounds": 10,
    })
    (args.output / "runner.pid").write_text(str(os.getpid()))
    (args.output / "started_at").write_text(datetime.now().isoformat())
    (args.output / "state").write_text("running")
    summaries = []
    try:
        for model in MODELS:
            for precision in ("FP32", "BF16", "FP16"):
                case_dir = args.output / model.replace(":", "_") / precision
                case_dir.mkdir(parents=True)
                print(f"START {model} {precision}", flush=True)

                def worker(index):
                    output = case_dir / f"instance_{index:02d}"
                    command = [
                        "taskset", "-c", ",".join(map(str, binding[index]["cpus"])),
                        str(binary), "--model-root", str(args.model_root.resolve()),
                        "--images", str(args.images.resolve()), "--model", model,
                        "--precisions", precision, "--mode", args.mode, "-batch", "0",
                        "--threads", "6", "--warmup", "3", "--iterations", "200",
                        "--rounds", "10", "--output", str(output),
                    ]
                    with (case_dir / f"instance_{index:02d}.log").open("w") as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        raise RuntimeError(f"{model} {precision} instance {index} exit {completed.returncode}; see {log.name}")
                    payload = json.loads((output / "results.json").read_text())
                    if not payload["openvino"]["build"].startswith("2025.4.1"):
                        raise RuntimeError("Unexpected OpenVINO build")
                    if payload["status"] != "PASS" or len(payload["cases"]) != 1:
                        raise RuntimeError(f"Failed/incomplete results: {output}")
                    result = payload["cases"][0]
                    expected = 12 if precision == "FP16" and "vits" in model else 0
                    if result["sdpa_bf16_nodes"] != expected or result["image_count"] != 10:
                        raise RuntimeError(f"Wrong model configuration: {output}")
                    if args.mode == "performance" and result["performance"]["samples"] != 2000:
                        raise RuntimeError(f"Incorrect per-image timing coverage: {output}")
                    if args.mode == "stability" and result["stability"]["rounds"] != 10:
                        raise RuntimeError(f"Incomplete stability rounds: {output}")
                    return result

                with ThreadPoolExecutor(max_workers=15) as pool:
                    results = list(pool.map(worker, range(15)))
                row = {"model": model, "precision": precision, "instances": 15, "status": "PASS"}
                if args.mode == "performance":
                    fps = [result["performance"]["throughput_fps"] for result in results]
                    row.update(
                        throughput_per_instance_fps=sum(fps) / 15,
                        throughput_fps=sum(fps),
                        p99_ms=max(result["performance"]["p99_ms"] for result in results),
                    )
                else:
                    with (case_dir / "comparison.log").open("w") as log:
                        status = subprocess.run(
                            [str(binary), "--mode", "compare-stability",
                             "--output", str(case_dir / "comparison.json"), "--stability-dir", str(case_dir)],
                            stdout=log, stderr=subprocess.STDOUT,
                        )
                    if status.returncode not in (0, 1):
                        raise RuntimeError(f"Comparator error: {case_dir}; see comparison.log")
                    report = json.loads((case_dir / "comparison.json").read_text())
                    row.update(
                        status=report["status"], output_groups=len(report["groups"]),
                        binary_files=report["binary_files"], processes=len(report["processes"]),
                    )
                    expected_groups = 10 * len(results[0]["outputs"])
                    if (len(report["processes"]) != 15 or len(report["groups"]) != expected_groups
                            or report["binary_files"] != 15 * 10 * expected_groups):
                        raise RuntimeError(f"Incomplete cross-process comparison coverage: {case_dir}")
                summaries.append(row)
                save(args.output / "summary.json", summaries)
                fields = list(dict.fromkeys(key for entry in summaries for key in entry))
                with (args.output / "summary.tsv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(summaries)
                print(f"DONE {json.dumps(row)}", flush=True)
        passed = len(summaries) == 21 and all(row["status"] == "PASS" for row in summaries)
        (args.output / "state").write_text("completed" if passed else "failed_comparison")
        return 0 if passed else 1
    except Exception as error:
        (args.output / "state").write_text("error")
        (args.output / "error.txt").write_text(str(error) + "\n")
        raise
    finally:
        (args.output / "finished_at").write_text(datetime.now().isoformat())


if __name__ == "__main__":
    raise SystemExit(main())
