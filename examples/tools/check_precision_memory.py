#!/usr/bin/env python3
"""Reproduce CPU FP32/BF16 memory differences in fresh, sequential processes."""

import argparse
from collections import Counter
import csv
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from benchmark_common import MODELS, require_runtime, save


def memory_snapshot(pid):
    result = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            result[parts[0].rstrip(":") + "_kib"] = int(parts[1])
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("VmRSS:", "VmHWM:", "VmSize:", "Threads:"):
            result[parts[0].rstrip(":")] = int(parts[1])
    return result


def worker(args):
    import numpy as np
    import openvino as ov
    from python_ir_benchmark import ARTIFACTS, load_inputs, md5_file

    args.output.mkdir(parents=True, exist_ok=False)
    require_runtime(ov.get_version())
    report = {
        "status": "RUNNING", "model": args.model, "precision": args.precision,
        "openvino": ov.get_version(), "pid": os.getpid(),
        "config": {}, "stages": [], "trim_diagnostic": args.trim,
        "keep_model": args.keep_model, "mmap": args.mmap,
        "affinity": sorted(os.sched_getaffinity(0)),
        "max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA", "default"),
    }

    def stage(name):
        record = {"name": name, "time_ns": time.time_ns(), **memory_snapshot(os.getpid()),
                  "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["stages"].append(record)
        save(args.output / "result.json", report)
        print(json.dumps(record), flush=True)

    def trim():
        gc.collect()
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc.malloc_trim(0)

    try:
        stage("imports")
        samples = load_inputs(args.manifest, args.model)
        stage("inputs_loaded")
        core = ov.Core()
        core.set_property("CPU", {"ENABLE_MMAP": args.mmap})
        core.get_versions("CPU")
        stage("plugin_loaded")
        artifact = ARTIFACTS[args.model]
        xml = args.model_root / artifact / f"{artifact}.xml"
        report["ir"] = {"xml": str(xml.resolve()), "xml_md5": md5_file(xml),
                        "bin_md5": md5_file(xml.with_suffix(".bin")), "bin_bytes": xml.with_suffix(".bin").stat().st_size}
        model = core.read_model(str(xml))
        size = 224 if args.model.startswith("dinov3") else int(args.model.split(":")[1])
        model.reshape([1, 3, size, size])
        report["original_constants"] = dict(Counter(
            node.get_element_type().get_type_name() for node in model.get_ops()
            if node.get_type_name() == "Constant"))
        stage("model_loaded")
        config = {
            "PERFORMANCE_HINT": args.hint, "NUM_STREAMS": args.streams if args.streams else "AUTO",
            "INFERENCE_NUM_THREADS": args.threads,
            "INFERENCE_PRECISION_HINT": "f32" if args.precision == "FP32" else "bf16",
            "PERF_COUNT": True,
        }
        report["requested_config"] = config
        compiled = core.compile_model(model, "CPU", config)
        stage("compiled_model_alive")
        report["config"] = {key: str(compiled.get_property(key)) for key in
                            (*config, "ENABLE_CPU_PINNING", "ENABLE_HYPER_THREADING")}
        if not args.keep_model:
            del model
            gc.collect()
        stage("source_model_policy_applied")
        if args.trim:
            report["trim_after_compile"] = trim()
            stage("after_compile_trim")
        request = compiled.create_infer_request()
        stage("request_created")
        finite = True
        for round_index in range(args.rounds):
            for _, tensor in samples:
                request.set_input_tensor(tensor)
                request.infer(share_outputs=True)
            if round_index in (0, 2, args.rounds - 1):
                stage(f"inference_round_{round_index + 1}")
        for port_index in range(len(compiled.outputs)):
            finite = finite and bool(np.isfinite(request.get_output_tensor(port_index).data).all())
        if not finite:
            raise RuntimeError("Non-finite inference output")
        stage("steady")
        if args.trim:
            report["trim_after_inference"] = trim()
            stage("after_inference_trim")
        # Inspect the execution graph after all measured snapshots to avoid
        # counting Python graph-wrapper allocations as inference memory.
        runtime = compiled.get_runtime_model()
        nodes = []
        for node in runtime.get_ordered_ops():
            nodes.append({
                "name": node.get_friendly_name(),
                "rt_info": {key: value.astype(str) for key, value in node.get_rt_info().items()},
                "outputs": [{"dtype": output.get_element_type().get_type_name(),
                             "shape": str(output.get_partial_shape())} for output in node.outputs()],
            })
        save(args.output / "runtime_graph.json", nodes)
        report["runtime_precision_counts"] = dict(Counter(
            node["rt_info"].get("runtimePrecision", "") for node in nodes))
        report["layer_type_counts"] = dict(Counter(
            node["rt_info"].get("layerType", "") for node in nodes))
        report["status"] = "PASS"
        save(args.output / "result.json", report)
        return 0
    except Exception as error:
        report.update(status="ERROR", error=str(error))
        save(args.output / "result.json", report)
        raise


def controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "configuration.json", {
        "python": sys.executable, "threads": args.threads, "rounds": args.rounds,
        "repeats": args.repeats, "models": args.models, "trim": args.trim,
        "keep_model": args.keep_model, "mmap": args.mmap,
        "hint": args.hint, "streams": args.streams,
        "max_cpu_isa": args.max_isa,
        "affinity": sorted(os.sched_getaffinity(0)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "One fresh process and one compiled CPU model per case; sequential runs.",
    })
    (args.output / "state").write_text("running")
    (args.output / "runner.pid").write_text(str(os.getpid()))
    rows = []
    try:
        for model in args.models:
            for repeat in range(1, args.repeats + 1):
                order = ("FP32", "BF16") if repeat % 2 else ("BF16", "FP32")
                for precision in order:
                    root = args.output / model.replace(":", "_") / f"repeat_{repeat}" / precision
                    root.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--model", model, "--precision", precision,
                        "--model-root", str(args.model_root.resolve()),
                        "--manifest", str(args.manifest.resolve()),
                        "--output", str(root), "--threads", str(args.threads),
                        "--rounds", str(args.rounds),
                    ]
                    if args.trim:
                        command.append("--trim")
                    if args.keep_model:
                        command.append("--keep-model")
                    if not args.mmap:
                        command.append("--no-mmap")
                    command += ["--hint", args.hint, "--streams", str(args.streams)]
                    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1")
                    if args.max_isa:
                        environment["ONEDNN_MAX_CPU_ISA"] = args.max_isa
                    print(f"START {model} {precision} repeat={repeat}", flush=True)
                    sampled_peak = 0
                    with root.with_suffix(".log").open("w") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                        try:
                            deadline = time.monotonic() + 1800
                            while process.poll() is None:
                                try:
                                    for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                                        if line.startswith("VmRSS:"):
                                            sampled_peak = max(sampled_peak, int(line.split()[1]))
                                except FileNotFoundError:
                                    pass
                                if time.monotonic() > deadline:
                                    raise TimeoutError(f"Memory test exceeded 30 minutes: {root}")
                                time.sleep(0.01)
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                    if process.returncode:
                        raise RuntimeError(f"Worker exit {process.returncode}: {root.with_suffix('.log')}")
                    payload = json.loads((root / "result.json").read_text())
                    if payload["status"] != "PASS":
                        raise RuntimeError(f"Failed memory case: {root}")
                    row = {"model": model, "precision": precision, "repeat": repeat,
                           "sampled_process_peak_rss_mib": sampled_peak / 1024}
                    for record in payload["stages"]:
                        for metric in ("Rss_kib", "Pss_kib", "Anonymous_kib", "Private_Dirty_kib", "VmHWM"):
                            row[f"{record['name']}_{metric}_mib"] = record[metric] / 1024
                    rows.append(row)
                    save(args.output / "summary.json", rows)
                    fields = list(dict.fromkeys(key for entry in rows for key in entry))
                    with (args.output / "summary.tsv").open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"DONE {model} {precision} steady RSS={row['steady_Rss_kib_mib']:.2f} MiB", flush=True)
        (args.output / "state").write_text("completed")
        return 0
    except Exception as error:
        (args.output / "state").write_text("error")
        (args.output / "error.txt").write_text(str(error) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--hint", choices=("LATENCY", "THROUGHPUT"), default="LATENCY")
    parser.add_argument("--streams", type=int, default=1, help="0 selects AUTO")
    parser.add_argument("--max-isa", choices=("AVX512_CORE_BF16", "AVX512_CORE"),
                        help="Diagnostic oneDNN ISA cap; changes kernels, not a production recommendation")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--trim", action="store_true", help="Diagnostic malloc_trim, not a production fix")
    parser.add_argument("--keep-model", action="store_true", help="Retain source ov.Model through inference")
    parser.add_argument("--no-mmap", dest="mmap", action="store_false")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--precision", choices=("FP32", "BF16"))
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.repeats) < 1 or args.streams < 0:
        parser.error("threads, rounds and repeats must be positive")
    if args.worker and (not args.model or not args.precision):
        parser.error("worker requires model and precision")
    return worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
