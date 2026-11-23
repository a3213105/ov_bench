#!/usr/bin/env python3
"""Snapshot CPU compilation in isolated processes; never create requests or infer."""

import argparse
import csv
from collections import defaultdict
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
        "fsmblks", "uordblks", "fordblks", "keepcost")]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(stage, directory, allocator):
    allocated = allocator()
    row = {"stage": stage, "timestamp_steady_ns": time.monotonic_ns()}
    for name in ("uordblks", "fordblks", "hblkhd"):
        row[f"allocator_{name}_bytes"] = getattr(allocated, name)
    text = Path("/proc/self/smaps_rollup").read_text()
    (directory / f"{stage}.smaps_rollup").write_text(text)
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == "kB":
            row[fields[0].rstrip(":") + "_kib"] = int(fields[1])
    (directory / f"{stage}.maps").write_text(Path("/proc/self/maps").read_text())
    return row


def worker(args):
    import openvino as ov

    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    libc = ctypes.CDLL("libc.so.6")
    allocator = libc.mallinfo2
    allocator.argtypes = []
    allocator.restype = Mallinfo2
    os.environ["OV_CPU_COMPILE_MEMORY_STATS_PATH"] = str(directory / "compile.jsonl")
    package = Path(ov.__file__).resolve().parent
    report = {"model": args.model.stem, "precision": args.precision,
              "openvino": ov.get_version(), "package": str(package), "pid": os.getpid(),
              "source_sha256": sha256(args.model), "snapshots": [], "status": "RUNNING",
              "scope": "compile only; no InferRequest, inputs, inference, trim or model reshape"}

    def record(name):
        row = snapshot(name, directory, allocator)
        report["snapshots"].append(row)
        save(directory / "result.json", report)
        print(name, round(row["Rss_kib"] / 1024, 2), "MiB", flush=True)

    try:
        core = ov.Core()
        core.get_versions("CPU")
        record("A0_plugin_loaded")
        model = core.read_model(args.model)
        record("A1_model_loaded")
        config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                  "INFERENCE_NUM_THREADS": args.threads, "ENABLE_CPU_PINNING": False,
                  "PERF_COUNT": False,
                  "INFERENCE_PRECISION_HINT": {"FP32": "f32", "BF16": "bf16", "FP16": "f16"}[args.precision]}
        report["config"] = config
        compiled = core.compile_model(model, "CPU", config)
        record("A2_compile_returned")
        records = [json.loads(line) for line in (directory / "compile.jsonl").read_text().splitlines()]
        names = [row["stage"] for row in records]
        for required in ("C0_compile_entry", "C1_model_cloned", "C2_transformations_complete",
                         "C4_graph_initialized", "C5_primitives_created",
                         "C6_cleanup_complete", "C3_compiled_model_constructed"):
            if required not in names:
                raise RuntimeError(f"Installed wheel lacks required snapshot: {required}")
        report["compile_snapshots"] = records
        maps = (directory / "A2_compile_returned.maps").read_text()
        loaded = set()
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and ".so" in fields[5]:
                path = Path(fields[5])
                if path.is_file() and ("libopenvino" in path.name or "libtbb" in path.name):
                    loaded.add(path.resolve())
        if not any("intel_cpu_plugin" in path.name for path in loaded):
            raise RuntimeError("CPU plugin absent from loaded mappings")
        if any(not path.is_relative_to(package) for path in loaded):
            raise RuntimeError(f"Mixed OpenVINO/TBB libraries: {sorted(map(str, loaded))}")
        report["loaded_libraries"] = {str(path): sha256(path) for path in sorted(loaded)}
        del records
        del model
        gc.collect()
        record("A3_source_model_released_diagnostic")
        del compiled
        gc.collect()
        record("A4_compiled_model_released_diagnostic")
        del core
        gc.collect()
        record("A5_core_released_diagnostic")
        analyze_descriptors(directory)
        report["status"] = "PASS"
        save(directory / "result.json", report)
    except Exception as error:
        report.update(status="ERROR", error=str(error))
        save(directory / "result.json", report)
        raise


def metrics(row):
    return {"rss_mib": row["Rss_kib"] / 1024,
            "pss_mib": row["Pss_kib"] / 1024,
            "anonymous_mib": row["Anonymous_kib"] / 1024,
            "private_dirty_mib": row["Private_Dirty_kib"] / 1024,
            "live_mib": (row["allocator_uordblks_bytes"] + row["allocator_hblkhd_bytes"]) / 2**20,
            "free_arena_mib": row["allocator_fordblks_bytes"] / 2**20}


def analyze_descriptors(directory):
    allocation_path = directory / "compile.jsonl.descriptor_allocations.jsonl"
    if not allocation_path.exists():
        return
    allocations = [json.loads(line) for line in allocation_path.read_text().splitlines()]
    retention = [json.loads(line) for line in
                 (directory / "compile.jsonl.descriptor_retention.jsonl").read_text().splitlines()]
    types = defaultdict(lambda: {"nodes": 0, "live_delta_bytes": 0, "candidates": 0})
    for row in allocations:
        group = types[row["type"]]
        for key, value in (("nodes", 1), ("live_delta_bytes", row["live_delta_bytes"]),
                           ("candidates", row["candidates"])):
            group[key] += value
    ranked = sorted(({"type": name, **values} for name, values in types.items()),
                    key=lambda row: row["live_delta_bytes"], reverse=True)
    stages = {}
    for stage in ("after_selection", "after_cleanup"):
        nodes = [row for row in retention if row["stage"] == stage and "node" in row]
        graphs = [row for row in retention if row["stage"] == stage and "scope" in row]
        totals = {"nodes": len(nodes), "graphs": graphs}
        for key in ("candidates", "selected", "unselected", "candidate_vector_capacity_bytes",
                    "unselected_slot_bytes", "spare_slot_bytes", "selected_port_capacity_bytes",
                    "unselected_port_capacity_bytes", "dnnl_descriptor_count", "dnnl_vector_capacity_bytes"):
            totals[key] = sum(row.get(key, 0) for row in nodes)
        totals["dynamic_nodes"] = sum(row["dynamic"] for row in nodes)
        totals["dynamic_unselected"] = sum(row["unselected"] for row in nodes if row["dynamic"])
        totals["static_unselected"] = totals["unselected"] - totals["dynamic_unselected"]
        stages[stage] = totals
    summary = {"allocation_nodes": len(allocations),
               "allocation_live_delta_bytes": sum(row["live_delta_bytes"] for row in allocations),
               "types": ranked, "retention": stages,
               "scope": "Measured allocation intervals and known container/object payloads, "
                        "not a proof of reclaimability or full allocator accounting."}
    save(directory / "descriptor_summary.json", summary)
    with (directory / "descriptor_types.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("type", "nodes", "live_delta_bytes", "candidates"), delimiter="\t")
        writer.writeheader()
        writer.writerows(ranked)


def summarize(output, results):
    rows = []
    totals = []
    for result in results:
        app = result["snapshots"]
        chain = [*app[:2], *result["compile_snapshots"], *app[2:]]
        previous = None
        for current in chain:
            values = metrics(current)
            row = {"model": result["model"], "precision": result["precision"],
                   "repeat": result["repeat"], "stage": current["stage"], **values}
            row.update({"delta_" + key: values[key] - metrics(previous)[key] if previous else 0
                        for key in values})
            for key in ("model_nodes", "constants", "constant_unique_logical_span_bytes", "cpu_nodes", "cpu_edges"):
                row[key] = current.get(key)
            rows.append(row)
            previous = current
        loaded, returned = metrics(app[1]), metrics(app[2])
        peak = max(metrics(row)["rss_mib"] for row in chain[:len(chain) - 3])
        totals.append({"model": result["model"], "precision": result["precision"],
                       "repeat": result["repeat"], "loaded_rss_mib": loaded["rss_mib"],
                       "returned_rss_mib": returned["rss_mib"], "sampled_peak_rss_mib": peak,
                       "compile_rss_mib": returned["rss_mib"] - loaded["rss_mib"],
                       "compile_live_mib": returned["live_mib"] - loaded["live_mib"],
                       "source_release_live_mib": metrics(app[3])["live_mib"] - returned["live_mib"],
                       "compiled_release_live_mib": metrics(app[4])["live_mib"] - metrics(app[3])["live_mib"]})
    for filename, entries in (("snapshots.tsv", rows), ("summary.tsv", totals)):
        if entries:
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)
    save(output / "summary.json", {"cases": totals, "snapshots": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/home/sgui/2026/models/pred_model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--precisions", nargs="+", choices=("FP32", "BF16", "FP16"), default=["FP32"])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--precision", choices=("FP32", "BF16", "FP16"), default="FP32")
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.timeout) <= 0:
        parser.error("threads, repeats and timeout must be positive")
    if args.worker:
        if not args.model:
            parser.error("--worker requires --model")
        return worker(args)
    models = sorted(args.model_root.glob("*.onnx"))
    if not models:
        parser.error("No ONNX models found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME",
                "OV_CPU_COMPILE_MEMORY_STATS_PATH", "OVBENCH_CPU_PLUGIN", "OVBENCH_CPU_PLUGIN_CONFIG"):
        environment.pop(key, None)
    results = []
    failures = []
    save(output / "configuration.json", {"python": sys.executable, "threads": args.threads,
         "repeats": args.repeats, "precisions": args.precisions, "script_sha256": sha256(Path(__file__)),
         "affinity": sorted(os.sched_getaffinity(0)),
         "environment": {key: value for key, value in environment.items()
                         if key.startswith(("MALLOC_", "ONEDNN_", "DNNL_", "OMP_"))}})
    for model in models:
        for precision in args.precisions:
            for repeat in range(1, args.repeats + 1):
                case = output / f"{model.stem}_{precision}_{repeat}"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", str(model.resolve()),
                           "--output", str(case), "--precision", precision, "--threads", str(args.threads)]
                print("START", case.name, flush=True)
                with case.with_suffix(".log").open("w") as log:
                    try:
                        completed = subprocess.run(command, env=environment, stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=args.timeout)
                        if completed.returncode:
                            raise RuntimeError(f"exit {completed.returncode}")
                        result = json.loads((case / "result.json").read_text())
                        if result["status"] != "PASS":
                            raise RuntimeError(result.get("error", "incomplete worker"))
                        result["repeat"] = repeat
                        results.append(result)
                    except (subprocess.TimeoutExpired, RuntimeError, OSError, ValueError) as error:
                        failures.append({"case": case.name, "error": str(error)})
                summarize(output, results)
                save(output / "failures.json", failures)
                print("END", case.name, flush=True)
    (output / "state").write_text("ERROR\n" if failures else "PASS\n")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())