#!/usr/bin/env python3
"""Compile-time release experiments in isolated native processes."""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=("FP32", "FP16", "BF16"), default="FP32")
    parser.add_argument("--experiment", choices=("mlas-source", "fc-compile-artifacts", "graph-disconnect"),
                        default="mlas-source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.experiment == "fc-compile-artifacts"
    graph_disconnect = args.experiment == "graph-disconnect"
    release_flag, events_suffix = {
        "mlas-source": ("OV_CPU_EXPERIMENTAL_RELEASE_MLAS_SOURCE", ".mlas_source_release.jsonl"),
        "fc-compile-artifacts": ("OV_CPU_EXPERIMENTAL_RELEASE_FC_COMPILE_ARTIFACTS", ".fc_compile_artifacts.jsonl"),
        "graph-disconnect": ("OV_CPU_EXPERIMENTAL_DISCONNECT_OV_GRAPH", ".graph_disconnect.jsonl"),
    }[args.experiment]
    args.output.mkdir()
    cases = {}
    clean_env = os.environ.copy()
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "OVBENCH_THREE_BATCHES",
                "OVBENCH_CPU_PLUGIN", "OVBENCH_CPU_PLUGIN_CONFIG",
                "OV_CPU_EXPERIMENTAL_RELEASE_MLAS_SOURCE", "OV_CPU_COMPILE_MEMORY_STATS_PATH",
                "OV_CPU_RUNTIME_MEMORY_STATS_PATH", "OV_CPU_EXPERIMENTAL_RELEASE_FC_COMPILE_ARTIFACTS",
                "OV_CPU_EXPERIMENTAL_DISCONNECT_OV_GRAPH"):
        clean_env.pop(key, None)
    clean_env.update(OVBENCH_VERIFY_STABILITY="1", OV_CPU_EXPERIMENTAL_RELEASE_ELTWISE_FACTORY="0")

    for dynamic in (False, True):
        for release in (False, True):
            name = ("release" if release else "baseline") + ("_dynamic" if dynamic else "_fixed")
            baseline_name = "baseline_dynamic" if dynamic else "baseline_fixed"
            if release and cases[baseline_name]["eligible_count"] == 0:
                cases[name] = {
                    "status": "NOT_APPLICABLE", "reason": "Baseline has no eligible release targets",
                    "native_pass": False, "release_confirmed": False, "result": None, "stages": [],
                }
                print(name, "NOT_APPLICABLE: no eligible release targets", flush=True)
                save(args.output / "summary.json", {
                    "status": "RUNNING", "precision": args.precision, "cases": cases})
                continue
            directory = args.output / name
            trace = args.output / (name + ".jsonl")
            env = clean_env.copy()
            env["OV_CPU_COMPILE_MEMORY_STATS_PATH"] = str(trace)
            if dynamic:
                env["OVBENCH_THREE_BATCHES"] = "1"
            if release:
                env[release_flag] = "1"
            command = [str(args.binary), str(args.model_root), str(args.inputs), args.model,
                       "ov", args.precision, "6", "3" if dynamic else "10", str(directory)]
            print(args.experiment, name, flush=True)
            with (args.output / (name + ".log")).open("w") as log:
                process = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            row = {
                "returncode": process.returncode,
                "signal": signal.Signals(-process.returncode).name if process.returncode < 0 else None,
                "result": None, "error": None,
                "stages": read_lines(directory / "stages.jsonl"),
                "release_events": read_lines(Path(str(trace) + events_suffix)),
            }
            for key, filename in (("result", "result.json"), ("error", "error.json")):
                path = directory / filename
                if path.exists():
                    row[key] = json.loads(path.read_text())
            complete = [event for event in row["release_events"] if event["stage"] == "complete"]
            if graph_disconnect:
                row["release_confirmed"] = bool(complete) and all(
                    event["disconnected_edges"] > 0
                    and event["disconnected_edges"] == event["input_edges"] + event["control_edges"]
                    for event in complete)
            elif artifacts:
                row["release_confirmed"] = any(
                    event["artifact"] == "factory" and event["objects"] > 0 for event in complete)
            else:
                row["release_confirmed"] = bool(complete) and all(
                    event["buffers"] > 0 and event["released_logical_bytes"] > 0 for event in complete)
            row["release_live_delta_bytes"] = sum(event.get("live_delta_bytes", 0) for event in complete)
            row["native_pass"] = (
                process.returncode == 0 and row["result"] is not None
                and row["result"]["status"] == "PASS"
                and row["result"]["finite_outputs_checked_each_iteration"]
                and row["result"]["precision"] == args.precision
                and row["result"]["effective_config"]["INFERENCE_PRECISION_HINT"]
                == {"FP32": "f32", "FP16": "f16", "BF16": "bf16"}[args.precision])
            weights_path = Path(str(trace) + ".mlas_weights.jsonl")
            row["mlas_request_count"] = None
            row["eligible_count"] = None
            if row["native_pass"]:
                required_trace = Path(str(trace) + events_suffix) if artifacts or graph_disconnect else weights_path
                if not required_trace.exists():
                    row["native_pass"] = False
                    row["validation_error"] = "Missing target trace; cannot establish release applicability"
                else:
                    row["mlas_request_count"] = len(read_lines(weights_path)) if weights_path.exists() else None
                    if graph_disconnect:
                        inventory = [event for event in row["release_events"]
                                     if event["stage"] in ("inventory", "complete")]
                        if len(inventory) != 1:
                            row["native_pass"] = False
                            row["validation_error"] = "Expected exactly one completed graph inventory"
                        else:
                            row["eligible_count"] = inventory[0]["input_edges"] + inventory[0]["control_edges"]
                    else:
                        row["eligible_count"] = sum(
                            event["eligible"] and event["artifact"] == "factory"
                            for event in row["release_events"]) if artifacts else row["mlas_request_count"]
            if graph_disconnect and release and row["native_pass"]:
                baseline_inventory = [event for event in cases[baseline_name]["release_events"]
                                      if event["stage"] == "inventory"]
                row["release_confirmed"] = row["release_confirmed"] and len(baseline_inventory) == 1 and all(
                    event[field] == baseline_inventory[0][field]
                    for event in complete for field in ("nodes", "input_edges", "control_edges"))
            if artifacts and release and row["native_pass"]:
                expected = {(event["node"], event["artifact"], event["objects"])
                            for event in cases[baseline_name]["release_events"] if event["eligible"]}
                released = {(event["node"], event["artifact"], event["objects"]) for event in complete}
                row["release_confirmed"] = row["release_confirmed"] and released == expected
            cases[name] = row
            save(args.output / "summary.json", {
                "status": "RUNNING", "precision": args.precision, "cases": cases})

    comparisons = {}
    passed = True
    applicable = False
    for suffix in ("fixed", "dynamic"):
        baseline = cases["baseline_" + suffix]
        candidate = cases["release_" + suffix]
        if candidate.get("status") == "NOT_APPLICABLE":
            comparisons[suffix] = {
                "status": "NOT_APPLICABLE", "reason": candidate["reason"],
                "baseline_native_pass": baseline["native_pass"], "release_confirmed": False,
            }
            passed = passed and baseline["native_pass"]
            continue
        applicable = True
        hashes_match = (
            baseline["native_pass"] and candidate["native_pass"]
            and baseline["result"]["sample_output_md5"] == candidate["result"]["sample_output_md5"])
        shapes_match = suffix != "dynamic" or (
            baseline["native_pass"] and candidate["native_pass"]
            and baseline["result"]["inference_shapes"] == candidate["result"]["inference_shapes"])
        stage_deltas = []
        baseline_stages = {(row["stage"], row["iteration"]): row for row in baseline["stages"]}
        for row in candidate["stages"]:
            key = row["stage"], row["iteration"]
            if key not in baseline_stages:
                continue
            before = baseline_stages[key]
            stage_deltas.append({
                "stage": key[0], "iteration": key[1],
                **{metric: (row[field] - before[field]) / 1024
                   for metric, field in (("rss_delta_mib", "Rss_kib"),
                                         ("pss_delta_mib", "Pss_kib"),
                                         ("uss_delta_mib", "uss_kib"))},
            })
            if "allocator_uordblks_bytes" in row and "allocator_uordblks_bytes" in before:
                stage_deltas[-1]["allocator_live_delta_mib"] = (
                    row["allocator_uordblks_bytes"] + row["allocator_hblkhd_bytes"]
                    - before["allocator_uordblks_bytes"] - before["allocator_hblkhd_bytes"]) / 1048576
        comparisons[suffix] = {
            "status": "PASS" if hashes_match and shapes_match and candidate["release_confirmed"] else "FAIL",
            "hashes_match": hashes_match, "shapes_match": shapes_match,
            "release_confirmed": candidate["release_confirmed"],
            "stage_deltas_release_minus_baseline": stage_deltas,
        }
        passed = passed and hashes_match and shapes_match and candidate["release_confirmed"]
    summary = {
        "status": ("PASS" if applicable else "NOT_APPLICABLE") if passed else "FAIL",
        "model": args.model, "precision": args.precision, "experiment": args.experiment,
        "warning": ("Destructive OV data/control-edge disconnection, including subgraphs; Constant buffers untouched. Shape inference may crash or return wrong outputs."
                    if graph_disconnect else "Only exact low-precision oneDNN FC executors owning their attributes are eligible."
                    if artifacts else "Destructive owner invalidation, not a supported optimization. Other consumers and export may break."),
        "limits": "One process per case with instrumentation; same-precision hashes, not cross-precision accuracy. No-MLAS baselines are explicitly not applicable. Not proof of general safety.",
        "cases": cases, "comparisons": comparisons,
    }
    save(args.output / "summary.json", summary)
    print(summary["status"], args.experiment, args.output, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
