#!/usr/bin/env python3
"""Summarize per-inference throughput from prediction-memory cases.json as CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


def load_cases(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases")
    if not isinstance(payload, list):
        raise ValueError("cases JSON must be a list or an object containing a cases list")
    return payload


def summarize(cases):
    groups = defaultdict(list)
    max_repeat = 0
    for case in cases:
        model = str(case.get("model", ""))
        kind = str(case.get("kind", case.get("backend", "")))
        precision = str(case.get("precision", ""))
        if not model or not kind or not precision:
            raise ValueError("Each case must include model, kind/backend, and precision")
        repeat = case.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"Invalid repeat number for {model}/{kind}/{precision}: {repeat}")
        max_repeat = max(max_repeat, repeat)
        groups[(model, kind, precision)].append(case)

    rows = []
    for (model, kind, precision), attempts in sorted(groups.items()):
        fps_by_repeat = {}
        latencies = []
        failed_runs = 0
        seen_repeats = set()
        for case in attempts:
            repeat = case.get("repeat", 1)
            if repeat in seen_repeats:
                raise ValueError(f"Duplicate repeat {repeat} for {model}/{kind}/{precision}")
            seen_repeats.add(repeat)
            result = case.get("result") or {}
            latency = result.get("mean_infer_ms")
            successful = (
                case.get("status") == "PASS"
                and result.get("status", "PASS") == "PASS"
                and isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                and math.isfinite(latency)
                and latency > 0
            )
            if successful:
                latencies.append(float(latency))
                fps_by_repeat[repeat] = 1000.0 / float(latency)
            else:
                failed_runs += 1

        fps_values = list(fps_by_repeat.values())
        successful_runs = len(fps_values)
        if successful_runs == 0:
            status = "FAIL"
        elif failed_runs:
            status = "PARTIAL"
        else:
            status = "PASS"
        rows.append({
            "model": model,
            "kind": kind,
            "precision": precision,
            "attempted_runs": len(attempts),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "median_mean_infer_ms": statistics.median(latencies) if latencies else "",
            "median_throughput_fps": statistics.median(fps_values) if fps_values else "",
            **{f"run_{repeat}_throughput_fps": fps_by_repeat.get(repeat, "")
               for repeat in range(1, max_repeat + 1)},
            "status": status,
        })
    return rows


def write_csv(output: Path, rows):
    if not rows:
        raise ValueError("throughput summary has no cases")
    fields = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=",", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Input cases.json")
    parser.add_argument("--output", type=Path, help="Output CSV (default: sibling throughput_summary.csv)")
    args = parser.parse_args()

    output = args.output or args.cases.with_name("throughput_summary.csv")
    rows = summarize(load_cases(args.cases))
    if not rows:
        parser.error("cases JSON contains no test cases")
    write_csv(output, rows)
    print(f"throughput_rows={len(rows)}")
    print(f"throughput_csv={output}")


if __name__ == "__main__":
    main()
