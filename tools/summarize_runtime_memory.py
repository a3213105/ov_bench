#!/usr/bin/env python3
"""Summarize opt-in, serialized CPU inference allocation diagnostics."""

import argparse
import csv
import json
from pathlib import Path


def write_table(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(path, expected_calls):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    totals = [row for row in rows if row["stage"] == "request_total"]
    if len(totals) != expected_calls:
        raise RuntimeError(f"{path}: expected {expected_calls} completed requests, got {len(totals)}")
    ids = sorted({row["inference_id"] for row in totals})
    if len(ids) != expected_calls:
        raise RuntimeError(f"{path}: duplicate inference IDs")
    order = {identifier: index for index, identifier in enumerate(ids, 1)}
    for row in rows:
        if row["inference_id"] not in order:
            raise RuntimeError(f"{path}: incomplete or nested request trace")
        row["iteration"] = order[row["inference_id"]]
    allocations = [row for row in rows if row["kind"] == "allocation"]
    groups = {}
    for row in allocations:
        key = (row["iteration"], row["depth"], row["type"], row["stage"])
        group = groups.setdefault(key, dict(zip(("iteration", "depth", "type", "stage"), key),
                                           count=0, live_delta_bytes=0, mmap_delta_bytes=0,
                                           free_delta_bytes=0))
        group["count"] += 1
        for field in ("live_delta_bytes", "mmap_delta_bytes", "free_delta_bytes"):
            group[field] += row[field]
    ranked = sorted(groups.values(), key=lambda row: (row["iteration"], -row["live_delta_bytes"]))
    phases = []
    for row in totals:
        identifier = row["inference_id"]
        request_phases = [item for item in allocations if item["inference_id"] == identifier
                          and item["stage"].startswith("request.")]
        graph_phases = [item for item in request_phases if item["stage"] == "request.graph_infer"]
        if len(graph_phases) != 1:
            raise RuntimeError(f"{path}: expected one graph-infer phase")
        graph = graph_phases[0]
        graph_children = [item for item in allocations if item["inference_id"] == identifier
                          and item["depth"] == graph["depth"] + 1]
        phases.append({
            "iteration": row["iteration"],
            "request_live_delta_bytes": row["live_delta_bytes"],
            "request_rss_delta_mib": (row["rss_after_kib"] - row["rss_before_kib"]) / 1024,
            "request_unattributed_bytes": row["live_delta_bytes"]
            - sum(item["live_delta_bytes"] for item in request_phases),
            "graph_live_delta_bytes": graph["live_delta_bytes"],
            "graph_unattributed_bytes": graph["live_delta_bytes"]
            - sum(item["live_delta_bytes"] for item in graph_children),
        })
    buffers = [row for row in rows if row["kind"] == "buffer"]
    inventory = []
    for iteration in range(1, expected_calls + 1):
        events = [row for row in buffers if row["iteration"] == iteration]
        weights = [row for row in events if row["stage"].startswith("dnnl.weights_")]
        # Address/size pairs describe observations, not retained allocation ownership.
        unique_weights = {(row["target_address"], row["target_bytes"]) for row in weights}
        unique_sources = {(row["source_address"], row["source_bytes"]) for row in weights}
        scratch = {(row["target_address"], row["target_bytes"]) for row in events
                   if row["stage"] == "dnnl.scratchpad_buffer"}
        inventory.append({
            "iteration": iteration, "weight_events": len(weights),
            "weight_created_events": sum(row["created"] for row in weights),
            "private_cache_hits": sum(row["stage"] == "dnnl.weights_private_hit" for row in weights),
            "observed_unique_source_bytes": sum(size for _, size in unique_sources),
            "observed_unique_weight_bytes": sum(size for _, size in unique_weights),
            "observed_unique_scratchpad_bytes": sum(size for _, size in scratch),
            "largest_scratchpad_bytes": max((size for _, size in scratch), default=0),
            "largest_shared_scratchpad_capacity_bytes": max(
                (row["target_bytes"] for row in events if row["stage"] == "dnnl.scratchpad_capacity"), default=0),
        })
    stem = str(path)
    write_table(Path(stem + ".by_stage.tsv"), ranked,
                ["iteration", "depth", "type", "stage", "count", "live_delta_bytes",
                 "mmap_delta_bytes", "free_delta_bytes"])
    write_table(Path(stem + ".intervals.tsv"), phases, list(phases[0]))
    write_table(Path(stem + ".buffers.tsv"), inventory, list(inventory[0]))
    top_nodes = sorted((row for row in allocations if row["node_id"]),
                       key=lambda row: row["live_delta_bytes"], reverse=True)[:100]
    columns = ["iteration", "graph_id", "inference_id", "depth", "node_id", "node", "type",
               "stage", "live_delta_bytes", "mmap_delta_bytes", "free_delta_bytes"]
    write_table(Path(stem + ".top_nodes.tsv"),
                [{key: row[key] for key in columns} for row in top_nodes], columns)
    summary = {
        "status": "PASS", "calls": expected_calls,
        "scope": "Diagnostic only: serialized shape/parameter updates, process-wide glibc deltas; no per-node RSS attribution.",
        "limits": "Nested rows are inclusive; sum only equal depth. Buffer address/size observations are not retained ownership or peak memory. Shared/overlapping spans and reused addresses may overcount.",
        "optimization_guidance": {
            "weights": "Multiple cached layouts or repeated creation are investigation candidates, not proof of waste; validate reuse and dynamic-shape needs before eviction.",
            "scratchpad": "Compare net allocation with observed shared addresses and maximum size; do not sum all per-node logical sizes as physical memory.",
            "execution": "First-only node.execute growth may be lazy JIT/cache allocation or page touching; correlate request RSS with allocator deltas.",
            "shape_buffers": "node.update_shapes growth may be essential outputs/intermediates; investigate capacity reuse and lifetime planning, not unconditional deletion.",
        },
        "intervals": phases, "buffers": inventory, "by_stage": ranked,
    }
    Path(stem + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calls", type=int, required=True)
    args = parser.parse_args()
    if args.calls <= 0:
        parser.error("--calls must be positive")
    result = summarize(args.trace, args.calls)
    print("Runtime memory summary:", args.trace, json.dumps(result["intervals"]))


if __name__ == "__main__":
    main()
