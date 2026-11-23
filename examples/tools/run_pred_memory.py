#!/usr/bin/env python3
"""Run isolated C++ prediction memory cases and summarize their stage snapshots."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from summarize_pred_throughput import summarize as summarize_throughput, write_csv as write_throughput_csv


MODELS = {
    "attention": "26Q2_round13_run2_e40_no_map_attn",
    "ped": "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine",
    "vehicle": "vehicle_q2_release_v0_w_leading_25",
}
CASES = (("ort", "FP32"), ("ov", "FP32"), ("ov", "BF16"), ("ov", "FP16"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(rows, name, iteration=None):
    matches = [r for r in rows if r["stage"] == name and
               (iteration is None or r["iteration"] == iteration)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name}/{iteration} stage, got {len(matches)}")
    return matches[0]


def summarize_compile_records(rows):
    if not rows or any(row.get("schema_version") != 1 for row in rows):
        raise ValueError("Missing or unsupported CPU compile ledger")
    stages = [row["stage"] for row in rows]
    required = ("C0_compile_entry", "C1_model_cloned", "C2_transformations_complete",
                "C3_compiled_model_constructed")
    for name in required:
        if stages.count(name) != 1:
            raise ValueError(f"Expected one {name} in isolated compile ledger")
    if [stages.index(name) for name in required] != sorted(stages.index(name) for name in required):
        raise ValueError("CPU compile checkpoints are out of order")
    graphs = {}
    for row in rows:
        if "graph_id" in row:
            graphs.setdefault(row["graph_id"], []).append(row["stage"])
    graph_stages = ["C4_graph_initialized", "C5_primitives_created", "C6_cleanup_complete"]
    complete = [names for names in graphs.values() if names == graph_stages]
    if not complete:
        raise ValueError("No complete initialized/activated/cleaned CPU graph")
    waterfall = []
    for previous, current in zip(rows, rows[1:]):
        row = {"from": previous["stage"], "to": current["stage"],
               "graph_id": current.get("graph_id"),
               "elapsed_ms": (current["timestamp_steady_ns"] -
                              previous["timestamp_steady_ns"]) / 1e6}
        if row["elapsed_ms"] < 0:
            raise ValueError("Nonmonotonic compile ledger timestamp")
        for key in ("Rss_kib", "Pss_kib", "Anonymous_kib", "Private_Dirty_kib",
                    "allocator_uordblks_bytes", "allocator_fordblks_bytes", "allocator_hblkhd_bytes"):
            if key in current and key in previous:
                row["delta_" + key] = current[key] - previous[key]
        waterfall.append(row)
    return {"records": rows, "waterfall": waterfall, "complete_graphs": len(complete),
            "accounted_fraction": None,
            "scope": "Process deltas, structural counts and deduplicated Constant logical spans; "
                     "not allocator capacity or complete object ownership accounting."}


def summarize(rows, result):
    total = result["iterations"]
    first = stage(rows, "inference", 1)
    final = stage(rows, "steady", total)
    stage(rows, "inference", total)
    inference = [r for r in rows if r["stage"] == "inference"]
    expected = sorted({1, total, *range(10, total + 1, 10)})
    if [r["iteration"] for r in inference] != expected:
        raise ValueError("Missing/duplicate/unordered inference checkpoints")
    baseline = stage(rows, "process_baseline")
    inputs = stage(rows, "inputs_loaded")
    warm_iteration = min(10, total)
    warm = stage(rows, "inference", warm_iteration)
    last_half = stage(rows, "inference", total // 2) if total % 20 == 0 else warm
    summary = {
        "backend": result["backend"], "precision": result["precision"],
        "iterations": total, "mean_infer_ms": result["mean_infer_ms"],
        "measured_seconds": final["elapsed_seconds"],
        "warm_iteration": warm_iteration, "tail_start_iteration": last_half["iteration"],
    }
    for metric in ("Rss", "Pss", "private", "Anonymous"):
        key = metric + "_kib"
        for label, row in (("baseline", baseline), ("inputs", inputs),
                           ("first", first), ("warm", warm), ("steady", final)):
            summary[f"{label}_{metric}_mib"] = row[key] / 1024
        for label, row in (("baseline", baseline), ("inputs", inputs),
                           ("first", first), ("warm", warm), ("tail", last_half)):
            summary[f"steady_minus_{label}_{metric}_mib"] = (final[key] - row[key]) / 1024
    summary["peak_hwm_mib"] = final["VmHWM_kib"] / 1024
    primary = rows[:rows.index(final) + 1]
    summary["peak_sampled_Rss_mib"] = max(r["Rss_kib"] for r in primary) / 1024
    summary["first_to_final_seconds"] = final["elapsed_seconds"] - first["elapsed_seconds"]
    summary["runtime_version"] = result.get(
        "openvino_version" if result["backend"] == "ov" else "onnxruntime_version", "unknown")
    summary["compile_ms"] = result.get("compile_ms")
    summary["first_infer_ms"] = first.get("last_infer_ms")
    if result["backend"] == "ov":
        loaded = stage(rows, "model_loaded")
        compiled = stage(rows, "model_compiled")
        for metric in ("Rss", "Pss", "Anonymous", "Private_Dirty"):
            key = metric + "_kib"
            if key in loaded and key in compiled:
                summary[f"compile_delta_{metric}_mib"] = (compiled[key] - loaded[key]) / 1024
        for metric in ("uordblks", "fordblks", "hblkhd"):
            key = "allocator_" + metric + "_bytes"
            if key in loaded and key in compiled:
                summary[f"compile_delta_{metric}_mib"] = (compiled[key] - loaded[key]) / (1024 * 1024)
    return summary


def report(output, cases, configuration):
    all_stages = []
    summaries = []
    failed = []
    compile_ledgers = []
    throughput_cases = []
    for short, backend, precision, directory in cases:
        result_path = directory / "result.json"
        if not result_path.exists():
            failed.append(f"{short}/{backend}/{precision}: see {directory.name}/run.log")
            throughput_cases.append({"model": short, "kind": backend, "precision": precision,
                                     "repeat": 1, "status": "ERROR", "result": {}})
            continue
        result = json.loads(result_path.read_text())
        throughput_cases.append({"model": short, "kind": backend, "precision": precision,
                                 "repeat": 1, "status": result.get("status"), "result": result})
        if result["status"] != "PASS":
            failed.append(f"{short}/{backend}/{precision}: {result}")
            continue
        rows = [json.loads(line) for line in (directory / "stages.jsonl").read_text().splitlines()]
        summary = summarize(rows, result)
        summary["model"] = short
        summary["runtime_precision_counts"] = result.get("runtime_precision_counts", {})
        summary["same_input_stability_checked"] = result.get("same_input_stability_checked", False)
        if backend == "ov":
            maps = (directory / "loaded_maps.txt").read_text()
            plugin = configuration.get("cpu_plugin")
            if plugin and plugin not in maps:
                failed.append(f"{short}/{precision}: requested CPU plugin absent from loaded maps")
            if configuration.get("compile_memory_stats"):
                try:
                    records = [json.loads(line) for line in
                               (directory / "compile_memory_stats.jsonl").read_text().splitlines()]
                    compile_ledgers.append({"model": short, "precision": precision,
                                            **summarize_compile_records(records)})
                except (OSError, ValueError, KeyError) as error:
                    failed.append(f"{short}/{precision}: invalid compile ledger: {error}")
        summaries.append(summary)
        for row in rows:
            all_stages.append({"model": short, "backend": backend, "precision": precision, **row})
    save(output / "summary.json", {"cases": summaries, "failed": failed})
    save(output / "compile_memory.json", {"cases": compile_ledgers})
    throughput_rows = summarize_throughput(throughput_cases)
    write_throughput_csv(output / "throughput_summary.csv", throughput_rows)
    for name, rows in (("stages.tsv", all_stages),
                       ("summary.tsv", [{k: v for k, v in r.items()
                                         if k != "runtime_precision_counts"} for r in summaries])):
        if rows:
            with (output / name).open("w", newline="") as stream:
                fields = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

    lines = [
        "# 本机 pred 三模型 C++ 内存增长实测",
        "",
        f"- 状态：{'全部完成' if not failed else '存在失败，见下文'}；每组 "
        f"{configuration['iterations']} 次（第一次包含在总次数内），"
        f"{configuration['threads']} 推理线程，1 轮；独立新进程串行运行。",
        "- CPU：" + configuration["cpu_model"] + "；进程可用 CPU：" +
        ",".join(str(c) for c in configuration["affinity"]) + "。",
        "- 实际框架版本：" + "; ".join(sorted({
            f"{row['backend']} {row['runtime_version']}" for row in summaries})) +
        "。库路径及 SHA256 见 configuration.json 和各组 loaded_maps.txt。",
        "- 所有实际推理均为 C++；Python 仅编排和统计。沿用当前 load_model、"
        "random_inputs、stable TopK 和默认 SDPA 策略；原始 ONNX 不修改、不 reshape。",
        "- batch=1，seed=43（按模型 ID 加盐），10 份预先生成的输入循环使用；"
        "四种后端/精度读取完全相同的二进制和 MD5 清单。所有输出每次检查有限性，"
        "本轮不做跨后端数值精度认证。",
        "- OV：LATENCY、NUM_STREAMS=1、单 InferRequest、CPU pinning NO、PERF_COUNT NO、"
        "无模型缓存；源 ov::Model 保留至 steady，匹配当前 C++ 使用方式。",
        "- ORT：CPUExecutionProvider、ORT_SEQUENTIAL、ORT_ENABLE_ALL、inter-op=1、"
        "CPU arena/memory pattern 默认开启；每次 Run 前释放上一份输出，不积累结果。",
        "- 不执行 malloc_trim，不调整 allocator、ISA 或系统缓存；继承本会话 CPU affinity，"
        "不更改其他任务。进程加载的两个框架共享库均计入各自 baseline，"
        "但 ORT 不加载 OV CPU 插件。主要比较也提供扣除 baseline/输入后的增量。",
        "- RSS/PSS/私有内存来自 /proc/self/smaps_rollup；1 MiB=1024 KiB。"
        "VmHWM 是内核粗略 RSS 计数的历史峰值，不是当前 RSS；本机该值低于精确"
        "smaps RSS，不能当作精确峰值。表中同时列出主要阶段 smaps RSS 的采样最大值"
        "（不保证捕获阶段内部瞬时峰值）。PSS 会受其他进程共享库驻留影响。",
        f"- 这是固定形状、固定大小输入池、固定请求数的 {configuration['iterations']} 次压力测试，"
        "不是按小时运行，也不覆盖变 batch、重复编译、多请求或所有业务输入。",
        "",
        "## 当前 RSS 及阶段增量（MiB）",
        "",
        "阶段数字是绝对当前 RSS。OV 的加载、编译可拆分；ORT 的 Session 创建"
        "同时加载/优化/初始化，不能强行拆成两个等价阶段。",
        "",
        f"| 模型 | 后端/精度 | baseline | 输入已加载 | 插件/Env | 模型加载 | 编译/ORT Session | 请求创建 | 首次推理 | {configuration['iterations']}次后 | 采样最大RSS | 内核粗略HWM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        directory = output / f"{row['model']}_{row['backend']}_{row['precision']}"
        stages = [json.loads(s) for s in (directory / "stages.jsonl").read_text().splitlines()]

        def rss(name):
            return f"{stage(stages, name)['Rss_kib'] / 1024:.2f}"

        is_ov = row["backend"] == "ov"
        lines.append(
            f"| {row['model']} | {row['backend'].upper()} {row['precision']} | "
            f"{row['baseline_Rss_mib']:.2f} | {row['inputs_Rss_mib']:.2f} | "
            f"{rss('plugin_loaded' if is_ov else 'ort_env_created')} | "
            f"{rss('model_loaded') if is_ov else '合并于Session'} | "
            f"{rss('model_compiled' if is_ov else 'session_created_load_and_optimize')} | "
            f"{rss('request_created') if is_ov else '不适用'} | "
            f"{row['first_Rss_mib']:.2f} | {row['steady_Rss_mib']:.2f} | "
            f"{row['peak_sampled_Rss_mib']:.2f} | {row['peak_hwm_mib']:.2f} |"
        )
    lines += [
        "", "## 推理阶段增长与稳定性（MiB）", "",
        "第10次已覆盖整个输入池；同时给出首次→最终、第10次→最终和后半程增量，"
        "以区分初次初始化/输入覆盖和持续增长。负数表示下降；单轮小幅变化不等于泄漏。",
        "",
        "| 模型 | 后端/精度 | 最终RSS−baseline | 最终RSS−输入基线 | 首次→最终RSS | 第10次→最终RSS | 后半程RSS增量 | 最终PSS | 首次→最终PSS | 最终私有内存 | 实测秒数 | 平均推理ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        values = [row[k] for k in (
            "steady_minus_baseline_Rss_mib", "steady_minus_inputs_Rss_mib",
            "steady_minus_first_Rss_mib", "steady_minus_warm_Rss_mib", "steady_minus_tail_Rss_mib",
            "steady_Pss_mib", "steady_minus_first_Pss_mib", "steady_private_mib",
            "measured_seconds", "mean_infer_ms")]
        lines.append(f"| {row['model']} | {row['backend'].upper()} {row['precision']} | " +
                     " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "## OV 相对 ONNX Runtime FP32 的最终内存差异", "",
              "绝对差额=OV最终−ORT最终；百分比=差额/ORT最终。扣基线差额比较两组"
              "（steady−process_baseline），不是权重文件大小或泄漏量。", "",
              "| 模型 | OV精度 | RSS差额MiB | RSS差额% | 扣baseline的RSS增长差额MiB | PSS差额MiB |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["backend"] != "ov":
            continue
        refs = [r for r in summaries if r["model"] == row["model"] and r["backend"] == "ort"]
        if not refs:
            continue
        ref = refs[0]
        delta = row["steady_Rss_mib"] - ref["steady_Rss_mib"]
        lines.append(f"| {row['model']} | {row['precision']} | {delta:+.2f} | "
                     f"{delta / ref['steady_Rss_mib'] * 100:+.2f}% | "
                     f"{row['steady_minus_baseline_Rss_mib'] - ref['steady_minus_baseline_Rss_mib']:+.2f} | "
                     f"{row['steady_Pss_mib'] - ref['steady_Pss_mib']:+.2f} |")
    lines += ["", "## 各 OV 阶段的 RSS 净增长（MiB）", "",
              "均相对紧邻阶段；首次推理相对 inputs_bound（不是相对编译结束）。", "",
              "| 模型 | 精度 | 加载−插件 | 编译−加载 | 请求−编译 | 绑定−请求 | 首次−绑定 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["backend"] != "ov":
            continue
        directory = output / f"{row['model']}_{row['backend']}_{row['precision']}"
        stages = [json.loads(s) for s in (directory / "stages.jsonl").read_text().splitlines()]
        points = [stage(stages, name)["Rss_kib"] for name in
                  ("plugin_loaded", "model_loaded", "model_compiled", "request_created", "inputs_bound")]
        points.append(stage(stages, "inference", 1)["Rss_kib"])
        deltas = [(b - a) / 1024 for a, b in zip(points, points[1:])]
        lines.append(f"| {row['model']} | {row['precision']} | " +
                     " | ".join(f"{v:+.2f}" for v in deltas) + " |")
    lines += ["", "## Runtime 精度证据", "",
              "precision hint 不代表整个图统一低精度；计数为 runtimePrecision 节点标签，"
              "不是参数字节数或算力比例。Runtime 图检查在 steady 之后执行，"
              "不污染主要内存阶段；after_runtime_inspection、source_model_released_diagnostic、"
              "backend_released 只属附加诊断，不用于主要比较。", ""]
    for row in summaries:
        if row["backend"] == "ov":
            lines.append(f"- {row['model']} {row['precision']}: "
                         f"`{json.dumps(row['runtime_precision_counts'], sort_keys=True)}`")
    lines += ["", "## Compile 测量与统计范围", "",
              "| 模型 | 后端/精度 | Compile/Session ms | 首次推理 ms | 同输入哈希检查 |",
              "|---|---|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['model']} | {row['backend']} {row['precision']} | "
                     f"{row['compile_ms']} | {row['first_infer_ms']} | "
                     f"{row['same_input_stability_checked']} |")
    lines += ["", "compile_memory.json 保存 C0-C6 原始采样、按真实执行顺序的增量和图 ID。"
              "C3 在 CompiledModel 构造返回后，因此一般晚于 C4/C5/C6。"
              "Constant 去重值是地址区间覆盖的逻辑字节数，不是 allocation capacity；"
              "结构计数不是对象持有字节，accounted_fraction 为 null，不声称完成 80% 归因。",
              "开启源码日志/哈希检查会引入观测开销。性能准入应在关闭这些选项时，"
              "用多轮独立进程与基线对照。uordblks 不含 mmap allocation，hblkhd 单独记录。",
              "本工具不替代精度、dynamic reshape、export/import、model cache、多 InferRequest "
              "及 ASan/UBSan 准入测试。", ""]
    if failed:
        lines += ["", "## 失败", "", *failed]
    lines += ["", "## 复现与归档", "",
              "- configuration.json：命令、CPU、库/插件/模型 SHA256、测试源码 SHA256。",
              "- inputs/inputs.json 与二进制：原 C++ 生成的10份输入和 MD5。",
              "- stages.tsv / summary.tsv / summary.json：完整阶段和每10次采样数据。",
              "- throughput_summary.csv：逗号分隔的模型/后端/精度吞吐汇总；FPS=1000/mean_infer_ms。",
              "- 每组 result.json、run.log、stages.jsonl、loaded_maps.txt；OV另有 runtime_graph.json。",
              "- source_snapshot/：本轮 C++ 和编排源码；pred_memory：本轮二进制。",
              "- runtime_lib/ 是本机库的 SONAME 链接，不复制/更换本机框架安装。", ""]
    (output / "report_zh.md").write_text("\n".join(lines))
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ort-library", type=Path, required=True)
    parser.add_argument("--cpu-plugin", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--compile-memory-stats", action="store_true")
    parser.add_argument("--verify-stability", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.threads, args.timeout_seconds) <= 0:
        parser.error("iterations, threads and timeout must be positive")
    if args.verify_stability and args.iterations < 20:
        parser.error("--verify-stability requires at least 20 iterations to repeat every input")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime_lib").mkdir()
    (output / "runtime_lib/libonnxruntime.so.1").symlink_to(args.ort_library.resolve())
    registry = ET.Element("ie")
    plugins = ET.SubElement(registry, "plugins")
    ET.SubElement(plugins, "plugin", name="CPU", location=str(args.cpu_plugin.resolve()))
    ET.ElementTree(registry).write(output / "cpu_plugins.xml", encoding="utf-8", xml_declaration=True)
    shutil.copy2(args.binary, output / "pred_memory")
    shutil.copytree(args.prepared, output / "inputs")
    source = Path(__file__).resolve().parent
    (output / "source_snapshot").mkdir()
    for name in ("CMakeLists.txt", "src", "include", "tests", "run_pred_memory.py",
                 "random_reference_worker.py"):
        path = source / name
        target = output / "source_snapshot" / name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    configuration = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "iterations": args.iterations, "threads": args.threads, "repeats": 1,
        "compile_memory_stats": args.compile_memory_stats,
        "verify_stability": args.verify_stability,
        "cpu_plugin": str(args.cpu_plugin.resolve()),
        "affinity": sorted(os.sched_getaffinity(0)), "uname": list(os.uname()),
        "cpu_model": next(line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "native_binary_sha256": sha256(output / "pred_memory"),
        "models": {name: {"id": model, "path": str((args.model_root / "pred_model" /
                    (model + ".onnx")).resolve()), "sha256": sha256(args.model_root / "pred_model" /
                    (model + ".onnx"))} for name, model in MODELS.items()},
        "libraries": {str(p.resolve()): sha256(p) for p in (args.ort_library, args.cpu_plugin)},
        "source_sha256": {str(p.relative_to(source)): sha256(p) for p in
                          (source / "src").glob("*.cpp")},
        "input_manifest_sha256": sha256(output / "inputs/inputs.json"),
        "allocator_environment": {key: value for key, value in os.environ.items() if
                                  key.startswith(("MALLOC_", "ONEDNN_", "DNNL_")) or key == "LD_PRELOAD"},
        "commands": [],
    }
    save(output / "configuration.json", configuration)
    (output / "state").write_text("RUNNING\n")
    environment = dict(os.environ)
    environment.pop("OV_CPU_COMPILE_MEMORY_STATS_PATH", None)
    environment.pop("OVBENCH_VERIFY_STABILITY", None)
    environment["OVBENCH_CPU_PLUGIN"] = str(args.cpu_plugin.resolve())
    environment["OVBENCH_CPU_PLUGIN_CONFIG"] = str(output / "cpu_plugins.xml")
    if args.verify_stability:
        environment["OVBENCH_VERIFY_STABILITY"] = "1"
    environment["LD_LIBRARY_PATH"] = str(output / "runtime_lib") + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    cases = []
    errors = []
    for short, model in MODELS.items():
        for backend, precision in CASES:
            directory = output / f"{short}_{backend}_{precision}"
            command = [str(output / "pred_memory"), str(args.model_root.resolve()),
                       str(output / "inputs/inputs.json"), model, backend, precision,
                       str(args.threads), str(args.iterations), str(directory)]
            cases.append((short, backend, precision, directory))
            configuration["commands"].append(command)
            save(output / "configuration.json", configuration)
            print(f"START {short} {backend} {precision}", flush=True)
            case_environment = dict(environment)
            if backend == "ov" and args.compile_memory_stats:
                case_environment["OV_CPU_COMPILE_MEMORY_STATS_PATH"] = str(
                    directory / "compile_memory_stats.jsonl")
            # The native executable creates its own exclusive case directory.
            log_path = output / f"{directory.name}.log"
            with log_path.open("w") as log:
                try:
                    completed = subprocess.run(command, env=case_environment, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
                    if completed.returncode:
                        errors.append(f"{directory.name}: exit {completed.returncode}")
                except subprocess.TimeoutExpired:
                    errors.append(f"{directory.name}: timeout {args.timeout_seconds}s")
            if directory.is_dir():
                log_path.rename(directory / "run.log")
            print(f"END {short} {backend} {precision}", flush=True)
    configuration["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    configuration["errors"] = errors
    save(output / "configuration.json", configuration)
    passed = report(output, cases, configuration) and not errors
    (output / "state").write_text("PASS\n" if passed else "ERROR\n")
    if not passed:
        raise RuntimeError(f"Memory run had failures: {errors}; see report_zh.md")
    print(f"PASS: {output / 'report_zh.md'}", flush=True)


if __name__ == "__main__":
    main()
