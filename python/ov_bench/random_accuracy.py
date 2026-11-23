"""Random-accuracy + tolerance-search, ported from ``src/random_accuracy.cpp``.

Generates a fresh shared random input set each iteration and compares every
low-precision InferRequest against a high-fidelity reference. The reference is
either an in-process OpenVINO FP32 session or an external worker
(``tools/random_reference_worker.py``) speaking line-delimited JSON over
stdin/stdout — the same protocol the C++ engine uses over a socketpair.
"""

import json
import os
import select
import tempfile
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from . import tensor_io
from .application import (apply_default_sdpa_bf16, load_model, model_path,
                          random_inputs)
from .session import Session, copy_outputs
from .tensor_io import compare_outputs, default_tolerance, load_tensor, save_tensor, tensor_info
from .utils import OVRuntimeError, absolute, fail, md5_file, read_json, tsv_json_value, tsv_row

_NATIVE_IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RandomAccuracyOptions:
    model_root: str
    output: str
    reference_root: str = ""
    reference_config: str = ""
    shapes: str = ""
    python: str = "python3"
    worker: str = ""
    precisions: list = field(default_factory=lambda: ["FP32"])
    iterations: int = 10
    seed: int = 43
    threads: int = 2
    timeout_seconds: int = 300
    threshold: float = 0.99
    atol: dict = field(default_factory=dict)
    rtol: dict = field(default_factory=dict)
    cpu_pinning: bool = None
    profiling: bool = False
    sdpa_bf16: bool = True
    tolerance_search: bool = False
    resample_common_cosine_failures: bool = False


class _Scratch:
    def __init__(self):
        root = Path(tempfile.gettempdir()) / ("ovbench-%d" % os.getuid())
        root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="random-accuracy-", dir=str(root)))

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)


class _ReferenceWorker:
    """Line-delimited JSON RPC bridge to random_reference_worker.py."""

    def __init__(self, options, config_path, directory, log_path):
        import subprocess
        self.timeout = options.timeout_seconds
        self._buffer = b""
        try:
            # On large hosts the worker's OpenVINO/OpenMP runtimes spin up
            # 2*core-count global arenas and can stall in futex queues during
            # dlopen/init, so READY may never arrive within the step timeout.
            # Cap its thread pools (inference itself runs with options.threads).
            cap = max(4, 2 * max(1, options.threads))
            environment = dict(os.environ)
            environment["TBB_MAX_THREAD_NUM"] = str(cap)
            environment["OMP_NUM_THREADS"] = str(cap)
            self._process = subprocess.Popen(
                [options.python, options.worker, "--config", str(config_path),
                 "--directory", str(directory)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=open(str(log_path), "wb"), env=environment)
        except Exception as error:
            raise OVRuntimeError("Cannot start reference worker: %s" % error)

    def _send_all(self, data):
        view = memoryview(data)
        while view:
            written = self._process.stdin.write(view)
            view = view[written:]
        self._process.stdin.flush()

    def _read_available(self):
        while True:
            remaining = max(0, int((self._deadline - time.monotonic()) * 1000))
            if remaining <= 0:
                raise OVRuntimeError("Reference worker timed out; see reference_worker.log")
            ready, _ready, _err = select.select([self._process.stdout], [], [],
                                                min(remaining, 1000) / 1000.0)
            if not ready:
                continue
            chunk = os.read(self._process.stdout.fileno(), 4096)
            if not chunk:
                raise OVRuntimeError("Reference worker exited without a response; "
                                     "see reference_worker.log")
            self._buffer += chunk
            if len(self._buffer) > 16 * 1024 * 1024:
                raise OVRuntimeError("Reference worker response exceeds 16 MiB")

    def send(self, message):
        try:
            self._send_all((json.dumps(message) + "\n").encode("utf-8"))
        except Exception:
            raise OVRuntimeError("Reference worker send failed; see reference_worker.log")

    def receive(self):
        self._deadline = time.monotonic() + self.timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline != -1:
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1:]
                result = json.loads(line.decode("utf-8"))
                if result.get("status") == "error":
                    raise OVRuntimeError("Reference worker: " + result["error"])
                return result
            self._read_available()

    def finish(self):
        self.send({"command": "close"})
        try:
            self._process.stdin.close()
        except Exception:
            pass
        self._process.wait(timeout=self.timeout)
        if self._process.returncode != 0:
            raise OVRuntimeError("Reference worker exited unsuccessfully; "
                                 "see reference_worker.log")


def _port_name(port, index, prefix):
    names = list(port.get_names())
    return prefix + str(index) if not names else min(names)


def _summary(path, cases):
    header = ["model", "precision", "sdpa_bf16_nodes", "images", "mean_ms", "p99_ms",
              "throughput_fps", "accuracy_passed", "md5_consistent", "stable_outputs_finite",
              "status"]
    lines = [tsv_row(header)]
    for case in cases:
        lines.append(tsv_row([case["model"], case["precision"],
                              tsv_json_value(case["sdpa_bf16_nodes"]),
                              tsv_json_value(case["image_count"]), "", "", "",
                              tsv_json_value(case["accuracy"]["passed_images"]), "", "",
                              case["status"]]))
    with open(path, "w") as handle:
        handle.write("".join(lines))


def _output_summary(path, cases):
    header = ["model", "precision", "output", "iterations", "passed", "mean_cosine",
              "min_cosine", "max_abs", "atol", "rtol", "out_of_tolerance_elements",
              "elements", "out_of_tolerance_fraction", "status", "abs_error_variance"]
    lines = [tsv_row(header)]
    for case in cases:
        totals = {}
        for iteration in case["accuracy"]["records"]:
            for output in iteration["outputs"]:
                t = totals.setdefault(output["name"], {"count": 0, "passed": 0,
                    "cosine_count": 0, "error_count": 0, "outside": 0, "elements": 0,
                    "atol": 0, "rtol": 0, "sum": 0.0, "minimum": 1.0, "maximum": 0.0,
                    "error_elements": 0, "error_mean": 0.0, "error_m2": 0.0})
                t["count"] += 1
                t["passed"] += 1 if output["status"] == "PASS" else 0
                t["outside"] += output["out_of_tolerance_elements"]
                t["elements"] += output["elements"]
                t["atol"] = output["atol"]
                t["rtol"] = output["rtol"]
                if output["cosine_similarity"] is not None:
                    cosine = output["cosine_similarity"]
                    t["cosine_count"] += 1
                    t["sum"] += cosine
                    t["minimum"] = min(t["minimum"], cosine)
                if output["max_abs"] is not None:
                    t["error_count"] += 1
                    t["maximum"] = max(t["maximum"], output["max_abs"])
                    number = output["elements"]
                    mean = output["mean_abs"]
                    variance = output["abs_error_variance"]
                    delta = mean - t["error_mean"]
                    combined = t["error_elements"] + number
                    t["error_m2"] += variance * number + delta * delta * t["error_elements"] \
                        * number / combined
                    t["error_mean"] += delta * number / combined
                    t["error_elements"] = combined
        for name, t in totals.items():
            lines.append(tsv_row([
                case["model"], case["precision"], name, str(t["count"]), str(t["passed"]),
                (t["sum"] / t["cosine_count"]) if t["cosine_count"] else "",
                t["minimum"] if t["cosine_count"] else "",
                t["maximum"] if t["error_count"] else "",
                t["atol"], t["rtol"], str(t["outside"]), str(t["elements"]),
                (t["outside"] / t["elements"]) if t["elements"] else 0.0,
                "PASS" if t["passed"] == t["count"] else "FAIL",
                (max(0.0, t["error_m2"] / t["error_elements"])
                 if t["error_count"] == t["count"] and t["error_elements"] else "")]))
    with open(path, "w") as handle:
        handle.write("".join(lines))


def _precision_metrics(path, cases):
    header = ["model", "precision", "iteration", "output", "cosine", "max_abs",
              "normalized_rmse", "mismatched_elements", "status"]
    lines = [tsv_row(header)]
    for case in cases:
        for iteration in case["accuracy"]["records"]:
            for output in iteration["outputs"]:
                lines.append(tsv_row([case["model"], case["precision"],
                                      tsv_json_value(iteration["iteration"]), output["name"],
                                      tsv_json_value(output["cosine_similarity"]),
                                      tsv_json_value(output["max_abs"]),
                                      tsv_json_value(output["normalized_rmse"]),
                                      tsv_json_value(output["mismatched_elements"]),
                                      output["status"]]))
    with open(path, "w") as handle:
        handle.write("".join(lines))


def _seed_generator(seed, model_id):
    entropy = seed.to_bytes(8, "little", signed=False) + model_id.encode("utf-8")
    digest = int.from_bytes(__import__("hashlib").sha256(entropy).digest()[:8], "little")
    return np.random.default_rng(digest)


def run_random_accuracy(specs, o):
    if o.resample_common_cosine_failures and (o.tolerance_search or
                                              "BF16" not in o.precisions or
                                              "FP16" not in o.precisions):
        fail("Resampling requires random-accuracy with both BF16 and FP16")
    if o.tolerance_search and (o.precisions != ["FP32"] or o.atol or o.rtol):
        fail("Tolerance search requires FP32 only with default base tolerances")
    if os.path.exists(o.output):
        raise OVRuntimeError("Output directory already exists: " + o.output)
    os.makedirs(o.output)

    report = {"schema_version": 2, "status": "RUNNING", "mode": "random-accuracy",
              "seed": o.seed, "iterations": o.iterations, "threads": o.threads,
              "threshold": o.threshold, "cases": [],
              "openvino": {"build": str(getattr(ov, "__version__", "unknown"))},
              "input_distribution": "FP32 normal(0,0.1); boolean Bernoulli(0.75); "
                                    "prediction mask groups nondegenerate",
              "execution": "All selected InferRequests coexist; fresh shared input each "
                           "iteration; sequential inference",
              "pass_criterion": "Every output must pass tolerance and cosine checks"}
    if o.resample_common_cosine_failures:
        report["sampling"] = {"enabled": True, "models": {}, "criterion":
                              "Accept if all outputs pass cosine checks in BF16 OR FP16",
                              "selection_bias": "Conditional random sample",
                              "max_attempts": None}
    _write_report(o.output, report)
    if o.tolerance_search:
        report["mode"] = "tolerance-search"
    try:
        configs = read_json(o.reference_config) if o.reference_config else {}
        shape_configs = read_json(o.shapes) if o.shapes else {}
        for spec in specs:
            _run_single_model(spec, o, report, configs, shape_configs)
        all_passed = all(case["status"] == "PASS" for case in report["cases"])
        report["status"] = "PASS" if all_passed else "FAIL"
        _write_report(o.output, report)
        return 0 if all_passed else 1
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        _write_report(o.output, report)
        raise


def _write_report(output, report):
    cleaned = _clean(report)
    path = os.path.join(output, "results.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(cleaned, handle, indent=2)
        handle.write("\n")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_clean(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _run_single_model(spec, o, report, configs, shape_configs):
    native_root = o.model_root
    if spec.family == "prediction" and not os.path.exists(model_path(spec, native_root)):
        native_root = o.reference_root
    model = load_model(spec, native_root)
    folder = os.path.join(o.output, safe_id(spec.id))
    os.makedirs(folder)
    rejected_handle = None
    if o.resample_common_cosine_failures:
        os.makedirs(os.path.join(folder, "rejected_inputs"), exist_ok=True)
        rejected_handle = open(os.path.join(folder, "rejected_inputs.jsonl"), "w")
        report["sampling"]["models"][spec.id] = {"attempted": 0, "accepted": 0,
                                                 "rejected": 0,
                                                 "rejected_log":
                                                 (safe_id(spec.id) + "/rejected_inputs.jsonl")}
    scratch = _Scratch()
    try:
        return _run_model_body(spec, o, report, model, folder, scratch, configs, shape_configs,
                               rejected_handle)
    finally:
        if rejected_handle:
            rejected_handle.close()
        scratch.cleanup()


def safe_id(model_id):
    return model_id.replace(":", "_")


def _run_model_body(spec, o, report, model, folder, scratch, configs, shape_configs,
                    rejected_handle):
    if o.tolerance_search:
        base_atol, base_rtol = 1e-5, 1e-4
    input_names, output_names = [], []
    for i in range(len(model.inputs)):
        input_names.append(_port_name(model.input(i), i, "input_"))
    for i in range(len(model.outputs)):
        output_names.append(spec.output_names[i] if i < len(spec.output_names)
                            else "output_%d" % i)
    config = {"model": spec.id, "family": spec.family, "artifact": spec.artifact,
              "model_root": absolute(native_root_of(o, spec)), "reference_root":
              absolute(o.reference_root), "threads": o.threads, "input_names": input_names,
              "output_names": output_names,
              "reference_config": configs.get(spec.id, {})}
    with open(os.path.join(folder, "reference_config.json"), "w") as handle:
        json.dump(_clean(config), handle, indent=2)
        handle.write("\n")
    reference = _ReferenceWorker(o, Path(os.path.join(folder, "reference_config.json")),
                                 scratch.path, Path(os.path.join(folder, "reference_worker.log")))
    ready = reference.receive()
    if ready.get("status") != "ready":
        raise OVRuntimeError("Reference worker did not report ready")
    ov_reference = ready.get("backend") == "OpenVINO FP32"
    shape_overrides = shape_configs.get(spec.id, {})
    generator = _seed_generator(o.seed, spec.id)

    sessions = {}
    patched = {}

    def compile(precision):
        graph = model.clone()
        patched[precision] = apply_default_sdpa_bf16(spec, precision, graph, o.sdpa_bf16)
        sessions[precision] = Session(graph, precision, o.threads, o.cpu_pinning, o.profiling)

    for precision in o.precisions:
        compile(precision)
    if ov_reference and "FP32" not in sessions:
        compile("FP32")
    resident = list(o.precisions)
    if ov_reference and "FP32" not in resident:
        resident.append("FP32")

    case_indices = {}
    for precision in o.precisions:
        tolerance = default_tolerance(precision)
        if precision in o.atol:
            tolerance["atol"] = o.atol[precision]
        if precision in o.rtol:
            tolerance["rtol"] = o.rtol[precision]
        case_indices[precision] = len(report["cases"])
        effective = {}
        for key in ("INFERENCE_NUM_THREADS", "NUM_STREAMS", "INFERENCE_PRECISION_HINT",
                    "ENABLE_CPU_PINNING"):
            try:
                effective[key] = sessions[precision].compiled.get_property(key).as_string()
            except Exception:
                effective[key] = ""
        ports = [{"name": output_names[i], "port": i,
                  "names": sorted(list(sessions[precision].compiled.output(i).get_names()))}
                 for i in range(len(output_names))]
        report["cases"].append({"model": spec.id, "family": spec.family,
                                "precision": precision, "sdpa_bf16_nodes": patched[precision],
                                "image_count": o.iterations, "reference": ready, "seed": o.seed,
                                "effective_config": effective,
                                "source": {"path": absolute(str(model_path(spec, native_root_of(o, spec)))),
                                           "md5": md5_file(str(model_path(spec, native_root_of(o, spec))))},
                                "resident_precisions": resident, "outputs": ports,
                                "status": "RUNNING",
                                "accuracy": {"images": o.iterations, "passed_images": 0,
                                             "atol": tolerance["atol"], "rtol": tolerance["rtol"],
                                             "records": []}})
        os.makedirs(os.path.join(folder, precision))
    _write_report(o.output, report)

    search_blocked = False
    required_scale = 0.0
    limiting_output = None
    accepted = 0
    iteration = 0
    while accepted < o.iterations:
        inputs = random_inputs(spec, model, generator, shape_overrides)
        input_descriptors = []
        temporary = []
        for i in range(len(inputs)):
            path = scratch.path / ("input_%d_%d.bin" % (iteration, i))
            descriptor = save_tensor(inputs[i][1], path)
            descriptor["name"] = inputs[i][0]
            descriptor["file"] = str(path)
            input_descriptors.append(descriptor)
            temporary.append(path)

        def infer(precision):
            session = sessions[precision]
            for name, tensor in inputs:
                session.request().set_tensor(name, tensor)
            session.request().infer()
            return copy_outputs(session.request())

        if ov_reference:
            baseline = infer("FP32")
        else:
            reference.send({"command": "infer", "iteration": iteration,
                            "inputs": input_descriptors})
            response = reference.receive()
            if response.get("status") != "ok" or len(response["outputs"]) != len(output_names):
                raise OVRuntimeError("Invalid reference output response")
            baseline = []
            for i in range(len(output_names)):
                descriptor = response["outputs"][i]
                if descriptor["name"] != output_names[i]:
                    raise OVRemoteError()
                baseline.append(load_tensor(descriptor, scratch.path))
                temporary.append(descriptor["file"])

        candidate_records = {}
        low_precision_cosine_passed = False
        for precision in o.precisions:
            accuracy_config = report["cases"][case_indices[precision]]["accuracy"]
            tolerance = {"atol": accuracy_config["atol"], "rtol": accuracy_config["rtol"]}
            if ov_reference and precision == "FP32":
                actual = baseline
            else:
                actual = infer(precision)
            if len(actual) != len(baseline) or len(actual) != len(output_names):
                raise OVRuntimeError("Output count mismatch")
            rows = []
            passed = True
            for i in range(len(actual)):
                row = compare_outputs(baseline[i], actual[i], o.threshold,
                                      tolerance["atol"], tolerance["rtol"])
                row["name"] = output_names[i]
                row["port"] = i
                row["reference_md5"] = tensor_info(baseline[i])["md5"]
                row["md5"] = tensor_info(actual[i])["md5"]
                passed = passed and row["status"] == "PASS"
                if o.tolerance_search:
                    real = baseline[i].element_type.get_type_name() in tensor_io.REAL_DTYPES
                    if not row["finite"] or (real and not row["cosine_passed"]) or \
                            (not real and not row["exact_equal"]):
                        if not search_blocked:
                            limiting_output = {"iteration": iteration, "name": output_names[i],
                                               "reason": "Nonfinite, cosine, or exact integer check failed"}
                        search_blocked = True
                    elif row["required_tolerance_scale"] is None:
                        raise OVRuntimeError("No finite common tolerance scale for output")
                    elif row["required_tolerance_scale"] > required_scale:
                        required_scale = row["required_tolerance_scale"]
                        if not search_blocked:
                            limiting_output = {"iteration": iteration, "name": output_names[i]}
                rows.append(row)
            for i in range(len(inputs)):
                if tensor_info(inputs[i][1])["md5"] != input_descriptors[i]["md5"]:
                    raise OVRuntimeError("Input modified during inference")
            recorded_inputs = [dict(d) for d in input_descriptors]
            for descriptor in recorded_inputs:
                descriptor.pop("file", None)
            if precision in ("BF16", "FP16"):
                low_precision_cosine_passed |= all(r["cosine_passed"] for r in rows)
            candidate_records[precision] = {"iteration": iteration,
                                            "image": "random_%d" % iteration,
                                            "inputs": recorded_inputs, "outputs": rows,
                                            "status": "PASS" if passed else "FAIL"}
        valid = not o.resample_common_cosine_failures or low_precision_cosine_passed
        if not valid:
            rejected_folder = os.path.join(folder, "rejected_inputs", "random_%d" % iteration)
            os.makedirs(rejected_folder)
            saved_inputs = [dict(d) for d in input_descriptors]
            for i in range(len(inputs)):
                dest = os.path.join(rejected_folder, "input_%d.bin" % i)
                shutil.copyfile(temporary[i], dest)
                saved_inputs[i]["file"] = os.path.relpath(dest, folder)
            rejected_handle.write(json.dumps(_clean({"model": spec.id, "iteration": iteration,
                "seed": o.seed, "threshold": o.threshold, "reference": ready,
                "inputs": saved_inputs, "reason": "Both BF16 and FP16 failed cosine",
                "precisions": candidate_records})) + "\n")
            rejected_handle.flush()
        else:
            for precision in o.precisions:
                record = candidate_records[precision]
                if o.resample_common_cosine_failures:
                    record["accepted_index"] = accepted
                passed = record["status"] == "PASS"
                accuracy = report["cases"][case_indices[precision]]["accuracy"]
                if passed:
                    accuracy["passed_images"] += 1
                accuracy["records"].append(record)
                report["cases"][case_indices[precision]]["status"] = \
                    "PASS" if accuracy["passed_images"] == accepted + 1 else "FAIL"
                if not o.tolerance_search:
                    all_passed_local = passed
            accepted += 1
        if o.resample_common_cosine_failures:
            sampling = report["sampling"]["models"][spec.id]
            sampling["attempted"] = iteration + 1
            sampling["accepted"] = accepted
            sampling["rejected"] = iteration + 1 - accepted
        for precision in o.precisions:
            with open(os.path.join(folder, precision, "result.json"), "w") as handle:
                json.dump(_clean(report["cases"][case_indices[precision]]), handle, indent=2)
                handle.write("\n")
        for path in temporary:
            try:
                os.remove(path)
            except OSError:
                pass
        _write_report(o.output, report)
        _summary(os.path.join(o.output, "summary.tsv"), report["cases"])
        _output_summary(os.path.join(o.output, "outputs.tsv"), report["cases"])
        _precision_metrics(os.path.join(o.output, "precision_metrics.tsv"), report["cases"])
        if search_blocked:
            break
        iteration += 1
    reference.finish()
    if o.tolerance_search:
        _finalize_tolerance_search(spec, o, report, case_indices, folder, required_scale,
                                   limiting_output, search_blocked)


class _OVRemoteError(Exception):
    pass


def OVRemoteError():
    return _OVRemoteError("Reference output name mismatch")


def native_root_of(o, spec):
    root = o.model_root
    if spec.family == "prediction" and not os.path.exists(model_path(spec, root)):
        return o.reference_root
    return root


def _outward_product(base, scale):
    exact = float(base) * float(scale)
    return float(np.nextafter(np.float64(exact), np.inf))


def _finalize_tolerance_search(spec, o, report, case_indices, folder, required_scale,
                               limiting_output, search_blocked):
    import math
    case = report["cases"][case_indices["FP32"]]
    case["tolerance_search"] = {"status": "FAIL", "base_atol": 1e-5, "base_rtol": 1e-4,
                                "minimum_scale_allowed": 1, "required_scale": required_scale,
                                "limiting_output": limiting_output, "scale": None,
                                "atol": None, "rtol": None}
    search = case["tolerance_search"]
    if search_blocked:
        search["reason"] = "Cosine/nonfinite/integer failure; tolerance relaxation cannot fix"
        case["status"] = "FAIL"
        return
    low = 1.0
    high = 1.0
    trials = []
    while high < required_scale:
        trials.append({"scale": high, "passed": False, "stage": "expand"})
        low = high
        if high > float(np.finfo(np.float64).max) / 2:
            raise OVRuntimeError("Tolerance search scale overflow")
        high *= 2
    trials.append({"scale": high, "passed": True, "stage": "bracket"})
    while high - low > 1e-8 * max(1.0, high):
        mid = low + (high - low) / 2
        passed = mid >= required_scale
        trials.append({"scale": mid, "passed": passed, "stage": "bisect"})
        if passed:
            high = mid
        else:
            low = mid
    atol = _outward_product(1e-5, high)
    rtol = _outward_product(1e-4, high)
    search.update({"scale": high, "atol": atol, "rtol": rtol, "lower_scale": low,
                   "relative_search_resolution": 1e-8, "trials": trials,
                   "method": "Exact per-element max(error/(base_atol+base_rtol*abs(reference))); "
                             "expansion/bisection of this monotone predicate"})
    case["calibration_accuracy"] = case["accuracy"]
    _write_report(o.output, report)
    verification = RandomAccuracyOptions(
        model_root=o.model_root, output=os.path.join(folder, "verification"),
        reference_root=o.reference_root, reference_config=o.reference_config,
        shapes=o.shapes, python=o.python, worker=o.worker, precisions=["FP32"],
        iterations=o.iterations, seed=o.seed, threads=o.threads,
        timeout_seconds=o.timeout_seconds, threshold=o.threshold, atol={"FP32": atol},
        rtol={"FP32": rtol}, cpu_pinning=o.cpu_pinning, profiling=o.profiling,
        sdpa_bf16=o.sdpa_bf16, tolerance_search=False)
    code = run_random_accuracy([spec], verification)
    verified = read_json(os.path.join(verification.output, "results.json"))
    verified_case = verified["cases"][0]
    original = case["calibration_accuracy"]["records"]
    final_records = verified_case["accuracy"]["records"]
    if len(original) != len(final_records):
        raise OVRuntimeError("Verification input count mismatch")
    for i in range(len(original)):
        if original[i]["inputs"] != final_records[i]["inputs"]:
            raise OVRuntimeError("Verification random input hashes differ from calibration")
    case["accuracy"] = verified_case["accuracy"]
    case["status"] = "PASS" if code == 0 else "FAIL"
    search["status"] = case["status"]
    search["verification_report"] = str(os.path.join(verification.output, "results.json"))
    search["verification_input_hashes_equal"] = True
    if code != 0:
        search["reason"] = "Fresh verification failed; no verified minimum claimed"
    with open(os.path.join(folder, "FP32", "result.json"), "w") as handle:
        json.dump(_clean(case), handle, indent=2)
        handle.write("\n")
    _write_report(o.output, report)
    _summary(os.path.join(o.output, "summary.tsv"), report["cases"])
    _output_summary(os.path.join(o.output, "outputs.tsv"), report["cases"])
    _precision_metrics(os.path.join(o.output, "precision_metrics.tsv"), report["cases"])
