"""Benchmark engine: option parsing + execution of every mode.

Python port of ``src/main.cpp`` (``ov_bench``). Produces the same
``results.json`` / ``summary.tsv`` layout, per-model directories and side files
so the shell entrypoints under ``tools/`` can drive the Python backend in place
of the native binary.
"""

import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from . import tensor_io
from .application import (ModelSpec, apply_default_sdpa_bf16, find_model, load_model,
                          model_path, model_specs, random_inputs, validate_input,
                          validate_inputs)
from .image import preprocess_image
from . import session as _session
from .session import Session, copy_outputs
from .stability import compare_stability_directories, process_identity, stability_filename
from .utils import (OVRuntimeError, OVBError, absolute, fail, md5_file, read_json,
                    tsv_json_value, tsv_row, write_json)

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _write_json(path, value):
    write_json(str(path), _clean(value))


def image_files(source):
    supported = {".jpg", ".jpeg", ".png"}
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in supported:
            fail("Supported image formats: JPEG, PNG")
        return [source]
    files = []
    for name in sorted(source.iterdir()):
        if name.is_file() and name.suffix.lower() in supported:
            files.append(name)
    if not files:
        raise OVRuntimeError("No JPEG/PNG images found")
    return files


# ---------------------------------------------------------------------------
# Options / parsing
# ---------------------------------------------------------------------------

@dataclass
class Options:
    model_root: str = ""
    images: str = ""
    manifest: str = ""
    output: str = ""
    start_signal: str = ""
    model_file: str = ""
    input_shapes: dict = field(default_factory=dict)
    random_samples: int = 10
    input_count: int = 10
    input_count_explicit: bool = False
    reference_root: str = ""
    reference_config: str = ""
    random_shapes: str = ""
    reference_python: str = "python3"
    seed: int = 43
    model: str = "all"
    precision: str = "FP32,BF16,FP16"
    mode: str = "performance"
    batch: int = 0
    batches: list = field(default_factory=lambda: [0])
    batch_explicit: bool = False
    warmup: int = 3
    iterations: int = 200
    rounds: int = 10
    iterations_explicit: bool = False
    duration_seconds: float = None
    step_timeout_seconds: int = 300
    threads: int = 6
    threshold: float = 0.99
    atol: dict = field(default_factory=dict)
    rtol: dict = field(default_factory=dict)
    stability_dirs: list = field(default_factory=list)
    cpu_plugin: str = ""
    prepared: bool = False
    sdpa_bf16: bool = True
    save_performance_outputs: bool = True
    resample_common_cosine_failures: bool = False
    cpu_pinning: bool = None
    hyper_threading: bool = None
    profiling: bool = True


_HEADERS = (
    "ov_bench --model-root DIR --output NEW_DIR [options]\n"
    "  --model all|MODEL_ID[,MODEL_ID]  --precisions FP32,BF16,FP16\n"
    "  --model-file FILE               Test an unregistered ONNX/XML model with native random inputs\n"
    "  --input-shape NAME=DIM[,DIM...]  Override one input shape; repeat per input\n"
    "  --samples 10 --seed 43          Generic random dataset size and seed\n"
    "  --input-count 10                Prediction random input tensor count\n"
    "  --images FILE_OR_DIR            Native JPEG/PNG preprocessing\n"
    "  --manifest FILE --prepared-inputs  Raw input/FP32-reference dataset\n"
    "  --mode performance|accuracy|random-accuracy|stability|all|preprocess|single\n"
    "  --batch 0|N[,0|N...]            Dynamic batch (0) or static batch N; default: 0\n"
    "  --mode self-test                 Run native regression tests; no model/output required\n"
    "  --mode gather-guard [--cpu-plugin FILE]  Verify Gather BF16 output and memory guards\n"
    "  --mode tolerance-search --precisions FP32  Search fixed-ratio FP32 tolerances, then verify\n"
    "  --mode compare-stability --stability-dir DIR [--stability-dir DIR ...] --output NEW.json\n"
    "  --threads 6 --warmup 3 --iterations 200 --rounds 10\n"
    "  --duration SECONDS             Timed inference duration instead of --iterations (exclusive)\n"
    "  --step-timeout-seconds 300       Per-stage timeout in single mode\n"
    "  --threshold 0.99               Cosine threshold; ViT FP16 SDPA BF16 rewrite\n"
    "  --no-sdpa-bf16                 Disable the FP16 SDPA rewrite flag\n"
    "  --start-signal FILE            Wait after warmup for multi-process release\n"
    "  --no-performance-output-files  Keep output checks without saving check_*.bin\n"
    "  --cpu-pinning YES|NO           Override OpenVINO internal pinning\n"
    "  --hyper-threading YES|NO       Override ENABLE_HYPER_THREADING\n"
    "  --perf-count YES|NO            Enable or disable OpenVINO profiling\n"
)


def _positive_integer(value, zero=False):
    if not value or any(c not in "0123456789" for c in value):
        fail("Expected nonnegative integer, got: " + value)
    number = int(value)
    if ((not zero and number == 0) or number > 1000000000):
        fail("Integer out of range: " + value)
    return number


def _split(value):
    items = value.split(",")
    if not items or value.endswith(","):
        fail("Invalid CSV: " + value)
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            fail("Empty or duplicate CSV item: " + value)
        seen.add(item)
        result.append(item)
    return result


def _parse_yes_no(value, key):
    if value not in ("YES", "NO"):
        fail(key + " must be YES or NO")
    return value == "YES"


def _parse_tolerances(key, value, target):
    for item in _split(value):
        separator = item.find("=")
        if separator <= 0 or separator + 1 == len(item):
            fail(key + " requires PRECISION=VALUE[,PRECISION=VALUE]")
        precision = item[:separator]
        if precision not in ("FP32", "BF16", "FP16"):
            fail("Unsupported tolerance precision: " + precision)
        text = item[separator + 1:]
        try:
            tolerance = float(text)
        except ValueError:
            fail(key + " values must be finite and nonnegative")
        if not math.isfinite(tolerance) or tolerance < 0:
            fail(key + " values must be finite and nonnegative")
        if precision in target:
            fail("Duplicate " + key + " precision: " + precision)
        target[precision] = tolerance


def parse(argv):
    o = Options()
    index = 1
    while index < len(argv):
        key = argv[index]
        if key == "--help":
            print(_HEADERS)
            return "help"
        if key == "--prepared-inputs":
            o.prepared = True
            index += 1
            continue
        if key == "--no-performance-output-files":
            o.save_performance_outputs = False
            index += 1
            continue
        if key == "--no-sdpa-bf16":
            o.sdpa_bf16 = False
            index += 1
            continue
        if key == "--resample-common-cosine-failures":
            o.resample_common_cosine_failures = True
            index += 1
            continue
        if index + 1 >= len(argv):
            fail("Missing value for " + key)
        value = argv[index + 1]
        index += 2
        if key == "--model-root":
            o.model_root = value
        elif key == "--model-file":
            o.model_file = value
        elif key == "--samples":
            o.random_samples = _positive_integer(value)
        elif key == "--input-count":
            o.input_count = _positive_integer(value)
            o.input_count_explicit = True
        elif key == "--input-shape":
            separator = value.find("=")
            if separator <= 0:
                fail("--input-shape requires NAME=DIM[,DIM...]")
            name = value[:separator]
            dimensions = value[separator + 1:]
            shape = []
            if dimensions:
                parts = dimensions.split(",")
                if dimensions.endswith(","):
                    fail("Invalid --input-shape dimension list")
                for part in parts:
                    shape.append(_positive_integer(part))
            if name in o.input_shapes:
                fail("Duplicate --input-shape input name")
            o.input_shapes[name] = shape
        elif key == "--output":
            o.output = value
        elif key in ("-batch", "--batch"):
            values = ["0"] if value == "-1" else _split(value)
            unique = set()
            o.batches = []
            for item in values:
                batch = _positive_integer(item, zero=True)
                if batch in unique:
                    fail("Duplicate batch value: " + item)
                unique.add(batch)
                o.batches.append(batch)
            o.batch = o.batches[0]
            o.batch_explicit = True
        elif key == "--images":
            o.images = value
        elif key == "--manifest":
            o.manifest = value
        elif key == "--start-signal":
            o.start_signal = value
        elif key == "--reference-root":
            o.reference_root = value
        elif key == "--reference-config":
            o.reference_config = value
        elif key == "--reference-python":
            o.reference_python = value
        elif key == "--random-shapes":
            o.random_shapes = value
        elif key == "--stability-dir":
            o.stability_dirs.append(value)
        elif key == "--cpu-plugin":
            o.cpu_plugin = value
        elif key == "--seed":
            o.seed = _positive_integer(value, zero=True)
        elif key == "--hyper-threading":
            o.hyper_threading = _parse_yes_no(value, "--hyper-threading")
        elif key == "--cpu-pinning":
            o.cpu_pinning = _parse_yes_no(value, "--cpu-pinning")
        elif key == "--perf-count":
            o.profiling = _parse_yes_no(value, "--perf-count")
        elif key == "--model":
            o.model = value
        elif key == "--precisions":
            o.precision = value
        elif key == "--mode":
            o.mode = value
        elif key == "--warmup":
            o.warmup = _positive_integer(value, zero=True)
        elif key == "--iterations":
            o.iterations = _positive_integer(value)
            o.iterations_explicit = True
        elif key == "--duration":
            try:
                duration = float(value)
            except ValueError:
                fail("--duration must be finite and positive seconds")
            if not math.isfinite(duration) or duration <= 0:
                fail("--duration must be finite and positive seconds")
            o.duration_seconds = duration
        elif key == "--rounds":
            o.rounds = _positive_integer(value)
        elif key == "--threads":
            o.threads = _positive_integer(value)
        elif key == "--step-timeout-seconds":
            o.step_timeout_seconds = _positive_integer(value)
        elif key in ("--atol", "--rtol"):
            _parse_tolerances(key, value, o.atol if key == "--atol" else o.rtol)
        elif key == "--threshold":
            try:
                o.threshold = float(value)
            except ValueError:
                fail("threshold must be finite in [-1,1)")
            if not math.isfinite(o.threshold) or o.threshold < -1 or o.threshold >= 1:
                fail("threshold must be finite in [-1,1)")
        else:
            fail("Unknown option: " + key)

    batch_modes = ("performance", "accuracy", "stability", "all", "preprocess", "single")
    if o.mode not in batch_modes and (len(o.batches) > 1 or any(b != 0 for b in o.batches)):
        fail("Batch lists and static batch values require a model execution mode")
    if o.duration_seconds is not None:
        if o.iterations_explicit:
            fail("--iterations and --duration are mutually exclusive")
        if o.mode not in ("performance", "all"):
            fail("--duration requires --mode performance or all")
    if o.input_count_explicit and o.mode not in ("performance", "all"):
        fail("--input-count requires --mode performance or all")
    if o.input_count_explicit and (o.prepared or o.manifest or o.model_file):
        fail("--input-count requires generated prediction inputs, not a manifest or --model-file")
    if (o.atol or o.rtol) and o.mode != "random-accuracy":
        fail("--atol and --rtol require --mode random-accuracy")
    if o.resample_common_cosine_failures and o.mode != "random-accuracy":
        fail("--resample-common-cosine-failures requires --mode random-accuracy")
    if o.mode == "tolerance-search" and o.precision != "FP32":
        fail("tolerance-search requires --precisions FP32")
    if o.stability_dirs and o.mode != "compare-stability":
        fail("--stability-dir requires --mode compare-stability")
    if o.cpu_plugin and o.mode != "gather-guard":
        fail("--cpu-plugin requires --mode gather-guard")
    if o.mode in ("self-test", "gather-guard"):
        return o
    if o.mode == "compare-stability":
        if not o.output or os.path.exists(o.output) or not o.stability_dirs:
            fail("compare-stability requires --stability-dir and --output naming a new JSON file")
        return o
    if o.model_file:
        if not (o.model_file and Path(o.model_file).is_file()):
            fail("Model file not found: " + o.model_file)
        if o.model_root or o.model != "all" or o.prepared or o.manifest or o.images:
            fail("--model-file generates random inputs; do not specify model-root, model, images or manifest")
        if o.mode not in ("performance", "accuracy", "stability", "all", "preprocess"):
            fail("--model-file supports performance, accuracy, stability, all and preprocess")
        o.model_file = absolute(o.model_file)
        o.model_root = str(Path(o.model_file).parent)
    elif o.input_shapes or o.random_samples != 10:
        fail("--input-shape and --samples require --model-file")
    if not o.output or not o.model_root:
        fail("--output and --model-root are required")
    if o.hyper_threading is not None and o.mode in ("random-accuracy", "tolerance-search"):
        fail("--hyper-threading is not supported by random-accuracy or tolerance-search")
    execution_modes = ("performance", "accuracy", "stability", "all", "preprocess",
                       "single", "random-accuracy", "tolerance-search")
    if o.mode not in execution_modes:
        fail("Unsupported mode: " + o.mode)
    if o.mode in ("all", "stability") and o.rounds < 2:
        fail("Stability requires at least two rounds")
    if (o.prepared or o.mode == "single") and not o.manifest:
        fail("Prepared inputs and accuracy require --manifest")
    if o.mode == "single" and (not o.prepared or o.model == "all" or
                               len(_split(o.model)) != 1 or len(_split(o.precision)) != 1 or
                               o.step_timeout_seconds == 0):
        fail("Single mode requires --prepared-inputs, one model, one precision, "
             "and a positive --step-timeout-seconds")
    if (not o.prepared and not o.model_file and not o.images and
            o.mode not in ("random-accuracy", "tolerance-search")):
        fail("Native image input requires --images")
    precisions = _split(o.precision)
    if o.resample_common_cosine_failures and ("BF16" not in precisions or "FP16" not in precisions):
        fail("--resample-common-cosine-failures requires both BF16 and FP16")
    for precision in precisions:
        if precision not in ("FP32", "BF16", "FP16"):
            fail("Unsupported precision: " + precision)
    if o.mode in ("random-accuracy", "tolerance-search") and (o.prepared or o.manifest or
                                                              o.images or o.start_signal):
        fail("Random accuracy generates fresh inputs; do not specify a dataset, images, or start signal")
    if o.mode not in ("random-accuracy", "tolerance-search") and (o.reference_root or
            o.reference_config or o.random_shapes or o.reference_python != "python3" or
            (o.seed != 43 and not o.model_file)):
        fail("Random input/reference options require --mode random-accuracy")
    if not o.reference_python or o.step_timeout_seconds == 0:
        fail("Reference Python and timeout must be nonempty and positive")
    return o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_id(model_id):
    return model_id.replace(":", "_")


def batch_directory_name(batch):
    return "batch_dynamic" if batch == 0 else "batch_" + str(batch)


def write_summary(path, cases, include_batch=False):
    header = ["model", "precision", "sdpa_bf16_nodes", "images", "mean_ms", "p99_ms",
              "throughput_fps", "accuracy_passed", "md5_consistent", "stable_outputs_finite",
              "status", "timed_inferences", "elapsed_seconds", "effective_threads"]
    if include_batch:
        header.insert(2, "batch")
    lines = [tsv_row(header)]
    for row in cases:
        fields = [row["model"], row["precision"], tsv_json_value(row["sdpa_bf16_nodes"]),
                  tsv_json_value(row["image_count"])]
        if include_batch:
            fields.insert(2, tsv_json_value(row["batch"]))
        for metric in ("mean_ms", "p99_ms", "throughput_fps"):
            fields.append(tsv_json_value(row["performance"][metric]) if "performance" in row else "")
        fields.append(tsv_json_value(row["accuracy"]["passed_images"]) if "accuracy" in row else "")
        fields.append(tsv_json_value(row["stability"]["md5_consistent"]) if "stability" in row else "")
        fields.append(tsv_json_value(row["stability"]["all_outputs_finite"]) if "stability" in row else "")
        fields.append(row["status"])
        fields.append(tsv_json_value(row["performance"]["samples"]) if "performance" in row else "")
        fields.append(tsv_json_value(row["performance"]["elapsed_seconds"]) if "performance" in row else "")
        if "effective_config" in row and "INFERENCE_NUM_THREADS" in row["effective_config"]:
            fields.append(tsv_json_value(row["effective_config"]["INFERENCE_NUM_THREADS"]))
        else:
            fields.append("")
        lines.append(tsv_row(fields))
    with open(str(path), "w") as handle:
        handle.write("".join(lines))


def _partial_to_list(partial):
    result = []
    for dim in partial:
        result.append(dim.get_length() if dim.is_static else None)
    return result


def output_ports(compiled):
    ports = []
    for port in compiled.outputs:
        ports.append({
            "names": sorted(list(port.get_names())),
            "shape": _partial_to_list(port.get_partial_shape()),
            "dtype": port.get_element_type().get_type_name(),
        })
    return ports


def input_ports(model):
    inputs = []
    for index in range(len(model.inputs)):
        port = model.input(index)
        inputs.append({
            "index": index,
            "names": sorted(list(port.get_names())),
            "dtype": port.get_element_type().get_type_name(),
            "partial_shape": str(port.get_partial_shape()),
        })
    return inputs


def runtime_info(compiled):
    nodes = []
    try:
        ordered = compiled.get_runtime_model().get_ordered_ops()
    except Exception:
        ordered = []
    for node in ordered:
        metadata = {}
        try:
            for key, value in node.get_rt_info().items():
                metadata[key] = str(value)
        except Exception:
            pass
        try:
            output_types = [o.get_element_type().get_type_name() for o in node.outputs()]
        except Exception:
            output_types = []
        nodes.append({"name": node.get_friendly_name(), "type": node.get_type_name(),
                      "output_types": output_types, "rt_info": metadata})
    return {"nodes": nodes}


def reshape_model_batch(model, batch):
    if batch <= 0:
        return
    for index in range(len(model.inputs)):
        port = model.input(index)
        partial = port.get_partial_shape()
        rank = partial.rank
        if rank.is_dynamic or rank.get_length() == 0:
            fail("Static batch reshape requires ranked, non-scalar model inputs")
        if partial[0].is_static and partial[0].get_length() != batch:
            fail("Requested batch %d conflicts with static input batch %d for %s" % (
                batch, partial[0].get_length(), port.get_any_name()))
        shape = [dim.get_length() if dim.is_static else -1 for dim in partial]
        shape[0] = batch
        model.reshape({port.get_any_name(): shape})


def _zero_tensor(tensor):
    from .tensor_io import _as_real, element_type_of, to_numpy
    name = element_type_of(tensor)
    if name not in tensor_io.REAL_DTYPES:
        fail("Unsupported real output dtype")
    values = _as_real(to_numpy(tensor), name)
    return bool(np.all(values == 0))


def compare_prediction_outputs(reference, actual, output_names, threshold):
    if len(reference) != len(actual):
        raise OVRuntimeError("Prediction output count mismatch")
    finite = True
    exact_outputs = True
    real_reference = []
    real_actual = []
    exact_mismatches = []
    exact_output_count = 0
    for port, (ref, act) in enumerate(zip(reference, actual)):
        if list(ref.shape) != list(act.shape):
            raise OVRuntimeError("Prediction output shape mismatch at port " + str(port))
        ref_info = tensor_io.tensor_info(ref)
        act_info = tensor_io.tensor_info(act)
        finite = finite and ref_info["finite"] and act_info["finite"]
        if ref_info["dtype"] in tensor_io.REAL_DTYPES:
            real_reference.append(ref)
            real_actual.append(act)
        else:
            exact_output_count += 1
            equal = (ref_info["dtype"] == act_info["dtype"] and
                     ref_info["md5"] == act_info["md5"])
            if not equal:
                exact_outputs = False
                exact_mismatches.append(output_names[port] if port < len(output_names) else str(port))
    row = {"finite": finite, "exact_outputs_equal": exact_outputs,
           "exact_output_count": exact_output_count, "exact_output_mismatches": exact_mismatches}
    if not finite:
        for key in ("cosine_similarity", "normalized_rmse", "mean_abs", "max_abs", "abs_error_variance"):
            row[key] = None
        row["status"] = "FAIL"
        return row
    real_outputs_equal = True
    if not real_reference:
        for key in ("cosine_similarity", "normalized_rmse", "mean_abs", "max_abs", "abs_error_variance"):
            row[key] = None
    else:
        ref_zero = all(_zero_tensor(t) for t in real_reference)
        act_zero = all(_zero_tensor(t) for t in real_actual)
        if ref_zero or act_zero:
            real_outputs_equal = ref_zero and act_zero
            for key in ("cosine_similarity", "normalized_rmse", "mean_abs", "max_abs",
                        "abs_error_variance"):
                row[key] = None
            row["zero_norm"] = True
        else:
            diff = tensor_io.compare(real_reference, real_actual)
            row["cosine_similarity"] = diff["cosine"]
            row["normalized_rmse"] = diff["normalized_rmse"]
            row["mean_abs"] = diff["mean_abs"]
            row["max_abs"] = diff["max_abs"]
            row["abs_error_variance"] = diff["abs_error_variance"]
            real_outputs_equal = diff["cosine"] > threshold
    row["status"] = "PASS" if (exact_outputs and real_outputs_equal) else "FAIL"
    return row


def bind_inputs(request, sample):
    _session.bind_request(request, request.get_compiled_model(), list(sample.inputs))


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

class Sample:
    def __init__(self, image, inputs, references):
        self.image = image
        self.inputs = inputs
        self.references = references


def _seed_generator(seed, model_id):
    # Deterministic per (seed, model id); mirrors the C++ seed_seq intent.
    entropy = seed.to_bytes(8, "little", signed=False) + model_id.encode("utf-8")
    digest = int.from_bytes(__import__("hashlib").sha256(entropy).digest()[:8], "little")
    return np.random.default_rng(digest)


def samples_for(spec, model, o, manifest):
    entries = {}
    found = False
    if o.manifest:
        if manifest.get("schema_version") != 1:
            raise OVRuntimeError("Unsupported dataset schema")
        models_seen = set()
        for entry in manifest["models"]:
            model_id = entry["model"]
            if model_id in models_seen:
                raise OVRuntimeError("Duplicate model in dataset: " + model_id)
            models_seen.add(model_id)
            if model_id != spec.id:
                continue
            found = True
            for sample in entry["samples"]:
                image = sample["image"]
                if image in entries:
                    raise OVRuntimeError("Duplicate dataset image")
                entries[image] = sample
        if not found:
            raise OVRuntimeError("Model missing from manifest: " + spec.id)
    samples = []
    if o.prepared:
        base = Path(o.manifest).parent
        for image, entry in entries.items():
            inputs = []
            if "inputs" in entry:
                for descriptor in entry["inputs"]:
                    inputs.append((descriptor["name"], tensor_io.load_tensor(descriptor, base)))
            else:
                inputs.append(("", tensor_io.load_tensor(entry["input"], base)))
            samples.append(Sample(image, inputs, entry.get("outputs", [])))
            if o.mode == "single":
                break
    else:
        if spec.family in ("prediction", "generic"):
            generator = _seed_generator(o.seed, spec.id)
            count = o.input_count if spec.family == "prediction" else o.random_samples
            for sample in range(count):
                name = "synthetic_%02d" % sample
                inputs = random_inputs(spec, model, generator, o.input_shapes)
                samples.append(Sample(name, inputs, []))
        else:
            for path in image_files(o.images):
                name = Path(path).name
                if o.manifest and name not in entries:
                    raise OVRuntimeError("Missing image reference: " + name)
                tensor = preprocess_image(path, spec.family, spec.input_size,
                                          spec.preprocess_scale)
                references = entries[name].get("outputs", []) if name in entries else []
                samples.append(Sample(name, [("", tensor)], references))
    if not samples:
        raise OVRuntimeError("Empty input dataset: " + spec.id)
    if spec.family in ("prediction", "generic"):
        for sample in samples:
            validate_inputs(model, sample.inputs)
    else:
        for sample in samples:
            if len(sample.inputs) != 1:
                raise OVRuntimeError("Image models require exactly one input tensor")
            validate_input(model, sample.inputs[0][1])
    return samples


# ---------------------------------------------------------------------------
# Single-mode timed stages
# ---------------------------------------------------------------------------

class _Timed:
    def __init__(self, timings, name, timeout_seconds, marker_path):
        self.timings = timings
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.marker_path = marker_path
        self.start = None
        self._previous = None

    def _handler(self, _signum, _frame):
        try:
            record = {"status": "TIMEOUT", "stage": self.name,
                      "timeout_seconds": self.timeout_seconds}
            with open(self.marker_path, "w") as handle:
                handle.write(__import__("json").dumps(record) + "\n")
        except Exception:
            pass
        sys.stderr.write("TIMEOUT: %s exceeded its configured deadline\n" % self.name)
        sys.stderr.flush()
        os._exit(124)

    def __enter__(self):
        if self.timeout_seconds and self.timeout_seconds > 0:
            self._previous = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.timeout_seconds)
        self.start = time.monotonic()
        sys.stderr.write("[single] %s.begin timeout_seconds=%d\n" % (self.name, self.timeout_seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.monotonic() - self.start) * 1000.0
        if self._previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._previous)
        if exc_type is None:
            sys.stderr.write("[single] %s.end elapsed_ms=%g\n" % (self.name, elapsed_ms))
        else:
            sys.stderr.write("[single] %s.error elapsed_ms=%g\n" % (self.name, elapsed_ms))
        self.timings[self.name] = elapsed_ms
        return False  # propagate exceptions naturally


def _openvino_version():
    try:
        version = ov.get_version()
        return {"build": str(version[0]), "description": str(version[1])}
    except Exception:
        return {"build": str(__import__("openvino").__version__), "description": ""}


def _default_reference_worker():
    override = os.environ.get("OVBENCH_REFERENCE_WORKER")
    if override:
        return override
    return str(_REPO_ROOT / "tools" / "random_reference_worker.py")


def run(o):
    import copy
    from . import selftest

    if len(o.batches) > 1 and o.start_signal:
        fail("Multiple --batch values cannot be combined with --start-signal; "
             "run batch variants separately")
    if o.mode == "gather-guard":
        return selftest.run_gather_guard_test(o.cpu_plugin)
    if o.mode == "self-test":
        return selftest.run_random_accuracy_core_tests()
    if o.mode == "compare-stability":
        report = compare_stability_directories(o.stability_dirs)
        write_json(str(o.output), _clean(report))
        print("%s: %d groups, %d binaries" % (report["status"], len(report["groups"]),
                                              report["binary_files"]))
        return 0 if report["status"] == "PASS" else 1

    if len(o.batches) > 1:
        return _run_multi_batch(o)

    ids = [] if o.model == "all" else _split(o.model)
    if o.model_file:
        stem = Path(o.model_file).stem
        model_id = stem if stem not in ("", ".", "..") else "custom_model"
        specs = [ModelSpec(model_id, "generic", o.model_file, 0, 0)]
    elif not ids:
        specs = model_specs()
    else:
        specs = [find_model(i) for i in ids]

    if o.mode in ("random-accuracy", "tolerance-search"):
        return _run_random_accuracy(o, specs)

    if o.mode == "single" and (len(specs) != 1 or len(_split(o.precision)) != 1 or not o.prepared):
        fail("Single mode requires one model, one precision, and prepared inputs")

    if o.start_signal and (o.mode != "performance" or len(specs) != 1 or
                           len(_split(o.precision)) != 1 or
                           not os.path.isdir(os.path.dirname(o.start_signal)) or
                           os.path.exists(o.start_signal)):
        fail("--start-signal requires one performance case and a new signal in an existing directory")

    manifest = read_json(o.manifest) if o.manifest else {}
    if os.path.exists(o.output):
        raise OVRuntimeError("Output directory already exists: " + o.output)
    os.makedirs(o.output, exist_ok=True)

    report = {
        "schema_version": 1, "status": "RUNNING",
        "openvino": _openvino_version(),
        "mode": o.mode, "threads": o.threads, "batch": o.batch, "warmup": o.warmup,
        "iterations": None if o.duration_seconds is not None else o.iterations,
        "duration_seconds": o.duration_seconds,
        "rounds": o.rounds, "threshold": o.threshold,
        "step_timeout_seconds": o.step_timeout_seconds,
        "prepared_inputs": o.prepared, "sdpa_bf16": o.sdpa_bf16,
        "random_input_count": o.input_count,
        "model_root": absolute(o.model_root),
        "images": o.images, "manifest": o.manifest,
        "manifest_md5": md5_file(o.manifest) if o.manifest else "",
        "reference_backend": (manifest.get("reference_backend", "") if o.manifest
                              else "OpenVINO FP32 self-reference"),
        "cases": [],
    }
    _write_json(os.path.join(o.output, "results.json"), report)
    single = o.mode == "single"
    marker_path = os.path.join(o.output, "timeout.json") if single else None
    all_passed = True
    process = process_identity()
    report["process_identity"] = process
    try:
        for spec in specs:
            stage_timings = {}
            source_model = model_path(spec, o.model_root)
            if str(source_model).endswith(".xml"):
                xml_path = str(source_model)
                bin_path = os.path.splitext(str(source_model))[0] + ".bin"
                provenance = {"xml": absolute(xml_path), "xml_md5": md5_file(xml_path),
                              "bin_md5": md5_file(bin_path)}
            else:
                provenance = {"onnx": absolute(str(source_model)),
                              "onnx_md5": md5_file(str(source_model))}

            if single:
                with _Timed(stage_timings, "read_model", o.step_timeout_seconds, marker_path):
                    model = load_model(spec, o.model_root)
                    reshape_model_batch(model, o.batch)
            else:
                model = load_model(spec, o.model_root)
                reshape_model_batch(model, o.batch)

            model_dir = os.path.join(o.output, safe_id(spec.id))
            os.makedirs(model_dir, exist_ok=True)
            if spec.family == "generic":
                report["input_analysis"] = {"model_file": o.model_file, "seed": o.seed,
                                            "inputs": input_ports(model)}
                _write_json(os.path.join(o.output, "results.json"), report)
                _write_json(os.path.join(model_dir, "input_analysis.json"),
                            report["input_analysis"])
                print("MODEL_INPUTS " + _clean(report["input_analysis"]).__str__())

            if single:
                with _Timed(stage_timings, "load_inputs", o.step_timeout_seconds, marker_path):
                    samples = samples_for(spec, model, o, manifest)
            else:
                samples = samples_for(spec, model, o, manifest)

            if spec.family == "generic":
                inputs_desc = input_ports(model)
                for index in range(len(model.inputs)):
                    generated = [{"sample": s.image,
                                  "tensor": tensor_io.tensor_info(s.inputs[index][1])}
                                 for s in samples]
                    inputs_desc[index]["samples"] = generated
                analysis = {"model_file": o.model_file, "seed": o.seed, "samples": len(samples),
                            "inputs": inputs_desc,
                            "generation": "Floating: normal(0,0.1); integer: uniform {0,1}; "
                                          "boolean: Bernoulli(0.75)",
                            "scope": "Synthetic inputs do not infer index bounds, shape tensor "
                                     "values, or application semantics."}
                _write_json(os.path.join(model_dir, "input_analysis.json"), analysis)
                report["input_analysis"] = analysis
                _write_json(os.path.join(o.output, "results.json"), report)
                print("INPUT_ANALYSIS " + str(_clean(analysis)))

            if o.mode == "preprocess":
                rows = []
                for i in range(len(samples)):
                    input_descriptors = []
                    for input_index, (name, tensor) in enumerate(samples[i].inputs):
                        name_text = "input_%d_%d.bin" % (i, input_index)
                        descriptor = tensor_io.save_tensor(tensor, os.path.join(model_dir, name_text))
                        descriptor["file"] = name_text
                        if name:
                            descriptor["name"] = name
                        input_descriptors.append(descriptor)
                    if spec.family in ("prediction", "generic"):
                        rows.append({"image": samples[i].image, "inputs": input_descriptors})
                    else:
                        rows.append({"image": samples[i].image, "input": input_descriptors[0]})
                _write_json(os.path.join(model_dir, "inputs.json"),
                            {"model": spec.id, "samples": rows})
                continue

            for precision in _split(o.precision):
                out_dir = os.path.join(model_dir, precision)
                os.makedirs(out_dir, exist_ok=True)
                graph = model.clone()
                patched = apply_default_sdpa_bf16(spec, precision, graph, o.sdpa_bf16)
                setup_start = time.monotonic()
                if single:
                    with _Timed(stage_timings, "compile_model", o.step_timeout_seconds, marker_path):
                        session = Session(graph, precision, o.threads, o.cpu_pinning,
                                          o.profiling, o.hyper_threading)
                else:
                    session = Session(graph, precision, o.threads, o.cpu_pinning,
                                      o.profiling, o.hyper_threading)
                request = session.request()
                compiled = session.compiled
                result = {"model": spec.id, "family": spec.family, "precision": precision,
                          "batch": o.batch, "sdpa_bf16_nodes": patched, "image_count":
                          len(samples), "ir": provenance,
                          "setup_seconds": time.monotonic() - setup_start,
                          "outputs": output_ports(compiled), "status": "PASS"}
                result["effective_config"] = _session.get_effective(compiled)

                if single:
                    sample = samples[0]
                    with _Timed(stage_timings, "bind_inputs", o.step_timeout_seconds, marker_path):
                        bind_inputs(request, sample)
                    with _Timed(stage_timings, "inference", o.step_timeout_seconds, marker_path):
                        request.infer()
                    with _Timed(stage_timings, "copy_outputs", o.step_timeout_seconds, marker_path):
                        outputs = copy_outputs(request)
                    output_rows = []
                    finite = True
                    with _Timed(stage_timings, "inspect_outputs", o.step_timeout_seconds, marker_path):
                        for port in range(len(outputs)):
                            info = tensor_io.tensor_info(outputs[port])
                            finite = finite and info["finite"]
                            output_rows.append({"port": port, "names": sorted(list(compiled.output(port).get_names())),
                                                "dtype": info["dtype"], "shape": info["shape"],
                                                "finite": info["finite"], "md5": info["md5"]})
                    result["step_timings_ms"] = stage_timings
                    result["single_inference"] = {"sample": sample.image, "completed": True,
                                                  "finite_outputs": finite,
                                                  "output_count": len(outputs),
                                                  "outputs": output_rows,
                                                  "inference_ms": stage_timings["inference"]}
                    if not finite:
                        result["status"] = "FAIL"
                    sys.stderr.write("[single] completed sample=%s finite_outputs=%s\n"
                                     % (sample.image, str(finite).lower()))

                def infer(sample):
                    bind_inputs(request, sample)
                    request.infer()

                if o.mode in ("performance", "all"):
                    _run_performance(o, request, samples, result, out_dir)
                if o.mode in ("accuracy", "all"):
                    _run_accuracy(o, spec, model, request, compiled, samples, result, precision,
                                  marker_path)
                if o.mode in ("stability", "all"):
                    _run_stability(o, spec, request, samples, result, precision, out_dir,
                                   patched, process)
                if not single:
                    _write_json(os.path.join(out_dir, "runtime_graph.json"), runtime_info(compiled))
                    result["runtime_graph"] = "runtime_graph.json"

                all_passed = all_passed and result["status"] == "PASS"
                report["cases"].append(result)
                _write_json(os.path.join(out_dir, "result.json"), result)
                _write_json(os.path.join(o.output, "results.json"), report)
                write_summary(os.path.join(o.output, "summary.tsv"), report["cases"],
                              o.batch_explicit)
                print("%s %s %s" % (spec.id, precision, result["status"]))

        report["status"] = "PASS" if all_passed else "FAIL"
        _write_json(os.path.join(o.output, "results.json"), report)
        return 0 if all_passed else 1
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        if os.path.isdir(o.output):
            _write_json(os.path.join(o.output, "results.json"), report)
        raise


def _run_multi_batch(o):
    import copy
    if os.path.exists(o.output):
        raise OVRuntimeError("Output directory already exists: " + o.output)
    os.makedirs(o.output)
    combined = {"schema_version": 1, "status": "RUNNING", "mode": o.mode,
                "batch_variants": [], "cases": []}
    passed = True
    for batch in o.batches:
        directory = batch_directory_name(batch)
        options = copy.deepcopy(o)
        options.batch = batch
        options.batches = [batch]
        options.batch_explicit = False
        options.output = os.path.join(o.output, directory)
        code = run(options)
        report = read_json(os.path.join(options.output, "results.json"))
        combined["batch_variants"].append({"batch": batch, "directory": directory,
                                           "status": report.get("status", "ERROR")})
        for item in report.get("cases", []):
            item["batch"] = batch
            item["batch_directory"] = directory
            combined["cases"].append(item)
        passed = passed and code == 0 and report.get("status", "ERROR") == "PASS"
    combined["status"] = "PASS" if passed else "FAIL"
    _write_json(os.path.join(o.output, "results.json"), combined)
    write_summary(os.path.join(o.output, "summary.tsv"), combined["cases"], include_batch=True)
    return 0 if passed else 1


def _run_random_accuracy(o, specs):
    from . import random_accuracy as _ra
    random = _ra.RandomAccuracyOptions(
        model_root=o.model_root, output=o.output,
        reference_root=o.reference_root, reference_config=o.reference_config,
        shapes=o.random_shapes, python=o.reference_python,
        worker=_default_reference_worker(), precisions=_split(o.precision),
        iterations=o.iterations, seed=o.seed, threads=o.threads,
        timeout_seconds=o.step_timeout_seconds, threshold=o.threshold,
        atol=o.atol, rtol=o.rtol, cpu_pinning=o.cpu_pinning, profiling=o.profiling,
        sdpa_bf16=o.sdpa_bf16, tolerance_search=(o.mode == "tolerance-search"),
        resample_common_cosine_failures=o.resample_common_cosine_failures)
    if not random.reference_root:
        sibling = Path(absolute(o.model_root)).parent / "models"
        random.reference_root = str(sibling) if sibling.is_dir() else o.model_root
    return _ra.run_random_accuracy(specs, random)


def _run_performance(o, request, samples, result, out_dir):
    n = len(samples)
    sample_index = 0
    reuse = n == 1

    def next_sample():
        nonlocal sample_index
        sample = samples[sample_index]
        sample_index = (sample_index + 1) % n
        return sample

    if reuse:
        bind_inputs(request, samples[0])
    for _ in range(o.warmup):
        if reuse:
            request.infer()
        else:
            bind_inputs(request, next_sample())
            request.infer()
    if o.start_signal:
        _write_json(os.path.join(o.output, "ready.json"), {"status": "READY"})
        deadline = time.monotonic() + 600
        while not os.path.exists(o.start_signal):
            if os.path.exists(os.path.join(os.path.dirname(o.start_signal), "abort.signal")):
                raise OVRuntimeError("Multi-process wave aborted")
            if time.monotonic() > deadline:
                raise OVRuntimeError("Timed out waiting for start signal")
            time.sleep(0.01)
    latencies = []
    indices = []
    measurement_start = time.monotonic()

    def active():
        if o.duration_seconds is not None:
            return not latencies or (time.monotonic() - measurement_start) < o.duration_seconds
        return len(latencies) < o.iterations

    while active():
        indices.append(sample_index)
        sample = next_sample()
        if not reuse:
            bind_inputs(request, sample)
        start = time.monotonic()
        request.infer()
        latencies.append((time.monotonic() - start) * 1000.0)
    elapsed_seconds = time.monotonic() - measurement_start
    for _ in range(o.warmup):
        if reuse:
            request.infer()
        else:
            bind_inputs(request, next_sample())
            request.infer()
    first_shape = list(samples[0].inputs[0][1].shape)
    batch_size = first_shape[0] if first_shape else 1
    stats = tensor_io.statistics(latencies, batch_size)
    result["performance"] = {
        "mean_ms": stats["mean_ms"], "p50_ms": stats["p50_ms"], "p90_ms": stats["p90_ms"],
        "p99_ms": stats["p99_ms"], "throughput_fps": stats["samples"] * batch_size / elapsed_seconds,
        "inference_throughput_fps": stats["throughput_fps"], "samples": stats["samples"],
        "iterations": stats["samples"],
        "measurement_mode": "duration" if o.duration_seconds is not None else "iterations",
        "requested_iterations": None if o.duration_seconds is not None else o.iterations,
        "requested_duration_seconds": o.duration_seconds,
        "elapsed_seconds": elapsed_seconds, "warmup_iterations": o.warmup,
        "cooldown_iterations": o.warmup, "output_files_saved": o.save_performance_outputs,
        "input_sequence": "cyclic; continues across warmup, performance and cooldown",
        "latency_unit": "single_input_batch", "batch_size": batch_size,
    }
    _write_json(os.path.join(out_dir, "latencies_ms.json"), latencies)
    _write_json(os.path.join(out_dir, "sample_indices.json"), indices)
    validity = []
    finite_all = True
    for i in range(n):
        bind_inputs(request, samples[i])
        request.infer()
        outputs = copy_outputs(request)
        finite = True
        output_metadata = []
        for p in range(len(outputs)):
            if o.save_performance_outputs:
                descriptor = tensor_io.save_tensor(outputs[p], os.path.join(
                    out_dir, "check_%d_%d.bin" % (i, p)))
            else:
                descriptor = tensor_io.tensor_info(outputs[p])
            finite = finite and descriptor["finite"]
            descriptor["output_port"] = p
            output_metadata.append(descriptor)
        validity.append({"image": samples[i].image, "finite": finite, "outputs": output_metadata})
        flat_finite = finite
        finite_all = finite_all and flat_finite
        if not finite_all:
            result["status"] = "FAIL"
    _write_json(os.path.join(out_dir, "output_validity.json"), validity)
    result["output_validity"] = "output_validity.json"


def _run_accuracy(o, spec, model, request, compiled, samples, result, precision, marker_path):
    comparisons = []
    passed = 0
    native_self_reference = all(len(s.references) == 0 for s in samples)
    reference_session = None
    if native_self_reference and precision != "FP32":
        def ref_infer(sample):
            pass
        reference_session = Session(model.clone(), "FP32", o.threads, o.cpu_pinning,
                                    o.profiling, o.hyper_threading)
    manifest_base = Path(o.manifest).parent if o.manifest else Path(".")

    def infer(sample):
        bind_inputs(request, sample)
        request.infer()

    for sample in samples:
        if not native_self_reference and len(sample.references) != len(compiled.outputs):
            raise OVRuntimeError("Reference output count mismatch: " + sample.image)
        reference = []
        if native_self_reference:
            if precision == "FP32":
                infer(sample)
                reference = copy_outputs(request)
            else:
                reference_request = reference_session.request()
                bind_inputs(reference_request, sample)
                reference_request.infer()
                reference = copy_outputs(reference_request)
        else:
            port_index = 0
            for desc in sample.references:
                if desc.get("name") and desc["name"] not in set(
                        compiled.output(port_index).get_names()):
                    raise OVRuntimeError("Reference output name mismatch: " + sample.image)
                reference.append(tensor_io.load_tensor(desc, manifest_base))
                port_index += 1
        infer(sample)
        actual = copy_outputs(request)
        if spec.family in ("prediction", "generic"):
            row = compare_prediction_outputs(reference, actual, spec.output_names, o.threshold)
        else:
            diff = tensor_io.compare(reference, actual)
            row = {"finite": diff["finite"],
                   "status": "PASS" if (diff["finite"] and diff["cosine"] > o.threshold) else "FAIL"}
            for key in ("normalized_rmse", "mean_abs", "max_abs", "abs_error_variance"):
                row[key] = diff[key] if diff["finite"] else None
            row["cosine_similarity"] = diff["cosine"] if diff["finite"] else None
        if row["status"] == "PASS":
            passed += 1
        row["image"] = sample.image
        comparisons.append(row)
    result["accuracy"] = {"passed_images": passed, "images": len(samples), "records": comparisons}
    if passed != len(samples):
        result["status"] = "FAIL"


def _run_stability(o, spec, request, samples, result, precision, out_dir, patched, process):
    n = len(samples)
    first = [[] for _ in range(n)]
    files = []
    consistent = True
    finite = True

    def infer(sample):
        bind_inputs(request, sample)
        request.infer()

    for round_index in range(o.rounds):
        folder = os.path.join(out_dir, "round_%d" % (round_index + 1))
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            infer(samples[i])
            outputs = copy_outputs(request)
            if round_index == 0:
                first[i] = []
            for p in range(len(outputs)):
                configuration = "sdpa_bf16" if patched else "original"
                filename = stability_filename({"image": samples[i].image, "model": spec.id,
                                               "precision": precision, "configuration": configuration,
                                               "process": process, "round": round_index + 1,
                                               "port": p})
                path = os.path.join(folder, filename)
                descriptor = tensor_io.save_tensor(outputs[p], path)
                if round_index == 0:
                    first[i].append(descriptor)
                base = first[i][p]
                equal = all(descriptor[k] == base[k] for k in ("md5", "dtype", "shape", "bytes"))
                consistent = consistent and equal
                finite = finite and descriptor["finite"]
                descriptor["file"] = os.path.relpath(path, o.output)
                descriptor["round"] = round_index + 1
                descriptor["image"] = samples[i].image
                descriptor["output_port"] = p
                descriptor["process"] = process
                descriptor["configuration"] = configuration
                descriptor["matches_round_1"] = equal
                files.append(descriptor)
    _write_json(os.path.join(out_dir, "files.json"), files)
    result["stability"] = {"rounds": o.rounds, "binary_files": len(files),
                           "md5_consistent": consistent, "all_outputs_finite": finite}
    if not consistent or not finite:
        result["status"] = "FAIL"


def main(argv):
    try:
        o = parse(list(argv))
    except (OVBError, OVRuntimeError) as error:
        sys.stderr.write("ERROR: %s\n" % error)
        return 2 if isinstance(error, OVBError) else 1
    if o == "help":
        return 0
    try:
        return run(o)
    except (OVBError, OVRuntimeError) as error:
        sys.stderr.write("ERROR: %s\n" % error)
        return 2 if isinstance(error, OVBError) else 1
    except Exception as error:
        sys.stderr.write("ERROR: %s\n" % error)
        return 1
