"""OpenVINO Core / session wrapper, ported from the ``ovbench::Session`` class
in ``src/core.cpp``.

The compile configuration mirrors the C++ engine exactly: LATENCY performance
hint, a single stream, the requested thread count, an explicit inference
precision, and optional CPU-pinning / hyper-threading overrides.
"""

import numpy as np
import openvino as ov

from .utils import OVRuntimeError, fail

PRECISION_ELEMENT = {"FP32": "f32", "BF16": "bf16", "FP16": "f16"}

# Configuration keys echoed into results, in the same order as the C++ binary.
EFFECTIVE_CONFIG_KEYS = (
    "INFERENCE_NUM_THREADS",
    "NUM_STREAMS",
    "PERFORMANCE_HINT",
    "INFERENCE_PRECISION_HINT",
    "PERF_COUNT",
    "ENABLE_CPU_PINNING",
    "ENABLE_HYPER_THREADING",
)


def make_config(precision, threads, cpu_pinning=None, profiling=True, hyper_threading=None):
    """Build the ov::AnyMap equivalent as a Python dict."""
    config = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": 1,
        "INFERENCE_NUM_THREADS": threads,
        "INFERENCE_PRECISION_HINT": PRECISION_ELEMENT[precision],
        "PERF_COUNT": "YES" if profiling else "NO",
    }
    if cpu_pinning is not None:
        config["ENABLE_CPU_PINNING"] = bool(cpu_pinning)
    if hyper_threading is not None:
        config["ENABLE_HYPER_THREADING"] = bool(hyper_threading)
    return config


def _compile_strict(core, model, config):
    """Compile, progressively dropping properties the plugin marks unsupported."""

    def _compile(remaining):
        try:
            core.compile_model(model, "CPU", dict(remaining))
            return True
        except Exception:
            return False

    if _compile(config):
        return core.compile_model(model, "CPU", dict(config))
    remaining = {k: v for k, v in config.items() if k != "ENABLE_PROFILING"}
    if "PERF_COUNT" in config:
        remaining.pop("PERF_COUNT", None)
    if _compile(remaining):
        return core.compile_model(model, "CPU", dict(remaining))
    for key in ("ENABLE_CPU_PINNING", "ENABLE_HYPER_THREADING", "ENABLE_PROFILING"):
        if key in config:
            trial = {k: v for k, v in config.items() if k != key}
            if _compile(trial):
                return core.compile_model(model, "CPU", dict(trial))
    raise OVRuntimeError("Unable to compile model with the requested CPU configuration")


class Session:
    """Compiled model + a single InferRequest, like the native engine."""

    def __init__(self, model, precision, threads, cpu_pinning=None, profiling=True,
                 hyper_threading=None):
        if model is None:
            fail("Session model must not be null")
        if not (isinstance(threads, int) and threads > 0):
            fail("Inference threads must be positive")
        if precision not in PRECISION_ELEMENT:
            fail("Precision must be FP32, BF16, or FP16")
        core = ov.Core()
        self.compiled = _compile_strict(core, model, make_config(
            precision, threads, cpu_pinning, profiling, hyper_threading))
        self._request = self.compiled.create_infer_request()

    def request(self):
        return self._request

    def compiled_model(self):
        return self.compiled


def copy_outputs(request):
    """C++ ``ovbench::copy_outputs``: own a copy of each output tensor."""
    compiled = request.get_compiled_model()
    outputs = []
    for index in range(len(compiled.outputs)):
        outputs.append(request.get_output_tensor(index))
    return outputs


def port_name(port):
    """First declared name of an input/output port (possibly ''), exact string."""
    names = port.get_names()
    if not names:
        return ""
    return next(iter(names))


def bind_request(request, compiled, entries):
    """Bind generated ``[(name, ov.Tensor)]`` to a request.

    ``set_tensor`` requires the model's declared port *name* (string). When a port
    has no declared name, fall back to ``set_input_tensor`` (position/type based),
    which is exact for single-input models.
    """
    inputs = compiled.inputs
    for index, (name, tensor) in enumerate(entries):
        if index < len(inputs):
            declared = port_name(inputs[index])
            if declared:
                request.set_tensor(declared, tensor)
                continue
        if name:
            request.set_tensor(name, tensor)
        else:
            request.set_input_tensor(tensor)


def _any_to_str(any):
    try:
        value = any.as_string()
        if isinstance(value, str) and value:
            return value
    except Exception:
        pass
    text = str(any)
    # ov.Any wrapping an element type renders as "<Type: 'float32'>" -> clean it.
    import re
    match = re.search(r"'([^']*)'", text)
    return match.group(1) if match else text


def get_effective(compiled, keys=EFFECTIVE_CONFIG_KEYS):
    result = {}
    for key in keys:
        try:
            result[key] = _any_to_str(compiled.get_property(key))
        except Exception:
            result[key] = ""
    return result
