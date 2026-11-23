"""Tensor I/O and numerical comparison, ported from ``src/core.cpp``.

Dtype handling follows the OpenVINO *Python* conventions of the installed SDK:
real values are transported through their numpy views (f32 -> float32,
f64 -> float64, f16 *and* bf16 -> the float16 value view the SDK exposes),
integers through their native numpy integers, booleans through bool. Real math
runs in float64 exactly like the C++ engine, and integer equality/metrics use
exact (Python ``int``) arithmetic so values above 2**53 compare correctly.

All md5 / byte provenance is computed on these deterministic numpy views, so
results are reproducible *within* the Python backend for a fixed seed. Binary
``.bin`` datasets are plain native-endian value payloads for the dtypes above.
"""

import hashlib
import math
import os

import numpy as np
import openvino as ov

from .utils import OVRuntimeError, checked_tensor_bytes, fail, finite_or_fail

# element type -> numpy value dtype used for reads / writes / provenance.
VALUE_DTYPES = {
    "boolean": np.dtype(bool),
    "f16": np.dtype(np.float16),
    "bf16": np.dtype(np.float16),  # SDK exposes bf16 through a float16 view
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
}

REAL_DTYPES = {"f16", "bf16", "f32", "f64"}
INTEGER_DTYPES = set(VALUE_DTYPES) - REAL_DTYPES - {"boolean"}

_PRECISION_ELEMENT = {"FP32": "f32", "BF16": "bf16", "FP16": "f16"}


def element_type_of(tensor):
    return tensor.element_type.get_type_name()


def to_numpy(tensor):
    """Copy an ov::Tensor to a numpy value array (C++ ``ovbench::owned``)."""
    name = element_type_of(tensor)
    if name not in VALUE_DTYPES:
        fail("Unsupported tensor dtype: " + name)
    shape = [int(d) for d in tensor.shape]
    array = np.asarray(tensor.data).reshape(shape)
    if array.dtype != VALUE_DTYPES[name]:
        array = array.astype(VALUE_DTYPES[name])
    return array.copy()


def from_numpy(array, element_type="f32", shape=None):
    ov_type = getattr(ov.Type, element_type, None)
    if ov_type is None:
        fail("Unsupported tensor dtype: " + str(element_type))
    if ov_type.get_type_name() not in VALUE_DTYPES:
        fail("Unsupported tensor dtype: " + ov_type.get_type_name())
    arr = np.asarray(array)
    if shape is None:
        shape = list(arr.shape)
    if not shape:
        shape = [1]
    tensor = ov.Tensor(ov_type, list(shape))
    view = np.asarray(tensor.data).reshape(-1)
    view[:] = arr.reshape(-1).astype(view.dtype, copy=False)
    return tensor


def _to_double(array, name):
    if name in REAL_DTYPES or name in INTEGER_DTYPES:
        return array.astype(np.float64)
    if name == "boolean":
        return array.astype(np.uint8).astype(np.float64)
    fail("Unsupported tensor dtype: " + name)


# Backwards-compatible alias: ``_zero_tensor`` in engine.py imports ``_as_real``.
_as_real = _to_double


def _int_error(reference, actual):
    """Exact per-element |actual - reference| for integer tensors (float64 out)."""
    flat_r, flat_a = reference.reshape(-1), actual.reshape(-1)
    if ("u" in np.dtype(flat_r.dtype).kind or "u" in np.dtype(flat_a.dtype).kind
            or str(flat_r.dtype) == "int64" or str(flat_a.dtype) == "int64"):
        return np.asarray([abs(int(x) - int(y)) for x, y in zip(flat_r, flat_a)],
                          dtype=np.float64).reshape(flat_r.shape)
    diff = flat_r.astype(np.int64) - flat_a.astype(np.int64)
    return np.abs(diff).astype(np.float64).reshape(flat_r.shape)


def finite_array(array, name):
    if name in REAL_DTYPES:
        return bool(np.all(np.isfinite(array.astype(np.float64))))
    return True


def tensor_info(tensor):
    """dtype/shape/bytes/md5/finite, same fields as C++ ``tensor_info``."""
    name = element_type_of(tensor)
    if name not in VALUE_DTYPES:
        fail("Unsupported tensor dtype: " + name)
    shape = [int(d) for d in tensor.shape]
    array = to_numpy(tensor)
    raw = array.tobytes()
    return {
        "dtype": name,
        "shape": shape,
        "bytes": checked_tensor_bytes(array.itemsize, shape),
        "md5": hashlib.md5(raw).hexdigest(),
        "finite": finite_array(array, name),
    }


def md5_buffer_of_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(64 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_tensor(tensor, path):
    """Exclusive create + MD5 readback verification, like the C++ version."""
    info = tensor_info(tensor)
    array = to_numpy(tensor)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(array.tobytes())
    size = os.path.getsize(path)
    readback = md5_buffer_of_file(path)
    if size != info["bytes"] or readback != info["md5"]:
        raise OVRuntimeError("Saved tensor MD5 readback mismatch: %s" % path)
    return info


def load_tensor(descriptor, base):
    """C++ ``load_tensor``: descriptor -> tensor with size/md5/finite checks."""
    if not isinstance(descriptor, dict):
        fail("Tensor descriptor must be an object")
    dtype_name = descriptor["dtype"]
    if dtype_name not in VALUE_DTYPES:
        fail("Unsupported tensor dtype: " + dtype_name)
    shape = descriptor["shape"]
    if not isinstance(shape, list):
        fail("Tensor shape must be an array")
    shape = [int(d) for d in shape]
    if any(d <= 0 for d in shape):
        fail("Tensor dimension out of range")
    itemsize = VALUE_DTYPES[dtype_name].itemsize
    byte_count = checked_tensor_bytes(itemsize, shape)
    if "bytes" in descriptor and int(descriptor["bytes"]) != byte_count:
        fail("Descriptor byte size mismatch")
    filename = descriptor.get("file")
    if not filename:
        fail("Tensor file path must not be empty")
    path = os.path.join(base, filename)
    if not os.path.isfile(path):
        raise OVRuntimeError("Cannot open tensor file: %s" % path)
    if os.path.getsize(path) != byte_count:
        raise OVRuntimeError("Tensor file byte size mismatch: %s" % path)
    array = np.fromfile(path, dtype=VALUE_DTYPES[dtype_name]).reshape(shape)
    data_md5 = hashlib.md5(array.tobytes()).hexdigest()
    if "md5" in descriptor:
        checksum = str(descriptor["md5"]).lower()
        if len(checksum) != 32:
            fail("Descriptor MD5 must contain 32 hexadecimal characters")
        try:
            int(checksum, 16)
        except ValueError:
            fail("Descriptor MD5 must contain 32 hexadecimal characters")
        if checksum != md5_buffer_of_file(path) or checksum != data_md5:
            raise OVRuntimeError("Tensor MD5 mismatch: %s" % path)
    if "finite" in descriptor:
        expected = bool(descriptor["finite"])
        actual = finite_array(array, dtype_name)
        if expected != actual:
            raise OVRuntimeError("Tensor finite metadata mismatch")
    return from_numpy(array, dtype_name, shape=shape)


def is_real_dtype(name):
    return name in REAL_DTYPES


# ---------------------------------------------------------------------------
# Mathematical comparisons
# ---------------------------------------------------------------------------

class RunningVariance:
    """C++ ``AbsoluteErrorMoments``: Welford running stats over |error|."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, values):
        if values.size == 0:
            return
        chunk_mean = float(values.mean())
        chunk_m2 = float(((values - chunk_mean) ** 2).sum())
        count = self.count
        size = int(values.size)
        delta = chunk_mean - self.mean
        combined = count + size
        self.mean += delta * size / combined
        self.m2 += chunk_m2 + delta * delta * count * size / combined
        self.count = combined

    def variance(self):
        return max(0.0, self.m2 / self.count)


def statistics(latencies_ms, batch_size=1):
    """Latency vector -> mean/p50/p90/p99/throughput (C++ ``statistics``)."""
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        fail("Latency samples must not be empty")
    if batch_size <= 0:
        fail("Batch size must be positive")
    if not bool(np.all(np.isfinite(values) & (values > 0))):
        fail("Latency samples must be positive and finite")
    ordered = np.sort(values)

    def percentile(fraction):
        position = fraction * (ordered.size - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        value = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        return finite_or_fail(float(value), "Percentile")

    mean = finite_or_fail(float(ordered.sum(dtype=np.float64) / ordered.size), "Mean")
    return {
        "mean_ms": mean,
        "p50_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "p99_ms": percentile(0.99),
        "throughput_fps": finite_or_fail(1000.0 * batch_size / mean, "Throughput"),
        "samples": int(ordered.size),
    }


def default_tolerance(precision):
    if precision == "FP32":
        return {"atol": 1e-5, "rtol": 1e-4}
    if precision == "BF16":
        return {"atol": 1e-3, "rtol": 1e-2}
    if precision == "FP16":
        return {"atol": 1e-4, "rtol": 1e-3}
    fail("Unsupported tolerance precision: " + precision)


def compare_outputs(expected, actual, threshold=0.99, atol=1e-5, rtol=1e-4):
    """C++ ``compare_output``: per-output metrics + PASS/FAIL."""
    if not (math.isfinite(threshold) and -1 <= threshold < 1):
        fail("Invalid cosine threshold")
    if not (math.isfinite(atol) and atol >= 0 and math.isfinite(rtol) and rtol >= 0):
        fail("Tolerances must be finite and nonnegative")
    if is_real_dtype(element_type_of(expected)) != is_real_dtype(element_type_of(actual)):
        fail("Output real/integer category mismatch")
    if list(expected.shape) != list(actual.shape):
        fail("Output shape mismatch")
    reference = to_numpy(expected)
    output = to_numpy(actual)
    count = int(reference.size)
    real = is_real_dtype(element_type_of(expected))

    r64 = _to_double(reference, element_type_of(expected))
    o64 = _to_double(output, element_type_of(actual))
    finite_mask = np.isfinite(r64) & np.isfinite(o64)
    finite = bool(finite_mask.all())
    nonfinite = int(count - np.count_nonzero(finite_mask))
    mismatches = int(np.count_nonzero(np.where(finite_mask, r64 != o64, False)))
    if real:
        error = np.where(finite_mask, np.abs(o64 - r64), 0.0)
        allowance = atol + rtol * np.abs(r64)
        outside = int(np.count_nonzero(np.where(finite_mask, error > allowance, False)))
        scaled = np.where(finite_mask & (allowance > 0),
                          error / np.maximum(allowance, np.finfo(np.float64).tiny), 0.0)
        required_scale = finite_or_fail(float(scaled.max()) if count else 0.0,
                                        "Required tolerance scale")
        if required_scale and float(np.float64(required_scale)) < required_scale:
            required_scale = math.nextafter(required_scale, math.inf)
    else:
        error = _int_error(reference, output)
        outside = int(np.count_nonzero(np.not_equal(reference, output)))
        mismatches = outside
        required_scale = None
    error = np.where(finite_mask, error, 0.0)

    moments = RunningVariance()
    moments.add(error)
    dot = float(np.dot(np.where(finite_mask, r64, 0.0), np.where(finite_mask, o64, 0.0)))
    xnorm = float(np.sum(np.where(finite_mask, r64 * r64, 0.0)))
    ynorm = float(np.sum(np.where(finite_mask, o64 * o64, 0.0)))
    squared = float(np.sum(np.where(finite_mask, error * error, 0.0)))
    absolute_error = float(np.sum(np.where(finite_mask, error, 0.0)))
    maximum = float(np.max(np.where(finite_mask, error, 0.0))) if count else 0.0

    row = {
        "reference_dtype": element_type_of(expected),
        "dtype": element_type_of(actual),
        "shape": [int(d) for d in actual.shape],
        "elements": count,
        "finite": finite,
        "nonfinite_elements": nonfinite,
        "mismatched_elements": mismatches,
        "atol": atol if real else 0.0,
        "rtol": rtol if real else 0.0,
        "out_of_tolerance_elements": outside,
        "out_of_tolerance_fraction": (outside / count) if count else 0.0,
        "tolerance_passed": outside == 0,
        "cosine_passed": False,
        "exact_equal": nonfinite == 0 and mismatches == 0,
        "cosine_similarity": None,
        "normalized_rmse": None,
        "mean_abs": None,
        "max_abs": None,
        "abs_error_variance": None,
        "required_tolerance_scale": required_scale if real else None,
        "status": "FAIL",
    }
    if not finite:
        return row
    row["mean_abs"] = finite_or_fail(absolute_error / count, "Mean absolute error")
    row["max_abs"] = finite_or_fail(maximum, "Max absolute error")
    row["abs_error_variance"] = finite_or_fail(moments.variance(),
                                               "Absolute error variance")
    if xnorm > 0:
        row["normalized_rmse"] = finite_or_fail(math.sqrt(squared / xnorm), "NRMSE")
    elif squared == 0:
        row["normalized_rmse"] = 0.0
    zero_norm = xnorm == 0 or ynorm == 0
    row["zero_norm"] = zero_norm
    cosine = None
    if not zero_norm:
        cosine = finite_or_fail(
            max(-1.0, min(1.0, dot / math.sqrt(xnorm * ynorm))), "Cosine")
        row["cosine_similarity"] = cosine
    cosine_passed = (mismatches == 0) if zero_norm else (
        cosine is not None and cosine > threshold)
    row["cosine_passed"] = cosine_passed
    passed = (cosine_passed and outside == 0) if real else mismatches == 0
    row["status"] = "PASS" if passed else "FAIL"
    return row


def compare(expected_list, actual_list):
    """C++ ``compare``: aggregate cosine / NRMSE / abs metrics over all outputs."""
    if not expected_list:
        fail("Comparison outputs must not be empty")
    if len(expected_list) != len(actual_list):
        fail("Output count mismatch")
    for index, (expected, actual) in enumerate(zip(expected_list, actual_list)):
        if list(expected.shape) != list(actual.shape):
            fail("Output shape mismatch at port " + str(index))
    dot = 0.0
    reference_norm = 0.0
    actual_norm = 0.0
    squared_error = 0.0
    absolute_error = 0.0
    maximum = 0.0
    count = 0
    finite = True
    moments = RunningVariance()
    for expected, actual in zip(expected_list, actual_list):
        name_e = element_type_of(expected)
        name_a = element_type_of(actual)
        array_r = _to_double(to_numpy(expected), name_e)
        array_a = _to_double(to_numpy(actual), name_a)
        mask = np.isfinite(array_r) & np.isfinite(array_a)
        finite = finite and bool(mask.all())
        error = np.where(mask, np.abs(array_a - array_r), 0.0)
        moments.add(error)
        dot = float(np.longdouble(dot) +
                    np.sum(np.where(mask, array_r * array_a, 0.0), dtype=np.longdouble))
        reference_norm = float(np.longdouble(reference_norm) +
                               np.sum(np.where(mask, array_r * array_r, 0.0),
                                      dtype=np.longdouble))
        actual_norm = float(np.longdouble(actual_norm) +
                            np.sum(np.where(mask, array_a * array_a, 0.0), dtype=np.longdouble))
        squared_error = float(np.longdouble(squared_error) +
                              np.sum(np.where(mask, error * error, 0.0), dtype=np.longdouble))
        absolute_error = float(np.longdouble(absolute_error) +
                               np.sum(np.where(mask, error, 0.0), dtype=np.longdouble))
        if error.size:
            maximum = max(maximum, float(error.max()))
        count += int(error.size)
    if not finite:
        return {"finite": False, "cosine": 0, "normalized_rmse": 0, "mean_abs": 0,
                "max_abs": 0, "abs_error_variance": 0}
    if not (reference_norm > 0 and actual_norm > 0):
        fail("Cosine similarity is undefined for zero-norm outputs")
    cosine = finite_or_fail(max(-1.0, min(1.0, dot / math.sqrt(reference_norm * actual_norm))),
                            "Cosine")
    return {
        "finite": True,
        "cosine": cosine,
        "normalized_rmse": finite_or_fail(math.sqrt(squared_error / reference_norm), "NRMSE"),
        "mean_abs": finite_or_fail(absolute_error / count, "Mean absolute error"),
        "max_abs": finite_or_fail(maximum, "Max absolute error"),
        "abs_error_variance": finite_or_fail(moments.variance(),
                                             "Absolute error variance"),
    }


def zero_tensor_real(tensor):
    name = element_type_of(tensor)
    values = _to_double(to_numpy(tensor), name)
    return bool(np.all(values == 0))


def precision_element_type(precision):
    return _PRECISION_ELEMENT[precision]
