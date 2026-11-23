"""Small helpers shared across the Python ov_bench backend.

Mirrors the free functions used by the C++ engine (JSON/TSV writing with the
same escaping rules, MD5 provenance, checked byte-size math).
"""

import hashlib
import json
import math
import os
import sys


class OVBError(ValueError):
    """User / specification error; maps to exit code 2 (like the C++ binary)."""


class OVRuntimeError(RuntimeError):
    """Execution or I/O error; maps to exit code 1 (like the C++ binary)."""


def fail(message, runtime=False):
    raise (OVRuntimeError if runtime else OVBError)(message)


# ---------------------------------------------------------------------------
# JSON / TSV
# ---------------------------------------------------------------------------

def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, default=list)
        handle.write("\n")


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except OSError as exc:
        raise OVRuntimeError("Cannot open JSON: %s" % path)
    except json.JSONDecodeError as exc:
        raise OVRuntimeError("Cannot parse JSON: %s (%s)" % (path, exc))


def tsv_escape(value):
    text = str(value)
    if not any(ch in text for ch in '\t"\r\n'):
        return text
    text = text.replace('"', '""')
    return '"%s"' % text


def tsv_row(fields):
    return "\t".join(tsv_escape(field) for field in fields) + "\n"


def tsv_json_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

CHUNK = 64 * 1024


def md5_file(path):
    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                digest.update(block)
    except OSError:
        raise OVRuntimeError("Cannot open file for MD5: %s" % path)
    return digest.hexdigest()


def checked_tensor_bytes(itemsize, shape):
    """C++ tensor_bytes(): reject zero dims and 64-bit overflow."""
    if itemsize <= 0:
        fail("Tensor byte size overflow")
    total = itemsize
    for dimension in shape:
        if dimension == 0:
            fail("Tensor shape contains a zero dimension")
        if total > sys.maxsize // dimension:
            fail("Tensor byte size overflow")
        total *= dimension
    return total


def absolute(path):
    return os.path.abspath(path)


def finite_or_fail(value, metric):
    if not math.isfinite(value):
        raise OverflowError("%s is not representable" % metric)
    return value
