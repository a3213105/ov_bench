"""Stability filename encoding + cross-directory comparison.

Python port of ``src/stability.cpp`` (``ovbench::stability_*``).
"""

import os
import re
import secrets

from .utils import OVRuntimeError, fail
from .utils import md5_file as _md5_file


def _hex(value):
    return value.hex()


def _unhex(value):
    if not value or len(value) % 2:
        fail("Invalid encoded filename field")
    return bytes.fromhex(value)


def _encode(value):
    out = []
    for byte in value.encode("latin-1", "replace"):
        c = chr(byte)
        if c.isalnum() or c in ".-":
            out.append(c)
        else:
            out.append("%" + format(byte, "02x"))
    return "".join(out)


def _decode(value):
    out = bytearray()
    i = 0
    while i < len(value):
        if value[i] == "%":
            if i + 2 >= len(value):
                fail("Invalid percent escape")
            out += _unhex(value[i + 1:i + 3])
            i += 3
        else:
            out.append(ord(value[i]))
            i += 1
    return out.decode("latin-1", "replace")


def _random_id():
    return _hex(secrets.token_bytes(8))


def _number(value, allow_zero):
    try:
        n = int(value)
    except ValueError:
        fail("Invalid stability round/port")
    if n < 0 or (not allow_zero and n == 0):
        fail("Invalid stability round/port")
    return n


_PROCESS_RE = re.compile(r"[0-9]+-[0-9a-f]{16}")


def process_identity():
    return "%d-%s" % (os.getpid(), _random_id())


def stability_filename(identity):
    process = identity["process"]
    image = identity["image"]
    model = identity["model"]
    precision = identity["precision"]
    configuration = identity["configuration"]
    round_ = identity["round"]
    port = identity["port"]
    if not image or not model or not configuration or round_ == 0 or \
            not _PROCESS_RE.fullmatch(process) or \
            precision not in ("FP32", "BF16", "FP16"):
        fail("Invalid stability output identity")
    name = ("stb2__" + _encode(image) + "__" + precision + "__" + process +
            "__" + str(round_) + "__" + _encode(model) + "__" + str(port) +
            "__" + _encode(configuration) + "__" + _random_id() + ".bin")
    if len(name.encode("utf-8")) > 255:
        fail("Encoded stability filename exceeds 255 bytes")
    return name


_FILENAME_RE = re.compile(
    r"^stb2__((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__(FP32|BF16|FP16)__"
    r"([0-9]+-[0-9a-f]{16})__([0-9]+)__((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__([0-9]+)__"
    r"((?:[A-Za-z0-9.-]|%[0-9a-f]{2})+)__[0-9a-f]{16}\.bin$")


def parse_stability_filename(filename):
    match = _FILENAME_RE.fullmatch(filename)
    if not match:
        fail("Malformed stability filename: " + filename)
    return {
        "image": _decode(match.group(1)),
        "model": _decode(match.group(5)),
        "precision": match.group(2),
        "configuration": _decode(match.group(7)),
        "process": match.group(3),
        "round": _number(match.group(4), False),
        "port": _number(match.group(6), True),
    }


def compare_stability_directories(directories):
    if not directories:
        fail("At least one result directory is required")
    groups = {}
    seen = set()
    processes = set()
    ignored = 0
    for directory in directories:
        directory = str(directory)
        if not os.path.isdir(directory):
            fail("Not a directory: " + directory)
        found = 0
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if not filename.endswith(".bin"):
                    continue
                if filename.startswith("stb1__"):
                    fail("Unsupported legacy stability filename (expected stb2): " + filename)
                if not filename.startswith("stb2__"):
                    ignored += 1
                    continue
                found += 1
                path = os.path.realpath(os.path.join(root, filename))
                if path in seen:
                    continue
                seen.add(path)
                identity = parse_stability_filename(filename)
                processes.add(identity["process"])
                key = (identity["model"], identity["image"], identity["precision"],
                       identity["port"], identity["configuration"])
                groups.setdefault(key, []).append({
                    "file": path, "process": identity["process"], "round": identity["round"],
                    "md5": _md5_file(path), "bytes": os.path.getsize(path)})
        if not found:
            raise OVRuntimeError("No named stability binaries found in: " + directory)
    if len(processes) < 2:
        raise OVRuntimeError("Cross-process comparison requires at least two distinct "
                             "process identities")
    results = []
    passed = True
    for (model, image, precision, port, configuration), files in sorted(groups.items()):
        rounds_by_process = {}
        duplicate = False
        consistent = True
        base_md5 = files[0]["md5"]
        base_bytes = files[0]["bytes"]
        for row in files:
            rounds = rounds_by_process.setdefault(row["process"], set())
            if row["round"] in rounds:
                duplicate = True
            rounds.add(row["round"])
            consistent = consistent and row["md5"] == base_md5 and row["bytes"] == base_bytes
        missing = [process for process in sorted(processes) if process not in rounds_by_process]
        complete = not missing
        if rounds_by_process:
            expected_rounds = min(rounds_by_process.values(), key=lambda v: sorted(v)[0])
            if expected_rounds and (min(expected_rounds) != 1 or max(expected_rounds) != len(expected_rounds)):
                complete = False
            for values in rounds_by_process.values():
                if values != expected_rounds:
                    complete = False
        else:
            complete = False
        ok = consistent and complete and not duplicate
        passed = passed and ok
        results.append({
            "model": model, "image": image, "precision": precision,
            "output_port": port, "configuration": configuration,
            "processes": len(rounds_by_process), "files": files,
            "md5_consistent": consistent, "missing_processes": missing,
            "rounds_complete": complete, "duplicate_process_round": duplicate,
            "status": "PASS" if ok else "FAIL",
        })
    return {
        "status": "PASS" if passed else "FAIL",
        "processes": sorted(processes),
        "binary_files": len(seen),
        "ignored_binary_files": ignored,
        "groups": results,
        "scope": "Filename-based cross-process byte consistency; not a numerical "
                 "accuracy or finite-value check.",
    }
