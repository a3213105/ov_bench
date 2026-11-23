"""Verify, back up and atomically deploy a CPU-only OpenVINO backport."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "runtime", "binary", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.runtime / "libs/libopenvino_intel_cpu_plugin.so"
    original = args.output / "libopenvino_intel_cpu_plugin.original.so"
    patched = args.output / "libopenvino_intel_cpu_plugin.patched.so"
    shutil.copy2(target, original)
    shutil.copy2(args.candidate, patched)
    environment = dict(os.environ, LD_LIBRARY_PATH=str(args.runtime / "libs"))

    def guard(plugin, log):
        with log.open("w") as stream:
            subprocess.run([str(args.binary), "--mode", "gather-guard",
                            "--cpu-plugin", str(plugin.resolve())],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    guard(patched, args.output / "candidate-guard.log")
    temporary = target.with_name(target.name + f".deploy-{os.getpid()}")
    replaced = False
    try:
        shutil.copy2(patched, temporary)
        if digest(temporary) != digest(patched):
            raise RuntimeError("Candidate copy checksum mismatch")
        os.replace(temporary, target)
        replaced = True
        guard(target, args.output / "deployed-guard.log")
        import openvino as ov
        core = ov.Core()
        versions = {name: str(version) for name, version in core.get_versions("CPU").items()}
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "libopenvino" in line})
        if str(target.resolve()) not in libraries:
            raise RuntimeError(f"Deployed CPU plugin was not loaded: {libraries}")
        report = {"status": "DEPLOYED", "core": ov.get_version(), "cpu_versions": versions,
                  "target": str(target), "original_sha256": digest(original),
                  "patched_sha256": digest(target), "loaded_libraries": libraries,
                  "backup": str(original), "scope": "CPU plugin only; other runtime files unchanged",
                  "source_base": "82bbf0292c5bd490bc25a383e63d6461d3ee05c5",
                  "backport": "7ba34b7ae0efc5d390970054230decd0a6ca1349"}
        (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        if replaced:
            shutil.copy2(original, temporary)
            os.replace(temporary, target)
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
