"""Verify shared random inputs and summarize both runtime matrices."""

import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    versions = {}
    hashes = {}
    for version in ("2023.3", "2025.4.1-patched"):
        folder = root / version
        versions[version] = json.loads((folder / "summary.json").read_text())
        hashes[version] = {}
        for path in folder.glob("*/results.json"):
            report = json.loads(path.read_text())
            for case in report["cases"]:
                hashes[version][(case["model"], case["precision"])] = [
                    [(item["name"], item["md5"]) for item in row["inputs"]]
                    for row in case["accuracy"]["records"]
                ]
    left, right = hashes.values()
    shared = left.keys() & right.keys()
    mismatches = [list(key) for key in sorted(shared)
                  if left[key][:min(len(left[key]), len(right[key]))] !=
                  right[key][:min(len(left[key]), len(right[key]))]]
    statuses = [value["status"] for value in versions.values()]
    status = "ERROR" if mismatches or "ERROR" in statuses else ("PASS" if all(
        value == "PASS" for value in statuses) else "FAIL")
    report = {
        "status": status, "versions": versions,
        "shared_input_cases_checked": len(shared), "input_hash_mismatches": mismatches,
        "input_comparison": (
            "Accepted input prefix; rejection sampling may select different subsets; mismatch is not a paired comparison"
            if any("sampling" in value for value in versions.values()) else
            "Shared input prefix; cosine fail-fast may produce different iteration counts"),
        "note": "ERROR includes unsupported models or incomplete runs; FAIL means completed numerical failures.",
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2))
    (root / "status").write_text(status + "\n")


if __name__ == "__main__":
    main()
