"""Summarize independent model runs without hiding numerical failures or errors."""

import csv
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    cases = []
    errors = []
    search_mode = False
    sampling_policies = {}
    for exit_path in sorted((root / "logs").glob("*.exit_code")):
        name = exit_path.name.removesuffix(".exit_code")
        code = int(exit_path.read_text())
        report_path = root / name / "results.json"
        if not report_path.is_file():
            errors.append({"model": name, "exit_code": code, "error": "Missing results.json"})
            continue
        report = json.loads(report_path.read_text())
        if report.get("sampling", {}).get("enabled"):
            sampling_policies[name] = report["sampling"]
        search_mode |= report["mode"] == "tolerance-search"
        if code not in (0, 1) or report["status"] not in ("PASS", "FAIL"):
            errors.append({"model": name, "exit_code": code, "error": report.get("error", report["status"])})
        for case in report["cases"]:
            records = case["accuracy"]["records"]
            hashes = [tuple(x["md5"] for x in row["inputs"]) for row in records]
            search = case.get("tolerance_search", {})
            early_failure = search.get("status") == "FAIL" and "scale" in search and search["scale"] is None
            requested = report["iterations"]
            sampling = report.get("sampling", {}).get("models", {}).get(case["model"])
            complete = (len(records) == requested and len(set(hashes)) == requested) or (
                early_failure and 0 < len(records) <= requested and len(set(hashes)) == len(records))
            if sampling is not None:
                complete &= (sampling["accepted"] == requested and
                             sampling["attempted"] == sampling["accepted"] + sampling["rejected"])
            cases.append({
                "model": case["model"], "precision": case["precision"],
                "backend": case["reference"]["backend"], "iterations": len(records),
                "unique_input_sets": len(set(hashes)), "complete": complete,
                "passed": case["accuracy"]["passed_images"], "status": case["status"],
                "threads": case["effective_config"]["INFERENCE_NUM_THREADS"],
                "result": str(report_path),
                "scale": search.get("scale"), "atol": search.get("atol"), "rtol": search.get("rtol"),
                "search_reason": search.get("reason", ""),
                "sampling_enabled": sampling is not None,
                "attempted_inputs": sampling["attempted"] if sampling is not None else len(records),
                "rejected_inputs": sampling["rejected"] if sampling is not None else 0,
            })
    expected_cases = 10 if search_mode else 30
    if len(cases) != expected_cases or any(not c["complete"] for c in cases):
        errors.append({"error": f"Incomplete matrix: expected {expected_cases} complete cases, got {len(cases)} cases"})
    status = "ERROR" if errors else ("PASS" if all(c["status"] == "PASS" for c in cases) else "FAIL")
    payload = {"status": status, "cases": cases, "errors": errors}
    if sampling_policies:
        payload["sampling"] = sampling_policies
    (root / "summary.json").write_text(json.dumps(payload, indent=2))
    keys = ["model", "precision", "backend", "iterations", "unique_input_sets", "passed", "status", "threads"]
    if search_mode:
        keys += ["scale", "atol", "rtol", "search_reason"]
    if sampling_policies:
        keys += ["sampling_enabled", "attempted_inputs", "rejected_inputs"]
    with (root / "summary.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(keys)
        writer.writerows([[case[key] for key in keys] for case in cases])
    (root / "status").write_text(status + "\n")


if __name__ == "__main__":
    main()
