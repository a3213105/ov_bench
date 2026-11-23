import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "summarize_random_accuracy.py"


class RandomAccuracySummaryTests(unittest.TestCase):
    def test_matrix_retains_sampling_policy_and_requested_accepted_count(self):
        with tempfile.TemporaryDirectory(prefix="sampling-summary-") as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            for index in range(10):
                model = f"model_{index}"
                (root / model).mkdir()
                (root / "logs" / f"{model}.exit_code").write_text("1")
                records = [{"iteration": attempt, "inputs": [{"md5": str(attempt)}]}
                           for attempt in (1, 3)]
                report = {
                    "iterations": 2, "mode": "random-accuracy", "status": "FAIL",
                    "sampling": {
                        "enabled": True, "selection_bias": "Conditional random sample",
                        "models": {model: {"accepted": 2, "rejected": 2, "attempted": 4}},
                    },
                    "cases": [{
                        "model": model, "precision": precision, "status": "FAIL",
                        "reference": {"backend": "OpenVINO FP32"},
                        "effective_config": {"INFERENCE_NUM_THREADS": "2"},
                        "accuracy": {"records": records, "passed_images": 0},
                    } for precision in ("FP32", "BF16", "FP16")],
                }
                (root / model / "results.json").write_text(json.dumps(report))
            subprocess.run([sys.executable, str(SCRIPT), str(root)], check=True)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["status"], "FAIL")
            self.assertEqual(summary["errors"], [])
            self.assertEqual(len(summary["sampling"]), 10)
            for case in summary["cases"]:
                self.assertTrue(case["complete"])
                self.assertTrue(case["sampling_enabled"])
                self.assertEqual((case["iterations"], case["attempted_inputs"], case["rejected_inputs"]),
                                 (2, 4, 2))
            with (root / "summary.tsv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream, delimiter="\t"))["rejected_inputs"], "2")
            path = root / "model_0" / "results.json"
            incomplete = json.loads(path.read_text())
            incomplete["sampling"]["models"]["model_0"]["accepted"] = 1
            path.write_text(json.dumps(incomplete))
            subprocess.run([sys.executable, str(SCRIPT), str(root)], check=True)
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
