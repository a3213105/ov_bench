import unittest
from pathlib import Path
import sys
import csv
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_pred_memory import summarize, summarize_compile_records
from summarize_pred_throughput import summarize as summarize_throughput, write_csv


class MemorySummaryTests(unittest.TestCase):
    def test_throughput_csv_is_comma_delimited_and_keeps_failures_blank(self):
        cases = [
            {"model": "attention", "kind": "release", "precision": "FP32",
             "repeat": 1, "status": "PASS", "result": {"mean_infer_ms": 10.0}},
            {"model": "attention", "kind": "release", "precision": "FP32",
             "repeat": 2, "status": "PASS", "result": {"mean_infer_ms": 20.0}},
            {"model": "attention", "kind": "release", "precision": "BF16",
             "repeat": 1, "status": "FAILED", "result": {}},
        ]
        rows = summarize_throughput(cases)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "throughput.csv"
            write_csv(output, rows)
            text = output.read_text(encoding="utf-8")
            self.assertIn("model,kind,precision", text)
            self.assertNotIn("\t", text)
            parsed = list(csv.DictReader(text.splitlines()))
        fp32 = next(row for row in parsed if row["precision"] == "FP32")
        self.assertEqual(fp32["median_throughput_fps"], "75.0")
        self.assertEqual(fp32["run_1_throughput_fps"], "100.0")
        self.assertEqual(fp32["run_2_throughput_fps"], "50.0")
        failed = next(row for row in parsed if row["precision"] == "BF16")
        self.assertEqual(failed["failed_runs"], "1")
        self.assertEqual(failed["median_throughput_fps"], "")
        self.assertEqual(failed["status"], "FAIL")

    def rows(self):
        rows = []
        for name, iteration, rss in (
            ("process_baseline", 0, 100), ("inputs_loaded", 0, 120),
            ("model_loaded", 0, 130), ("model_compiled", 0, 180),
            ("inference", 1, 200), ("inference", 10, 220),
            ("inference", 20, 224), ("steady", 20, 224),
        ):
            rows.append({
                "stage": name, "iteration": iteration, "Rss_kib": rss * 1024,
                "Pss_kib": (rss - 10) * 1024, "private_kib": (rss - 20) * 1024,
                "Anonymous_kib": (rss - 30) * 1024, "VmHWM_kib": 240 * 1024,
                "elapsed_seconds": iteration + 1,
            })
        return rows

    def result(self):
        return {"backend": "ov", "precision": "BF16", "iterations": 20, "mean_infer_ms": 2}

    def test_exact_stage_deltas_and_units(self):
        result = summarize(self.rows(), self.result())
        self.assertEqual(result["steady_Rss_mib"], 224)
        self.assertEqual(result["steady_minus_baseline_Rss_mib"], 124)
        self.assertEqual(result["steady_minus_inputs_Rss_mib"], 104)
        self.assertEqual(result["steady_minus_first_Rss_mib"], 24)
        self.assertEqual(result["steady_minus_warm_Rss_mib"], 4)
        self.assertEqual(result["steady_minus_tail_Rss_mib"], 4)
        self.assertEqual(result["steady_Pss_mib"], 214)
        self.assertEqual(result["peak_hwm_mib"], 240)
        self.assertEqual(result["peak_sampled_Rss_mib"], 224)

    def test_missing_final_checkpoint_is_error(self):
        with self.assertRaises(ValueError):
            summarize([r for r in self.rows() if r["iteration"] != 20], self.result())

    def test_missing_middle_checkpoint_is_error(self):
        with self.assertRaises(ValueError):
            summarize([r for r in self.rows() if r["iteration"] != 10], self.result())

    def test_duplicate_checkpoint_is_error(self):
        rows = self.rows()
        rows.insert(6, dict(rows[5]))
        with self.assertRaises(ValueError):
            summarize(rows, self.result())

    def test_negative_growth_preserved(self):
        rows = self.rows()
        rows[-1]["Rss_kib"] = 210 * 1024
        result = summarize(rows, self.result())
        self.assertEqual(result["steady_minus_warm_Rss_mib"], -10)

    def test_post_measurement_graph_allocations_excluded_from_peak(self):
        rows = self.rows()
        graph = dict(rows[-1], stage="after_runtime_inspection", Rss_kib=999 * 1024)
        rows.append(graph)
        self.assertEqual(summarize(rows, self.result())["peak_sampled_Rss_mib"], 224)

    def test_compile_metrics_and_actual_runtime(self):
        rows = self.rows()
        loaded = dict(rows[1], stage="model_loaded", allocator_uordblks_bytes=1024 * 1024)
        compiled = dict(loaded, stage="model_compiled", Rss_kib=150 * 1024,
                        allocator_uordblks_bytes=4 * 1024 * 1024)
        rows[2:4] = [loaded, compiled]
        result = dict(self.result(), openvino_version="patched-build", compile_ms=42.5)
        summary = summarize(rows, result)
        self.assertEqual(summary["runtime_version"], "patched-build")
        self.assertEqual(summary["compile_ms"], 42.5)
        self.assertEqual(summary["compile_delta_Rss_mib"], 30)
        self.assertEqual(summary["compile_delta_uordblks_mib"], 3)

    def test_ort_does_not_require_ov_compile_stages(self):
        result = dict(self.result(), backend="ort", onnxruntime_version="actual-ort")
        self.assertEqual(summarize(self.rows(), result)["runtime_version"], "actual-ort")

    def compile_records(self):
        names = ["C0_compile_entry", "C1_model_cloned", "C2_transformations_complete",
                 "C4_graph_initialized", "C5_primitives_created", "C6_cleanup_complete",
                 "C3_compiled_model_constructed"]
        rows = []
        for index, name in enumerate(names):
            row = {"schema_version": 1, "stage": name,
                   "timestamp_steady_ns": index * 1000000, "Rss_kib": 100 + index}
            if index in (3, 4, 5):
                row["graph_id"] = 123
            rows.append(row)
        return rows

    def test_compile_ledger_actual_order_and_no_fake_ownership(self):
        result = summarize_compile_records(self.compile_records())
        self.assertEqual(result["complete_graphs"], 1)
        self.assertEqual(len(result["waterfall"]), 6)
        self.assertEqual(result["waterfall"][0]["delta_Rss_kib"], 1)
        self.assertIsNone(result["accounted_fraction"])

    def test_compile_ledger_missing_graph_cleanup_is_error(self):
        rows = [row for row in self.compile_records() if row["stage"] != "C6_cleanup_complete"]
        with self.assertRaises(ValueError):
            summarize_compile_records(rows)

    def test_duplicate_compile_is_not_an_isolated_case(self):
        rows = self.compile_records()
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            summarize_compile_records(rows)


if __name__ == "__main__":
    unittest.main()
