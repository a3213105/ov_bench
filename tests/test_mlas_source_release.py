import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_mlas_source_release as experiment


class MlasSourceReleaseTests(unittest.TestCase):
    def run_experiment(self, precision="FP32", *, no_mlas=False, missing_trace=False,
                       mismatch=False, crash=False, wrong_hint=False, artifacts=False,
                       eligible=True, incomplete_release=False, graph_disconnect=False):
        commands = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"

            def native(command, *, env, **kwargs):
                commands.append(command)
                directory = Path(command[-1])
                directory.mkdir()
                release = env.get("OV_CPU_EXPERIMENTAL_DISCONNECT_OV_GRAPH" if graph_disconnect
                                  else "OV_CPU_EXPERIMENTAL_RELEASE_FC_COMPILE_ARTIFACTS" if artifacts
                                  else "OV_CPU_EXPERIMENTAL_RELEASE_MLAS_SOURCE") == "1"
                if release and crash:
                    return SimpleNamespace(returncode=-11)
                dynamic = "OVBENCH_THREE_BATCHES" in env
                iterations = 3 if dynamic else 10
                result = {
                    "status": "PASS", "precision": command[5],
                    "finite_outputs_checked_each_iteration": True,
                    "effective_config": {"INFERENCE_PRECISION_HINT": "wrong" if wrong_hint else
                                         {"FP32": "f32", "FP16": "f16", "BF16": "bf16"}[command[5]]},
                    "sample_output_md5": [["changed" if mismatch and release else "same"]
                                          for _ in range(iterations)],
                    "inference_shapes": [{"iteration": batch, "inputs": {"input": [batch, 2]},
                                          "outputs": [[batch, 2]]} for batch in range(1, 4)]
                                        if dynamic else [],
                }
                experiment.save(directory / "result.json", result)
                memory = 1024 if release else 2048
                (directory / "stages.jsonl").write_text(json.dumps({
                    "stage": "model_compiled", "iteration": 0,
                    "Rss_kib": memory, "Pss_kib": memory, "uss_kib": memory}) + "\n")
                trace = env["OV_CPU_COMPILE_MEMORY_STATS_PATH"]
                if not missing_trace:
                    Path(trace + ".mlas_weights.jsonl").write_text("" if no_mlas else '{"node":"fc"}\n')
                    if graph_disconnect:
                        event = {"stage": "complete" if release else "inventory",
                                 "nodes": 5, "input_edges": 4, "control_edges": 1,
                                 "disconnected_edges": (4 if incomplete_release else 5) if release else 0,
                                 "expired_nodes": 2 if release else 0, "surviving_nodes": 3 if release else 5,
                                 "live_delta_bytes": -128 if release else 0}
                        Path(trace + ".graph_disconnect.jsonl").write_text(json.dumps(event) + "\n")
                    if artifacts:
                        events = [{"stage": "complete" if release and eligible else "inventory",
                                   "node": "fc", "artifact": kind, "eligible": eligible,
                                   "objects": 1, "live_delta_bytes": -128 if release else 0}
                                  for kind in ("factory", "node_post_ops")]
                        if release and incomplete_release:
                            events.pop()
                        Path(trace + ".fc_compile_artifacts.jsonl").write_text(
                            "".join(json.dumps(event) + "\n" for event in events))
                if release:
                    Path(trace + ".mlas_source_release.jsonl").write_text(json.dumps({
                        "stage": "complete", "buffers": 1, "released_logical_bytes": 1024}) + "\n")
                return SimpleNamespace(returncode=0)

            arguments = ["check_mlas_source_release.py", "--binary", "/fake/pred_memory",
                         "--model-root", "/fake/models", "--inputs", "/fake/inputs",
                         "--model", "ped", "--precision", precision, "--output", str(output)]
            if artifacts:
                arguments += ["--experiment", "fc-compile-artifacts"]
            if graph_disconnect:
                arguments += ["--experiment", "graph-disconnect"]
            with patch.object(sys, "argv", arguments), patch.object(
                    experiment.subprocess, "run", side_effect=native), contextlib.redirect_stdout(io.StringIO()):
                code = experiment.main()
            return code, json.loads((output / "summary.json").read_text()), commands

    def test_all_precisions_use_same_precision_release_baseline(self):
        for precision in ("FP32", "FP16", "BF16"):
            with self.subTest(precision=precision):
                code, summary, commands = self.run_experiment(precision)
                self.assertEqual(code, 0)
                self.assertEqual(summary["status"], "PASS")
                self.assertEqual(summary["precision"], precision)
                self.assertEqual(len(commands), 4)
                self.assertTrue(all(command[5] == precision for command in commands))
                for comparison in summary["comparisons"].values():
                    self.assertTrue(comparison["hashes_match"])
                    self.assertTrue(comparison["shapes_match"])
                    delta = comparison["stage_deltas_release_minus_baseline"][0]
                    self.assertEqual(delta["rss_delta_mib"], -1)

    def test_no_mlas_runs_both_baselines_but_does_not_claim_release(self):
        code, summary, commands = self.run_experiment("BF16", no_mlas=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "NOT_APPLICABLE")
        self.assertEqual(len(commands), 2)
        self.assertEqual([command[-2] for command in commands], ["10", "3"])
        for comparison in summary["comparisons"].values():
            self.assertEqual(comparison["status"], "NOT_APPLICABLE")
            self.assertFalse(comparison["release_confirmed"])

    def test_missing_trace_is_failure_not_no_mlas(self):
        code, summary, commands = self.run_experiment("FP16", missing_trace=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(len(commands), 4)
        self.assertIn("validation_error", summary["cases"]["baseline_fixed"])

    def test_output_mismatch_fails(self):
        code, summary, _ = self.run_experiment("FP16", mismatch=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["comparisons"]["fixed"]["hashes_match"])

    def test_crash_is_recorded_and_other_cases_still_run(self):
        code, summary, commands = self.run_experiment(crash=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 4)
        self.assertEqual(summary["cases"]["release_fixed"]["signal"], "SIGSEGV")

    def test_wrong_effective_hint_fails(self):
        code, summary, _ = self.run_experiment("BF16", wrong_hint=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")

    def test_artifacts_release_runs_without_mlas(self):
        for precision in ("FP16", "BF16"):
            with self.subTest(precision=precision):
                code, summary, commands = self.run_experiment(precision, artifacts=True, no_mlas=True)
                self.assertEqual(code, 0)
                self.assertEqual(summary["status"], "PASS")
                self.assertEqual(len(commands), 4)
                self.assertEqual(summary["cases"]["release_fixed"]["release_live_delta_bytes"], -256)

    def test_artifacts_ineligible_executor_is_not_deleted(self):
        code, summary, commands = self.run_experiment(artifacts=True, eligible=False)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "NOT_APPLICABLE")
        self.assertEqual(len(commands), 2)

    def test_artifacts_missing_log_fails(self):
        code, summary, _ = self.run_experiment("FP16", artifacts=True, missing_trace=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")

    def test_artifacts_incomplete_deletion_fails(self):
        code, summary, _ = self.run_experiment("BF16", artifacts=True, incomplete_release=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["cases"]["release_fixed"]["release_confirmed"])

    def test_graph_disconnection_checks_output_and_all_edges(self):
        code, summary, commands = self.run_experiment("BF16", graph_disconnect=True, no_mlas=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 4)
        self.assertEqual(summary["experiment"], "graph-disconnect")
        self.assertTrue(summary["cases"]["release_fixed"]["release_confirmed"])

    def test_graph_partial_disconnection_fails(self):
        code, summary, _ = self.run_experiment("FP16", graph_disconnect=True, incomplete_release=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["cases"]["release_fixed"]["release_confirmed"])

    def test_graph_crash_is_captured(self):
        code, summary, commands = self.run_experiment("BF16", graph_disconnect=True, crash=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 4)
        self.assertEqual(summary["cases"]["release_fixed"]["signal"], "SIGSEGV")

    def test_graph_missing_inventory_fails(self):
        code, summary, _ = self.run_experiment("FP16", graph_disconnect=True, missing_trace=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
