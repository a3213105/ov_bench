import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_runtime_memory import summarize


class RuntimeMemoryTests(unittest.TestCase):
    def row(self, stage, depth, live, node=0):
        return {
            "graph_id": 1, "inference_id": 10, "stage": stage, "depth": depth,
            "node_id": node, "node": "fc" if node else "", "type": "FullyConnected" if node else "Request",
            "kind": "allocation", "live_delta_bytes": live, "mmap_delta_bytes": 0,
            "free_delta_bytes": -live, "rss_before_kib": 1024, "rss_after_kib": 2048,
        }

    def test_nested_steps_not_double_counted_and_buffers_deduplicated(self):
        rows = [self.row("request_total", 0, 100), self.row("request.graph_infer", 0, 90),
                self.row("graph.allocate_memory", 1, 10), self.row("node.prepare_params", 1, 75, 4),
                self.row("dnnl.prepare_weights", 2, 70, 4)]
        event = {
            "graph_id": 1, "inference_id": 10, "stage": "dnnl.weights_result", "depth": 3,
            "node_id": 4, "node": "fc", "type": "FullyConnected", "kind": "buffer",
            "source_address": 100, "source_bytes": 128, "target_address": 200,
            "target_bytes": 64, "created": True, "private_cache_entries": 1,
        }
        rows.extend([event, dict(event, stage="dnnl.weights_private_hit", created=False)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = summarize(path, 1)
            self.assertEqual(summary["intervals"][0]["request_unattributed_bytes"], 10)
            self.assertEqual(summary["intervals"][0]["graph_unattributed_bytes"], 5)
            self.assertEqual(summary["buffers"][0]["observed_unique_weight_bytes"], 64)
            self.assertEqual(summary["buffers"][0]["private_cache_hits"], 1)
            self.assertEqual(summary["buffers"][0]["weight_created_events"], 1)
            self.assertEqual(summary["intervals"][0]["request_rss_delta_mib"], 1)
            self.assertTrue(Path(str(path) + ".top_nodes.tsv").exists())

    def test_missing_completed_request_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(RuntimeError, "completed requests"):
                summarize(path, 3)

    def test_missing_graph_phase_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.jsonl"
            path.write_text(json.dumps(self.row("request_total", 0, 0)) + "\n")
            with self.assertRaisesRegex(RuntimeError, "graph-infer phase"):
                summarize(path, 1)


if __name__ == "__main__":
    unittest.main()
