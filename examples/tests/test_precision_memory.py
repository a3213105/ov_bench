import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_precision_memory import memory_snapshot


class MemorySnapshotTests(unittest.TestCase):
    def test_reads_real_rollup_and_status(self):
        memory = memory_snapshot(os.getpid())
        self.assertGreater(memory["Rss_kib"], 0)
        self.assertGreater(memory["Pss_kib"], 0)
        self.assertGreater(memory["VmHWM"], 0)
        self.assertGreaterEqual(memory["Threads"], 1)

    def test_parses_units_and_ignores_mapping_header(self):
        rollup = "0000-ffff ---p 00000000 00:00 0 [rollup]\nRss: 120 kB\nPss: 100 kB\nAnonymous: 60 kB\n"
        status = "Name:\tworker\nVmRSS:\t120 kB\nVmHWM:\t200 kB\nThreads:\t6\n"
        with patch.object(Path, "read_text", side_effect=[rollup, status]):
            memory = memory_snapshot(123)
        self.assertEqual(memory["Rss_kib"], 120)
        self.assertEqual(memory["Pss_kib"], 100)
        self.assertEqual(memory["Anonymous_kib"], 60)
        self.assertEqual(memory["VmHWM"], 200)
        self.assertEqual(memory["Threads"], 6)

    def test_missing_process_is_not_silently_defaulted(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError), self.assertRaises(FileNotFoundError):
            memory_snapshot(123)


if __name__ == "__main__":
    unittest.main()
