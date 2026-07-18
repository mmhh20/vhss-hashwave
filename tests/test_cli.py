import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
ENV = {**os.environ, "PYTHONPATH": ROOT}


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "hashwave", *args],
            cwd=ROOT,
            env=ENV,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return json.loads(result.stdout)

    def test_memory_demo_cli(self):
        payload = self.run_cli("memory-demo")
        self.assertEqual(payload["prediction"], "token")

    def test_demo_cli(self):
        payload = self.run_cli(
            "demo", "--generations", "20", "--beam", "16", "--per-parent", "32", "--seed", "cli", "--json"
        )
        result = payload["hash_evolution"]
        self.assertGreaterEqual(result["unique_evaluations"], 1)
        self.assertGreaterEqual(result["train_mae"], 0)

    def test_version_cli(self):
        result = subprocess.run(
            [sys.executable, "-m", "hashwave", "--version"],
            cwd=ROOT, env=ENV, text=True, capture_output=True, timeout=30, check=True,
        )
        self.assertEqual(result.stdout.strip(), "VHSS 1.2.0")


if __name__ == "__main__":
    unittest.main()
