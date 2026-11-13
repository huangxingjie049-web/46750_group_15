"""Reproduce all Assignment 2 Q1 results in dependency order."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = ("base_case.py", "calculate_metrics.py", "sensitivity.py", "settlement_analysis.py")


def main():
    for script in STEPS:
        print(f"Running {script}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
