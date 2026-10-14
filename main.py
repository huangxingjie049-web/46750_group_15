"""The sole root entry point: reproduce report experiments or run tests."""

import argparse
from pathlib import Path
import subprocess
import sys

# Anchor default paths to the package so running from another directory is safe.
ROOT = Path(__file__).resolve().parent


# Parse the command line, select report tasks, and run them in a fixed order.
def main(argv=None):
    """Dispatch named tasks; outputs are anchored to the repository by default."""
    from src.experiments import TASKS, run_task
    from src.report import verify_report
    from src.report_plotting import render_report_figures
    from src.model import write_json

    # Expose task selection, output location, plot control, and smoke-test mode.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        choices=["all", "test"] + list(TASKS),
        default="all",
        help="Default: all report experiments.",
    )
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Parent directory; each task gets a separate subdirectory.",
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Save numerical outputs only."
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: sparse battery scans without plateau search; NOT report reproduction.",
    )
    args = p.parse_args(argv)
    # Listing tasks is read-only and exits before any scenario is solved.
    if args.list_tasks:
        for k, v in TASKS.items():
            print(f"{k:16} {v}")
        return
    # Run pytest with the same Python interpreter selected by the user.
    if args.task == "test":
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT
            )
        )
    # Avoid duplicate aliases in the full report run; calibration is nested in battery tasks.
    tasks = (
        ["1f", "2b", "2c", "2d", "2e", "3e", "3f", "3g", "3g-sensitivity"]
        if args.task == "all"
        else [args.task]
    )
    # Execute tasks sequentially, with a separate subdirectory for each one.
    for task in tasks:
        print(f"Running {task}: {TASKS[task]}", flush=True)
        result = run_task(
            task,
            args.output_dir / task,
            plots=(task == "3g-pi" and not args.no_plots),
            quick=args.quick,
        )
        print(
            f"Validated {task}; results: {(args.output_dir/task).resolve()}", flush=True
        )
    # Generate report pictures from this run's numerical files, after all tasks finish.
    if tasks != ["3g-pi"]:
        report = verify_report(args.output_dir, tasks=tasks, quick=args.quick)
        if not args.no_plots:
            figures = render_report_figures(
                args.output_dir, tasks=tasks, quick=args.quick
            )
            report["figures"] = figures
            report["figure_count"] = len(figures)
            write_json(args.output_dir / "report" / "manifest.json", report)
        print(
            f"Report check: {report['checked_count']} numerical checkpoints matched; "
            f"{report['figure_count']} report figures freshly rendered.",
            flush=True,
        )
    print(
        f"All selected experiments completed. Outputs: {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
