"""Figures must be drawn from solved data, without report image files."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from src.experiments import run_task

ROOT = Path(__file__).resolve().parents[1]


# Default CLI must solve and draw with no original artwork available.
def test_default_cli_solves_and_draws_without_report_images(tmp_path):
    assert not list((ROOT / "report_reference").rglob("*.png"))
    assert not list((ROOT / "report_reference").rglob("*.jpg"))
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--task",
            "1f",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads((tmp_path / "report/manifest.json").read_text())
    assert m["all_checked_numbers_match"] and m["figure_count"] == 2
    for f in m["figures"]:
        with Image.open(tmp_path / "report" / f["path"]) as image:
            assert image.size == (1800, 1800)
        assert f["method"] == "Matplotlib rendering from computed numerical outputs"
        assert f["sha256"] != f["reference_sha256"] and f["source_files"]


# Changed plotted decisions must change the image and its captured artists.
def test_report_renderer_uses_current_csv_decisions(tmp_path):
    from src.report_plotting import render_report_figures

    run_task("1f", tmp_path / "1f", plots=False)
    a = render_report_figures(tmp_path, tasks=["1f"])
    before = hashlib.sha256(
        (tmp_path / "report" / a[0]["path"]).read_bytes()
    ).hexdigest()
    p = tmp_path / "1f/runs/Q1_caseA/hourly.csv"
    h = pd.read_csv(p)
    h.loc[h.hour == 3, "load"] += 0.25
    h.to_csv(p, index=False)
    b = render_report_figures(tmp_path, tasks=["1f"])
    after = hashlib.sha256(
        (tmp_path / "report" / b[0]["path"]).read_bytes()
    ).hexdigest()
    assert before != after
    data = json.loads((tmp_path / "report" / b[0]["plot_data"]).read_text())
    bars = next(x for x in data["axes"][2]["bars"] if x["label"] == "load")
    assert bars["height"][3] == pytest.approx(6.25)


# Numeric failures must stop before report drawing begins.
def test_report_detects_changed_solver_results(tmp_path):
    from src.report import verify_report

    run_task("1f", tmp_path / "1f", plots=False)
    p = tmp_path / "1f/runs/Q1_caseA/metrics.json"
    j = json.loads(p.read_text())
    j["objective"] += 0.1
    p.write_text(json.dumps(j))
    with pytest.raises(RuntimeError, match="numerical"):
        verify_report(tmp_path, tasks=["1f"])
    assert not (tmp_path / "report/figures").exists()


# No numbered report figure may be silently omitted from the generation plan.
def test_report_contract_covers_all_numbered_figures():
    c = json.loads((ROOT / "report_reference/contract.json").read_text())
    assert [f["figure"] for f in c["figures"]] == list(range(1, 27))
    assert len(c["checks"]) == 426


# Single-task drawing must not require unrelated questions to have run.
def test_single_q2e_task_generates_its_three_report_figures(tmp_path):
    from src.report_plotting import render_report_figures

    run_task("2e", tmp_path / "2e", plots=False)
    figs = render_report_figures(tmp_path, tasks=["2e"])
    assert [f["figure"] for f in figs] == [9, 10, 11]
    assert all((tmp_path / "report" / f["path"]).exists() for f in figs)


# The explicit numerical-only flag should leave no image in a clean output folder.
def test_no_plots_disables_rendering_in_a_clean_directory(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--task",
            "1f",
            "--no-plots",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert not list(tmp_path.rglob("*.png"))
    assert (
        json.loads((tmp_path / "report/manifest.json").read_text())["figure_count"] == 0
    )


# Report response hours and actual solved data must agree at the plotting boundary.
def test_linear_response_plots_report_hours_and_actual_solutions(tmp_path):
    from src.report_plotting import Runs, coefficient_response
    import matplotlib.pyplot as plt

    run_task("2b", tmp_path / "2b", plots=False)
    fig = coefficient_response(Runs(tmp_path), "2b", "linear", [7, 9, 11, 18])
    ax = fig.axes[0]
    assert [line.get_label() for line in ax.lines] == [
        "Hour 7",
        "Hour 9",
        "Hour 11",
        "Hour 18",
    ]
    values = pd.read_csv(tmp_path / "2b/sweep_metrics.csv").cL
    expected = [
        pd.read_csv(tmp_path / f"2b/runs/cL_{v:g}/hourly.csv")
        .set_index("hour")
        .loc[7, "load"]
        for v in values
    ]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
    plt.close(fig)
