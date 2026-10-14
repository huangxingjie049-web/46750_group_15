"""Experiment design and integration checks, independent of plotting aesthetics."""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from src.experiments import TASKS, run_task, mean_preserving_prices, calibrate_pi
from src.data_loader import load_question


# Check fixed means and spread scaling for the declared alpha scenarios.
def test_volatility_preserves_mean_and_order():
    p = load_question("Q3_battery").energy_price
    # Check flat, reduced, baseline, and increased variability with the supplied price profile.
    for a in [0, 0.5, 1, 1.5]:
        x = mean_preserving_prices(p, a)
        assert x.mean() == pytest.approx(p.mean())
        assert np.ptp(x) == pytest.approx(a * np.ptp(p))
    assert np.all(mean_preserving_prices(p, 0) == p.mean())


# Check the calibrated value and its stability across finite-difference steps.
def test_calibration_uses_forecasts_and_stable_local_value(tmp_path):
    value, table = calibrate_pi(tmp_path, plots=False)
    assert value == pytest.approx(1.473684210526, abs=2e-6)
    assert np.ptp(table["pi"]) < 1e-5


# Exercise each task pipeline and inspect its required numerical outputs.
@pytest.mark.parametrize("task", ["1f", "2b", "2c", "2d", "2e", "3d", "3e", "3f", "3g"])
def test_each_task_runs_validates_and_exports(task, tmp_path):
    out = tmp_path / task
    result = run_task(task, out, plots=False, quick=True)
    assert (out / "manifest.json").exists()
    assert (out / "summary.csv").exists()
    assert result["all_validations_passed"]
    if task == "2e":
        assert (out / "temporal_comparison.csv").exists()
    if task == "3g":
        assert (out / "without_daily_requirement.csv").exists()


# Check discoverable CLI tasks and the single-root-entry-point layout.
def test_all_task_selectors_have_help_and_no_duplicate_root_runners():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "main.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "3g-sensitivity" in r.stdout
    assert list(root.glob("run_*.py")) == []
    assert "2b" in TASKS and "2e" in TASKS


# Compare the initial-storage RHS dual with the independent finite difference.
def test_calibration_dual_has_same_sign_as_inventory_value(tmp_path):
    import json

    pi, table = calibrate_pi(tmp_path, plots=False)
    payload = json.loads((tmp_path / "pi_estimate.json").read_text())
    assert payload["initial_energy_dual"] == pytest.approx(pi, abs=2e-6)


# Check price means, reference values, and nondecreasing sizing gains.
def test_battery_sensitivity_design_and_monotonic_sizing(tmp_path):
    import pandas as pd

    result = run_task("3g-sensitivity", tmp_path, plots=False, quick=True)
    assert result["all_validations_passed"]
    e = pd.read_csv(tmp_path / "price_variability.csv")
    assert np.ptp(e.price_mean) < 1e-12
    assert list(e.alpha) == [0, 0.5, 1, 1.5]
    assert e.loc[e.alpha == 1, "inventory_adjusted_value"].iloc[0] == pytest.approx(
        3.16720938, abs=2e-6
    )
    assert e.loc[e.alpha == 1, "marginal_capacity_right"].iloc[0] == pytest.approx(
        0.462368421, abs=2e-6
    )
    # A larger feasible battery resource cannot lower its optimized operating value.
    for name, key in [("capacity_scan.csv", "capacity"), ("power_scan.csv", "power")]:
        for _, g in pd.read_csv(tmp_path / name).groupby("alpha"):
            assert np.all(np.diff(g.sort_values(key).inventory_adjusted_value) >= -2e-6)
    p = pd.read_csv(tmp_path / "power_scan.csv")
    assert np.max(abs(p.loc[p.power == 0, "inventory_adjusted_value"])) < 2e-6


# Exercise the full saturation search and compare its rounded report thresholds.
def test_full_plateau_search_reproduces_report_thresholds(tmp_path):
    import pandas as pd

    # Exercise dense scans and the bisection search, not the sparse smoke shortcut.
    run_task("3g-sensitivity", tmp_path, plots=False, quick=False)
    t = pd.read_csv(tmp_path / "conditional_plateaus.csv").set_index("alpha")
    assert bool(t.loc[0, "capacity_already_at_base"])
    for alpha, cap, power in [(0.5, 4.85, 2.19), (1.0, 7.38, 4.21), (1.5, 7.71, 4.21)]:
        assert t.loc[alpha, "capacity_plateau"] == pytest.approx(cap, abs=0.006)
        assert t.loc[alpha, "power_plateau"] == pytest.approx(power, abs=0.006)


# Ensure disabling figures retains all underlying numerical experiments.
def test_no_plots_keeps_numerical_response_experiments(tmp_path):
    result = run_task("2d", tmp_path / "2d", plots=False)
    assert result["run_count"] == 49
    calibrate_pi(tmp_path / "pi", plots=False)
    assert (tmp_path / "pi/value_curve.csv").exists()
