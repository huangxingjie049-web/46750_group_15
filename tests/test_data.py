"""Original-data preservation and correct ratio-to-physical-unit conversions."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load_question

ROOT = Path(__file__).resolve().parents[1]


# Guard the exact original JSON bytes against accidental edits.
def test_original_baseline_data_hashes():
    for relative, expected in json.loads(
        (ROOT / "docs/data_hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


# Check catalogue-ratio conversions, battery efficiency, and next-day horizons.
def test_physical_units_and_forecasts():
    d = load_question("Q3_battery")
    assert d.pv_available.sum() == pytest.approx(26.94)
    assert d.reference_load.sum() == pytest.approx(32.34)
    assert d.battery_initial_soc_kWh == pytest.approx(2)
    assert d.min_daily_energy_kWh == 40
    assert (
        d.battery_charging_efficiency * d.battery_discharging_efficiency
        == pytest.approx(0.9025)
    )
    assert d.n_hours == 24
    for v in [
        d.energy_price_next_day,
        d.reference_load_next_day,
        d.pv_available_next_day,
    ]:
        assert len(v) == 24


# Verify the controlled A/B comparison uses the same price and PV profiles.
def test_q1_cases_differ_only_in_pv_cost_for_relevant_inputs():
    a = load_question("Q1_caseA")
    b = load_question("Q1_caseB")
    assert a.pv_marginal_cost == 1.41 and b.pv_marginal_cost == 1.47
    for name in ["energy_price", "pv_available"]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
