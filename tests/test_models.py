"""Independent mathematical checks for every model family, including edge cases."""

from dataclasses import replace
import numpy as np
import pytest
from src.data_loader import load_question
from src.model import solve
from src.validation import validate, independent_schedule


# Compare each supplied base result with both report values and the independent oracle.
@pytest.mark.parametrize(
    "case,kind,energy,objective",
    [
        ("Q1_caseA", "utility", 35.94, 2.6088),
        ("Q1_caseB", "utility", 12, 1.95),
        ("Q2_linear", "linear", 23.1, -44.3142),
        ("Q2_quadratic", "quadratic", 22.765, -39.971025),
        ("Q3", "quadratic", 40, -63.3112847826087),
    ],
)
def test_report_bases_and_independent_oracle(case, kind, energy, objective):
    d = load_question(case)
    r = solve(d, kind)
    assert validate(r, d)["passed"]
    assert r.metrics["daily_energy"] == pytest.approx(energy, abs=2e-6)
    assert r.metrics["objective"] == pytest.approx(objective, abs=2e-6)
    assert independent_schedule(d, kind)["objective"] == pytest.approx(
        objective, abs=2e-6
    )


# Check valid results pass and a corrupted load fails validation in every family.
@pytest.mark.parametrize(
    "kind,case",
    [
        ("utility", "Q1_caseA"),
        ("linear", "Q2_linear"),
        ("quadratic", "Q2_quadratic"),
        ("coupled", "Q2_quadratic"),
    ],
)
def test_each_family_accounting_and_tamper_detection(kind, case):
    d = load_question(case)
    r = solve(d, kind, coupling=1 if kind == "coupled" else 0)
    assert validate(r, d)["passed"]
    # Deliberately corrupt balance without re-solving; the validator must catch the edit.
    r.hourly.loc[8, "load"] += 0.1
    assert not validate(r, d)["passed"]


# At a price tie, compare optimal value without assuming a unique load schedule.
def test_linear_tie_is_compared_by_value_not_schedule():
    d = replace(load_question("Q2_linear"), linear_disutility=1.41)
    r = solve(d, "linear")
    assert validate(r, d)["passed"]
    assert r.metrics["objective"] == pytest.approx(
        independent_schedule(d, "linear")["objective"], abs=2e-6
    )


# Exercise zero and out-of-bounds references directly at the model interface.
def test_quadratic_zero_and_outside_reference():
    d = load_question("Q2_quadratic")
    for ref in [np.zeros(24), np.full(24, -1.0), np.full(24, 8.0)]:
        s = replace(d, reference_load=ref)
        assert validate(solve(s, "quadratic"), s)["passed"]


# Check daily-constraint activation, its removal, and an infeasible requirement.
def test_daily_activation_and_removal():
    d = load_question("Q3")
    # Include inactive, binding, reference-total, and maximum-feasible daily requirements.
    for emin in [0, 20, 22.765, 32.34, 40, 144]:
        s = replace(d, min_daily_energy_kWh=emin)
        r = solve(s, "quadratic")
        assert validate(r, s)["passed"]
        assert r.metrics["daily_energy"] == pytest.approx(max(emin, 22.765), abs=2e-6)
    s = replace(d, min_daily_energy_kWh=None)
    assert solve(s, "quadratic").metrics["objective"] == pytest.approx(
        -39.971025, abs=2e-6
    )
    # A daily requirement above the physical maximum must be rejected before solving.
    with pytest.raises(ValueError):
        solve(replace(d, min_daily_energy_kWh=145), "quadratic")


# Check complete discomfort accounting and the zero-coupling limit.
def test_coupled_penalty_is_counted_and_zero_recovers_q2c():
    d = load_question("Q2_quadratic")
    r = solve(d, "coupled", coupling=1)
    assert validate(r, d)["passed"]
    assert r.metrics["coupling_disutility"] > 0
    assert r.metrics["objective"] == pytest.approx(
        -r.metrics["procurement_cost"] - r.metrics["disutility"], abs=1e-6
    )
    # When cS=0 the coupled objective reduces exactly to the ordinary quadratic model.
    zero = solve(d, "coupled", coupling=0)
    base = solve(d, "quadratic")
    np.testing.assert_allclose(zero.hourly.load, base.hourly.load, atol=2e-6)


# Verify battery feasibility/KKT and both value/cyclic terminal treatments.
@pytest.mark.parametrize("mode", ["value", "cyclic"])
def test_battery_physical_kkt_and_terminal(mode):
    d = load_question("Q3_battery")
    r = solve(d, "quadratic", terminal=mode, pi=1.4736842105263157)
    assert validate(r, d)["passed"]
    assert r.metrics["simultaneous_charge_discharge_hours"] == 0
    if mode == "cyclic":
        assert r.metrics["final_soc"] == pytest.approx(2, abs=1e-7)
    else:
        assert r.metrics["objective"] == pytest.approx(-57.19670698864, abs=2e-6)


# With idle storage, check the terminal reward is counted once for retained inventory.
def test_zero_power_inventory_and_terminal_reward_once():
    d = replace(
        load_question("Q3_battery"), battery_max_charge_kW=0, battery_max_discharge_kW=0
    )
    r = solve(d, "quadratic", terminal="value", pi=1.5)
    assert validate(r, d)["passed"]
    assert r.metrics["objective"] == pytest.approx(-63.3112847826087 + 3, abs=2e-6)
    assert r.metrics["terminal_credit"] == pytest.approx(3, abs=1e-8)


# Reproduce the report battery consumption after removing only the daily minimum.
def test_battery_without_daily_requirement():
    d = replace(load_question("Q3_battery"), min_daily_energy_kWh=None)
    r = solve(d, "quadratic", terminal="value", pi=1.4736842105263157)
    assert validate(r, d)["passed"]
    assert r.metrics["daily_energy"] == pytest.approx(23.24, abs=2e-6)


# Construct a feasible but suboptimal schedule to challenge the KKT validator.
def test_feasible_but_nonoptimal_battery_schedule_is_rejected():
    """KKT complementarity must validate the exported schedule, not stale x."""
    d = load_question("Q3_battery")
    r = solve(d, "quadratic", pi=1.4736842105263157)
    h = r.hourly
    # Replace optimal battery dispatch with idle storage while keeping a feasible energy balance.
    h["charge"] = 0.0
    h["discharge"] = 0.0
    h["soc"] = 2.0
    h["import"] = (h.load - h.pv).clip(lower=0)
    h["export"] = (h.pv - h.load).clip(lower=0)
    h["procurement_cost"] = (
        d.pv_marginal_cost * h.pv
        + h.effective_import_price * h["import"]
        - h.effective_export_price * h["export"]
    )
    # Update accounting to ensure failure comes from optimality checks rather than a stale total.
    r.metrics["procurement_cost"] = float(h.procurement_cost.sum())
    r.metrics["objective"] = (
        -r.metrics["procurement_cost"]
        - r.metrics["disutility"]
        + 2 * 1.4736842105263157
    )
    assert not validate(r, d)["passed"]


# Compare sizing duals with objective changes under small resource expansions.
def test_capacity_and_common_power_duals_match_finite_differences():
    d = load_question("Q3_battery")
    pi = 1.4736842105263157
    base = solve(d, "quadratic", pi=pi)
    # Use a small positive size change and compare its objective slope with the relevant dual sum.
    eps = 1e-4
    cap = solve(replace(d, battery_capacity_kWh=4 + eps), "quadratic", pi=pi)
    power = solve(
        replace(d, battery_max_charge_kW=1.5 + eps, battery_max_discharge_kW=1.5 + eps),
        "quadratic",
        pi=pi,
    )
    assert (
        cap.metrics["objective"] - base.metrics["objective"]
    ) / eps == pytest.approx(base.metrics["marginal_capacity"], abs=2e-4)
    assert (
        power.metrics["objective"] - base.metrics["objective"]
    ) / eps == pytest.approx(base.metrics["marginal_power"], abs=2e-4)
