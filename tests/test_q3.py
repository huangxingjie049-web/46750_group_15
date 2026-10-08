"""Q3 invariants and limiting cases, independent of any plotted appearance."""
from dataclasses import replace
import numpy as np
import pytest
from src.data_loader import load_question
from src.model_q2_quadratic import QuadraticConsumerModel
from src.model_q3 import DailyEnergyConsumerModel
from src.validation_q3 import validate_solution, independent_solution


def solve(data):
    return DailyEnergyConsumerModel(data).build().solve()


def test_baseline_and_independent_optimum():
    d = load_question('Q3')
    r = solve(d)
    check = validate_solution(r, d)
    assert check['passed']
    assert r.meta['daily_energy'] == pytest.approx(40, abs=1e-6)
    assert r.meta['above_reference_hours'] > 0
    assert r.meta['minimum_energy_multiplier_mu'] > 0
    assert r.objective == pytest.approx(independent_solution(d)['objective'], abs=1e-6)


def test_zero_requirement_recovers_q2c():
    d = replace(load_question('Q3'), min_daily_energy_kWh=0.)
    r = solve(d)
    q2 = QuadraticConsumerModel(replace(d, min_daily_energy_kWh=None)).build().solve()
    np.testing.assert_allclose(r.hourly['load'], q2.hourly['load'], atol=1e-6)
    assert r.objective == pytest.approx(q2.objective, abs=1e-6)
    assert abs(r.meta['minimum_energy_multiplier_mu']) < 1e-6


def test_tightening_requirement_cannot_improve_surplus():
    d = load_question('Q3')
    low, high = solve(d), solve(replace(d, min_daily_energy_kWh=41.))
    assert high.objective <= low.objective + 1e-6


def test_shadow_price_by_two_sided_finite_difference():
    d = load_question('Q3')
    r = solve(d)
    step = 1e-3
    left = solve(replace(d, min_daily_energy_kWh=d.min_daily_energy_kWh-step))
    right = solve(replace(d, min_daily_energy_kWh=d.min_daily_energy_kWh+step))
    slope = (right.objective-left.objective)/(2*step)
    assert slope == pytest.approx(-r.meta['minimum_energy_multiplier_mu'], abs=2e-4)


def test_single_hour_analytic_case():
    d = replace(load_question('Q3'), hours=np.array([0]),
                energy_price=np.array([2.]), pv_available=np.array([0.]),
                reference_load=np.array([3.]), min_daily_energy_kWh=4.)
    r = solve(d)
    assert r.hourly.loc[0, 'load'] == pytest.approx(4., abs=1e-6)
    assert r.objective == pytest.approx(-11., abs=1e-6)
    assert r.meta['minimum_energy_multiplier_mu'] == pytest.approx(4.5, abs=1e-6)
    assert validate_solution(r, d)['passed']


@pytest.mark.parametrize('value', [145., float('nan'), -1.])
def test_invalid_requirement_rejected(value):
    with pytest.raises(ValueError):
        solve(replace(load_question('Q3'), min_daily_energy_kWh=value))


def test_battery_not_silently_ignored():
    with pytest.raises(ValueError):
        solve(load_question('Q3_battery'))


def test_repeated_build_rejected():
    model = DailyEnergyConsumerModel(load_question('Q3')).build()
    with pytest.raises(RuntimeError):
        model.build()


def test_q2c_still_rejects_daily_requirement():
    with pytest.raises(ValueError):
        QuadraticConsumerModel(load_question('Q3')).build()
