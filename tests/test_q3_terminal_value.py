from dataclasses import replace
import numpy as np
import pytest
from src.data_loader import load_question


def one_hour():
    d=load_question('Q3_battery')
    return replace(d,hours=np.arange(1),energy_price_next_day=np.array([2.]),
        pv_available_next_day=np.zeros(1),reference_load_next_day=np.ones(1),
        load_min_kWh=1,load_max_kWh=1,min_daily_energy_kWh=1,
        import_tariff=.5,export_tariff=.5,battery_capacity_kWh=1,
        battery_initial_soc_kWh=.5,battery_max_charge_kW=2,battery_max_discharge_kW=2)


def test_forecasts_replaced_without_mutating_today():
    from src.terminal_value_q3 import next_day_data
    d=load_question('Q3_battery');original=d.energy_price.copy()
    nd=next_day_data(d,1)
    np.testing.assert_array_equal(nd.energy_price,d.energy_price_next_day)
    np.testing.assert_array_equal(nd.pv_available,d.pv_available_next_day)
    np.testing.assert_array_equal(nd.reference_load,d.reference_load_next_day)
    np.testing.assert_array_equal(d.energy_price,original)
    assert nd.battery_initial_soc_kWh==1


def test_marginal_value_matches_avoided_import_hand_calculation():
    from src.terminal_value_q3 import estimate_terminal_value
    info,_,_=estimate_terminal_value(one_hour(),epsilon=.01)
    assert info['pi_estimate_DKK_per_kWh']==pytest.approx(.95*2.5,abs=1e-6)
    assert info['baseline_initial_energy_dual_DKK_per_kWh']==pytest.approx(2.375,abs=1e-6)


def test_fixed_end_target_does_not_follow_changed_initial_energy():
    from src.terminal_value_q3 import estimate_terminal_value
    info,rows,_=estimate_terminal_value(one_hour(),epsilon=.01,end_soc=.2)
    assert info['pi_estimate_DKK_per_kWh']==pytest.approx(2.375,abs=1e-6)
    for row in rows: assert row['final_soc_kWh']==pytest.approx(.2,abs=1e-6)


def test_initial_zero_uses_right_difference():
    from src.terminal_value_q3 import estimate_terminal_value
    info,_,_=estimate_terminal_value(one_hour(),initial_energy=0,epsilon=.01)
    assert info['left_slope_DKK_per_kWh'] is None
    assert info['pi_estimate_DKK_per_kWh']==pytest.approx(2.375,abs=1e-6)

@pytest.mark.parametrize('kwargs',[{'epsilon':0},{'epsilon':float('nan')},{'initial_energy':-1},{'end_soc':2}])
def test_invalid_estimation_settings(kwargs):
    from src.terminal_value_q3 import estimate_terminal_value
    with pytest.raises(ValueError): estimate_terminal_value(one_hour(),**kwargs)
