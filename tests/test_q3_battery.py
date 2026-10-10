from dataclasses import replace
import numpy as np
import pytest
from src.data_loader import load_question
from src.model_q3 import DailyEnergyConsumerModel


def solve(d, mode='cyclic', pi=0):
    from src.model_q3_battery import BatteryConsumerModel
    m=BatteryConsumerModel(d,terminal_mode=mode,terminal_value=pi).build()
    return m,m.solve()


def test_cyclic_returns_initial_and_improves_current_cost():
    d=load_question('Q3_battery')
    m,r=solve(d)
    assert r.hourly.soc.iloc[-1]==pytest.approx(d.battery_initial_soc_kWh,abs=1e-6)
    base=DailyEnergyConsumerModel(load_question('Q3')).build().solve()
    assert r.meta['total_cost']<=base.meta['total_cost']+1e-6


def test_zero_capacity_recovers_no_battery():
    d=replace(load_question('Q3_battery'),battery_capacity_kWh=0,battery_initial_soc_kWh=0)
    _,r=solve(d)
    base=DailyEnergyConsumerModel(load_question('Q3')).build().solve()
    assert r.objective==pytest.approx(base.objective,abs=2e-6)


def test_terminal_value_counted_once_and_envelope():
    d=load_question('Q3_battery')
    _,r=solve(d,'value',2)
    assert r.objective==pytest.approx(-r.meta['total_cost']+2*r.hourly.soc.iloc[-1],abs=1e-6)
    _,lo=solve(d,'value',2-1e-3)
    _,hi=solve(d,'value',2+1e-3)
    assert (hi.objective-lo.objective)/.002==pytest.approx(r.hourly.soc.iloc[-1],abs=.002)


def test_high_value_retains_more_than_zero_value():
    d=load_question('Q3_battery')
    _,lo=solve(d,'value',0)
    _,hi=solve(d,'value',10)
    assert hi.hourly.soc.iloc[-1]>lo.hourly.soc.iloc[-1]+1
    assert hi.hourly.soc.iloc[-1]==pytest.approx(4,abs=1e-6)


def test_validation_detects_corrupted_soc():
    from src.validation_q3_battery import validate_battery_solution
    d=load_question('Q3_battery');m,r=solve(d)
    assert validate_battery_solution(m,r)['passed']
    r.hourly.loc[0,'soc']+=.1
    assert not validate_battery_solution(m,r)['passed']

@pytest.mark.parametrize('mode,pi',[('bad',0),('value',-1),('value',np.nan),('cyclic',1)])
def test_invalid_terminal_settings(mode,pi):
    with pytest.raises(ValueError): solve(load_question('Q3_battery'),mode,pi)


def test_two_hour_arbitrage_matches_hand_calculation():
    d=replace(load_question('Q3_battery'),hours=np.arange(2),energy_price=np.array([1.,3.]),
        reference_load=np.zeros(2),pv_available=np.zeros(2),load_min_kWh=0,load_max_kWh=0,
        min_daily_energy_kWh=0,import_tariff=.1,export_tariff=.1,
        battery_capacity_kWh=1,battery_initial_soc_kWh=0,battery_max_charge_kW=1,
        battery_max_discharge_kW=1,battery_charging_efficiency=1,battery_discharging_efficiency=1)
    _,r=solve(d)
    assert r.objective==pytest.approx(1.8,abs=1e-6)
    assert r.hourly.loc[0,'charge']==pytest.approx(1,abs=1e-6)
    assert r.hourly.loc[1,'discharge']==pytest.approx(1,abs=1e-6)
