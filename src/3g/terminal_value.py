"""Estimate a local terminal-energy value from a next-day QP.

A separate, FIXED day-2 end condition is held constant in all perturbations.
This approximates the future value function with a local linear slope; it is
not a full multi-day optimisation or a universal constant energy price.
"""
from dataclasses import replace
import numpy as np
from src.model_q3_battery import BatteryConsumerModel
from src.validation_q3_battery import validate_battery_solution


def next_day_data(today,initial_energy):
    arrays={}
    for name in ('energy_price','pv_available','reference_load'):
        values=getattr(today,name+'_next_day')
        if values is None:
            raise ValueError(f'Missing {name}_next_day forecast.')
        values=np.asarray(values,dtype=float)
        if values.shape!=(today.n_hours,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name}_next_day forecast.')
        arrays[name]=values.copy()
    return replace(today,question='Q3_next_day',battery_initial_soc_kWh=float(initial_energy),
        battery_final_soc_kWh=None,**arrays)


def solve_next_day(today,initial_energy,end_soc=None):
    d=next_day_data(today,initial_energy)
    # No day-3 reward. Optional FIXED day-2 end-energy equality is added below.
    m=BatteryConsumerModel(d,terminal_mode='value',terminal_value=0).build()
    if end_soc is not None:
        m.con['fixed_next_day_terminal_soc']=m.m.addConstr(
            m.var['soc'][d.hours[-1]]==end_soc,name='fixed_next_day_terminal_soc')
    r=m.solve();checks=validate_battery_solution(m,r)
    if end_soc is not None:
        err=float(abs(r.hourly.soc.iloc[-1]-end_soc))
        checks['fixed_terminal_soc_error']=err
        checks['passed']=checks['passed'] and err<=checks['tolerance']
    if not checks['passed']: raise RuntimeError(f'Next-day validation failed: {checks}')
    # storage_balance[0] is e0 - eta_ch*C0 + D0/eta_dis = initial_energy.
    dual=float(m.con['storage_balance'][d.hours[0]].Pi)
    record={'initial_energy_kWh':float(initial_energy),'objective_DKK':float(r.objective),
        'total_cost_DKK':r.meta['total_cost'],'final_soc_kWh':r.meta['final_soc_kWh'],
        'initial_energy_dual_DKK_per_kWh':dual,'validation':checks}
    return record,r


def estimate_terminal_value(today,initial_energy=None,epsilon=.01,end_soc=None):
    cap=today.battery_capacity_kWh
    initial=today.battery_initial_soc_kWh if initial_energy is None else initial_energy
    if cap is None or not np.isfinite(cap) or cap<=0: raise ValueError('Capacity must be positive.')
    if initial is None or not np.isfinite(initial) or not 0<=initial<=cap: raise ValueError('Initial energy must be within capacity.')
    if not np.isfinite(epsilon) or epsilon<=0: raise ValueError('epsilon must be finite and positive.')
    if end_soc is not None and (not np.isfinite(end_soc) or not 0<=end_soc<=cap): raise ValueError('Fixed terminal target must be within capacity.')
    points=sorted(set([max(0.,initial-epsilon),float(initial),min(cap,initial+epsilon)]))
    records=[];baseline=None
    for e in points:
        record,r=solve_next_day(today,e,end_soc)
        records.append(record)
        if e==initial: baseline=r
    centre=next(v for v in records if v['initial_energy_kWh']==initial)
    left=next((v for v in records if v['initial_energy_kWh']<initial),None)
    right=next((v for v in records if v['initial_energy_kWh']>initial),None)
    ls=None if left is None else (centre['objective_DKK']-left['objective_DKK'])/(initial-left['initial_energy_kWh'])
    rs=None if right is None else (right['objective_DKK']-centre['objective_DKK'])/(right['initial_energy_kWh']-initial)
    # Average directional slopes at interior points; one-sided at boundaries.
    estimate=float(np.mean([v for v in (ls,rs) if v is not None]))
    if estimate < -2e-5:
        raise RuntimeError('Negative marginal value: inspect forecasts and modelling assumptions.')
    dual=centre['initial_energy_dual_DKK_per_kWh']
    fd_bounds_ok=(rs is None or dual>=rs-2e-4) and (ls is None or dual<=ls+2e-4)
    if not fd_bounds_ok: raise RuntimeError('Finite differences are inconsistent with initial-energy dual.')
    info={'pi_estimate_DKK_per_kWh':max(0.,estimate),
        'expansion_energy_kWh':float(initial),'epsilon_kWh':float(epsilon),
        'left_slope_DKK_per_kWh':ls,'right_slope_DKK_per_kWh':rs,
        'baseline_initial_energy_dual_DKK_per_kWh':dual,
        'dual_within_directional_slope_bounds':bool(fd_bounds_ok),
        'next_day_end_treatment':'free end energy, zero day-3 terminal value' if end_soc is None else 'fixed end energy, independent of perturbed initial energy',
        'fixed_next_day_end_energy_kWh':end_soc,
        'method':'Average left/right finite-difference slopes; one-sided at capacity boundary.',
        'assumptions':'Use next-day price/PV/reference forecasts; retain today\'s tariffs, PV cost, cQ, daily minimum, battery capacity/power/efficiencies.',
        'limitation':'A local linear approximation; value depends on stored quantity and day-2/day-3 boundary assumptions.'}
    return info,records,baseline
