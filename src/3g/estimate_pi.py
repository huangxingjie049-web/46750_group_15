"""Estimate pi from next-day forecasts, optionally run today using that estimate."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_question
from src.terminal_value_q3 import estimate_terminal_value,solve_next_day
from run_q3 import save_figure

ROOT=Path(__file__).resolve().parents[2]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epsilon',type=float,default=.01,help='Energy perturbation in kWh.')
    p.add_argument('--initial-energy',type=float,default=None,help='Expansion point; default 2 kWh from supplied data.')
    p.add_argument('--next-day-end-soc',type=float,default=None,help='Fixed day-2 end target in kWh; default is free day-2 end with zero day-3 reward.')
    p.add_argument('--emin',type=float,default=None)
    p.add_argument('--cq',type=float,default=None)
    p.add_argument('--run-today',action='store_true',help='Run run_q3g.py --mode both using the computed pi.')
    p.add_argument('--sweep',action='store_true',help='Also run today\'s illustrative g(v) sweeps; requires --run-today.')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'Q3g_pi_estimation')
    a=p.parse_args(argv)
    if a.sweep and not a.run_today: p.error('--sweep requires --run-today.')
    d=load_question('Q3_battery')
    if a.emin is not None: d=replace(d,min_daily_energy_kWh=a.emin)
    if a.cq is not None: d=replace(d,quadratic_disutility=a.cq)
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    info,records,baseline=estimate_terminal_value(d,a.initial_energy,a.epsilon,a.next_day_end_soc)
    baseline.save(out,tag='baseline')
    pd.DataFrame([{k:v for k,v in r.items() if k!='validation'} for r in records]).to_csv(out/'finite_difference_runs.csv',index=False)
    (out/'validation.json').write_text(json.dumps(records,indent=2))
    # Check numerical robustness to perturbation size.
    stability=[]
    for eps in sorted(set([a.epsilon/10,a.epsilon,a.epsilon*10])):
        v,_,_=estimate_terminal_value(d,a.initial_energy,eps,a.next_day_end_soc)
        stability.append({'epsilon_kWh':eps,'pi_estimate_DKK_per_kWh':v['pi_estimate_DKK_per_kWh'],
            'left_slope':v['left_slope_DKK_per_kWh'],'right_slope':v['right_slope_DKK_per_kWh']})
    pd.DataFrame(stability).to_csv(out/'epsilon_stability.csv',index=False)
    # Value curve gives context: a single pi may be inaccurate away from E0.
    curve=[]
    for energy in np.linspace(0,d.battery_capacity_kWh,9):
        record,_=solve_next_day(d,float(energy),a.next_day_end_soc)
        curve.append(record)
    pd.DataFrame([{k:v for k,v in r.items() if k!='validation'} for r in curve]).to_csv(out/'next_day_value_curve.csv',index=False)
    (out/'curve_validation.json').write_text(json.dumps(curve,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    e=np.array([r['initial_energy_kWh'] for r in curve]);w=np.array([r['objective_DKK'] for r in curve])
    reference=info['expansion_energy_kWh'];base_w=baseline.objective;pi=info['pi_estimate_DKK_per_kWh']
    axes[0].plot(e,w,'o-',label='Next-day optimum')
    axes[0].plot(e,base_w+pi*(e-reference),'--',label='Local linear approximation')
    axes[0].set(xlabel='Next-day initial energy (kWh)',ylabel='Optimal objective (DKK)');axes[0].legend(fontsize=8)
    axes[1].plot(e,[r['initial_energy_dual_DKK_per_kWh'] for r in curve],'o-',label='Initial-energy marginal value')
    axes[1].axhline(pi,ls='--',label='Estimated pi')
    axes[1].set(xlabel='Next-day initial energy (kWh)',ylabel='DKK per stored kWh');axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2);ax.axvline(reference,color='grey',ls=':')
    save_figure(fig,out,'next_day_energy_value')
    info['epsilon_stability']=stability
    info['forecast_inputs']={'price_DKK_per_kWh':d.energy_price_next_day.tolist(),
        'pv_available_kWh':d.pv_available_next_day.tolist(),
        'reference_load_kWh':d.reference_load_next_day.tolist(),
        'minimum_daily_energy_kWh':d.min_daily_energy_kWh,'cq':d.quadratic_disutility,
        'import_tariff':d.import_tariff,'export_tariff':d.export_tariff,
        'pv_marginal_cost':d.pv_marginal_cost,'capacity_kWh':d.battery_capacity_kWh,
        'charge_limit_kW':d.battery_max_charge_kW,'discharge_limit_kW':d.battery_max_discharge_kW,
        'eta_charge':d.battery_charging_efficiency,'eta_discharge':d.battery_discharging_efficiency}
    (out/'pi_estimate.json').write_text(json.dumps(info,indent=2))
    print(f'Estimated pi = {pi:.9f} DKK per stored kWh')
    print(f'Expansion point = {reference:g} kWh; epsilon = {a.epsilon:g} kWh')
    print(f'Next-day end treatment: {info["next_day_end_treatment"]}')
    print(pd.DataFrame(stability).to_string(index=False))
    print(f'Outputs: {out.resolve()}')
    if a.run_today:
        cmd=[sys.executable,str(ROOT/'main.py'),'--task','3g','--mode','both','--pi',str(pi),'--output-dir',str(out/'today_using_estimated_pi')]
        if a.sweep: cmd.append('--sweep')
        if a.emin is not None: cmd+=['--emin',str(a.emin)]
        if a.cq is not None: cmd+=['--cq',str(a.cq)]
        sys.stdout.flush();subprocess.run(cmd,check=True)

if __name__=='__main__': main()
