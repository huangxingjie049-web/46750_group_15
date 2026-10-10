"""Reproduce Q3(d)--(e): python run_q3.py. No sensitivity sweep or battery yet."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_question
from src.model_q2_quadratic import QuadraticConsumerModel
from src.model_q3 import DailyEnergyConsumerModel
from src.validation_q3 import validate_solution

ROOT = Path(__file__).resolve().parents[2]


def save_figure(fig, out, name):
    fig.tight_layout()
    for extension in ['png','pdf']:
        fig.savefig(out/f'{name}.{extension}',dpi=180)
    plt.close(fig)


def plot_comparison(data, baseline, unconstrained, out):
    """Schedules, deviations and prices on a shared hourly axis."""
    h,u,t = baseline.hourly,unconstrained.hourly,data.hours
    fig,axes = plt.subplots(3,1,sharex=True,figsize=(11,8))
    axes[0].step(t,data.reference_load,where='mid',color='black',ls='--',label='Reference')
    axes[0].step(t,u['load'],where='mid',label='Q2(c): no daily requirement')
    axes[0].step(t,h['load'],where='mid',label=f'Q3: E_min = {data.min_daily_energy_kWh:g} kWh')
    axes[0].set_ylabel('Consumption (kWh / 1 h)')
    axes[0].legend(fontsize=9,ncol=2)
    axes[1].bar(t-.2,u['deviation'],width=.4,label='Q2(c) deviation')
    axes[1].bar(t+.2,h['deviation'],width=.4,label='Q3 deviation')
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set_ylabel('Deviation (kWh / 1 h)');axes[1].legend(ncol=2)
    axes[2].step(t,h['effective_import_price'],where='mid',label='Import price')
    axes[2].step(t,h['effective_export_price'],where='mid',label='Export price')
    axes[2].axhline(data.pv_marginal_cost,ls='--',color='purple',label='PV marginal cost')
    axes[2].axhline(baseline.meta['minimum_energy_multiplier_mu'],ls=':',color='black',label='Daily multiplier mu')
    axes[2].set(xlabel='Hour',ylabel='DKK/kWh');axes[2].legend(ncol=2,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25);ax.set_xticks(range(0,data.n_hours,2))
    fig.suptitle('Q3(e): same-data comparison with the unconstrained quadratic consumer')
    save_figure(fig,out,'q3e_comparison')

    fig,ax = plt.subplots(figsize=(11,4))
    ax.step(t,data.reference_load,where='mid',color='black',ls='--',label='Reference')
    ax.step(t,u['load'],where='mid',label='Q2(c): no daily requirement')
    ax.step(t,h['load'],where='mid',label='Q3: daily requirement')
    ax.set(xlabel='Hour',ylabel='Consumption (kWh / 1 h)')
    ax.set_xticks(range(0,data.n_hours,2));ax.legend();ax.grid(alpha=.25)
    save_figure(fig,out,'q3e_load_comparison')

    fig,axes = plt.subplots(2,1,sharex=True,figsize=(11,6))
    for ax,r,label in zip(axes,[unconstrained,baseline],['Q2(c): no daily requirement','Q3: daily requirement']):
        f=r.hourly
        ax.bar(t,f['import'],label='Import')
        ax.bar(t,-f['export'],label='Export (negative)')
        ax.step(t,f['pv'],where='mid',color='orange',label='PV produced')
        ax.step(t,f['load'],where='mid',color='black',label='Load')
        ax.set(title=label,ylabel='Energy (kWh / 1 h)');ax.legend(ncol=4);ax.grid(alpha=.25)
    limits=[ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
    axes[-1].set_xlabel('Hour');axes[-1].set_xticks(range(0,data.n_hours,2))
    save_figure(fig,out,'q3e_energy_flows')


def export_metrics(table, out):
    table.to_csv(out/'comparison_metrics.csv',index=False)
    rows = [
        ('Procurement cost (DKK)','procurement_cost'),
        ('Quadratic disutility (DKK)','disutility'),
        ('Total cost (DKK)','total_cost'),
        ('Net utility (DKK)','net_utility'),
        ('Daily consumption (kWh)','daily_energy'),
        ('Absolute deviation (kWh)','absolute_deviation'),
        ('Hours above reference','above_reference_hours'),
        ('Hours at a load bound','binding_hours'),
    ]
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
           r'\caption{Q3(e): comparison on identical input data. Only Q3 imposes the daily minimum of '+
           f'{table.iloc[1]["minimum_daily_energy"]:g}'+r' kWh.}',
           r'\label{tab:q3e_metrics}',r'\begin{tabular}{lrr}',r'\toprule',
           r'Metric & Q2(c), no daily requirement & Q3 \\\midrule']
    for label,key in rows:
        vals=[table.iloc[i][key] for i in range(2)]
        fmt=(lambda x:f'{int(x)}') if key in ['above_reference_hours','binding_hours'] else (lambda x:f'{x:.3f}')
        lines.append(f'{label} & {fmt(vals[0])} & {fmt(vals[1])}'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (out/'q3e_metrics.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def reproduce(emin=None,cq=None,output_dir=None):
    d=load_question('Q3')
    if emin is not None: d=replace(d,min_daily_energy_kWh=emin)
    if cq is not None: d=replace(d,quadratic_disutility=cq)
    out=Path(output_dir) if output_dir else ROOT/'results'/'Q3'
    out.mkdir(parents=True,exist_ok=True)
    # Remove ONLY the daily constraint; keep the exact same reference and prices.
    du=replace(d,question='Q3_unconstrained',min_daily_energy_kWh=None)
    qm=DailyEnergyConsumerModel(d).build()
    baseline=qm.solve()
    unconstrained=QuadraticConsumerModel(du).build().solve()
    checks={'Q3_base':validate_solution(baseline,d),
            'same_data_unconstrained':validate_solution(unconstrained,du)}
    if not all(v['passed'] for v in checks.values()):
        raise RuntimeError(f'Validation failed: {checks}')
    baseline.save(out);unconstrained.save(out)
    qm.m.write(str(out/'q3_model.lp'))
    rows=[]
    for name,r in [('Q2(c)_same_data',unconstrained),('Q3',baseline)]:
        rows.append({'model':name,**r.meta,
                     'import_kWh':float(r.hourly['import'].sum()),
                     'export_kWh':float(r.hourly['export'].sum()),
                     'pv_kWh':float(r.hourly['pv'].sum())})
    table=pd.DataFrame(rows);export_metrics(table,out)
    pd.DataFrame({'hour':d.hours,'reference_load':d.reference_load,
                  'unconstrained_load':unconstrained.hourly['load'].to_numpy(),
                  'q3_load':baseline.hourly['load'].to_numpy(),
                  'q3_minus_unconstrained':(baseline.hourly['load']-unconstrained.hourly['load']).to_numpy()}
                 ).to_csv(out/'comparison_hourly.csv',index=False)

    # A few targeted validation probes, not the Q3(f) sensitivity analysis.
    zero_data=replace(d,min_daily_energy_kWh=0.)
    zero=DailyEnergyConsumerModel(zero_data).build().solve()
    zero_error=float(np.max(abs(zero.hourly['load']-unconstrained.hourly['load'])))
    zero_check={'max_load_error_kWh':zero_error,
                'objective_error_DKK':abs(zero.objective-unconstrained.objective)}
    zero_check['passed']=max(zero_check.values())<=2e-6
    checks['zero_requirement_recovers_Q2c']=zero_check
    step=1e-3
    if step < d.min_daily_energy_kWh < d.n_hours*d.load_max_kWh-step:
        left=DailyEnergyConsumerModel(replace(d,min_daily_energy_kWh=d.min_daily_energy_kWh-step)).build().solve()
        right=DailyEnergyConsumerModel(replace(d,min_daily_energy_kWh=d.min_daily_energy_kWh+step)).build().solve()
        left_slope=(baseline.objective-left.objective)/step
        right_slope=(right.objective-baseline.objective)/step
        pi=baseline.duals['minimum_energy']
        # At a regime boundary one-sided slopes can differ. Keep both as evidence.
        fd={'step_kWh':step,'left_slope_DKK_per_kWh':left_slope,
            'right_slope_DKK_per_kWh':right_slope,'gurobi_pi':pi,
            'central_slope_error':abs((left_slope+right_slope)/2-pi),
            'monotonicity_passed':bool(left.objective>=baseline.objective-2e-6 and right.objective<=baseline.objective+2e-6)}
        fd['passed']=fd['monotonicity_passed'] and fd['central_slope_error']<2e-4
        checks['minimum_energy_finite_difference']=fd
    if not all(v['passed'] for v in checks.values()):
        raise RuntimeError(f'Validation probes failed: {checks}')
    (out/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    hypotheses={
        'reference_daily_energy_kWh':float(d.reference_load.sum()),
        'unconstrained_daily_energy_kWh':unconstrained.meta['daily_energy'],
        'minimum_daily_energy_kWh':d.min_daily_energy_kWh,
        'forced_above_reference_expected':bool(d.min_daily_energy_kWh>d.reference_load.sum()+1e-6),
        'above_reference_observed_hours':baseline.meta['above_reference_hours'],
        'above_reference_hours':[int(t) for t in d.hours if baseline.hourly.loc[t,'deviation']>1e-6],
        'daily_energy_equals_max_of_minimum_and_unconstrained':bool(abs(baseline.meta['daily_energy']-
            max(d.min_daily_energy_kWh,unconstrained.meta['daily_energy']))<2e-6),
        'note':'These observations test Q3(c) in the supplied positive-price regime; they are not universal claims for other data.'}
    (out/'hypothesis_checks.json').write_text(json.dumps(hypotheses,indent=2),encoding='utf-8')
    snapshot={'question':'Q3','hours':d.hours.tolist(),'energy_price':d.energy_price.tolist(),
              'pv_available':d.pv_available.tolist(),'reference_load':d.reference_load.tolist(),
              'load_min_kWh':d.load_min_kWh,'load_max_kWh':d.load_max_kWh,
              'pv_marginal_cost':d.pv_marginal_cost,'import_tariff':d.import_tariff,
              'export_tariff':d.export_tariff,'cq':d.quadratic_disutility,
              'minimum_daily_energy_kWh':d.min_daily_energy_kWh,
              'solver_version':baseline.meta['solver_version'],
              'duals':'Report mu = -minimum_energy.Pi; report lambda_t = -balance[t].Pi.'}
    (out/'input_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    plot_comparison(d,baseline,unconstrained,out)
    print(table[['model','procurement_cost','disutility','net_utility','daily_energy','absolute_deviation','above_reference_hours']].to_string(index=False))
    print(f'Q3 mu = {baseline.meta["minimum_energy_multiplier_mu"]:.6f} DKK/kWh')
    print('All validation checks passed.')
    print(f'Outputs: {out.resolve()}')
    return baseline,unconstrained,checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emin',type=float,default=None,help='Override E_min (kWh); default is the supplied Q3 value.')
    parser.add_argument('--cq',type=float,default=None,help='Override c_Q (DKK/kWh^2).')
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args(argv)
    reproduce(args.emin,args.cq,args.output_dir)


if __name__=='__main__':
    main()
