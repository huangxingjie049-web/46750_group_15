"""Reproduce all Q2(c) experiments: python run_q2c.py (or --cq 1 --sweep 0.1 0.5 1 2 5)."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data_loader import load_question
from src.model_q2_quadratic import QuadraticConsumerModel

DEFAULT_SWEEP=[0.05,0.1,0.2,0.5,1.,2.,5.,10.]

def linear_benchmark(d):
    """Independent piecewise-linear reference, not the teammate's Q2(b) implementation.

    Use the provided linear coefficient. At a tie choose the smallest load;
    this selection need not match another LP solver's equally optimal choice.
    """
    cL=load_question('Q2_linear').linear_disutility
    rows=[]
    for t in d.hours:
        a,b=d.energy_price[t]+d.import_tariff,d.energy_price[t]-d.export_tariff
        p,ref=d.pv_available[t],d.reference_load[t]
        def evaluate(L):
            P=p if d.pv_marginal_cost<b else (min(L,p) if d.pv_marginal_cost<a else 0.)
            I,E=max(L-P,0),max(P-L,0)
            C=d.pv_marginal_cost*P+a*I-b*E
            return C+cL*abs(L-ref),P,I,E,C
        candidates=sorted(set([d.load_min_kWh,d.load_max_kWh,float(np.clip(ref,d.load_min_kWh,d.load_max_kWh)),float(np.clip(p,d.load_min_kWh,d.load_max_kWh))]))
        L=min(candidates,key=lambda x:round(evaluate(x)[0],10))
        cost,P,I,E,C=evaluate(L)
        rows.append({'hour':int(t),'load':L,'pv':P,'import':I,'export':E,'procurement_cost':C,'disutility':cL*abs(L-ref),'absolute_deviation':abs(L-ref)})
    return pd.DataFrame(rows).set_index('hour')

def reproduce(cq=None,sweep=DEFAULT_SWEEP,output_dir=None):
    d=load_question('Q2_quadratic')
    if cq is not None: d=replace(d,quadratic_disutility=cq)
    out=Path(output_dir or 'results/Q2_quadratic');out.mkdir(parents=True,exist_ok=True)
    base=QuadraticConsumerModel(d).build().solve();base.save(out)
    runs={}
    for q in sorted(set(sweep or [])):
        result=QuadraticConsumerModel(replace(d,quadratic_disutility=q)).build().solve()
        result.save(out,tag=f'cq_{q:g}')
        runs[q]=result
    rows=[dict(result.meta,status=result.status) for result in runs.values()]
    if rows:
        pd.DataFrame(rows).to_csv(out/'sweep_metrics.csv',index=False)
        latex = [r"\begin{table}[htbp]\centering",
                 r"\caption{Q2(c) coefficient sweep; $c^Q$ is in DKK/kWh$^2$, costs in DKK/day, energy and absolute deviation in kWh/day.}",
                 r"\begin{tabular}{rrrrrr}\toprule",
                 r"$c^Q$ & $C$ & $D$ & $E^{\mathrm{day}}$ & $A$ & $N_{\mathrm{bind}}$\\\midrule"]
        for row in rows:
            latex.append(f"{row['cq']:g} & {row['procurement_cost']:.3f} & {row['disutility']:.3f} & {row['daily_energy']:.3f} & {row['absolute_deviation']:.3f} & {row['binding_hours']}" + r"\\")
        latex += [r"\bottomrule\end{tabular}", r"\end{table}"]
        (out/'q2c_metrics.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    h=base.hourly;hours=d.hours
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(11,8))
    axes[0].step(hours,d.reference_load,where='mid',color='black',ls='--',label='Reference')
    axes[0].step(hours,h['load'],where='mid',label='Optimal load')
    axes[0].step(hours,d.pv_available,where='mid',label='PV available',alpha=.65)
    axes[0].set_ylabel('Energy (kWh / 1 h)');axes[0].legend(ncol=3)
    axes[1].bar(hours,h['import'],label='Import');axes[1].bar(hours,-h['export'],label='Export (negative)')
    axes[1].step(hours,h['pv'],where='mid',color='orange',label='PV produced')
    axes[1].set_ylabel('Energy (kWh / 1 h)');axes[1].legend(ncol=3)
    axes[2].step(hours,h['effective_import_price'],where='mid',label='Effective import price')
    axes[2].step(hours,h['effective_export_price'],where='mid',label='Effective export price')
    axes[2].axhline(d.pv_marginal_cost,color='purple',ls='--',label='PV marginal cost')
    axes[2].set(xlabel='Hour',ylabel='DKK/kWh');axes[2].legend(ncol=3)
    for ax in axes: ax.grid(alpha=.25);ax.set_xticks(hours)
    fig.suptitle(f'Q2(c) base case: cQ={d.quadratic_disutility:g} DKK/kWh²; total cost={base.meta["total_cost"]:.3f} DKK')
    fig.tight_layout();fig.savefig(out/'q2c_base.png',dpi=180);fig.savefig(out/'q2c_base.pdf');plt.close(fig)
    if runs:
        qs=np.array(list(runs));tab=pd.DataFrame(rows)
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for t in [5,8,12,18]: axes[0,0].plot(qs,[r.hourly.loc[t,'load'] for r in runs.values()],'.-',label=f'Hour {t}; ref={d.reference_load[t]:g}')
        axes[0,0].set_ylabel('Load (kWh / 1 h)');axes[0,0].legend(fontsize=8)
        axes[0,1].plot(qs,tab['daily_energy'],'.-',label='Optimal daily energy');axes[0,1].axhline(d.reference_load.sum(),ls='--',color='black',label='Reference daily energy')
        axes[0,1].set_ylabel('Energy (kWh/day)');axes[0,1].legend(fontsize=8)
        axes[1,0].plot(qs,tab['absolute_deviation'],'.-',label='Absolute deviation');axes[1,0].set_ylabel('Absolute deviation (kWh/day)')
        axes[1,1].plot(qs,tab['procurement_cost'],'.-',label='Procurement cost');axes[1,1].plot(qs,tab['disutility'],'.-',label='Disutility');axes[1,1].set_ylabel('DKK/day');axes[1,1].legend(fontsize=8)
        for ax in axes.flat: ax.set_xscale('log');ax.set_xlabel('cQ (DKK/kWh²)');ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/'q2c_sweep.png',dpi=180);fig.savefig(out/'q2c_sweep.pdf');plt.close(fig)
    lin=linear_benchmark(d);lin.to_csv(out/'linear_reference_benchmark.csv')
    fig,ax=plt.subplots(figsize=(11,4))
    ax.step(hours,d.reference_load,where='mid',ls='--',color='black',label='Reference')
    ax.step(hours,h['load'],where='mid',label=f'Quadratic cQ={d.quadratic_disutility:g}')
    ax.step(hours,lin['load'],where='mid',label='Independent linear benchmark cL=1.43')
    ax.set(xlabel='Hour',ylabel='Energy (kWh / 1 h)',title='Base-profile comparison; coefficients have different units')
    ax.legend();ax.grid(alpha=.25);ax.set_xticks(hours);fig.tight_layout();fig.savefig(out/'q2c_linear_comparison.png',dpi=180);fig.savefig(out/'q2c_linear_comparison.pdf');plt.close(fig)
    # First release from the lower bound: local slope to its right.
    thresholds=[]
    for t in hours:
        a,b=d.energy_price[t]+d.import_tariff,d.energy_price[t]-d.export_tariff
        k=float(np.clip(d.pv_marginal_cost,b,a)) if d.load_min_kWh<d.pv_available[t] else a
        gap=d.reference_load[t]-d.load_min_kWh
        thresholds.append({'hour':int(t),'reference_load':d.reference_load[t],'lower_bound_release_cq':k/(2*gap) if gap>0 else None})
    pd.DataFrame(thresholds).to_csv(out/'lower_bound_thresholds.csv',index=False)
    (out/'input_snapshot.json').write_text(json.dumps({'question':d.question,'price':d.energy_price.tolist(),'pv_available':d.pv_available.tolist(),'reference_load':d.reference_load.tolist(),'pv_marginal_cost':d.pv_marginal_cost,'import_tariff':d.import_tariff,'export_tariff':d.export_tariff,'load_min':d.load_min_kWh,'load_max':d.load_max_kWh,'base_cq':d.quadratic_disutility,'sweep':list(runs),'solver_version':base.meta['solver_version']},indent=2))
    print(json.dumps(base.meta,indent=2))
    if rows: print(pd.DataFrame(rows)[['cq','procurement_cost','disutility','daily_energy','absolute_deviation','binding_hours']].to_string(index=False))
    print(f'Outputs: {out.resolve()}')
    return base,runs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cq',type=float,default=None)
    p.add_argument('--sweep',nargs='+',type=float,default=DEFAULT_SWEEP)
    p.add_argument('--base-only',action='store_true')
    p.add_argument('--output-dir',default='results/Q2_quadratic')
    args=p.parse_args();reproduce(args.cq,[] if args.base_only else args.sweep,args.output_dir)
if __name__=='__main__':main()
