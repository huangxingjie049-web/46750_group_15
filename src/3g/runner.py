"""Q3(g)(iii)--(v): two terminal treatments, comparisons and optional sweeps."""
import argparse
from dataclasses import replace,fields
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_question
from src.model_q3 import DailyEnergyConsumerModel
from src.model_q3_battery import BatteryConsumerModel
from src.validation_q3 import validate_solution
from src.validation_q3_battery import validate_battery_solution
from run_q3 import save_figure

ROOT=Path(__file__).resolve().parents[2]

def checked_solve(d,mode,pi):
    m=BatteryConsumerModel(d,terminal_mode=mode,terminal_value=pi).build()
    r=m.solve();v=validate_battery_solution(m,r)
    if not v['passed']: raise RuntimeError(f'{mode} validation failed: {v}')
    return m,r,v


def row(label,r,base,pi):
    # Same initial stock held idle is the reference for the value formulation.
    # W_batt - (W_no_batt + pi*E0) avoids counting initial inventory as profit.
    return {'scenario':label,**r.meta,'objective_DKK':r.objective,
        'import_kWh':float(r.hourly['import'].sum()),'export_kWh':float(r.hourly['export'].sum()),
        'pv_kWh':float(r.hourly.pv.sum()),
        'current_day_cost_saving_DKK':base.meta['total_cost']-r.meta['total_cost'],
        'inventory_adjusted_battery_value_DKK':r.objective-base.objective-pi*r.meta.get('initial_soc_kWh',0)}


def plots(d,results,out):
    t=d.hours
    fig,axes=plt.subplots(len(results),1,sharex=True,figsize=(11,2.7*len(results)))
    for ax,(label,r) in zip(axes,results.items()):
        ax.step(t,r.hourly.load,where='mid',label='Load',color='black')
        ax.step(t,r.hourly.pv,where='mid',label='PV',color='orange')
        ax.bar(t,r.hourly['import'],alpha=.5,label='Import')
        ax.bar(t,-r.hourly['export'],alpha=.5,label='Export (negative)')
        ax.set(title=label,ylabel='kWh per hour');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Hour')
    save_figure(fig,out,'q3g_energy_flows')
    fig,ax=plt.subplots(figsize=(11,4))
    ax.step(t,d.reference_load,where='mid',ls='--',color='black',label='Reference')
    for label,r in results.items(): ax.step(t,r.hourly.load,where='mid',label=label)
    ax.set(xlabel='Hour',ylabel='Load (kWh per hour)');ax.legend();ax.grid(alpha=.2)
    save_figure(fig,out,'q3g_load_comparison')
    battery=[(label,r) for label,r in results.items() if 'soc' in r.hourly]
    fig,axes=plt.subplots(len(battery),2,figsize=(12,3.6*len(battery)),squeeze=False)
    for axs,(label,r) in zip(axes,battery):
        h=r.hourly
        axs[0].bar(t,h.charge,label='Charge');axs[0].bar(t,-h.discharge,label='Discharge (negative)')
        axs[0].set(title=label,xlabel='Hour',ylabel='Power (kW)');axs[0].legend()
        # Initial state at boundary 0; e_t occurs at boundary t+1.
        axs[1].step(np.arange(d.n_hours+1),np.r_[d.battery_initial_soc_kWh,h.soc],where='post',label='Stored energy')
        axs[1].axhline(d.battery_initial_soc_kWh,ls='--',color='grey',label='Initial energy')
        axs[1].axhline(d.battery_capacity_kWh,ls=':',color='black',label='Capacity')
        axs[1].set(xlabel='Hour boundary',ylabel='Stored energy (kWh)',ylim=(-.1,d.battery_capacity_kWh+.3))
        axs[1].legend(fontsize=8)
        for ax in axs: ax.grid(alpha=.2)
    save_figure(fig,out,'q3g_battery_dispatch')


def sensitivity(d,modes,base,args,out):
    plan={'status':'ILLUSTRATIVE ranges; replace with team-agreed values before final reporting',
        'environment_parameter':'multiplicative energy-price scale (tariffs and PV cost fixed)',
        'price_scales':args.price_scales,'technical_parameter':'battery capacity (initial energy and power limits fixed)',
        'capacities_kWh':args.capacities,'terminal_values_DKK_per_kWh':args.pi_values,
        'method':'Each parameter varied independently; all other inputs fixed.'}
    (out/'sensitivity_plan.json').write_text(json.dumps(plan,indent=2))
    rows=[]
    for mode,pi in modes:
        for parameter,values in [('price_scale',args.price_scales),('capacity_kWh',args.capacities),('terminal_value',args.pi_values if mode=='value' else [])]:
            for value in values:
                ds=replace(d,energy_price=d.energy_price*value) if parameter=='price_scale' else replace(d,battery_capacity_kWh=value) if parameter=='capacity_kWh' else d
                this_pi=value if parameter=='terminal_value' else pi
                _,r,v=checked_solve(ds,mode,this_pi)
                # Re-solve no battery under the SAME environmental inputs.
                nb=no_battery_data(ds);ref=DailyEnergyConsumerModel(nb).build().solve()
                item=row(mode,r,ref,this_pi)
                rows.append({'parameter':parameter,'parameter_value':value,'validation_passed':v['passed'],**item})
    table=pd.DataFrame(rows);table.to_csv(out/'sensitivity_metrics.csv',index=False)
    for parameter,group in table.groupby('parameter'):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for mode,g in group.groupby('scenario'):
            g=g.sort_values('parameter_value')
            for ax,key,title in zip(axes,['total_cost','final_soc_kWh','inventory_adjusted_battery_value_DKK'],['Current-day total cost (DKK)','Final energy (kWh)','Inventory-adjusted battery value (DKK)']):
                ax.plot(g.parameter_value,g[key],marker='o',label=mode);ax.set(xlabel=parameter,ylabel=title);ax.grid(alpha=.2);ax.legend()
        save_figure(fig,out,'q3g_sensitivity_'+parameter)


def no_battery_data(d):
    return replace(d,question='Q3g_no_battery',**{f.name:None for f in fields(d) if f.name.startswith('battery_')})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cyclic','value','both'],default='cyclic')
    p.add_argument('--pi',type=float,default=None,help='Explicit terminal value in DKK/kWh; required for value/both.')
    p.add_argument('--sweep',action='store_true',help='Run illustrative g(v) independent sweeps.')
    p.add_argument('--price-scales',type=float,nargs='+',default=[.5,1,1.5])
    p.add_argument('--capacities',type=float,nargs='+',default=[2,4,8])
    p.add_argument('--pi-values',type=float,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--emin',type=float,default=None)
    p.add_argument('--cq',type=float,default=None)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'Q3g')
    args=p.parse_args(argv)
    if args.mode in ('value','both') and args.pi is None: p.error('--pi is required: Joe has not yet specified the terminal value.')
    if args.mode=='cyclic' and args.pi is not None: p.error('--pi applies only to value/both modes.')
    d=load_question('Q3_battery')
    if args.emin is not None: d=replace(d,min_daily_energy_kWh=args.emin)
    if args.cq is not None: d=replace(d,quadratic_disutility=args.cq)
    if args.sweep:
        for vals in [args.price_scales,args.capacities]:
            if len(set(vals))<3 or not all(np.isfinite(v) and v>0 for v in vals): p.error('Each g(v) sweep needs at least three distinct positive finite values.')
        if any(v<d.battery_initial_soc_kWh for v in args.capacities): p.error('Capacity must be at least the fixed initial energy.')
    modes=([('cyclic',0)] if args.mode in ('cyclic','both') else [])+([('value',args.pi)] if args.mode in ('value','both') else [])
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    nd=no_battery_data(d);base=DailyEnergyConsumerModel(nd).build().solve()
    checks={'no_battery':validate_solution(base,nd)}
    if not checks['no_battery']['passed']: raise RuntimeError(checks)
    base.save(out);results={'No battery':base};rows=[row('no_battery',base,base,0)]
    for mode,pi in modes:
        m,r,v=checked_solve(d,mode,pi);checks[mode]=v
        r.save(out,tag=mode);m.m.write(str(out/f'q3g_{mode}.lp'))
        results['Cyclic' if mode=='cyclic' else f'Terminal value: pi={pi:g}']=r
        rows.append(row(mode,r,base,pi))
    table=pd.DataFrame(rows);table.to_csv(out/'comparison_metrics.csv',index=False)
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    snapshot={f.name:(getattr(d,f.name).tolist() if isinstance(getattr(d,f.name),np.ndarray) else getattr(d,f.name)) for f in fields(d)}
    snapshot.update(terminal_modes=modes,terminal_reward_counted_once=True)
    (out/'input_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    plots(d,results,out)
    if args.sweep: sensitivity(d,modes,base,args,out)
    selected=['scenario','total_cost','objective_DKK','final_soc_kWh','current_day_cost_saving_DKK','inventory_adjusted_battery_value_DKK','simultaneous_charge_discharge_hours']
    print(table.reindex(columns=selected).to_string(index=False))
    print('All feasibility/accounting/KKT checks passed.')
    print(f'Outputs: {out.resolve()}')

if __name__=='__main__': main()
