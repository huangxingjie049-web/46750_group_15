"""Q1 steps 1-3: reproducible base-case experiment, figures and independent LP check.

Run python base_case.py. Only input JSONs in data/ are used; they are not edited.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from main import ROOT, CommunityModel, load_data, net_import_shares, TOL
from consistency_checks import report_equation_checks, terminal_condition_check


def highs_check(model):
    """Solve the assembled LP independently with HiGHS; compare cost J in DKK.

    This checks the numerical solution, not an independent physical formulation.
    Different optimal dispatches are expected for linear preferences.
    """
    m = model.model
    variables, rows = m.getVars(), m.getConstrs()
    A = m.getA().tocsr()
    rhs = np.array([r.RHS for r in rows])
    eq = np.array([r.Sense == '=' for r in rows])
    signs = np.array([-1 if r.Sense == '>' else 1 for r in rows])
    ub = ~eq
    bounds = [(v.LB if v.LB > -1e90 else None,
               v.UB if v.UB < 1e90 else None) for v in variables]
    result = linprog(np.array([v.Obj for v in variables]),
        A_ub=A[ub].multiply(signs[ub,None]).tocsr(), b_ub=rhs[ub]*signs[ub],
        A_eq=A[eq], b_eq=rhs[eq], bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    error = abs(result.fun-m.ObjVal)
    if error > TOL:
        raise AssertionError(f'HiGHS objective difference {error}')
    return dict(success=True, objective_difference_DKK=error,
                note='Independent solver, same assembled LP; dispatch may differ.')


MODEL_LABELS = dict(R='R: Reference case', I='I: Individual contracts model',
                    C='C: Community contract model')
MODEL_STYLES = dict(
    R=dict(color='#111111', linestyle='--', linewidth=1.8),
    I=dict(color='#cc1526', linestyle='-', linewidth=1.9),
    C=dict(color='#0057b8', linestyle='-.', linewidth=1.9,
           marker='o', markersize=3.5, markerfacecolor='white', markevery=3))


def style_axes(axes):
    """Use a clean shared layout with subdued horizontal guides and open spines."""
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#444444')
        ax.grid(axis='y',color='#dddddd',linewidth=.6,alpha=.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.set(xlim=(-.4,23.4),xticks=[0,4,8,12,16,20,23],xlabel='Hour')


def figures(results, data, out):
    """Save content-only titles, full model legends and consistent black/red/blue curves."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    x=np.arange(24)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':12,'axes.labelsize':11,
                         'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(12.8,4.9),sharex=True)
        for key,result in results.items():
            for ax,field in zip(axes,['import_kWh','feeder_exceedance_kWh']):
                ax.step(x,[r[field] for r in result.hourly],where='mid',
                        label=MODEL_LABELS[key],**MODEL_STYLES[key])
        cap=np.array([r['cap_kW'] for r in results['R'].hourly])
        cap_label=(f'Feeder capacity ({cap[0]:g} kW)' if np.allclose(cap,cap[0])
                   else 'Feeder capacity')
        axes[0].plot(x,cap,color='#858585',linestyle=':',linewidth=1.4,
                     label=cap_label,zorder=1)
        axes[0].set(ylabel='Import (kW)',title='(a) Feeder import')
        axes[1].set(ylabel='Exceedance (kW)',title='(b) Capacity exceedance')
        style_axes(axes)
        fig.suptitle('Feeder import and capacity exceedance',fontsize=15,y=.97)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),
                   ncol=2,frameon=False,fontsize=10,handlelength=3.2,columnspacing=3)
        fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.26,wspace=.24)
        for suffix in ('png','pdf'): fig.savefig(out/f'feeder_comparison.{suffix}',dpi=300)
        plt.close(fig)

        fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharex=True)
        ref=np.sum([a.ref for a in data.members],axis=0)
        for key,result in results.items():
            rows=result.members
            aggregate=lambda field: np.array([sum(r[field] for r in rows
                                                  if r['hour']==t) for t in range(24)])
            series=[aggregate('L_kWh')-ref,
                    aggregate('charge_kW')-aggregate('discharge_kW'),aggregate('PV_kWh')]
            for ax,y in zip(axes,series):
                ax.plot(x,y,label=MODEL_LABELS[key],**MODEL_STYLES[key])
        axes[0].set(ylabel='Load deviation (kWh)',title='(a) Load relative to reference')
        axes[1].set(ylabel='Charge minus discharge (kW)',title='(b) Battery dispatch')
        axes[2].set(ylabel='PV output (kWh)',title='(c) PV generation')
        style_axes(axes)
        fig.suptitle('Load, battery and PV dispatch',fontsize=15,y=.97)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),
                   ncol=3,frameon=False,fontsize=10,handlelength=3.2,columnspacing=3)
        fig.subplots_adjust(left=.06,right=.985,top=.80,bottom=.23,wspace=.32)
        for suffix in ('png','pdf'): fig.savefig(out/f'dispatch_response.{suffix}',dpi=300)
        plt.close(fig)


def replot_saved_results():
    """Regenerate figures from saved CSV dispatch; do not re-solve the optimization."""
    from types import SimpleNamespace
    out=ROOT/'results'/'base_case'
    def read_records(path):
        with path.open(encoding='utf-8',newline='') as f:
            records=list(csv.DictReader(f))
        for row in records:
            for key,value in row.items():
                try: row[key]=float(value)
                except (ValueError,TypeError): pass
        return records
    results={key:SimpleNamespace(hourly=read_records(out/key/'hourly.csv'),
                                members=read_records(out/key/'members.csv'))
             for key in MODEL_LABELS}
    figures(results,load_data(ROOT/'data'),out)
    print('Regenerated both figures from saved dispatch:',out)


def run_base_case(out=None):
    """Return model/result dictionaries after base-case and limit-case validation."""
    out=Path(out) if out is not None else ROOT/'results'/'base_case'
    out.mkdir(parents=True,exist_ok=True)
    data=load_data(ROOT/'data')
    assert len(data.members)==16
    assert sorted(set(a.cL for a in data.members))==[.25,.5,1.,2.]
    assert sum(a.capacity>0 for a in data.members)==8
    assert sum(max(a.pv)>0 for a in data.members)==8
    models={'R':CommunityModel(data,'R')}
    results={'R':models['R'].run()}
    shares=net_import_shares(results['R'])
    for key in ('I','C'):
        models[key]=CommunityModel(data,key,shares=shares)
        results[key]=models[key].run()
    checks={key:highs_check(model) for key,model in models.items()}
    equation_checks={key:report_equation_checks(model) for key,model in models.items()}
    terminal_check=terminal_condition_check(data)
    official=data.contract['reference_model_R_by_day']['d1']
    for field, tolerance in [('daily_import_kWh',.05),('peak_import_kW',.005),('cost_DKK',.005)]:
        actual=results['R'].metrics['total_cost_DKK' if field=='cost_DKK' else field]
        assert abs(actual-official[field])<tolerance,(field,actual,official[field])
    assert results['C'].metrics['total_cost_DKK']<=results['I'].metrics['total_cost_DKK']+TOL
    for result in results.values():
        assert result.metrics['battery_overlap_hours']==result.metrics['trade_overlap_hours']==0
        assert abs(result.metrics['total_cost_DKK']-sum(result.metrics[f] for f in
            ['energy_cost_DKK','tariff_cost_DKK','disutility_cost_DKK','PV_cost_DKK','penalty_DKK']))<TOL
    limit=CommunityModel(data,'I',shares=shares,beta=0,alpha=0,cap=[1e5]*24).run()
    error=abs(limit.metrics['W_DKK']-results['R'].metrics['W_DKK'])
    assert error<TOL
    summary=[]
    for key,result in results.items():
        models[key].save(out/key)
        row=dict(result.metrics)
        row['gain_vs_R_DKK']=results['R'].metrics['total_cost_DKK']-row['total_cost_DKK']
        summary.append(row)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    payload=dict(results=summary,independent_solver=checks,shares=shares,
        report_equation_checks=equation_checks, terminal_condition_check=terminal_check,
        no_contract_limit_error_DKK=error,official_R=official,
        reference_exceedance_note='Official 66.5 kWh is dispatch-dependent; run audit_reference_ties.py.',
        scope='Q1 suggested steps 1-3 only; no sensitivity or settlement study.')
    (out/'validation_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    figures(results,data,out)
    print('All base-case checks passed, including independent HiGHS solutions.')
    for r in summary:
        print(f"{r['variant']}: cost={r['total_cost_DKK']:.6f} DKK; peak={r['peak_import_kW']:.6f} kW; "
              f"exceedance={r['feeder_exceedance_kWh']:.6f} kWh; hours={r['hours_above_cap']}")
    return data,models,results,payload


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true',help='Redraw saved results without solving')
    args=parser.parse_args()
    replot_saved_results() if args.plots_only else run_base_case()
