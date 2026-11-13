"""Reproduce Q1 step 5: coarse search, threshold refinement and parameter sweeps.
Run: python sensitivity.py
Use the supplied JSON data and the split imports of Q1 equations (15) and (17).
"""
from pathlib import Path
import csv, json
import numpy as np
from main import CommunityModel, load_data, net_import_shares, GRB, gp, TOL

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'sensitivity'
STEP4=ROOT/'results'/'contract_value_metrics.csv'
OUT.mkdir(parents=True, exist_ok=True)
D=load_data(ROOT/'data')
rm=CommunityModel(D,'R'); R=rm.run(); SHARES=net_import_shares(R)
BASE_CAP=list(D.contract['capacity_limit_kW'])
JR=R.metrics['J_DKK']; cache={}; certificates=[]

def solve(parameter, value, variant):
    value=round(float(value),10); key=(parameter,value,variant)
    if key in cache:return cache[key]
    cap=[k*value/1.3 for k in BASE_CAP] if parameter=='f' else BASE_CAP
    if variant=='R':
        metrics=dict(R.metrics); hourly=R.hourly; member_rows=R.members; slope=0.; binding=0
        physical_exceedance=0.
    else:
        kw={'beta':value} if parameter=='beta' else {'alpha':value} if parameter=='alpha' else {'cap':cap}
        model=CommunityModel(D,variant,shares=SHARES,**kw); result=model.run()
        metrics=dict(result.metrics); hourly=result.hourly; member_rows=result.members
        certificates.append(max(result.certificate.values()))
        physical_exceedance = (
            metrics['feeder_exceedance_kWh'] if variant == 'C' else
            sum(max(0., r['import_kWh']-SHARES[r['member']]*cap[r['hour']])
                for r in member_rows)
        )
        if model.alpha > (1-model.beta)*D.tau_imp:
            assert abs(metrics['charged_exceedance_kWh']-physical_exceedance)<TOL
        if parameter=='beta':
            slope=-sum(D.tau_imp*h['import_within_cap_kWh']+D.tau_exp*h['export_kWh']
                       for h in hourly)
        elif parameter=='alpha':
            slope=metrics['charged_exceedance_kWh']
        else:
            slope=sum(c.Pi*(BASE_CAP[t]/1.3)*(SHARES[i] if variant=='I' else 1.) for k,c in model.constraints.cap.items() for i,t in ([k] if variant=='I' else [(None,k)]))
        binding=sum(abs(c.Slack)<TOL for c in model.constraints.cap.values())
    V=sum(max(0,h['import_kWh']-cap[h['hour']]) for h in hourly)
    row=dict(parameter=parameter,value=value,model=variant,J=metrics['J_DKK'],gain_vs_R=JR-metrics['J_DKK'],V=V,peak=max(h['import_kWh'] for h in hourly),H=sum(h['import_kWh']>cap[h['hour']]+TOL for h in hourly),charged_exceedance=metrics['charged_exceedance_kWh'],cap_kW=cap[0],energy_cost=metrics['energy_cost_DKK'],tariff_cost=metrics['tariff_cost_DKK'],disutility_cost=metrics['disutility_cost_DKK'],PV_cost=metrics['PV_cost_DKK'],penalty=metrics['penalty_DKK'],cost_slope=slope,binding_caps=binding,physical_contract_exceedance=physical_exceedance)
    row['pooling_gain']=None
    assert abs(row['J']-sum(row[k] for k in ('energy_cost','tariff_cost',
        'disutility_cost','PV_cost','penalty')))<TOL
    for level in [1,2,3,4]:
        rows=[r for r in member_rows if r['member'].startswith('C'+str(level))]
        row['deviation_class_'+str(level)]=sum(r['deviation_kWh'] for r in rows)
    row['battery_throughput']=sum(r['charge_kW']+r['discharge_kW'] for r in member_rows)
    row['PV_generation']=sum(r['PV_kWh'] for r in member_rows)
    if variant!='R':model.model.dispose()
    cache[key]=row
    return row

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def add_pooling_gains(rows):
    for row in rows:
        p, x = row['parameter'], row['value']
        row['pooling_gain'] = solve(p,x,'I')['J']-solve(p,x,'C')['J']


def verify_base_metrics():
    """Match each sweep's base point to the six metrics from Step 4."""
    fields = {'J':'J_DKK', 'gain_vs_R':'gain_vs_R_DKK',
              'V':'feeder_exceedance_kWh', 'peak':'peak_import_kW',
              'H':'hours_above_cap', 'pooling_gain':'gain_C_vs_I_DKK'}
    base = {m:solve('alpha',10.,m) for m in ('R','I','C')}
    add_pooling_gains(base.values())
    compared_to_step4 = STEP4.is_file()
    if compared_to_step4:
        with STEP4.open(newline='', encoding='utf-8') as handle:
            expected = {row['variant']:row for row in csv.DictReader(handle)}
    else:
        # A standalone sensitivity run still checks agreement across its base points.
        expected = {m:{column:row[field] for field,column in fields.items()}
                    for m,row in base.items()}

    errors = []
    for p,x in [('beta',.75),('f',1.3),('alpha',10.)]:
        for m in ('R','I','C'):
            row = solve(p,x,m)
            for field,column in fields.items():
                if field=='pooling_gain' and m!='C':
                    continue
                error = abs(row[field]-float(expected[m][column]))
                if error>=TOL:
                    raise ValueError(f'{p} base point, {m}: {field} differs by {error:g}')
                errors.append(error)
    return dict(compared_to_step4_csv=compared_to_step4, max_absolute_difference=max(errors))


coarse={'beta':list(np.linspace(0,1,21)), 'f':sorted(set([.65,1.,1.3]+list(np.arange(.75,8.001,.25)))), 'alpha':[0,.1,.25,.5,.75,1,1.25,1.5,2,2.5,3,3.5,4,4.2,4.5,5,6,8,10,15,20,30]}
coarse_rows=[solve(p,x,m) for p,xx in coarse.items() for x in xx for m in ['R','I','C']]
add_pooling_gains(coarse_rows)
base_check=verify_base_metrics()
print('Coarse search complete:',len(cache),'model points',flush=True)

def bisect(parameter,variant,field,target,lo,hi,tolerance=1e-7):
    a=solve(parameter,lo,variant)[field]-target
    b=solve(parameter,hi,variant)[field]-target
    if a*b>0:return None
    for _ in range(45):
        mid=(lo+hi)/2; c=solve(parameter,mid,variant)[field]-target
        if hi-lo<tolerance:break
        if a*c<=0:hi=mid;b=c
        else:lo=mid;a=c
    return dict(parameter=parameter,model=variant,field=field,target=target,lo=lo,hi=hi,value=(lo+hi)/2)

thresholds=[]
for p,xx in coarse.items():
    for m in ['I','C']:
        for lo,hi in zip(xx[:-1],xx[1:]):
            a=solve(p,lo,m)['J']-JR;b=solve(p,hi,m)['J']-JR
            if a*b<0:
                t=bisect(p,m,'J',JR,lo,hi);t['kind']='signing';thresholds.append(t)

floors={}
for variant in ['I','C']:
    model=CommunityModel(D,variant,shares=SHARES);model.build()
    model.model.setObjective(gp.quicksum(model.variables.excess.values()),GRB.MINIMIZE)
    model.model.optimize();assert model.model.Status==GRB.OPTIMAL
    floors[variant]=model.model.ObjVal;model.model.dispose()
    # Locate the first penalty reaching the minimum possible charged exceedance.
    lo,hi=0.,30.
    assert solve('alpha',hi,variant)['charged_exceedance']<=floors[variant]+1e-5
    for _ in range(28):
        mid=(lo+hi)/2
        if solve('alpha',mid,variant)['charged_exceedance']<=floors[variant]+1e-5:hi=mid
        else:lo=mid
    thresholds.append(dict(parameter='alpha',model=variant,kind='minimum_exceedance_plateau',value=hi,lo=lo,hi=hi,minimum_charged_exceedance=floors[variant]))

grids={}
for p,xx in coarse.items():
    extra=[];width={'beta':.05,'f':.15,'alpha':.5}[p]
    for t in thresholds:
        if t['parameter']==p:
            x=t['value'];extra.extend(np.linspace(max(xx[0],x-width),min(xx[-1],x+width),21));extra.extend([x-width/100,x+width/100])
    # Refine intervals in which optimal cost slopes change.
    def refine(lo,hi,depth):
        if depth==0:return
        if any(abs(solve(p,lo,m)['cost_slope']-solve(p,hi,m)['cost_slope'])>1e-5 for m in ['I','C']):
            mid=(lo+hi)/2;extra.append(mid)
            refine(lo,mid,depth-1);refine(mid,hi,depth-1)
    for lo,hi in zip(xx[:-1],xx[1:]):refine(lo,hi,2)
    grids[p]=sorted(set(round(float(x),9) for x in xx+extra if xx[0]<=x<=xx[-1]))

rows=[solve(p,x,m) for p,xx in grids.items() for x in xx for m in ['R','I','C']]
for p,xx in grids.items():
    for x in xx:assert solve(p,x,'C')['J']<=solve(p,x,'I')['J']+TOL
    for m in ['I','C']:
        costs=[solve(p,x,m)['J'] for x in xx]
        assert all((b>=a-1e-5 if p=='alpha' else b<=a+1e-5) for a,b in zip(costs[:-1],costs[1:])),(p,m)
add_pooling_gains(rows)
save_csv(OUT/'coarse_search.csv',coarse_rows)
save_csv(OUT/'sensitivity_metrics.csv',rows)
(OUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
for p,x in sorted({key[:2] for key in cache}):
    evaluation = [cache.get((p,x,m)) for m in ('R','I','C')]
    if evaluation[1] is not None and evaluation[2] is not None:
        add_pooling_gains(row for row in evaluation if row is not None)
save_csv(OUT/'all_evaluations.csv',list(cache.values()))
summary=dict(reference_cost=JR,reference_exceedance=R.metrics['feeder_exceedance_kWh'],shares=SHARES,minimum_charged_exceedance=floors,thresholds=thresholds,grid_points={k:len(v) for k,v in grids.items()},unique_model_points=len(cache),max_optimality_certificate_residual=max(certificates),base_case_metrics_check=base_check,notes=['Each experiment changes one contract parameter only.','R dispatch and cap shares fixed from one reference solve.','Cap = supplied hourly cap * f/1.3; shape unchanged.','Above-cap imports pay the day-ahead price plus the penalty.','Charged exceedance is the LP variable x. Physical contract exceedance is reconstructed from metered flows and may differ at low penalties.','R exceedance re-evaluated at each cap; R objective is constant.','Fixed primal-simplex solver; alternate LP optima can change dispatch metrics.','No assertion of monotonic aggregate V for I: penalty targets individual excess.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

print(json.dumps(summary,indent=2),flush=True)
from plot_results import plot
plot()
