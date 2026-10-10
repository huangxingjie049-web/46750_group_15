"""Physical/accounting checks and continuous-QP KKT residuals."""
import numpy as np

def validate_battery_solution(model,result,tol=2e-6):
    d,h=model.data,result.hourly
    prev=np.r_[d.battery_initial_soc_kWh,h.soc.to_numpy()[:-1]]
    dynamics=h.soc.to_numpy()-prev-d.battery_charging_efficiency*h.charge.to_numpy()+h.discharge.to_numpy()/d.battery_discharging_efficiency
    checks={'storage_balance_max':float(np.max(abs(dynamics))),
        'power_balance_max':float((h.pv+h['import']+h.discharge-h.load-h['export']-h.charge).abs().max()),
        'daily_requirement_violation':float(max(0,d.min_daily_energy_kWh-h.load.sum()))}
    for name,lo,hi in [('load',d.load_min_kWh,d.load_max_kWh),('pv',0,d.pv_available),('import',0,np.inf),('export',0,np.inf),('charge',0,d.battery_max_charge_kW),('discharge',0,d.battery_max_discharge_kW),('soc',0,d.battery_capacity_kWh)]:
        checks[name+'_bound_violation']=float(max(0,np.max(lo-h[name]),np.max(h[name]-hi)))
    if model.terminal_mode=='cyclic': checks['terminal_soc_error']=float(abs(h.soc.iloc[-1]-d.battery_initial_soc_kWh))
    cost=float((d.quadratic_disutility*(h.load-d.reference_load)**2+d.pv_marginal_cost*h.pv+(d.energy_price+d.import_tariff)*h['import']-(d.energy_price-d.export_tariff)*h['export']).sum())
    checks['objective_error']=float(abs(result.objective+cost-model.terminal_value*h.soc.iloc[-1]))
    m=model.m;vars_=m.getVars();cons=m.getConstrs()
    x=np.array([v.X for v in vars_]);pi=np.array([c.Pi for c in cons])
    rhs=np.array([c.RHS for c in cons]);A=m.getA();slack=A@x-rhs
    grad=np.zeros(len(vars_));index={v.VarName:i for i,v in enumerate(vars_)}
    for i,t in enumerate(d.hours):
        grad[index[model.var['load'][t].VarName]]=-2*d.quadratic_disutility*(h.loc[t,'load']-d.reference_load[i])
        grad[index[model.var['pv'][t].VarName]]=-d.pv_marginal_cost
        grad[index[model.var['import'][t].VarName]]=-(d.energy_price[i]+d.import_tariff)
        grad[index[model.var['export'][t].VarName]]=d.energy_price[i]-d.export_tariff
    grad[index[model.var['soc'][d.hours[-1]].VarName]]=model.terminal_value
    checks['stationarity_max']=float(np.max(abs(grad-A.T@pi)))
    checks['complementarity_max']=float(np.max(abs(pi*slack)))
    checks['dual_sign_violation']=float(max(max(0,p) if c.Sense=='>' else max(0,-p) if c.Sense=='<' else 0 for c,p in zip(cons,pi)))
    checks['passed']=bool(max(checks.values())<=tol)
    checks['tolerance']=tol
    checks['simultaneous_charge_discharge_hours']=result.meta['simultaneous_charge_discharge_hours']
    checks['note']='Overlap is reported for g(ii)/(iv), not automatically a feasibility error.'
    return checks
