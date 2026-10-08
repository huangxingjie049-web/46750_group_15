"""Q2(c): convex quadratic reference-based disutility, without daily coupling."""
from dataclasses import dataclass
import json
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from .model_q1 import FlexibleConsumerModel, Results, _dual

@dataclass
class QuadraticResults(Results):
    def save(self, folder: Path | str, tag: str = '') -> None:
        super().save(folder, tag)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        (Path(folder)/f'{stem}_metrics.json').write_text(json.dumps(self.meta,indent=2),encoding='utf-8')

class QuadraticConsumerModel(FlexibleConsumerModel):
    """Reuse the Q1 solve interface; change the preference term and accounting."""
    def _check_intertemporal_inputs(self):
        """Q2(c) has no daily energy or storage; subclasses may add coupling."""
        if self.data.min_daily_energy_kWh is not None or self.data.battery_capacity_kWh is not None:
            raise ValueError('Daily energy requirements and storage belong to Q3, not Q2(c).')

    def _add_intertemporal_constraints(self):
        """Extension hook; the Q2(c) model has only hourly constraints."""

    def build(self):
        d,m,T=self.data,self.m,self.T
        if d.reference_load is None or d.quadratic_disutility is None:
            raise ValueError('Q2(c) requires a reference profile and quadratic coefficient.')
        if not np.isfinite(d.quadratic_disutility) or d.quadratic_disutility<=0:
            raise ValueError('The quadratic disutility coefficient must be finite and positive.')
        if d.linear_disutility not in (None,0.) or d.consumption_utility not in (None,0.):
            raise ValueError('Q2(c) replaces utility and has no linear disutility term.')
        self._check_intertemporal_inputs()
        if not (0<=d.load_min_kWh<=d.load_max_kWh) or np.any(d.pv_available<0):
            raise ValueError('Invalid load or PV bounds.')
        if np.any(d.energy_price+d.import_tariff <= d.energy_price-d.export_tariff):
            raise ValueError('This implementation requires a strictly positive trading spread.')
        if not np.isfinite(d.reference_load).all():
            raise ValueError('Reference consumption must be finite.')
        m.Params.BarConvTol=1e-10
        m.Params.FeasibilityTol=1e-9
        m.Params.OptimalityTol=1e-9
        for name in ['load','pv','import','export']:
            self.var[name]=m.addVars(T,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name=name)
        L,P,I,E=(self.var[n] for n in ['load','pv','import','export'])
        self.con['balance']=m.addConstrs((P[t]+I[t]-L[t]-E[t]==0 for t in T),name='balance')
        for name,v,low,high in [('load',L,d.load_min_kWh,d.load_max_kWh),('pv',P,0.,d.pv_available)]:
            self.con[name+'_lower']=m.addConstrs((v[t]>=low for t in T),name=name+'_lower')
            self.con[name+'_upper']=m.addConstrs((v[t]<= (high[t] if name=='pv' else high) for t in T),name=name+'_upper')
        for name,v in [('import',I),('export',E)]:
            self.con[name+'_lower']=m.addConstrs((v[t]>=0 for t in T),name=name+'_lower')
        m.setObjective(gp.quicksum(
            -d.quadratic_disutility*(L[t]-d.reference_load[t])**2
            -d.pv_marginal_cost*P[t]-(d.energy_price[t]+d.import_tariff)*I[t]
            +(d.energy_price[t]-d.export_tariff)*E[t] for t in T),GRB.MAXIMIZE)
        self._add_intertemporal_constraints()
        m.update()
        return self

    def _extract_results(self,status):
        d,T=self.data,list(self.T)
        h=pd.DataFrame(index=pd.Index(T,name='hour'))
        for name,values in [('price',d.energy_price),('pv_available',d.pv_available),('reference_load',d.reference_load)]: h[name]=values
        for name,v in self.var.items(): h[name]=[v[t].X for t in T]
        duals = {}
        for name,c in self.con.items():
            if isinstance(c, gp.tupledict):
                h['dual_'+name]=[_dual(c[t]) for t in T]
            else:
                duals[name] = float(_dual(c))
        h['effective_import_price']=d.energy_price+d.import_tariff
        h['effective_export_price']=d.energy_price-d.export_tariff
        h['deviation']=h['load']-h['reference_load']
        h['absolute_deviation']=h['deviation'].abs()
        h['disutility']=d.quadratic_disutility*h['deviation']**2
        h['procurement_cost']=d.pv_marginal_cost*h['pv']+h['effective_import_price']*h['import']-h['effective_export_price']*h['export']
        h['curtailment']=h['pv_available']-h['pv']
        tol=1e-6
        lower=np.abs(h['load']-d.load_min_kWh)<=tol
        upper=np.abs(h['load']-d.load_max_kWh)<=tol
        h['lower_bound_binding']=lower
        h['upper_bound_binding']=upper
        h['deviation_bound_binding']=lower|upper
        C,D=float(h['procurement_cost'].sum()),float(h['disutility'].sum())
        meta={'cq':float(d.quadratic_disutility),'procurement_cost':C,'disutility':D,
              'total_cost':C+D,'net_utility':-C-D,'daily_energy':float(h['load'].sum()),
              'absolute_deviation':float(h['absolute_deviation'].sum()),
              'squared_deviation':float((h['deviation']**2).sum()),
              'binding_hours':int((lower|upper).sum()),'lower_bound_hours':int(lower.sum()),'upper_bound_hours':int(upper.sum()),
              'binding_tolerance':tol,'above_reference_hours':int((h['deviation']>tol).sum()),
              'simultaneous_trade_hours':int(((h['import']>tol)&(h['export']>tol)).sum()),
              'balance_residual_max':float((h['pv']+h['import']-h['load']-h['export']).abs().max()),
              'objective_accounting_error':float(abs(self.m.ObjVal+C+D)),
              'solver':'Gurobi','solver_version':'.'.join(map(str,gp.gurobi.version()))}
        return QuadraticResults(d.question,status,float(self.m.ObjVal),h,duals,meta)
