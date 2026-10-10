"""Continuous Q3(g) battery QP. One-hour intervals; no binary exclusion."""
from dataclasses import replace
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.model_q3 import DailyEnergyConsumerModel

class BatteryConsumerModel(DailyEnergyConsumerModel):
    def __init__(self,data,terminal_mode='cyclic',terminal_value=0.):
        super().__init__(data)
        self.terminal_mode=terminal_mode
        self.terminal_value=float(terminal_value)

    def _check_intertemporal_inputs(self):
        d=self.data
        self.data=replace(d,battery_capacity_kWh=None)
        try: super()._check_intertemporal_inputs()
        finally: self.data=d
        if self.terminal_mode not in ('cyclic','value'):
            raise ValueError('terminal_mode must be cyclic or value.')
        if not np.isfinite(self.terminal_value) or self.terminal_value<0:
            raise ValueError('Terminal value must be finite and nonnegative.')
        if self.terminal_mode=='cyclic' and self.terminal_value!=0:
            raise ValueError('Cyclic mode uses pi=0.')
        for name in ('battery_capacity_kWh','battery_max_charge_kW','battery_max_discharge_kW','battery_initial_soc_kWh'):
            v=getattr(d,name)
            if v is None or not np.isfinite(v) or v<0: raise ValueError(f'Invalid {name}.')
        if d.battery_initial_soc_kWh>d.battery_capacity_kWh: raise ValueError('Initial energy exceeds capacity.')
        for name in ('battery_charging_efficiency','battery_discharging_efficiency'):
            v=getattr(d,name)
            if v is None or not np.isfinite(v) or not 0<v<=1: raise ValueError(f'Invalid {name}.')
        if d.battery_final_soc_kWh is not None:
            raise ValueError('Choose terminal_mode instead of battery_final_soc_kWh.')

    def _add_intertemporal_constraints(self):
        super()._add_intertemporal_constraints()
        d,m,T=self.data,self.m,list(self.T)
        for name in ('charge','discharge','soc'): self.var[name]=m.addVars(T,lb=-GRB.INFINITY,name=name)
        C,D,S=(self.var[n] for n in ('charge','discharge','soc'))
        for name,v,upper in [('charge',C,d.battery_max_charge_kW),('discharge',D,d.battery_max_discharge_kW),('soc',S,d.battery_capacity_kWh)]:
            self.con[name+'_lower']=m.addConstrs((v[t]>=0 for t in T),name=name+'_lower')
            self.con[name+'_upper']=m.addConstrs((v[t]<=upper for t in T),name=name+'_upper')
        m.remove(list(self.con['balance'].values()))
        L,P,I,E=(self.var[n] for n in ('load','pv','import','export'))
        self.con['balance']=m.addConstrs((P[t]+I[t]+D[t]-L[t]-E[t]-C[t]==0 for t in T),name='battery_balance')
        self.con['storage_balance']=gp.tupledict()
        for i,t in enumerate(T):
            previous=d.battery_initial_soc_kWh if i==0 else S[T[i-1]]
            self.con['storage_balance'][t]=m.addConstr(S[t]-previous-d.battery_charging_efficiency*C[t]+D[t]/d.battery_discharging_efficiency==0,name=f'storage_balance[{t}]')
        if self.terminal_mode=='cyclic': self.con['terminal_soc']=m.addConstr(S[T[-1]]==d.battery_initial_soc_kWh,name='terminal_soc')
        else:
            m.update()
            # Count terminal reward ONCE, outside the hourly sum.
            m.setObjective(m.getObjective()+self.terminal_value*S[T[-1]],GRB.MAXIMIZE)

    def _extract_results(self,status):
        r=super()._extract_results(status);h,d=r.hourly,self.data
        terminal=float(h.soc.iloc[-1]);credit=self.terminal_value*terminal
        h['battery_loss']=(1-d.battery_charging_efficiency)*h.charge+(1/d.battery_discharging_efficiency-1)*h.discharge
        overlap=(h.charge>1e-6)&(h.discharge>1e-6)
        r.meta.update(terminal_mode=self.terminal_mode,terminal_value_DKK_per_kWh=self.terminal_value,
            initial_soc_kWh=float(d.battery_initial_soc_kWh),final_soc_kWh=terminal,
            terminal_credit_DKK=credit,net_utility=float(r.objective),
            adjusted_cost_DKK=float(r.meta['total_cost']-credit),
            objective_accounting_error=float(abs(r.objective+r.meta['total_cost']-credit)),
            balance_residual_max=float((h.pv+h['import']+h.discharge-h.load-h['export']-h.charge).abs().max()),
            simultaneous_charge_discharge_hours=int(overlap.sum()),
            simultaneous_charge_discharge_indices=[int(t) for t in h.index[overlap]],
            charge_kWh=float(h.charge.sum()),discharge_kWh=float(h.discharge.sum()),
            battery_losses_kWh=float(h.battery_loss.sum()))
        return r
