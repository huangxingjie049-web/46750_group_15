"""Audit dispatch directly against the latest report equations, not just solver rows."""
from copy import deepcopy
from main import CommunityModel, TOL


def report_equation_checks(model):
    """Return physical/accounting residuals in kWh/DKK for a solved report model."""
    d, v, m = model.data, model.variables, model.model
    errors = dict(load_bounds=0., daily_minimum=0., deviation_split=0.,
                  absolute_deviation=0., net_definition=0., battery_dynamics=0.,
                  asset_bounds=0., terminal_lower_bound=0., balance=0.,
                  cap=0., objective_reconstruction_DKK=0.)
    def keep(key, value): errors[key] = max(errors[key], value)
    private = 0.
    for a in d.members:
        i = a.id
        keep('daily_minimum', max(0.,a.minimum-sum(v.L[i,t].X for t in range(24))))
        if model.terminal == 'minimum':
            assert model.constraints.terminal[i].Sense == '>'
        keep('terminal_lower_bound', max(0.,a.initial-v.e[i,23].X))
        for t in range(24):
            load, pv, ch, dis, e, q = [x[i,t].X for x in (v.L,v.P,v.ch,v.dis,v.e,v.net)]
            dp, dn = v.d_plus[i,t].X, v.d_minus[i,t].X
            keep('load_bounds',max(0.,a.lower[t]-load,load-a.upper[t]))
            keep('deviation_split',abs(load-a.ref[t]-dp+dn))
            keep('absolute_deviation',abs(dp+dn-abs(load-a.ref[t])))
            keep('asset_bounds',max(0.,-pv,pv-a.pv[t],-ch,ch-a.charge,
                                   -dis,dis-a.discharge,-e,e-a.capacity,-dp,-dn))
            keep('net_definition',abs(q-load+pv-ch+dis))
            previous=a.initial if t==0 else v.e[i,t-1].X
            keep('battery_dynamics',abs(e-previous-a.rho_ch*ch+dis/a.rho_dis))
            private += a.c*pv+a.cL*(dp+dn)
            if model.variant != 'C':
                keep('balance',abs(q-v.imp[i,t].X+v.exp[i,t].X))
                if model.variant=='I':
                    keep('cap',max(0.,v.imp[i,t].X-v.excess[i,t].X-model.shares[i]*model.cap[t]))
    grid = 0.
    discount = 1. if model.variant=='R' else 1-model.beta
    for t in range(24):
        if model.variant=='C':
            imp, exp, excess = v.imp[t].X, v.exp[t].X, v.excess[t].X
            keep('balance',abs(sum(v.net[a.id,t].X for a in d.members)-imp+exp))
            keep('cap',max(0.,imp-excess-model.cap[t]))
        else:
            imp=sum(v.imp[a.id,t].X for a in d.members)
            exp=sum(v.exp[a.id,t].X for a in d.members)
            excess=sum(v.excess[a.id,t].X for a in d.members) if model.variant=='I' else 0.
        grid += ((d.prices[t]+discount*d.tau_imp)*imp
                 -(d.prices[t]-discount*d.tau_exp)*exp+model.alpha*excess)
    errors['objective_reconstruction_DKK']=abs(m.ObjVal-private-grid)
    assert m.ModelSense==1, 'Report minimizes J'
    assert max(errors.values())<TOL,errors
    return errors


def terminal_condition_check(data):
    """Distinguish terminal >= from equality using a named negative-price case.

    One battery-owning member with initial SOC zero; all prices 0 except
    hour 23 at -5 DKK/kWh. Zero initial SOC isolates the terminal restriction.
    Raw input data stay unchanged. This diagnostic is not the base case.
    """
    d=deepcopy(data)
    d.members=[next(a for a in d.members if a.capacity>0)]
    d.members[0].initial=0.
    d.prices=[0.]*23+[-5.]
    lower_model=CommunityModel(d,'R',terminal='minimum')
    lower=lower_model.run()
    equal_model=CommunityModel(d,'R',terminal='cyclic')
    equal=equal_model.run()
    a=d.members[0]
    end=lower_model.variables.e[a.id,23].X
    assert end>a.initial+TOL
    assert lower.metrics['J_DKK']<equal.metrics['J_DKK']-TOL
    return dict(case='single_battery_member_zero_initial_SOC_negative_last_hour_price',
        initial_SOC_kWh=a.initial, minimum_terminal_SOC_kWh=end,
        cyclic_terminal_SOC_kWh=equal_model.variables.e[a.id,23].X,
        minimum_cost_DKK=lower.metrics['J_DKK'], cyclic_cost_DKK=equal.metrics['J_DKK'],
        passed=True, note='Demonstrates the two terminal constraints are different models.')
