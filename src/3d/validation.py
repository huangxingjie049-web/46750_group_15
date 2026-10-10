"""Independent reduced-model solution and physical/KKT checks for Q3.

The reference method eliminates PV/import/export analytically, minimizes each
piecewise quadratic hourly cost, then bisects the daily multiplier. It does
not use Gurobi or the full QP constraints to obtain its solution.
"""
import numpy as np


def _hour_solution(d, t, mu):
    a = d.energy_price[t]+d.import_tariff
    b = d.energy_price[t]-d.export_tariff
    c, cap, ref, q = d.pv_marginal_cost, d.pv_available[t], d.reference_load[t], d.quadratic_disutility
    lo, hi = d.load_min_kWh, d.load_max_kWh
    # Below PV availability the marginal procurement cost is clip(c,b,a);
    # above it the marginal cost is a. Endpoints include the kink and bounds.
    kink = float(np.clip(cap, lo, hi))
    candidates = [lo, hi, kink]
    if cap > lo:
        candidates.append(float(np.clip(ref+(mu-np.clip(c,b,a))/(2*q),lo,kink)))
    if cap < hi:
        candidates.append(float(np.clip(ref+(mu-a)/(2*q),kink,hi)))

    def flows(load):
        pv = cap if c <= b else (min(load,cap) if c <= a else 0.)
        imp, exp = max(load-pv,0.), max(pv-load,0.)
        cost = c*pv+a*imp-b*exp+q*(load-ref)**2
        return pv,imp,exp,cost

    load = min(candidates,key=lambda x:flows(x)[3]-mu*x)
    return load, *flows(load)


def independent_solution(d):
    """Solve reduced convex problem with a 1-D search, including inactive E_min."""
    target = d.min_daily_energy_kWh or 0.
    if target > d.n_hours*d.load_max_kWh:
        raise ValueError('Infeasible daily requirement.')
    def schedule(mu):
        return np.asarray([_hour_solution(d,t,mu) for t in range(d.n_hours)])
    values, mu = schedule(0.), 0.
    if values[:,0].sum() < target-1e-10:
        low, high = 0., 1.
        while schedule(high)[:,0].sum() < target-1e-10:
            high *= 2
        for _ in range(90):
            mid = (low+high)/2
            if schedule(mid)[:,0].sum() < target:
                low = mid
            else:
                high = mid
        mu = high
        values = schedule(mu)
    return {'load':values[:,0], 'objective':float(-values[:,4].sum()), 'mu':float(mu)}


def validate_solution(result, d, tolerance=2e-6):
    """Check feasibility, accounting, KKT and independent optimal objective.

    Report multipliers use the maximization Lagrangian in (31). Raw Gurobi
    balance Pi has the opposite sign to lambda; upper-bound Pi is beta/delta,
    whereas lower-bound Pi is minus alpha/gamma/eta/theta.
    """
    h = result.hourly
    L,P,I,E = (h[n].to_numpy() for n in ['load','pv','import','export'])
    lam = -h['dual_balance'].to_numpy()
    alpha,beta,gamma,delta,eta,theta = (
        -h['dual_load_lower'].to_numpy(), h['dual_load_upper'].to_numpy(),
        -h['dual_pv_lower'].to_numpy(), h['dual_pv_upper'].to_numpy(),
        -h['dual_import_lower'].to_numpy(), -h['dual_export_lower'].to_numpy())
    mu = -result.duals.get('minimum_energy',0.)
    emin = d.min_daily_energy_kWh or 0.
    slack = L.sum()-emin
    a,b = d.energy_price+d.import_tariff,d.energy_price-d.export_tariff
    q,c = d.quadratic_disutility,d.pv_marginal_cost
    cost = float(np.sum(c*P+a*I-b*E+q*(L-d.reference_load)**2))
    def maxabs(x):
        return float(np.max(np.abs(x)))
    feasibility = float(max(0.,np.max(d.load_min_kWh-L),np.max(L-d.load_max_kWh),
                            np.max(-P),np.max(P-d.pv_available),np.max(-I),np.max(-E),-slack))
    stationarity = np.concatenate([
        -2*q*(L-d.reference_load)+mu+alpha-beta-lam,
        -c+lam+gamma-delta, -a+lam+eta, b-lam+theta])
    complementarity = np.concatenate([
        alpha*(L-d.load_min_kWh), beta*(d.load_max_kWh-L), gamma*P,
        delta*(d.pv_available-P), eta*I, theta*E, [mu*slack]])
    w = mu+alpha-beta-lam
    # The daily constant -mu*E_min occurs ONCE, outside the hourly sum.
    dual_objective = float(np.sum(w*w/(4*q)+d.reference_load*w-alpha*d.load_min_kWh
                                  +beta*d.load_max_kWh+delta*d.pv_available)-mu*emin)
    reference = independent_solution(d)
    checks = {
        'balance_residual_kWh': maxabs(P+I-L-E),
        'bound_and_daily_violation_kWh': feasibility,
        'objective_accounting_error_DKK': abs(result.objective+cost),
        'stationarity_residual': maxabs(stationarity),
        'complementarity_residual': maxabs(complementarity),
        'dual_sign_violation': float(max(0.,-min(mu,alpha.min(),beta.min(),gamma.min(),
                                               delta.min(),eta.min(),theta.min()))),
        'primal_dual_gap_DKK': abs(dual_objective-result.objective),
        'independent_objective_error_DKK': abs(reference['objective']-result.objective),
        'independent_load_error_kWh': maxabs(reference['load']-L),
        'simultaneous_trade_kWh': float(np.minimum(I,E).max()),
    }
    return {'passed': all(x <= tolerance for x in checks.values()),
            'tolerance':tolerance,'checks':checks,
            'dual_objective_DKK':dual_objective,'independent_mu':reference['mu']}
