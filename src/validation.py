"""Independent procurement oracle and physical/accounting/KKT certificates.

All models are concave maximizations over affine constraints. KKT residuals
therefore certify global optimality; no unique dual vector is assumed at ties.
"""

import numpy as np
from .model import TOL


# Compute the cheapest PV/grid allocation for a fixed hourly load.
def procurement(d, t, load):
    """Independent minimum procurement in DKK for fixed hourly load in kWh."""
    a = d.energy_price[t] + d.import_tariff
    b = d.energy_price[t] - d.export_tariff
    c = d.pv_marginal_cost
    cap = d.pv_available[t]
    # Generate all available PV when c<=b; use up to local demand when b<c<=a;
    # generate none when c>a. At a tie this selects one optimal allocation.
    pv = cap if c <= b else min(load, cap) if c <= a else 0.0
    return c * pv + a * max(load - pv, 0) - b * max(pv - load, 0)


# Provide a solver-independent benchmark for separable model families.
def independent_schedule(d, kind):
    """Solve scalar segments and bisect daily mu, independently of Gurobi.

    Utility/absolute penalties: enumerate breakpoints; quadratic: include
    segment stationary points. Objective ties need no schedule agreement.
    """

    # For a trial daily multiplier, minimize each independent hourly cost.
    def schedule(mu):
        loads = []
        values = []
        for t in range(d.n_hours):
            lo, hi = d.load_min_kWh, d.load_max_kWh
            cap = d.pv_available[t]
            ref = 0 if d.reference_load is None else d.reference_load[t]
            # Affine-cost minima occur at bounds/reference/PV breakpoints; include all feasible ones.
            candidates = [
                lo,
                hi,
                float(np.clip(cap, lo, hi)),
                float(np.clip(ref, lo, hi)),
            ]
            # Add stationary points on both procurement segments, clipped to their segment bounds.
            if kind == "quadratic":
                a = d.energy_price[t] + d.import_tariff
                b = d.energy_price[t] - d.export_tariff
                kink = float(np.clip(cap, lo, hi))
                q = d.quadratic_disutility
                if cap > lo:
                    candidates.append(
                        float(
                            np.clip(
                                ref
                                + (mu - np.clip(d.pv_marginal_cost, b, a)) / (2 * q),
                                lo,
                                kink,
                            )
                        )
                    )
                if cap < hi:
                    candidates.append(
                        float(np.clip(ref + (mu - a) / (2 * q), kink, hi))
                    )

            # Evaluate true hourly procurement plus preference cost for a candidate load.
            def cost(x):
                preference = (
                    -d.consumption_utility * x
                    if kind == "utility"
                    else (
                        d.linear_disutility * abs(x - ref)
                        if kind == "linear"
                        else d.quadratic_disutility * (x - ref) ** 2
                    )
                )
                return procurement(d, t, x) + preference

            # Minimize cost-mu*load; ties choose the smallest load for a deterministic oracle.
            x = min(candidates, key=lambda x: (cost(x) - mu * x, x))
            loads.append(x)
            values.append(cost(x))
        return np.array(loads), -sum(values)

    # Start without a daily shadow price; a slack daily requirement needs no adjustment.
    x, value = schedule(0)
    mu = 0.0
    # A daily minimum adds one aggregate feasibility condition.
    if d.min_daily_energy_kWh is not None and x.sum() < d.min_daily_energy_kWh - 1e-9:
        if kind != "quadratic":
            raise ValueError("Daily oracle supports strictly convex quadratic loads.")
        # Expand the mu bracket until the daily minimum is reached, then bisect it.
        low, high = 0.0, 1.0
        while schedule(high)[0].sum() < d.min_daily_energy_kWh - 1e-9:
            high *= 2
        # The strictly convex hourly response gives a monotone aggregate load for bisection.
        for _ in range(85):
            mid = (low + high) / 2
            if schedule(mid)[0].sum() < d.min_daily_energy_kWh:
                low = mid
            else:
                high = mid
        mu = high
        x, value = schedule(mu)
    # Return the independently optimized load, true objective, and daily multiplier.
    return {"load": x, "objective": float(value), "mu": float(mu)}


# Recalculate feasibility, accounting, and optimality from the delivered schedule.
def validate(result, data, tolerance=TOL):
    """Recompute physical/accounting/KKT conditions for any model family.

    Outputs: residuals and pass status. Uncoupled/no-battery models additionally
    use independent scalar optimization. DKK, kWh and stationarity tolerances
    share the small numerical threshold, with units recorded per check.
    """
    r, d = result, data
    h = r.hourly
    s = r.specification
    kind = s["kind"]
    battery = "soc" in h

    # Measure the largest absolute residual, including empty arrays safely.
    def mx(x):
        return float(np.max(np.abs(x))) if np.size(x) else 0.0

    # Measure the largest lower/upper-bound violation; feasible points return zero.
    def violation(lo, x, hi=np.inf):
        return float(max(0, np.max(lo - x), np.max(x - hi)))

    # Read the exported decisions, so validation can detect changes after the original solve.
    L, P, I, E = (h[n].to_numpy() for n in ["load", "pv", "import", "export"])
    charge = h.charge.to_numpy() if battery else np.zeros(d.n_hours)
    discharge = h.discharge.to_numpy() if battery else np.zeros(d.n_hours)
    delta = L - h.reference_load.to_numpy()
    # Recalculate discomfort independently of stored model metrics.
    local = (
        np.zeros(d.n_hours)
        if kind == "utility"
        else (
            d.linear_disutility * abs(delta)
            if kind == "linear"
            else d.quadratic_disutility * delta**2
        )
    )
    coupling = s["coupling_DKK_per_kWh2"]
    # Each adjacent sum contributes one coupling penalty, consistent with the report formula.
    pairs = delta[1:] + delta[:-1]
    D = float(local.sum() + coupling * (pairs**2).sum())
    # Rebuild procurement with PV cost, effective import price, and export revenue.
    C = float(
        np.sum(
            d.pv_marginal_cost * P
            + (d.energy_price + d.import_tariff) * I
            - (d.energy_price - d.export_tariff) * E
        )
    )
    U = float(d.consumption_utility * L.sum()) if kind == "utility" else 0.0
    # Value mode rewards end SOC once; cyclic mode has no terminal reward.
    credit = (
        s["pi_DKK_per_kWh"] * float(h.soc.iloc[-1])
        if battery and s["terminal_mode"] == "value"
        else 0.0
    )
    # Check physical balances/bounds and reconcile saved costs with recomputed amounts.
    checks = {
        "balance_kWh": mx(P + I + discharge - L - E - charge),
        "load_bounds_kWh": violation(d.load_min_kWh, L, d.load_max_kWh),
        "pv_bounds_kWh": violation(0, P, d.pv_available),
        "trade_nonnegative_kWh": max(violation(0, I), violation(0, E)),
        "procurement_accounting_DKK": abs(C - r.metrics["procurement_cost"]),
        "disutility_accounting_DKK": abs(D - r.metrics["disutility"]),
        "objective_accounting_DKK": abs(U - C - D + credit - r.metrics["objective"]),
        "simultaneous_trade_kWh": (
            mx(np.minimum(I, E))
            if np.all(
                d.energy_price + d.import_tariff > d.energy_price - d.export_tariff
            )
            else 0.0
        ),
    }
    # A daily minimum adds one aggregate feasibility condition.
    if d.min_daily_energy_kWh is not None:
        checks["daily_minimum_kWh"] = max(0, d.min_daily_energy_kWh - L.sum())
    # Positive linear penalty implies z equals absolute deviation at an optimum.
    if kind == "linear":
        checks["absolute_auxiliary_kWh"] = mx(h.z.to_numpy() - abs(delta))
    # Use previous end SOC (or initial SOC at t=0) to verify every storage transition.
    if battery:
        soc = h.soc.to_numpy()
        prev = np.r_[d.battery_initial_soc_kWh, soc[:-1]]
        checks.update(
            soc_dynamics_kWh=mx(
                soc
                - prev
                - d.battery_charging_efficiency * charge
                + discharge / d.battery_discharging_efficiency
            ),
            soc_bounds_kWh=violation(0, soc, d.battery_capacity_kWh),
            charge_bounds_kW=violation(0, charge, d.battery_max_charge_kW),
            discharge_bounds_kW=violation(0, discharge, d.battery_max_discharge_kW),
        )
        if s["terminal_mode"] == "cyclic":
            checks["cyclic_terminal_kWh"] = abs(soc[-1] - d.battery_initial_soc_kWh)
        if (
            np.all(d.energy_price - d.export_tariff > 0)
            and d.battery_charging_efficiency * d.battery_discharging_efficiency < 1
        ):
            checks["simultaneous_battery_kW"] = mx(np.minimum(charge, discharge))
    # Gurobi Pi is the derivative wrt RHS; Lagrangian f - Pi*(Ax-b).
    A = r.certificate["A"]
    rows = r.certificate["rows"]
    # Validate the delivered hourly decisions, not an earlier solver snapshot.
    x = np.array(
        [
            float(h.loc[int(n.split("[")[1][:-1]), n.split("[")[0]])
            for n in r.certificate["variable_names"]
        ]
    )
    pi = np.array([a["pi"] for a in rows])
    rhs = np.array([a["rhs"] for a in rows])
    residual = A @ x - rhs
    # Check every exported linear row, including auxiliary rows not in physical summaries.
    row_violation = [
        (
            max(0, v)
            if row["sense"] == "<"
            else max(0, -v) if row["sense"] == ">" else abs(v)
        )
        for v, row in zip(residual, rows)
    ]
    checks["all_explicit_rows_feasibility"] = max(row_violation, default=0.0)
    # Differentiate the objective analytically instead of trusting Gurobi reduced costs.
    derivatives = {
        "pv": np.full(d.n_hours, -d.pv_marginal_cost),
        "import": -(d.energy_price + d.import_tariff),
        "export": d.energy_price - d.export_tariff,
    }
    if kind == "utility":
        derivatives["load"] = np.full(d.n_hours, d.consumption_utility)
    elif kind == "linear":
        derivatives["load"] = np.zeros(d.n_hours)
        derivatives["z"] = np.full(d.n_hours, -d.linear_disutility)
    else:
        gradient = -2 * d.quadratic_disutility * delta
        # Each load belongs to up to two adjacent pairs, so both neighbors affect its gradient.
        if kind == "coupled":
            gradient[:-1] -= 2 * coupling * pairs
            gradient[1:] -= 2 * coupling * pairs
        derivatives["load"] = gradient
    # Use previous end SOC (or initial SOC at t=0) to verify every storage transition.
    if battery:
        for name in ("charge", "discharge", "soc"):
            derivatives[name] = np.zeros(d.n_hours)
        if s["terminal_mode"] == "value":
            derivatives["soc"][-1] = s["pi_DKK_per_kWh"]
    # Order analytical derivatives exactly as the solver columns in certificate A.
    gradient = np.array(
        [
            derivatives[n.split("[")[0]][int(n.split("[")[1][:-1])]
            for n in r.certificate["variable_names"]
        ]
    )
    # For max f with these raw Pi values, stationarity is grad(f)-A.T@Pi=0.
    checks["KKT_stationarity"] = mx(gradient - A.T @ pi)
    checks["KKT_complementarity"] = mx(pi * residual)
    # Maximization requires Pi>=0 for <= rows and Pi<=0 for >= rows; equalities are free.
    signs = [
        max(0, -p) if row["sense"] == "<" else max(0, p) if row["sense"] == ">" else 0
        for p, row in zip(pi, rows)
    ]
    checks["KKT_dual_sign"] = max(signs, default=0.0)
    checks["dual_gap_DKK"] = abs(float(pi @ residual))
    # Only separable no-battery models use the scalar oracle; all families still receive KKT checks.
    if not battery and kind != "coupled":
        oracle = independent_schedule(d, kind)
        checks["independent_objective_DKK"] = abs(
            oracle["objective"] - r.metrics["objective"]
        )
        if kind == "quadratic":
            checks["independent_load_kWh"] = mx(oracle["load"] - L)
    # A run passes only when every computed residual is within the stated tolerance.
    return {
        "passed": bool(all(v <= tolerance for v in checks.values())),
        "tolerance": tolerance,
        "checks": checks,
        "scope": "physical, full preference accounting, independently differentiated KKT; scalar oracle for separable models",
    }
