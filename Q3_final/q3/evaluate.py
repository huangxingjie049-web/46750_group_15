"""Evaluation of a posted price vector on a set of preference scenarios (in-sample or out-of-sample).

For every scenario the members respond to the FIXED prices (``follower_lp.community_response``) and the manager
settles at the connection point exactly as in Q2 (``Q2_final/Q2_2/manager_accounting.py``):

    import_t = [sum_i q_i,t]+, export_t = [-sum_i q_i,t]+, excess_t = [import_t - cap_t]+          (kWh)
    external = sum_t [ p_t (import_t - export_t) + (1 - beta)(tau_imp (import_t - excess_t) + tau_exp export_t)
                       + alpha excess_t ]
    (Q1 convention since 13 Nov: the discounted tariff applies to imports within the cap; imports above the cap pay
    the day-ahead price plus the penalty. In the optimisation this is a_d import + (alpha - (1 - beta) tau_imp) excess.)
    community cost J = sum_i (PV cost + disutility) + external                                     (DKK)
    internal revenue = sum_t x_t sum_i q_i,t;   budget surplus = revenue - external                 (DKK)

Metrics per scenario (point of view in brackets):
    J, physical, external, penalty                     [community]
    J_C, regret = J - J_C                              [community; reference: central dispatch C with perfect
                                                        information on the same scenario, the lower bound of J]
    exceedance_kWh = sum_t excess_t, hours_above_cap, max_exceedance_kW, peak_import_kW     [DSO; reference: cap]
    budget_surplus                                      [manager; BB holds if 0]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .follower_lp import central_dispatch, community_response

HOUR_TOL = 1e-6   # kWh; an hour counts as "above the cap" if its excess exceeds this


def contract_terms(data, alpha: float | None = None, beta: float | None = None, cap_factor: float = 1.0) -> dict:
    """Contract of the Q1 data with optional overrides (alpha DKK/kWh, beta, cap multiplied by cap_factor)."""
    c = data.contract
    beta = float(c["tariff_discount_ratio"]) if beta is None else float(beta)
    alpha = float(c["penalty_DKK_per_kWh"]) if alpha is None else float(alpha)
    cap = cap_factor * np.array(c["capacity_limit_kW"], dtype=float)
    p = np.array(data.prices, dtype=float)
    exc_cost = alpha - (1 - beta) * float(data.tau_imp)
    assert exc_cost > 0, "penalty must exceed the discounted import tariff it replaces"
    return {"alpha": alpha, "beta": beta, "cap": cap, "p": p, "tau_imp": float(data.tau_imp),
            "tau_exp": float(data.tau_exp), "exc_cost": exc_cost,
            "a_d": p + (1 - beta) * float(data.tau_imp), "b_d": p - (1 - beta) * float(data.tau_exp)}


def settle(response: dict, prices, k: dict) -> dict:
    """Manager settlement and DSO metrics of one community response (see module docstring)."""
    imp, exp, exc = (np.asarray(response[key]) for key in ("imports", "exports", "excess"))
    energy = float(np.dot(k["p"], imp - exp))
    tariff = float((1 - k["beta"]) * (k["tau_imp"] * (imp - exc).sum() + k["tau_exp"] * exp.sum()))
    penalty = float(k["alpha"] * exc.sum())
    external = energy + tariff + penalty
    physical = float(sum(r["physical"] for r in response["members"].values()))
    revenue = float(np.dot(prices, response["net"]))
    return {
        "J": physical + external, "physical": physical, "external": external, "energy": energy, "tariff": tariff,
        "penalty": penalty, "revenue": revenue, "budget_surplus": revenue - external,
        "import_kWh": float(imp.sum()), "export_kWh": float(exp.sum()), "peak_import_kW": float(imp.max()),
        "exceedance_kWh": float(exc.sum()), "hours_above_cap": int((exc > HOUR_TOL).sum()),
        "max_exceedance_kW": float(exc.max()),
    }


def evaluate(prices, datasets: list, rule: str = "independent", contract_kw: dict | None = None):
    """Evaluate ``prices`` on every scenario in ``datasets`` (Q1 namespaces from ``q1_bridge.scenario_data``).

    Returns (per-scenario metrics DataFrame, hourly import array [n_scenarios x 24], per-member cost DataFrame).
    ``contract_kw`` may hold ``tie_alpha``: under the optimistic rule the members' ties are then broken with this
    exceedance weight (cap first) instead of the DSO penalty, while every cost is still settled at the true penalty.
    This is the solution concept of SDP^CC, whose leader keeps the cap whenever a tie allows it.
    """
    prices = np.asarray(prices, dtype=float)
    kw = dict(contract_kw or {})
    tie_alpha = kw.pop("tie_alpha", None)   # optimistic tie-break that puts the cap first (chance-constrained models)
    rows, hourly, members = [], [], []
    for d in datasets:
        k = contract_terms(d, **kw)
        k_sel = dict(k, exc_cost=float(tie_alpha)) if (tie_alpha is not None and rule == "optimistic") else k
        resp = community_response(d, prices, k_sel, rule=rule)
        row = {"scenario": d.scenario_id, "rule": rule, **settle(resp, prices, k),
               "value_tol": resp.get("value_tol", 0.0)}
        row["J_C"] = settle(central_dispatch(d, k), np.zeros(24), k)["J"]
        row["regret"] = row["J"] - row["J_C"]
        rows.append(row)
        hourly.append(np.asarray(resp["imports"]))
        for i, r in resp["members"].items():
            members.append({"scenario": d.scenario_id, "member": i, "private_cost": r["objective"],
                            "physical": r["physical"], "payment": r["payment"]})
    return pd.DataFrame(rows), np.vstack(hourly), pd.DataFrame(members)


def summarize(df: pd.DataFrame, cols=("J", "regret", "penalty", "exceedance_kWh", "hours_above_cap", "peak_import_kW",
                                      "budget_surplus")) -> pd.DataFrame:
    """Distribution summary (mean, quantiles, CVaR_0.9 of the upper tail) and frequency of any exceedance."""
    out = {}
    for c in cols:
        v = np.sort(df[c].to_numpy(dtype=float))
        tail = v[int(np.floor(0.9 * len(v))):]
        out[c] = {"mean": v.mean(), "min": v.min(), "q05": np.quantile(v, 0.05), "median": np.median(v),
                  "q95": np.quantile(v, 0.95), "max": v.max(), "cvar90_upper": tail.mean()}
    s = pd.DataFrame(out).T
    s.attrs["share_of_scenarios_with_exceedance"] = float((df["exceedance_kWh"] > HOUR_TOL).mean())
    return s
