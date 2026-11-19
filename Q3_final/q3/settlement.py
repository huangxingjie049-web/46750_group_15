"""Individual rationality in expectation and zero-sum fixed transfers (the Q2 extension, applied under uncertainty).

For a price vector x and a scenario set (in-sample or out-of-sample), with the optimistic selection rule:

    V_i(s)  member i's private cost under x in scenario s (PV cost + disutility + payments), DKK
    R_i(s)  member i's cost alone in the reference model R in the same scenario (no contract, full tariffs), DKK
    margin_i = E_s[R_i(s)] - E_s[V_i(s)]          (>= 0: individually rational in expectation)

Members with a negative margin are compensated by fixed transfers paid by the members with a positive margin, in
proportion to their margins (the rule of Q2_final/Q2_2/run_manager_parameterized.py, mode TRANSFER). Transfers sum to
zero and do not change the prices or the dispatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import evaluate
from .follower_lp import reference_cost


def ir_in_expectation(prices, datasets: list, rule: str = "optimistic", contract_kw: dict | None = None) -> pd.DataFrame:
    _, _, members = evaluate(prices, datasets, rule=rule, contract_kw=contract_kw)
    ref = [{"scenario": d.scenario_id, "member": m.id, "R": reference_cost(m, d.prices, d.tau_imp, d.tau_exp)}
           for d in datasets for m in d.members]
    df = members.merge(pd.DataFrame(ref), on=["scenario", "member"])
    out = df.groupby("member", sort=False).agg(E_V=("private_cost", "mean"), E_R=("R", "mean"))
    worst = (df.private_cost - df.R).groupby(df.member, sort=False).max()
    share_worse = ((df.private_cost - df.R) > 1e-6).groupby(df.member, sort=False).mean()
    out["worst_V_minus_R"] = worst
    out["share_of_scenarios_worse_than_R"] = share_worse
    out["margin"] = out.E_R - out.E_V
    deficit = float(np.clip(-out.margin, 0, None).sum())
    gains = float(np.clip(out.margin, 0, None).sum())
    if gains + 1e-9 >= deficit:
        out["transfer"] = np.where(out.margin < 0, out.margin, deficit * np.clip(out.margin, 0, None) / max(gains, 1e-12))
    else:
        out["transfer"] = np.nan
    out["margin_after_transfer"] = out.margin - out.transfer
    out.attrs.update(deficit=deficit, gains=gains, feasible=bool(gains + 1e-9 >= deficit))
    return out.reset_index()
