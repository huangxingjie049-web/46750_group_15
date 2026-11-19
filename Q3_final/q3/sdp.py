"""Question 3: scenario-based dynamic pricing SDP^x as ONE parametrized Gurobi model.

The manager posts one price vector x_t (24 values, DKK/kWh) before the members' preferences are known. For each
in-sample joint scenario s (probability pi_s) every member best-responds to x with that scenario's reference profile,
daily minimum and hourly bounds. Each (member, scenario) lower level is replaced by its KKT system, built by the
validated Q2 function ``Q2_final/Q2_2/kkt_member_builder.add_member_kkt`` (SOS1 complementarity, strong-duality
payment linearization). The prices are shared; everything else is scenario-indexed:

    community, scenario s:  P_imp_{s,t} - P_exp_{s,t} = sum_i q_{i,s,t},   P_imp_{s,t} <= cap_t + z_{s,t}
    cost_s = sum_i (PV cost + disutility)_{i,s}
             + sum_t [ p_t (P_imp - P_exp) + (1 - beta)(tau_imp (P_imp - z) + tau_exp P_exp) + alpha z ]_{s,t}
    revenue_s = sum_i payment_{i,s}  (= dual objective - physical cost, linear)

Switches (``SDPConfig``):
    risk = "neutral"   min  sum_s pi_s cost_s                                                        SDP^E
    risk = "cvar"      min  (1 - w) sum_s pi_s cost_s + w [ eta + 1/(1 - a) sum_s pi_s u_s ],
                       u_s >= cost_s - eta, u_s >= 0                    (Rockafellar-Uryasev)        SDP^CVaR
    cap = "penalty"    the contract: exceedance allowed in every scenario at alpha DKK/kWh
    cap = "chance"     exceedance only in scenarios with y_s = 1, sum_s pi_s y_s <= epsilon:
                       z_{s,t} <= M y_s                                                              SDP^CC
    bb  = "expectation"  sum_s pi_s (revenue_s - external_s) = 0                         (base, as the assignment)
    bb  = "scenario"     revenue_s = external_s for every s                                         (variation)
    bb  = "none"         no budget constraint ("cap only": validation against C and the cost of BB)
    Budget balance is ELASTIC by default: residual r+ - r- added to the equality and bb_penalty (r+ + r-) to the
    objective. With prices fixed at a start vector the hard equality is usually infeasible (no complete MIP start),
    while the elastic form always has one; the reported bb_residual shows whether BB holds at the solution.
Individual rationality is not a constraint (as in Q2): it is checked ex post, in expectation, and settled with
zero-sum fixed transfers (``q3/settlement.py``).

Size: with N scenarios and the 16 base members, the model has N copies of the 16 KKT systems
(``model_size`` reports variables, constraints, SOS1 pairs and binaries for the solved instance).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .evaluate import contract_terms
from .follower_lp import community_response
from .paths import Q2_DIR, load_module
from .q1_bridge import scenario_data

BIG_M_EXCESS = 60.0     # kWh: hourly import <= 16 x (3 kWh + 0.75 kW) = 60


@dataclass
class SDPConfig:
    """One named configuration of the parametrized model (superscript in the report: SDP^E, SDP^CVaR, SDP^CC)."""
    name: str = "SDP_E"
    risk: str = "neutral"            # "neutral" | "cvar"
    cvar_alpha: float = 0.9          # CVaR level a (upper tail of cost)
    cvar_weight: float = 0.5         # w in [0, 1]: weight of CVaR in the objective
    cap: str = "penalty"             # "penalty" | "chance"
    epsilon: float = 0.2             # chance constraint: max probability of any exceedance
    bb: str = "expectation"          # "expectation" | "scenario" | "none"
    price_lb: float = -15.0          # DKK/kWh, as in Q2
    price_ub: float = 15.0
    time_limit: float = 600.0        # s
    mip_gap: float = 1e-3
    threads: int = 0
    seed: int = 0
    warm_start: str = "auto"         # "auto" (C prices if bb = none, else DP prices), "C", "DP", "none" or a prices.json
    start_time_limit: float = 120.0  # s, phase 1 of the warm start (model solved with the prices fixed)
    bb_penalty: float = 100.0        # DKK per DKK of budget-balance residual (elastic BB); 0 = hard equality
    mip_focus: int = 1               # Gurobi MIPFocus (1: find feasible solutions first)
    contract: dict = field(default_factory=dict)   # overrides for evaluate.contract_terms: alpha, beta, cap_factor

    def check(self):
        assert self.risk in ("neutral", "cvar") and self.cap in ("penalty", "chance")
        assert self.bb in ("expectation", "scenario", "none")
        assert 0 < self.cvar_alpha < 1 and 0 <= self.cvar_weight <= 1 and 0 <= self.epsilon <= 1


_KKT = None
_CONVERT = None


def _q2():
    """The Q2 KKT builder and member converter, loaded by path (never modified)."""
    global _KKT, _CONVERT
    if _KKT is None:
        _KKT = load_module("q2_kkt_member_builder", Q2_DIR / "Q2_2" / "kkt_member_builder.py").add_member_kkt
        _CONVERT = load_module("q2_data_interface", Q2_DIR / "Q2_1" / "data_interface.py").convert_member
    return _KKT, _CONVERT


def pool_files(set_name: str) -> list[Path]:
    """Every price vector already found for ``set_name`` (pooled multi-start): all solved runs of that scenario set
    and, for in5, the archived N = 5 vectors in data/n5_vectors. A price vector is a valid start for any variant."""
    from .paths import Q3_DATA_DIR, RESULTS_DIR
    files = sorted((RESULTS_DIR / "sdp").glob(f"*_{set_name}_bb-*/prices.json"))
    if set_name == "in5":
        files += sorted((Q3_DATA_DIR / "n5_vectors").glob("*.json"))
    return files


def _start_prices(cfg: SDPConfig):
    """Candidate start price vectors (list of (label, prices)); None for no warm start.

    ``warm_start`` is "none", "auto", or a comma-separated list of C, DP and/or prices.json paths (multi-start: phase 1
    is solved for every candidate and the best complete solution is used). auto: cap only -> C dual prices (DP = C);
    with budget balance -> the solved deterministic baseline with the same budget rule if it exists
    (results/sdp/SDP_E_expected_bb-<bb>/prices.json), otherwise the Q2 DP prices."""
    from . import prices as pr
    from .paths import RESULTS_DIR
    key = cfg.warm_start
    if key == "none":
        return None
    if key == "auto":
        baseline = RESULTS_DIR / "sdp" / f"SDP_E_expected_bb-{cfg.bb}" / "prices.json"
        key = "C" if cfg.bb == "none" else (str(baseline) if baseline.exists() else "DP")
    cands, seen = [], set()

    def add(label, x):
        sig = tuple(np.round(np.asarray(x, dtype=float), 6))
        if sig not in seen:                      # identical price vectors are evaluated once
            seen.add(sig)
            cands.append((label, np.asarray(x, dtype=float)))

    for k in [k.strip() for k in key.split(",") if k.strip()]:
        if k.lower().startswith("pool:"):
            for f in pool_files(k.split(":", 1)[1]):
                add(str(f), pr.load_prices(str(f))[1])
            continue
        if k.lower().endswith(".json") and not Path(k).exists():
            print(f"warm start candidate skipped (file not found): {k}", flush=True)
            continue
        add(k, pr.load_prices(k)[1])
    print(f"warm start: {len(cands)} distinct candidate price vectors", flush=True)
    if not cands and cfg.warm_start != "auto":
        return _start_prices(replace(cfg, warm_start="auto"))
    return cands or None


def build(base, scenarios: list[dict], cfg: SDPConfig):
    """Build (not solve) the SDP model. Returns (model, handles)."""
    import gurobipy as gp
    from gurobipy import GRB, quicksum

    cfg.check()
    add_member_kkt, convert_member = _q2()
    T = range(24)
    m = gp.Model(cfg.name)
    m.Params.TimeLimit = cfg.time_limit
    m.Params.MIPGap = cfg.mip_gap
    m.Params.Threads = cfg.threads
    m.Params.Seed = cfg.seed
    m.Params.MIPFocus = cfg.mip_focus
    x = m.addVars(T, lb=cfg.price_lb, ub=cfg.price_ub, name="price")

    datasets = [scenario_data(base, s) for s in scenarios]
    pis = [float(s["probability"]) for s in scenarios]
    assert abs(sum(pis) - 1) < 1e-9
    sc = []
    for k, d in enumerate(datasets):
        terms = contract_terms(d, **cfg.contract)
        blocks = {mem.id: add_member_kkt(m, convert_member(mem), x, f"{mem.id}_s{k}") for mem in d.members}
        imp = m.addVars(T, lb=0.0, name=f"P_imp_s{k}")
        exp = m.addVars(T, lb=0.0, name=f"P_exp_s{k}")
        exc = m.addVars(T, lb=0.0, name=f"excess_s{k}")
        for t in T:
            m.addConstr(imp[t] - exp[t] == quicksum(b["net"][t] for b in blocks.values()), name=f"balance_s{k}_{t}")
            m.addConstr(imp[t] <= float(terms["cap"][t]) + exc[t], name=f"cap_s{k}_{t}")
        physical = quicksum(b["physical_cost"] for b in blocks.values())
        external = quicksum(float(terms["p"][t]) * (imp[t] - exp[t])
                            + (1 - terms["beta"]) * (terms["tau_imp"] * (imp[t] - exc[t]) + terms["tau_exp"] * exp[t])
                            + terms["alpha"] * exc[t] for t in T)        # tariff only within the cap (Q1)
        revenue = quicksum(b["payment_linear"] for b in blocks.values())
        sc.append(dict(data=d, terms=terms, blocks=blocks, imp=imp, exp=exp, exc=exc,
                       physical=physical, external=external, revenue=revenue, cost=physical + external))

    expected_cost = quicksum(pi * s["cost"] for pi, s in zip(pis, sc))
    handles = dict(x=x, scenarios=sc, pis=pis, expected_cost=expected_cost)

    if cfg.cap == "chance":
        y = m.addVars(len(sc), vtype=GRB.BINARY, name="exempt")
        for k, s in enumerate(sc):
            for t in T:
                m.addConstr(s["exc"][t] <= BIG_M_EXCESS * y[k], name=f"chance_s{k}_{t}")
        m.addConstr(quicksum(pi * y[k] for k, pi in enumerate(pis)) <= cfg.epsilon + 1e-9, name="chance_budget")
        handles["exempt"] = y

    bb_slack = []
    elastic = cfg.bb_penalty > 0

    def bb_row(expr, label):
        if elastic:
            rp = m.addVar(lb=0.0, name=f"bb_res_pos_{label}")
            rn = m.addVar(lb=0.0, name=f"bb_res_neg_{label}")
            bb_slack.append((rp, rn))
            m.addConstr(expr + rn - rp == 0, name=f"budget_balance_{label}")
        else:
            m.addConstr(expr == 0, name=f"budget_balance_{label}")

    if cfg.bb == "expectation":
        bb_row(quicksum(pi * (s["revenue"] - s["external"]) for pi, s in zip(pis, sc)), "expected")
    elif cfg.bb == "scenario":
        for k, s in enumerate(sc):
            bb_row(s["revenue"] - s["external"], f"s{k}")
    handles["bb_slack"] = bb_slack
    bb_pen = cfg.bb_penalty * quicksum(rp + rn for rp, rn in bb_slack) if bb_slack else 0.0

    if cfg.risk == "cvar":
        eta = m.addVar(lb=-GRB.INFINITY, name="VaR")
        u = m.addVars(len(sc), lb=0.0, name="cvar_excess")
        for k, s in enumerate(sc):
            m.addConstr(u[k] >= s["cost"] - eta, name=f"cvar_s{k}")
        cvar = eta + (1.0 / (1 - cfg.cvar_alpha)) * quicksum(pi * u[k] for k, pi in enumerate(pis))
        handles.update(eta=eta, u=u, cvar=cvar)
        m.setObjective((1 - cfg.cvar_weight) * expected_cost + cfg.cvar_weight * cvar + bb_pen, GRB.MINIMIZE)
    else:
        m.setObjective(expected_cost + bb_pen, GRB.MINIMIZE)
    m.update()
    return m, handles


def set_start(m, handles, prices) -> dict:
    """MIP start: the prices plus, in every scenario, the members' optimistic best responses to them (computed
    with HiGHS by q3.follower_lp) and the community flows. Gurobi completes the dual variables itself."""
    if prices is None:
        return {"warm_start": "none"}
    prices = np.asarray(prices, dtype=float)
    for t in range(24):
        handles["x"][t].Start = float(prices[t])
    for s in handles["scenarios"]:
        resp = community_response(s["data"], prices, s["terms"], rule="optimistic")
        for i, blk in s["blocks"].items():
            for key, vals in resp["members"][i]["primal"].items():
                if blk.get(key) is not None:
                    for t in range(24):
                        blk[key][t].Start = float(vals[t])
        for t in range(24):
            s["imp"][t].Start = float(resp["imports"][t])
            s["exp"][t].Start = float(resp["exports"][t])
            s["exc"][t].Start = float(resp["excess"][t])
    m.update()
    return {"warm_start_prices": prices.tolist()}


def two_phase_start(m, handles, candidates, cfg: SDPConfig) -> dict:
    """Warm start in two phases (a partial MIP start of prices + primal responses is often NOT completed by Gurobi:
    "Another try with MIP start").

    Phase 1: for every candidate price vector, fix the prices and solve the model (the members' KKT systems then only
    select the optimistic best response; fast), with tight feasibility tolerances so that the solution is accepted
    as a MIP start. Phase 2: free the prices and pass the COMPLETE best phase-1 solution as MIP start. If no phase 1
    finds a solution, fall back to the partial start of the first candidate.
    """
    if not candidates:
        return {"warm_start": "none"}
    if isinstance(candidates, np.ndarray):
        candidates = [("given", candidates)]
    x = handles["x"]
    bounds = [(x[t].LB, x[t].UB) for t in range(24)]
    tol = (m.Params.FeasibilityTol, m.Params.IntFeasTol, m.Params.NumericFocus)
    m.Params.FeasibilityTol, m.Params.IntFeasTol, m.Params.NumericFocus = 1e-9, 1e-9, 2
    m.Params.TimeLimit = min(cfg.start_time_limit, cfg.time_limit)
    best, tried = None, []
    for label, prices in candidates:
        for t in range(24):
            x[t].LB = x[t].UB = float(prices[t])
        m.reset()
        m.optimize()
        rec = {"start": label, "status": int(m.Status), "runtime_s": float(m.Runtime)}
        if m.SolCount > 0:
            rec["objective"] = float(m.ObjVal)
            if best is None or m.ObjVal < best[0]:
                best = (float(m.ObjVal), label, prices, [(v, v.X) for v in m.getVars()])
        tried.append(rec)
    for t in range(24):
        x[t].LB, x[t].UB = bounds[t]
    m.Params.FeasibilityTol, m.Params.IntFeasTol, m.Params.NumericFocus = tol
    m.Params.TimeLimit = cfg.time_limit
    m.reset()
    info = {"phase1": tried, "phase1_runtime_s": float(sum(r["runtime_s"] for r in tried))}
    if best is not None:
        for v, val in best[3]:
            v.Start = val
        info.update(warm_start=f"complete solution of phase 1 ({best[1]})", phase1_objective=best[0],
                    warm_start_prices=[float(v) for v in best[2]])
    else:
        set_start(m, handles, candidates[0][1])
        info["warm_start"] = f"partial ({candidates[0][0]}; phase 1 found no solution)"
    m.update()
    return info


def kkt_residuals(m, handles) -> dict:
    """Validation of a solution (mandatory checks of the assignment), DKK and kWh:

    * max_complementarity: largest min(|a|, |b|) over all SOS1 pairs (0 if every complementarity holds);
    * max_strong_duality_gap: largest |primal private cost - dual objective| over all (member, scenario);
    * max_member_suboptimality: largest (model private cost - optimal value of the member LP at the same prices,
      solved independently with HiGHS) over all (member, scenario); >= 0 up to tolerances, 0 if every lower level
      is solved to optimality.
    """
    from .follower_lp import best_response

    comp = 0.0
    for sos in m.getSOSs():
        _, vs, _ = m.getSOS(sos)
        comp = max(comp, min(abs(vs[0].X), abs(vs[1].X)))
    x = np.array([handles["x"][t].X for t in range(24)])
    gap, subopt = 0.0, 0.0
    for s in handles["scenarios"]:
        members = {mem.id: mem for mem in s["data"].members}
        for i, blk in s["blocks"].items():
            net = np.array([blk["net"][t].getValue() for t in range(24)])
            primal = blk["physical_cost"].getValue() + float(np.dot(x, net))
            gap = max(gap, abs(primal - blk["dual_objective"].getValue()))
            subopt = max(subopt, primal - best_response(members[i], x)["objective"])
    return {"max_complementarity": comp, "max_strong_duality_gap": gap, "max_member_suboptimality": subopt}


def model_size(m) -> dict:
    return {"variables": m.NumVars, "constraints": m.NumConstrs, "sos1_pairs": m.NumSOS,
            "binaries": m.NumBinVars}


def solve(base, scenarios: list[dict], cfg: SDPConfig, out_dir: Path | None = None) -> dict:
    """Build, warm-start, solve and report one configuration on one in-sample set.

    Writes (if out_dir): config.json, prices.json, in_sample.csv (per scenario), solve.json (status, objective,
    bound, gap, runtime, size)."""
    import pandas as pd
    from . import prices as pr

    t0 = time.time()
    m, h = build(base, scenarios, cfg)
    build_time = time.time() - t0
    start_info = two_phase_start(m, h, _start_prices(cfg), cfg)
    m.optimize()
    report = {"config": asdict(cfg), "N": len(scenarios), "status": int(m.Status), "solutions": int(m.SolCount),
              "runtime_s": float(m.Runtime), "build_s": build_time, **model_size(m), **start_info}
    report["total_runtime_s"] = report["runtime_s"] + start_info.get("phase1_runtime_s", 0.0)
    if m.SolCount == 0:
        report["message"] = "no incumbent within the time limit (this does not prove infeasibility)"
    else:
        x = np.array([h["x"][t].X for t in range(24)])
        report.update(objective=float(m.ObjVal), bound=float(m.ObjBound), gap=float(m.MIPGap),
                      expected_cost=float(h["expected_cost"].getValue()), prices=x.tolist(),
                      bb_residual=float(sum(rp.X + rn.X for rp, rn in h["bb_slack"])))
        report.update(kkt_residuals(m, h))
        if "cvar" in h:
            report.update(cvar=float(h["cvar"].getValue()), var=float(h["eta"].X))
        rows = []
        for k, s in enumerate(h["scenarios"]):
            exc = np.array([s["exc"][t].X for t in range(24)])
            imp = np.array([s["imp"][t].X for t in range(24)])
            rows.append({"scenario": k, "probability": h["pis"][k], "cost": s["cost"].getValue(),
                         "physical": s["physical"].getValue(), "external": s["external"].getValue(),
                         "revenue": s["revenue"].getValue(),
                         "budget_surplus": s["revenue"].getValue() - s["external"].getValue(),
                         "penalty": s["terms"]["alpha"] * exc.sum(), "exceedance_kWh": exc.sum(),
                         "hours_above_cap": int((exc > 1e-6).sum()), "peak_import_kW": imp.max(),
                         "exempt": int(round(h["exempt"][k].X)) if "exempt" in h else 0})
        report["in_sample"] = rows
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pr.save_prices(x, out_dir / "prices.json", {"model": cfg.name, "N": len(scenarios)})
            pd.DataFrame(rows).to_csv(out_dir / "in_sample.csv", index=False)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "solve.json").write_text(json.dumps({k: v for k, v in report.items() if k != "in_sample"},
                                                       indent=1), encoding="utf-8")
    return report
