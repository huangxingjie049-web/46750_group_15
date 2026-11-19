"""Members' best responses at FIXED internal prices, solved with HiGHS (scipy.optimize.milp).

Used for the out-of-sample evaluation of Question 3: once the manager has posted a price vector x (24 values,
DKK/kWh), every member of a scenario solves its own linear program (the follower of Q2, ``Q2_final/Q2_1/follower.py``,
same constraints and the same terminal condition e_23 >= E_0):

    min  sum_t [ c P_t + cL (d+_t + d-_t) + x_t q_t ]
    s.t. lower_t <= l_t <= upper_t,  l_t - ref_t = d+_t - d-_t,  sum_t l_t >= E_min,  0 <= P_t <= Pbar_t,
         battery: e_t = e_{t-1} + rho_ch p_ch_t - p_dis_t / rho_dis,  0 <= e_t <= Ebar,  power bounds,  e_23 >= E_0,
         q_t = l_t - P_t + p_ch_t - p_dis_t.

With a linear disutility the best response is generally a SET (a member at a price exactly cL away from its internal
value is indifferent). The community's import, and hence the cap exceedance, depends on which element is selected.
Three selection rules are implemented:

    "independent"  each member's LP is solved on its own; HiGHS returns one vertex (an arbitrary tie-break),
                   as the independently solved members of Q2 (Section 2.3 of the report);
    "optimistic"   among all profiles in which every member is optimal, the one with the LOWEST community cost
                   (the manager's selection assumed by the optimistic bilevel model) -- a linear program;
    "pessimistic"  ... the one with the HIGHEST community cost (worst case for the manager) -- a MILP with 48
                   binaries (import/export sign and exceedance indicator of each hour).

A different solver (HiGHS) from the one that computes the prices (Gurobi) is used on purpose: the evaluation does not
reuse any solver state of the pricing model.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

T = 24
BIG_M = 100.0          # kWh; |community net import| <= 16 x (3 + 0.75) = 60 and exports <= 24 + 6 = 30
VALUE_TOLS = (1e-6, 1e-5, 1e-4)   # DKK; a member's cost may exceed its own optimal value by at most this in the
                                  # joint selection. The joint MILP of the pessimistic rule can be declared infeasible
                                  # at 1e-6 because of solver tolerances (prices up to 15 DKK/kWh multiply the flows);
                                  # the tolerance is then relaxed step by step and the value used is reported.


@contextlib.contextmanager
def _quiet_stdout():
    """Silence HiGHS's C-level messages (e.g. "HighsMipSolverData::transformNewIntegerFeasibleSolution") that bypass
    Python's sys.stdout; a no-op when stdout has no file descriptor (e.g. under pytest capture)."""
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


class LP:
    """Minimal sparse model builder for scipy.optimize.milp (minimisation)."""

    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.rows, self.cols, self.vals, self.rlo, self.rhi = [], [], [], [], []

    def var(self, n, lb=0.0, ub=np.inf, cost=0.0, integer=False):
        start = len(self.lb)
        self.lb += list(np.broadcast_to(lb, n).astype(float))
        self.ub += list(np.broadcast_to(ub, n).astype(float))
        self.c += list(np.broadcast_to(cost, n).astype(float))
        self.integ += [1 if integer else 0] * n
        return np.arange(start, start + n)

    def row(self, coeffs: dict, lo=-np.inf, hi=np.inf):
        r = len(self.rlo)
        for j, v in coeffs.items():
            if v != 0.0:
                self.rows.append(r); self.cols.append(j); self.vals.append(v)
        self.rlo.append(lo); self.rhi.append(hi)

    def add_cost(self, coeffs: dict, scale=1.0):
        for j, v in coeffs.items():
            self.c[j] += scale * v

    def solve(self, sense=1.0):
        n = len(self.lb)
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.rlo), n)).tocsr()
        with _quiet_stdout():
            res = milp(c=sense * np.array(self.c), integrality=np.array(self.integ),
                       bounds=Bounds(np.array(self.lb), np.array(self.ub)),
                       constraints=[LinearConstraint(A, np.array(self.rlo), np.array(self.rhi))],
                       options={"presolve": True, "mip_rel_gap": 1e-9})
        if res.status != 0:
            raise RuntimeError(f"HiGHS status {res.status}: {res.message}")
        return res.x


def _add(d: dict, j: int, v: float):
    d[j] = d.get(j, 0.0) + v


@dataclass
class MemberBlock:
    """Index sets of one member inside an LP and its linear expressions (dict index -> coefficient)."""
    id: str
    L: np.ndarray
    P: np.ndarray
    Dp: np.ndarray
    Dm: np.ndarray
    Ch: np.ndarray | None
    Dis: np.ndarray | None
    E: np.ndarray | None
    q: list = field(default_factory=list)          # 24 expressions: net import (kWh)
    physical: dict = field(default_factory=dict)  # PV cost + disutility (DKK)


def add_member(lp: LP, m) -> MemberBlock:
    """Add one member (a Q1 SimpleNamespace) to ``lp``: variables, physical constraints, net-import expressions."""
    L = lp.var(T, lb=m.lower, ub=m.upper)
    P = lp.var(T, lb=0.0, ub=m.pv)
    Dp = lp.var(T); Dm = lp.var(T)
    for t in range(T):
        lp.row({L[t]: 1.0, Dp[t]: -1.0, Dm[t]: 1.0}, lo=m.ref[t], hi=m.ref[t])
    lp.row({int(j): 1.0 for j in L}, lo=m.minimum)
    Ch = Dis = E = None
    if m.capacity > 0:
        Ch = lp.var(T, ub=m.charge); Dis = lp.var(T, ub=m.discharge); E = lp.var(T, ub=m.capacity)
        for t in range(T):
            coeffs = {E[t]: 1.0, Ch[t]: -m.rho_ch, Dis[t]: 1.0 / m.rho_dis}
            if t > 0:
                coeffs[E[t - 1]] = -1.0
            rhs = m.initial if t == 0 else 0.0
            lp.row(coeffs, lo=rhs, hi=rhs)
        lp.row({E[T - 1]: 1.0}, lo=m.initial)            # terminal condition of Q1/Q2: e_23 >= E_0
    blk = MemberBlock(m.id, L, P, Dp, Dm, Ch, Dis, E)
    for t in range(T):
        q = {int(L[t]): 1.0, int(P[t]): -1.0}
        if Ch is not None:
            q[int(Ch[t])] = 1.0; q[int(Dis[t])] = -1.0
        blk.q.append(q)
        _add(blk.physical, int(P[t]), m.c)
        _add(blk.physical, int(Dp[t]), m.cL); _add(blk.physical, int(Dm[t]), m.cL)
    return blk


def private_cost(blk: MemberBlock, prices) -> dict:
    """Member's own objective: physical cost + payment sum_t x_t q_t (dict index -> coefficient)."""
    out = dict(blk.physical)
    for t in range(T):
        for j, v in blk.q[t].items():
            _add(out, j, prices[t] * v)
    return out


def _value(expr: dict, x: np.ndarray) -> float:
    return float(sum(v * x[j] for j, v in expr.items()))


def _primal(blk: MemberBlock, x: np.ndarray) -> dict:
    """Hourly primal values of one member (kWh / kW): used as MIP start of the pricing models."""
    out = {"load": x[blk.L], "pv": x[blk.P], "d_plus": x[blk.Dp], "d_minus": x[blk.Dm]}
    if blk.Ch is not None:
        out.update(charge=x[blk.Ch], discharge=x[blk.Dis], soc=x[blk.E])
    return out


def best_response(m, prices) -> dict:
    """Solve one member's LP at fixed prices. Returns objective, physical cost, payment and the hourly net import."""
    lp = LP()
    blk = add_member(lp, m)
    obj = private_cost(blk, prices)
    lp.add_cost(obj)
    x = lp.solve()
    q = np.array([_value(blk.q[t], x) for t in range(T)])
    phys = _value(blk.physical, x)
    return {"objective": _value(obj, x), "physical": phys, "payment": float(np.dot(prices, q)), "q": q,
            "primal": _primal(blk, x)}


def community_response(data, prices, contract, rule: str = "independent") -> dict:
    """Response of all members of ``data`` (Q1 namespace) to ``prices`` under a selection rule.

    contract: dict with ``cap`` (24 kW), ``exc_cost`` (DKK/kWh of excess on top of ``a_d``), ``a_d`` and ``b_d`` (24
    discounted effective import / export prices, DKK/kWh); see evaluate.contract_terms. Returns hourly net/import/export/excess (kWh) and per-member results.
    """
    prices = np.asarray(prices, dtype=float)
    indep = {m.id: best_response(m, prices) for m in data.members}
    if rule == "independent":
        net = sum(r["q"] for r in indep.values())
        imp, exp = np.maximum(net, 0.0), np.maximum(-net, 0.0)
        exc = np.maximum(imp - contract["cap"], 0.0)
        members = {i: {"objective": r["objective"], "physical": r["physical"], "payment": r["payment"], "q": r["q"],
                       "primal": r["primal"]} for i, r in indep.items()}
        return {"net": net, "imports": imp, "exports": exp, "excess": exc, "members": members}
    if rule not in ("optimistic", "pessimistic"):
        raise ValueError("rule must be independent, optimistic or pessimistic")

    last_error = None
    for tol in VALUE_TOLS:
        try:
            lp, blocks, imp, exp, exc = _joint_model(data, prices, contract, indep, rule, tol)
            x = lp.solve(sense=-1.0 if rule == "pessimistic" else 1.0)
            break
        except RuntimeError as err:
            last_error = err
    else:
        raise RuntimeError(f"joint {rule} selection failed for every tolerance: {last_error}")
    members = {}
    for blk in blocks:
        q = np.array([_value(blk.q[t], x) for t in range(T)])
        phys = _value(blk.physical, x)
        members[blk.id] = {"objective": phys + float(np.dot(prices, q)), "physical": phys,
                           "payment": float(np.dot(prices, q)), "q": q, "primal": _primal(blk, x)}
    net = sum(r["q"] for r in members.values())
    return {"net": net, "imports": x[imp], "exports": x[exp], "excess": x[exc], "members": members,
            "member_optimal_values": {i: r["objective"] for i, r in indep.items()}, "value_tol": tol}


def _joint_model(data, prices, contract, indep, rule, tol):
    """Joint LP (optimistic) or MILP (pessimistic) over all members' optimal sets; see community_response."""
    lp = LP()
    blocks = [add_member(lp, m) for m in data.members]
    if indep is not None:                                # every member stays on its optimal set
        for blk in blocks:
            lp.row(private_cost(blk, prices), hi=indep[blk.id]["objective"] + tol)
    imp = lp.var(T); exp = lp.var(T); exc = lp.var(T)
    cost: dict = {}
    for blk in blocks:
        for j, v in blk.physical.items():
            _add(cost, j, v)
    for t in range(T):
        bal = {int(imp[t]): -1.0, int(exp[t]): 1.0}
        for blk in blocks:
            for j, v in blk.q[t].items():
                _add(bal, j, v)
        lp.row(bal, lo=0.0, hi=0.0)                       # sum_i q_i,t = import_t - export_t
        lp.row({int(imp[t]): 1.0, int(exc[t]): -1.0}, hi=contract["cap"][t])
        _add(cost, int(imp[t]), contract["a_d"][t]); _add(cost, int(exp[t]), -contract["b_d"][t])
        _add(cost, int(exc[t]), contract["exc_cost"])     # alpha - (1 - beta) tau_imp: no tariff above the cap
    if rule == "pessimistic":
        s = lp.var(T, ub=1.0, integer=True); b = lp.var(T, ub=1.0, integer=True)
        for t in range(T):
            lp.row({int(imp[t]): 1.0, int(s[t]): -BIG_M}, hi=0.0)                    # import only if s = 1
            lp.row({int(exp[t]): 1.0, int(s[t]): BIG_M}, hi=BIG_M)                   # export only if s = 0
            lp.row({int(exc[t]): 1.0, int(imp[t]): -1.0, int(b[t]): BIG_M}, hi=BIG_M - contract["cap"][t])
            lp.row({int(exc[t]): 1.0, int(b[t]): -BIG_M}, hi=0.0)                    # excess = [import - cap]+
    lp.add_cost(cost)
    return lp, blocks, imp, exp, exc



def central_dispatch(data, contract) -> dict:
    """Model C of Q1 on one scenario (perfect information, direct control): the joint model WITHOUT the members'
    optimality constraints, minimising community cost. Its cost J^C(s) is the lower bound of any price vector's cost
    in that scenario and serves as the reference ("regret" = J(x, s) - J^C(s))."""
    lp, blocks, imp, exp, exc = _joint_model(data, None, contract, None, "optimistic", 0.0)
    x = lp.solve()
    members = {}
    for blk in blocks:
        q = np.array([_value(blk.q[t], x) for t in range(T)])
        members[blk.id] = {"objective": _value(blk.physical, x), "physical": _value(blk.physical, x),
                           "payment": 0.0, "q": q, "primal": _primal(blk, x)}
    net = sum(r["q"] for r in members.values())
    return {"net": net, "imports": x[imp], "exports": x[exp], "excess": x[exc], "members": members}


def reference_cost(m, p, tau_imp: float, tau_exp: float) -> float:
    """Cost of one member alone in the reference model R of Q1 (no contract, full tariffs, own meter), DKK:
    min sum_t [ c P_t + cL (d+_t + d-_t) + (p_t + tau_imp) I_t - (p_t - tau_exp) E_t ],  q_t = I_t - E_t.
    Used for individual rationality: the member's private cost under a price vector is compared with this value
    on the same scenario."""
    lp = LP()
    blk = add_member(lp, m)
    imp = lp.var(T); exp = lp.var(T)
    cost = dict(blk.physical)
    for t in range(T):
        row = {int(imp[t]): -1.0, int(exp[t]): 1.0}
        for j, v in blk.q[t].items():
            _add(row, j, v)
        lp.row(row, lo=0.0, hi=0.0)
        _add(cost, int(imp[t]), p[t] + tau_imp); _add(cost, int(exp[t]), -(p[t] - tau_exp))
    lp.add_cost(cost)
    x = lp.solve()
    return float(sum(v * x[j] for j, v in cost.items()))
