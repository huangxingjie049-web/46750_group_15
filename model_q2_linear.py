"""Question 2(b): continuous LP with absolute-deviation disutility.

Energy variables are kWh per one-hour period; money is DKK.
This model reuses the input loader and naming conventions of Question 1.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from .data_loader import InputData


@dataclass
class LinearDisutilityResults:
    """Hourly decisions and separately accounted daily metrics."""
    hourly: pd.DataFrame
    metrics: dict

    def save(self, folder: Path, tag: str = "base") -> None:
        """Save physical hourly quantities and DKK daily metrics."""
        folder.mkdir(parents=True, exist_ok=True)
        self.hourly.to_csv(folder / f"{tag}_hourly.csv", index_label="hour")
        (folder / f"{tag}_metrics.json").write_text(
            json.dumps(self.metrics, indent=2), encoding="utf-8")


class LinearDisutilityModel:
    """Build and solve max(-procurement cost - c_L * absolute deviation)."""

    def __init__(self, data: InputData, verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model("Q2_linear")
        self.m.Params.OutputFlag = int(verbose)
        self.var = {}
        self.con = {}

    def build(self):
        """Declare five continuous variables per hour and linear constraints."""
        d, m, T = self.data, self.m, self.T
        if d.reference_load is None or d.linear_disutility is None or d.linear_disutility <= 0:
            raise ValueError("A reference profile and c_L > 0 are required")
        if np.any(d.energy_price + d.import_tariff < d.energy_price - d.export_tariff):
            raise ValueError("Import price must not be below export price")
        for name in ["import", "export", "abs_deviation"]:
            self.var[name] = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name=name)
        for name in ["load", "pv"]:
            self.var[name] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        I, E, L, P, z = [self.var[k] for k in ["import", "export", "load", "pv", "abs_deviation"]]
        self.con["balance"] = m.addConstrs((P[t]+I[t]-L[t]-E[t] == 0 for t in T), name="balance")
        self.con["load_lower"] = m.addConstrs((L[t] >= d.load_min_kWh for t in T), name="load_lower")
        self.con["load_upper"] = m.addConstrs((L[t] <= d.load_max_kWh for t in T), name="load_upper")
        self.con["pv_lower"] = m.addConstrs((P[t] >= 0 for t in T), name="pv_lower")
        self.con["pv_upper"] = m.addConstrs((P[t] <= d.pv_available[t] for t in T), name="pv_upper")
        self.con["deviation_pos"] = m.addConstrs((z[t] >= L[t]-d.reference_load[t] for t in T), name="deviation_pos")
        self.con["deviation_neg"] = m.addConstrs((z[t] >= d.reference_load[t]-L[t] for t in T), name="deviation_neg")
        self.cost = gp.quicksum(d.pv_marginal_cost*P[t]+(d.energy_price[t]+d.import_tariff)*I[t]
                                -(d.energy_price[t]-d.export_tariff)*E[t] for t in T)
        self.disutility = d.linear_disutility * gp.quicksum(z[t] for t in T)
        m.setObjective(-self.cost-self.disutility, GRB.MAXIMIZE)
        return self

    def solve(self) -> LinearDisutilityResults:
        """Optimize, check status, and compute metrics from physical decisions."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Q2 solver status: {self.m.Status}")
        d = self.data
        h = pd.DataFrame(index=pd.Index(self.T, name="hour"))
        h["reference_load"] = d.reference_load
        h["pv_available"] = d.pv_available
        h["price"] = d.energy_price
        h["a_t"] = d.energy_price+d.import_tariff
        h["b_t"] = d.energy_price-d.export_tariff
        for name, variables in self.var.items():
            h[name] = [variables[t].X for t in self.T]
        for name, constraints in self.con.items():
            h[f"dual_{name}"] = [constraints[t].Pi for t in self.T]
        h["deviation"] = h["load"]-h["reference_load"]
        h["procurement_cost"] = d.pv_marginal_cost*h["pv"]+h["a_t"]*h["import"]-h["b_t"]*h["export"]
        h["disutility"] = d.linear_disutility*h["deviation"].abs()
        h["curtailment"] = h["pv_available"]-h["pv"]
        eps = 1e-6
        h["bound_binding"] = (abs(h["load"]-d.load_min_kWh)<=eps) | (abs(h["load"]-d.load_max_kWh)<=eps)
        metrics = {
            "c_L": float(d.linear_disutility), "status": "OPTIMAL", "objective_DKK": float(self.m.ObjVal),
            "C_DKK": float(h["procurement_cost"].sum()), "D_DKK": float(h["disutility"].sum()),
            "E_day_kWh": float(h["load"].sum()), "A_kWh": float(h["deviation"].abs().sum()),
            "N_bind": int(h["bound_binding"].sum()),
            "N_bind_positive_reference": int((h["bound_binding"] & (h["reference_load"]>eps)).sum()),
            "import_kWh": float(h["import"].sum()), "export_kWh": float(h["export"].sum()),
            "pv_kWh": float(h["pv"].sum()), "curtailment_kWh": float(h["curtailment"].sum()),
        }
        return LinearDisutilityResults(h, metrics)


def verify_solution(result: LinearDisutilityResults, data: InputData, tol: float = 1e-6) -> None:
    """Check feasibility/accounting and an independent hourly vertex enumeration.

    The enumeration computes optimal PV/trade for each scalar load candidate,
    independently of the LP auxiliary-variable construction. Ties may differ in
    schedule, so compare objectives rather than a chosen tied solution.
    """
    h = result.hourly
    assert abs(h["pv"]+h["import"]-h["load"]-h["export"]).max() < tol
    assert (h["load"] >= data.load_min_kWh-tol).all() and (h["load"] <= data.load_max_kWh+tol).all()
    assert (h["pv"] >= -tol).all() and (h["pv"] <= h["pv_available"]+tol).all()
    assert (h[["import", "export", "abs_deviation"]] >= -tol).all().all()
    assert abs(h["abs_deviation"]-h["deviation"].abs()).max() < tol
    assert (np.minimum(h["import"], h["export"]) < tol).all()
    assert abs(result.metrics["objective_DKK"]+result.metrics["C_DKK"]+result.metrics["D_DKK"]) < tol
    c, cL = data.pv_marginal_cost, data.linear_disutility
    for t, row in h.iterrows():
        a, b, cap, ref = row["a_t"], row["b_t"], row["pv_available"], row["reference_load"]
        candidates = {data.load_min_kWh, data.load_max_kWh,
                      np.clip(ref, data.load_min_kWh, data.load_max_kWh),
                      np.clip(cap, data.load_min_kWh, data.load_max_kWh)}
        values = []
        for load in candidates:
            pv = cap if c <= b else (min(load, cap) if c <= a else 0)
            values.append(c*pv+a*max(load-pv, 0)-b*max(pv-load, 0)+cL*abs(load-ref))
        assert abs(min(values)-row["procurement_cost"]-row["disutility"]) < tol, f"hour {t}"
    if (h["reference_load"] >= data.load_min_kWh).all() and (h["reference_load"] <= data.load_max_kWh).all():
        assert (h["load"] <= h["reference_load"]+tol).all()
