"""Reproduce Question 2(b): python main_q2_linear.py.

Loads Q2_linear without changing JSON inputs, solves the base and c_L sweep,
checks each LP against independent scalar enumeration, and saves CSV/JSON/PNG.
"""
from dataclasses import replace
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp

from src.data_loader import load_question
from src.model_q2_linear import LinearDisutilityModel, verify_solution


def plot_results(base, runs, table, out):
    """Save schedule, load-response, and daily metrics figures with units."""
    h = base.hourly
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(h.index, h.reference_load, where="mid", label="Reference profile", linewidth=2)
    ax.step(h.index, h.load, where="mid", label="Optimal load", linewidth=2)
    ax.plot(h.index, h.pv_available, "--", color="grey", label="Available PV", alpha=.7)
    ax.set(xlabel="Hour", ylabel="Energy per hour (kWh)", title="Q2(b) base case: c_L = 1.43 DKK/kWh", xticks=range(0,24,2))
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"base_schedule.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for t in [7, 9, 11, 18]:
        ax.plot(table.c_L, [r.hourly.loc[t,"load"] for r in runs], ".-", label=f"Hour {t}")
    ax.set(xlabel="c_L (DKK/kWh)", ylabel="Optimal hourly consumption (kWh)", title="Load response; exact thresholds may admit multiple optima")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"load_response.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    for ax, cols, unit in zip(axes.flat, [["C_DKK","D_DKK"],["E_day_kWh"],["A_kWh"],["N_bind"]], ["DKK","kWh","kWh","hours"]):
        for col in cols: ax.plot(table.c_L, table[col], ".-", label=col)
        ax.set(xlabel="c_L (DKK/kWh)", ylabel=unit); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"sweep_metrics.png", dpi=180); plt.close(fig)


def main():
    """Run the supplied base case and threshold-focused one-parameter sweep."""
    data = load_question("Q2_linear")
    out = Path(__file__).resolve().parent/"results"/"Q2_linear"
    out.mkdir(parents=True, exist_ok=True)
    base = LinearDisutilityModel(data).build().solve()
    verify_solution(base, data); base.save(out)
    thresholds = [1.41, 1.45, 1.50, 1.60, 1.70, 1.90, 2.00, 2.10]
    representatives = [.5, 1.4, 1.43, 1.47, 1.55, 1.65, 1.8, 1.95, 2.05, 2.2]
    values = sorted(set(representatives+[round(k+offset,6) for k in thresholds for offset in [-.001,0,.001]]))
    rows, runs = [], []
    for value in values:
        scenario = replace(data, linear_disutility=value)
        result = LinearDisutilityModel(scenario).build().solve()
        verify_solution(result, scenario)
        result.save(out/"sweep", f"cL_{value:.3f}")
        rows.append(result.metrics); runs.append(result)
    table = pd.DataFrame(rows)
    table.to_csv(out/"sweep_metrics.csv", index=False)
    assert (np.diff(table.E_day_kWh)>=-1e-6).all()
    assert (np.diff(table.A_kWh)<=1e-6).all()
    compact = table[table.c_L.isin(representatives)]
    compact.to_csv(out/"representative_metrics.csv", index=False)
    changes = []
    for k in thresholds:
        low = runs[values.index(round(k-.001,6))].hourly
        high = runs[values.index(round(k+.001,6))].hourly
        changed = list(low.index[abs(high.load-low.load)>1e-6])
        assert changed, f"No change found at {k}"
        for t in changed:
            changes.append({"c_L_threshold":k,"hour":int(t),"load_below_kWh":low.loc[t,"load"],"load_above_kWh":high.loc[t,"load"]})
    pd.DataFrame(changes).to_csv(out/"threshold_changes.csv",index=False)
    (out/"validation.json").write_text(json.dumps({"runs_verified":len(runs)+1,"checks":"LP feasibility, auxiliary tightness, accounting, no simultaneous trade, independent vertex enumeration, no above-reference load, monotonic sweep, threshold changes", "gurobi_version":gp.gurobi.version(),"numpy_version":np.__version__,"pandas_version":pd.__version__},indent=2),encoding="utf-8")
    plot_results(base, runs, table, out)
    print("BASE",json.dumps(base.metrics,indent=2))
    print(compact[["c_L","C_DKK","D_DKK","E_day_kWh","A_kWh","N_bind"]].to_string(index=False))
    print(f"Verified {len(runs)+1} LP runs. Results: {out}")


if __name__ == "__main__":
    main()
