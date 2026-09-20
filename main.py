"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import numpy as np
import argparse

from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model_q1 import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)

    # 新增测试 new test
    # --- 1(d).i verification: no simultaneous import & export -------------------
    hr = results.hourly
    both_active = (hr["import"] > 1e-6) & (hr["export"] > 1e-6)
    n_both = int(both_active.sum())
    print(f"[check 1.d.i] hours with simultaneous import&export: {n_both}")
    if n_both > 0:
        print(hr.loc[both_active, ["import", "export"]])
    else:
        print("[check 1.d.i] OK: g_import * g_export = 0 in every hour (tau_imp+tau_exp > 0)")

    # --- 1(d).ii verification: three regimes of the load ------------------------
    hr = results.hourly
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    u = data.consumption_utility
    Lmin, Lmax = data.load_min_kWh, data.load_max_kWh

    R1 = (u > p_imp)
    R2 = (u < p_exp)
    R3 = ~(R1 | R2)

    print(f"\n[check 1.d.ii] R1 (L=Lmax={Lmax}): hours {list(np.where(R1)[0])}")
    print(f"[check 1.d.ii] R2 (L=Lmin={Lmin}): hours {list(np.where(R2)[0])}")
    print(f"[check 1.d.ii] R3 (L=PV):          hours {list(np.where(R3)[0])}")
    print(f"[check 1.d.ii] L at R1 should = {Lmax}, actual = {hr.loc[R1,'load'].unique()}")
    print(f"[check 1.d.ii] L at R2 should = {Lmin}, actual = {hr.loc[R2,'load'].unique()}")
    print(f"[check 1.d.ii] L at R3 should = PV_available, max err = {(hr.loc[R3,'load']-hr.loc[R3,'pv_available']).abs().max():.2e}")


    # --- 1(d).iii verification: PV curtailment (revised) ----------------------
    hr = results.hourly
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    c_pv = data.pv_marginal_cost
    u = data.consumption_utility

    R1 = u > p_imp
    R2 = u < p_exp
    R3 = ~(R1 | R2)
    marginal_value = np.where(R1, p_imp, np.where(R2, p_exp, u))

    theoretical_curtail = c_pv > marginal_value + 1e-6
    actual_curtail = (hr["pv_available"] > 1e-6) & (hr["pv"] < hr["pv_available"] - 1e-6)

    print(f"\n[check 1.d.iii] hours where c_pv > marginal value (theory): {list(np.where(theoretical_curtail)[0])}")
    print(f"[check 1.d.iii] solver actually curtails in:                {list(np.where(actual_curtail)[0])}")
    print(
        "[check 1.d.iii] note: these hours satisfy the curtailment condition "
        "c_pv > marginal value, but PV_available = 0, so no actual curtailment can occur."
    )

    # --- 1(d).iv verification: extreme PV bounds (revised) --------------------
    from src.scenarios import set_load_preferences

    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    c_pv = data.pv_marginal_cost
    u = data.consumption_utility

    # Scenario A: L_min = 1.0 (PV < L_min in many hours)
    data_a = set_load_preferences(data, load_min_kWh=1.0)
    res_a = FlexibleConsumerModel(data_a).build().solve()
    hr_a = res_a.hourly

    R1 = u > p_imp
    R2 = u < p_exp
    R3 = ~(R1 | R2)

    print("\n[check 1.d.iv-A] L_min=1.0")
    print(f"  R1 hours (L should = L_max=6.0): {list(np.where(R1)[0])}")
    print(f"    actual load: {hr_a.loc[R1, 'load'].values}")
    print(f"  R2 hours (L should = L_min=1.0): {list(np.where(R2)[0])}")
    print(f"    actual load: {hr_a.loc[R2, 'load'].values}")
    print(f"  R3 hours (L changed from PV to L_min=1.0): {list(np.where(R3)[0])}")
    print(f"    actual load: {hr_a.loc[R3, 'load'].values}")

    # Scenario B: L_max = 4.0 (PV > L_max at noon)
    data_b = set_load_preferences(data, load_max_kWh=4.0)
    res_b = FlexibleConsumerModel(data_b).build().solve()
    hr_b = res_b.hourly

    print("\n[check 1.d.iv-B] L_max=4.0")
    mask_b = data.pv_available > 4.0
    print(f"  hours with PV > L_max: {list(np.where(mask_b)[0])}")
    print(f"    which regime: R1={list(np.where(R1 & mask_b)[0])}, "
          f"R2={list(np.where(R2 & mask_b)[0])}, "
          f"R3={list(np.where(R3 & mask_b)[0])}")
    print(f"    load: {hr_b.loc[mask_b, 'load'].values}")
    print(f"    pv actual: {hr_b.loc[mask_b, 'pv'].values}")
    print(f"    export: {hr_b.loc[mask_b, 'export'].values}")
    print(f"    note: if c_pv > p_exp, surplus PV is curtailed not exported")



    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
