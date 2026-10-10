"""Entry point: run a selected model and save results and figures.

Usage:
    python main.py --question Q1_caseA
        Run the Q1 Case A base case.

    python main.py --question Q1_caseB
        Run the Q1 Case B base case.

    python main.py --question Q2_quadratic
        Run the Q2(c) base case only.

    python main.py --question Q2_quadratic --sweep
        Run the Q2(c) base case and quadratic coefficient sweep.

    python main.py --question Q1_caseA --scenarios
        Run Q1 Case A and the example sensitivity scenarios.

    python main.py
        Run Q1 Case A by default.

Outputs are saved to results/<question>/.
Q2(c) can also be run independently with: python run_q2c.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model_q1 import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


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


def main(argv=None):
    parser=argparse.ArgumentParser(description="Q1(e): solve and plot Case A or B.")
    parser.add_argument('--case',choices=['Q1_caseA','Q1_caseB'],default='Q1_caseA')
    parser.add_argument('--scenarios',action='store_true')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args(argv)
    out=args.output_dir or RESULTS_DIR/args.case
    out.mkdir(parents=True,exist_ok=True)
    if not args.show: matplotlib.use('Agg')
    base=run_base_case(args.case,out,args.show)
    if args.scenarios and base is not None: run_scenarios(args.case,out)
    print(f'Outputs written to {out}')
