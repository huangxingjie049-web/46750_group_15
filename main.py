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
    parser.add_argument("--cq", type=float, default=None, help="Q2(c) quadratic coefficient")
    parser.add_argument("--sweep", action="store_true", help="run the Q2(c) coefficient sweep")
    args = parser.parse_args()

    if args.question == "Q2_quadratic":
        if args.scenarios or args.show:
            parser.error("For Q2(c), use --sweep; figures are saved to disk without --show.")
        from run_q2c import reproduce, DEFAULT_SWEEP
        reproduce(args.cq, DEFAULT_SWEEP if args.sweep else [])
        return
    if not args.question.startswith("Q1_"):
        parser.error("This project currently implements Q1 and Q2_quadratic only.")
    if args.cq is not None or args.sweep:
        parser.error("--cq and --sweep are for Q2_quadratic only.")

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
