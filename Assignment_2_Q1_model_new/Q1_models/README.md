# Assignment 2 Q1 — Is the community worth forming?

This project reproduces Group 15's base case, contract-value metrics,
contract sensitivity analysis, and member settlement for Question 1 of
DTU course 46750.

## Setup and run

Use Python 3.10 or later with a Gurobi licence that can solve the full
models (up to 4,224 variables in the base case). From `Q1_models/`, run:

```sh
python -m pip install -r requirements.txt
python run_all.py
```

All steps use the supplied JSON inputs in `data/`. They write to `results/`
and leave the inputs unchanged. The model class in `main.py` implements
all three variants and is also used by the sensitivity sweeps.

| Step | Command | Output |
| --- | --- | --- |
| Base case | `python base_case.py` | `results/base_case/` — comparison, cost components, dispatch, duals, figures, and validation with Gurobi and HiGHS |
| Contract value | `python calculate_metrics.py` | `results/contract_value_metrics.csv` — three cost metrics and three DSO import metrics from the saved base case |
| Sensitivity | `python sensitivity.py` | `results/sensitivity/` — discount, cap factor, and penalty sweeps, thresholds, and PNG/PDF figures |
| Settlement | `python settlement_analysis.py` | `results/settlement/` — hourly prices and transfers, member costs, individual rationality, and manager budget |

Contract value and settlement use the base-case dispatch. The sensitivity
script compares its base points with the contract-value CSV when it is
available. To redraw figures from the saved results, run
`python base_case.py --plots-only` and `python plot_results.py`.

## Results and conventions

| Model | Total cost (DKK/day) | Peak import (kW) | Feeder exceedance (kWh/day) | Hours above cap |
| --- | ---: | ---: | ---: | ---: |
| R · No contract | 398.24 | 27.60 | 66.70 | 9 |
| I · Individual contracts | 480.13 | 11.08 | 4.29 | 3 |
| C · Community contract | 372.16 | 9.39 | 0.00 | 0 |

The models use one-hour intervals and a minimum terminal state of charge
equal to each battery's initial state. I and C split imports in the
objective and energy balance. The discounted import tariff applies to
within-cap imports. The above-cap tier pays the day-ahead price plus the
penalty. In the low-penalty sweep, the model can use that tier before the
within-cap tier is full. The results report physical exceedance from the
metered flows separately from the quantity assigned to the above-cap tier.

The settlement uses the negative Gurobi `Pi` of C's balance (17c) as the
internal price. The manager retains 163.77 DKK/day of capacity rent.
Individual rationality holds for 3 of 16 members against R and 8 of 16
against I. Alternate LP optima can change dispatch and dual values.

## Notebook and report material

- `Assignment2_Q1_base_case.ipynb` walks through the base case. Run it in
  Jupyter with the same dependencies and a Gurobi licence. To rebuild the
  notebook source, run `python build_notebook.py`.
- `answer_overleaf.tex` contains the mathematical formulation. The
  `report/` folder contains the contract-value and sensitivity sections.
  In Overleaf, use `preview_contract_value_metrics.tex` or
  `preview_sensitivity.tex` as the main document to compile a section preview.
- `python main.py --variant all` runs the model variants directly.
  `python audit_reference_ties.py` checks how alternate optimal reference
  schedules affect measured exceedance.
