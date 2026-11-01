# Assignment 2 - Q1: Reproduce the results

## Setup

Use Python 3.10 or later and a valid Gurobi licence. Extract the ZIP and open a terminal in the `Q1_models` folder. Keep the Python files and `data/` folder together.

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the base case

```powershell
python base_case.py
```

Outputs are saved in `results/base_case/`:

- `comparison.csv`: model comparison table.
- `validation_summary.json`: validation results.
- `feeder_comparison.png` / `.pdf`: feeder import and capacity exceedance.
- `dispatch_response.png` / `.pdf`: load, battery and PV dispatch.
- `R/`, `I/`, `C/`: dispatch CSVs, model metrics, duals, LP files and validation results.

To redraw the figures from existing results without solving again:

```powershell
python base_case.py --plots-only
```

## Additional saved outputs

```powershell
python main.py --variant all
python audit_reference_ties.py
```

These commands reproduce the outputs in `results/minimum/`, including `reference_tie_audit.json`. To run one model, replace `all` with `R`, `I` or `C`.

## Notebook

Open `Assignment2_Q1_base_case.ipynb` in Jupyter or VS Code, select the Python environment containing the installed dependencies, and run all cells. Keep the notebook in the `Q1_models` folder.
