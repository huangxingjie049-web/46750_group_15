# DTU 46750 — Group 15 assignment code

All implemented tasks can be selected through **main.py**. The original input
files under **data/** are preserved byte-for-byte. No branches were merged or
pushed. This package reorganises the uploaded working tree, not other branches.

## Install and run

From the repository root (Windows CMD / PowerShell):

```text
python -m pip install -r requirements.txt
python main.py --list-tasks
python main.py --help
```

Gurobi requires your existing academic licence. Tests additionally need pytest.

| Task | Command | Default outputs |
|---|---|---|
| Q1(e), Case A | `python main.py --task 1e --case Q1_caseA` | results/Q1_caseA |
| Q1(e), Case B | `python main.py --task 1e --case Q1_caseB` | results/Q1_caseB |
| Q1(f), A/B comparison | `python main.py --task 1f` | results/Q1f |
| Q2(c), base only | `python main.py --task 2c` | results/Q2_quadratic |
| Q2(c), coefficient sweep | `python main.py --task 2c --sweep` | results/Q2_quadratic |
| Q3(d), solve and validate | `python main.py --task 3d` | results/Q3d |
| Q3(e), compare and plot | `python main.py --task 3e` | results/Q3 |
| Q3(g), cyclic battery | `python main.py --task 3g` | results/Q3g |
| Q3(g), compare cyclic/value | `python main.py --task 3g --mode both --pi 2` | results/Q3g |
| Q3(g), include sensitivity | `python main.py --task 3g --mode both --pi 2 --sweep` | results/Q3g |
| Estimate pi from next day | `python main.py --task 3g-pi` | results/Q3g_pi_estimation |
| Estimate pi and run today | `python main.py --task 3g-pi --run-today` | results/Q3g_pi_estimation/today_using_estimated_pi |

`python main.py` still runs Q1 Case A. Use `--output-dir PATH` to keep different
runs separate. Relative output paths are interpreted from your current working
directory; default paths are anchored to the repository.

View each task's options:

```text
python main.py --task 3g --help
python main.py --task 3g-pi --help
```

Examples with overrides:

```text
python main.py --task 3e --emin 40 --cq 1
python main.py --task 2c --sweep 0.1 0.5 1 2 5
python main.py --task 3g --mode both --pi 2 --sweep --price-scales 0.5 1 1.5 --capacities 2 4 8 --pi-values 0 1 2 3 4
python main.py --task 3g-pi --next-day-end-soc 2 --run-today --output-dir results/pi_fixed2
python -m pytest -q
```

Q1's original example scenarios remain available with
`python main.py --task 1e --case Q1_caseA --scenarios`. They are examples, not
newly designed assignment experiments.

## Source layout

| Folder/file | Contents |
|---|---|
| src/1e/ | Q1 model.py and runner.py |
| src/1f/ | Runner comparing the existing Case A / Case B models |
| src/2c/ | Quadratic model.py and runner.py (figures/sweep) |
| src/3d/ | Daily-energy model.py, validation.py, runner.py |
| src/3e/ | Comparison/figures runner.py; reuses 3d model |
| src/3g/ | Battery model.py, validation.py, runner.py, terminal_value.py, estimate_pi.py |
| src/data_loader.py | Original shared data loader, unchanged |
| src/model.py | Original teacher model template, unchanged |
| src/plotting.py, src/scenarios.py | Existing shared utilities, unchanged |
| tests/ | Automatic checks across models and command routing |
| data/ | Original appliance/bus/case JSON files, unchanged |
| docs/data_preservation.json | SHA-256 of all original data files |
| docs/README_original.md | Original uploaded README, retained |
| docs/verification/ | Logs from executed tasks |

The folder names **1e, 3d, 3g, ...** match the requested assignment labels.
Because normal Python import syntax cannot use a numeric-leading package name,
main.py loads them with `importlib.import_module`. Within a task, relative
imports such as `from .model import ...` are valid. Use main.py to execute tasks;
do not execute nested runner.py files directly.

Old root scripts and src/model_q*.py / validation_q*.py are small compatibility
wrappers. The actual implementation now resides in the task folders. They
preserve old commands and imports used by teammates/tests; they are not duplicate
models. Existing commands like `python run_q3.py` still work.
The aliases `--question Q1_caseA`, `Q1_caseB`, `Q2_quadratic`, `Q3`,
`Q3_battery` also work with main.py. Do not combine conflicting selectors.

The uploaded working tree does not contain implementations for Q2(b), Q2(e)
or Q3(f). Their input data/the report are retained; the dispatcher does not
pretend those tasks are implemented. Other branches still require integration.
Q3(e) is the base comparison; **Q3(f)** is the no-battery sensitivity question;
Q3(g)(v) is the battery sensitivity question.

## Model assumptions and report consistency

- Q3(d) adds sum(load) >= minimum daily energy to Q2(c).
- Q3(g) uses initial energy 2 kWh, capacity 4 kWh, charge/discharge limits
  1.5 kW, efficiencies .95/.95, one-hour intervals. These match the supplied
  JSON and the report's g(i) section. Storage e_t is AFTER hour t.
- Cyclic mode fixes day-1 final energy to initial energy.
- Value mode has no cyclic equality and adds pi*final_energy ONCE to W.
  Report equation (41a) currently places this reward inside the hourly sum;
  move it outside. The report itself was not edited.
- Battery charging is not counted as load toward the daily requirement.
- No binary/nonlinear charge-discharge exclusion is imposed. Overlap is
  recorded for checking the theoretical prediction in g(ii), not silently banned.
- pi=2 and default sweep ranges remain **illustrative**, not choices agreed by Joe.
- Current-day total_cost differs from the objective including terminal value.
  Remaining inventory value is not realised current-day revenue.
- Inventory-adjusted battery value compares with keeping identical initial
  stock idle and valuing it with the same pi; see comparison_metrics.csv.

### Next-day calibration

The uploaded ZIP did not yet contain the next-day add-on; it is included here
under src/3g. It uses supplied next-day price/PV/reference forecasts, retaining
other supplied parameters. Default day-2 end energy is free with zero day-3
reward; this boundary assumption needs a stated justification.

`--next-day-end-soc 2` instead fixes day-2 end energy to 2 kWh in every
perturbation. It does NOT change with initial energy. Default expansion point
is 2 kWh, perturbation .01 kWh; left/right differences estimate pi, checked
against the initial-energy dual and perturbations .001/.01/.1 kWh. A value
curve is also saved to expose the local approximation's limitations.

## Updating your Windows working repository

This is a source package without .git or Python caches. Keep your local .git
folder and history. Before copying, make sure your working changes are saved
in Git or a separate backup. Copy the package's contents into the existing
repository root, replacing files when requested. Keep original data/ intact;
the packaged versions are identical to the uploaded data.

There is no need to delete old runner/model files: their packaged versions are
compatibility wrappers. Then run `python main.py --list-tasks` and
`python -m pytest -q`. No commit, push or merge has been performed for you.
