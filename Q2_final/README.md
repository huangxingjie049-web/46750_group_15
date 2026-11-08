# Question 2 — Uniform Pricing for an Energy Community

**DTU 46750 · Optimization in Modern Power Systems · Assignment 2 · Group 15**

This folder contains the final Python implementation, validation scripts and figures for Question 2. It studies whether a common hourly electricity price can coordinate the decisions of 16 community members while respecting the manager's contractual requirements.

## 1. Repository layout

`Q2_final` is intended to sit **next to** `Q1_models` at the repository root:

```text
46750_group_15/
├── Q1_models/                 # Q1 inputs and central-dispatch results
├── Q2_final/
│   ├── Q2_1/
│   │   ├── data_interface.py
│   │   ├── follower.py
│   │   ├── kkt_validation.py
│   │   ├── run_kkt_validation.py
│   │   └── verify_manual_kkt.py
│   ├── Q2_2/
│   │   ├── kkt_member_builder.py
│   │   ├── community_response.py
│   │   ├── manager_accounting.py
│   │   ├── run_manager_parameterized.py
│   │   ├── run_manager_ir_relaxation.py
│   │   ├── check_q2_step3_c_prices.py
│   │   ├── plot_q2_figures.py
│   │   ├── outside_costs_R.json
│   │   ├── dp_bb_start.sol
│   │   └── ir_relax_cost_limited.sol
│   └── Q2figures/
│       ├── Q2_3_feeder_net_exchange_comparison.png
│       ├── Q2_5_social_cost_vs_capacity.png
│       └── Q2_5_compensation_vs_capacity.png
└── ...
```

The implementation uses Q1 data and the Q1 central-dispatch outputs, so **do not move `Q2_final` into `Q2_models` or remove the required Q1 files**. The Q1 C price-response check reads `Q1_models/results/base_case/C/hourly.csv` and `members.csv`.

## 2. Requirements

Use the repository's existing Python environment and dependency files (`requirements.txt` / `environment.yaml` at the repository root). The main dependencies include `gurobipy` and `matplotlib`. Running the optimization models requires a valid Gurobi licence.

The reported tests were run with Gurobi 13.0.3. From the **repository root**, activate your environment and run the commands below. All commands use relative paths from that root.

## 3. Main scripts and experiments

| Script | Purpose |
| --- | --- |
| `Q2_1/data_interface.py` | Load and adapt Q1 member and contract data. |
| `Q2_1/follower.py` | Solve individual members' price-response problems. |
| `Q2_1/run_kkt_validation.py` | Test follower primal/dual feasibility and KKT conditions. |
| `Q2_1/verify_manual_kkt.py` | Independently check KKT stationarity equations. |
| `Q2_2/kkt_member_builder.py` | Construct follower KKT constraints inside the manager model. |
| `Q2_2/run_manager_parameterized.py` | Main parameterized manager model (DP, BB, strict IR and post-solution transfers). |
| `Q2_2/run_manager_ir_relaxation.py` | Minimize the maximum individual-rationality relaxation, subject to the model's other constraints. |
| `Q2_2/check_q2_step3_c_prices.py` | Check Q1 C dual prices and illustrate alternative optimal follower responses. |
| `Q2_2/plot_q2_figures.py` | Regenerate the three report figures from the saved numerical data and Q1 C outputs. |

### 3.1 Validate the follower model

```powershell
python Q2_final/Q2_1/run_kkt_validation.py
python Q2_final/Q2_1/verify_manual_kkt.py
```

The independent stationarity test covers 16 members and 2,496 equations. The largest observed residual was `4.441e-16`.

### 3.2 Run the manager model

Open `Q2_final/Q2_2/run_manager_parameterized.py` and set the configuration variables near the beginning of the script before executing it. The configurations used in the study are:

- `MODE = "DP"`: dynamic pricing without budget balance.
- `MODE = "BB"`: dynamic pricing with budget balance.
- `MODE = "STRICT_IR"`: dynamic pricing with budget balance and individual rationality imposed directly.
- `MODE = "TRANSFER"`: solve the BB dispatch and then compute member-specific, zero-sum fixed transfers.

Set `CAP_MULTIPLIER` to `0.90`, `1.00` or `1.10` for the capacity sensitivity study. The base contractual capacity is `9.39 kW`. The tested base BB configuration uses `MODE = "BB"`, `CAP_MULTIPLIER = 1.00` and the saved warm start `dp_bb_start.sol`.

```powershell
python Q2_final/Q2_2/run_manager_parameterized.py
```

**Important:** `TRANSFER` does not jointly optimize transfer payments and dispatch. Fixed transfers are calculated after finding a BB dispatch and do not change the marginal price incentives in that selected solution.

### 3.3 Test price indifference and feeder response

```powershell
python Q2_final/Q2_2/check_q2_step3_c_prices.py
```

The test verifies that the Q1 C schedules are individually optimal at the corresponding dual prices, while independently solved follower models may select different, equally optimal schedules. The displayed alternative response is **one** such selection; it is **not** a proven worst-case response.

### 3.4 Run the IR relaxation experiment

```powershell
python Q2_final/Q2_2/run_manager_ir_relaxation.py
```

This experiment has a `120 s` time limit and may replace `Q2_2/ir_relax_cost_limited.sol` with the feasible solution from the current run. Preserve the supplied file if you want to retain the reported incumbent. The objective of this model is the IR relaxation `epsilon`, **not** community social cost. The community cost is reported separately in the output. Gurobi's stopping status and bound must be checked before making optimality claims.

### 3.5 Regenerate the report figures

```powershell
python Q2_final/Q2_2/plot_q2_figures.py
```

The script saves three PNGs to `Q2_final/Q2figures/`. The sensitivity values in the plotting script come from previous optimization runs; running the plotting script **does not** re-solve the sensitivity cases. Existing PNGs with the same filenames are overwritten.

## 4. Selected reported results

Values below refer to solutions actually found by the numerical experiments, and costs are in DKK.

| Experiment | Community social cost | Notes |
| --- | ---: | --- |
| Q1 R (no-contract reference) | 398.241694 | Reference used for savings and member-level IR comparisons. |
| Q1 C (central dispatch) | 372.162434 | Centralized benchmark. |
| DP, base capacity | 373.574814 | Feasible solution; not claimed globally optimal. |
| DP + BB, base capacity | 373.574814 | BB residual close to zero; 0.3781% reported MIP gap. |
| IR relaxation, base capacity | 389.475351 | Best feasible `epsilon = 1.783904`; time limit reached, no proof of minimum epsilon. |

For the BB solution followed by fixed transfers:

| Capacity multiplier | Capacity (kW) | Community cost (DKK) | Compensation required (DKK) |
| ---: | ---: | ---: | ---: |
| 0.90 | 8.451 | 392.061073 | 94.616017 |
| 1.00 | 9.390 | 373.574814 | 80.276676 |
| 1.10 | 10.329 | 361.593065 | 44.019571 |

At each tested capacity, seven members violate IR *before* fixed transfers, while the computed zero-sum transfers make all 16 members individually rational for the selected solution. This is an ex-post settlement result, not a guarantee against other individually optimal follower responses.

The strict DP+BB+IR model did not find a feasible incumbent during the tested `180 s` run. **This does not prove infeasibility.** Likewise, the IR-relaxation run stopped at its time limit, so `epsilon = 1.783904` is the best value found rather than a proven global minimum.

## 5. Reproducibility notes

- Run commands from the repository root, keeping `Q1_models` and `Q2_final` as siblings.
- Check that Q1 input files and `Q1_models/results/base_case/C/{hourly.csv,members.csv}` are present.
- The `.sol` files are saved Gurobi solutions, not Python source files. Their use depends on the corresponding model structure.
- Optimizer runtime, incumbents and gaps can vary with Gurobi version, hardware, parameters and thread scheduling.
- For each run, record the selected mode, capacity multiplier, feasible objective, best bound, optimality gap and solver status.
- The original development scripts are outside `Q2_final`; this folder is the streamlined submission version.
