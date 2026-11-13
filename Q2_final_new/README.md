# Question 2 — Uniform Pricing for an Energy Community

**DTU 46750 · Optimization in Modern Power Systems · Assignment 2 · Group 15**

This folder contains the final Python implementation, validation scripts and figures for Question 2. It studies whether a common hourly electricity price can coordinate the decisions of 16 community members while respecting the manager's contractual requirements.

## 1. Repository layout

`Q2_final` is intended to sit **next to** `Q1_models` at the repository root:

```text

46750_group_15/
├── Assignment_2_Q1_model_new/
│   └── Q1_models/
│       ├── main.py
│       ├── data/
│       └── results/
│           └── base_case/
│               ├── R/
│               ├── I/
│               └── C/
│                   ├── hourly.csv
│                   └── members.csv
│
├── Q2_final_new/
│   ├── README.md
│   ├── Q2_1/
│   │   ├── data_interface.py
│   │   ├── follower.py
│   │   ├── kkt_validation.py
│   │   ├── run_kkt_validation.py
│   │   └── verify_manual_kkt.py
│   │
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
│   │   ├── ir_relax_cost_limited.sol
│   │   └── ir_relax_q1new.sol
│   │
│   └── Q2figures/
│       ├── Q2_3_feeder_net_exchange_comparison.png
│       ├── Q2_5_social_cost_vs_capacity.png
│       └── Q2_5_compensation_vs_capacity.png

```


The Q2 implementation uses the updated Q1 data and results from `Assignment_2_Q1_model_new/Q1_models/`. The `Q2_final_new` and `Assignment_2_Q1_model_new` folders must remain at the same directory level, as shown above.

The data interface loads the updated Q1 inputs, while the Q1 C price-response validation reads `Assignment_2_Q1_model_new/Q1_models/results/base_case/C/hourly.csv` and `members.csv`. These paths are defined relative to the Python script locations.


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
python Q2_final_new/Q2_1/run_kkt_validation.py
python Q2_final_new/Q2_1/verify_manual_kkt.py
```

The independent stationarity test covers 16 members and 2,496 equations. 
The follower KKT validation covers all 16 members. All primal feasibility, variable bound, dual feasibility, stationarity, complementarity and strong duality checks passed. The largest primal constraint residual was `1.277e-15`, and the largest primal-dual gap was `7.105e-15`, both within numerical tolerance.



### 3.2 Run the manager model

Open `Q2_final_new/Q2_2/run_manager_parameterized.py` and set the configuration variables near the beginning of the script.

The available modes are:

- `MODE = "DP"`: Dynamic pricing without budget balance.
- `MODE = "BB"`: Dynamic pricing with strict budget balance.
- `MODE = "STRICT_IR"`: Dynamic pricing with budget balance and individual rationality imposed directly for all members.
- `MODE = "TRANSFER"`: Solve the BB model and then calculate member-specific, zero-sum fixed transfers.

The base-case configuration uses:

```python
CAP_MULTIPLIER = 1.00
TIME_LIMIT = 180
MIP_GAP = 0.02
USE_WARM_START = False
```

The base contractual capacity is `9.39 kW`. For the capacity sensitivity analysis, `CAP_MULTIPLIER` is set to `0.90`, `1.00` and `1.10`, corresponding to capacity limits of `8.451`, `9.390` and `10.329 kW`.

The updated Q1 formulation separates within-cap imports and excess imports. Wholesale energy costs are calculated for total imports and exports, while discounted import grid tariffs apply only to within-cap imports. Excess imports are charged the DSO penalty.

The saved `dp_bb_start.sol` belongs to the previous model formulation and must not be used as a warm start for the updated experiments.

Run the model from the repository root:

```powershell
python Q2_final_new/Q2_2/run_manager_parameterized.py
```

**Important:** The `TRANSFER` mode does not jointly optimize dispatch and transfer payments. It first solves the BB model and then calculates fixed transfers for the selected dispatch. These transfers do not change the marginal price incentives.

### 3.3 Test price indifference and feeder response

Run:

```powershell
python Q2_final_new/Q2_2/check_q2_step3_c_prices.py
```

This script reads the updated Q1 C results and verifies whether the centralized member dispatches are individually optimal at the corresponding hourly dual prices.

For the updated Q1 case, all 16 centralized member dispatches are individually optimal at the Q1 C dual prices. However, all 16 independently solved follower LPs select different net-import schedules while achieving the same minimum private costs.

The resulting aggregate feeder response differs from the centralized dispatch:

- Centralized Q1 C social cost: `372.162434 DKK`.
- Independent-response social cost: `756.793059 DKK`.
- Centralized peak import: `9.390 kW`.
- Independent-response peak import: `26.000 kW`.
- Independent-response total capacity excess: `33.881053 kWh`.
- Independent-response capacity penalty: `338.810526 DKK`.
- Hours exceeding the contractual capacity: `6`.

These results demonstrate that supporting the centralized dispatch with uniform prices does not guarantee that independently selected optimal responses reproduce the same feeder schedule.

The reported independent response is one possible selection among multiple follower-optimal dispatches. It is not a proven globally worst-case response.

### 3.4 Run the IR relaxation experiment

Run:

```powershell
python Q2_final_new/Q2_2/run_manager_ir_relaxation.py
```

The IR relaxation model minimizes a common nonnegative relaxation variable `epsilon`, which represents the maximum permitted individual IR violation.

The experiment also imposes strict budget balance and a community social-cost threshold equal to the aggregate Q1 R reference cost.

The updated experiment uses a `120 s` time limit and does not use the previous Q1 warm start.

The best feasible solution found in the updated experiment has:

- Best feasible epsilon: `10.114500 DKK`.
- Best bound on epsilon: `0.000000 DKK`.
- MIP gap: `100%`.
- Community social cost: `393.344934 DKK`.
- Budget balance residual: approximately zero.

The solver stopped at the time limit with a feasible solution. Therefore, the reported epsilon is the best value found during this run, not a proven global minimum.

The updated solution is saved as:

`Q2_final_new/Q2_2/ir_relax_q1new.sol`

The previous `ir_relax_cost_limited.sol` belongs to the earlier model formulation and should not be used to reproduce the updated results.

### 3.5 Regenerate the report figures

Run:

```powershell
python Q2_final_new/Q2_2/plot_q2_figures.py
```

The script generates three PNG figures in `Q2_final_new/Q2figures/`:

- `Q2_3_feeder_net_exchange_comparison.png`
- `Q2_5_social_cost_vs_capacity.png`
- `Q2_5_compensation_vs_capacity.png`

The Q2.3 figure compares the updated Q1 C centralized feeder schedule with one independently selected follower-optimal response.

The Q2.5 figures use the verified updated Manager results for capacity multipliers `0.90`, `1.00` and `1.10`.

The plotting script uses stored numerical values and Q1 C CSV data. It does not rerun the optimization experiments. Existing PNG files with the same names are overwritten.

## 4. Selected reported results

The following values correspond to the updated Q1 formulation and the Q2 experiments performed with this implementation. All monetary values are in DKK.

### 4.1 Reference and manager experiments

| Experiment | Community social cost (DKK) | Solver result |
| --- | ---: | --- |
| Q1 R reference | 398.241694 | No-contract reference |
| Q1 C centralized dispatch | 372.162434 | Centralized benchmark |
| DP, base capacity | 373.602476 | Feasible solution, 0.3854% MIP gap |
| DP + BB, base capacity | 374.578640 | Feasible solution, 0.6450% MIP gap |
| IR relaxation, base capacity | 393.344934 | Best feasible epsilon = 10.114500 |
| STRICT_IR, base capacity | Not available | No feasible incumbent within 180 s |
| TRANSFER, base capacity | 374.578640 | BB dispatch followed by fixed transfers |

The DP model does not impose budget balance. Its reported budget surplus is therefore not required to be zero.

The BB and TRANSFER experiments satisfy the budget balance checks within numerical tolerance.

The STRICT_IR experiment reached the 180-second time limit without finding a feasible incumbent. This result does not prove that the strict IR model is infeasible.

The IR relaxation experiment also reached its time limit without proving the minimum possible epsilon.

### 4.2 Capacity sensitivity analysis

The BB model was solved with three capacity multipliers.

| Capacity multiplier | Capacity (kW) | Social cost (DKK) | Compensation required (DKK) | MIP gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.90 | 8.451 | 392.128750 | 94.628706 | 0.0188% |
| 1.00 | 9.390 | 374.578640 | 81.619893 | 0.6450% |
| 1.10 | 10.329 | 361.593065 | 45.174571 | 0.0000% |

The reported social costs are the best feasible values found by Gurobi. The solution at multiplier `1.10` has a zero reported optimality gap, while the other cases have small nonzero gaps.

The results show lower community social costs and lower compensation requirements as the contractual import capacity increases.

The compensation values represent the total IR deficits associated with the selected BB dispatches. They are not the globally minimum compensation requirements over all feasible dispatches.

At all three tested capacities, seven members violate individual rationality before transfers.

### 4.3 Fixed transfers and individual rationality

For the base-capacity TRANSFER experiment, the BB dispatch has:

- Community social cost: `374.578640 DKK`.
- Total compensation required: `81.619893 DKK`.
- Available positive member gains: `105.282947 DKK`.
- Aggregate member saving relative to R: `23.663054 DKK`.
- Members violating IR before transfers: `7`.
- Members satisfying IR after transfers: `16/16`.
- Net transfers: approximately zero.

The available positive member gains exceed the required compensation. Therefore, a zero-sum transfer allocation can make all 16 members individually rational for the selected BB dispatch.

The transfers are calculated after optimization and do not change the physical dispatch or marginal price incentives. The result demonstrates feasibility of the reported settlement for the selected response, not robustness against every possible follower-optimal response.

### 4.4 Numerical validation

The updated experiments include checks for:

- Budget balance residual.
- Linearized and directly calculated internal payments.
- Individual follower optimality.
- Feeder capacity excess consistency.
- Individual-rationality margins.
- Zero-sum transfers and post-transfer IR, where applicable.

The tested DP, BB and TRANSFER solutions passed the implemented numerical validation checks.

The reported solver gaps and time limits should be retained when interpreting the results. A feasible solution is not necessarily a proven global optimum.

## 5. Reproducibility notes

- Keep `Q2_final_new` and `Assignment_2_Q1_model_new` at the same directory level.
- The updated Q1 data must be available under `Assignment_2_Q1_model_new/Q1_models/data/`.
- The Q1 C price-response experiment requires `Assignment_2_Q1_model_new/Q1_models/results/base_case/C/hourly.csv` and `members.csv`.
- Run the listed Python commands from the repository root. Alternatively, run the individual scripts directly using PyCharm with the same project directory structure.
- The Manager model uses `USE_WARM_START = False` for the updated experiments.
- The older `dp_bb_start.sol` and `ir_relax_cost_limited.sol` files belong to the previous model formulation. They are retained only as historical solution files and should not be treated as updated Q1 results.
- The updated IR relaxation solution is stored in `ir_relax_q1new.sol`.
- Gurobi solution files are optimizer outputs, not Python source files. Their compatibility depends on the corresponding model formulation.
- The plotting script contains numerical values from the reported experiments and does not automatically rerun the Manager model.
- Runtime, incumbent solutions and reported MIP gaps may vary with solver version, hardware, parameters and solver execution.
- For reproducibility, record the selected mode, capacity multiplier, solver status, best feasible objective, best bound and MIP gap.
- The original Q2 development files are stored separately. `Q2_final_new` contains the updated implementation for the revised Q1 model.
