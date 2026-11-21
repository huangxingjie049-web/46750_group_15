# Question 2 — Uniform Pricing for an Energy Community

**DTU 46750 · Modern Power Systems Optimization · Assignment 2 · Group 15**

This repository contains the Question 2 implementation of a bilevel uniform hourly pricing model for a 16-member energy community, its follower KKT reformulation, numerical validations and sensitivity analysis. Question 2 uses the updated Question 1 model as its common data and settlement reference.

## 1. Folder layout

Keep the two project folders at the same level:

```text
46750_group_15/
├── Assignment_2_Q1_model_new/
│   └── Q1_models/
│       ├── main.py
│       ├── base_case.py
│       ├── data/
│       └── results/base_case/{R,I,C}/
└── Q2_final_new/
    ├── README.md
    ├── Q2_1/
    │   ├── data_interface.py
    │   ├── follower.py
    │   ├── kkt_validation.py
    │   ├── run_kkt_validation.py
    │   └── verify_manual_kkt.py
    ├── Q2_2/
    │   ├── kkt_member_builder.py
    │   ├── community_response.py
    │   ├── manager_accounting.py
    │   ├── check_q2_step3_c_prices.py
    │   ├── generate_cap_outside_costs.py
    │   ├── run_manager_parameterized.py
    │   ├── run_manager_ir_relaxation.py
    │   ├── plot_q2_figures.py
    │   ├── outside_costs_RI.json
    │   ├── outside_costs_RI_cap_0p90.json
    │   ├── outside_costs_RI_cap_1p00.json
    │   ├── outside_costs_RI_cap_1p10.json
    │   └── ir_relax_RI_Q1new.sol
    └── Q2figures/
        ├── Q2_3_feeder_net_exchange_comparison.png
        ├── Q2_5_social_cost_vs_capacity.png
        └── Q2_5_compensation_vs_capacity.png
```

The relative paths in `Q2_1/data_interface.py` and the Q2 scripts assume this layout. Other `.sol` files in `Q2_2` may belong to earlier experiments and should not be used to reproduce the final results.

## 2. Model and settlement conventions

The manager chooses one uniform hourly internal price, while members individually minimize their private costs. The follower linear programs are embedded in the manager problem using KKT conditions. The manager objective is the sum of physical member costs and external community settlement costs. Internal payments are transfers within the community and are not added to social cost.

The current Question 1 settlement convention is used consistently in Question 2:

- Wholesale prices apply to total feeder imports and exports.
- The discounted import grid tariff applies **only to imports within contractual capacity**, not to excess imports.
- The discounted export tariff applies to exports.
- Imports exceeding capacity incur the DSO penalty `alpha = 10 DKK/kWh` on the excess energy.
- Feeder import and export are mutually exclusive in each hour; excess import is explicitly split from within-cap import.

The standard feeder capacity is `9.390 kW`. The sensitivity cases use `8.451 kW` (0.90), `9.390 kW` (1.00) and `10.329 kW` (1.10). The internal price domain is bounded between zero and the model's economic upper bound of `12.625 DKK/kWh`; this is a numerical modeling bound, not a DSO contractual requirement.

**Outside options:** Individual IR is compared against each member's cheaper Q1 R or I option, calculated at the same capacity multiplier. The sum of individual outside costs differs from the minimum of the aggregate R and I costs.

## 3. Environment and reproduction

Use the Python dependencies from the supplied project, including `gurobipy`, `numpy`, `scipy` and `matplotlib`, and an academic or other valid Gurobi licence. The saved final logs were obtained with **Gurobi 13.0.3**. Solver runtimes and incumbent solutions can vary by computer and parameter settings.

From the `46750_group_15` directory, use:

```powershell
python Q2_final_new/Q2_1/run_kkt_validation.py
python Q2_final_new/Q2_1/verify_manual_kkt.py
python Q2_final_new/Q2_2/check_q2_step3_c_prices.py
python Q2_final_new/Q2_2/generate_cap_outside_costs.py
python Q2_final_new/Q2_2/manager_accounting.py
```

The Q1 benchmark costs can also be verified from `Assignment_2_Q1_model_new/Q1_models/results/base_case/{R,I,C}/metrics.json`. The Q1 base-case workflow is implemented in `Q1_models/base_case.py`. The Q1 `main.py` may write outputs into a separate `results/minimum` directory; ensure that its output directories are writable before rerunning it.

### 3.1 Parameterized manager experiments

Edit the configuration near the top of `Q2_final_new/Q2_2/run_manager_parameterized.py`, then run:

```powershell
python Q2_final_new/Q2_2/run_manager_parameterized.py
```

The recorded final manager experiments used `MIP_GAP = 0.02`, `TIME_LIMIT = 180` seconds and `USE_WARM_START = False` (although most runs terminated much earlier). The seven recorded configurations are:

| Experiment | `MODE` | `CAP_MULTIPLIER` |
| --- | --- | ---: |
| Base dynamic pricing | `"DP"` | 1.00 |
| Dynamic pricing + budget balance | `"BB"` | 1.00 |
| BB dispatch + fixed transfers | `"TRANSFER"` | 1.00 |
| Strict individual rationality, base capacity | `"STRICT_IR"` | 1.00 |
| Strict individual rationality, higher capacity | `"STRICT_IR"` | 1.10 |
| BB capacity sensitivity, lower capacity | `"BB"` | 0.90 |
| BB capacity sensitivity, higher capacity | `"BB"` | 1.10 |

Each run loads `outside_costs_RI_cap_0p90.json`, `outside_costs_RI_cap_1p00.json` or `outside_costs_RI_cap_1p10.json` automatically, as appropriate. When reproducing the recorded experiments, set `TIME_LIMIT = 180` explicitly: the copy of the script in an earlier project archive had a 60-second default, while the final manager logs used 180 seconds.

**Mode definitions:** `DP` does not impose budget balance. `BB` adds exact budget balance but not individual IR. `STRICT_IR` imposes both BB and all 16 individual IR constraints. `TRANSFER` solves the same BB optimization and then computes member-specific fixed, zero-sum payments for that selected dispatch. Transfers are postprocessing, not jointly optimized with dispatch, and do not alter marginal incentives.

### 3.2 IR relaxation

The separate `run_manager_ir_relaxation.py` uses the **base capacity (1.00)** and 16 nonnegative individual IR-slack variables. It specifies lexicographic objectives: **first minimize the sum of the 16 IR slacks**, then minimize community social cost. It does **not** minimize one common maximum-violation `epsilon`, and the earlier community cost threshold constraint is not active in the final code.

```powershell
python Q2_final_new/Q2_2/run_manager_ir_relaxation.py
```

The recorded run used a 300-second time limit, 2% relative MIP-gap target and no previously saved warm start. The first-priority objective **did not converge**, and the second-priority social-cost objective was **not reached**. The recorded feasible solution therefore has no claim to lexicographic optimality. The solution is saved as `Q2_2/ir_relax_RI_Q1new.sol`. Keep that file; other historic `.sol` files are not substitutes.

`run_manager_ir_stage2.py` is a **legacy script** with an obsolete fixed slack and older tariff and outside-option conventions. It is **not part of the reported final reproducible workflow and should not be run**.

## 4. Numerical validation and price indifference

The follower KKT validation passed for all 16 members, covering primal and dual feasibility, stationarity, complementarity and strong duality. The largest primal constraint residual in the recorded tests was about `1.277e-15`; the largest primal-dual gap was about `7.105e-15 DKK`. Independent manual stationarity testing covered **2,496 equations** across 16 members, with a largest residual of `4.441e-16`.

`check_q2_step3_c_prices.py` tests the Q1 C centralized dispatch using the Q1 C dual prices. The central dispatch is individually follower-optimal for **all 16 members**. However, separately solved follower LPs can choose different, equally optimal dispatches; in the recorded selection, all 16 had different net-import schedules. The aggregate example yielded:

| Indicator | Central Q1 C dispatch | One independent follower-optimal selection |
| --- | ---: | ---: |
| Community social cost (DKK) | 372.162434 | 756.793059 |
| Peak feeder import (kW) | 9.390 | 26.000 |
| Total capacity excess (kWh) | 0 | 33.881053 |
| DSO capacity penalty (DKK) | 0 | 338.810526 |
| Hours exceeding capacity | 0 | 6 |

This demonstrates a possible coordination failure caused by **follower indifference**. The independent outcome is one valid selection, **not a certified worst-case (pessimistic) response**.

## 5. Final recorded numerical results

All monetary amounts are DKK for the 24-hour study period. Results below refer to the updated Q1 formulation; the Q2 manager values and solver statuses were checked against the final collection of execution logs (`Q2_new22_result.docx`). Q1 reference figures are also consistent with the Q1 saved `metrics.json` files.

### 5.1 Q1 benchmarks and outside options

| Q1 contract | Community cost (DKK) |
| --- | ---: |
| R | 398.241694 |
| I | 480.133426 |
| C | 372.162434 |

| Capacity multiplier | Sum of member-specific min(R, I) outside costs (DKK) |
| ---: | ---: |
| 0.90 | 394.328928 |
| 1.00 | 392.450402 |
| 1.10 | 390.835702 |

### 5.2 Manager results

| Experiment | Feasible community social cost (DKK) | Final MIP gap | IR / status |
| --- | ---: | ---: | --- |
| DP, cap 1.00 | 373.357334 | 0.3200% | 13 members violate IR; BB not imposed |
| BB, cap 1.00 | 379.353689 | 1.8957% | 8 members violate IR; BB passes |
| TRANSFER, cap 1.00 | 379.353689 | 1.8957% | 16/16 satisfy IR after zero-sum transfers |
| BB, cap 0.90 | 398.027213 | 1.5005% | 8 members violate IR before transfers |
| BB, cap 1.10 | 363.835787 | 0.6164% | 8 members violate IR before transfers |
| STRICT_IR, cap 1.00 | — | — | 180 s time limit, no feasible incumbent |
| STRICT_IR, cap 1.10 | 361.593065 | 0.0000% | 16/16 satisfy IR and BB |

`STRICT_IR` at cap 1.00 terminated with `Status = 9` and `SolCount = 0` after about 180 seconds. **This does not prove mathematical infeasibility.** In contrast, the cap 1.10 strict-IR model found a feasible solution with the same cost as the certified lower bound (`361.593065 DKK`), hence reported zero MIP gap. Because the cap 1.10 ordinary BB model has the same lower bound and a larger feasible incumbent, the strict-IR solution also attains the BB model's global lower bound within numerical tolerance. The feasible BB solutions in the table are not all claimed to be globally optimal.

### 5.3 Capacity sensitivity with the BB mode

| Capacity multiplier | Feeder capacity (kW) | BB feasible social cost (DKK) | IR deficit / compensation (DKK) | Members violating IR | MIP gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 8.451 | 398.027213 | 88.784323 | 8 | 1.5005% |
| 1.00 | 9.390 | 379.353689 | 57.420845 | 8 | 1.8957% |
| 1.10 | 10.329 | 363.835787 | 48.714523 | 8 | 0.6164% |

The reported feasible BB costs and individual IR deficits decrease as capacity increases. The compensation amounts describe the **selected BB incumbents** and are not proven minimum compensation levels across all feasible solutions. At cap 0.90, the selected BB solution has only `85.086039 DKK` of positive member gains against `88.784323 DKK` of IR deficits, so its specific dispatch cannot be made fully individually rational using only zero-sum transfers. This does not establish infeasibility for every possible dispatch.

### 5.4 Fixed transfers at base capacity

The cap 1.00 BB dispatch has `57.420845 DKK` of member IR deficits and `70.517558 DKK` of positive member gains. The available gains cover the deficits, leaving `13.096712 DKK` in aggregate member gains. The recorded postprocessing found net transfers of approximately `-3.553e-15 DKK`, with **16/16 members satisfying IR after transfers**. These transfers preserve the selected dispatch and exact BB settlement. This is feasibility for the selected optimistic follower response, not robustness against every other follower-optimal response.

### 5.5 IR relaxation: feasible solution, not a proven optimum

The separately recorded 300-second IR-relaxation run (reviewed prior to the final Word compilation) yielded a feasible sum of individual IR slacks of **11.04281109 DKK** and a community social cost of **382.63810589 DKK**. Seven members violated IR; total model slack matched actual violations within numerical tolerance. The solver stopped at `TIME_LIMIT` with a **100% MIP gap for the first-priority slack objective** and a zero lower bound. It did not progress to minimizing social cost as a second objective. Thus `11.04281109 DKK` is a feasible upper bound on the minimum total IR slack, **not a proven minimum**. For a complete self-contained results record, add this run's full console output to the final Word compilation.

### 5.6 Fixed-price accounting diagnostic

`manager_accounting.py` checks a fixed-price independent follower response, **not an optimized manager solution**. It reported a community cost of `1700.008913 DKK`, including `1413.921053 DKK` of DSO excess-capacity penalties from `141.392105 kWh` of exceedance across 11 hours. The manager's fixed-price budget deficit was `1433.139934 DKK`. The independent accounting checks passed. These diagnostic values must not be mistaken for the optimized DP/BB social costs.

## 6. Figures

Run:

```powershell
python Q2_final_new/Q2_2/plot_q2_figures.py
```

The script uses saved Q1 C CSV data and hard-coded values from the recorded BB sensitivity runs to generate:

1. `Q2_3_feeder_net_exchange_comparison.png` — centralized versus one independently selected follower-optimal net feeder exchange.
2. `Q2_5_social_cost_vs_capacity.png` — social cost for the three recorded BB incumbent solutions (`398.027213`, `379.353689`, `363.835787 DKK`).
3. `Q2_5_compensation_vs_capacity.png` — aggregate IR deficits of the corresponding BB incumbents (`88.784323`, `57.420845`, `48.714523 DKK`).

**Before committing to Git**, make sure the local copy of `plot_q2_figures.py` contains these latest values. The previously shared archive still had the old arrays, even though the updated images were subsequently shown. The first figure can remain visually unchanged because its source is the Q1 C price-response comparison, not the BB sensitivity arrays.

## 7. Limitations and reproducibility notes

- The bilevel KKT reformulation implements an **optimistic selection** from possibly multiple follower-optimal responses. The Q1 C example highlights the importance of this qualification.
- Gurobi's message `Optimal solution found (tolerance ...)` means the configured stopping tolerance was met. It does not mean the MIP gap is zero.
- The base-capacity strict-IR time-limit result is **inconclusive on feasibility**. Do not report it as mathematically infeasible.
- The first-priority IR-relaxation run is **not** a proven minimum and its second-priority objective was not completed.
- Fixed post-optimization transfers apply only to the selected BB dispatch; they do not guarantee IR under every alternative follower-optimal selection.
- Keep the updated Q1 and Q2 folders adjacent. Historic `.sol` files, older plotting arrays and the legacy Stage 2 script must not be used in the final reported workflow.
- Before reproduction, record `MODE`, `CAP_MULTIPLIER`, Gurobi parameters, solver status, best feasible objective, best bound, MIP gap and validation residuals. The reported values are taken from the saved execution records rather than new optimization runs performed for this README.
