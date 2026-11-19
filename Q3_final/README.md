# Question 3 — Steering consumers whose response is uncertain

**DTU 46750 · Assignment 2 · Group 15**

`Q3_final/` sits at the repository root **next to `Q1_models/` and `Q2_final/`**. It reads their code and outputs and
never modifies them:

| Used from | What |
|---|---|
| `Q1_models/main.py` → `load_data` | the 16 members of the base case (d1) |
| `Q1_models/data/` | the supplied JSON inputs (identical to the starter package) |
| `Q1_models/results/base_case/C/` | C dual prices (`hourly.csv`) and C cost (`metrics.json`) |
| `Q2_final/Q2_2/dp_bb_start.sol` | the price vector reported for DP / DP+BB in Section 2 |
| `Q2_final/Q2_2/kkt_member_builder.py` | `add_member_kkt`: one member's KKT system (SOS1, strong-duality payment) |
| `Q2_final/Q2_1/data_interface.py` | `convert_member`: Q1 member → KKT builder input |

**Folder detection (13 Nov).** The updated Q1 and Q2 are used automatically when they exist:
`Assignment_2_Q1_model_new/Q1_models` (else `Q1_models`) and `Q2_final_new` (else `Q2_final`); `validate` and
`sdp-validate` print the folders in use. Override with the environment variables `Q3_Q1_DIR` / `Q3_Q2_DIR`.
Contract accounting follows the updated Q1: the discounted tariff applies to imports within the cap; imports above
the cap pay the day-ahead price plus the penalty `alpha` (in the models: `a_d` import + (alpha - (1-beta) tau_imp) excess).

## Layout

```text
Q3_final/
├── run_q3.py                 one command per experiment (see below)
├── q3/
│   ├── scenario_generation.py   starter-package module, copied UNCHANGED
│   ├── scenarios.py             draw / check / save the scenario sets (seeds below)
│   ├── q1_bridge.py             scenario -> Q1 member data (ref, E_min, hourly bounds recomputed)
│   ├── follower_lp.py           members' best responses at fixed prices (HiGHS); selection rules; central dispatch
│   ├── evaluate.py              settlement at the connection point, DSO metrics, regret vs perfect information
│   ├── sdp.py                   SDP^x: ONE parametrized Gurobi model (risk-neutral / CVaR, penalty / chance cap,
│   │                            budget balance in expectation / per scenario / off) + warm start + KKT residuals
│   ├── settlement.py            individual rationality in expectation and zero-sum fixed transfers
│   ├── figures.py               report figures, read from results/ only
│   ├── prices.py                C and DP price vectors (read only)
│   └── paths.py
├── data/preference_uncertainty_params.json   starter file, unchanged
├── scenarios/                 generated scenario sets (part of the deliverable)
├── results/                   outputs of every experiment; results/q3_summary.csv = all models in one table
├── figures/                   report figures (PDF + PNG)
└── tests/test_q3.py
```


## Reproduce everything with one command

`Q3_final/run_all.ps1` runs every experiment of the table below in the right order (about 9-10 h on a 16-thread laptop),
one log per step in `Q3_final/logs/run_all/`, and stops at the first failing step. From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1 -DryRun   # check paths, print the plan
powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1           # run everything
powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1 -From 12  # resume at step 12
```

Warm-start candidates that do not exist yet are skipped, so the script also runs on a fresh checkout. The N = 5 and N = 10 models
start from `--warm pool:in5` / `pool:in10`: every price vector already found for that scenario set (all runs in
`results/sdp` and, for N = 5, the archived vectors in `data/n5_vectors`) is evaluated at fixed prices, and the best
complete solution starts the MILP (pooled multi-start). A variant therefore never ends above a solution that is
feasible for it (e.g. SDP^E is never worse than an SDP^CC solution), and re-running the script never makes a result
worse. The script is the authoritative list of commands; the table rows below show the individual experiments.

`data/n5_vectors/{A,B,C,D}.json` archive four N = 5 price vectors returned by different runs (12-14 Nov): A, B and C
have the same in-sample cost in every scenario (370.626 DKK) but very different out-of-sample regret; D (370.385 DKK)
is the best solution found. `run_q3.py select --oos` evaluates them in sample, on the validation scenarios (the
N = 10 set without the N = 5 set) and out of sample (Section 3.4). The steps below can
also be run one by one.

## Where each number of the report comes from

| Report (Section 3) | Content | File (all written by the commands below) |
|---|---|---|
| 3.1, Table `tab:q3_baseline`, Figure `fig:q3_baseline` | DP and DP^BB out of sample | `results/sdp/SDP_E_expected_bb-{none,expectation}/`, `figures/q3_baseline_imports.pdf` |
| 3.2, Table `tab:q3_sdp`, Figures `fig:q3_prices`, `fig:q3_regret` | SDP^E, N = 5 and 10 | `results/q3_summary.csv`, `figures/q3_prices.pdf`, `figures/q3_regret_ecdf.pdf` |
| 3.3, Table `tab:q3_risk` (and appendix `tab:q3_risk_sweep`) | SDP^CVaR, SDP^CC, penalty 10 and 3 | `results/q3_summary.csv`, `results/sdp/<run>/` |
| 3.4, Figure `fig:q3_n` | accuracy against performance over N | `results/nstudy/summary.csv`, `figures/q3_n_timing.pdf` |
| 3.4, Table `tab:q3_n5` | near-optimal N = 5 vectors x^(1)-x^(4) = `data/n5_vectors/{A,B,C,D}.json` | `results/selection_in5_on_in10.csv` |
| 3.5, Table `tab:q3_deviation` | regret grouped by the deviation of the daily reference energy | `results/regret_by_deviation.csv` |
| Appendix, `tab:q3_indifference`, `tab:q3_size` | selection rules, model size | `results/sdp/<run>/eval_oos/metrics_{rule}.csv`, `results/sdp/<run>/solve.json` |

`figures/q3_n_study.pdf`, `q3_penalty.pdf` and `q3_prices_cap_only.pdf` are produced as well but not used in the
report. The results in `results/` are those of the final run (15 Nov, pooled multi-start; N study 16 Nov).

## Experiments — one command each (run from the repository root, in this order)

The order matters only for the warm starts: later runs start from the prices of earlier ones (`--warm`), which saves
solution time but does not change the model. Times are for a 16-thread laptop with Gurobi 13.

| # | Experiment (report section) | Command | Time |
|---|---|---|---|
| E0 | Scenario sets and checks (3.1, 3.4) | `python Q3_final/run_q3.py scenarios` | s |
| E0 | Evaluator validation at θ = 1 (3.1) | `python Q3_final/run_q3.py validate` | s |
| E0 | SDP validation: DP = C, identical copies (3.1) | `python Q3_final/run_q3.py sdp-validate` | s |
| E1 | Q1/Q2 price vectors out of sample (3.1, appendix) | `python Q3_final/run_q3.py baseline` | 2 min |
| E2 | Deterministic baseline DP, cap only (3.1) | `python Q3_final/run_q3.py sdp --model E --set expected --bb none` | 1 min |
| E2 | Deterministic baseline DP^BB (3.1) | `python Q3_final/run_q3.py sdp --model E --set expected --time-limit 600` | 15 min |
| E3 | SDP^E, N = 5, cap only (3.2, 3.4) | `python Q3_final/run_q3.py sdp --model E --set in5 --bb none` | 5 min |
| E3 | SDP^E, N = 10, cap only (3.4) | `python Q3_final/run_q3.py sdp --model E --set in10 --bb none --warm Q3_final/results/sdp/SDP_E_in5_bb-none/prices.json` | 5 min |
| E3 | SDP^E, N = 5, BB in expectation — base (3.2) | `python Q3_final/run_q3.py sdp --model E --set in5 --time-limit 1800 --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json` | 35 min |
| E3 | SDP^E, N = 10, BB in expectation (3.2, 3.4); best start = SDP^CC ε = 0, N = 10 | `python Q3_final/run_q3.py sdp --model E --set in10 --time-limit 900 --warm Q3_final/results/sdp/SDP_CC_eps0_in10_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in10_bb-expectation/prices.json` | 25 min |
| E4 | SDP^CVaR, α = 0.9, w = 0.5 — base (3.3) | `python Q3_final/run_q3.py sdp --model CVaR --set in5 --time-limit 1800` | 30 min |
| E4 | SDP^CVaR sweep over w at α = 0.9; w = 0.01 is SDP^E with a CVaR tie-break (3.3, 3.4) | `python Q3_final/run_q3.py sweep --model CVaR --param weight --values 0.01,0.25,0.75,1 --time-limit 600 --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json` | 50 min |
| E4 | SDP^CVaR sweep over α at w = 0.5 (3.4) | `python Q3_final/run_q3.py sweep --model CVaR --param alpha --values 0.4,0.6,0.8 --weight 0.5 --time-limit 600 --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json` | 40 min |
| E5 | SDP^CC sweep over ε (3.3, 3.4) | `python Q3_final/run_q3.py sweep --model CC --param epsilon --values 0,0.2,0.4,0.6 --time-limit 600 --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json` | 50 min |
| E4 | SDP^CVaR, N = 10 (3.4) | `python Q3_final/run_q3.py sdp --model CVaR --set in10 --time-limit 900 --warm Q3_final/results/sdp/SDP_CC_eps0_in10_bb-expectation/prices.json,Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in10_bb-expectation/prices.json` | 25 min |
| E5 | SDP^CC sweep over ε at N = 10 (3.3, 3.4) | `python Q3_final/run_q3.py sweep --model CC --param epsilon --values 0,0.3,0.6 --set in10 --time-limit 1800 --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json` | 2 h |
| E5 | Validation-based choice between the in-sample-equivalent N = 5 price vectors (3.4) | `python Q3_final/run_q3.py select --candidates Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_CC_eps0_in5_bb-expectation/prices.json` | 1 min |
| E7 | Penalty α^DSO = 3 DKK/kWh (regime where exceedance is worth paying), SDP^E, N = 5 (3.4) | `python Q3_final/run_q3.py sdp --model E --penalty 3 --time-limit 900 --rules optimistic --warm Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_E_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_CC_eps0_in5_bb-expectation/prices.json` | 20 min |
| E7 | idem, SDP^CVaR w = 0.5 | `python Q3_final/run_q3.py sdp --model CVaR --penalty 3 --time-limit 900 --rules optimistic --warm Q3_final/results/sdp/SDP_E_pen3_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json` | 20 min |
| E7 | idem, SDP^CC sweep over ε | `python Q3_final/run_q3.py sweep --model CC --param epsilon --values 0,0.2,0.4,0.6 --penalty 3 --time-limit 900 --warm Q3_final/results/sdp/SDP_E_pen3_in5_bb-expectation/prices.json,Q3_final/results/sdp/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json` | 75 min |
| E7 | Re-evaluate SDP^CC runs with the cap-first tie-break of the optimistic rule (already applied by `sdp`/`sweep` since 13 Nov) | `python Q3_final/run_q3.py reeval --runs <run folder names>` | 2 min per run |
| E8 | Accuracy against performance over nested N = 1, 2, 5, 10, 20: SDP^E with BB, the same start (DP^BB prices), time limit (1800 s) and gap for every N, so that solution times are comparable; writes `results/nstudy/` (3.4) | `python Q3_final/run_q3.py nstudy` (`--dry-run`: sets and checks only) | 1-2 h |
| E6 | Summary table of all models | `python Q3_final/run_q3.py summary` | s |
| E6 | All report figures, and the regret grouped by demand deviation (`results/regret_by_deviation.csv`, 3.5) | `python Q3_final/run_q3.py figures` | s |
| T | Tests | `python -m pytest Q3_final/tests -q` | 15 s |

`run_q3.py <command> --help` lists every option (time limit, MIP gap, threads, warm-start candidates, evaluation
rules, budget-balance penalty). With N = 5 equiprobable scenarios, CVaR at α ≥ 0.8 is the worst scenario and the
chance constraint can only take ε ∈ {0, 0.2, 0.4, …}.

Each `sdp` run writes `results/sdp/<model>_<set>_bb-<bb>/`: `solve.json` (configuration, status, objective, bound, gap,
runtime, model size, warm start, KKT residuals), `prices.json`, `in_sample.csv` (per scenario, as targeted by the
model), `eval_<set>/` (in-sample re-evaluation of the prices with the optimistic rule), `eval_oos/` (100 out-of-sample
scenarios, three selection rules) and `ir_oos.csv` (individual rationality in expectation and transfers).

Python packages: `numpy`, `pandas`, `scipy`, `matplotlib` (evaluation and figures, HiGHS through scipy) and `gurobipy`
with an academic licence (pricing models; the restricted licence is too small).

## Pricing model and solution strategy

`q3/sdp.py` builds one Gurobi model per configuration: one shared price vector, N copies of the 16 KKT systems of Q2,
scenario-indexed community balance, cap and costs, and the switches risk (neutral / CVaR), cap (penalty / chance),
budget balance (expectation / scenario / none). Budget balance is elastic (residual penalised at 100 DKK/DKK, reported
as `bb_residual`; 0 in every reported run).

Warm start in two phases: (1) for every candidate price vector, the prices are fixed and the model is solved with
tight tolerances (the KKT systems only select the optimistic responses); (2) the prices are freed and the best complete
phase-1 solution is passed as MIP start. Several candidates (multi-start) protect against poor local incumbents: the
first SDP^E run with N = 5 stopped at 371.14 DKK, and the CVaR run later exposed a feasible SDP^E solution with an
expected cost of 370.63 DKK.

## Scenarios and seeds

Base study of the assignment: each member's reference profile is `rho_t ** theta`, `theta ~ Uniform[0.8, 1.2]`
(scaled Beta, kappa = 2, mean 1), independent per member, aligned merging; the daily minimum and the hourly bounds
follow the profile; `cL` is fixed.

| Set | Size | Seed | Note |
|---|---|---|---|
| `in5` | 5 | 1 | first 5 scenarios of `in10` (nested; checked) |
| `in10` | 10 | 1 | |
| `oos` | 100 | 1000 | same set for every model (common random numbers); disjoint from in-sample (checked) |
| `expected` | 1 | – | theta = 1: the deterministic baseline |
| `in1`, `in2`, `in20` | 1, 2, 20 | 1 | N study only (`nstudy`): first 1, 2 scenarios of `in20`; `in5` and `in10` are its first 5 and 10 (nested; checked in `results/nstudy/scenario_checks.json`) |

Checks (`results/scenario_checks.json`): theta within bounds, sample mean vs set mean, each scenario's daily minimum
equals its reference energy, stored indices recover the independent draws, probabilities sum to one.

## Evaluation of a price vector

For each scenario the members respond to the fixed prices; the manager settles exactly as in
`Q2_final/Q2_2/manager_accounting.py`. Because the best response of a linear-disutility member is a set, three
selection rules are reported:

* `optimistic` — the cheapest profile for the community among all members' optimal responses (LP; the solution
  concept of the bilevel models; main metric);
* `independent` — each member solved alone (an arbitrary vertex, as in Section 2.3 of the report);
* `pessimistic` — the most expensive one (MILP, 48 binaries).

A member's cost in a joint selection may exceed its own optimum by at most `value_tol` (1e-6 DKK, relaxed to 1e-5 or
1e-4 when HiGHS declares the joint MILP infeasible for tolerance reasons; the value used is stored per scenario).

Metrics per scenario: community cost `J` and components, `J_C` (central dispatch with perfect information on the same
scenario) and `regret = J - J_C`, exceedance (kWh), hours above the cap, peak import, penalty, manager budget surplus.
Summaries: mean, quantiles, upper-tail CVaR at 90 %, share of scenarios with any exceedance.

## Model size

Every (member, scenario) pair adds one Q2 KKT system (linear in N and in the number of members):

| N | variables | constraints | SOS1 pairs |
|---|---|---|---|
| 1 | 8,016 | 4,488 | 3,480 |
| 5 | 39,984 | 22,440 | 17,400 |
| 10 | 79,944 | 44,880 | 34,800 |

Budget balance adds 1 constraint (2 residual variables), CVaR N + 1 continuous variables and N constraints, the chance
constraint N binaries and 24 N + 1 constraints.

## Validation

* The bridge reproduces the Q1 members exactly at theta = 1.
* Central dispatch on the expected scenario: `J = 372.162434` = Q1 C.
* Optimistic response at the C dual prices: `J = 372.162434` = Q1 C.
* SDP with N = 1, theta = 1, cap only (`sdp-validate`): 372.162434 = C, proven optimal; three identical copies: same.
* After every solve, `solve.json` reports the largest complementarity product, the largest strong-duality gap and the
  largest suboptimality of any (member, scenario) against its own LP solved independently with HiGHS.
* In-sample re-evaluation: the prices are re-evaluated on the in-sample set with the optimistic rule; the value can
  only be lower than or equal to the model's (printed after each solve).
* IR at the reported DP prices, theta = 1: 7 members worse off than R, compensation 80.277 DKK, gains 104.944 DKK
  (= Q2, Section 2.4).
