"""Question 3 - one command per experiment (run from the repository root).

    python Q3_final/run_q3.py scenarios     # draw, check and save the in-sample (N = 5, 10) and out-of-sample (100) sets
    python Q3_final/run_q3.py validate      # theta = 1 checks: the evaluator reproduces the Q1 C cost at the C prices
    python Q3_final/run_q3.py evaluate --prices DP --set oos --rules optimistic
                                            # evaluate one price vector on one scenario set
    python Q3_final/run_q3.py baseline      # experiment E1: C and DP prices on the 100 out-of-sample scenarios,
                                            # under the three selection rules (Section 3.1 of the report)
    python Q3_final/run_q3.py sdp-validate  # SDP with one scenario reproduces DP = C (needs Gurobi)
    python Q3_final/run_q3.py sdp --model E --set in5 --bb expectation
                                            # solve SDP^E on 5 in-sample scenarios, then evaluate out of sample
    python Q3_final/run_q3.py nstudy        # accuracy against performance over N = 1, 2, 5, 10, 20 (nested), same
                                            # start, time limit and gap for every N (Section 3.4 of the report)

Outputs are written under Q3_final/results/ and Q3_final/scenarios/. No Q1 or Q2 file is modified.
Evaluation needs numpy, pandas and scipy only (HiGHS); the stochastic pricing models (next step) need gurobipy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q3 import evaluate as ev                      # noqa: E402
from q3 import prices as pr                        # noqa: E402
from q3 import q1_bridge as qb                     # noqa: E402
from q3 import scenarios as sc                     # noqa: E402
from q3.paths import Q1_C_DIR, RESULTS_DIR, SCENARIO_DIR  # noqa: E402

RULES = ("independent", "optimistic", "pessimistic")


def scenario_set(name: str) -> list[dict]:
    """Named scenario sets: expected (theta = 1), in5, in10 (seed 1, nested), oos (100, seed 1000)."""
    if name == "expected":
        return [sc.expected_scenario()]
    if name in ("in5", "in10"):
        return sc.draw(int(name[2:]), sc.SEED_IN_SAMPLE)
    if name == "oos":
        return sc.draw(sc.OUT_OF_SAMPLE_SIZE, sc.SEED_OUT_OF_SAMPLE)
    raise ValueError(f"unknown scenario set {name!r}")


def cmd_scenarios(_args) -> None:
    sets = {"in5": (5, sc.SEED_IN_SAMPLE), "in10": (10, sc.SEED_IN_SAMPLE), "oos": (100, sc.SEED_OUT_OF_SAMPLE)}
    drawn, checks = {}, {}
    for name, (n, seed) in sets.items():
        drawn[name] = sc.draw(n, seed)
        checks[name] = sc.check_set(drawn[name], seed)
        sc.save(sc.to_record(drawn[name], name, seed, ("profile",)), SCENARIO_DIR / f"{name}.json")
    checks["in5_nested_in_in10"] = sc.check_nested(drawn["in5"], drawn["in10"])
    ids = list(drawn["oos"][0]["theta"]["profile"])
    overlap = any(abs(a["theta"]["profile"][c] - b["theta"]["profile"][c]) < 1e-12
                  for a in drawn["in10"] for b in drawn["oos"] for c in ids)
    checks["oos_disjoint_from_in_sample"] = not overlap
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scenario_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    for name, c in checks.items():
        print(name, c)
    flags = [v for c in checks.values() for k, v in (c.items() if isinstance(c, dict) else [("", c)])
             if isinstance(v, bool)]
    assert all(flags), "a scenario check failed"
    print(f"\nScenario sets saved in {SCENARIO_DIR}; all checks passed.")


def cmd_validate(_args) -> dict:
    """theta = 1: (i) the bridge reproduces the Q1 members; (ii) at the C dual prices the optimistic selection
    reproduces the C cost of Q1 (the C dispatch is a best response, and no profile can be cheaper than C)."""
    from q3.paths import describe
    print(f"using {describe()}", flush=True)
    base = qb.load_base()
    d = qb.scenario_data(base, sc.expected_scenario())
    same = all(np.allclose(a.ref, b.ref) and np.allclose(a.lower, b.lower) and np.allclose(a.upper, b.upper)
               and abs(a.minimum - b.minimum) < 1e-12 for a, b in zip(base.members, d.members))
    j_c = json.loads((Q1_C_DIR / "metrics.json").read_text(encoding="utf-8"))["J_DKK"]
    df, _, _ = ev.evaluate(pr.c_dual_prices(), [d], rule="optimistic")
    out = {"bridge_reproduces_Q1_members": bool(same), "J_C_Q1": j_c, "J_optimistic_at_C_prices": float(df.J[0]),
           "difference": float(df.J[0] - j_c)}
    out["passed"] = bool(same and abs(out["difference"]) < 1e-4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "validation_theta1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    assert out["passed"], "theta = 1 validation failed"
    return out


def run_evaluation(price_name: str, set_name: str, rules, contract_kw=None, label: str | None = None,
                   folder: Path | None = None) -> pd.DataFrame:
    default_label, x = pr.load_prices(price_name)
    label = label or default_label
    base = qb.load_base()
    data = [qb.scenario_data(base, s) for s in scenario_set(set_name)]
    folder = folder or RESULTS_DIR / f"eval_{label}_{set_name}"
    folder.mkdir(parents=True, exist_ok=True)
    pr.save_prices(x, folder / "prices.json", {"source": price_name})
    summaries = []
    for rule in rules:
        df, hourly, members = ev.evaluate(x, data, rule=rule, contract_kw=contract_kw)
        df.to_csv(folder / f"metrics_{rule}.csv", index=False)
        members.to_csv(folder / f"members_{rule}.csv", index=False)
        pd.DataFrame(hourly, columns=[f"h{t:02d}" for t in range(24)]).assign(scenario=df.scenario).to_csv(
            folder / f"hourly_import_{rule}.csv", index=False)
        s = ev.summarize(df)
        s.insert(0, "rule", rule); s.insert(0, "prices", label)
        s["share_with_exceedance"] = s.attrs["share_of_scenarios_with_exceedance"]
        summaries.append(s)
        print(f"\n== prices {label} | set {set_name} | rule {rule} | "
              f"share of scenarios with exceedance {s.attrs['share_of_scenarios_with_exceedance']:.2f}")
        print(s[["mean", "min", "median", "q95", "max", "cvar90_upper"]].round(3).to_string())
    out = pd.concat(summaries).rename_axis("metric").reset_index()
    out.to_csv(folder / "summary.csv", index=False)
    return out


def cmd_evaluate(args) -> None:
    run_evaluation(args.prices, args.set, args.rules.split(","))


def cmd_baseline(_args) -> None:
    cmd_validate(None)
    tables = [run_evaluation(p, "oos", RULES) for p in ("C", "DP")]
    tables += [run_evaluation(p, "expected", RULES) for p in ("C", "DP")]
    pd.concat(tables).to_csv(RESULTS_DIR / "E1_baseline_summary.csv", index=False)
    print(f"\nE1 summary written to {RESULTS_DIR / 'E1_baseline_summary.csv'}")


# ----------------------------------------------------------------------------------------------------------------------
# Stochastic pricing models (Gurobi)
# ----------------------------------------------------------------------------------------------------------------------
def sdp_config(args):
    from q3.sdp import SDPConfig
    risk = "cvar" if args.model == "CVaR" else "neutral"
    cap = "chance" if args.model == "CC" else "penalty"
    name = {"E": "SDP_E", "CVaR": f"SDP_CVaR_a{args.alpha:g}_w{args.weight:g}", "CC": f"SDP_CC_eps{args.epsilon:g}"}[args.model]
    penalty = getattr(args, "penalty", None)
    contract = {} if penalty is None else {"alpha": float(penalty)}
    if penalty is not None:
        name += f"_pen{penalty:g}"
    return SDPConfig(contract=contract, name=name, risk=risk, cvar_alpha=args.alpha, cvar_weight=args.weight, cap=cap,
                     epsilon=args.epsilon, bb=args.bb, time_limit=args.time_limit, mip_gap=args.gap,
                     threads=args.threads, warm_start=args.warm, bb_penalty=args.bb_penalty,
                     mip_focus=args.mip_focus)


def sdp_label(cfg, set_name: str) -> str:
    return f"{cfg.name}_{set_name}_bb-{cfg.bb}"


def cmd_sdp(args) -> dict:
    """Solve one SDP configuration on one in-sample set, then evaluate its prices (in-sample re-check and the
    100 out-of-sample scenarios) and the members' individual rationality in expectation."""
    from q3 import sdp
    from q3.settlement import ir_in_expectation
    cfg = sdp_config(args)
    base = qb.load_base()
    scen = scenario_set(args.set)
    folder = RESULTS_DIR / "sdp" / sdp_label(cfg, args.set)
    rep = sdp.solve(base, scen, cfg, out_dir=folder)
    keys = ("status", "solutions", "objective", "expected_cost", "bound", "gap", "bb_residual", "runtime_s",
            "total_runtime_s", "warm_start", "phase1_objective", "variables", "constraints",
            "sos1_pairs", "binaries", "max_complementarity", "max_strong_duality_gap", "max_member_suboptimality")
    print("\n" + json.dumps({k: rep.get(k) for k in keys}, indent=1))
    if rep["solutions"] == 0 or args.no_eval:
        return rep
    _post_solve(cfg, args.set, folder, rep, args.rules.split(","))
    return rep


TIE_ALPHA_CC = 100.0   # DKK/kWh: cap-first tie-break of the optimistic rule for SDP^CC (see evaluate.evaluate)


def eval_contract(cfg) -> dict:
    """Contract overrides used to evaluate a configuration: its own penalty, and for SDP^CC the cap-first tie-break."""
    kw = dict(cfg.contract)
    if cfg.cap == "chance":
        kw["tie_alpha"] = TIE_ALPHA_CC
    return kw


def _post_solve(cfg, set_name: str, folder: Path, rep: dict, rules) -> None:
    """In-sample re-check, out-of-sample evaluation and IR in expectation of a solved configuration."""
    from q3.settlement import ir_in_expectation
    kw = eval_contract(cfg)
    prices_file = str(folder / "prices.json")
    ins = run_evaluation(prices_file, set_name, ("optimistic",), contract_kw=kw, label=cfg.name,
                         folder=folder / f"eval_{set_name}")
    model_mean = float(rep["expected_cost"])
    note = ("cap-first tie-break: equal for epsilon = 0, may be higher for epsilon > 0" if "tie_alpha" in kw
            else "re-evaluation can only be lower or equal")
    print(f"\nIn-sample check: model expected cost {model_mean:.4f} DKK vs re-evaluated (optimistic) "
          f"{float(ins.loc[ins.metric == 'J', 'mean'].iloc[0]):.4f} DKK ({note})")
    run_evaluation(prices_file, "oos", list(rules), contract_kw=kw, label=cfg.name, folder=folder / "eval_oos")
    base = qb.load_base()
    data = [qb.scenario_data(base, s) for s in scenario_set("oos")]
    ir = ir_in_expectation(np.array(rep["prices"]), data, contract_kw=kw)
    ir.to_csv(folder / "ir_oos.csv", index=False)
    print(f"\nIR in expectation (out-of-sample, optimistic): {int((ir.margin < -1e-6).sum())} members worse off than R; "
          f"compensation {ir.attrs['deficit']:.3f} DKK, gains {ir.attrs['gains']:.3f} DKK")


def cmd_reeval(args) -> None:
    """Re-run the evaluation of solved configurations from their saved prices (no model is solved)."""
    from q3.sdp import SDPConfig
    for name in [n.strip() for n in args.runs.split(",") if n.strip()]:
        folder = RESULTS_DIR / "sdp" / name
        rep = json.loads((folder / "solve.json").read_text(encoding="utf-8"))
        cfg = SDPConfig(**rep["config"])
        set_name = name.split(f"{cfg.name}_", 1)[1].rsplit("_bb-", 1)[0]
        print(f"\n######## re-evaluating {name} (set {set_name}, contract {eval_contract(cfg)})", flush=True)
        _post_solve(cfg, set_name, folder, rep, args.rules.split(","))
    cmd_summary(None)


def cmd_sweep(args) -> None:
    """Sensitivity on the tuning parameter of a stochastic variant (Q3.6): solve one configuration per value, each
    warm-started from the given candidates, the previous point of the sweep and any earlier run of the same point."""
    import copy as _copy
    previous = None
    for v in [float(a) for a in args.values.split(",")]:
        a = _copy.copy(args)
        setattr(a, args.param, v)
        cfg = sdp_config(a)
        own = RESULTS_DIR / "sdp" / sdp_label(cfg, a.set) / "prices.json"
        cands = [c for c in args.warm.split(",") if c and c != "auto"]
        cands += [str(f) for f in (previous, own) if f is not None and Path(f).exists()]
        a.warm = ",".join(dict.fromkeys(cands)) or "auto"
        print(f"\n######## {cfg.name} on {a.set}: {args.param} = {v:g}  (warm: {a.warm})", flush=True)
        rep = cmd_sdp(a)
        if rep.get("solutions", 0) > 0:
            previous = RESULTS_DIR / "sdp" / sdp_label(cfg, a.set) / "prices.json"
    cmd_summary(None)


def cmd_select(args) -> pd.DataFrame:
    """Validation-based tie-break between price vectors that are equivalent in sample (N = 5): every candidate is
    evaluated (optimistic rule) on the validation scenarios = the N = 10 set without the nested N = 5 set (seed 1,
    never used by the N = 5 models and disjoint from the out-of-sample set). The candidate with the lowest mean
    realised cost is selected; regret, exceedance and budget are reported for transparency."""
    from q3.paths import Q3_DATA_DIR
    base = qb.load_base()
    n_in = int(args.inner[2:])
    ins = [qb.scenario_data(base, s) for s in scenario_set(args.inner)]
    val = [qb.scenario_data(base, s) for s in scenario_set(args.outer)[n_in:]]
    oos = [qb.scenario_data(base, s) for s in scenario_set("oos")] if args.oos else []
    kw = {} if args.penalty is None else {"alpha": args.penalty}
    cands = ([c.strip() for c in args.candidates.split(",") if c.strip()] if args.candidates
             else [str(f) for f in sorted((Q3_DATA_DIR / "n5_vectors").glob("*.json"))])
    rows = []
    for c in cands:
        _, x = pr.load_prices(c)
        meta = json.loads(Path(c).read_text(encoding="utf-8")) if c.endswith(".json") else {}
        run = meta.get("label") or (Path(c).parent.name if c.endswith(".json") else c)
        row = {"candidate": run}
        for tag, data in (("in", ins), ("val", val), ("oos", oos)):
            if not data:
                continue
            df, _, _ = ev.evaluate(x, data, "optimistic", kw)
            row.update({f"{tag}_J_mean": df.J.mean(), f"{tag}_regret_mean": df.regret.mean(),
                        f"{tag}_regret_max": df.regret.max(),
                        f"{tag}_regret_cvar90_upper": float(ev.summarize(df).loc["regret", "cvar90_upper"]),
                        f"{tag}_share_exceed": float((df.exceedance_kWh > 1e-6).mean()),
                        f"{tag}_budget_mean": df.budget_surplus.mean(), f"{tag}_budget_min": df.budget_surplus.min()})
        row["source"] = meta.get("source", c)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("val_J_mean").reset_index(drop=True)
    out["selected"] = out.index == 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pen = "" if args.penalty is None else f"_pen{args.penalty:g}"
    f = RESULTS_DIR / f"selection_{args.inner}_on_{args.outer}{pen}.csv"
    out.to_csv(f, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"validation scenarios: {len(val)} ({args.outer} without {args.inner})")
        print(out.round(3).to_string(index=False))
    print(f"\nwritten: {f}")
    return out


def _row_from_run(folder: Path) -> dict | None:
    """One line of the Q3 summary table from a run folder written by ``run_q3.py sdp``."""
    f = folder / "solve.json"
    if not f.exists():
        return None
    rep = json.loads(f.read_text(encoding="utf-8"))
    cfg = rep["config"]
    row = {"run": folder.name, "model": cfg["name"], "N": rep["N"], "bb": cfg["bb"], "risk": cfg["risk"],
           "cvar_alpha": cfg["cvar_alpha"] if cfg["risk"] == "cvar" else None,
           "cvar_weight": cfg["cvar_weight"] if cfg["risk"] == "cvar" else None,
           "epsilon": cfg["epsilon"] if cfg["cap"] == "chance" else None,
           "penalty": float(cfg.get("contract", {}).get("alpha", 10.0)),
           "status": rep["status"], "objective": rep.get("objective"), "expected_cost_in": rep.get("expected_cost"),
           "bound": rep.get("bound"), "gap": rep.get("gap"), "bb_residual": rep.get("bb_residual"),
           "runtime_s": rep.get("total_runtime_s", rep.get("runtime_s")), "variables": rep.get("variables"),
           "sos1_pairs": rep.get("sos1_pairs"), "binaries": rep.get("binaries"),
           "max_complementarity": rep.get("max_complementarity"),
           "max_strong_duality_gap": rep.get("max_strong_duality_gap"),
           "max_member_suboptimality": rep.get("max_member_suboptimality")}
    ins = folder / "in_sample.csv"
    if ins.exists():
        d = pd.read_csv(ins)
        row.update(in_share_exceed=float((d.exceedance_kWh > 1e-6).mean()), in_exceed_mean=float(d.exceedance_kWh.mean()),
                   in_budget_min=float(d.budget_surplus.min()))
    for set_dir in list(folder.glob("eval_in*")) + list(folder.glob("eval_expected")):
        sm = pd.read_csv(set_dir / "summary.csv")
        r = sm[(sm.rule == "optimistic") & (sm.metric == "regret")]
        if len(r):
            row["in_regret_mean"] = float(r["mean"].iloc[0])
    sm_path = folder / "eval_oos" / "summary.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path)
        opt = sm[sm.rule == "optimistic"].set_index("metric")
        for metric, cols in (("J", ("mean",)), ("regret", ("mean", "max", "cvar90_upper")),
                             ("exceedance_kWh", ("mean", "max")), ("penalty", ("mean",)),
                             ("budget_surplus", ("mean", "min"))):
            for c in cols:
                row[f"oos_{metric}_{c}"] = float(opt.loc[metric, c])
        row["oos_share_exceed"] = float(opt["share_with_exceedance"].iloc[0])
        for rule in ("independent", "pessimistic"):
            r = sm[(sm.rule == rule) & (sm.metric == "J")]
            if len(r):
                row[f"oos_J_mean_{rule}"] = float(r["mean"].iloc[0])
    ir = folder / "ir_oos.csv"
    if ir.exists():
        d = pd.read_csv(ir)
        row.update(ir_members_worse=int((d.margin < -1e-6).sum()), ir_compensation=float((-d.margin).clip(lower=0).sum()),
                   ir_gains=float(d.margin.clip(lower=0).sum()))
    return row


def cmd_summary(_args) -> pd.DataFrame:
    """All solved configurations in one table: results/q3_summary.csv (main source of the Q3 tables)."""
    rows = [r for f in sorted((RESULTS_DIR / "sdp").glob("*")) if f.is_dir() and not f.name.startswith("validate")
            for r in [_row_from_run(f)] if r is not None]
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "q3_summary.csv"
    df.to_csv(out, index=False)
    cols = [c for c in ("run", "objective", "gap", "runtime_s", "oos_J_mean", "oos_regret_mean", "oos_regret_cvar90_upper",
                        "oos_share_exceed", "oos_budget_surplus_mean", "ir_members_worse") if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].round(3).to_string(index=False))
    print(f"\nwritten: {out}")
    return df


def cmd_sdp_validate(args) -> None:
    """(V1) N = 1, theta = 1, cap only: the SDP is the deterministic DP and must reach the C cost (DP = C in the
    optimistic sense); (V2) three identical copies of the theta = 1 scenario give the same optimal value."""
    from q3.paths import describe
    print(f"using {describe()}", flush=True)
    from q3 import sdp
    from q3.sdp import SDPConfig
    j_c = json.loads((Q1_C_DIR / "metrics.json").read_text(encoding="utf-8"))["J_DKK"]
    base = qb.load_base()
    e = sc.expected_scenario()
    dup = []
    for k in range(3):
        d = dict(e); d["id"] = f"copy{k}"; d["probability"] = 1 / 3; dup.append(d)
    out = {"J_C_Q1": j_c}
    for tag, scen in (("V1_N1", [e]), ("V2_three_copies", dup)):
        cfg = SDPConfig(name=f"validate_{tag}", bb="none", time_limit=args.time_limit, mip_gap=args.gap,
                        threads=args.threads, warm_start="C")
        rep = sdp.solve(base, scen, cfg, out_dir=RESULTS_DIR / "sdp" / f"validate_{tag}")
        out[tag] = {k: rep.get(k) for k in ("status", "objective", "bound", "gap", "runtime_s", "variables",
                                            "sos1_pairs")}
        if rep.get("objective") is not None:
            out[tag]["difference_to_C"] = rep["objective"] - j_c
    out["passed"] = all(abs(out[t].get("difference_to_C", 1e9)) < 1e-3 for t in ("V1_N1", "V2_three_copies"))
    (RESULTS_DIR / "validation_sdp.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


NSTUDY_DIR = RESULTS_DIR / "nstudy"
DP_BB_PRICES = RESULTS_DIR / "sdp" / "SDP_E_expected_bb-expectation" / "prices.json"


def nested_sets(sizes) -> dict[int, list[dict]]:
    """In-sample sets of every size, all taken from one draw of the largest size (seed 1): the first n scenarios,
    equiprobable. The sets are nested by construction, and the n = 5 and n = 10 sets equal in5 and in10."""
    import copy as _copy
    big = sc.draw(max(sizes), sc.SEED_IN_SAMPLE)
    out = {}
    for n in sizes:
        sub = _copy.deepcopy(big[:n])
        for s in sub:
            s["probability"] = 1.0 / n
        out[n] = sub
    return out


def cmd_nstudy(args) -> pd.DataFrame:
    """Accuracy against performance (Section 3.4): SDP^E with budget balance in expectation on nested in-sample sets
    of increasing size N. Every N uses the same start (the DP^BB prices, phase 1 then the MILP), the same time limit,
    gap and threads, so that the solution times can be compared; the pooled multi-start of the main runs is NOT used.
    Each solution is re-evaluated in sample and on the same 100 out-of-sample scenarios (optimistic rule).
    Outputs: results/nstudy/N<n>/ (solve.json, prices.json, eval_in, eval_oos), results/nstudy/summary.csv and
    scenarios/in<n>.json. The main results under results/sdp are not touched."""
    from q3 import sdp
    from q3.sdp import SDPConfig
    sizes = sorted({int(v) for v in args.values.split(",")})
    warm = args.warm or str(DP_BB_PRICES)
    if warm.endswith(".json") and not Path(warm).exists():
        raise SystemExit(f"start prices not found: {warm} (run the DP^BB baseline first, run_all.ps1 step 7)")
    sets = nested_sets(sizes)
    checks = {f"in{n}": sc.check_set(sets[n], sc.SEED_IN_SAMPLE) for n in sizes}
    checks["nested"] = all(sc.check_nested(sets[a], sets[b]) for a, b in zip(sizes, sizes[1:]))
    # the draw indices are checked on the largest set; the smaller sets are its first scenarios (nested), so a re-draw
    # with their own size is not the reference (with one scenario the starter returns the mean, theta = 1)
    for n in sizes[:-1]:
        checks[f"in{n}"]["indices_recover_draws"] = "via nested largest set"
    for n in sizes:
        sc.save(sc.to_record(sets[n], f"in{n}", sc.SEED_IN_SAMPLE, ("profile",)), SCENARIO_DIR / f"in{n}.json")
    NSTUDY_DIR.mkdir(parents=True, exist_ok=True)
    (NSTUDY_DIR / "scenario_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    assert checks["nested"] and checks[f"in{sizes[-1]}"]["indices_recover_draws"] is True
    assert all(c["profile_within_bounds"] and c["emin_equals_reference_energy"] and c["probabilities_sum_to_one"]
               for k, c in checks.items() if k != "nested")
    print(f"nested in-sample sets {sizes} (seed {sc.SEED_IN_SAMPLE}) checked and saved; start prices: {warm}", flush=True)
    if args.dry_run:
        print("dry run: no model is solved")
        return pd.DataFrame()

    base = qb.load_base()
    oos = [qb.scenario_data(base, s) for s in scenario_set("oos")]
    rows = []
    summary_file = NSTUDY_DIR / "summary.csv"
    for n in sizes:
        folder = NSTUDY_DIR / f"N{n}"
        cfg = SDPConfig(name="SDP_E", bb="expectation", time_limit=args.time_limit, mip_gap=args.gap,
                        threads=args.threads, warm_start=warm, mip_focus=args.mip_focus)
        print(f"\n######## N = {n}: SDP^E, budget balance in expectation (time limit {args.time_limit:g} s)", flush=True)
        rep = sdp.solve(base, sets[n], cfg, out_dir=folder)
        row = {"N": n, "status": rep["status"], "time_limit_reached": rep["status"] == 9,
               "build_s": rep.get("build_s"), "phase1_s": rep.get("phase1_runtime_s"), "milp_s": rep.get("runtime_s"),
               "total_s": rep.get("total_runtime_s"), "variables": rep.get("variables"),
               "constraints": rep.get("constraints"), "sos1_pairs": rep.get("sos1_pairs"),
               "objective": rep.get("objective"), "bound": rep.get("bound"), "gap": rep.get("gap"),
               "start_objective": rep.get("phase1_objective")}
        if rep["solutions"] > 0:
            x = np.array(rep["prices"])
            for tag, data in (("in", [qb.scenario_data(base, s) for s in sets[n]]), ("oos", oos)):
                df, hourly, _ = ev.evaluate(x, data, rule="optimistic")
                (folder / f"eval_{tag}").mkdir(parents=True, exist_ok=True)
                df.to_csv(folder / f"eval_{tag}" / "metrics_optimistic.csv", index=False)
                row.update({f"{tag}_J_mean": df.J.mean(), f"{tag}_regret_mean": df.regret.mean(),
                            f"{tag}_regret_cvar90": float(ev.summarize(df).loc["regret", "cvar90_upper"]),
                            f"{tag}_share_exceed": float((df.exceedance_kWh > 1e-6).mean()),
                            f"{tag}_exceed_mean_kWh": df.exceedance_kWh.mean(),
                            f"{tag}_budget_mean": df.budget_surplus.mean(), f"{tag}_budget_min": df.budget_surplus.min()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_file, index=False)        # written after every N (safe to interrupt)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, indent=1), flush=True)
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out[[c for c in ("N", "status", "total_s", "variables", "objective", "gap", "in_J_mean", "oos_J_mean",
                               "oos_regret_mean", "oos_regret_cvar90", "oos_share_exceed") if c in out]].round(3)
              .to_string(index=False))
    print(f"\nwritten: {summary_file}")
    return out


def cmd_figures(_args) -> None:
    """Regenerate every report figure for which the experiment outputs exist (no model is solved)."""
    from q3 import figures as fg
    sdp_dir = RESULTS_DIR / "sdp"
    have = lambda name: (sdp_dir / name / "solve.json").exists() and (sdp_dir / name / "eval_oos").exists()
    written = []
    if all(have(f.name) and (f / "eval_expected").exists() for _, f, _ in fg.default_baseline_runs()):
        written.append(fg.baseline_imports(fg.default_baseline_runs()))
    main_runs = [n for n in ("SDP_E_expected_bb-expectation", "SDP_E_in10_bb-expectation") if have(n)]
    if len(main_runs) == 2:
        written.append(fg.prices([sdp_dir / n for n in main_runs]))
    cap_runs = [n for n in ("SDP_E_expected_bb-none", "SDP_E_in5_bb-none") if have(n)]
    if len(cap_runs) == 2:
        written.append(fg.prices([sdp_dir / n for n in cap_runs], out=fg.FIG_DIR / "q3_prices_cap_only", sdp=Path("-")))
    dist_runs = [r for r in fg.default_regret_runs() if have(r[1].name)]
    if len(dist_runs) >= 2:
        written.append(fg.regret_distribution(dist_runs))
    summary = cmd_summary(None)
    if not summary.empty and (summary.penalty == 3).any():
        written.append(fg.penalty_study(summary))
    if not summary.empty and summary[summary.risk == "neutral"].N.nunique() >= 2:
        written.append(fg.n_study(summary))
    if all(have(n) for n in ("SDP_E_expected_bb-none", "SDP_E_expected_bb-expectation", "SDP_E_in10_bb-expectation")):
        written.append(fg.regret_by_deviation())        # table of Section 3.5 (csv)
    if (RESULTS_DIR / "nstudy" / "summary.csv").exists():
        written.append(fg.n_timing(RESULTS_DIR / "nstudy" / "summary.csv"))
    print("\nfigures written:" if written else "no figure: run the experiments first")
    for w in written:
        print("  ", w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios").set_defaults(fn=cmd_scenarios)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    e = sub.add_parser("evaluate")
    e.add_argument("--prices", default="DP", help="C, DP or a prices.json file")
    e.add_argument("--set", default="oos", choices=["expected", "in5", "in10", "oos"])
    e.add_argument("--rules", default="optimistic", help="comma-separated: independent,optimistic,pessimistic")
    e.set_defaults(fn=cmd_evaluate)
    sub.add_parser("baseline").set_defaults(fn=cmd_baseline)

    def solver_args(p):
        p.add_argument("--time-limit", type=float, default=600.0, help="Gurobi time limit (s)")
        p.add_argument("--gap", type=float, default=1e-3, help="relative MIP gap")
        p.add_argument("--threads", type=int, default=0)

    sd = sub.add_parser("sdp", help="solve one stochastic pricing model and evaluate its prices")
    sd.add_argument("--model", choices=["E", "CVaR", "CC"], default="E")
    sd.add_argument("--set", choices=["in5", "in10", "expected"], default="in5")
    sd.add_argument("--bb", choices=["expectation", "scenario", "none"], default="expectation")
    sd.add_argument("--alpha", type=float, default=0.9, help="CVaR level")
    sd.add_argument("--weight", type=float, default=0.5, help="CVaR weight in the objective")
    sd.add_argument("--epsilon", type=float, default=0.2, help="chance constraint: max probability of exceedance")
    sd.add_argument("--warm", default="auto", help="auto | C | DP | none | prices.json")
    sd.add_argument("--bb-penalty", type=float, default=100.0,
                    help="DKK per DKK of budget-balance residual (elastic BB); 0 = hard equality")
    sd.add_argument("--mip-focus", type=int, default=1, help="Gurobi MIPFocus")
    sd.add_argument("--rules", default="optimistic,independent,pessimistic", help="out-of-sample selection rules")
    sd.add_argument("--penalty", type=float, default=None,
                    help="DSO penalty alpha (DKK/kWh) for model and evaluation; default = contract value (10)")
    sd.add_argument("--no-eval", action="store_true", help="solve only")
    solver_args(sd)
    sd.set_defaults(fn=cmd_sdp)
    sv = sub.add_parser("sdp-validate", help="SDP with one scenario (or identical copies) reproduces DP = C")
    solver_args(sv)
    sv.set_defaults(fn=cmd_sdp_validate)
    ns = sub.add_parser("nstudy", help="accuracy against performance over nested N, same start and limits (Q3.5)")
    ns.add_argument("--values", default="1,2,5,10,20", help="in-sample sizes N (nested, seed 1)")
    ns.add_argument("--warm", default="", help="start prices for every N (default: the DP^BB prices)")
    ns.add_argument("--mip-focus", type=int, default=1)
    ns.add_argument("--dry-run", action="store_true", help="draw, check and save the nested sets only")
    solver_args(ns)
    ns.set_defaults(fn=cmd_nstudy, time_limit=1800.0)
    sub.add_parser("figures", help="regenerate the report figures from saved results").set_defaults(fn=cmd_figures)
    sw = sub.add_parser("sweep", help="tuning-parameter study of a stochastic variant (Q3.6)")
    sw.add_argument("--model", choices=["E", "CVaR", "CC"], default="CVaR")
    sw.add_argument("--param", choices=["weight", "alpha", "epsilon"], default="weight")
    sw.add_argument("--values", required=True, help="comma-separated values, e.g. 0.25,0.5,0.75,1")
    sw.add_argument("--set", choices=["in5", "in10"], default="in5")
    sw.add_argument("--bb", choices=["expectation", "scenario", "none"], default="expectation")
    sw.add_argument("--alpha", type=float, default=0.9)
    sw.add_argument("--weight", type=float, default=0.5)
    sw.add_argument("--epsilon", type=float, default=0.2)
    sw.add_argument("--warm", default="", help="extra start candidates (comma-separated prices.json files)")
    sw.add_argument("--rules", default="optimistic")
    sw.add_argument("--penalty", type=float, default=None, help="DSO penalty alpha (DKK/kWh); default = contract (10)")
    sw.add_argument("--bb-penalty", type=float, default=100.0)
    sw.add_argument("--mip-focus", type=int, default=1)
    sw.add_argument("--no-eval", action="store_true")
    solver_args(sw)
    sw.set_defaults(fn=cmd_sweep)
    sub.add_parser("summary", help="all solved configurations in one table").set_defaults(fn=cmd_summary)
    re_ = sub.add_parser("reeval", help="re-evaluate solved runs from their saved prices (no solve)")
    re_.add_argument("--runs", required=True, help="comma-separated run folder names under results/sdp")
    re_.add_argument("--rules", default="optimistic")
    re_.set_defaults(fn=cmd_reeval)
    se = sub.add_parser("select", help="validation-based choice between in-sample-equivalent price vectors")
    se.add_argument("--candidates", default=None,
                    help="comma-separated prices.json files (or C, DP); default: the archived data/n5_vectors")
    se.add_argument("--oos", action="store_true", help="also evaluate every candidate out of sample")
    se.add_argument("--inner", default="in5", choices=["in5"], help="in-sample set of the candidates")
    se.add_argument("--outer", default="in10", choices=["in10"], help="nested larger set; its extra scenarios validate")
    se.add_argument("--penalty", type=float, default=None, help="DSO penalty alpha (DKK/kWh) used in the evaluation")
    se.set_defaults(fn=cmd_select)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
