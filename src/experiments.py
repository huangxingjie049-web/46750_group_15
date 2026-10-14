"""Report experiments: immutable inputs, explicit scenarios, checked saved outputs."""

from dataclasses import replace
from pathlib import Path
import hashlib
import platform
import sys
import numpy as np
import pandas as pd
from .data_loader import load_question, InputData
from .model import solve, write_json
from .validation import validate
from . import plotting as plots

# Public task descriptions are shared by CLI help and the experiment dispatcher.
TASKS = {
    "1e": "Q1 Case A model and primal/dual results",
    "1f": "Q1 Cases A/B comparison",
    "2b": "Linear disutility base and threshold-focused coefficient sweep",
    "2c": "Quadratic disutility base and coefficient sweep",
    "2d": "Q1/linear/quadratic comparison at low/base/high weights",
    "2e": "Coupled quadratic base and alternating/block price experiment",
    "3d": "Daily-energy model with physical/KKT/independent verification",
    "3e": "Daily-energy base vs Q2(c)",
    "3f": "Daily minimum and c_Q sensitivity",
    "3g": "Calibrated battery, cyclic alternative and no-daily-requirement comparison",
    "3g-pi": "Next-day terminal value calibration",
    "3g-sensitivity": "Mean-preserving price variability, capacity, power and saturation scans",
}


# Create a battery-free comparison while retaining all consumer/grid inputs.
def no_battery(d):
    """Remove battery parameters, retaining every consumer/grid input."""
    # dataclasses.replace creates a new scenario without editing the baseline JSON.
    return replace(
        d, **{k: None for k in d.__dataclass_fields__ if k.startswith("battery_")}
    )


# Change price spreads about the mean without changing the daily mean.
def mean_preserving_prices(prices, alpha):
    """Scale deviations about mean price; alpha is dimensionless and nonnegative."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")
    p = np.asarray(prices, dtype=float)
    # alpha=0 gives flat prices; alpha=1 recovers the baseline; the mean is unchanged.
    return p.mean() + alpha * (p - p.mean())


class Experiment:
    """Own one reproducible task and its validation/output manifest."""

    # Create the task output folder and collect runs for its final manifest.
    def __init__(self, out, task, plots_enabled=True):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.plots = plots_enabled
        self.rows = []
        self.runs = []

    # Use the same solve -> validate -> save pipeline for every scenario.
    def run(
        self,
        label,
        data,
        kind="quadratic",
        *,
        coupling=0.0,
        terminal="value",
        pi=0.0,
        **fields,
    ):
        """Solve/validate/save a labeled scenario; failure stops the experiment."""
        # Each scenario receives its own LP, hourly CSV, input snapshot, and validation record.
        folder = self.out / "runs" / label
        r = solve(
            data,
            kind,
            coupling=coupling,
            terminal=terminal,
            pi=pi,
            lp_path=folder / "model.lp",
        )
        # Validate immediately after solving and stop before exporting misleading successful outputs.
        v = validate(r, data)
        if not v["passed"]:
            raise RuntimeError(f"{label} validation failed: {v}")
        r.save(folder, data, v)
        # Collect scalar metrics for the task summary while preserving detailed per-run outputs.
        row = {"scenario": label, **fields, **r.metrics}
        self.rows.append(row)
        self.runs.append(
            {
                "scenario": label,
                "path": str(folder.relative_to(self.out)),
                "validation_passed": v["passed"],
            }
        )
        return r

    # Aggregate validated scenarios into CSV, LaTeX, and reproducibility metadata.
    def finish(self):
        """Write complete summary, readable LaTeX table and dependency manifest."""
        table = pd.DataFrame(self.rows)
        table.to_csv(self.out / "summary.csv", index=False)
        columns = [
            c
            for c in [
                "scenario",
                "daily_energy",
                "procurement_cost",
                "disutility",
                "objective",
                "mu",
            ]
            if c in table
        ]
        # Plain LaTeX needs no optional pandas Styler/Jinja2 dependency.
        tex = [
            "\\begin{tabular}{" + ("l" + "r" * (len(columns) - 1)) + "}",
            "\\hline",
            " & ".join(c.replace("_", r"\_") for c in columns) + r" \\",
            "\\hline",
        ]
        for _, row in table[columns].iterrows():
            tex.append(
                " & ".join(
                    str(v).replace("_", r"\_") if isinstance(v, str) else f"{v:.6f}"
                    for v in row
                )
                + r" \\"
            )
        tex += ["\\hline", "\\end{tabular}"]
        (self.out / "summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        # The manifest records validated runs, environment, figure names, and input byte hashes.
        payload = {
            "task": self.task,
            "all_validations_passed": all(r["validation_passed"] for r in self.runs),
            "run_count": len(self.runs),
            "runs": self.runs,
            "python": sys.version,
            "platform": platform.platform(),
            "units": "energy kWh, battery power kW, monetary values DKK, mu DKK/kWh",
            "figures": [p.name for p in sorted(self.out.glob("*.pdf"))],
            "input_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(
                    (Path(__file__).resolve().parents[1] / "data").glob("*.json")
                )
            },
        }
        write_json(self.out / "manifest.json", payload)
        return payload


# Use next-day forecasts to estimate the marginal value of initial stored energy.
def calibrate_pi(out, plots=True):
    """Estimate next-day marginal inventory value at 2 kWh (DDK/kWh).

    Uses supplied next-day forecasts and free day-2 end SOC with zero day-3
    value. Central differences .001/.01/.1 are retained with one-sided slopes.
    Returns selected price and epsilon stability table; all runs are validated.
    """
    exp = Experiment(out, "3g-pi", plots)
    # Calibrate from next-day forecasts while retaining the consumer and battery parameters.
    today = load_question("Q3_battery")
    d = replace(
        today,
        energy_price=today.energy_price_next_day.copy(),
        pv_available=today.pv_available_next_day.copy(),
        reference_load=today.reference_load_next_day.copy(),
    )
    # Day 2 ends freely with zero day-3 reward; this is not a joint two-day optimization.
    base = exp.run("next_day_initial_2", d, terminal="value", pi=0.0)
    rows = []
    # Perturb initial inventory symmetrically to check a stable local terminal-value estimate.
    for eps in [0.001, 0.01, 0.1]:
        low = exp.run(
            f"initial_minus_{eps:g}",
            replace(d, battery_initial_soc_kWh=2 - eps),
            terminal="value",
            pi=0.0,
        )
        high = exp.run(
            f"initial_plus_{eps:g}",
            replace(d, battery_initial_soc_kWh=2 + eps),
            terminal="value",
            pi=0.0,
        )
        # Compare one-sided slopes and average them to obtain the centered finite difference.
        left = (base.metrics["objective"] - low.metrics["objective"]) / eps
        right = (high.metrics["objective"] - base.metrics["objective"]) / eps
        rows.append(
            {
                "epsilon": eps,
                "left_slope": left,
                "right_slope": right,
                "pi": (left + right) / 2,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "epsilon_stability.csv", index=False)
    # Use the report epsilon of 0.01 kWh and preserve its full computed precision.
    pi = float(table.loc[table.epsilon == 0.01, "pi"].iloc[0])
    write_json(
        exp.out / "pi_estimate.json",
        {
            "pi_DKK_per_kWh": pi,
            "expansion_initial_soc_kWh": 2.0,
            "epsilon_kWh": 0.01,
            "day2_end": "free; zero day3 terminal value",
            "initial_energy_dual": float(base.hourly.dual_storage_balance.iloc[0]),
            "stability_range": float(np.ptp(table.pi)),
            "note": "Local linear approximation to future value; not a full two-day optimization.",
        },
    )
    # Always save the numerical value curve, even when figure generation is disabled.
    curve = []
    for x in np.linspace(0, 4, 17):
        r = exp.run(
            f"value_curve_{x:g}",
            replace(d, battery_initial_soc_kWh=float(x)),
            terminal="value",
            pi=0.0,
        )
        curve.append({"initial_soc": x, "objective": r.metrics["objective"]})
    c = pd.DataFrame(curve)
    c.to_csv(exp.out / "value_curve.csv", index=False)
    if plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(c.initial_soc, c.objective, "o-", label="Next-day optimum")
        ax.plot(
            c.initial_soc,
            base.metrics["objective"] + pi * (c.initial_soc - 2),
            "--",
            label="Local tangent at 2 kWh",
        )
        ax.set(
            xlabel="Initial stored energy (kWh)", ylabel="Next-day net utility (DKK)"
        )
        ax.legend()
        plots_module = globals()["plots"]
        plots_module.save(fig, exp.out, "next_day_value_curve")
    exp.finish()
    return pi, table


# Q1(e-f): solve the supplied Cases A and B and export their comparison.
def q1(exp):
    """Q1 solve supplied A/B inputs; compare value at degenerate ties."""
    results = {}
    for case in ["Q1_caseA", "Q1_caseB"]:
        d = load_question(case)
        r = exp.run(case, d, "utility")
        results[case] = r
        if exp.plots:
            plots.inputs(d, exp.out / case)
            plots.schedule(r, d, exp.out, case + "_overview")
    pd.DataFrame([{"case": k, **r.metrics} for k, r in results.items()]).to_csv(
        exp.out / "case_comparison.csv", index=False
    )


# Q2(b): compare the linear base with coefficients around price thresholds.
def q2b(exp):
    """34-run linear sweep with just-below/at/above all report thresholds."""
    d = load_question("Q2_linear")
    base = exp.run("base", d, "linear")
    # Sample just below, exactly at, and just above each linear price threshold.
    thresholds = [1.41, 1.45, 1.5, 1.6, 1.7, 1.9, 2.0, 2.1]
    reps = [0.5, 1.4, 1.43, 1.47, 1.55, 1.65, 1.8, 1.95, 2.05, 2.2]
    values = sorted(
        set(
            reps
            + [
                round(k + offset, 6)
                for k in thresholds
                for offset in [-0.001, 0, 0.001]
            ]
        )
    )
    rows = []
    runs = {}
    for value in values:
        r = exp.run(
            f"cL_{value:g}", replace(d, linear_disutility=value), "linear", cL=value
        )
        rows.append({"cL": value, **r.metrics})
        runs[value] = r
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "sweep_metrics.csv", index=False)
    # Stronger comfort should not reduce daily load or increase absolute deviation here.
    if np.any(np.diff(table.daily_energy) < -2e-6) or np.any(
        np.diff(table.absolute_deviation) > 2e-6
    ):
        raise RuntimeError("Linear sweep monotonicity failed.")
    table[table.cL.isin(reps)].to_csv(
        exp.out / "representative_metrics.csv", index=False
    )
    # Identify the hours whose loads jump across each tested linear threshold.
    changes = []
    for k in thresholds:
        low, high = runs[round(k - 0.001, 6)], runs[round(k + 0.001, 6)]
        for t in np.where(abs(high.hourly.load - low.hourly.load) > 2e-6)[0]:
            changes.append(
                {
                    "threshold": k,
                    "hour": t,
                    "load_below": low.hourly.loc[t, "load"],
                    "load_above": high.hourly.loc[t, "load"],
                }
            )
    pd.DataFrame(changes).to_csv(exp.out / "threshold_changes.csv", index=False)
    if exp.plots:
        plots.schedule(base, d, exp.out, "q2b_base")
        plots.coefficient_sweep(
            table, [runs[x] for x in values], d, exp.out, "q2b_sweep", "cL"
        )


# Q2(c): solve the quadratic base and the report coefficient sweep.
def q2c(exp):
    """Report quadratic base plus its eight coefficient scenarios."""
    d = load_question("Q2_quadratic")
    base = exp.run("base", d)
    runs = []
    rows = []
    # Use the report quadratic weights; only the discomfort coefficient changes.
    for q in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        r = exp.run(f"cQ_{q:g}", replace(d, quadratic_disutility=q), cQ=q)
        runs.append(r)
        rows.append({"cQ": q, **r.metrics})
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "sweep_metrics.csv", index=False)
    # The weighted convex formulation implies nonincreasing squared deviation as cQ rises.
    if np.any(np.diff(table.squared_deviation) > 2e-6):
        raise RuntimeError("Quadratic deviation monotonicity failed.")
    if exp.plots:
        plots.schedule(base, d, exp.out, "q2c_base")
        plots.coefficient_sweep(table, runs, d, exp.out, "q2c_sweep", "cQ")


# Q2(d): compare preference shapes at low/base/high weights and selected hours.
def q2d(exp):
    """Compare preference shapes; different coefficient units are not welfare weights."""
    linear = load_question("Q2_linear")
    quad = load_question("Q2_quadratic")
    results = {}
    # These pairs compare response shapes, not equal welfare preferences across different units.
    for label, cl, cq in [("low", 0.5, 0.05), ("base", 1.43, 1), ("high", 2.2, 10)]:
        l = exp.run(
            label + "_linear", replace(linear, linear_disutility=cl), "linear", cL=cl
        )
        q = exp.run(
            label + "_quadratic",
            replace(quad, quadratic_disutility=cq),
            "quadratic",
            cQ=cq,
        )
        if label == "base":
            results = {"Linear": l, "Quadratic": q}
    utility = exp.run("Q1_caseA", load_question("Q1_caseA"), "utility")
    if exp.plots:
        plots.compare_load(results, quad, exp.out, "q2d_base_profiles")
        plots.compare_load(
            {**results, "Q1 utility": utility}, quad, exp.out, "q2d_three_models"
        )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    # Use the report/old-runner sample points; keep numerical sweeps even without plotting.
    for i, (kind, source, weights, unit) in enumerate(
        [
            (
                "linear",
                linear,
                sorted(
                    set(
                        [0.5, 1.4, 1.43, 1.47, 1.55, 1.65, 1.8, 1.95, 2.05, 2.2]
                        + [
                            round(k + offset, 6)
                            for k in [1.41, 1.45, 1.5, 1.6, 1.7, 1.9, 2, 2.1]
                            for offset in [-0.001, 0, 0.001]
                        ]
                    )
                ),
                "DKK/kWh",
            ),
            ("quadratic", quad, [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10], "DKK/kWh²"),
        ]
    ):
        responses = []
        for weight in weights:
            ds = replace(
                source,
                **(
                    {"linear_disutility": float(weight)}
                    if kind == "linear"
                    else {"quadratic_disutility": float(weight)}
                ),
            )
            rr = exp.run(f"response_{kind}_{weight:.8g}", ds, kind)
            responses.append(rr)
        if exp.plots:
            ax = axes[i]
            for t in [8, 11, 12, 18]:
                ax.plot(
                    weights,
                    [rr.hourly.loc[t, "load"] for rr in responses],
                    label=f"Hour {t}",
                )
            ax.set(xlabel=f"{kind} coefficient ({unit})", ylabel="Load (kWh / 1 h)")
            ax.legend(ncol=4)
            if kind == "quadratic":
                ax.set_xscale("log")
    if exp.plots:
        plots.save(fig, exp.out, "q2d_coefficient_response")
    write_json(
        exp.out / "comparison_notes.json",
        {
            "interpretation": "cL (DKK/kWh) and cQ (DKK/kWh²) have different units; weighted costs across weights are not a fixed-preference welfare ranking.",
            "hour8_quadratic_base_kWh": float(
                results["Quadratic"].hourly.loc[8, "load"]
            ),
            "Q1": "Linear marginal consumption utility; changes preferences and therefore is not a common-reference comfort baseline.",
        },
    )


# Q2(e): compare separable/coupled preferences under two price-timing patterns.
def q2e(exp):
    """Use the report SUM-of-adjacent-deviations penalty, not a ramp penalty."""
    d = load_question("Q2_quadratic")
    base = exp.run("base_coupled", d, "coupled", coupling=1.0)
    sep = exp.run("base_separable", d)
    rows = []
    profiles = {}
    # Both price patterns contain twelve low and twelve high prices; only timing changes.
    scenarios = {
        "alternating": np.tile([1.0, 2.0], 12),
        "block": np.tile(np.repeat([1.0, 2.0], 6), 2),
    }
    linear = load_question("Q2_linear")
    # Hold physical profiles fixed and compare all three preference models under each pattern.
    for label, price in scenarios.items():
        ds = replace(d, energy_price=price)
        for kind in ["linear", "quadratic", "coupled"]:
            data = replace(linear, energy_price=price) if kind == "linear" else ds
            r = exp.run(
                label + "_" + kind,
                data,
                kind,
                coupling=1.0 if kind == "coupled" else 0.0,
            )
            profiles[label + " " + kind] = r
            rows.append({"price_scenario": label, "model": kind, **r.metrics})
    pd.DataFrame(rows).to_csv(exp.out / "temporal_comparison.csv", index=False)
    write_json(
        exp.out / "experiment_design.json",
        {
            "formula": "D=cQ*sum(delta_t²)+cS*sum_{t=1}^{23}(delta_t+delta_{t-1})²",
            "cS_DKK_per_kWh2": 1.0,
            "price_profiles": scenarios,
            "fairness": "Same mean 1.5, same 12 high/12 low hours, fixed PV/reference/bounds. Changed alignment with PV/reference is part of timing effect; equal mean alone does not remove it.",
            "warning": "Sum penalty discourages sustained same-sign deviations, not rapid load changes. Total disutility includes BOTH terms.",
        },
    )
    if exp.plots:
        plots.compare_load(
            {"Coupled": base, "Separable": sep}, d, exp.out, "q2e_base_comparison"
        )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        for name, p in scenarios.items():
            ax.step(d.hours, p, where="post", label=name)
        ax.set(xlabel="Hour t", ylabel="Market price (DKK/kWh)")
        ax.legend()
        plots.save(fig, exp.out, "q2e_price_profiles")
        for name in scenarios:
            plots.compare_load(
                {k: v for k, v in profiles.items() if k.startswith(name)},
                d,
                exp.out,
                f"q2e_{name}_loads",
            )


# Q3(d-e): compare the daily-energy requirement with its removal.
def q3base(exp):
    """Solve daily requirement and compare with the same consumer without it."""
    d = load_question("Q3")
    r = exp.run("daily_40", d)
    # Remove only the daily duty to isolate its effect on the same consumer.
    b = exp.run("without_requirement", replace(d, min_daily_energy_kWh=None))
    write_json(
        exp.out / "hypothesis_checks.json",
        {
            "daily_requirement_binding": abs(r.metrics["minimum_energy_slack"]) < 2e-6,
            "positive_mu": r.metrics["mu"] > 0,
            "forced_exceedance": d.min_daily_energy_kWh > d.reference_load.sum(),
            "energy_rule_error": abs(
                r.metrics["daily_energy"]
                - max(d.min_daily_energy_kWh, b.metrics["daily_energy"])
            ),
        },
    )
    if exp.plots:
        plots.schedule(r, d, exp.out, "q3e_comparison")
        plots.flows({"Q2(c)": b, "Q3": r}, d, exp.out, "q3e_energy_flows")
        plots.compare_load({"Q2(c)": b, "Q3": r}, d, exp.out, "q3e_load_comparison")


# Q3(f): vary the daily minimum and comfort coefficient in separate experiments.
def q3f(exp):
    """Report two independent no-battery sweeps with matching unconstrained runs."""
    d = load_question("Q3")
    rows = []
    profiles = {}
    bases = {}
    # Separate Emin sensitivity (cQ=1) from cQ sensitivity (Emin=30 kWh).
    for experiment, values in [
        ("emin", [20, 22.765, 25, 32.34, 40, 48]),
        ("cq", [0.1, 0.5, 1, 2, 5, 10]),
    ]:
        for value in values:
            q = 1.0 if experiment == "emin" else value
            emin = value if experiment == "emin" else 30.0
            ds = replace(d, quadratic_disutility=q, min_daily_energy_kWh=emin)
            # Cache the matching unconstrained baseline for each comfort coefficient.
            if q not in bases:
                bases[q] = exp.run(
                    f"unconstrained_cQ_{q:g}", replace(ds, min_daily_energy_kWh=None)
                )
            b = bases[q]
            r = exp.run(f"{experiment}_{value:g}", ds)
            if (
                abs(r.metrics["daily_energy"] - max(emin, b.metrics["daily_energy"]))
                > 2e-6
            ):
                raise RuntimeError("Daily energy rule failed.")
            rows.append(
                {
                    "experiment": experiment,
                    "parameter_value": value,
                    "cQ": q,
                    "Emin": emin,
                    "unconstrained_daily_energy": b.metrics["daily_energy"],
                    # Measure the requirement burden using C+D, including discomfort.
                    "incremental_total_cost": r.metrics["total_cost"]
                    - b.metrics["total_cost"],
                    **r.metrics,
                }
            )
            profiles[(experiment, value)] = (r, b)
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "sensitivity_metrics.csv", index=False)
    if exp.plots:
        for experiment in ["emin", "cq"]:
            plots.daily_sensitivity(table, profiles, d, exp.out, experiment)


# Separate raw objective improvement from the credit for initial inventory.
def battery_value(r, base, pi, d):
    """Raw net-utility improvement and initial-inventory-adjusted value in DKK."""
    # Raw operating value compares optimal objectives under the same current-day inputs.
    value = r.metrics["objective"] - base.metrics["objective"]
    return {
        "battery_value": value,
        # Subtract pi times initial internal energy to distinguish inventory credit from operating gain.
        "inventory_adjusted_value": value - pi * d.battery_initial_soc_kWh,
    }


# Q3(g): calibrate pi, compare terminal treatments, and remove the daily constraint.
def q3g(exp):
    """Calibrated base, cyclic alternative and battery/no-battery daily removal."""
    # Calibrate once before the current-day battery comparisons; reuse this pi in all scenarios.
    pi, _ = calibrate_pi(exp.out / "calibration", plots=exp.plots)
    d = load_question("Q3_battery")
    base = exp.run("no_battery", no_battery(d))
    r = exp.run("value_battery", d, pi=pi)
    cyc = exp.run("cyclic_battery", d, terminal="cyclic")
    rows = []
    for label, result, price in [
        ("no_battery", base, 0.0),
        ("terminal_value", r, pi),
        ("cyclic", cyc, 0.0),
    ]:
        rows.append(
            {
                "scenario": label,
                **result.metrics,
                **battery_value(result, base, price, d),
            }
        )
    pd.DataFrame(rows).to_csv(exp.out / "comparison_metrics.csv", index=False)
    # The no-daily comparison retains the battery and the same terminal treatment.
    without = replace(d, min_daily_energy_kWh=None)
    rb = exp.run("battery_without_daily_requirement", without, pi=pi)
    nb = exp.run("no_battery_without_daily_requirement", no_battery(without))
    pd.DataFrame(
        [
            {"scenario": k, **v.metrics}
            for k, v in [
                ("without_daily_no_battery", nb),
                ("without_daily_battery", rb),
            ]
        ]
    ).to_csv(exp.out / "without_daily_requirement.csv", index=False)
    if exp.plots:
        plots.battery_base(r, base, d, exp.out)
        plots.compare_load(
            {"No battery": nb, "Battery": rb},
            without,
            exp.out,
            "q3g_without_daily_requirement",
        )


# Q3(g)(v): scan capacity/power under fixed-mean price scenarios.
def q3g_sensitivity(exp, quick=False):
    """Report capacity/power/volatility sweeps, right marginals and plateau search.

    Initial inventory stays 2 kWh. pi is calibrated once and stays fixed across
    current-day scenarios. Each environment has its own no-battery benchmark.
    Separate conditional plateaus do not identify jointly optimal sizing.
    """
    # Calibrate once before the current-day battery comparisons; reuse this pi in all scenarios.
    pi, _ = calibrate_pi(exp.out / "calibration", plots=exp.plots)
    d = load_question("Q3_battery")
    # Combine report sizing points with dense grids; quick mode omits the dense grids.
    capacities = sorted(
        set(
            [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
            + ([] if quick else list(np.arange(2, 12.001, 0.25)))
        )
    )
    powers = sorted(
        set(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            + ([] if quick else list(np.arange(0, 6.001, 0.1)))
        )
    )
    # Avoid resolving identical sizing scenarios encountered by scans and plateau searches.
    cache = {}
    cap_rows = []
    power_rows = []
    env_rows = []
    plateaus = []
    # Probe just above a size to obtain a right marginal at possibly nondifferentiable points.
    precision = 1e-4
    # Each price environment gets its own no-battery benchmark under the same altered prices.
    for alpha in [0.0, 0.5, 1.0, 1.5]:
        ds = replace(d, energy_price=mean_preserving_prices(d.energy_price, alpha))
        base = exp.run(f"alpha_{alpha:g}_no_battery", no_battery(ds), alpha=alpha)

        # Reuse a validated battery run for a given alpha, capacity, and common power.
        def evaluate(cap, power):
            key = (alpha, round(cap, 10), round(power, 10))
            if key not in cache:
                # Change capacity/common power while retaining fixed initial energy of 2 kWh.
                scenario = replace(
                    ds,
                    battery_capacity_kWh=float(cap),
                    battery_max_charge_kW=float(power),
                    battery_max_discharge_kW=float(power),
                )
                r = exp.run(
                    f"alpha_{alpha:g}_E_{cap:.8g}_P_{power:.8g}",
                    scenario,
                    pi=pi,
                    alpha=alpha,
                    capacity=cap,
                    power=power,
                )
                cache[key] = (
                    r,
                    {
                        **battery_value(r, base, pi, d),
                        "marginal_capacity": r.metrics["marginal_capacity"],
                        "marginal_power": r.metrics["marginal_power"],
                    },
                )
            return cache[key]

        r, metrics = evaluate(4, 1.5)
        # Measure capacity and common-power marginals just above their base sizes.
        _, right_e = evaluate(4 + precision, 1.5)
        _, right_p = evaluate(4, 1.5 + precision)
        env_rows.append(
            {
                "alpha": alpha,
                "price_mean": float(ds.energy_price.mean()),
                "price_range": float(np.ptp(ds.energy_price)),
                **metrics,
                "marginal_capacity_right": right_e["marginal_capacity"],
                "marginal_power_right": right_p["marginal_power"],
                "capacity_10pct_first_order": 0.1 * 4 * right_e["marginal_capacity"],
                "power_10pct_first_order": 0.1 * 1.5 * right_p["marginal_power"],
            }
        )
        # Capacity scan holds both power limits at 1.5 kW.
        for cap in capacities:
            r, m = evaluate(cap, 1.5)
            _, right = evaluate(cap + precision, 1.5)
            cap_rows.append(
                {
                    "alpha": alpha,
                    "capacity": cap,
                    **m,
                    "marginal_capacity": right["marginal_capacity"],
                }
            )
        # Common-power scan holds energy capacity at 4 kWh.
        for power in powers:
            r, m = evaluate(4, power)
            _, right = evaluate(4, power + precision)
            power_rows.append(
                {
                    "alpha": alpha,
                    "power": power,
                    **m,
                    "marginal_power": right["marginal_power"],
                }
            )

        # Locate the expansion size beyond which the right marginal value vanishes.
        def plateau(resource, low, high):
            # Vary only the requested resource and hold the other resource at its base size.
            def eval_size(x):
                return (
                    evaluate(x, 1.5)[1] if resource == "capacity" else evaluate(4, x)[1]
                )

            key = "marginal_" + resource
            # If the base right marginal is already zero, no expansion is needed in this scan.
            if eval_size(low + precision)[key] <= 1e-6:
                return low, True
            if eval_size(high + precision)[key] > 1e-6:
                raise RuntimeError(f"{resource} plateau not reached within range.")
            # Bisect the transition to marginal<=1e-6; plateau sizes have numerical search precision.
            for _ in range(19):
                mid = (low + high) / 2
                if eval_size(mid + precision)[key] > 1e-6:
                    low = mid
                else:
                    high = mid
            return high, False

        # Full report mode searches conditional plateaus; quick mode only provides smoke outputs.
        if not quick:
            ce, already_e = plateau("capacity", 4, 12)
            pp, already_p = plateau("power", 1.5, 6)
            plateaus.append(
                {
                    "alpha": alpha,
                    "capacity_plateau": ce,
                    "capacity_already_at_base": already_e,
                    "capacity_value": evaluate(ce + precision, 1.5)[1][
                        "inventory_adjusted_value"
                    ],
                    "power_plateau": pp,
                    "power_already_at_base": already_p,
                    "power_value": evaluate(4, pp + precision)[1][
                        "inventory_adjusted_value"
                    ],
                }
            )
    ct = pd.DataFrame(cap_rows)
    pt = pd.DataFrame(power_rows)
    et = pd.DataFrame(env_rows)
    ct.to_csv(exp.out / "capacity_scan.csv", index=False)
    pt.to_csv(exp.out / "power_scan.csv", index=False)
    et.to_csv(exp.out / "price_variability.csv", index=False)
    pd.DataFrame(plateaus).to_csv(exp.out / "conditional_plateaus.csv", index=False)
    # Compact report tables; dense scans remain available for reproducing curves.
    ct[(ct.alpha == 1) & ct.capacity.isin([2, 3, 4, 5, 6, 8, 12])].to_csv(
        exp.out / "table14_capacity.csv", index=False
    )
    pt[(pt.alpha == 1) & pt.power.isin([0, 0.5, 1, 1.5, 2, 3, 4, 6])].to_csv(
        exp.out / "table15_power.csv", index=False
    )
    et.to_csv(exp.out / "table16_variability.csv", index=False)
    pd.DataFrame(plateaus).to_csv(exp.out / "table17_plateaus.csv", index=False)
    # Count consecutive above/below-mean price blocks for the report duration interpretation.
    blocks = []
    for p in d.energy_price:
        sign = bool(p > d.energy_price.mean())
        if not blocks or blocks[-1]["above_mean"] != sign:
            blocks.append({"above_mean": sign, "hours": 1})
        else:
            blocks[-1]["hours"] += 1
    write_json(
        exp.out / "experiment_design.json",
        {
            "price_formula": "mean(p)+alpha*(p-mean(p))",
            "alphas": [0, 0.5, 1, 1.5],
            "pi_fixed": pi,
            "initial_soc_kWh_fixed": 2.0,
            "capacity_scan_power_kW": 1.5,
            "power_scan_capacity_kWh": 4.0,
            "right_marginal_perturbation": precision,
            "plateau_marginal_tolerance": 1e-6,
            "quick": quick,
            "price_blocks": blocks,
            "median_block_hours": float(np.median([b["hours"] for b in blocks])),
            "equal_to_mean_convention": "assigned to not-above group; no equal-to-mean hour in baseline",
            "scope": "Values exclude investment cost. H*P is illustrative throughput, not a proof of a universal saturation formula.",
        },
    )
    if exp.plots:
        plots.sizing(
            ct[ct.alpha == 1],
            exp.out,
            "q3g_capacity_base",
            "capacity",
            "marginal_capacity",
            "kWh",
        )
        plots.sizing(
            pt[pt.alpha == 1],
            exp.out,
            "q3g_power_base",
            "power",
            "marginal_power",
            "kW",
        )
        plots.sizing(
            ct,
            exp.out,
            "q3g_capacity_by_variability",
            "capacity",
            "marginal_capacity",
            "kWh",
        )
        plots.sizing(
            pt, exp.out, "q3g_power_by_variability", "power", "marginal_power", "kW"
        )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        for a in [0, 0.5, 1, 1.5]:
            axes[0].step(
                d.hours,
                mean_preserving_prices(d.energy_price, a),
                where="post",
                label=f"alpha={a:g}",
            )
        axes[0].set(xlabel="Hour t", ylabel="Market price (DKK/kWh)")
        axes[0].legend(ncol=4)
        for key in ["battery_value", "inventory_adjusted_value"]:
            axes[1].plot(et.alpha, et[key], "o-", label=key)
        axes[1].set(xlabel="Price variability alpha", ylabel="Value (DKK)")
        axes[1].legend()
        plots.save(fig, exp.out, "q3g_price_variability")


# Map the public task name to its experiment and write the final summary.
def run_task(task, out, plots=True, quick=False):
    """Run one named task into its own folder; return a checked manifest."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task}")
    exp = Experiment(out, task, plots)
    # Standalone calibration owns its own Experiment and writes its own detailed outputs.
    if task == "3g-pi":
        calibrate_pi(out, plots)
        return {"all_validations_passed": True}
    # Aliases 1e/1f and 3d/3e intentionally share their underlying experiment functions.
    fn = {
        "1e": q1,
        "1f": q1,
        "2b": q2b,
        "2c": q2c,
        "2d": q2d,
        "2e": q2e,
        "3d": q3base,
        "3e": q3base,
        "3f": q3f,
        "3g": q3g,
    }
    if task == "3g-sensitivity":
        q3g_sensitivity(exp, quick)
    else:
        fn[task](exp)
    # Summarize the task only after all scenario validations and optional figures complete.
    return exp.finish()
