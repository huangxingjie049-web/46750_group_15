"""One parameterized convex LP/QP for all report models.

Hourly energy is kWh in a 1 h interval; battery power is kW. Explicit linear
bounds expose all Pi values. No artificial exclusion of simultaneous operation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from .data_loader import InputData

# Residual/count tolerance in the fixed kWh/kW/DKK units used by this assignment.
TOL = 2e-6


# Convert scientific Python values into ordinary JSON-compatible values.
def json_value(value):
    """Convert numpy/path/scalar objects for documented JSON outputs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value)}")


# Keep saved metadata readable and reject NaN/Infinity in audit files.
def write_json(path, payload):
    """Write UTF-8 structured output; arrays carry units in their field names."""
    Path(path).write_text(
        json.dumps(payload, indent=2, default=json_value, allow_nan=False),
        encoding="utf-8",
    )


@dataclass
class Result:
    """Detached primal/dual result; no live solver objects are retained."""

    hourly: pd.DataFrame
    metrics: dict
    specification: dict
    certificate: dict

    # Export the completed result or figure to the requested output directory.
    def save(self, folder, data, validation):
        """Save one run: hourly CSV, JSON metrics/inputs/validation and dual rows."""
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        self.hourly.to_csv(out / "hourly.csv", index_label="hour")
        write_json(out / "metrics.json", self.metrics)
        write_json(
            out / "input_snapshot.json",
            {"data": asdict(data), "model": self.specification},
        )
        write_json(out / "validation.json", validation)
        pd.DataFrame(self.certificate["rows"]).to_csv(
            out / "constraint_duals.csv", index=False
        )


# Reject unsupported or infeasible inputs before allocating a solver model.
def _check(data, kind, coupling, terminal, pi):
    """Validate the supported report domain before a solver is constructed."""
    # Use a short alias for the case; no input fields are modified.
    d = data
    # The preference switch selects only the four report objective families.
    if kind not in ("utility", "linear", "quadratic", "coupled"):
        raise ValueError("Unknown preference kind.")
    if not 0 <= d.load_min_kWh <= d.load_max_kWh:
        raise ValueError("Invalid load bounds.")
    # Prices and PV availability must match the horizon and contain finite values.
    for a in (d.energy_price, d.pv_available):
        if len(a) != d.n_hours or not np.isfinite(a).all():
            raise ValueError("Invalid hourly data.")
    if np.any(d.pv_available < 0):
        raise ValueError("Negative PV availability.")
    # Grid capacity limits are outside the report model and must not be ignored.
    if d.max_import_kW is not None or d.max_export_kW is not None:
        raise ValueError("Report assumes unrestricted grid trades.")
    # A positive import/export spread excludes profitable simultaneous grid trading.
    if np.any(d.energy_price + d.import_tariff <= d.energy_price - d.export_tariff):
        raise ValueError("Require a strictly positive trading spread.")
    # Q1 requires marginal utility; Q2/Q3 require reference load and positive discomfort weight.
    if kind == "utility":
        if d.consumption_utility is None or not np.isfinite(d.consumption_utility):
            raise ValueError("Missing utility.")
    else:
        if d.reference_load is None or not np.isfinite(d.reference_load).all():
            raise ValueError("Missing reference.")
        weight = d.linear_disutility if kind == "linear" else d.quadratic_disutility
        if weight is None or not np.isfinite(weight) or weight <= 0:
            raise ValueError("Disutility weight must be positive.")
    # Nonnegative coupling preserves convexity of the disutility function.
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("Coupling coefficient must be nonnegative.")
    if kind != "coupled" and coupling != 0:
        raise ValueError("Coupling belongs to the coupled model.")
    # The daily requirement cannot exceed the sum of hourly load upper bounds.
    e = d.min_daily_energy_kWh
    if e is not None and (
        not np.isfinite(e) or not 0 <= e <= d.n_hours * d.load_max_kWh
    ):
        raise ValueError("Infeasible daily minimum.")
    # Validate terminal treatment before using a linear value for stored energy.
    if not np.isfinite(pi) or pi < 0:
        raise ValueError("Terminal price must be finite and nonnegative.")
    if terminal not in ("value", "cyclic"):
        raise ValueError("Invalid terminal mode.")
    # Storage inputs must be physical, including initial energy within capacity.
    if d.battery_capacity_kWh is not None:
        for x in (
            d.battery_capacity_kWh,
            d.battery_max_charge_kW,
            d.battery_max_discharge_kW,
            d.battery_initial_soc_kWh,
        ):
            if x is None or not np.isfinite(x) or x < 0:
                raise ValueError("Invalid battery input.")
        if d.battery_initial_soc_kWh > d.battery_capacity_kWh:
            raise ValueError("Initial inventory exceeds capacity.")
        for x in (d.battery_charging_efficiency, d.battery_discharging_efficiency):
            if x is None or not 0 < x <= 1:
                raise ValueError("Invalid efficiency.")


# Build the shared LP/QP, solve it, and detach its decisions and dual values.
def solve(
    data: InputData,
    kind="quadratic",
    *,
    coupling=0.0,
    terminal="value",
    pi=0.0,
    lp_path=None,
):
    """Solve max utility - procurement - discomfort + terminal credit.

    Inputs: InputData in kWh/kW/DKK; kind; coupling in DKK/kWh²; pi in
    DKK/kWh of internal energy. Outputs: detached Result with hourly duals.
    The optional lp_path exports the actual mathematical model for auditing.
    """
    _check(data, kind, coupling, terminal, pi)
    # Use a short alias for the case; no input fields are modified.
    d = data
    # One index represents one one-hour interval; battery powers need no extra time factor.
    T = list(range(d.n_hours))
    battery = d.battery_capacity_kWh is not None
    # Record the exact model choices with every run for later reproduction.
    specification = {
        "kind": kind,
        "coupling_DKK_per_kWh2": float(coupling),
        "terminal_mode": terminal if battery else None,
        "pi_DKK_per_kWh": float(pi if battery and terminal == "value" else 0),
        "interval_hours": 1.0,
        "solver_method": "primal simplex (QP/LP)",
        "raw_dual_convention": "Pi = derivative of maximized objective with respect to RHS",
    }
    # Keep solver resources local to this solve and silence routine solver output.
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("group15_consumer", env=env) as m:
            # Use primal simplex for reproducible LP/QP duals and tight feasibility tolerances.
            m.Params.Method = 0
            m.Params.FeasibilityTol = 1e-9
            m.Params.OptimalityTol = 1e-9
            var = {}
            con = {}
            # All models share physical decisions; add z or battery decisions only when needed.
            families = (
                ["load", "pv", "import", "export"]
                + (["z"] if kind == "linear" else [])
                + (["charge", "discharge", "soc"] if battery else [])
            )
            # Declare unrestricted variables, then impose all bounds as explicit dual-bearing rows.
            for name in families:
                var[name] = m.addVars(T, lb=-GRB.INFINITY, name=name)

            # Add explicit bound rows so their RHS sensitivities can be exported as Pi.
            def bounds(name, lo, hi=None):
                con[name + "_lower"] = m.addConstrs(
                    (var[name][t] >= lo for t in T), name=name + "_lower"
                )
                if hi is not None:
                    con[name + "_upper"] = m.addConstrs(
                        (var[name][t] <= (hi[t] if np.ndim(hi) else hi) for t in T),
                        name=name + "_upper",
                    )

            # Apply physical load/PV bounds and grid nonnegativity to every hour.
            bounds("load", d.load_min_kWh, d.load_max_kWh)
            bounds("pv", 0, d.pv_available)
            bounds("import", 0)
            bounds("export", 0)
            L, P, I, E = (var[k] for k in ["load", "pv", "import", "export"])
            C = var.get("charge")
            D = var.get("discharge")
            S = var.get("soc")
            # Supply equals consumption, export, and charging; discharge contributes to supply.
            con["balance"] = m.addConstrs(
                (
                    P[t]
                    + I[t]
                    + (D[t] if battery else 0)
                    - L[t]
                    - E[t]
                    - (C[t] if battery else 0)
                    == 0
                    for t in T
                ),
                name="balance",
            )
            # Only consumer load counts toward the daily duty; battery charging does not.
            if d.min_daily_energy_kWh is not None:
                con["minimum_energy"] = m.addConstr(
                    gp.quicksum(L[t] for t in T) >= d.min_daily_energy_kWh,
                    name="minimum_energy",
                )
            # Generation and imports cost money; exports reduce procurement cost.
            procurement = gp.quicksum(
                d.pv_marginal_cost * P[t]
                + (d.energy_price[t] + d.import_tariff) * I[t]
                - (d.energy_price[t] - d.export_tariff) * E[t]
                for t in T
            )
            # Q1 rewards load by u*L; all other families subtract reference-based discomfort.
            if kind == "utility":
                preference = gp.quicksum(d.consumption_utility * L[t] for t in T)
            elif kind == "linear":
                bounds("z", 0)
                z = var["z"]
                # z >= L-reference and z >= reference-L linearize the absolute deviation.
                con["deviation_pos"] = m.addConstrs(
                    (z[t] - L[t] >= -d.reference_load[t] for t in T),
                    name="deviation_pos",
                )
                con["deviation_neg"] = m.addConstrs(
                    (z[t] + L[t] >= d.reference_load[t] for t in T),
                    name="deviation_neg",
                )
                preference = -d.linear_disutility * gp.quicksum(z[t] for t in T)
            else:
                preference = -d.quadratic_disutility * gp.quicksum(
                    (L[t] - d.reference_load[t]) ** 2 for t in T
                )
                # The report uses a SUM of consecutive deviations, not a load-ramping difference.
                if kind == "coupled":
                    preference -= coupling * gp.quicksum(
                        (
                            L[t]
                            - d.reference_load[t]
                            + L[t - 1]
                            - d.reference_load[t - 1]
                        )
                        ** 2
                        for t in T[1:]
                    )
            # Storage adds efficiency losses and time coupling while retaining a continuous QP.
            if battery:
                bounds("charge", 0, d.battery_max_charge_kW)
                bounds("discharge", 0, d.battery_max_discharge_kW)
                bounds("soc", 0, d.battery_capacity_kWh)
                # soc[t] is end-of-hour energy; the first hour starts from the supplied initial stock.
                con["storage_balance"] = m.addConstrs(
                    (
                        S[t]
                        - (d.battery_initial_soc_kWh if t == 0 else S[t - 1])
                        - d.battery_charging_efficiency * C[t]
                        + D[t] / d.battery_discharging_efficiency
                        == 0
                        for t in T
                    ),
                    name="storage_balance",
                )
                # Cyclic mode restores initial inventory and receives no terminal value reward.
                if terminal == "cyclic":
                    con["terminal_soc"] = m.addConstr(
                        S[T[-1]] == d.battery_initial_soc_kWh, name="terminal_soc"
                    )
            # Credit final internal energy once, outside the hourly sum, in value mode only.
            m.setObjective(
                preference
                - procurement
                + (pi * S[T[-1]] if battery and terminal == "value" else 0),
                GRB.MAXIMIZE,
            )
            m.update()
            # Export the actual mathematical model before solving for independent inspection.
            if lp_path is not None:
                Path(lp_path).parent.mkdir(parents=True, exist_ok=True)
                m.write(str(lp_path))
            # Time the optimization itself and require an optimal solver status.
            start = time.perf_counter()
            m.optimize()
            elapsed = time.perf_counter() - start
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Optimization failed: Gurobi status {m.Status}")
            # Create the exported hourly table from solver decisions and the case input profiles.
            h = pd.DataFrame(index=pd.Index(T, name="hour"))
            h["price"] = d.energy_price
            h["pv_available"] = d.pv_available
            h["reference_load"] = (
                d.reference_load
                if d.reference_load is not None
                else np.zeros(d.n_hours)
            )
            h["effective_import_price"] = d.energy_price + d.import_tariff
            h["effective_export_price"] = d.energy_price - d.export_tariff
            for name, family in var.items():
                h[name] = [family[t].X for t in T]
            # Hourly dual rows become CSV columns; daily/terminal row duals are scalar values.
            scalar_duals = {}
            for name, family in con.items():
                if isinstance(family, gp.tupledict):
                    h["dual_" + name] = [family[t].Pi for t in T]
                else:
                    scalar_duals[name] = family.Pi
            # Recompute physical deviations and procurement from the solved decisions.
            h["deviation"] = h.load - h.reference_load
            h["absolute_deviation"] = h.deviation.abs()
            h["curtailment"] = h.pv_available - h.pv
            h["procurement_cost"] = (
                d.pv_marginal_cost * h.pv
                + h.effective_import_price * h["import"]
                - h.effective_export_price * h["export"]
            )
            h["utility"] = d.consumption_utility * h.load if kind == "utility" else 0.0
            # Keep local discomfort separate from the coupling term for transparent accounting.
            h["local_disutility"] = (
                0.0
                if kind == "utility"
                else (
                    d.linear_disutility * h.absolute_deviation
                    if kind == "linear"
                    else d.quadratic_disutility * h.deviation**2
                )
            )
            h["coupling_disutility"] = 0.0
            # Assign each pair penalty to its later hour; each pair appears exactly once.
            if kind == "coupled":
                h.loc[T[1:], "coupling_disutility"] = (
                    coupling
                    * (h.deviation.to_numpy()[1:] + h.deviation.to_numpy()[:-1]) ** 2
                )
            h["disutility"] = h.local_disutility + h.coupling_disutility
            # Aggregate report statistics; counts use TOL to exclude numerical rounding noise.
            metrics = {
                "status": "OPTIMAL",
                "objective": float(m.ObjVal),
                "net_utility": float(m.ObjVal),
                "procurement_cost": float(h.procurement_cost.sum()),
                "utility": float(h.utility.sum()),
                "disutility": float(h.disutility.sum()),
                "local_disutility": float(h.local_disutility.sum()),
                "coupling_disutility": float(h.coupling_disutility.sum()),
                "daily_energy": float(h.load.sum()),
                "absolute_deviation": float(h.absolute_deviation.sum()),
                "squared_deviation": float((h.deviation**2).sum()),
                "import_kWh": float(h["import"].sum()),
                "export_kWh": float(h["export"].sum()),
                "pv_kWh": float(h.pv.sum()),
                "curtailment_kWh": float(h.curtailment.sum()),
                "binding_hours": int(
                    (
                        (abs(h.load - d.load_min_kWh) <= TOL)
                        | (abs(h.load - d.load_max_kWh) <= TOL)
                    ).sum()
                ),
                "above_reference_hours": int((h.deviation > TOL).sum()),
                "above_reference_energy": float(h.deviation.clip(lower=0).sum()),
                "below_reference_energy": float((-h.deviation).clip(lower=0).sum()),
                "simultaneous_trade_hours": int(
                    ((h["import"] > TOL) & (h["export"] > TOL)).sum()
                ),
                # For a maximized objective with load sum >= Emin, the report multiplier is -Pi.
                "mu": float(-scalar_duals.get("minimum_energy", 0)),
                "minimum_energy_slack": (
                    float(h.load.sum() - d.min_daily_energy_kWh)
                    if d.min_daily_energy_kWh is not None
                    else None
                ),
                "solver_version": ".".join(map(str, gp.gurobi.version())),
                "runtime_seconds": elapsed,
                "variables": m.NumVars,
                "linear_constraints": m.NumConstrs,
                "terminal_credit": 0.0,
                "initial_soc": None,
                "final_soc": None,
            }
            # Current-day total cost excludes terminal credit; objective includes it when applicable.
            metrics["total_cost"] = metrics["procurement_cost"] + metrics["disutility"]
            # Report inventory, losses, overlap counts, and resource expansion sensitivities.
            if battery:
                metrics.update(
                    initial_soc=float(d.battery_initial_soc_kWh),
                    final_soc=float(h.soc.iloc[-1]),
                    terminal_credit=(
                        float(pi * h.soc.iloc[-1]) if terminal == "value" else 0.0
                    ),
                    simultaneous_charge_discharge_hours=int(
                        ((h.charge > TOL) & (h.discharge > TOL)).sum()
                    ),
                    charge_kWh=float(h.charge.sum()),
                    discharge_kWh=float(h.discharge.sum()),
                    battery_losses_kWh=float(
                        (
                            (1 - d.battery_charging_efficiency) * h.charge
                            + (1 / d.battery_discharging_efficiency - 1) * h.discharge
                        ).sum()
                    ),
                    # Increasing capacity relaxes every SOC upper row, so its marginal is their Pi sum.
                    marginal_capacity=float(h.dual_soc_upper.sum()),
                    # Common power raises BOTH charge and discharge limits; sum both sets of Pi.
                    marginal_power=float(
                        h.dual_charge_upper.sum() + h.dual_discharge_upper.sum()
                    ),
                )
            # Store all explicit rows; validation independently differentiates preference/cost.
            rows = [
                {"name": c.ConstrName, "sense": c.Sense, "rhs": c.RHS, "pi": c.Pi}
                for c in m.getConstrs()
            ]
            # Retain A, row senses/RHS/duals, and variable names for independent KKT checks.
            certificate = {
                "A": m.getA(),
                "rows": rows,
                "variable_names": [v.VarName for v in m.getVars()],
                "x": np.array([v.X for v in m.getVars()]),
            }
            return Result(h, metrics, specification, certificate)
