"""Headless report figures generated exclusively from computed results."""

from pathlib import Path
import tempfile
import matplotlib

# The Agg backend saves figures without a desktop or interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Apply consistent styling across all report figures.
plt.rcParams.update(
    {"font.size": 9, "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 110}
)


# Export the completed result or figure to the requested output directory.
def save(fig, out, name):
    """Save the same figure to PNG (180 dpi) and vector PDF, then close it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Write a temporary image then replace the target, avoiding partially written final figures.
    for ext in ("png", "pdf"):
        with tempfile.NamedTemporaryFile(dir=out, suffix="." + ext, delete=False) as f:
            temporary = Path(f.name)
        try:
            fig.savefig(temporary, format=ext, dpi=180, bbox_inches="tight")
            temporary.replace(out / f"{name}.{ext}")
        finally:
            if temporary.exists():
                temporary.unlink()
    # Release figure memory so long experiment sweeps do not accumulate open figures.
    plt.close(fig)


# Show the effective prices and physical input profiles before interpreting results.
def inputs(d, out):
    """Plot hourly price/tariffs and PV/reference input in physical units."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    # Separate wholesale prices from the effective prices faced by imports and exports.
    axes[0].step(d.hours, d.energy_price, where="post", label="Market price")
    axes[0].step(
        d.hours, d.energy_price + d.import_tariff, where="post", label="Import price"
    )
    axes[0].step(
        d.hours, d.energy_price - d.export_tariff, where="post", label="Export price"
    )
    axes[0].axhline(d.pv_marginal_cost, ls=":", color="black", label="PV marginal cost")
    axes[0].set_ylabel("Price (DKK/kWh)")
    axes[0].legend(ncol=4)
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    axes[1].bar(
        d.hours, d.pv_available, label="Available PV", color="orange", alpha=0.6
    )
    if d.reference_load is not None:
        axes[1].step(d.hours, d.reference_load, where="post", label="Reference")
    axes[1].set(xlabel="Hour t (interval [t, t+1))", ylabel="Energy (kWh / 1 h)")
    axes[1].legend()
    save(fig, out, "input_profiles")


# Show the solved load, physical dispatch, and hourly economic signals.
def schedule(r, d, out, name):
    """Plot load, grid/PV dispatch and economic prices from one solved model."""
    h = r.hourly
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].step(d.hours, h.load, where="post", label="Load", color="C0")
    if r.specification["kind"] != "utility":
        axes[0].step(
            d.hours,
            h.reference_load,
            where="post",
            label="Reference",
            ls="--",
            color="black",
        )
    axes[0].step(
        d.hours,
        h.pv_available,
        where="post",
        label="Available PV",
        ls=":",
        color="orange",
    )
    axes[0].set_ylabel("Load (kWh / 1 h)")
    axes[0].legend(ncol=3)
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    # Plot exports below zero for readability; stored export decisions remain nonnegative.
    axes[1].bar(d.hours, h["import"], label="Import", alpha=0.6)
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    axes[1].bar(d.hours, -h["export"], label="Export (negative)", alpha=0.6)
    axes[1].step(d.hours, h.pv, where="post", label="PV generation", color="orange")
    axes[1].set_ylabel("Energy (kWh / 1 h)")
    axes[1].legend(ncol=3)
    # Compare effective prices with the balance value; report lambda is negative raw balance Pi.
    # Include boundary 0 initial SOC, followed by each end-of-hour inventory at boundary t+1.
    axes[2].step(d.hours, h.effective_import_price, where="post", label="Import price")
    # Include boundary 0 initial SOC, followed by each end-of-hour inventory at boundary t+1.
    axes[2].step(d.hours, h.effective_export_price, where="post", label="Export price")
    # Include boundary 0 initial SOC, followed by each end-of-hour inventory at boundary t+1.
    axes[2].step(
        d.hours, -h.dual_balance, where="post", label="Balance value lambda", ls="--"
    )
    axes[2].axhline(d.pv_marginal_cost, ls=":", color="black", label="PV cost")
    if d.min_daily_energy_kWh is not None:
        axes[2].axhline(r.metrics["mu"], ls="-.", color="C4", label="Daily mu")
    axes[2].set(xlabel="Hour t", ylabel="Marginal value (DKK/kWh)")
    axes[2].legend(ncol=3)
    save(fig, out, name)


# Overlay solved loads against a common reference using consistent hourly axes.
def compare_load(results, d, out, name):
    """Compare solved hourly loads on one consistent axis and common inputs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if d.reference_load is not None:
        ax.step(
            d.hours,
            d.reference_load,
            where="post",
            label="Reference",
            color="black",
            ls=":",
        )
    for label, r in results.items():
        ax.step(d.hours, r.hourly.load, where="post", label=label)
    ax.set(xlabel="Hour t", ylabel="Load (kWh / 1 h)")
    ax.legend(ncol=3)
    save(fig, out, name)


# Compare load, PV, import, and export in separate panels with identical units.
def flows(results, d, out, name):
    """Show one panel per flow with the same units across compared models."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    # One panel per physical flow keeps units and model comparisons consistent.
    for ax, key in zip(axes, ["load", "pv", "import", "export"]):
        for label, r in results.items():
            ax.step(d.hours, r.hourly[key], where="post", label=label)
        ax.set_ylabel(f"{key.title()} (kWh / 1 h)")
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Hour t")
    save(fig, out, name)


# Connect coefficient changes to hourly responses and daily outcome metrics.
def coefficient_sweep(table, runs, d, out, name, parameter):
    """Plot daily metrics and representative hourly responses to one weight."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    x = table[parameter]
    # Selected hours illustrate different PV/reference/price regimes in coefficient response.
    for t in [8, 11, 12, 18]:
        axes[0, 0].plot(
            x, [r.hourly.loc[t, "load"] for r in runs], "o-", ms=3, label=f"Hour {t}"
        )
    axes[0, 0].set_ylabel("Load (kWh / 1 h)")
    axes[0, 0].legend()
    for ax, key, label in zip(
        list(axes.flat)[1:],
        [
            "daily_energy",
            "absolute_deviation",
            "binding_hours",
            "procurement_cost",
            "disutility",
        ],
        [
            "Daily energy (kWh)",
            "Absolute deviation (kWh)",
            "Bound hours (count)",
            "Procurement (DKK)",
            "Disutility (DKK)",
        ],
    ):
        ax.plot(x, table[key], "o-", ms=3)
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel(
            parameter + " (" + ("DKK/kWh" if parameter == "cL" else "DKK/kWh²") + ")"
        )
        if parameter == "cQ":
            ax.set_xscale("log")
    save(fig, out, name)


# Show both aggregate sensitivity and matched with/without load profiles.
def daily_sensitivity(table, profiles, d, out, experiment):
    """Plot daily outcomes and up to six with/without hourly comparisons."""
    # Choose one sensitivity experiment and pair each case with its unconstrained counterpart.
    rows = table[table.experiment == experiment]
    x = rows.parameter_value
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(x, rows.daily_energy, "o-", label="With requirement")
    axes[0, 0].plot(x, rows.unconstrained_daily_energy, "s--", label="Without")
    axes[0, 0].axhline(
        d.reference_load.sum(), ls=":", color="black", label="Reference total"
    )
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Daily energy (kWh)")
    for k in ["procurement_cost", "disutility", "total_cost"]:
        axes[0, 1].plot(x, rows[k], "o-", label=k)
    axes[0, 1].legend()
    axes[0, 1].set_ylabel("Cost (DKK)")
    axes[1, 0].plot(x, rows.mu, "o-")
    axes[1, 0].set_ylabel("Daily mu (DKK/kWh)")
    axes[1, 1].plot(x, rows.squared_deviation, "o-")
    axes[1, 1].set_ylabel("Squared deviation (kWh²)")
    for ax in axes.flat:
        ax.set_xlabel("E_min (kWh)" if experiment == "emin" else "c_Q (DKK/kWh²)")
        if experiment == "cq":
            ax.set_xscale("log")
    save(fig, out, f"q3f_{experiment}_summary")
    # Use six hourly panels so every reported sensitivity scenario can be inspected.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        r, b = profiles[(experiment, row.parameter_value)]
        ax.step(
            d.hours,
            d.reference_load,
            where="post",
            ls=":",
            color="black",
            label="Reference",
        )
        ax.step(
            d.hours, b.hourly.load, where="post", ls="--", label="Without requirement"
        )
        ax.step(d.hours, r.hourly.load, where="post", label="With requirement")
        ax.set(
            title=f"{experiment}={row.parameter_value:g}; mu={row.mu:.3f}",
            xlabel="Hour t",
            ylabel="Load (kWh / 1 h)",
        )
        ax.legend(fontsize=7)
    save(fig, out, f"q3f_{experiment}_load_profiles")


# Show dispatch, stored energy, grid flows, and changes in load deviation.
def battery_base(r, base, d, out):
    """Generate report-equivalent battery dispatch, flow and deviation figures."""
    h = r.hourly
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for key in ["effective_import_price", "effective_export_price"]:
        axes[0].step(d.hours, h[key], where="post", label=key)
    axes[0].set_ylabel("Price (DKK/kWh)")
    axes[0].legend()
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    axes[1].bar(d.hours, h.charge, label="Charge")
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    axes[1].bar(d.hours, -h.discharge, label="Discharge (negative)")
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend()
    # Include boundary 0 initial SOC, followed by each end-of-hour inventory at boundary t+1.
    axes[2].step(
        np.arange(d.n_hours + 1),
        np.r_[d.battery_initial_soc_kWh, h.soc],
        where="post",
        label="SOC",
    )
    axes[2].axhline(d.battery_capacity_kWh, ls=":", color="black", label="Capacity")
    axes[2].axhline(d.battery_initial_soc_kWh, ls="--", label="Initial SOC")
    axes[2].set(xlabel="Hour boundary (e_t at t+1)", ylabel="Stored energy (kWh)")
    axes[2].legend()
    save(fig, out, "q3g_base_dispatch")
    flows({"No battery": base, "Battery": r}, d, out, "q3g_flow_comparison")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for label, result in [("No battery", base), ("Battery", r)]:
        axes[0].step(d.hours, result.hourly.deviation, where="post", label=label)
    axes[0].set_ylabel("Signed deviation (kWh)")
    axes[0].legend()
    # Physical PV availability and reference load are inputs, not optimized dispatch.
    axes[1].bar(d.hours, h.absolute_deviation - base.hourly.absolute_deviation)
    axes[1].set(xlabel="Hour t", ylabel="Change in absolute\ndeviation (kWh)")
    save(fig, out, "q3g_deviation_comparison")


# Compare operating gains and marginal values for conditional sizing scans.
def sizing(table, out, name, xkey, marginal, unit):
    """Plot value/marginal and value per unit, omitting division at zero size."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # Display gain and marginal value on separate axes because their units differ.
    right = axes[0].twinx()
    for alpha, g in table.groupby("alpha"):
        g = g.sort_values(xkey)
        line = axes[0].plot(
            g[xkey], g.inventory_adjusted_value, label=f"alpha={alpha:g}"
        )[0]
        right.plot(g[xkey], g[marginal], ls="--", color=line.get_color())
        # Average gain per unit is undefined at zero size, so exclude those points.
        positive = g[g[xkey] > 0]
        axes[1].plot(
            positive[xkey],
            positive.inventory_adjusted_value / positive[xkey],
            color=line.get_color(),
            label=f"alpha={alpha:g}",
        )
    axes[0].set_ylabel("Inventory-adjusted value (DKK)")
    right.set_ylabel(f"Marginal value (DKK/{unit})")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    axes[1].set(xlabel=f"{xkey} ({unit})", ylabel=f"Average gain (DKK/{unit})")
    save(fig, out, name)
