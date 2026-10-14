"""Freshly render report-panel counterparts from computed CSV/JSON outputs.

These plots use the report panel choices and the user's earlier drawing code.
Every curve is read from newly computed CSV/JSON outputs. Original report images
are neither bundled nor read by this renderer.
"""

from pathlib import Path
from contextlib import contextmanager
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data_loader import load_question


# The file reader exposes only numerical run outputs, never archived picture pixels.
class Runs:
    """Access a completed experiment output parent, caching hourly schedules."""

    def __init__(self, root, aliases=None):
        self.root = Path(root)
        self.aliases = aliases or {}
        self.cache = {}
        self.accessed = set()

    def hourly(self, task, scenario):
        path = (
            self.root / self.aliases.get(task, task) / "runs" / scenario / "hourly.csv"
        )
        self.accessed.add(path)
        if path not in self.cache:
            self.cache[path] = pd.read_csv(path).set_index("hour")
        return self.cache[path]

    def table(self, task, name):
        path = self.root / self.aliases.get(task, task) / name
        self.accessed.add(path)
        return pd.read_csv(path)

    # Experimental price inputs are saved by the solver task, not recreated from picture pixels.
    def design(self, task):
        path = self.root / self.aliases.get(task, task) / "experiment_design.json"
        self.accessed.add(path)
        return json.loads(path.read_text())

    # Scalar metrics supply annotations without reading report literals.
    def metrics(self, task, scenario):
        path = (
            self.root
            / self.aliases.get(task, task)
            / "runs"
            / scenario
            / "metrics.json"
        )
        self.accessed.add(path)
        return json.loads(path.read_text())


# Capture the actual Matplotlib data rather than reconstruct a second, possibly different plot.
def plotted_data(fig):
    """Export line/bar coordinates, labels, and styles for independent figure-data auditing."""
    axes = []
    for ax in fig.axes:
        lines = []
        for line in ax.lines:
            lines.append(
                {
                    "label": line.get_label(),
                    "x": np.asarray(line.get_xdata(), dtype=float).tolist(),
                    "y": np.asarray(line.get_ydata(), dtype=float).tolist(),
                    "drawstyle": line.get_drawstyle(),
                    "linestyle": line.get_linestyle(),
                    "color": str(line.get_color()),
                }
            )
        bars = []
        for container in ax.containers:
            if hasattr(container, "patches"):
                bars.append(
                    {
                        "label": container.get_label(),
                        "x_center": [
                            p.get_x() + p.get_width() / 2 for p in container.patches
                        ],
                        "height": [p.get_height() for p in container.patches],
                    }
                )
        axes.append(
            {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(),
                "lines": lines,
                "bars": bars,
            }
        )
    return {
        "method": "Captured from Matplotlib artists drawn from computed CSV/JSON data",
        "axes": axes,
    }


# Build only figures belonging to the selected task, with no unrelated CSV prerequisites.
def figure_builders(runs):
    """Map report figure numbers to deferred builders using freshly solved run files."""
    h = runs.hourly
    m = runs.metrics
    t = runs.table
    return {
        1: lambda: q1_overview(h("1f", "Q1_caseA"), load_question("Q1_caseA")),
        2: lambda: q1_overview(h("1f", "Q1_caseB"), load_question("Q1_caseB")),
        3: lambda: base_load(h("2b", "base"), "Q2(b) base case: c_L = 1.43 DKK/kWh"),
        4: lambda: coefficient_response(runs, "2b", "linear", [7, 9, 11, 18]),
        5: lambda: quadratic_base(h("2c", "base"), m("2c", "base")["total_cost"]),
        6: lambda: quadratic_sweep(runs),
        7: lambda: model_comparison(h("2d", "base_linear"), h("2d", "base_quadratic")),
        8: lambda: combined_response(runs),
        9: lambda: coupled_comparison(h("2e", "base_coupled")),
        10: lambda: coupled_comparison(
            h("2e", "base_coupled"), h("2e", "base_separable")
        ),
        11: lambda: timing_prices(runs.design("2e")["price_profiles"]),
        12: lambda: daily_base(
            h("3e", "daily_40"),
            h("3e", "without_requirement"),
            m("3e", "daily_40")["mu"],
        ),
        13: lambda: daily_flows(h("3e", "daily_40"), h("3e", "without_requirement")),
        14: lambda: sensitivity_summary(t("3f", "sensitivity_metrics.csv"), "emin"),
        15: lambda: sensitivity_profiles(
            runs, t("3f", "sensitivity_metrics.csv"), "emin"
        ),
        16: lambda: sensitivity_summary(t("3f", "sensitivity_metrics.csv"), "cq"),
        17: lambda: sensitivity_profiles(
            runs, t("3f", "sensitivity_metrics.csv"), "cq"
        ),
        18: lambda: battery_dispatch(h("3g", "value_battery")),
        19: lambda: battery_flows(h("3g", "value_battery"), h("3g", "no_battery")),
        20: lambda: battery_deviations(h("3g", "value_battery"), h("3g", "no_battery")),
        21: lambda: sizing(
            t("3g-sensitivity", "capacity_scan.csv"),
            t("3g-sensitivity", "conditional_plateaus.csv"),
            "capacity",
        ),
        22: lambda: sizing(
            t("3g-sensitivity", "power_scan.csv"),
            t("3g-sensitivity", "conditional_plateaus.csv"),
            "power",
        ),
        23: lambda: variability(t("3g-sensitivity", "price_variability.csv")),
        24: lambda: sizing(
            t("3g-sensitivity", "capacity_scan.csv"),
            t("3g-sensitivity", "conditional_plateaus.csv"),
            "capacity",
            True,
        ),
        25: lambda: sizing(
            t("3g-sensitivity", "power_scan.csv"),
            t("3g-sensitivity", "conditional_plateaus.csv"),
            "power",
            True,
        ),
        26: lambda: model_comparison(
            h("2d", "base_linear"), h("2d", "base_quadratic"), False
        ),
    }


# Save every report-named PNG/JPG directly from Figure.savefig, never from a source picture.
def render_report_figures(results_root, tasks=None, quick=False):
    """Generate selected report figures, vector PDFs, captured artist data, and provenance.

    The ordinary full run creates all 26 figures at their original LaTeX paths.
    Quick mode omits the four sizing figures that require full plateau searches.
    """
    from .report import REPORT_TASKS, ALIASES
    from .model import write_json

    parent = Path(results_root)
    selected = REPORT_TASKS if tasks is None else tasks
    aliases = {ALIASES.get(task, task): task for task in selected}
    runs = Runs(parent, aliases)
    builders = figure_builders(runs)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1] / "report_reference/contract.json"
        ).read_text()
    )
    output = parent / "report"
    output.mkdir(parents=True, exist_ok=True)
    index = []
    with report_style():
        for reference in contract["figures"]:
            number = reference["figure"]
            if reference["task"] not in aliases or (
                quick and number in [21, 22, 24, 25]
            ):
                continue
            runs.accessed.clear()
            fig = builders[number]()
            # The final-report Q3 artwork uses larger publication labels than the old default theme.
            font_sizes = {
                12: 18,
                13: 16,
                18: 14,
                19: 12,
                20: 14,
                21: 14,
                22: 14,
                23: 18,
                24: 18,
                25: 18,
            }
            if number in font_sizes:
                scale = font_sizes[number] / 10
                for axis in fig.axes:
                    axis.xaxis.label.set_fontsize(
                        axis.xaxis.label.get_fontsize() * scale
                    )
                    axis.yaxis.label.set_fontsize(
                        axis.yaxis.label.get_fontsize() * scale
                    )
                    axis.title.set_fontsize(axis.title.get_fontsize() * scale)
                    axis.tick_params(labelsize=font_sizes[number])
                    legend = axis.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontsize(text.get_fontsize() * scale)
            # Physical size and DPI preserve the reference file's image dimensions.
            dpi = reference["dpi"]
            fig.set_size_inches(
                (reference["width_pixels"] + 1e-6) / dpi,
                (reference["height_pixels"] + 1e-6) / dpi,
            )
            fig.tight_layout()
            target = output / reference["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=dpi)
            fig.savefig(target.with_suffix(".pdf"))
            datafile = target.with_suffix(".plot_data.json")
            write_json(datafile, plotted_data(fig))
            plt.close(fig)
            source_files = [
                {
                    "path": str(p.relative_to(parent)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(runs.accessed)
            ]
            # Prices in Figure 11 are experimental inputs: retain the actual input definition too.
            if number == 11:
                design = parent / aliases["2e"] / "experiment_design.json"
                source_files.append(
                    {
                        "path": str(design.relative_to(parent)),
                        "sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
                    }
                )
            index.append(
                {
                    "figure": number,
                    "path": reference["path"],
                    "pdf": str(target.with_suffix(".pdf").relative_to(output)),
                    "plot_data": str(datafile.relative_to(output)),
                    "source_tex": reference["source_tex"],
                    "source_files": source_files,
                    "method": "Matplotlib rendering from computed numerical outputs",
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "reference_sha256": reference["sha256"],
                    "width_pixels": reference["width_pixels"],
                    "height_pixels": reference["height_pixels"],
                }
            )
    pd.DataFrame(index).drop(columns=["source_files"], errors="ignore").to_csv(
        output / "figure_index.csv", index=False
    )
    write_json(output / "figure_generation.json", index)
    return index


# Scope styling to report rendering so imported diagnostic modules cannot change it.
@contextmanager
def report_style():
    # Restore the original runners' defaults, rather than inherit diagnostic styling.
    with plt.rc_context(
        {
            **dict(plt.rcParamsDefault),
            "backend": "Agg",
            "font.size": 10,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
        }
    ):
        yield


# Q1/Q2 and Q3(f) retain the old runners' midpoint display; Q3(e/g) use interval boundaries.
"""Draw an hourly result column on the supplied Matplotlib axes.

    Inputs: hourly DataFrame, column name, legend label, and plot style.
    Values retain the selected column's units. Updates ax; returns None.
    """

def step(ax, h, column, label, **style):
    where = style.pop("where", "mid")
    x = h.index.to_numpy()
    y = h[column].to_numpy()
    # A post-step final interval must extend to boundary 24, not stop at hour 23.
    if where == "post":
        x = np.r_[x, x[-1] + 1]
        y = np.r_[y, y[-1]]
    ax.step(x, y, where=where, label=label, **style)


# Common hourly axes keep Figure 3 onward comparable with the zero-based report index.
def hourly_axis(ax, ylabel="Energy per hour (kWh)"):
    ax.set(xlabel="Hour", ylabel=ylabel)
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.25)


# This shared base panel retains the report's reference / optimal / PV colour meanings.
def base_load(h, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    step(ax, h, "reference_load", "Reference profile", color="C0", linewidth=2)
    step(ax, h, "load", "Optimal load", color="C1", linewidth=2)
    ax.plot(
        h.index, h.pv_available, "--", color="gray", alpha=0.7, label="Available PV"
    )
    hourly_axis(ax)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(title)
    ax.legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Selected hours are supplied explicitly to prevent silently changing report examples.
def coefficient_response(runs, task, kind, hours):
    table = runs.table(task, "sweep_metrics.csv")
    key = "cL" if kind == "linear" else "cQ"
    fig, ax = plt.subplots(figsize=(10, 4))
    for t in hours:
        values = [runs.hourly(task, f"{key}_{x:g}").loc[t, "load"] for x in table[key]]
        ax.plot(table[key], values, ".-", label=f"Hour {t}")
    if kind == "quadratic":
        ax.set_xscale("log")
    ax.set(
        xlabel=f"{key} (" + ("DKK/kWh" if kind == "linear" else "DKK/kWh²") + ")",
        ylabel="Optimal hourly consumption (kWh)",
        title="Load response; exact thresholds may admit multiple optima",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Four Q1 panels retain the original report structure; duals use an explicit sign legend.
def _hour_reference_lines(ax, hours):
    """Original Q1 runner styling: one light vertical line per decision hour."""
    ax.set_xticks(hours)
    ax.grid(axis="x", color="0.85", ls=":", lw=0.7)
    ax.set_axisbelow(True)


def q1_overview(frame, data):
    """Four panels adapted from the user's plotting1.py; raw duals are freshly solved."""
    hr = frame
    h = hr.index.to_numpy()
    fig, (ax_limits, ax_price, ax_schedule, ax_duals) = plt.subplots(
        4, 1, sharex=True, figsize=(12, 12)
    )

    # 1. Electricity prices and the two Q1 marginal values.
    ax_price.step(h, data.energy_price, where="mid", color="k", label="energy price")
    ax_price.step(
        h,
        data.energy_price + data.import_tariff,
        where="mid",
        ls="--",
        label="price + import tariff",
    )
    ax_price.step(
        h,
        data.energy_price - data.export_tariff,
        where="mid",
        ls=":",
        label="price - export tariff",
    )
    if data.consumption_utility is not None:
        ax_price.axhline(
            data.consumption_utility,
            color="C2",
            ls="-.",
            lw=1.2,
            label=r"load utility $u_L$",
        )
    if data.pv_marginal_cost is not None:
        ax_price.axhline(
            data.pv_marginal_cost,
            color="C4",
            ls=(0, (3, 1, 1, 1)),
            lw=1.2,
            label=r"PV marginal cost $c^{\mathrm{PV}}$",
        )
    ax_price.set(ylabel="DKK/kWh", title="b) Electricity prices")
    ax_price.legend(fontsize=8, ncol=3)

    # 2. Available PV and the load bounds.
    ax_limits.fill_between(
        h,
        data.pv_available,
        step="mid",
        alpha=0.4,
        color="orange",
        label="PV available",
    )
    ax_limits.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax_limits.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    ax_limits.set(ylabel="kWh/h", title="a) PV and load preferences")
    ax_limits.legend(fontsize=8)

    # 3. Optimal schedule.
    width = 0.8
    ax_schedule.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    ax_schedule.bar(
        h,
        -hr["pv"],
        width,
        color="orange",
        alpha=0.7,
        label="PV used (negative = generation)",
    )
    ax_schedule.step(
        h,
        -hr["pv_available"],
        where="mid",
        color="orange",
        ls="--",
        lw=1,
        label="PV available",
    )
    ax_schedule.plot(
        h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)"
    )
    ax_schedule.axhline(0, color="grey", lw=0.8)
    ax_schedule.set(ylabel="kWh/h", title="c) Optimal schedule")
    ax_schedule_price = ax_schedule.twinx()
    ax_schedule_price.step(
        h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price"
    )
    ax_schedule_price.set_ylabel("DKK/kWh", color="C3")
    lines, labels = ax_schedule.get_legend_handles_labels()
    price_lines, price_labels = ax_schedule_price.get_legend_handles_labels()
    ax_schedule.legend(
        lines + price_lines,
        labels + price_labels,
        fontsize=8,
        ncol=3,
        loc="upper center",
    )

    # 4. Dual variables and the effective price signals.
    for column in [
        "dual_balance",
        "dual_pv_lower",
        "dual_pv_upper",
        "dual_load_lower",
        "dual_load_upper",
    ]:
        ax_duals.step(h, hr[column], where="mid", label=column.removeprefix("dual_"))
    ax_duals.step(
        h,
        data.energy_price + data.import_tariff,
        where="mid",
        color="grey",
        ls="--",
        lw=1,
        label="price + import tariff",
    )
    ax_duals.step(
        h,
        data.energy_price - data.export_tariff,
        where="mid",
        color="grey",
        ls=":",
        lw=1,
        label="price - export tariff",
    )
    ax_duals.set(xlabel="hour", ylabel="DKK/kWh", title="d) Dual variables")
    ax_duals.legend(fontsize=8, ncol=3)

    for ax in (ax_price, ax_limits, ax_schedule, ax_duals):
        _hour_reference_lines(ax, h)
        # Keep the shared x-axis aligned while showing every hourly label on every panel.
        ax.tick_params(axis="x", labelbottom=True)

    fig.suptitle(f"Q1 overview - {data.question}", fontsize=13)
    return fig


# Quadratic Figure 5 has load, procurement decisions, and effective-price panels.
def quadratic_base(h, total_cost):
    """Adapted directly from the user's run_q2c.py base-figure block."""
    d = load_question("Q2_quadratic")
    hours = d.hours
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    axes[0].step(
        hours, d.reference_load, where="mid", color="black", ls="--", label="Reference"
    )
    axes[0].step(hours, h["load"], where="mid", label="Optimal load")
    axes[0].step(hours, d.pv_available, where="mid", label="PV available", alpha=0.65)
    axes[0].set_ylabel("Energy (kWh / 1 h)")
    axes[0].legend(ncol=3)
    axes[1].bar(hours, h["import"], label="Import")
    axes[1].bar(hours, -h["export"], label="Export (negative)")
    axes[1].step(hours, h["pv"], where="mid", color="orange", label="PV produced")
    axes[1].set_ylabel("Energy (kWh / 1 h)")
    axes[1].legend(ncol=3)
    axes[2].step(
        hours, h["effective_import_price"], where="mid", label="Effective import price"
    )
    axes[2].step(
        hours, h["effective_export_price"], where="mid", label="Effective export price"
    )
    axes[2].axhline(
        d.pv_marginal_cost, color="purple", ls="--", label="PV marginal cost"
    )
    axes[2].set(xlabel="Hour", ylabel="DKK/kWh")
    axes[2].legend(ncol=3)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(hours)
    fig.suptitle(
        f"Q2(c) base case: cQ={d.quadratic_disutility:g} DKK/kWh²; total cost={total_cost:.3f} DKK"
    )
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


def quadratic_sweep(runs):
    """Original four-panel sweep and original representative hours."""
    d = load_question("Q2_quadratic")
    tab = runs.table("2c", "sweep_metrics.csv")
    qs = tab.cQ.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for t in [5, 8, 12, 18]:
        axes[0, 0].plot(
            qs,
            [runs.hourly("2c", f"cQ_{q:g}").loc[t, "load"] for q in qs],
            ".-",
            label=f"Hour {t}; ref={d.reference_load[t]:g}",
        )
    axes[0, 0].set_ylabel("Load (kWh / 1 h)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(qs, tab["daily_energy"], ".-", label="Optimal daily energy")
    axes[0, 1].axhline(
        d.reference_load.sum(), ls="--", color="black", label="Reference daily energy"
    )
    axes[0, 1].set_ylabel("Energy (kWh/day)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(qs, tab["absolute_deviation"], ".-", label="Absolute deviation")
    axes[1, 0].set_ylabel("Absolute deviation (kWh/day)")
    axes[1, 1].plot(qs, tab["procurement_cost"], ".-", label="Procurement cost")
    axes[1, 1].plot(qs, tab["disutility"], ".-", label="Disutility")
    axes[1, 1].set_ylabel("DKK/day")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("cQ (DKK/kWh²)")
        ax.grid(alpha=0.25)
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 7 and appendix Figure 26 consistently show quadratic blue / linear orange.
def model_comparison(linear, quad, two_panels=True):
    fig, axes = plt.subplots(
        2 if two_panels else 1, 1, figsize=(12, 9 if two_panels else 4.8)
    )
    axes = np.atleast_1d(axes)
    if two_panels:
        ax = axes[0]
        step(ax, linear, "reference_load", "Reference profile", color="C0", linewidth=2)
        step(ax, linear, "load", "Optimal load", color="C1", linewidth=2)
        ax.plot(
            linear.index,
            linear.pv_available,
            "--",
            color="gray",
            alpha=0.7,
            label="Available PV",
        )
        ax.set_title("Q2(b) base case: c_L = 1.43 DKK/kWh")
        ax.set_xticks(range(0, 24, 2))
    ax = axes[-1]
    step(ax, quad, "reference_load", "Reference", color="black", ls="--")
    step(ax, quad, "load", "Quadratic cQ=1", color="C0")
    step(ax, linear, "load", "Linear cL=1.43 (oracle verified)", color="C1")
    ax.set_title("Base-profile comparison; coefficients have different units")
    for ax in axes:
        hourly_axis(ax)
        ax.legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 8 uses the hours visible in each original report panel, not a common replacement set.
def combined_response(runs):
    """Recreate the report's two-panel comparison from Q2(d)'s freshly solved sweeps."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.82))
    summary = runs.table("2d", "summary.csv")
    for ax, kind, key, hours in [
        (axes[0], "linear", "cL", [7, 9, 11, 18]),
        (axes[1], "quadratic", "cQ", [5, 8, 12, 18]),
    ]:
        names = [n for n in summary.scenario if n.startswith("response_" + kind + "_")]
        names = sorted(names, key=lambda n: float(n.rsplit("_", 1)[1]))
        weights = [float(n.rsplit("_", 1)[1]) for n in names]
        for hour in hours:
            ax.plot(
                weights,
                [runs.hourly("2d", n).loc[hour, "load"] for n in names],
                ".-",
                label=f"Hour {hour}",
            )
        if kind == "quadratic":
            ax.set_xscale("log")
        ax.set(
            xlabel=key + " (" + ("DKK/kWh" if kind == "linear" else "DKK/kWh²") + ")",
            ylabel="Optimal hourly consumption (kWh)",
        )
        ax.set_title(
            "Load response; exact thresholds may admit multiple optima"
            if kind == "linear"
            else ""
        )
        ax.grid(alpha=0.25)
        ax.legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figures 9 and 10 are separate outputs, including available PV in the Figure 9 base panel.
def coupled_comparison(h, quad=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    step(ax, h, "reference_load", "Reference", color="black", ls="--")
    if quad is None:
        step(ax, h, "load", "Non-separable (cS=1)", color="C0")
        step(ax, h, "pv_available", "PV available", color="C1")
        ax.set_title("Q2(e) base case: non-separable disutility")
    else:
        step(
            ax,
            quad,
            "load",
            f"Quadratic (cQ=1), A={quad.absolute_deviation.sum():.2f}",
            color="C0",
        )
        step(
            ax,
            h,
            "load",
            f"Non-separable (cQ=1, cS=1), A={h.absolute_deviation.sum():.2f}",
            color="C1",
        )
        ax.set_title("Comparison: quadratic vs non-separable")
    hourly_axis(ax)
    ax.legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 11 changes only chronological price timing, retaining twelve low and high hours.
def timing_prices(price_profiles):
    fig, ax = plt.subplots(figsize=(11, 3))
    for label, key in [("Alternating", "alternating"), ("Block", "block")]:
        prices = price_profiles[key]
        ax.step(range(24), prices, where="mid", label=label)
    hourly_axis(ax, "Price (DKK/kWh)")
    ax.set_title("Price scenarios")
    ax.legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 12 restores the missing signed-deviation panel and the unconstrained load comparison.
def daily_base(h, b, mu):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 10))
    step(axes[0], h, "reference_load", "Reference", color="black", ls=":", where="post")
    step(axes[0], b, "load", "Q2(c)", color="C0", where="post")
    step(axes[0], h, "load", "Q3", color="C2", where="post")
    axes[0].set_title("(a) Consumption")
    axes[1].bar(
        h.index + 0.5, b.deviation, width=0.7, label="Q2(c)", color="C0", alpha=0.7
    )
    axes[1].bar(h.index + 0.5, h.deviation, width=0.7, label="Q3", color="C2")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("(b) Reference deviations")
    step(axes[2], h, "effective_import_price", "a_t", color="C0", where="post")
    step(axes[2], h, "effective_export_price", "b_t", color="C1", where="post")
    d = load_question("Q3")
    axes[2].axhline(d.pv_marginal_cost, color="C4", ls="--", label="c")
    axes[2].axhline(mu, color="gray", ls=":", label="mu")
    axes[2].set_title("(c) Prices and marginal values")
    for ax in axes:
        hourly_axis(ax)
        ax.legend(ncol=4, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
    axes[1].set_ylabel("Signed deviation (kWh)")
    axes[2].set_ylabel("Price (DKK/kWh)")
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 13 groups all physical flows by model in two panels, with negative export bars.
def daily_flows(h, b):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    for ax, r, label in [(axes[0], b, "Q2c"), (axes[1], h, "Q3")]:
        step(ax, r, "load", "L_t", color="black", where="post")
        step(ax, r, "pv", "P_t", color="orange", where="post")
        ax.bar(
            r.index + 0.5, r["import"], width=0.7, label="I_t", color="C0", alpha=0.6
        )
        ax.bar(
            r.index + 0.5, -r["export"], width=0.7, label="-E_t", color="C1", alpha=0.7
        )
        hourly_axis(ax)
        ax.set_title(label)
        ax.legend(ncol=4, loc="upper left")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
    limits = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Q3(f) figures retain the six report scenarios and matched unconstrained references.
def sensitivity_summary(table, experiment):
    rows = table[table.experiment == experiment]
    x = rows.parameter_value
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(x, rows.daily_energy, "o-", label="With daily requirement")
    axes[0, 0].plot(
        x, rows.unconstrained_daily_energy, "s--", label="Without requirement"
    )
    axes[0, 0].axhline(32.34, color="black", ls=":", label="Reference total")
    if experiment == "cq":
        axes[0, 0].axhline(30, color="gray", ls="-.", label="Daily minimum")
    axes[0, 0].set_ylabel("Daily energy (kWh)")
    for key, label in [
        ("procurement_cost", "Procurement"),
        ("disutility", "Disutility"),
        ("total_cost", "Total cost"),
    ]:
        axes[0, 1].plot(x, rows[key], "o-", label=label)
    axes[0, 1].set_ylabel("Cost (DKK)")
    axes[1, 0].plot(x, rows.mu, "o-")
    axes[1, 0].set_ylabel("Daily multiplier mu (DKK/kWh)")
    axes[1, 1].plot(x, rows.squared_deviation, "o-")
    axes[1, 1].set_ylabel("Unweighted squared deviation (kWh²)")
    for ax in axes.flat:
        ax.set_xlabel(
            "Minimum daily energy (kWh)"
            if experiment == "emin"
            else "Quadratic coefficient (DKK/kWh²)"
        )
        if experiment == "cq":
            ax.set_xscale("log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Q3(f): "
        + (
            "vary E_min; c_Q = 1"
            if experiment == "emin"
            else "vary c_Q; E_min = 30 kWh"
        )
    )
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# The same six-panel arrangement is used for minimum-energy and comfort-weight profiles.
def sensitivity_profiles(runs, table, experiment):
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 9))
    for ax, (_, row) in zip(
        axes.flat, table[table.experiment == experiment].iterrows()
    ):
        value = row.parameter_value
        h = runs.hourly("3f", f"{experiment}_{value:g}")
        b = runs.hourly("3f", f"unconstrained_cQ_{row.cQ:g}")
        step(ax, h, "reference_load", "Reference", color="black", ls=":")
        step(ax, b, "load", "Without requirement", color="C0", ls="--")
        step(ax, h, "load", "With requirement", color="C1")
        ax.set(
            title=f"{experiment} = {value:g}; mu = {row.mu:.3f}",
            xlabel="Hour",
            ylabel="Load (kWh / 1 h)",
        )
        ax.legend(fontsize=7)
    fig.suptitle("Q3(f): hourly load profiles")
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 18 plots stored energy at hour boundaries, including the initial stock at boundary 0.
"""Return a three-panel battery dispatch figure.

    Input: hourly solved results with energy in kWh, power in kW,
    and electricity prices in DKK/kWh.
    Output: a Matplotlib Figure showing prices, dispatch, and stored energy.
    """

def battery_dispatch(h):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    for key, label in [
        ("effective_import_price", "Import a_t"),
        ("effective_export_price", "Export b_t"),
    ]:
        step(axes[0], h, key, label, where="post")
    axes[0].set(title="(a) Effective prices", ylabel="DKK/kWh")
    axes[0].set_ylim(0, 3.15)
    axes[1].bar(h.index + 0.5, h.charge, width=0.65, label="Charge", color="#008773")
    axes[1].bar(
        h.index + 0.5,
        -h.discharge,
        width=0.65,
        label="Discharge (negative)",
        color="#CD642F",
    )
    for y in [-1.5, 0, 1.5]:
        axes[1].axhline(y, color="gray", ls=":" if y else "-", lw=1)
    axes[1].set(
        title="(b) Charging and discharging do not overlap", ylabel="Power (kW)"
    )
    axes[1].set_ylim(-1.9, 2)
    axes[2].step(
        range(25), np.r_[2, h.soc], where="post", color="#008773", label="Stored energy"
    )
    axes[2].axhline(4, color="black", ls=":", label="Capacity 4 kWh")
    axes[2].axhline(2, color="gray", ls="--", label="Initial 2 kWh")
    axes[2].set(
        title="(c) State of charge: storage limits are respected",
        xlabel="Elapsed time from start of day (h)",
        ylabel="Stored energy (kWh)",
        xlim=(0, 24),
    )
    axes[2].set_ylim(-0.1, 4.8)
    axes[2].set_xticks(range(0, 25, 4))
    for ax in axes:
        ax.legend(ncol=3, fontsize=8, loc="upper left")
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 19 preserves the 2x2 layout and the effective-price overlays on secondary axes.
def battery_flows(h, b):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(13, 8))
    for ax, key, title in zip(
        axes.flat,
        ["load", "pv", "import", "export"],
        [
            "(a) Consumption L_t",
            "(b) PV generation P_t",
            "(c) Grid imports I_t",
            "(d) Grid exports E_t",
        ],
    ):
        step(ax, b, key, "No battery", color="C0", where="post")
        step(
            ax, h, key, "Terminal-value battery", color="#008773", ls="--", where="post"
        )
        if key == "load":
            step(
                ax, h, "reference_load", "Reference", color="gray", ls=":", where="post"
            )
        right = ax.twinx()
        step(
            right,
            h,
            "effective_import_price",
            "Effective import price",
            color="#B59D69",
            ls=":",
            alpha=0.6,
            where="post",
        )
        step(
            right,
            h,
            "effective_export_price",
            "Effective export price",
            color="#B9678D",
            ls="-.",
            alpha=0.6,
            where="post",
        )
        right.set_ylabel("Effective price (DKK/kWh)")
        ax.set(title=title, xlabel="Hour", ylabel="Energy per hour (kWh)")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        right.set_ylim(0, 3.5)
        ax.legend(fontsize=7)
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 20 separately colors improvements and deterioration in hourly absolute deviation.
def battery_deviations(h, b):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    step(axes[0], b, "deviation", "No battery", color="C0", where="post")
    step(
        axes[0],
        h,
        "deviation",
        "Terminal-value battery",
        color="#008773",
        ls="--",
        where="post",
    )
    axes[0].set(
        title="(a) Hourly reference deviations", ylabel="Signed deviation (kWh)"
    )
    axes[0].legend()
    axes[0].set_ylim(-0.02, 0.65)
    diff = h.absolute_deviation - b.absolute_deviation
    axes[1].bar(
        h.index[diff < 0] + 0.5,
        diff[diff < 0],
        width=0.7,
        color="#008773",
        label="Smaller absolute deviation",
    )
    axes[1].bar(
        h.index[diff >= 0] + 0.5,
        diff[diff >= 0],
        width=0.7,
        color="#CD642F",
        label="Larger absolute deviation",
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        title="(b) Change in absolute deviation: improvement is not uniform across hours",
        xlabel="Hour",
        ylabel="Change in absolute deviation (kWh)",
    )
    axes[1].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figures 21/22/24/25 restore base-size markers, saturation annotations, and a line-style legend.
def sizing(table, plateaus, resource, all_alphas=False):
    key = resource
    marginal = "marginal_" + resource
    unit = "kWh" if resource == "capacity" else "kW"
    base = 4 if resource == "capacity" else 1.5
    groups = table if all_alphas else table[table.alpha == 1]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    right = axes[0].twinx()
    colors = {0: "gray", 0.5: "#467CAC", 1: "#258339", 1.5: "#C9667B"}
    for alpha, g in groups.groupby("alpha"):
        g = g.sort_values(key)
        color = colors[alpha] if all_alphas else "#252A31"
        axes[0].plot(
            g[key],
            g.inventory_adjusted_value,
            color=color,
            label=f"alpha={alpha:g}" if all_alphas else "Operating gain V_adj",
        )
        right.plot(
            g[key],
            g[marginal],
            color=color if all_alphas else "#2864B2",
            ls="--",
            label="Marginal gain",
        )
        positive = g[g[key] > 0]
        axes[1].plot(
            positive[key],
            positive.inventory_adjusted_value / positive[key],
            color=color if all_alphas else "#008773",
        )
    for ax in axes:
        ax.axvline(
            base, color="gray" if all_alphas else "#C58B8B", ls=":", label="Base case"
        )
    if not all_alphas:
        plateau = float(
            plateaus.loc[plateaus.alpha == 1, resource + "_plateau"].iloc[0]
        )
        for ax in axes:
            ax.axvline(plateau, color="gray", ls="-.", label="Saturation")
            ax.axvspan(plateau, groups[key].max(), color="gray", alpha=0.07)
        axes[0].annotate(
            f"Saturation\n{plateau:.2f} {unit}",
            xy=(plateau, groups.inventory_adjusted_value.max()),
            xytext=(8, -35),
            textcoords="offset points",
            color="gray",
        )
    axes[0].set_ylabel("Operating gain V_adj (DKK)")
    right.set_ylabel(f"Marginal gain (DKK/{unit})")
    axes[0].set_ylim(bottom=0)
    if not all_alphas:
        axes[0].set_ylim(0, 4.5 if resource == "capacity" else 5)
        right.set_ylim(0, 0.9 if resource == "capacity" else 6.2)
    axes[1].set(
        xlabel=f"{'Energy capacity' if resource=='capacity' else 'Common charge/discharge power'} ({unit})",
        ylabel=f"Gain per {unit} (DKK/{unit})",
    )
    lines, labels = axes[0].get_legend_handles_labels()
    rl, rb = right.get_legend_handles_labels()
    if all_alphas:
        # Two extra dummy lines make the solid/dashed meaning explicit without duplicating alpha labels.
        from matplotlib.lines import Line2D

        lines += [
            Line2D([], [], color="black", ls="-"),
            Line2D([], [], color="black", ls="--"),
        ]
        labels += ["Gain", "Marginal"]
    else:
        lines += rl
        labels += rb
    if all_alphas:
        axes[0].legend(
            lines,
            labels,
            ncol=6,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
        )
    else:
        axes[0].legend(lines, labels, ncol=2, fontsize=8, loc="upper left")
    axes[0].set_title(
        ("Fixed power: 1.5 kW" if resource == "capacity" else "Fixed capacity: 4 kWh")
        if all_alphas
        else (
            "Capacity sweep at 1.5 kW"
            if resource == "capacity"
            else "Common power sweep at 4 kWh"
        )
    )
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig


# Figure 23 combines the mean-preserving price profiles with both operating-value definitions.
def variability(table):
    prices = load_question("Q3_battery").energy_price
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for alpha, color in [
        (0, "gray"),
        (0.5, "#467CAC"),
        (1, "#258339"),
        (1.5, "#C9667B"),
    ]:
        values = prices.mean() + alpha * (prices - prices.mean())
        axes[0].step(
            range(25),
            np.r_[values, values[-1]],
            where="post",
            label=f"alpha={alpha:g}",
            color=color,
        )
    axes[0].set(
        title="(a) Market prices", xlabel="Hour", ylabel="Market price (DKK/kWh)"
    )
    axes[0].legend(ncol=4)
    axes[0].set_xlim(0, 24)
    axes[0].set_xticks(range(0, 25, 4))
    axes[1].plot(table.alpha, table.battery_value, "o-", label="V_battery")
    axes[1].plot(
        table.alpha, table.inventory_adjusted_value, "s--", label="V_adj", color="C2"
    )
    axes[1].set(
        title="(b) Battery value",
        xlabel="Price-spread factor alpha",
        ylabel="Battery value (DKK)",
    )
    axes[1].legend()
    for axis in fig.axes:
        axis.grid(alpha=0.25)
    return fig
