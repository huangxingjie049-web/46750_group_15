"""Figures of Question 3 (matplotlib, PDF + PNG). Every figure reads the files written by the experiments, so it can
be regenerated without solving any model: ``python Q3_final/run_q3.py figures``.

Colours: one fixed hue per model across all figures (validated categorical palette, slots 1-6, fixed order); the cap,
reference levels and theta = 1 are neutral ink, never a series colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import Q3_DIR, RESULTS_DIR

FIG_DIR = Q3_DIR / "figures"
COLORS = {"DP": "#2a78d6", "DP_BB": "#eb6834", "SDP_E": "#1baf7a", "SDP_CVaR": "#eda100", "SDP_CC": "#e87ba4",
          "SDP_E_cap": "#008300"}
LABELS = {"DP": "DP", "DP_BB": r"$\mathrm{DP}^{\mathrm{BB}}$", "SDP_E": r"$\mathrm{SDP}^{\mathrm{E}}$",
          "SDP_CVaR": r"$\mathrm{SDP}^{\mathrm{CVaR}}$", "SDP_CC": r"$\mathrm{SDP}^{\mathrm{CC}}$",
          "SDP_E_cap": r"$\mathrm{SDP}^{\mathrm{E}}$ (cap only)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
CAP_KW = 9.39
SDP = RESULTS_DIR / "sdp"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out: Path, tight: bool = False) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(bbox_inches="tight", pad_inches=0.03) if tight else {}
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, **extra)
    return out.with_suffix(".pdf")


def run_key(folder: Path) -> str | None:
    """Colour/label key of a run folder from its solve.json (None if the run does not exist)."""
    f = folder / "solve.json"
    if not f.exists():
        return None
    cfg = json.loads(f.read_text(encoding="utf-8"))["config"]
    n = json.loads(f.read_text(encoding="utf-8"))["N"]
    if cfg["risk"] == "cvar":
        return "SDP_CVaR"
    if cfg["cap"] == "chance":
        return "SDP_CC"
    if n == 1:
        return "DP" if cfg["bb"] == "none" else "DP_BB"
    return "SDP_E" if cfg["bb"] != "none" else "SDP_E_cap"


def _hourly(folder: Path, set_name: str) -> np.ndarray:
    df = pd.read_csv(folder / f"eval_{set_name}" / "hourly_import_optimistic.csv")
    return df[[f"h{t:02d}" for t in range(24)]].to_numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Section 3.1: where the deterministic baseline fails
# ---------------------------------------------------------------------------------------------------------------------
def baseline_imports(runs: list[tuple[str, Path, str]], out: Path = FIG_DIR / "q3_baseline_imports") -> Path:
    """Hourly community import of the deterministic baselines over the 100 out-of-sample scenarios
    (median and 5-95 % band) against the cap and the theta = 1 import (Section 3.1). Labels: share of days on which
    the hour is above the cap (shown when > 20 %). The axis is zoomed on the cap; evening imports (0-2.4 kW) lie below."""
    plt = _plt()
    hours = np.arange(24)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 2.9), sharey=True, layout="constrained")
    for ax, (label, folder, key) in zip(np.atleast_1d(axes), runs):
        color = COLORS[key]
        oos = _hourly(folder, "oos")
        theta1 = _hourly(folder, "expected")[0]
        q05, q50, q95 = np.quantile(oos, [0.05, 0.5, 0.95], axis=0)
        share = (oos > CAP_KW + 1e-6).mean(axis=0)
        ax.fill_between(hours, q05, q95, step="mid", color=color, alpha=0.22, linewidth=0, label="5–95 % of days")
        ax.step(hours, q50, where="mid", color=color, linewidth=2, label="median, out of sample")
        ax.step(hours, theta1, where="mid", color=INK, linewidth=1, linestyle=(0, (1, 1.5)), label=r"$\theta=1$")
        ax.axhline(CAP_KW, color=INK, linewidth=1, linestyle="--")
        ax.text(23.4, CAP_KW + 0.25, "cap 9.39 kW", ha="right", va="bottom", fontsize=7.5, color=INK)
        for t in np.flatnonzero(share > 0.2):
            ax.text(t, min(q95[t], 10.6) + 0.06, f"{share[t]:.0%}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(20.2, 8.12, "evening:\n0–2.4 kW", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_ylim(8.0, 10.9)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("hour", fontsize=8, color=MUTED)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.6, 23.6)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("community import (kW)", fontsize=8, color=MUTED)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    path = _save(fig, out)
    plt.close(fig)
    return path


def default_baseline_runs() -> list[tuple[str, Path, str]]:
    """(panel title, run folder written by ``run_q3.py sdp --set expected``, colour key)."""
    return [("DP", SDP / "SDP_E_expected_bb-none", "DP"),
            (r"$\mathrm{DP}^{\mathrm{BB}}$", SDP / "SDP_E_expected_bb-expectation", "DP_BB")]


# ---------------------------------------------------------------------------------------------------------------------
# Section 3.2 / 3.4: prices
# ---------------------------------------------------------------------------------------------------------------------
LEVELS = {1.41: r"$c^{PV}$", 1.91: "L1", 2.41: "L2", 3.41: "L3", 5.41: "L4"}
# N = 5, budget balance, penalty 10: price vectors returned by different runs (archived in data/n5_vectors).
# A, B, C have the same in-sample cost in every scenario; D is the best solution found.
N5_STYLES = {"A": ((0, (5, 1.5)), 1.5, INK), "B": ((0, (2, 1.5)), 1.5, MUTED), "C": ((0, (1, 1.2)), 1.5, MUTED),
             "D": ("-", 2.2, None)}


# report names of the archived vectors (the letters A-D of the files would clash with model C of Question 1)
N5_REPORT = {"A": r"$x^{(1)}$", "B": r"$x^{(2)}$", "C": r"$x^{(3)}$", "D": r"$x^{(4)}$"}


def n5_vectors() -> list[tuple[str, Path]]:
    from .paths import Q3_DATA_DIR
    return [(f.stem, f) for f in sorted((Q3_DATA_DIR / "n5_vectors").glob("*.json"))]


def _prices_of(folder: Path) -> np.ndarray:
    return np.array(json.loads((folder / "prices.json").read_text(encoding="utf-8"))["prices_DKK_per_kWh"])


def _price_axis(ax, ymax: float, hours=(0, 23)):
    for lv, name in LEVELS.items():
        ax.axhline(lv, color=GRID, linewidth=0.9, zorder=0)
        ax.text(hours[1] + 0.75, lv, name, fontsize=6.5, color=MUTED, va="center", ha="left")
    ax.set_ylim(1.0, ymax)
    ax.set_xlim(hours[0] - 0.6, hours[1] + 0.6)
    ax.set_xlabel("hour", fontsize=8, color=MUTED)
    _style(ax)


def _clipped_labels(ax, hours, series, ymax):
    """Values above ymax: one label per series, stacked so that equal hours do not overlap."""
    stack = {}
    for x, color in series:
        for t in np.flatnonzero(x > ymax):
            k = stack.get(t, 0)
            ax.text(t - 0.65, ymax - 0.12 - 0.42 * k, f"{x[t]:.2f}", ha="right", va="top", fontsize=6.3, color=color)
            stack[t] = k + 1


def prices(runs: list[Path], out: Path = FIG_DIR / "q3_prices", ymax: float = 6.0, sdp: Path | None = None) -> Path:
    """(a) Internal prices of the given models with the indifference levels c_PV + 2 c^L_k of the flexibility classes.
    (b) if the N = 5 runs exist: the three price vectors that are equivalent in sample (A, B, C), hours 12-23."""
    sdp = sdp or SDP
    plt = _plt()
    eq = n5_vectors() if sdp != Path("-") else []
    two = len(eq) >= 2
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(7.2 if two else 6.6, 2.9), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6, 1]} if two else None)
    ax = axes[0] if two else axes
    hours = np.arange(24)
    drawn = []
    for k, folder in enumerate(runs):
        key = run_key(folder)
        x = _prices_of(folder)
        lab = LABELS[key] + (f", N = {_n_of(folder)}" if key.startswith("SDP") else "")
        ax.step(hours, np.minimum(x, ymax), where="mid", color=COLORS[key], linewidth=2.0 - 0.4 * k, label=lab)
        drawn.append((x, COLORS[key]))
    _clipped_labels(ax, hours, drawn, ymax)
    _price_axis(ax, ymax)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylabel("internal price (DKK/kWh)", fontsize=8, color=MUTED)
    if two:
        ax.set_title("(a) deterministic and stochastic prices", fontsize=8, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    if two:
        ax2 = axes[1]
        h = np.arange(12, 24)
        drawn = []
        for name, f in eq:
            x = np.array(json.loads(f.read_text(encoding="utf-8"))["prices_DKK_per_kWh"])[12:]
            ls, lw, col = N5_STYLES.get(name, ("-", 1.2, MUTED))
            ax2.step(h, np.minimum(x, ymax), where="mid", color=col or COLORS["SDP_E"], linestyle=ls, linewidth=lw,
                     label=N5_REPORT.get(name, name), zorder=3 if name == "D" else 2)
            drawn.append((x, col or COLORS["SDP_E"]))
        clipped = [x for x, _ in drawn if x.max() > ymax]
        if clipped:                                   # one label for the (common) value above the axis
            t = int(np.argmax(clipped[0]))
            lo, hi = min(x[t] for x in clipped), max(x[t] for x in clipped)
            txt = f"{lo:.2f}" if hi - lo < 5e-3 else f"{lo:.2f}-{hi:.2f}"
            ax2.text(12 + t - 0.65, ymax - 0.12, txt, ha="right", va="top", fontsize=6.3, color=MUTED)
        for t0, t1 in ((14.5, 17.5), (20.5, 21.5)):
            ax2.axvspan(t0, t1, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        _price_axis(ax2, ymax, hours=(12, 23))
        ax2.set_xticks(range(12, 24, 3))
        ax2.set_title("(b) N = 5: solutions of different runs", fontsize=8, color=INK, loc="left")
        ax2.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=2, handlelength=2.6)
    path = _save(fig, out)
    plt.close(fig)
    return path


def _n_of(folder: Path) -> int:
    return int(json.loads((folder / "solve.json").read_text(encoding="utf-8"))["N"])


def regret_distribution(runs: list[tuple[str, Path, str, object]], out: Path = FIG_DIR / "q3_regret_ecdf") -> Path:
    """Empirical distribution (1 - ECDF) of the out-of-sample regret: share of days on which the regret exceeds r.
    ``runs``: (label, run folder, colour key, line style)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 2.9), layout="constrained")
    for lab, folder, key, ls in runs:
        r = np.sort(pd.read_csv(folder / "eval_oos" / "metrics_optimistic.csv").regret.to_numpy())
        share = 1.0 - np.arange(1, len(r) + 1) / len(r)
        ax.step(np.maximum(r, 1e-2), share, where="post", color=COLORS[key], linestyle=ls, linewidth=1.7, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("regret r (DKK, log scale)", fontsize=8, color=MUTED)
    ax.set_ylabel("share of days with regret > r", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02)
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    path = _save(fig, out)
    plt.close(fig)
    return path


def default_regret_runs(sdp: Path | None = None) -> list[tuple[str, Path, str, object]]:
    sdp = sdp or SDP
    return [("DP", sdp / "SDP_E_expected_bb-none", "DP", "-"),
            (LABELS["DP_BB"], sdp / "SDP_E_expected_bb-expectation", "DP_BB", "-"),
            (LABELS["SDP_E"] + ", N = 5 and 10", sdp / "SDP_E_in10_bb-expectation", "SDP_E", "-")]


def regret_by_deviation(runs: list[tuple[str, Path]] | None = None,
                        out: Path = RESULTS_DIR / "regret_by_deviation.csv",
                        edges=(-15.0, -5.0, 0.0, 5.0, 10.0, 15.0)) -> Path:
    """Section 3.5: out-of-sample regret and exceedance frequency grouped by the deviation of the community reference
    energy of each day from its expected value (theta = 1). A proxy for the level of member uncertainty, computed from
    saved evaluations only (no model is solved)."""
    from .paths import SCENARIO_DIR
    runs = runs or [("DP", SDP / "SDP_E_expected_bb-none"), ("DP_BB", SDP / "SDP_E_expected_bb-expectation"),
                    ("SDP_E", SDP / "SDP_E_in10_bb-expectation")]
    from .scenarios import expected_scenario
    oos = json.loads((SCENARIO_DIR / "oos.json").read_text(encoding="utf-8"))
    total = lambda e: float(sum(e.values())) if isinstance(e, dict) else float(e)
    energy = pd.Series([total(s["daily_reference_energy_kWh"]) for s in oos["scenarios"]], name="energy")
    e0 = 0.0                                           # community reference energy at theta = 1
    for member in expected_scenario()["members"].values():
        load = next(a for a in member.values() if "load_id" in a)
        e0 += load["max_load_KWh_per_hour"] * sum(load["reference_load_capacity_factor"])
    d = pd.DataFrame({"deviation_kWh": energy - e0})
    d["bin"] = pd.cut(d.deviation_kWh, list(edges))
    agg = {"days": ("deviation_kWh", "size")}
    for key, folder in runs:
        m = pd.read_csv(folder / "eval_oos" / "metrics_optimistic.csv").set_index("scenario").sort_index()
        d[f"regret_{key}"] = m.regret.to_numpy()
        d[f"exceed_{key}"] = (m.exceedance_kWh > 1e-6).to_numpy()
        agg[f"regret_{key}"] = (f"regret_{key}", "mean")
        agg[f"share_exceed_{key}"] = (f"exceed_{key}", "mean")
    table = d.groupby("bin", observed=True).agg(**agg).reset_index()
    table["bin"] = table["bin"].astype(str)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Section 3.3 / 3.4: penalty level, chance constraint and number of scenarios
# ---------------------------------------------------------------------------------------------------------------------
def penalty_study(summary: pd.DataFrame, out: Path = FIG_DIR / "q3_penalty") -> Path:
    """(a) In-sample expected cost of SDP^CC against epsilon at penalty 3 (SDP^E drawn at its in-sample exceedance
    frequency). (b) Targeted (in sample) against achieved (out of sample) share of days with exceedance; points above
    the diagonal break the in-sample bound. (c) Out-of-sample cost against exceedance frequency at penalty 3."""
    plt = _plt()
    d = summary[(summary.bb == "expectation") & summary.oos_share_exceed.notna()].copy()
    cc = d[d.epsilon.notna()]
    neu = d[d.epsilon.isna() & (d.risk == "neutral") & (d.N > 1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), layout="constrained")
    c3 = cc[(cc.penalty == 3) & (cc.N == 5)].sort_values("epsilon")
    e3 = neu[(neu.penalty == 3) & (neu.N == 5)]
    ax = axes[0]
    ax.plot(c3.epsilon, c3.expected_cost_in, marker="o", color=COLORS["SDP_CC"], label=LABELS["SDP_CC"])
    if len(e3):
        ax.plot(e3.in_share_exceed, e3.expected_cost_in, marker="D", linestyle="none", color=COLORS["SDP_E"],
                label=LABELS["SDP_E"])
    ax.set_xlabel(r"$\varepsilon$ / in-sample share with exceedance", fontsize=7.5, color=MUTED)
    ax.set_ylabel("in-sample expected cost (DKK)", fontsize=8, color=MUTED)
    ax.set_title(r"(a) penalty 3, N = 5", fontsize=8, color=INK, loc="left")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
    # drawn largest first so that coinciding points stay visible
    for (pen, n), mk, ms, ls, lab in (((10, 5), "s", 9, "--", "penalty 10, N = 5"),
                                      ((10, 10), "^", 8, ":", "penalty 10, N = 10"),
                                      ((3, 5), "o", 4.5, "-", "penalty 3, N = 5")):
        g = cc[(cc.penalty == pen) & (cc.N == n)].sort_values("epsilon")
        if len(g):
            ax.plot(g.epsilon, g.oos_share_exceed, marker=mk, markersize=ms, color=COLORS["SDP_CC"], linewidth=1.0,
                    linestyle=ls, markerfacecolor=COLORS["SDP_CC"] if pen == 3 else "white",
                    label=LABELS["SDP_CC"] + ", " + lab)
    if len(e3):
        ax.plot(e3.in_share_exceed, e3.oos_share_exceed, marker="D", linestyle="none", color=COLORS["SDP_E"],
                label=LABELS["SDP_E"] + ", penalty 3")
    ax.set_xlim(-0.04, 1.0)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel(r"targeted: $\varepsilon$ (in sample)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("achieved: share of days, out of sample", fontsize=7.5, color=MUTED)
    ax.set_title("(b) exceedance frequency", fontsize=8, color=INK, loc="left")
    ax = axes[2]
    if len(c3):
        ax.plot(c3.oos_share_exceed, c3.oos_J_mean, marker="o", linestyle="none", color=COLORS["SDP_CC"])
        groups = c3.groupby([c3.oos_share_exceed.round(3), c3.oos_J_mean.round(2)])
        for (x, y), g in groups:
            eps = ", ".join(f"{e:g}" for e in sorted(g.epsilon))
            ax.annotate(rf"$\varepsilon$ = {eps}", (x, y), textcoords="offset points", xytext=(6, -2), fontsize=6.3,
                        color=MUTED)
    if len(e3):
        ax.plot(e3.oos_share_exceed, e3.oos_J_mean, marker="D", linestyle="none", color=COLORS["SDP_E"])
    ax.set_xlabel("share of days with exceedance", fontsize=7.5, color=MUTED)
    ax.set_ylabel("mean cost, out of sample (DKK)", fontsize=7.5, color=MUTED)
    ax.set_title("(c) penalty 3, out of sample", fontsize=8, color=INK, loc="left")
    for ax in axes:
        _style(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    if not any("SDP}^{\\mathrm{E}}" in l for l in labels):
        handles, labels = handles + h0[1:], labels + l0[1:]
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False)
    path = _save(fig, out)
    plt.close(fig)
    return path


def n_study(summary: pd.DataFrame, out: Path = FIG_DIR / "q3_n_study", sdp: Path | None = None) -> Path:
    """Risk-neutral model against the number of in-sample scenarios N (1 = deterministic baseline), penalty 10:
    out-of-sample mean regret and regret CVaR_0.9 (with budget balance, the bar at N = 5 spans the archived solutions
    A-D of results/selection_in5_on_in10.csv) and solution time (hollow: time limit reached)."""
    sdp = sdp or SDP
    plt = _plt()
    d = summary[(summary.risk == "neutral") & summary.epsilon.isna() & (summary.penalty == 10)].sort_values("N")
    from .paths import RESULTS_DIR
    sel_file = RESULTS_DIR / "selection_in5_on_in10.csv"
    sel = pd.read_csv(sel_file) if sel_file.exists() else pd.DataFrame()
    have_range = {"oos_regret_mean", "oos_regret_cvar90_upper"} <= set(sel.columns)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    for bb, key, style in (("expectation", "SDP_E", "-"), ("none", "SDP_E_cap", "--")):
        g = d[d.bb == bb]
        if g.empty:
            continue
        c = COLORS[key]
        lab = "budget balance" if bb == "expectation" else "cap only"
        for ax, col in ((axes[0], "oos_regret_mean"), (axes[1], "oos_regret_cvar90_upper")):
            ax.plot(g.N, g[col], marker="o", color=c, linestyle=style, label=f"{lab}, out of sample")
            if bb == "expectation" and have_range:      # N = 5: spread over the archived solutions x^(1)-x^(4)
                vals = sel[col].to_numpy()
                xb = 5.35                                # beside the curve markers, so that nothing overlaps
                ax.vlines(xb, vals.min(), vals.max(), color=c, linewidth=1.0)
                for name, v in zip(sel.candidate, vals):
                    ax.plot(xb, v, marker="_", color=c, markersize=8)
                    if ax is axes[0]:                    # labels once; the order is the same in both panels
                        ax.annotate(N5_REPORT.get(name, name), (xb, v), textcoords="offset points", xytext=(5, -2),
                                    fontsize=6.3, color=MUTED,
                                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        t = g.runtime_s
        axes[2].plot(g.N, t, color=c, linestyle=style, label=lab)
        for n, v, st in zip(g.N, t, g.status):
            axes[2].plot(n, v, marker="o", color=c, markerfacecolor=c if st == 2 else "white")
    axes[0].set_ylabel("mean regret (DKK)", fontsize=7.5, color=MUTED)
    axes[1].set_ylabel(r"regret $\mathrm{CVaR}_{0.9}$ (DKK)", fontsize=7.5, color=MUTED)
    axes[2].set_ylabel("solution time (s, log)", fontsize=7.5, color=MUTED)
    axes[2].set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axes[2].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.set_xlabel("in-sample scenarios N", fontsize=8, color=MUTED)
        ax.set_xticks(sorted(d.N.unique()))
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=6.8, frameon=False)
    path = _save(fig, out, tight=True)                  # keeps the left axis label inside the page
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Section 3.4: accuracy against performance over nested N (run_q3.py nstudy)
# ---------------------------------------------------------------------------------------------------------------------
def n_timing(summary_csv: Path, out: Path = FIG_DIR / "q3_n_timing") -> Path:
    """Same start, time limit and gap for every N. (a) In-sample expected cost and out-of-sample mean cost.
    (b) Out-of-sample mean regret and regret CVaR_0.9. (c) Solution time (phase 1 + MILP; hollow: time limit
    reached, labelled with the remaining gap). Grey diamonds: best known solutions of the main runs (pooled
    multi-start) for N = 5 and 10; dotted: the DP^BB baseline out of sample."""
    plt = _plt()
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
    d = pd.read_csv(summary_csv).sort_values("N")
    ref = SDP / "SDP_E_expected_bb-expectation" / "eval_oos" / "metrics_optimistic.csv"
    base = pd.read_csv(ref) if ref.exists() else None
    best = None
    summ = RESULTS_DIR / "q3_summary.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        s = s[s.run.isin(["SDP_E_in5_bb-expectation", "SDP_E_in10_bb-expectation"])].copy()
        if len(s):
            s["N"] = s.run.map({"SDP_E_in5_bb-expectation": 5, "SDP_E_in10_bb-expectation": 10})
            best = s.sort_values("N")
    c, cb, cg = COLORS["SDP_E"], COLORS["DP_BB"], MUTED
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    ax = axes[0]
    ax.plot(d.N, d.objective, marker="s", color=c, linestyle="--", label="in sample (model)")
    ax.plot(d.N, d.oos_J_mean, marker="o", color=c, label="out of sample")
    if best is not None:
        ax.plot(best.N, best.objective, marker="D", linestyle="none", color=cg, markerfacecolor="white",
                label="best known, in sample")
        ax.plot(best.N, best.oos_J_mean, marker="D", linestyle="none", color=cg, label="best known, out of sample")
    if base is not None:
        ax.axhline(base.J.mean(), color=cb, linewidth=1.0, linestyle=":", label=LABELS["DP_BB"] + ", out of sample")
    ax.set_ylabel("expected cost (DKK)", fontsize=7.5, color=MUTED)
    ax = axes[1]
    ax.plot(d.N, d.oos_regret_mean, marker="o", color=c, label="mean")
    ax.plot(d.N, d.oos_regret_cvar90, marker="s", color=c, linestyle="--", label=r"$\mathrm{CVaR}_{0.9}$")
    if best is not None:
        ax.plot(best.N, best.oos_regret_mean, marker="D", linestyle="none", color=cg)
        ax.plot(best.N, best.oos_regret_cvar90_upper, marker="D", linestyle="none", color=cg)
    if base is not None:
        ax.axhline(base.regret.mean(), color=cb, linewidth=1.0, linestyle=":")
    ax.set_ylabel("out-of-sample regret (DKK)", fontsize=7.5, color=MUTED)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax = axes[2]
    ax.plot(d.N, d.total_s, color=c)
    for n, v, lim in zip(d.N, d.total_s, d.time_limit_reached):
        ax.plot(n, v, marker="o", color=c, markerfacecolor="white" if lim else c)
    hit = d[d.time_limit_reached.astype(bool)]
    if len(hit):
        gaps = ", ".join(f"{100 * g:.2f}%" for g in hit.gap)
        lim_s = round(float(hit.milp_s.max()), -2)          # the MILP time limit
        ax.text(0.97, 0.04, f"hollow: stopped after {lim_s:,.0f} s\nat gap {gaps}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.3, color=MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("solution time (s, log)", fontsize=7.5, color=MUTED)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(d.N.tolist()))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("in-sample scenarios N", fontsize=8, color=MUTED)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
    path = _save(fig, out, tight=True)
    plt.close(fig)
    return path
