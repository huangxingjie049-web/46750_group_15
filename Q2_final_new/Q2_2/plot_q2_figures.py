
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# 1. Paths
# =====================================================

HERE = Path(__file__).resolve().parent
Q2_ROOT = HERE.parent
PROJECT_ROOT = Q2_ROOT.parent

FIG_DIR = Q2_ROOT / "Q2figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    path = FIG_DIR / filename
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    print("Saved:", path)


# =====================================================
# 2. Verified Q2.5 sensitivity data
# =====================================================

multipliers = ["0.90", "1.00", "1.10"]


social_cost = [
    392.12875003,
    374.57864026,
    361.59306474
]

compensation = [
    94.628706,
    81.619893,
    45.174571
]



# =====================================================
# Figure 1: Community social cost
# =====================================================

fig, ax = plt.subplots(figsize=(7.2, 4.3))

ax.plot(
    multipliers,
    social_cost,
    marker="o",
    linewidth=2.2
)

for i, value in enumerate(social_cost):
    ax.annotate(
        f"{value:.2f}",
        (i, value),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center"
    )

ax.set_title(
    "Community social cost vs capacity multiplier"
)
ax.set_xlabel("Contract capacity multiplier")
ax.set_ylabel("Community social cost (DKK)")
ax.grid(alpha=0.25)
ax.margins(y=0.2)

fig.tight_layout()

save_figure(
    fig,
    "Q2_5_social_cost_vs_capacity.png"
)


# =====================================================
# Figure 2: IR compensation requirement
# =====================================================

fig, ax = plt.subplots(figsize=(7.2, 4.3))

ax.plot(
    multipliers,
    compensation,
    marker="o",
    linewidth=2.2
)

for i, value in enumerate(compensation):
    ax.annotate(
        f"{value:.2f}",
        (i, value),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center"
    )

ax.set_title(
    "IR compensation required vs capacity multiplier"
)
ax.set_xlabel("Contract capacity multiplier")
ax.set_ylabel("Total compensation required (DKK)")
ax.grid(alpha=0.25)
ax.margins(y=0.2)

fig.tight_layout()

save_figure(
    fig,
    "Q2_5_compensation_vs_capacity.png"
)


# =====================================================
# 3. Q2.3 feeder response comparison
# =====================================================

# Read updated Q1 C dispatch
c_file = (
    PROJECT_ROOT
    / "Assignment_2_Q1_model_new"
    / "Q1_models"
    / "results"
    / "base_case"
    / "C"
    / "hourly.csv"
)


if not c_file.exists():
    raise FileNotFoundError(
        f"Missing Q1 C hourly.csv: {c_file}"
    )

with c_file.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

by_hour = {
    int(r["hour"]): r
    for r in rows
}

if len(rows) != 24 or set(by_hour) != set(range(24)):
    raise ValueError("Expected Q1 C hours 0-23")


hours = list(range(24))

c_net = [
    float(by_hour[t]["import_kWh"])
    - float(by_hour[t]["export_kWh"])
    for t in hours
]

capacity = [
    float(by_hour[t]["cap_kW"])
    for t in hours
]


# Independent follower response:
# copied from the verified Q2.3B output.
# This is ONE optimal-response selection,
# NOT the mathematical worst-case response.


independent_net = [
    6.000, 10.000, 12.120, 8.120,
    6.000, 12.341, 17.760, 26.000,
    9.200, 6.720, 8.560, 5.600,
    0.000, 7.440, 7.600, 8.880,
    4.728, 8.640, -5.280, -12.000,
    0.000, 0.000, 2.4211, 12.000
]



# =====================================================
# Figure 3: Feeder net exchange
# =====================================================

fig, ax = plt.subplots(figsize=(10, 4.8))

ax.plot(
    hours,
    c_net,
    marker="o",
    markersize=3.5,
    linewidth=1.8,
    label="Q1 C centralized net exchange"
)

ax.plot(
    hours,
    independent_net,
    marker="s",
    markersize=3.5,
    linewidth=1.8,
    label="Independent optimal responses"
)

ax.plot(
    hours,
    capacity,
    linestyle="--",
    linewidth=1.5,
    label="Contract import capacity"
)

ax.axhline(0, color="gray", linewidth=0.9)

ax.set_title(
    "Q1 C vs independent follower-optimal response"
)
ax.set_xlabel("Hour")
ax.set_ylabel("Feeder net exchange (average kW)")
ax.set_xticks(hours)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()

save_figure(
    fig,
    "Q2_3_feeder_net_exchange_comparison.png"
)


print("\nQ2 FIGURES COMPLETE")
print("Output folder:", FIG_DIR)
print("Three PNG figures saved successfully.")
