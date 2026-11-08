"""Q2 manager: one parameterized entry point, reusing the validated KKT builder.

Place in Q2_models/Q2_2; keep Q1_models next to Q2_models.
First replication test: MODE='TRANSFER', CAP_MULTIPLIER=1.0.
"""
import json
import os
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB, quicksum

# Change only these settings between experiments.
MODE = "BB"  # 'DP', 'BB', 'STRICT_IR', 'TRANSFER'
CAP_MULTIPLIER = 1.00 # 1.00, 0.90, 1.10
TIME_LIMIT = 180
MIP_GAP = 0.02
USE_WARM_START = True

THIS_DIR = Path(__file__).resolve().parent
Q2_DIR = THIS_DIR.parent
sys.path.insert(0, str(Q2_DIR / "Q2_1"))
from data_interface import load_q1_data, convert_member
from follower import solve_member
from kkt_member_builder import add_member_kkt


def load_solution_start(model, filename):
    """Use bare filename so Gurobi does not parse non-ASCII Windows paths."""
    if not (THIS_DIR / filename).is_file():
        print(f"Warm start skipped: {filename} not found")
        return
    model.update()
    previous = Path.cwd()
    try:
        os.chdir(THIS_DIR)
        model.read(filename)
    finally:
        os.chdir(previous)
    print(f"Warm start read: {filename}")


def main():
    mode = MODE.upper().strip()
    if mode not in {"DP", "BB", "STRICT_IR", "TRANSFER"}:
        raise ValueError("MODE must be DP, BB, STRICT_IR or TRANSFER")
    if CAP_MULTIPLIER <= 0:
        raise ValueError("CAP_MULTIPLIER must be positive")

    data = load_q1_data()
    T = range(24)
    wholesale = list(data.prices)
    beta = float(data.contract["tariff_discount_ratio"])
    alpha = float(data.contract["penalty_DKK_per_kWh"])
    caps = [CAP_MULTIPLIER * float(c) for c in data.contract["capacity_limit_kW"]]
    outside = json.loads((THIS_DIR / "outside_costs_R.json").read_text(encoding="utf-8"))
    assert set(outside) == {member.id for member in data.members}

    print("\n=== PARAMETERIZED Q2 MANAGER ===")
    print(f"MODE={mode}; CAP_MULTIPLIER={CAP_MULTIPLIER:.2f}; CAP_HOUR_00={caps[0]:.4f} kW")
    print(f"Reference R sum = {sum(outside.values()):.8f} DKK")
    model = gp.Model("Q2_manager_parameterized")
    model.Params.OutputFlag = 1
    model.Params.MIPGap = MIP_GAP
    model.Params.TimeLimit = TIME_LIMIT
    x = model.addVars(T, lb=-15.0, ub=15.0, name="internal_price")

    blocks = {}
    for member in data.members:
        blocks[member.id] = add_member_kkt(model, convert_member(member), x, member.id)

    imports = model.addVars(T, lb=0.0, name="feeder_import")
    exports = model.addVars(T, lb=0.0, name="feeder_export")
    excess = model.addVars(T, lb=0.0, name="capacity_excess")
    for t in T:
        model.addConstr(
            imports[t] - exports[t] == quicksum(blocks[i]["net"][t] for i in blocks),
            name=f"feeder_balance_{t}",
        )
        model.addConstr(imports[t] <= caps[t] + excess[t], name=f"soft_capacity_{t}")

    physical_cost = quicksum(blocks[i]["physical_cost"] for i in blocks)
    energy_cost = quicksum(wholesale[t] * (imports[t] - exports[t]) for t in T)
    tariff_cost = (1 - beta) * quicksum(
        data.tau_imp * imports[t] + data.tau_exp * exports[t] for t in T
    )
    penalty_cost = alpha * quicksum(excess[t] for t in T)
    external_cost = energy_cost + tariff_cost + penalty_cost
    revenue_linear = quicksum(blocks[i]["payment_linear"] for i in blocks)
    social_cost = physical_cost + external_cost

    if mode in {"BB", "STRICT_IR", "TRANSFER"}:
        model.addConstr(revenue_linear == external_cost, name="budget_balance")
    if mode == "STRICT_IR":
        for i in blocks:
            model.addConstr(
                blocks[i]["private_cost_linear"] <= outside[i],
                name=f"individual_rationality_{i}",
            )

    model.setObjective(social_cost, GRB.MINIMIZE)
    if USE_WARM_START and abs(CAP_MULTIPLIER - 1.0) < 1e-12:
        if mode in {"DP", "BB", "TRANSFER"}:
            load_solution_start(model, "dp_bb_start.sol")
        elif mode == "STRICT_IR":
            # A relaxed-IR solution is not necessarily a feasible strict-IR start.
            print("Strict IR: no feasible warm start is known")

    model.optimize()
    print("\n=== Q2 UNIFIED SOLVER STATUS ===")
    print(f"Status: {model.Status}; solutions: {model.SolCount}; runtime: {model.Runtime:.2f}s")
    if model.SolCount == 0:
        print("No incumbent found. This does NOT prove the strict IR model infeasible.")
        return
    print(f"Feasible social cost: {model.ObjVal:.8f} DKK")
    print(f"Best bound: {model.ObjBound:.8f} DKK; gap: {model.MIPGap:.4%}")
    print(f"Physical cost: {physical_cost.getValue():.8f} DKK")
    print(f"External cost: {external_cost.getValue():.8f} DKK")
    print(f"DSO penalty: {penalty_cost.getValue():.8f} DKK")
    print(f"Peak import: {max(imports[t].X for t in T):.4f} kW")

    direct_revenue = sum(x[t].X * sum(blocks[i]["net"][t].getValue() for i in blocks) for t in T)
    bb_residual = direct_revenue - external_cost.getValue()
    payment_error = direct_revenue - revenue_linear.getValue()
    print(f"BB residual (diagnostic for DP): {bb_residual:.3e} DKK")
    print(f"Payment linearization error: {payment_error:.3e} DKK")
    assert abs(payment_error) < 1e-5
    if mode != "DP":
        assert abs(bb_residual) < 1e-5

    margins = {i: outside[i] - blocks[i]["private_cost_linear"].getValue() for i in blocks}
    violators = [i for i, margin in margins.items() if margin < -1e-5]
    deficit = sum(max(0.0, -m) for m in margins.values())
    gains = sum(max(0.0, m) for m in margins.values())
    print(f"IR violators BEFORE transfer: {len(violators)} {violators}")
    print(f"Total compensation required: {deficit:.6f} DKK")
    print(f"Available positive gains: {gains:.6f} DKK")
    print(f"Total community saving relative to R: {sum(margins.values()):.6f} DKK")

    if mode == "STRICT_IR":
        assert not violators, "Strict IR incumbent violates an IR constraint"
    if mode == "TRANSFER":
        if gains + 1e-6 < deficit:
            print("TRANSFER NOT FEASIBLE: member gains cannot cover IR deficits")
        else:
            transfers = {
                i: (m if m < 0.0 else deficit * m / gains)
                for i, m in margins.items()
            }
            final_margins = {i: margins[i] - transfers[i] for i in blocks}
            assert abs(sum(transfers.values())) < 1e-5
            assert min(final_margins.values()) >= -1e-5
            print(f"Net transfers: {sum(transfers.values()):.3e} DKK")
            print(f"Members meeting IR AFTER transfers: {sum(v >= -1e-5 for v in final_margins.values())}/16")

    optimized_prices = [x[t].X for t in T]
    max_gap = 0.0
    for member in data.members:
        direct = solve_member(convert_member(member), optimized_prices, output=False)
        gap = abs(direct["objective"] - blocks[member.id]["private_cost_linear"].getValue())
        max_gap = max(max_gap, gap)
    print(f"Maximum independent follower optimality error: {max_gap:.3e} DKK")
    assert max_gap < 1e-5
    print("Q2 UNIFIED VALIDATION PASSED")


if __name__ == "__main__":
    main()
