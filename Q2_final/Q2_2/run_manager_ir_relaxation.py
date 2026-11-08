
from pathlib import Path
import sys
import json
import os

import gurobipy as gp
from gurobipy import GRB, quicksum

Q2_ROOT = Path(__file__).resolve().parent.parent
Q2_1_DIR = Q2_ROOT / "Q2_1"

sys.path.insert(0, str(Q2_1_DIR))

from data_interface import load_q1_data, convert_member
from follower import solve_member
from kkt_member_builder import add_member_kkt

from community_response import aggregate_response
from manager_accounting import calculate_manager_accounting


def main():
    data = load_q1_data()

    outside_file = (
        Path(__file__).resolve().parent
        / "outside_costs_R.json"
    )

    outside_costs = json.loads(
        outside_file.read_text(encoding="utf-8")
    )

    T = range(24)

    prices = list(data.prices)
    contract = data.contract

    beta = float(contract["tariff_discount_ratio"])
    alpha = float(contract["penalty_DKK_per_kWh"])
    caps = list(contract["capacity_limit_kW"])

    # Original independent follower reference
    response = aggregate_response(data, prices)
    accounting = calculate_manager_accounting(
        data, response
    )

    model = gp.Model("manager_fixed_price_KKT")
    model.Params.OutputFlag = 1

    # Accept a solution within 2% of the best bound.
    model.Params.MIPGap = 0.001
    model.Params.MIPGapAbs = 1e-5
    # Stop after at most 180 seconds.
    model.Params.TimeLimit = 120
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.5

    # Prices are variables but fixed for this test.
    # Next stage: remove the equalities and optimize them.
    # Preliminary price bounds for the DP experiment.
    # These are numerical bounds, not contract terms.
    PRICE_LO = -15.0
    PRICE_HI = 15.0

    x = model.addVars(
        T,
        lb=PRICE_LO,
        ub=PRICE_HI,
        name="internal_price"
    )

    # Add all 16 follower KKT systems
    blocks = {}

    for source in data.members:
        member = convert_member(source)
        blocks[source.id] = add_member_kkt(
            model, member, x, source.id
        )

    # Community connection point
    imports = model.addVars(
        T, lb=0, name="feeder_import"
    )
    exports = model.addVars(
        T, lb=0, name="feeder_export"
    )
    excess = model.addVars(
        T, lb=0, name="capacity_excess"
    )

    for t in T:
        total_net = quicksum(
            blocks[i]["net"][t] for i in blocks
        )

        model.addConstr(
            imports[t] - exports[t] == total_net,
            name=f"feeder_balance_{t}"
        )

        model.addConstr(
            imports[t] <= caps[t] + excess[t],
            name=f"soft_capacity_{t}"
        )

    # Community physical costs
    physical_cost = quicksum(
        blocks[i]["physical_cost"]
        for i in blocks
    )

    # External wholesale, tariff and DSO costs
    energy_cost = quicksum(
        prices[t] * (imports[t] - exports[t])
        for t in T
    )

    tariff_cost = quicksum(
        (1 - beta) * (
            data.tau_imp * imports[t]
            + data.tau_exp * exports[t]
        )
        for t in T
    )

    penalty_cost = alpha * quicksum(
        excess[t] for t in T
    )

    external_cost = (
        energy_cost + tariff_cost + penalty_cost
    )


    # ===============================================
    # Strict Budget Balance (BB)
    # ===============================================
    # Linearized internal payments from follower KKT.
    # No price * quantity bilinear term is introduced.

    internal_revenue_linear = quicksum(
        blocks[i]["payment_linear"]
        for i in blocks
    )

    model.addConstr(
        internal_revenue_linear == external_cost,
        name="budget_balance"
    )


    # Common IR relaxation (DKK per member)
    epsilon = model.addVar(
        lb=0.0,
        name="IR_relaxation"
    )

    for member_id in blocks:
        model.addConstr(
            blocks[member_id]["private_cost_linear"]
            <= outside_costs[member_id] + epsilon,
            name=f"relaxed_IR_{member_id}"
        )

    # Cost threshold required by strict BB + IR
    model.addConstr(
        physical_cost + external_cost
        <= sum(outside_costs.values()),
        name="strict_IR_cost_threshold"
    )



    # Manager minimizes community social cost.
    model.setObjective(
        epsilon,
        GRB.MINIMIZE
    )

    # ============================================

    # =============================================
    # Load the DP+BB solution as a warm start
    # =============================================
    start_dir = Path(__file__).resolve().parent
    start_file = start_dir / "dp_bb_start.sol"

    if not start_file.is_file():
        raise FileNotFoundError(start_file)

    model.update()

    original_dir = Path.cwd()

    try:
        # Avoid passing a Chinese path to Gurobi.
        os.chdir(start_dir)

        # Read the solution using only its filename.
        model.read("dp_bb_start.sol")

    finally:
        os.chdir(original_dir)

    # The saved DP+BB solution has a maximum
    # IR violation of approximately 18.946881 DKK.
    epsilon.Start = 18.947

    print("\nWARM START PREPARED")
    print("Solution file:", start_file)
    print("Requested initial epsilon: 18.947 DKK")

    # Solve only once.
    model.optimize()


    if model.SolCount == 0:
        raise RuntimeError(
            f"No feasible solution. Status: {model.Status}"
        )

    print("\nSOLVER STATUS")
    print("-" * 50)
    print("Status:", model.Status)
    print("Best feasible objective:", model.ObjVal)
    print("Best bound:", model.ObjBound)
    print("MIP gap:", model.MIPGap)
    print("Runtime:", model.Runtime)

    if model.Status == GRB.TIME_LIMIT:
        print("Stopped at time limit with a feasible solution.")
    elif model.Status == GRB.OPTIMAL:
        print("Optimality tolerance satisfied.")
    else:
        print("Solver stopped with status:", model.Status)

    print("\nFIXED-PRICE MANAGER + KKT TEST")
    print("=" * 65)

    print(
        "Independent follower dispatch cost:",
        f'{accounting["community_cost"]:.8f}'
    )

    print(f"IR relaxation epsilon objective: {model.ObjVal:.8f}")

    print(
        "Difference:",
        f'{accounting["community_cost"] - model.ObjVal:.8f}'
    )

    print("\nKKT MANAGER COST BREAKDOWN")
    print("-" * 65)

    print("Physical cost:", physical_cost.getValue())
    print("Energy cost:", energy_cost.getValue())
    print("Grid tariff:", tariff_cost.getValue())
    print("DSO penalty:", penalty_cost.getValue())
    print("External cost:", external_cost.getValue())

    print(
        "Peak import:",
        max(imports[t].X for t in T)
    )
    print(
        "Total excess:",
        sum(excess[t].X for t in T)
    )

    #-----bb verify-----

    print("\nDP + BB VALIDATION")
    print("=" * 60)

    revenue_linear = internal_revenue_linear.getValue()
    payment_external = external_cost.getValue()

    # Independent numerical check using optimized prices.
    revenue_direct = sum(
        x[t].X * sum(
            blocks[i]["net"][t].getValue()
            for i in blocks
        )
        for t in T
    )

    bb_residual = revenue_direct - payment_external
    linearization_error = abs(
        revenue_direct - revenue_linear
    )

    print(f"Internal revenue (direct): {revenue_direct:.8f}")
    print(f"Internal revenue (linear): {revenue_linear:.8f}")
    print(f"External payment:          {payment_external:.8f}")
    print(f"BB residual:               {bb_residual:.3e}")
    print(f"Linearization error:       {linearization_error:.3e}")
    print(f"Community total cost:      {(physical_cost + external_cost).getValue():.8f}")
    print(f"Best bound:                {model.ObjBound:.8f}")
    print(f"MIP gap:                   {model.MIPGap:.4%}")

    assert abs(bb_residual) < 1e-5
    assert linearization_error < 1e-5

    print("\nDP + BB CHECKS PASSED")


    # Independent check of every member's optimal cost
    max_member_gap = 0.0

    for source in data.members:
        member = convert_member(source)

        optimized_prices = [x[t].X for t in T]

        direct = solve_member(
            member, optimized_prices, output=False
        )

        # x is fixed, hence private_cost is linear
        kkt_cost = blocks[source.id][
            "private_cost"
        ].getValue()

        gap = abs(direct["objective"] - kkt_cost)
        max_member_gap = max(max_member_gap, gap)

        assert gap < 1e-5, (
            f"Member optimality mismatch: {source.id}"
        )

    print("\nMax member optimality gap:", max_member_gap)

    # Any independently selected follower-optimal
    # dispatch is feasible for the optimistic model.

    print("\nDP PRICE OPTIMIZATION TEST PASSED")
    print("=" * 50)

    for t in T:
        print(
            f"Hour {t:02d}: "
            f"x={x[t].X:9.5f} "
            f"p={prices[t]:7.3f} "
            f"import={imports[t].X:9.4f}"
        )

    print("\nPRICE BOUND CHECK")
    at_lower = [
        t for t in T
        if abs(x[t].X - PRICE_LO) < 1e-5
    ]
    at_upper = [
        t for t in T
        if abs(x[t].X - PRICE_HI) < 1e-5
    ]

    print("Hours at lower price bound:", at_lower)
    print("Hours at upper price bound:", at_upper)


    print("\nCOMPARISON WITH Q1 CENTRAL DISPATCH")

    central_cost = 372.162434

    # Calculate actual community social cost
    social_cost = (
        physical_cost + external_cost
    ).getValue()

    print(f"Q1 C cost:            {central_cost:.6f} DKK")
    print(f"Community social cost:{social_cost:12.6f} DKK")
    print(f"Community cost - C:   {social_cost - central_cost:.6f} DKK")
    print(f"IR relaxation epsilon:{epsilon.X:12.6f} DKK")


    print("\nFIXED-PRICE MANAGER KKT TEST PASSED")

    # ----检查当前DP方案的预算收支（Budget Balance）-----

    print("\nDP BUDGET BALANCE CHECK")
    print("=" * 50)

    internal_revenue = sum(
        x[t].X * sum(
            blocks[i]["net"][t].getValue()
            for i in blocks
        )
        for t in T
    )

    external_payment = external_cost.getValue()

    budget_surplus = internal_revenue - external_payment

    print(f"Internal revenue: {internal_revenue:.6f} DKK")
    print(f"External payment: {external_payment:.6f} DKK")
    print(f"Budget surplus:   {budget_surplus:.6f} DKK")


    # ===============================================
    # IR diagnostic for the existing DP+BB solution
    # ===============================================
    outside_file = (
        Path(__file__).resolve().parent
        / "outside_costs_R.json"
    )

    outside_costs = json.loads(
        outside_file.read_text(encoding="utf-8")
    )

    print("\nIR DIAGNOSTIC FOR DP+BB SOLUTION")
    print("=" * 75)
    print(
        f"{'Member':10s} "
        f"{'Q1 R cost':>12s} "
        f"{'DP+BB cost':>12s} "
        f"{'IR margin':>12s}"
    )

    violating_members = []
    total_violation = 0.0

    for member_id in blocks:
        private_cost = blocks[member_id][
            "private_cost_linear"
        ].getValue()

        outside_cost = outside_costs[member_id]
        margin = outside_cost - private_cost

        print(
            f"{member_id:10s} "
            f"{outside_cost:12.6f} "
            f"{private_cost:12.6f} "
            f"{margin:12.6f}"
        )

        if margin < -1e-6:
            violating_members.append(member_id)
            total_violation += -margin

    print("-" * 75)
    print("Violating members:", violating_members)
    print("Number violating:", len(violating_members))
    print("Total IR violation:", round(total_violation, 6))

    print("\nIR DIAGNOSTIC COMPLETED")

    #----------

    print("\nIR RELAXATION RESULTS")
    print("=" * 55)

    print(f"Solver status: {model.Status}")
    print(f"Minimum found epsilon: {epsilon.X:.8f} DKK")
    print(f"Best bound on epsilon: {model.ObjBound:.8f} DKK")
    print(f"MIP gap: {model.MIPGap:.4%}")

    social_cost = (
        physical_cost + external_cost
    ).getValue()

    print(f"Community social cost: {social_cost:.8f} DKK")

    print("\nIndividual IR margins:")
    for member_id in blocks:
        actual_cost = blocks[member_id][
            "private_cost_linear"
        ].getValue()

        margin = outside_costs[member_id] - actual_cost

        print(f"{member_id:10s} {margin:12.6f}")

    print("\nIR RELAXATION CHECK COMPLETED")


    # Save the best feasible IR relaxation solution
    if model.SolCount > 0:
        output_dir = Path(__file__).resolve().parent
        original_dir = Path.cwd()

        try:
            os.chdir(output_dir)
            #model.write("ir_relax_best.sol")
            model.write("ir_relax_cost_limited.sol")
        finally:
            os.chdir(original_dir)

        print("\nIR relaxation solution saved:")
        print(output_dir / "ir_relax_cost_limited.sol")


    #---------------





if __name__ == "__main__":
    main()
