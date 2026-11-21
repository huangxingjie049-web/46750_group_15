
from pathlib import Path
import sys
import json
import os
import tempfile
import shutil

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
        / "outside_costs_RI.json"
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


    economic_upper_bound = max(
        p + (1 - beta) * data.tau_imp
        for p in prices
    ) + alpha


    # Original independent follower reference
    response = aggregate_response(data, prices)
    accounting = calculate_manager_accounting(
        data, response
    )

    model = gp.Model("manager_fixed_price_KKT")
    model.Params.OutputFlag = 1

    # Accept a solution within 2% of the best bound.
    model.Params.MIPGap = 0.02
    model.Params.MIPGapAbs = 1e-5
    # Stop after at most 180 seconds.
    model.Params.TimeLimit = 300
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.5


    # Internal prices are decision variables in the IR relaxation model.
    # Price bounds are used for numerical optimization.
    # These bounds are not part of the DSO contract.

    PRICE_LO = 0.0
    PRICE_HI = economic_upper_bound


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
    imports = model.addVars(T, lb=0.0, name="feeder_import")
    imports_cap = model.addVars(
        T, lb=0.0, name="feeder_import_within_cap"
    )
    exports = model.addVars(T, lb=0.0, name="feeder_export")
    excess = model.addVars(T, lb=0.0, name="capacity_excess")

    above_cap = model.addVars(
        T,
        vtype=GRB.BINARY,
        name="above_cap"
    )

    grid_import_mode = model.addVars(
        T,
        vtype=GRB.BINARY,
        name="grid_import_mode"
    )



    for t in T:
        total_net = quicksum(
            blocks[i]["net"][t] for i in blocks
        )

        model.addConstr(
            imports[t] == imports_cap[t] + excess[t],
            name=f"import_split_{t}"
        )

        model.addConstr(
            imports[t] - exports[t] == total_net,
            name=f"feeder_balance_{t}"
        )

        model.addGenConstrIndicator(
            grid_import_mode[t],
            True,
            exports[t] == 0.0,
            name=f"no_export_{t}"
        )

        model.addGenConstrIndicator(
            grid_import_mode[t],
            False,
            imports[t] == 0.0,
            name=f"no_import_{t}"
        )

        model.addConstr(
            imports_cap[t] <= caps[t],
            name=f"capacity_limit_{t}"
        )

        model.addGenConstrIndicator(
            above_cap[t],
            False,
            excess[t] == 0.0,
            name=f"no_excess_{t}"
        )

        model.addGenConstrIndicator(
            above_cap[t],
            True,
            imports_cap[t] == caps[t],
            name=f"cap_filled_{t}"
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
            data.tau_imp * imports_cap[t]
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

    # Individual IR relaxation: one slack for each member
    ir_slack = model.addVars(
        list(blocks.keys()),
        lb=0.0,
        name="IR_slack"
    )

    for member_id in blocks:
        model.addConstr(
            blocks[member_id]["private_cost_linear"]
            <= outside_costs[member_id] + ir_slack[member_id],
            name=f"relaxed_IR_{member_id}"
        )


    # Cost threshold required by strict BB + IR
    #model.addConstr(
        #physical_cost + external_cost
        #<= sum(outside_costs.values()),
        #name="strict_IR_cost_threshold"
    #)


    # Stage 1: Minimize total individual IR violations
    total_ir_slack = quicksum(
        ir_slack[member_id] for member_id in blocks
    )

    social_cost = physical_cost + external_cost


    # Lexicographic optimization:
    # First minimize total IR slack, then community social cost.
    model.ModelSense = GRB.MINIMIZE

    model.setObjectiveN(
        total_ir_slack,
        index=0,
        priority=2,
        weight=1.0,
        abstol=1e-6,
        reltol=0.0,
        name="min_total_IR_slack"
    )

    model.setObjectiveN(
        social_cost,
        index=1,
        priority=1,
        weight=1.0,
        abstol=1e-6,
        reltol=0.0,
        name="min_community_social_cost"
    )



    # ============================================
    # Warm start from the previous feasible solution.
    model.update()
    start_file = (
        Path(__file__).resolve().parent
        / "ir_relax_RI_updated_tariff.sol"
    )

    if False and start_file.exists():
        with tempfile.TemporaryDirectory(prefix="gurobi_start_") as temp_dir:
            temp_start = Path(temp_dir) / "ir_start.sol"
            shutil.copyfile(start_file, temp_start)

            model.read(str(temp_start))

        print("\nWarm start loaded from:", start_file)
    else:
        print("\nNo saved IR relaxation solution found.")



    # Solve only once.
    model.optimize()


    if model.SolCount == 0:
        raise RuntimeError(
            f"No feasible solution. Status: {model.Status}"
        )


    print("\nMULTI-OBJECTIVE SOLVER STATUS")
    print("-" * 50)
    print("Status:", model.Status)
    print(f"Runtime: {model.Runtime:.2f} seconds")
    print(f"Total IR slack (feasible): {total_ir_slack.getValue():.8f} DKK")
    print(f"Community social cost (feasible): {social_cost.getValue():.8f} DKK")

    print("\nObjective priority:")
    print("1. Minimize total individual IR slack")
    print("2. Minimize community social cost")


    if model.Status == GRB.TIME_LIMIT:
        print("Stopped at time limit with a feasible solution.")
    elif model.Status == GRB.OPTIMAL:
        print("Optimality tolerance satisfied.")
    else:
        print("Solver stopped with status:", model.Status)

    print("\nIR RELAXATION MANAGER + KKT TEST")

    print("=" * 65)

    print(
        "Independent follower dispatch cost:",
        f'{accounting["community_cost"]:.8f}'
    )

    print(f"Total individual IR slack: {total_ir_slack.getValue():.8f} DKK")

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


    # Check that capacity excess matches actual import exceedance.
    max_excess_error = max(
        abs(excess[t].X - max(0.0, imports[t].X - caps[t]))
        for t in T
    )
    print(f"Maximum capacity excess error: {max_excess_error:.3e} kW")
    assert max_excess_error < 1e-5, (
        "Capacity excess does not match actual exceedance"
    )


    # Check that the community does not import and export simultaneously.
    print("\n=== SIMULTANEOUS IMPORT / EXPORT CHECK ===")

    simultaneous_hours = [
        t for t in T
        if imports[t].X > 1e-5 and exports[t].X > 1e-5
    ]

    print(f"Hours with simultaneous import and export: {simultaneous_hours}")

    for t in simultaneous_hours:
        print(
            f"Hour {t:02d}: "
            f"Import = {imports[t].X:.6f} kW, "
            f"Export = {exports[t].X:.6f} kW"
        )

    assert not simultaneous_hours, (
        "Simultaneous feeder import and export detected"
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
    print(f"Total IR slack (feasible): {total_ir_slack.getValue():.8f} DKK")
    print("Multi-objective bounds and gaps: see Gurobi solver log")


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
    social_cost_value = social_cost.getValue()


    # Calculate actual community social cost
    social_cost = (
        physical_cost + external_cost
    ).getValue()

    print(f"Q1 C cost:            {central_cost:.6f} DKK")
    print(f"Community social cost:{social_cost_value:12.6f} DKK")
    print(f"Community cost - C:   {social_cost_value - central_cost:.6f} DKK")
    print(f"Total IR slack: {total_ir_slack.getValue():12.6f} DKK")


    print("\nIR RELAXATION MANAGER KKT TEST PASSED")

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
        / "outside_costs_RI.json"
    )

    outside_costs = json.loads(
        outside_file.read_text(encoding="utf-8")
    )

    print("\nIR DIAGNOSTIC FOR DP+BB SOLUTION")
    print("=" * 75)
    print(
        f"{'Member':10s} "
        f"{'Q1 R/I cost':>12s} "
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
    print(f"Total IR slack (feasible): {total_ir_slack.getValue():.8f} DKK")
    print(f"Community social cost: {social_cost_value:.8f} DKK")


    print("\nINDIVIDUAL IR SLACK RESULTS")
    print("-" * 55)

    for member_id in blocks:
        print(
            f"{member_id:10s} "
            f"Slack = {ir_slack[member_id].X:.6f} DKK"
        )


    print("\n=== IR SLACK CONSISTENCY CHECK ===")

    max_slack_error = 0.0
    actual_total_violation = 0.0

    for member_id in blocks:
        private_cost = blocks[member_id]["private_cost_linear"].getValue()
        reference_cost = outside_costs[member_id]

        actual_violation = max(0.0, private_cost - reference_cost)
        slack_value = ir_slack[member_id].X

        slack_error = abs(slack_value - actual_violation)
        max_slack_error = max(max_slack_error, slack_error)
        actual_total_violation += actual_violation

        print(
            f"{member_id:10s} "
            f"Actual violation = {actual_violation:.6f}, "
            f"Slack = {slack_value:.6f}, "
            f"Difference = {slack_error:.3e}"
        )

    print(f"Total actual IR violation: {actual_total_violation:.8f} DKK")
    print(f"Total model IR slack: {total_ir_slack.getValue():.8f} DKK")
    print(f"Maximum slack difference: {max_slack_error:.3e} DKK")

    print(f"Community social cost: {social_cost_value:.8f} DKK")

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

            model.write("ir_relax_RI_Q1new.sol")
        finally:
            os.chdir(original_dir)

        print("\nIR relaxation solution saved:")
        print(output_dir / "ir_relax_RI_Q1new.sol")


    #---------------





if __name__ == "__main__":
    main()
