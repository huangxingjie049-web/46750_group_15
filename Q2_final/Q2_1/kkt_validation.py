
from gurobipy import GRB


def validate_kkt(model, tol=1e-5):
    """
    Independently validate KKT conditions for a
    continuous linear minimization model.

    Uses Gurobi's Pi, Slack, RC and constraint rows.
    """

    if model.Status != GRB.OPTIMAL:
        raise ValueError("Model is not optimal.")

    if model.IsMIP:
        raise ValueError("KKT validation requires an LP.")

    variables = model.getVars()
    constraints = model.getConstrs()

    # ------------------------------------------------
    # 1. Primal feasibility
    # ------------------------------------------------
    max_constraint_violation = 0.0
    max_bound_violation = 0.0

    for con in constraints:
        lhs = model.getRow(con).getValue()
        rhs = con.RHS

        if con.Sense == "<":
            violation = max(0.0, lhs - rhs)
        elif con.Sense == ">":
            violation = max(0.0, rhs - lhs)
        else:
            violation = abs(lhs - rhs)

        max_constraint_violation = max(
            max_constraint_violation, violation
        )

    for var in variables:
        violation = max(
            0.0,
            var.LB - var.X,
            var.X - var.UB
        )
        max_bound_violation = max(
            max_bound_violation, violation
        )

    # ------------------------------------------------
    # 2. Dual feasibility and complementarity
    # ------------------------------------------------
    max_dual_violation = 0.0
    max_complementarity = 0.0

    for con in constraints:
        pi = con.Pi

        if con.Sense == "<":
            dual_violation = max(0.0, pi)
        elif con.Sense == ">":
            dual_violation = max(0.0, -pi)
        else:
            dual_violation = 0.0

        max_dual_violation = max(
            max_dual_violation, dual_violation
        )

        if con.Sense != "=":
            max_complementarity = max(
                max_complementarity,
                abs(pi * con.Slack)
            )

    # ------------------------------------------------
    # 3. Stationarity: c - A.T*pi - RC = 0
    # ------------------------------------------------
    dual_gradient = {
        var.VarName: 0.0 for var in variables
    }

    for con in constraints:
        row = model.getRow(con)

        for k in range(row.size()):
            var = row.getVar(k)
            coefficient = row.getCoeff(k)

            dual_gradient[var.VarName] += (
                coefficient * con.Pi
            )

    max_stationarity = 0.0
    max_reduced_cost_violation = 0.0

    for var in variables:
        residual = (
            var.Obj
            - dual_gradient[var.VarName]
            - var.RC
        )

        max_stationarity = max(
            max_stationarity, abs(residual)
        )

        at_lower = abs(var.X - var.LB) <= tol
        at_upper = abs(var.X - var.UB) <= tol

        # Reduced-cost sign at a variable bound:
        # lower: RC >= 0, upper: RC <= 0.
        # A fixed variable may have either sign.
        if not (at_lower and at_upper):
            if at_lower:
                rc_violation = max(0.0, -var.RC)
            elif at_upper:
                rc_violation = max(0.0, var.RC)
            else:
                rc_violation = abs(var.RC)

            max_reduced_cost_violation = max(
                max_reduced_cost_violation,
                rc_violation
            )

        # Complementarity for variable bounds
        if var.RC > 0:
            max_complementarity = max(
                max_complementarity,
                abs(var.RC * (var.X - var.LB))
            )
        elif var.RC < 0:
            max_complementarity = max(
                max_complementarity,
                abs(var.RC * (var.UB - var.X))
            )

    # ------------------------------------------------
    # 4. Strong duality
    # ------------------------------------------------
    primal_objective = model.ObjVal

    # For a minimization LP, Gurobi's Pi and RC
    # provide the following dual objective.
    dual_objective = sum(
        con.Pi * con.RHS
        for con in constraints
    )

    for var in variables:
        if var.RC > 0 and var.LB > -GRB.INFINITY / 2:
            dual_objective += var.RC * var.LB
        elif var.RC < 0 and var.UB < GRB.INFINITY / 2:
            dual_objective += var.RC * var.UB

    # Include objective constant, if any.
    dual_objective += model.ObjCon

    duality_gap = abs(
        primal_objective - dual_objective
    )

    # ------------------------------------------------
    # 5. Print validation results
    # ------------------------------------------------
    checks = {
        "Primal constraint residual":
            max_constraint_violation,
        "Variable bound residual":
            max_bound_violation,
        "Dual sign residual":
            max_dual_violation,
        "Reduced-cost sign residual":
            max_reduced_cost_violation,
        "Stationarity residual":
            max_stationarity,
        "Complementarity residual":
            max_complementarity,
        "Primal-dual gap":
            duality_gap,
    }

    print("\nKKT VALIDATION")
    print("-" * 55)

    for name, value in checks.items():
        status = "PASS" if value <= tol else "FAIL"
        print(f"{name:32s}: {value:.3e}  {status}")

    print("-" * 55)
    print(f"Primal objective: {primal_objective:.8f}")
    print(f"Dual objective:   {dual_objective:.8f}")

    if all(value <= tol for value in checks.values()):
        print("ALL KKT CHECKS PASSED")
    else:
        raise AssertionError("Some KKT checks failed.")

    return checks
