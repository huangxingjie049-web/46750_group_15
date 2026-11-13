
from gurobipy import Model, GRB, quicksum


def solve_member(member, prices, output=False, return_model=False):
    """
    Solve one member's best response to posted internal prices.

    All energy quantities refer to one-hour intervals.
    Prices and costs are in DKK/kWh.
    """

    T = range(24)

    assert len(prices) == 24
    assert len(member["reference"]) == 24
    assert len(member["load_min"]) == 24
    assert len(member["load_max"]) == 24
    assert len(member["pv_available"]) == 24

    m = Model("member_best_response")
    m.Params.OutputFlag = int(output)

    # 1. Member decisions
    load = m.addVars(T, lb=0, name="load")
    pv = m.addVars(T, lb=0, name="pv")
    d_plus = m.addVars(T, lb=0, name="d_plus")
    d_minus = m.addVars(T, lb=0, name="d_minus")

    # Net import can be positive or negative.
    q = m.addVars(T, lb=-GRB.INFINITY, name="net_import")

    # 2. Shared load, PV and deviation constraints
    for t in T:
        m.addConstr(
            load[t] >= member["load_min"][t],
            name=f"load_min_{t}"
        )
        m.addConstr(
            load[t] <= member["load_max"][t],
            name=f"load_max_{t}"
        )
        m.addConstr(
            pv[t] <= member["pv_available"][t],
            name=f"pv_max_{t}"
        )
        m.addConstr(
            load[t] - member["reference"][t]
            == d_plus[t] - d_minus[t],
            name=f"deviation_{t}"
        )

    # Daily minimum consumption
    m.addConstr(
        quicksum(load[t] for t in T)
        >= member["min_daily_energy"],
        name="daily_energy"
    )

    # 3. Optional battery
    battery = member.get("battery")
    charge = discharge = soc = None

    if battery is not None:
        charge = m.addVars(T, lb=0, name="charge")
        discharge = m.addVars(T, lb=0, name="discharge")
        soc = m.addVars(T, lb=0, name="soc")

        for t in T:
            m.addConstr(
                charge[t] <= battery["p_charge_max"],
                name=f"charge_max_{t}"
            )
            m.addConstr(
                discharge[t] <= battery["p_discharge_max"],
                name=f"discharge_max_{t}"
            )
            m.addConstr(
                soc[t] <= battery["capacity"],
                name=f"soc_max_{t}"
            )

            previous = (
                battery["initial_energy"]
                if t == 0 else soc[t - 1]
            )

            m.addConstr(
                soc[t] == previous
                + battery["eta_charge"] * charge[t]
                - discharge[t] / battery["eta_discharge"],
                name=f"soc_balance_{t}"
            )

        # Same terminal condition as Q1
        m.addConstr(
            soc[23] >= battery["initial_energy"],
            name="terminal_soc"
        )

    # 4. Net import balance
    for t in T:
        battery_net = (
            charge[t] - discharge[t]
            if battery is not None else 0
        )

        m.addConstr(
            q[t] == load[t] - pv[t] + battery_net,
            name=f"net_balance_{t}"
        )

    # 5. Individual objective
    pv_cost = quicksum(
        member["pv_cost"] * pv[t] for t in T
    )

    disutility = quicksum(
        member["disutility_rate"] *
        (d_plus[t] + d_minus[t])
        for t in T
    )

    payment = quicksum(
        prices[t] * q[t] for t in T
    )

    m.setObjective(
        pv_cost + disutility + payment,
        GRB.MINIMIZE
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Follower status: {m.Status}"
        )

    result = {
        "objective": m.ObjVal,
        "pv_cost": pv_cost.getValue(),
        "disutility": disutility.getValue(),
        "payment": payment.getValue(),
        "load": [load[t].X for t in T],
        "pv": [pv[t].X for t in T],
        "net_import": [q[t].X for t in T],
    }

    if battery is not None:
        result["charge"] = [charge[t].X for t in T]
        result["discharge"] = [discharge[t].X for t in T]
        result["soc"] = [soc[t].X for t in T]

    if return_model:
        result["_model"] = m

    return result
