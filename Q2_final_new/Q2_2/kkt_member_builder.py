# 统一KKT Builder，让16名消费者能够同时进入一个Manager模型。
# 这将是我们从单消费者验证走向真正动态定价优化的关键一步
import gurobipy as gp
from gurobipy import GRB, quicksum


def add_member_kkt(model, member, prices, member_id):
    """
    Add one member's complete KKT system to an
    existing Gurobi model.

    prices: 24 fixed numbers OR 24 Gurobi variables.
    Supports members with or without batteries.
    """
    T = range(24)
    if len(prices) != 24:
        raise ValueError("Expected 24 hourly prices.")

    name = str(member_id)
    bat = member["battery"]

    def v(label, lb=0):
        return model.addVars(
            T, lb=lb, name=f"{name}_{label}"
        )

    def scalar(label, lb=0):
        return model.addVar(
            lb=lb, name=f"{name}_{label}"
        )

    def complement(dual, slack):
        model.addSOS(GRB.SOS_TYPE1, [dual, slack])

    # --------------------------------------------------
    # 1. Primal variables
    # --------------------------------------------------
    load = v("load", lb=-GRB.INFINITY)
    pv = v("pv")
    d_plus = v("d_plus")
    d_minus = v("d_minus")

    # --------------------------------------------------
    # 2. Dual variables: load, PV and disutility
    # --------------------------------------------------
    eta_load_lo = v("eta_load_lo")
    eta_load_up = v("eta_load_up")
    eta_pv_lo = v("eta_pv_lo")
    eta_pv_up = v("eta_pv_up")

    psi_plus = v("psi_plus")
    psi_minus = v("psi_minus")
    theta = v("theta", lb=-GRB.INFINITY)
    mu = scalar("mu_daily")

    # --------------------------------------------------
    # 3. Nonnegative slacks
    # --------------------------------------------------
    s_load_lo = v("s_load_lo")
    s_load_up = v("s_load_up")
    s_pv_up = v("s_pv_up")
    s_daily = scalar("s_daily")

    # --------------------------------------------------
    # 4. Member physical constraints and K1-K4
    # --------------------------------------------------
    for t in T:
        x = prices[t]

        model.addConstr(
            load[t] - member["load_min"][t]
            == s_load_lo[t],
            name=f"{name}_load_lower_{t}"
        )

        model.addConstr(
            member["load_max"][t] - load[t]
            == s_load_up[t],
            name=f"{name}_load_upper_{t}"
        )

        model.addConstr(
            member["pv_available"][t] - pv[t]
            == s_pv_up[t],
            name=f"{name}_pv_upper_{t}"
        )

        model.addConstr(
            load[t] - member["reference"][t]
            - d_plus[t] + d_minus[t] == 0,
            name=f"{name}_deviation_{t}"
        )

        # K1: load
        model.addConstr(
            x - eta_load_lo[t] + eta_load_up[t]
            + theta[t] - mu == 0,
            name=f"{name}_K1_{t}"
        )

        # K2: PV
        model.addConstr(
            member["pv_cost"] - x
            - eta_pv_lo[t] + eta_pv_up[t] == 0,
            name=f"{name}_K2_{t}"
        )

        # K3 and K4: load deviation
        model.addConstr(
            member["disutility_rate"]
            - theta[t] - psi_plus[t] == 0,
            name=f"{name}_K3_{t}"
        )

        model.addConstr(
            member["disutility_rate"]
            + theta[t] - psi_minus[t] == 0,
            name=f"{name}_K4_{t}"
        )

        # Complementary slackness
        complement(eta_load_lo[t], s_load_lo[t])
        complement(eta_load_up[t], s_load_up[t])
        complement(eta_pv_lo[t], pv[t])
        complement(eta_pv_up[t], s_pv_up[t])
        complement(psi_plus[t], d_plus[t])
        complement(psi_minus[t], d_minus[t])

    model.addConstr(
        quicksum(load[t] for t in T)
        - member["min_daily_energy"] == s_daily,
        name=f"{name}_daily_minimum"
    )
    complement(mu, s_daily)

    # --------------------------------------------------
    # 5. Optional battery: primal and dual variables
    # --------------------------------------------------
    charge = discharge = soc = None

    if bat is not None:
        charge = v("charge")
        discharge = v("discharge")
        soc = v("soc")

        eta_ch_lo = v("eta_ch_lo")
        eta_ch_up = v("eta_ch_up")
        eta_dis_lo = v("eta_dis_lo")
        eta_dis_up = v("eta_dis_up")
        eta_soc_lo = v("eta_soc_lo")
        eta_soc_up = v("eta_soc_up")

        sigma = v("sigma", lb=-GRB.INFINITY)
        phi = scalar("phi_terminal")

        s_ch_up = v("s_ch_up")
        s_dis_up = v("s_dis_up")
        s_soc_up = v("s_soc_up")
        s_terminal = scalar("s_terminal")

        for t in T:
            x = prices[t]

            model.addConstr(
                bat["p_charge_max"] - charge[t]
                == s_ch_up[t],
                name=f"{name}_charge_upper_{t}"
            )

            model.addConstr(
                bat["p_discharge_max"] - discharge[t]
                == s_dis_up[t],
                name=f"{name}_discharge_upper_{t}"
            )

            model.addConstr(
                bat["capacity"] - soc[t]
                == s_soc_up[t],
                name=f"{name}_soc_upper_{t}"
            )

            previous = (
                bat["initial_energy"] if t == 0
                else soc[t - 1]
            )

            model.addConstr(
                soc[t] - previous
                - bat["eta_charge"] * charge[t]
                + discharge[t] / bat["eta_discharge"]
                == 0,
                name=f"{name}_soc_balance_{t}"
            )

            # K5: charging
            model.addConstr(
                x - bat["eta_charge"] * sigma[t]
                - eta_ch_lo[t] + eta_ch_up[t] == 0,
                name=f"{name}_K5_{t}"
            )

            # K6: discharging
            model.addConstr(
                -x + sigma[t] / bat["eta_discharge"]
                - eta_dis_lo[t] + eta_dis_up[t] == 0,
                name=f"{name}_K6_{t}"
            )

            # K7 / K8: state of charge
            if t < 23:
                model.addConstr(
                    sigma[t] - sigma[t + 1]
                    - eta_soc_lo[t] + eta_soc_up[t] == 0,
                    name=f"{name}_K7_{t}"
                )
            else:
                model.addConstr(
                    sigma[t]
                    - eta_soc_lo[t] + eta_soc_up[t]
                    - phi == 0,
                    name=f"{name}_K8"
                )

            complement(eta_ch_lo[t], charge[t])
            complement(eta_ch_up[t], s_ch_up[t])
            complement(eta_dis_lo[t], discharge[t])
            complement(eta_dis_up[t], s_dis_up[t])
            complement(eta_soc_lo[t], soc[t])
            complement(eta_soc_up[t], s_soc_up[t])

        model.addConstr(
            soc[23] - bat["initial_energy"]
            == s_terminal,
            name=f"{name}_terminal_soc"
        )
        complement(phi, s_terminal)

    # --------------------------------------------------
    # 6. Return expressions for the manager model
    # --------------------------------------------------
    net = {}
    for t in T:
        net[t] = load[t] - pv[t]
        if bat is not None:
            net[t] += charge[t] - discharge[t]

    physical_cost = quicksum(
        member["pv_cost"] * pv[t]
        + member["disutility_rate"]
        * (d_plus[t] + d_minus[t])
        for t in T
    )


    # --------------------------------------------------
    # 7. Strong-duality payment linearization
    # --------------------------------------------------
    # Dual objective from the KKT Lagrangian.
    # All products below are parameter * dual variable.

    dual_objective = quicksum(
        eta_load_lo[t] * member["load_min"][t]
        - eta_load_up[t] * member["load_max"][t]
        - theta[t] * member["reference"][t]
        - eta_pv_up[t] * member["pv_available"][t]
        for t in T
    )

    dual_objective += mu * member["min_daily_energy"]

    if bat is not None:
        dual_objective -= quicksum(
            eta_ch_up[t] * bat["p_charge_max"]
            + eta_dis_up[t] * bat["p_discharge_max"]
            + eta_soc_up[t] * bat["capacity"]
            for t in T
        )

        dual_objective += (
            -sigma[0] * bat["initial_energy"]
            + phi * bat["initial_energy"]
        )

    # Strong duality:
    # private_cost = physical_cost + payment
    #              = dual_objective
    #
    # Therefore payment = dual_objective - physical_cost.
    # This expression is LINEAR even if prices are variables.
    payment_linear = dual_objective - physical_cost


    # At fixed prices this is linear.
    # With variable prices it becomes quadratic,
    # so DO NOT insert it directly into a MILP objective.
    private_cost = physical_cost + quicksum(
        prices[t] * net[t] for t in T
    )

    return {
        "load": load,
        "pv": pv,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "net": net,
        "physical_cost": physical_cost,
        "private_cost": private_cost,

        # New linear expressions
        "dual_objective": dual_objective,
        "payment_linear": payment_linear,
        "private_cost_linear": dual_objective,
    }
