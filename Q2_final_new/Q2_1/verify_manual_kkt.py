
from data_interface import load_q1_data, convert_member
from follower import solve_member


def verify_member(source, prices):
    member = convert_member(source)
    result = solve_member(
        member, prices,
        output=False,
        return_model=True
    )
    model = result["_model"]

    def pi(name):
        con = model.getConstrByName(name)
        if con is None:
            raise RuntimeError(f"Missing constraint: {name}")
        return con.Pi

    def rc(name):
        var = model.getVarByName(name)
        if var is None:
            raise RuntimeError(f"Missing variable: {name}")
        return var.RC

    residuals = []

    def record(name, value):
        residuals.append((name, abs(value)))

    # Daily minimum energy multiplier
    daily_pi = pi("daily_energy")

    for t in range(24):
        x = prices[t]
        lam = pi(f"net_balance_{t}")
        dev = pi(f"deviation_{t}")

        # Load: K1, including explicit variable bound RC
        r_load = (
            lam
            - pi(f"load_min_{t}")
            - pi(f"load_max_{t}")
            - dev
            - daily_pi
            - rc(f"load[{t}]")
        )
        record(f"load[{t}]", r_load)

        # PV: K2
        r_pv = (
            member["pv_cost"]
            - lam
            - pi(f"pv_max_{t}")
            - rc(f"pv[{t}]")
        )
        record(f"pv[{t}]", r_pv)

        # Positive deviation: K3
        r_plus = (
            member["disutility_rate"]
            + dev
            - rc(f"d_plus[{t}]")
        )
        record(f"d_plus[{t}]", r_plus)

        # Negative deviation: K4
        r_minus = (
            member["disutility_rate"]
            - dev
            - rc(f"d_minus[{t}]")
        )
        record(f"d_minus[{t}]", r_minus)

        # Explicit net-import variable
        r_net = x - lam - rc(f"net_import[{t}]")
        record(f"net_import[{t}]", r_net)

        battery = member["battery"]

        if battery is not None:
            soc_pi = pi(f"soc_balance_{t}")

            # Charging: K5
            r_charge = (
                lam
                + battery["eta_charge"] * soc_pi
                - pi(f"charge_max_{t}")
                - rc(f"charge[{t}]")
            )
            record(f"charge[{t}]", r_charge)

            # Discharging: K6
            r_discharge = (
                -lam
                - soc_pi / battery["eta_discharge"]
                - pi(f"discharge_max_{t}")
                - rc(f"discharge[{t}]")
            )
            record(f"discharge[{t}]", r_discharge)

            # State of charge: K7 and K8
            next_pi = (
                pi(f"soc_balance_{t + 1}")
                if t < 23 else 0.0
            )
            terminal_pi = (
                pi("terminal_soc")
                if t == 23 else 0.0
            )

            r_soc = (
                -soc_pi
                + next_pi
                - pi(f"soc_max_{t}")
                - terminal_pi
                - rc(f"soc[{t}]")
            )
            record(f"soc[{t}]", r_soc)

    worst_name, worst_value = max(
        residuals,
        key=lambda item: item[1]
    )

    return {
        "member": source.id,
        "max_residual": worst_value,
        "worst_variable": worst_name,
        "checks": len(residuals),
    }


def main():
    data = load_q1_data()
    prices = list(data.prices)
    tolerance = 1e-7

    print("MANUAL STATIONARITY VALIDATION")
    print("-" * 70)

    overall_max = 0.0
    total_checks = 0

    for source in data.members:
        result = verify_member(source, prices)

        maximum = result["max_residual"]
        overall_max = max(overall_max, maximum)
        total_checks += result["checks"]

        status = "PASS" if maximum <= tolerance else "FAIL"

        print(
            f"{result['member']:10s} "
            f"max residual = {maximum:.3e} "
            f"{status:5s} "
            f"worst = {result['worst_variable']}"
        )

        if status == "FAIL":
            raise AssertionError(
                f"Manual stationarity failed for {source.id}"
            )

    print("-" * 70)
    print("Members checked:", len(data.members))
    print("Stationarity equations checked:", total_checks)
    print(f"Largest residual: {overall_max:.3e}")
    print("ALL MANUAL STATIONARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
