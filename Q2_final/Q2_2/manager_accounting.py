# manager的真实成本计算

from community_response import aggregate_response

from pathlib import Path
import sys

Q2_ROOT = Path(__file__).resolve().parent.parent
Q2_1_DIR = Q2_ROOT / "Q2_1"
sys.path.insert(0, str(Q2_1_DIR))

from data_interface import load_q1_data


def calculate_manager_accounting(data, response):
    """
    Calculate the manager's external settlement,
    congestion penalty and budget surplus.

    This evaluates a fixed dispatch. It does not
    optimize internal prices.
    """

    T = range(24)
    contract = data.contract

    beta = float(contract["tariff_discount_ratio"])
    alpha = float(contract["penalty_DKK_per_kWh"])
    caps = list(contract["capacity_limit_kW"])

    assert len(caps) == 24
    assert 0 <= beta <= 1
    assert alpha >= 0

    discounted_import_prices = [
        data.prices[t] + (1 - beta) * data.tau_imp
        for t in T
    ]

    discounted_export_prices = [
        data.prices[t] - (1 - beta) * data.tau_exp
        for t in T
    ]

    imports = response["imports"]
    exports = response["exports"]

    # One-hour intervals: kW excess = kWh excess.
    excess = [
        max(0.0, imports[t] - caps[t])
        for t in T
    ]

    energy_cost = sum(
        data.prices[t] * (imports[t] - exports[t])
        for t in T
    )

    grid_tariff_cost = sum(
        (1 - beta) * (
            data.tau_imp * imports[t]
            + data.tau_exp * exports[t]
        )
        for t in T
    )

    penalty_cost = alpha * sum(excess)

    external_cost = (
        energy_cost + grid_tariff_cost + penalty_cost
    )

    # Cross-check against discounted tariff formula.
    direct_external_cost = sum(
        discounted_import_prices[t] * imports[t]
        - discounted_export_prices[t] * exports[t]
        + alpha * excess[t]
        for t in T
    )

    assert abs(
        external_cost - direct_external_cost
    ) < 1e-7

    physical_cost = response["physical_cost"]
    internal_revenue = response["internal_payments"]

    community_cost = physical_cost + external_cost
    budget_surplus = internal_revenue - external_cost

    return {
        "caps": caps,
        "excess": excess,
        "energy_cost": energy_cost,
        "grid_tariff_cost": grid_tariff_cost,
        "penalty_cost": penalty_cost,
        "external_cost": external_cost,
        "physical_cost": physical_cost,
        "community_cost": community_cost,
        "internal_revenue": internal_revenue,
        "budget_surplus": budget_surplus,
    }


def main():
    data = load_q1_data()

    # Fixed diagnostic prices, NOT optimized DP.
    prices = list(data.prices)

    response = aggregate_response(data, prices)
    accounting = calculate_manager_accounting(
        data, response
    )

    print("\nMANAGER ACCOUNTING AT FIXED PRICES")
    print("=" * 65)

    print("Hour    Import    Cap      Excess")

    for t in range(24):
        print(
            f"{t:02d}      "
            f"{response['imports'][t]:7.3f}  "
            f"{accounting['caps'][t]:7.3f}  "
            f"{accounting['excess'][t]:7.3f}"
        )

    print("\nCOST BREAKDOWN")
    print("-" * 65)

    labels = [
        ("Wholesale energy cost",
         accounting["energy_cost"]),
        ("Discounted grid tariffs",
         accounting["grid_tariff_cost"]),
        ("DSO capacity penalty",
         accounting["penalty_cost"]),
        ("Manager external cost",
         accounting["external_cost"]),
        ("Member physical cost",
         accounting["physical_cost"]),
        ("Total community cost",
         accounting["community_cost"]),
        ("Internal net revenue",
         accounting["internal_revenue"]),
        ("Manager budget surplus",
         accounting["budget_surplus"]),
    ]

    for name, value in labels:
        print(f"{name:30s}: {value:12.6f} DKK")

    print("\nTotal excess (kWh):",
          round(sum(accounting["excess"]), 6))

    print("Hours exceeding cap:",
          sum(z > 1e-7 for z in accounting["excess"]))

    assert abs(
        accounting["community_cost"]
        - accounting["physical_cost"]
        - accounting["external_cost"]
    ) < 1e-7

    assert abs(
        accounting["budget_surplus"]
        - accounting["internal_revenue"]
        + accounting["external_cost"]
    ) < 1e-7

    print("\nMANAGER ACCOUNTING CHECKS PASSED")


if __name__ == "__main__":
    main()
