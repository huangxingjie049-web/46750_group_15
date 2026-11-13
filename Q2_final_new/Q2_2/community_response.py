
from pathlib import Path
import sys

# Locate the existing Q2.1 implementation.
Q2_ROOT = Path(__file__).resolve().parent.parent
Q2_1_DIR = Q2_ROOT / "Q2_1"

sys.path.insert(0, str(Q2_1_DIR))

from data_interface import load_q1_data, convert_member
from follower import solve_member


def aggregate_response(data, prices):
    """
    Solve all 16 independent followers at fixed prices.

    This is a diagnostic tool, not the optimal
    manager pricing model.
    """
    if len(prices) != 24:
        raise ValueError("Expected 24 hourly prices.")

    hourly_net = [0.0] * 24
    hourly_load = [0.0] * 24
    hourly_pv = [0.0] * 24

    total_physical_cost = 0.0
    total_internal_payments = 0.0
    member_results = {}

    for source in data.members:
        member = convert_member(source)

        result = solve_member(
            member, prices, output=False
        )

        member_results[source.id] = result

        total_physical_cost += (
            result["pv_cost"] + result["disutility"]
        )
        total_internal_payments += result["payment"]

        for t in range(24):
            hourly_net[t] += result["net_import"][t]
            hourly_load[t] += result["load"][t]
            hourly_pv[t] += result["pv"][t]

    if len(member_results) != 16:
        raise AssertionError("Expected 16 members.")

    # Connection-point net metering.
    imports = [max(q, 0.0) for q in hourly_net]
    exports = [max(-q, 0.0) for q in hourly_net]

    # Independent hourly balance verification.
    for t in range(24):
        assert abs(
            imports[t] - exports[t] - hourly_net[t]
        ) < 1e-7

    # Independent payment accounting check.
    payment_from_aggregate = sum(
        prices[t] * hourly_net[t]
        for t in range(24)
    )

    assert abs(
        payment_from_aggregate - total_internal_payments
    ) < 1e-6

    return {
        "member_results": member_results,
        "net": hourly_net,
        "load": hourly_load,
        "pv": hourly_pv,
        "imports": imports,
        "exports": exports,
        "physical_cost": total_physical_cost,
        "internal_payments": total_internal_payments,
    }


def main():
    data = load_q1_data()

    # Diagnostic price vector only.
    # The optimal DP prices will be decision variables
    # in the subsequent manager MILP.
    prices = list(data.prices)

    result = aggregate_response(data, prices)

    print("\nCOMMUNITY RESPONSE AT FIXED TEST PRICES")
    print("=" * 65)
    print("Members solved:", len(result["member_results"]))
    print("Test prices: x_t = p_t")

    print("\nHour   Load(kWh)    PV(kWh)    Net(kWh)    Import    Export")

    for t in range(24):
        print(
            f"{t:02d}     "
            f"{result['load'][t]:9.3f} "
            f"{result['pv'][t]:10.3f} "
            f"{result['net'][t]:10.3f} "
            f"{result['imports'][t]:9.3f} "
            f"{result['exports'][t]:9.3f}"
        )

    print("\nSUMMARY")
    print("-" * 65)

    print(
        "Total member physical cost (DKK):",
        round(result["physical_cost"], 6)
    )
    print(
        "Total internal payments (DKK):",
        round(result["internal_payments"], 6)
    )
    print(
        "Peak feeder import (kW):",
        round(max(result["imports"]), 6)
    )
    print(
        "Total feeder import (kWh):",
        round(sum(result["imports"]), 6)
    )
    print(
        "Total feeder export (kWh):",
        round(sum(result["exports"]), 6)
    )

    print("\nCOMMUNITY AGGREGATION CHECKS PASSED")


if __name__ == "__main__":
    main()
