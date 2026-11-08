
"""
Q2 Step 3:
Check whether the Q1 C centralized dispatch is individually
optimal when members face Q1 C's hourly dual prices.

This script only solves 16 independent follower LPs.
It does not modify the existing Manager models.
"""

import csv
import sys
from pathlib import Path

# ============================================================
# 1. Project paths
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

Q1_C_DIR = (
    PROJECT_ROOT
    / "Q1_models"
    / "results"
    / "base_case"
    / "C"
)

Q2_1_DIR = HERE.parent / "Q2_1"

sys.path.insert(0, str(Q2_1_DIR))

from data_interface import load_q1_data, convert_member
from follower import solve_member

TOL = 1e-5


# ============================================================
# 2. Read CSV exports
# ============================================================

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Q1 C export: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


# ============================================================
# 3. Main validation
# ============================================================

def main():

    h_rows = read_csv(Q1_C_DIR / "hourly.csv")
    m_rows = read_csv(Q1_C_DIR / "members.csv")

    if len(h_rows) != 24:
        raise ValueError(
            f"Expected 24 Q1 hours, found {len(h_rows)}"
        )

    # Q1 C dual prices
    prices = [0.0] * 24

    for r in h_rows:
        hour = int(r["hour"])

        prices[hour] = float(
            r["internal_price_DKK_per_kWh"]
        )

    # Q1 C member dispatch
    rows_by_member = {}

    for r in m_rows:
        mid = r["member"]

        rows_by_member.setdefault(
            mid, {}
        )[int(r["hour"])] = r

    data = load_q1_data()

    print(
        "\n=== Q2 STEP 3: Q1 C DUAL-PRICE FOLLOWER TEST ==="
    )

    print("Q1 C prices:", Q1_C_DIR / "hourly.csv")
    print("Q1 C dispatch:", Q1_C_DIR / "members.csv")

    print(
        "\nMember       C private cost   "
        "Follower minimum     Difference    Max net diff"
    )

    print("-" * 85)

    count_fail = 0
    count_different = 0

    total_c = 0.0
    total_f = 0.0

    different_hours = []

    # ========================================================
    # 4. Test all 16 independent followers
    # ========================================================

    for source in data.members:

        mid = source.id
        member = convert_member(source)

        by_hour = rows_by_member.get(mid, {})

        if len(by_hour) != 24:
            raise ValueError(
                f"{mid}: expected 24 rows, "
                f"got {len(by_hour)}"
            )

        c_private = 0.0
        c_net = []

        # Calculate each member's private cost
        # using the Q1 C dispatch and Q1 C prices.
        for t in range(24):

            r = by_hour[t]

            pv = float(r["PV_kWh"])
            d_plus = float(r["d_plus_kWh"])
            d_minus = float(r["d_minus_kWh"])
            net = float(r["net_kWh"])

            c_private += (
                member["pv_cost"] * pv
                + member["disutility_rate"]
                * (d_plus + d_minus)
                + prices[t] * net
            )

            c_net.append(net)

        # Independently solve this follower LP
        follower = solve_member(
            member,
            prices,
            output=False
        )

        delta = c_private - follower["objective"]

        net_diffs = [
            abs(
                c_net[t]
                - follower["net_import"][t]
            )
            for t in range(24)
        ]

        max_net_diff = max(net_diffs)

        if abs(delta) > TOL:
            count_fail += 1

        if max_net_diff > TOL:

            count_different += 1

            hours = [
                t
                for t, diff in enumerate(net_diffs)
                if diff > TOL
            ]

            different_hours.append(
                (mid, hours)
            )

        total_c += c_private
        total_f += follower["objective"]

        print(
            f"{mid:<11} "
            f"{c_private:14.6f} "
            f"{follower['objective']:18.6f} "
            f"{delta:14.3e} "
            f"{max_net_diff:13.6f}"
        )

    # ========================================================
    # 5. Final summary
    # ========================================================

    print("\n=== STEP 3 SUMMARY ===")

    print(
        "Members checked:",
        len(data.members)
    )

    print(
        "Members where C dispatch is NOT follower-optimal:",
        count_fail
    )

    print(
        "Members with different independent LP dispatch:",
        count_different
    )

    print(
        "Sum of C private costs:",
        f"{total_c:.8f} DKK"
    )

    print(
        "Sum of follower minimum costs:",
        f"{total_f:.8f} DKK"
    )

    print(
        "Total C-minus-minimum gap:",
        f"{total_c - total_f:.3e} DKK"
    )

    print("\nDifferent net-import hours:")

    for mid, hours in different_hours:
        print(f"{mid}: {hours}")

    # ========================================================
    # 6. Interpretation
    # ========================================================

    print("\n=== FINAL INTERPRETATION ===")

    if count_fail == 0:

        print(
            "Q1 C dispatch is individually optimal "
            "at Q1 C dual prices."
        )

        print(
            "Optimistic price support is verified."
        )

        if count_different > 0:
            print(
                "Some independent LPs choose different "
                "but equally optimal dispatches."
            )

            print(
                "These are candidates for further "
                "indifference analysis."
            )

    else:

        print(
            "Some Q1 C member dispatches are not "
            "follower-optimal under these prices."
        )

        print(
            "Check model consistency and price "
            "sign conventions before concluding."
        )


if __name__ == "__main__":
    main()

    # ============================================================
    # Q2.3 STEP 3B
    # Feeder and cost effects of alternative optimal responses
    #
    # Paste BELOW the existing:
    # if __name__ == "__main__":
    #     main()
    # ============================================================

    from community_response import aggregate_response
    from manager_accounting import calculate_manager_accounting


    def compare_c_and_independent_responses():

        print("\n" + "=" * 75)
        print("=== Q2 STEP 3B: FEEDER RESPONSE COMPARISON ===")
        print("=" * 75)

        # 1. Load Q1 C exports
        hourly_rows = read_csv(Q1_C_DIR / "hourly.csv")
        member_rows = read_csv(Q1_C_DIR / "members.csv")

        hourly_by_t = {
            int(r["hour"]): r
            for r in hourly_rows
        }

        if set(hourly_by_t) != set(range(24)):
            raise ValueError("Q1 C hourly data must contain hours 0-23")

        # 2. Load original community data
        data = load_q1_data()

        prices = [
            float(
                hourly_by_t[t]["internal_price_DKK_per_kWh"]
            )
            for t in range(24)
        ]

        # 3. Solve the 16 independent follower LPs
        response = aggregate_response(data, prices)

        # Use existing, previously validated accounting
        independent = calculate_manager_accounting(
            data, response
        )

        # 4. Reconstruct the centralized Q1 C dispatch
        c_imports = [
            float(hourly_by_t[t]["import_kWh"])
            for t in range(24)
        ]

        c_exports = [
            float(hourly_by_t[t]["export_kWh"])
            for t in range(24)
        ]

        c_net = [
            c_imports[t] - c_exports[t]
            for t in range(24)
        ]

        # Physical cost of Q1 C member dispatch
        members_by_id = {
            m.id: convert_member(m)
            for m in data.members
        }

        c_physical = 0.0
        c_internal_payments = 0.0

        for r in member_rows:
            mid = r["member"]
            t = int(r["hour"])

            member = members_by_id[mid]

            pv = float(r["PV_kWh"])
            d_plus = float(r["d_plus_kWh"])
            d_minus = float(r["d_minus_kWh"])
            net = float(r["net_kWh"])

            c_physical += (
                    member["pv_cost"] * pv
                    + member["disutility_rate"]
                    * (d_plus + d_minus)
            )

            c_internal_payments += prices[t] * net

        # Validate member net energy against Q1 feeder CSV
        c_member_net = [0.0] * 24

        for r in member_rows:
            t = int(r["hour"])
            c_member_net[t] += float(r["net_kWh"])

        max_c_balance_error = max(
            abs(c_member_net[t] - c_net[t])
            for t in range(24)
        )

        if max_c_balance_error > 1e-5:
            raise AssertionError(
                "Q1 C member net and feeder balance disagree: "
                f"{max_c_balance_error}"
            )

        # Reuse the same accounting for both dispatches
        c_response = {
            "imports": c_imports,
            "exports": c_exports,
            "physical_cost": c_physical,
            "internal_payments": c_internal_payments,
        }

        centralized = calculate_manager_accounting(
            data, c_response
        )

        # 5. Hourly comparison
        print("\nHOURLY FEEDER COMPARISON")
        print("-" * 75)

        print(
            "Hour   C net    Independent net   "
            "Difference   C excess   Independent excess"
        )

        for t in range(24):
            net_difference = (
                    response["net"][t] - c_net[t]
            )

            print(
                f"{t:02d}   "
                f"{c_net[t]:8.3f}   "
                f"{response['net'][t]:15.3f}   "
                f"{net_difference:10.3f}   "
                f"{centralized['excess'][t]:8.3f}   "
                f"{independent['excess'][t]:16.3f}"
            )

        # 6. Community-level indicators
        print("\n=== COMMUNITY COST COMPARISON ===")

        def show(label, a, b):
            print(
                f"{label:<32}"
                f"C={a:12.6f}   "
                f"Independent={b:12.6f}   "
                f"Delta={b - a:+12.6f}"
            )

        show(
            "Peak feeder import (kW)",
            max(c_imports),
            max(response["imports"])
        )

        show(
            "Total feeder imports (kWh)",
            sum(c_imports),
            sum(response["imports"])
        )

        show(
            "Total feeder exports (kWh)",
            sum(c_exports),
            sum(response["exports"])
        )

        show(
            "Total capacity excess (kWh)",
            sum(centralized["excess"]),
            sum(independent["excess"])
        )

        show(
            "Member physical cost (DKK)",
            centralized["physical_cost"],
            independent["physical_cost"]
        )

        show(
            "External energy cost (DKK)",
            centralized["energy_cost"],
            independent["energy_cost"]
        )

        show(
            "Grid tariff cost (DKK)",
            centralized["grid_tariff_cost"],
            independent["grid_tariff_cost"]
        )

        show(
            "Capacity penalty (DKK)",
            centralized["penalty_cost"],
            independent["penalty_cost"]
        )

        show(
            "Total external cost (DKK)",
            centralized["external_cost"],
            independent["external_cost"]
        )

        show(
            "SOCIAL COST (DKK)",
            centralized["community_cost"],
            independent["community_cost"]
        )

        show(
            "Manager budget surplus (DKK)",
            centralized["budget_surplus"],
            independent["budget_surplus"]
        )

        # 7. Additional checks
        c_exceed_hours = sum(
            e > 1e-7
            for e in centralized["excess"]
        )

        independent_exceed_hours = sum(
            e > 1e-7
            for e in independent["excess"]
        )

        feeder_diff_hours = [
            t for t in range(24)
            if abs(response["net"][t] - c_net[t]) > 1e-5
        ]

        print("\n=== STEP 3B SUMMARY ===")

        print(
            "Q1 C social cost:",
            f"{centralized['community_cost']:.8f} DKK"
        )

        print(
            "Independent-response social cost:",
            f"{independent['community_cost']:.8f} DKK"
        )

        print(
            "Social cost difference:",
            f"{independent['community_cost'] - centralized['community_cost']:+.8f} DKK"
        )

        print(
            "C hours exceeding capacity:",
            c_exceed_hours
        )

        print(
            "Independent hours exceeding capacity:",
            independent_exceed_hours
        )

        print(
            "Hours with different aggregate feeder net:",
            feeder_diff_hours
        )

        print(
            "Independent-response BB residual:",
            f"{independent['budget_surplus']:.8f} DKK"
        )

        print(
            "Q1 C reconstruction error against 372.162434:",
            f"{centralized['community_cost'] - 372.162434473684:+.3e} DKK"
        )

        print("\nIMPORTANT:")
        print(
            "This is ONE independent optimal-response selection."
        )
        print(
            "It is NOT a globally worst-case pessimistic response."
        )

        print("\nQ2 STEP 3B FINISHED")


    if __name__ == "__main__":
        compare_c_and_independent_responses()

