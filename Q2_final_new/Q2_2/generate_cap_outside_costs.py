
from pathlib import Path
import sys
import json

THIS_DIR = Path(__file__).resolve().parent
Q1_DIR = (
    THIS_DIR.parent.parent
    / "Assignment_2_Q1_model_new"
    / "Q1_models"
)

sys.path.insert(0, str(Q1_DIR))

from main import CommunityModel, load_data, net_import_shares


def member_costs(result, data, variant, beta, alpha):
    """Reconstruct individual costs from Q1 dispatch."""
    costs = {member.id: 0.0 for member in data.members}
    members = {member.id: member for member in data.members}

    discount = 1.0 if variant == "R" else 1.0 - beta

    for row in result.members:
        i = row["member"]
        t = row["hour"]
        member = members[i]

        physical = (
            member.cL * row["deviation_kWh"]
            + member.c * row["PV_kWh"]
        )

        imp = row["import_kWh"]
        exp = row["export_kWh"]


        if variant == "R":
            grid = (
                (data.prices[t] + data.tau_imp) * imp
                - (data.prices[t] - data.tau_exp) * exp
            )
        else:
            imp_cap = row["import_within_cap_kWh"]
            excess = row["charged_exceedance_kWh"]

            grid = (
                (data.prices[t] + discount * data.tau_imp)
                * imp_cap
                - (data.prices[t] - discount * data.tau_exp)
                * exp
                + (data.prices[t] + alpha) * excess
            )


        costs[i] += physical + grid

    return costs


def main():
    data = load_data(Q1_DIR / "data")

    beta = float(data.contract["tariff_discount_ratio"])
    alpha = float(data.contract["penalty_DKK_per_kWh"])

    reference = CommunityModel(
        data, variant="R", terminal="minimum"
    ).run()

    shares = net_import_shares(reference)
    r_costs = member_costs(reference, data, "R", beta, alpha)

    assert abs(
        sum(r_costs.values()) - reference.metrics["total_cost_DKK"]
    ) < 1e-5

    for multiplier in [0.90, 1.00, 1.10]:
        scaled_caps = [
            multiplier * float(cap)
            for cap in data.contract["capacity_limit_kW"]
        ]

        individual = CommunityModel(
            data,
            variant="I",
            shares=shares,
            cap=scaled_caps,
            terminal="minimum"
        ).run()

        i_costs = member_costs(
            individual, data, "I", beta, alpha
        )

        assert abs(
            sum(i_costs.values())
            - individual.metrics["total_cost_DKK"]
        ) < 1e-5

        outside = {
            i: min(r_costs[i], i_costs[i])
            for i in r_costs
        }

        filename = (
            f"outside_costs_RI_cap_{multiplier:.2f}"
            .replace(".", "p") + ".json"
        )

        output_file = THIS_DIR / filename

        output_file.write_text(
            json.dumps(outside, indent=2),
            encoding="utf-8"
        )

        print(f"\nCAP_MULTIPLIER = {multiplier:.2f}")
        print(f"R total cost = {sum(r_costs.values()):.6f} DKK")
        print(f"I total cost = {sum(i_costs.values()):.6f} DKK")
        print(f"R/I outside sum = {sum(outside.values()):.6f} DKK")
        print(f"Saved: {output_file.name}")

        if abs(multiplier - 1.0) < 1e-9:
            assert abs(
                sum(outside.values()) - 392.45040163
            ) < 1e-4, "Baseline R/I costs do not match"


if __name__ == "__main__":
    main()
