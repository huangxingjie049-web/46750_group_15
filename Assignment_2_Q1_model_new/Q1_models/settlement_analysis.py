"""Assignment 2, Question 1, step 6.

Run: python settlement_analysis.py
Reads the saved R, I and C base-case dispatch and duals. Uses Python's
standard library and writes the settlement results to results/settlement/.
"""

import argparse
import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT
TOL = 2e-6
MODELS = ("R", "I", "C")
BENCHMARKS = ("R", "I")
COMPONENTS = (
    "purchase_DKK", "sale_revenue_DKK", "penalty_DKK",
    "pv_cost_DKK", "load_deviation_cost_DKK",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_close(label, actual, expected):
    if not isclose(actual, expected, rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"{label}: calculated {actual:.9f}, expected {expected:.9f}")


def member_parameters(folder):
    """Use the input catalogue to identify assets and cost coefficients."""
    catalogue = read_json(folder / "appliance_params.json")
    community = read_json(folder / "params_community_base.json")
    loads = {row["load_id"]: row for row in catalogue["loads"]}
    pv_units = {row["DER_id"]: row for row in catalogue["DERs"]}
    batteries = {row["storage_id"] for row in catalogue["storages"]}
    members = {}
    for row in community["consumers"]:
        assets = row["list_appliances"]
        load = next(loads[key] for key in assets if key in loads)
        pv = next((pv_units[key] for key in assets if key in pv_units), None)
        category = ["Load"]
        if pv is not None:
            category.append("PV")
        if batteries.intersection(assets):
            category.append("battery")
        members[row["consumer_id"]] = (
            " + ".join(category),
            pv["marginal_cost_DKK_per_kWh"] if pv is not None else 0.0,
            load["linear_disutility_DKK_per_kWh"],
        )
    return members


def member_costs(base, members, tariffs, internal_price, prices, alpha):
    """Settle each model's own dispatch and retain its cost components."""
    costs = {}
    hourly_net = dict.fromkeys(internal_price, 0.0)
    member_hourly = []
    for model in MODELS:
        accounts = {i: dict.fromkeys(COMPONENTS, 0.0) for i in members}
        dispatch = read_csv(base / model / "members.csv")
        keys = {(row["member"], int(row["hour"])) for row in dispatch}
        expected = {(i, t) for i in members for t in internal_price}
        if keys != expected or len(keys) != len(dispatch):
            raise ValueError(f"{model}: incomplete or duplicate member-hour dispatch")
        for row in dispatch:
            i, t = row["member"], int(row["hour"])
            _, pv_rate, deviation_rate = members[i]
            account = accounts[i]
            account["pv_cost_DKK"] += pv_rate * float(row["PV_kWh"])
            deviation = float(row["d_plus_kWh"]) + float(row["d_minus_kWh"])
            account["load_deviation_cost_DKK"] += deviation_rate * deviation
            if model == "C":
                net = float(row["net_kWh"])
                physical_net = (float(row["L_kWh"]) - float(row["PV_kWh"])
                                + float(row["charge_kW"]) - float(row["discharge_kW"]))
                check_close(f"C net load {i}, hour {t}", net, physical_net)
                hourly_net[t] += net
                buy_price = sell_price = internal_price[t]
                imports, exports = max(net, 0.0), max(-net, 0.0)
                member_hourly.append({
                    "member": i,
                    "hour": t,
                    "net_import_kWh": net,
                    "internal_price_DKK_per_kWh": internal_price[t],
                    "net_payment_DKK": internal_price[t] * net,
                })
            else:
                buy_price, sell_price = tariffs[model][t]
                imports = float(row["import_kWh"])
                exports = float(row["export_kWh"])
                check_close(f"{model} meter balance {i}, hour {t}",
                            imports - exports, float(row["net_kWh"]))
                if model == "I":
                    within = float(row["import_within_cap_kWh"])
                    excess = float(row["charged_exceedance_kWh"])
                    check_close(f"I import split {i}, hour {t}", within + excess, imports)
                    account["purchase_DKK"] += prices[t] * excess
                    account["penalty_DKK"] += alpha * excess
                    imports = within
            account["purchase_DKK"] += buy_price * imports
            account["sale_revenue_DKK"] += sell_price * exports

        for account in accounts.values():
            account["net_electricity_bill_DKK"] = (
                account["purchase_DKK"] - account["sale_revenue_DKK"]
                + account["penalty_DKK"]
            )
            account["total_cost_DKK"] = (
                account["net_electricity_bill_DKK"] + account["pv_cost_DKK"]
                + account["load_deviation_cost_DKK"]
            )
        costs[model] = accounts
    return costs, hourly_net, member_hourly


def manager_budget(dispatch, dual, internal_price, hourly_net, tariffs, prices, alpha):
    """Check the manager's hourly cash balance and capacity rent."""
    hourly = []
    for t, row in sorted(dispatch.items()):
        price = internal_price[t]
        within = float(row["import_within_cap_kWh"])
        excess = float(row["charged_exceedance_kWh"])
        exports = float(row["export_kWh"])
        cap = float(row["cap_kW"])
        chi = -dual[f"cap[{t}]"]
        buy_price, sell_price = tariffs["I"][t]
        check_close(f"C total import, hour {t}", within + excess, float(row["import_kWh"]))
        check_close(f"C community balance, hour {t}", hourly_net[t], within + excess - exports)

        internal = price * hourly_net[t]
        purchase = buy_price * within + prices[t] * excess
        sale = sell_price * exports
        penalty = alpha * excess
        external = purchase - sale + penalty
        surplus = internal - external
        rent = chi * cap  # One-hour intervals, as in the model.
        check_close(f"Manager surplus equals capacity rent, hour {t}", surplus, rent)
        hourly.append({
            "hour": t,
            "internal_price_DKK_per_kWh": price,
            "capacity_dual_DKK_per_kW": chi,
            "cap_kW": cap,
            "import_within_cap_kWh": within,
            "exceedance_kWh": excess,
            "export_kWh": exports,
            "member_net_payments_DKK": internal,
            "external_purchase_DKK": purchase,
            "external_sale_revenue_DKK": sale,
            "exceedance_penalty_DKK": penalty,
            "external_net_cost_DKK": external,
            "manager_surplus_DKK": surplus,
            "capacity_rent_DKK": rent,
        })

    keys = (
        "member_net_payments_DKK", "external_purchase_DKK", "external_sale_revenue_DKK",
        "exceedance_penalty_DKK", "external_net_cost_DKK",
        "manager_surplus_DKK", "capacity_rent_DKK",
    )
    summary = {key: sum(row[key] for row in hourly) for key in keys}
    surplus = summary["manager_surplus_DKK"]
    summary["strictly_budget_balanced"] = abs(surplus) <= TOL
    summary["external_cost_covered"] = surplus >= -TOL
    return hourly, summary


def individual_rationality(members, costs):
    individual, by_type = [], {}
    for i, (category, _, deviation_rate) in members.items():
        row = {"member": i, "consumer_type": category, "load_deviation_rate_DKK_per_kWh": deviation_rate}
        for model in MODELS:
            row.update({f"{model}_{key}": value for key, value in costs[model][i].items()})
        for model in BENCHMARKS:
            gain = row[f"{model}_total_cost_DKK"] - row["C_total_cost_DKK"]
            row[f"gain_vs_{model}_DKK"] = gain
            row[f"IR_vs_{model}"] = gain >= -TOL
        row["IR_vs_best_alternative"] = all(row[f"IR_vs_{m}"] for m in BENCHMARKS)
        individual.append(row)
        by_type.setdefault(category, []).append(row)

    groups = []
    for category, rows in by_type.items():
        group = {"consumer_type": category, "member_count": len(rows)}
        for model in BENCHMARKS:
            group[f"IR_vs_{model}_count"] = sum(row[f"IR_vs_{model}"] for row in rows)
            group[f"mean_gain_vs_{model}_DKK"] = mean(row[f"gain_vs_{model}_DKK"] for row in rows)
        group["IR_vs_best_alternative_count"] = sum(row["IR_vs_best_alternative"] for row in rows)
        groups.append(group)
    return individual, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Q1 model directory containing data and results/base_case")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "settlement")
    args = parser.parse_args()
    inputs = args.model_dir / "data"
    base = args.model_dir / "results" / "base_case"

    bus = read_json(inputs / "bus_params.json")[0]
    contract = read_json(inputs / "dso_contract_params.json")
    alpha = contract["penalty_DKK_per_kWh"]
    members = member_parameters(inputs)
    prices = bus["energy_price_DKK_per_kWh"]
    fractions = {"R": 1.0, "I": 1.0 - contract["tariff_discount_ratio"]}
    tariffs = {
        model: [(p + fraction * bus["import_tariff_DKK_per_kWh"],
                 p - fraction * bus["export_tariff_DKK_per_kWh"]) for p in prices]
        for model, fraction in fractions.items()
    }
    dispatch = read_csv(base / "C" / "hourly.csv")
    community = {int(row["hour"]): row for row in dispatch}
    if set(community) != set(range(len(prices))) or len(community) != len(dispatch):
        raise ValueError("Incomplete or duplicate community hourly dispatch")
    dual = {row["name"]: float(row["Pi"]) for row in read_csv(base / "C" / "duals.csv")}
    internal_price = {t: -dual[f"community_balance[{t}]"] for t in community}
    for t, price in internal_price.items():
        check_close(f"Internal price from (17c), hour {t}", price,
                    float(community[t]["internal_price_DKK_per_kWh"]))

    costs, hourly_net, member_hourly = member_costs(
        base, members, tariffs, internal_price, prices, alpha
    )
    hourly, budget = manager_budget(
        community, dual, internal_price, hourly_net, tariffs, prices, alpha
    )
    individual, groups = individual_rationality(members, costs)
    objectives = {row["variant"]: float(row["J_DKK"]) for row in read_csv(base / "comparison.csv")}
    check_close("Manager net external cost matches the community dispatch",
                budget["external_net_cost_DKK"], read_json(base / "C" / "metrics.json")["grid_cost_DKK"])
    for model in MODELS:
        total = sum(row["total_cost_DKK"] for row in costs[model].values())
        expected = objectives[model]
        if model == "C":
            expected += budget["manager_surplus_DKK"]
        check_close(f"{model} member costs reconcile to the objective", total, expected)
    check_close("Manager receipts equal total member bills",
                budget["member_net_payments_DKK"],
                sum(row["net_electricity_bill_DKK"] for row in costs["C"].values()))
    for model in BENCHMARKS:
        budget[f"IR_vs_{model}_count"] = sum(row[f"IR_vs_{model}"] for row in individual)
    budget["IR_vs_best_alternative_count"] = sum(row["IR_vs_best_alternative"] for row in individual)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "hourly_settlement.csv", hourly)
    write_csv(out / "member_hourly_settlement.csv", member_hourly)
    write_csv(out / "member_settlement.csv", individual)
    write_csv(out / "consumer_type_results.csv", groups)
    (out / "budget_summary.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")

    print(f"Manager surplus: {budget['manager_surplus_DKK']:.3f} DKK/day")
    for model in BENCHMARKS:
        print(f"Individual rationality vs {model}: {budget[f'IR_vs_{model}_count']}/{len(members)}")
    print(f"Results saved to {out.resolve()}")


if __name__ == "__main__":
    main()
