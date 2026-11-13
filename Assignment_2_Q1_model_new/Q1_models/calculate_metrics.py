"""Recompute the six Q1 contract-value metrics from the saved base-case run."""

import argparse
import csv
from math import fsum, isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CASE = ROOT / "results" / "base_case"
TOL = 2e-6
HOURS = set(range(24))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def metered_import(row, variant):
    total = float(row["import_kWh"])
    if variant != "R":
        split = float(row["import_within_cap_kWh"]) + float(row["charged_exceedance_kWh"])
        if not isclose(total, split, rel_tol=0, abs_tol=TOL):
            raise ValueError(f"{variant}, hour {row['hour']}: metered import does not match its split")
    return total


def hourly_imports(base_case_dir, variant, hourly):
    """Use individual meters in R and I and the external community meter in C."""
    if variant == "C":
        return {t: metered_import(row, variant) for t, row in hourly.items()}

    imports = {t: [] for t in hourly}
    for row in read_csv(base_case_dir / variant / "members.csv"):
        imports[int(row["hour"])].append(metered_import(row, variant))
    if any(not values for values in imports.values()):
        raise ValueError(f"{variant}: missing individual meter readings")
    return {t: fsum(values) for t, values in imports.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-case-dir", type=Path, default=BASE_CASE)
    args = parser.parse_args()

    comparison = {row["variant"]: row
                  for row in read_csv(args.base_case_dir / "comparison.csv")}
    reference_cost = float(comparison["R"]["J_DKK"])
    individual_cost = float(comparison["I"]["J_DKK"])
    rows = []
    cap_profile = {}
    for variant in ("R", "I", "C"):
        saved = comparison[variant]
        records = read_csv(args.base_case_dir / variant / "hourly.csv")
        hourly = {int(row["hour"]): row for row in records}
        if hourly.keys() != HOURS or len(records) != len(HOURS):
            raise ValueError(f"{variant}: expected one feeder row per hour")
        caps = {t: float(row["cap_kW"]) for t, row in hourly.items()}
        if variant == "R":
            cap_profile = caps
        elif caps != cap_profile:
            raise ValueError(f"{variant}: cap profile differs from the reference")

        imports = hourly_imports(args.base_case_dir, variant, hourly)
        for t, imp in imports.items():
            if not isclose(imp, float(hourly[t]["import_kWh"]), rel_tol=0, abs_tol=TOL):
                raise ValueError(f"{variant}, hour {t}: feeder import differs from metered imports")
        exceedance = [max(0, imp - cap_profile[t]) for t, imp in imports.items()]
        cost = float(saved["J_DKK"])
        row = {
            "variant": variant,
            "J_DKK": cost,
            "gain_vs_R_DKK": reference_cost - cost,
            "gain_C_vs_I_DKK": individual_cost - cost if variant == "C" else None,
            "feeder_exceedance_kWh": fsum(exceedance),
            "peak_import_kW": max(imports.values()),
            "hours_above_cap": sum(value > TOL for value in exceedance),
        }
        for key in ("gain_vs_R_DKK", "feeder_exceedance_kWh",
                    "peak_import_kW", "hours_above_cap"):
            if not isclose(row[key], float(saved[key]), rel_tol=0, abs_tol=TOL):
                raise ValueError(f"{variant}: {key} disagrees with the saved base case")
        rows.append(row)

    output = ROOT / "results" / "contract_value_metrics.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified six contract-value metrics for R, I and C. Saved {output}")


if __name__ == "__main__":
    main()
