"""Compare freshly solved numerical results with the report; no image-file reads."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .model import write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "report_reference"
ALIASES = {"1e": "1f", "3d": "3e"}
REPORT_TASKS = ["1f", "2b", "2c", "2d", "2e", "3e", "3f", "3g", "3g-sensitivity"]


# Compare CSV selectors numerically where appropriate, without relying on string rounding.
def _select(frame, selector):
    chosen = frame
    for name, value in selector.items():
        if isinstance(value, (int, float)):
            chosen = chosen[
                np.isclose(chosen[name].astype(float), value, rtol=0, atol=1e-9)
            ]
        else:
            chosen = chosen[chosen[name] == value]
    if len(chosen) != 1:
        raise ValueError(f"Expected one row for {selector}; found {len(chosen)}")
    return chosen.iloc[0]


# Expected report values are only read for comparison; no solver result is overwritten.
def verify_report(results_root, tasks=None, quick=False):
    """Compare actual solver outputs with the report's printed numerical values.

    No figure is read or copied. main.py renders figures after this succeeds.
    Partial and quick runs remain explicitly identified in the manifest.
    """
    parent = Path(results_root)
    out = parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((REFERENCE / "contract.json").read_text(encoding="utf-8"))
    selected = REPORT_TASKS if tasks is None else list(tasks)
    actual = {ALIASES.get(task, task): task for task in selected}
    selected_set = set(actual)
    cache = {}
    rows = []
    missing = []
    skipped = []

    # Load real run files once and compare at the precision printed in the report.
    for check in contract["checks"]:
        canonical = check["file"].split("/")[0]
        if canonical not in selected_set:
            continue
        if quick and check["file"].endswith("table17_plateaus.csv"):
            skipped.append(check["name"])
            continue
        relative = Path(actual[canonical]).joinpath(*Path(check["file"]).parts[1:])
        path = parent / relative
        if not path.is_file():
            missing.append(str(relative))
            continue
        if path not in cache:
            cache[path] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".json"
                else pd.read_csv(path)
            )
        payload = cache[path]
        if path.suffix == ".json":
            value = payload[check["field"]]
        elif check.get("aggregate") == "max_abs":
            value = payload[check["field"]].abs().max()
        else:
            selected_row = _select(payload, check["selector"])
            value = selected_row[check["field"]]
            if check.get("divide_by"):
                value /= selected_row[check["divide_by"]]
        if check.get("subtract_file"):
            other = json.loads((parent / check["subtract_file"]).read_text())
            value -= other[check["field"]]
        value = float(value)
        error = abs(value - check["reported"])
        rows.append(
            {
                "checkpoint": check["name"],
                "source_tex": check["source_tex"],
                "computed_file": str(relative),
                "field": check["field"],
                "reported": check["reported"],
                "computed": value,
                "absolute_difference": error,
                "tolerance": check["tolerance"],
                "passed": bool(np.isfinite(value) and error <= check["tolerance"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "numerical_match.csv", index=False)
    failures = [row for row in rows if not row["passed"]]
    summary = {
        "authority": contract["authority"],
        "selected_tasks": selected,
        "complete_report_scope": set(REPORT_TASKS) <= selected_set and not quick,
        "checked_count": len(rows),
        "skipped_count": len(skipped),
        "all_checked_numbers_match": bool(rows and not failures and not missing),
        "missing_files": sorted(set(missing)),
        "failures": failures,
        "quick_skipped_checkpoints": skipped,
        "known_report_conflicts": contract["known_conflicts"],
        "image_method": contract["image_method"],
        "figures": [],
        "figure_count": 0,
    }
    write_json(out / "manifest.json", summary)

    # Failed numerical comparisons stop the CLI before any report drawing begins.
    if missing:
        raise RuntimeError(f"Report outputs missing: {sorted(set(missing))}")
    if failures or not rows:
        raise RuntimeError(
            f"Report numerical comparison failed: {failures or 'no matching checkpoints'}"
        )

    return summary
