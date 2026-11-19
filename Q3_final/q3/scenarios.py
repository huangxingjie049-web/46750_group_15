"""Question 3: drawing, checking and saving the preference scenarios.

All draws go through the starter module ``scenario_generation.py`` (copied unchanged into ``q3/``), with the
hyper-parameters of ``data/preference_uncertainty_params.json``. Base study: the reference profile of every member is
uncertain (rho_t ** theta, theta ~ Uniform[0.8, 1.2], independent per member, aligned merging); cL is fixed.

Seeds (report them):
    in-sample      seed 1     N = 5 and N = 10 are NESTED: the first 5 scenarios of N = 10 are the N = 5 set
                              (each member has its own random stream; checked in check_nested)
    out-of-sample  seed 1000  100 joint scenarios, the same set for every model compared (common random numbers)
The two seeds must differ: with the same seed, the first out-of-sample scenarios would equal the in-sample ones.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from . import scenario_generation as sg
from .paths import Q1_DATA_DIR, Q3_DATA_DIR

SEED_IN_SAMPLE = 1
SEED_OUT_OF_SAMPLE = 1000
IN_SAMPLE_SIZES = (5, 10)
OUT_OF_SAMPLE_SIZE = 100
UNCERTAINTY_FILE = "preference_uncertainty_params.json"


def raw_inputs() -> tuple[dict, dict, dict]:
    """(appliance catalogue, community, uncertainty rules) as raw dicts; the catalogue holds the d1 profiles."""
    app = sg.load_json("appliance_params.json", Q1_DATA_DIR)
    community = sg.load_json("params_community_base.json", Q1_DATA_DIR)
    rules = sg.load_json(UNCERTAINTY_FILE, Q3_DATA_DIR)
    return app, community, rules


def make_spec(rules: dict, community: dict, overrides: dict | None = None) -> dict:
    """Hyper-parameter specification {quantity: {consumer_id: BetaRule}}; ``overrides`` = {quantity: {lo, hi, mean, kappa}}
    applied to all members (e.g. {"profile": {"kappa": 8}} for a less uncertain community)."""
    ids = [c["consumer_id"] for c in community["consumers"]]
    spec = sg.rules_to_spec(rules, ids)
    for q, kw in (overrides or {}).items():
        sg.set_rule(spec, q, None, **kw)
    return spec


def draw(n: int, seed: int, quantities=("profile",), overrides: dict | None = None) -> list[dict]:
    """n equiprobable joint scenarios (aligned merging) of the base community, d1."""
    app, community, rules = raw_inputs()
    spec = make_spec(rules, community, overrides)
    return sg.draw_preference_scenarios(app, community, rules, n=n, seed=seed, quantities=tuple(quantities), spec=spec)


def expected_scenario() -> dict:
    """The deterministic baseline: theta = 1 for every member and quantity (the catalogue unchanged)."""
    app, community, _ = raw_inputs()
    ids = [c["consumer_id"] for c in community["consumers"]]
    members = sg.expected_preference_scenario(app, community)
    return {"id": "expected", "probability": 1.0, "index": {},
            "theta": {q: {cid: 1.0 for cid in ids} for q in sg.QUANTITIES}, "members": members}


# ----------------------------------------------------------------------------------------------------------------------
# Checks required by the assignment
# ----------------------------------------------------------------------------------------------------------------------
def _load(member: dict) -> dict:
    return next(a for a in member.values() if "load_id" in a)


def check_set(scenarios: list[dict], seed: int, quantities=("profile",), overrides: dict | None = None,
              tol: float = 1e-9) -> dict:
    """Validate one scenario set. Returns a dict of check results (raises nothing; the caller asserts).

    * theta inside [lo, hi] for every uncertain (quantity, member);
    * sample mean of theta vs the set mean (reported, not asserted: it converges with n);
    * every scenario's daily minimum equals its reference energy;
    * the indices stored in each joint scenario recover the independent draws (re-drawn with the same seed);
    * probabilities sum to one.
    """
    app, community, rules = raw_inputs()
    ids = [c["consumer_id"] for c in community["consumers"]]
    spec = make_spec(rules, community, overrides)
    for q in sg.QUANTITIES:
        sg.set_rule(spec, q, None, n_scenarios=(len(scenarios) if q in quantities else 1))
    scaling = sg.draw_scaling_scenarios(copy.deepcopy(spec), seed)
    out = {"n": len(scenarios), "seed": seed, "quantities": list(quantities)}
    for q in quantities:
        th = np.array([[s["theta"][q][cid] for cid in ids] for s in scenarios])
        lo = min(spec[q][c].lo for c in ids); hi = max(spec[q][c].hi for c in ids)
        out[f"{q}_theta_min"] = float(th.min()); out[f"{q}_theta_max"] = float(th.max())
        out[f"{q}_theta_mean"] = float(th.mean()); out[f"{q}_set_mean"] = float(spec[q][ids[0]].mean)
        out[f"{q}_within_bounds"] = bool(th.min() >= lo - tol and th.max() <= hi + tol)
    out["indices_recover_draws"] = all(
        abs(s["theta"][q][cid] - float(scaling[q][cid][s["index"][(q, cid)]])) < tol
        for s in scenarios for q in sg.QUANTITIES for cid in ids)
    out["emin_equals_reference_energy"] = all(
        abs(_load(s["members"][cid])["min_total_energy_per_day_kWh"]
            - _load(s["members"][cid])["max_load_KWh_per_hour"] * sum(_load(s["members"][cid])["reference_load_capacity_factor"])) < tol
        for s in scenarios for cid in ids)
    out["probabilities_sum_to_one"] = abs(sum(s["probability"] for s in scenarios) - 1.0) < tol
    return out


def check_nested(small: list[dict], large: list[dict], quantities=("profile",)) -> bool:
    """True if the first len(small) scenarios of ``large`` have the same theta as ``small`` (nested in-sample sets)."""
    ids = list(small[0]["theta"]["profile"])
    return all(abs(a["theta"][q][c] - b["theta"][q][c]) < 1e-12
               for a, b in zip(small, large) for q in quantities for c in ids)


# ----------------------------------------------------------------------------------------------------------------------
# Saving (the generated scenarios are part of the code deliverable)
# ----------------------------------------------------------------------------------------------------------------------
def to_record(scenarios: list[dict], name: str, seed: int, quantities, overrides: dict | None = None) -> dict:
    """JSON-serialisable description of a set: seeds, hyper-parameters, theta, indices and each member's daily
    reference energy (kWh). The full member data are re-derived from theta with ``scenario_generation``."""
    app, community, rules = raw_inputs()
    spec = make_spec(rules, community, overrides)
    hyper = {q: {k: v for k, v in vars(next(iter(spec[q].values()))).items() if k != "n_scenarios"}
             for q in quantities}
    return {
        "name": name, "seed": seed, "n": len(scenarios), "uncertain_quantities": list(quantities),
        "hyper_parameters": hyper, "overrides": overrides or {}, "merging": "aligned",
        "generator": "q3/scenario_generation.py (starter package, unchanged)",
        "scenarios": [{
            "id": s["id"], "probability": s["probability"], "theta": s["theta"],
            "index": {f"{q}|{c}": k for (q, c), k in s["index"].items()},
            "daily_reference_energy_kWh": {c: _load(m)["min_total_energy_per_day_kWh"] for c, m in s["members"].items()},
        } for s in scenarios],
    }


def save(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
