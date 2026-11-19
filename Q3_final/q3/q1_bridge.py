"""Bridge between the scenario generator of the starter package and the member data of Q1/Q2.

Q1 (``Q1_models/main.py: load_data``) represents every member as a ``SimpleNamespace`` with the fields
``id, cL, ref, minimum, lower, upper, pv, c, capacity, charge, discharge, rho_ch, rho_dis, initial`` (kWh, kW, DKK).
A preference scenario of Question 3 changes only the load part of a member:

    ref_t     = L_max * rho_t ** theta_profile          (kWh)
    minimum   = sum_t ref_t                              (kWh, the daily minimum follows the profile)
    lower_t   = max(L_min, ref_t - D),  upper_t = min(L_max, ref_t + D)   (kWh, D = 1 kWh max hourly deviation)
    cL        = cL_base * theta_cL                       (DKK/kWh, only when cL is uncertain)

The formulas are those of ``load_data`` so that the scenario theta = 1 reproduces the Q1 members exactly
(checked in ``tests/test_q3_scenarios.py``).
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

from .paths import Q1_DATA_DIR, Q1_DIR, load_module

_Q1 = None


def q1_module():
    """The Q1 module (``Q1_models/main.py``), loaded once by path."""
    global _Q1
    if _Q1 is None:
        _Q1 = load_module("q1_main", Q1_DIR / "main.py")
    return _Q1


def load_base() -> SimpleNamespace:
    """The base case of Q1: 16 members, day d1, prices, tariffs and contract (unchanged Q1 loader)."""
    return q1_module().load_data(Q1_DATA_DIR)


def _load_dict(scenario_member: dict) -> dict:
    """The flexible-load appliance dict of one member in one scenario (exactly one per member)."""
    loads = [a for a in scenario_member.values() if "load_id" in a]
    if len(loads) != 1:
        raise ValueError("every member must have exactly one flexible load")
    return loads[0]


def scenario_data(base: SimpleNamespace, scenario: dict) -> SimpleNamespace:
    """A copy of the Q1 data in which every member's load follows one joint preference scenario.

    Parameters
    ----------
    base : the output of :func:`load_base` (not modified).
    scenario : one joint scenario of ``scenario_generation.merge_scenarios`` (keys ``members``, ``theta``, ``id``).

    Returns
    -------
    SimpleNamespace with the same fields as ``base``; ``members`` are modified copies, plus ``scenario_id`` and
    ``theta`` (the scaling parameters used).
    """
    out = copy.copy(base)
    members = []
    for m in base.members:
        load = _load_dict(scenario["members"][m.id])
        lmax = float(load["max_load_KWh_per_hour"])
        lmin = float(load["min_load_KWh_per_hour"])
        dev = float(load["max_hourly_deviation_kWh"])
        ref = [lmax * float(r) for r in load["reference_load_capacity_factor"]]
        minimum = sum(ref)
        if abs(minimum - float(load["min_total_energy_per_day_kWh"])) > 1e-9:
            raise AssertionError(f"{m.id}: daily minimum does not equal the scenario's reference energy")
        new = copy.copy(m)
        new.ref = ref
        new.minimum = minimum
        new.lower = [max(lmin, r - dev) for r in ref]
        new.upper = [min(lmax, r + dev) for r in ref]
        new.cL = float(load["linear_disutility_DKK_per_kWh"])
        members.append(new)
    out.members = members
    out.scenario_id = scenario.get("id")
    out.theta = scenario.get("theta")
    return out
