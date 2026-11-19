"""Checks of the Question 3 code (run from the repository root: python -m pytest Q3_final/tests -q)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q3 import evaluate as ev                       # noqa: E402
from q3 import follower_lp as fl                    # noqa: E402
from q3 import prices as pr                         # noqa: E402
from q3 import q1_bridge as qb                      # noqa: E402
from q3 import scenarios as sc                      # noqa: E402
from q3.paths import Q1_C_DIR                       # noqa: E402

J_C_Q1 = json.loads((Q1_C_DIR / "metrics.json").read_text(encoding="utf-8"))["J_DKK"]


@pytest.fixture(scope="module")
def base():
    return qb.load_base()


@pytest.fixture(scope="module")
def expected(base):
    return qb.scenario_data(base, sc.expected_scenario())


def test_expected_scenario_reproduces_q1_members(base, expected):
    for a, b in zip(base.members, expected.members):
        assert np.allclose(a.ref, b.ref) and np.allclose(a.lower, b.lower) and np.allclose(a.upper, b.upper)
        assert abs(a.minimum - b.minimum) < 1e-12 and a.cL == b.cL


@pytest.mark.parametrize("n,seed", [(5, sc.SEED_IN_SAMPLE), (10, sc.SEED_IN_SAMPLE), (100, sc.SEED_OUT_OF_SAMPLE)])
def test_scenario_checks(n, seed):
    c = sc.check_set(sc.draw(n, seed), seed)
    assert c["profile_within_bounds"] and c["indices_recover_draws"]
    assert c["emin_equals_reference_energy"] and c["probabilities_sum_to_one"]
    assert abs(c["profile_theta_mean"] - 1.0) < 0.05


def test_in_sample_nested_and_out_of_sample_disjoint():
    s5, s10, oos = sc.draw(5, 1), sc.draw(10, 1), sc.draw(100, 1000)
    assert sc.check_nested(s5, s10)
    ids = list(oos[0]["theta"]["profile"])
    assert not any(abs(a["theta"]["profile"][c] - b["theta"]["profile"][c]) < 1e-12
                   for a in s10 for b in oos for c in ids)


def test_central_dispatch_reproduces_q1_C(expected):
    k = ev.contract_terms(expected)
    j = ev.settle(fl.central_dispatch(expected, k), np.zeros(24), k)["J"]
    assert abs(j - J_C_Q1) < 1e-4


def test_optimistic_response_at_C_prices_reproduces_C(expected):
    df, _, _ = ev.evaluate(pr.c_dual_prices(), [expected], rule="optimistic")
    assert abs(df.J[0] - J_C_Q1) < 1e-4
    assert abs(df.regret[0]) < 1e-4


def test_selection_rules_are_ordered(expected):
    """optimistic <= independent <= pessimistic community cost at the same prices."""
    x = pr.dp_reported_prices()
    j = {r: ev.evaluate(x, [expected], rule=r)[0].J[0] for r in ("optimistic", "independent", "pessimistic")}
    assert j["optimistic"] <= j["independent"] + 1e-6 <= j["pessimistic"] + 2e-6


def test_every_member_is_optimal_in_joint_selection(expected):
    k = ev.contract_terms(expected)
    x = pr.c_dual_prices()
    r = fl.community_response(expected, x, k, "pessimistic")
    for i, v in r["member_optimal_values"].items():
        assert r["members"][i]["objective"] <= v + r["value_tol"] + 1e-6


def test_sdp_kkt_model_at_fixed_C_prices(base):
    """The SDP model (Q2 KKT systems, N = 1, theta = 1, cap only) with the prices fixed at the C duals returns the
    C cost: the KKT reformulation selects the optimistic best response. Needs a Gurobi licence (skipped otherwise)."""
    gp = pytest.importorskip("gurobipy")
    from q3 import sdp
    cfg = sdp.SDPConfig(name="test_fixed_C", bb="none", time_limit=120, warm_start="none")
    try:
        m, h = sdp.build(base, [sc.expected_scenario()], cfg)
    except gp.GurobiError as err:
        pytest.skip(f"Gurobi licence: {err}")
    x = pr.c_dual_prices()
    for t in range(24):
        h["x"][t].LB = h["x"][t].UB = float(x[t])
    m.Params.OutputFlag = 0
    m.optimize()
    assert m.SolCount > 0
    assert abs(m.ObjVal - J_C_Q1) < 1e-3
