"""Folder layout of the repository, as seen from Question 3.

Q3_final/ sits next to Q1_models/ and Q2_final/ at the repository root. Question 3 READS the Q1 and Q2 code and
outputs (it never modifies them):

    Q1_models/main.py                       load_data(): the 16 members of the base case (d1)
    Q1_models/data/                         the supplied JSON inputs (identical to the starter package)
    Q1_models/results/base_case/C/          central dispatch C: hourly.csv (dual prices), metrics.json
    Q2_final/Q2_2/kkt_member_builder.py     add_member_kkt(): one member's KKT system inside a Gurobi model
    Q2_final/Q2_2/dp_bb_start.sol           the reported DP / DP+BB price vector (Gurobi solution file)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

Q3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Q3_DIR.parent


def _pick(env: str, *candidates: Path) -> Path:
    """Folder of a teammate's part: the environment variable ``env`` if set, else the first existing candidate."""
    if os.environ.get(env):
        return Path(os.environ[env]).resolve()
    return next((c for c in candidates if c.exists()), candidates[-1])


# The updated Q1 (13 Nov) and Q2 live in Assignment_2_Q1_model_new/Q1_models and Q2_final_new; the original layout
# (Q1_models, Q2_final) is used if those folders do not exist. Override with Q3_Q1_DIR / Q3_Q2_DIR.
Q1_DIR = _pick("Q3_Q1_DIR", REPO_ROOT / "Assignment_2_Q1_model_new" / "Q1_models", REPO_ROOT / "Q1_models")
Q1_DATA_DIR = Q1_DIR / "data"
Q1_C_DIR = Q1_DIR / "results" / "base_case" / "C"
Q2_DIR = _pick("Q3_Q2_DIR", REPO_ROOT / "Q2_final_new", REPO_ROOT / "Q2_final")


def describe() -> str:
    rel = lambda p: p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
    return f"Q1: {rel(Q1_DIR)}   Q2: {rel(Q2_DIR)}"
Q3_DATA_DIR = Q3_DIR / "data"
SCENARIO_DIR = Q3_DIR / "scenarios"
RESULTS_DIR = Q3_DIR / "results"


def load_module(name: str, path: Path):
    """Import a module from an explicit file path under a unique name.

    The Q1 and Q2 folders both use generic module names (``main``, ``follower``); loading them by path avoids
    collisions with the repository-root ``main.py``.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
