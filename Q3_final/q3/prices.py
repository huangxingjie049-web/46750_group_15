"""Price vectors produced by Q1 and Q2 (read only), and helpers to save / load Q3 price vectors.

    C   the dual of the internal balance of the central dispatch C (Q1_models/results/base_case/C/hourly.csv,
        column internal_price_DKK_per_kWh); supports the C dispatch in the optimistic sense (report, Section 2.3)
    DP  the reported DP / DP+BB price vector (Q2_final/Q2_2/dp_bb_start.sol, variables internal_price[t];
        community cost 373.5748 DKK in the report, Tables 6 and 8)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .paths import Q1_C_DIR, Q2_DIR


def c_dual_prices() -> np.ndarray:
    with open(Q1_C_DIR / "hourly.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["hour"]))
    return np.array([float(r["internal_price_DKK_per_kWh"]) for r in rows])


def dp_reported_prices() -> np.ndarray:
    pattern = re.compile(r"^internal_price\[(\d+)\]\s+(\S+)")
    x = {}
    for line in (Q2_DIR / "Q2_2" / "dp_bb_start.sol").read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            x[int(m.group(1))] = float(m.group(2))
    if sorted(x) != list(range(24)):
        raise ValueError("dp_bb_start.sol does not contain internal_price[0..23]")
    return np.array([x[t] for t in range(24)])


def save_prices(prices, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prices_DKK_per_kWh": [float(v) for v in prices], **(meta or {})}, indent=1),
                    encoding="utf-8")


def load_prices(name_or_path: str) -> tuple[str, np.ndarray]:
    """``C``, ``DP`` or the path of a JSON file written by :func:`save_prices`."""
    key = name_or_path.upper()
    if key == "C":
        return "C", c_dual_prices()
    if key == "DP":
        return "DP", dp_reported_prices()
    p = Path(name_or_path)
    return p.stem, np.array(json.loads(p.read_text(encoding="utf-8"))["prices_DKK_per_kWh"], dtype=float)
