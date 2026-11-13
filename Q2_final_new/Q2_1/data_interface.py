
from pathlib import Path
import sys

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
Q1_DIR = PROJECT_ROOT / "Assignment_2_Q1_model_new" / "Q1_models"

# Import the data loader from our teammate's Q1 code.
sys.path.insert(0, str(Q1_DIR))

from main import load_data


def load_q1_data():
    """Load the original 16-member community data."""
    return load_data(Q1_DIR / "data")


def convert_member(q1_member):
    """
    Convert a Q1 member into the input format
    required by Q2 follower.py.
    """

    battery = None

    if q1_member.capacity > 0:
        battery = {
            "capacity": q1_member.capacity,
            "p_charge_max": q1_member.charge,
            "p_discharge_max": q1_member.discharge,
            "eta_charge": q1_member.rho_ch,
            "eta_discharge": q1_member.rho_dis,
            "initial_energy": q1_member.initial,
        }

    return {
        "reference": list(q1_member.ref),
        "load_min": list(q1_member.lower),
        "load_max": list(q1_member.upper),
        "pv_available": list(q1_member.pv),
        "min_daily_energy": q1_member.minimum,
        "pv_cost": q1_member.c,
        "disutility_rate": q1_member.cL,
        "battery": battery,
    }
