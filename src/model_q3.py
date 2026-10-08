"""Q3(d): extend Q2(c) with sum(load) >= E_min; no battery yet.

Load variables are ENERGY consumed in each one-hour interval (kWh), so their
daily sum needs no extra time-step factor. The only added constraint is daily.
Objective and all hourly constraints are inherited from Q2(c).
"""
import gurobipy as gp
import numpy as np
from .model_q2_quadratic import QuadraticConsumerModel


class DailyEnergyConsumerModel(QuadraticConsumerModel):
    """Convex QP matching report equations (29a)--(29e)."""

    def build(self):
        if self.var or self.con:
            raise RuntimeError('build() may only be called once; create a fresh model per run.')
        return super().build()

    def _check_intertemporal_inputs(self):
        d = self.data
        if d.battery_capacity_kWh is not None:
            raise ValueError('This is the no-battery model of Q3(d); Q3(g) is not implemented here.')
        if d.min_daily_energy_kWh is None or not np.isfinite(d.min_daily_energy_kWh):
            raise ValueError('Q3 requires a finite minimum daily energy.')
        if d.min_daily_energy_kWh < 0 or d.min_daily_energy_kWh > d.n_hours*d.load_max_kWh:
            raise ValueError('Minimum daily energy must be between 0 and n_hours * load_max_kWh.')
        if d.max_import_kW is not None or d.max_export_kW is not None:
            raise ValueError('This formulation assumes unrestricted grid trading, as in the assignment.')

    def _add_intertemporal_constraints(self):
        self.con['minimum_energy'] = self.m.addConstr(
            gp.quicksum(self.var['load'][t] for t in self.T) >= self.data.min_daily_energy_kWh,
            name='minimum_energy')

    def _extract_results(self, status):
        result = super()._extract_results(status)
        d, h = self.data, result.hourly
        pi = result.duals['minimum_energy']
        # Maximize W with sum(L) >= E_min: Gurobi Pi = dW*/dE_min <= 0.
        # The report's nonnegative multiplier is therefore mu = -Pi.
        result.meta.update({
            'minimum_daily_energy': float(d.min_daily_energy_kWh),
            'reference_daily_energy': float(d.reference_load.sum()),
            'minimum_energy_slack_kWh': float(h['load'].sum()-d.min_daily_energy_kWh),
            'minimum_energy_binding': bool(abs(h['load'].sum()-d.min_daily_energy_kWh) <= 1e-6),
            'minimum_energy_gurobi_pi': float(pi),
            'minimum_energy_multiplier_mu': float(-pi),
            'above_reference_energy_kWh': float(h['deviation'].clip(lower=0).sum()),
            'below_reference_energy_kWh': float((-h['deviation']).clip(lower=0).sum()),
            'variables': int(self.m.NumVars),
            'linear_constraints': int(self.m.NumConstrs),
        })
        return result
