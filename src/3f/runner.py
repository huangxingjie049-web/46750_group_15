"""Q3(f): daily-energy and quadratic-disutility sensitivity experiments.

Run through main.py --task 3f. All input files are read-only.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_question
from src.model_q3 import DailyEnergyConsumerModel
from src.model_q2_quadratic import QuadraticConsumerModel
from src.validation_q3 import validate_solution

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMIN = [20, 22.765, 25, 32.34, 40, 48]
DEFAULT_CQ = [0.1, 0.5, 1, 2, 5, 10]
TOL = 2e-6


def _values(values, name, lower, upper=None, strictly_positive=False):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 3 or
            not np.isfinite(values).all() or len(np.unique(values)) != len(values)):
        raise ValueError(f'{name}: supply at least three distinct finite values.')
    if np.any(values <= lower if strictly_positive else values < lower):
        raise ValueError(f'{name}: values are outside the allowed lower bound.')
    if upper is not None and np.any(values > upper):
        raise ValueError(f'{name}: values exceed feasible daily consumption {upper:g}.')
    return sorted(values.tolist())


def _solve(data, out, name, checks):
    model_class = (QuadraticConsumerModel if data.min_daily_energy_kWh is None
                   else DailyEnergyConsumerModel)
    model = model_class(data).build()
    # At E_min equal to unconstrained energy, barrier can stop with tiny
    # artificial slack. Primal simplex resolves this boundary accurately.
    model.m.Params.Method = 0
    result = model.solve()
    check = validate_solution(result, data)
    checks[name] = check
    if not check['passed']:
        raise RuntimeError(f'{name}: validation failed: {check}')
    folder = out / 'runs' / name
    folder.mkdir(parents=True, exist_ok=True)
    result.save(folder)
    model.m.write(str(folder / 'model.lp'))
    model.m.dispose()
    return result


def _row(experiment, value, data, result, baseline):
    h = result.hourly
    return {
        'experiment': experiment, 'parameter_value': value,
        'minimum_daily_energy': data.min_daily_energy_kWh,
        'cq': data.quadratic_disutility,
        'daily_energy': float(h['load'].sum()),
        'unconstrained_daily_energy': baseline.meta['daily_energy'],
        'reference_daily_energy': float(data.reference_load.sum()),
        'procurement_cost': result.meta['procurement_cost'],
        'disutility': result.meta['disutility'],
        'total_cost': result.meta['total_cost'],
        'squared_deviation': float((h['deviation'] ** 2).sum()),
        'absolute_deviation': float(h['deviation'].abs().sum()),
        'above_reference_hours': int((h['deviation'] > TOL).sum()),
        'above_reference_energy_kWh': float(h['deviation'].clip(lower=0).sum()),
        'mu': result.meta['minimum_energy_multiplier_mu'],
        'daily_slack_kWh': result.meta['minimum_energy_slack_kWh'],
        'daily_constraint_binding': abs(result.meta['minimum_energy_slack_kWh']) <= TOL,
        'incremental_total_cost_DKK': result.meta['total_cost'] - baseline.meta['total_cost'],
        'load_difference_from_unconstrained_max_kWh': float(
            np.max(np.abs(h['load'].to_numpy() - baseline.hourly['load'].to_numpy()))),
    }


def _figure(fig, out, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def _plots(data, table, profiles, out):
    for experiment, xlabel in [('emin', 'Minimum daily energy (kWh)'),
                               ('cq', 'Quadratic coefficient (DKK/kWh²)')]:
        rows = table[table.experiment == experiment]
        x = rows.parameter_value
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].plot(x, rows.daily_energy, 'o-', label='With daily requirement')
        axes[0, 0].plot(x, rows.unconstrained_daily_energy, 's--', label='Without requirement')
        axes[0, 0].axhline(data.reference_load.sum(), color='black', ls=':', label='Reference total')
        if experiment == 'cq':
            axes[0, 0].axhline(rows.minimum_daily_energy.iloc[0], color='grey', ls='-.', label='Daily minimum')
        axes[0, 0].set_ylabel('Daily energy (kWh)')
        axes[0, 0].legend(fontsize=8)
        for key, label in [('procurement_cost', 'Procurement'), ('disutility', 'Disutility'),
                           ('total_cost', 'Total cost')]:
            axes[0, 1].plot(x, rows[key], 'o-', label=label)
        axes[0, 1].set_ylabel('Cost (DKK)'); axes[0, 1].legend(fontsize=8)
        axes[1, 0].plot(x, rows.mu, 'o-')
        axes[1, 0].set_ylabel('Daily multiplier mu (DKK/kWh)')
        axes[1, 1].plot(x, rows.squared_deviation, 'o-')
        axes[1, 1].set_ylabel('Unweighted squared deviation (kWh²)')
        for ax in axes.flat:
            ax.set_xlabel(xlabel); ax.grid(alpha=.25)
            if experiment == 'cq': ax.set_xscale('log')
        fig.suptitle('Q3(f): ' + (f'vary E_min; c_Q = {rows.cq.iloc[0]:g}' if experiment == 'emin'
                                  else f'vary c_Q; E_min = {rows.minimum_daily_energy.iloc[0]:g} kWh'))
        _figure(fig, out, f'q3f_{experiment}_summary')

        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)
        for ax, (_, row) in zip(axes.flat, rows.iterrows()):
            frame = profiles[(experiment, row.parameter_value)]
            ax.step(data.hours, data.reference_load, where='mid', color='black', ls=':', label='Reference')
            ax.step(data.hours, frame['unconstrained_load'], where='mid', ls='--', label='Without requirement')
            ax.step(data.hours, frame['load'], where='mid', label='With requirement')
            ax.set_title(f'{experiment} = {row.parameter_value:g}; mu = {row.mu:.3f}')
            ax.set(xlabel='Hour', ylabel='Load (kWh / 1 h)')
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(rows):]: ax.set_visible(False)
        fig.suptitle('Q3(f): hourly load profiles')
        _figure(fig, out, f'q3f_{experiment}_load_profiles')


def run_experiments(output_dir, emin_values=DEFAULT_EMIN, cq_values=DEFAULT_CQ,
                    fixed_cq=1., fixed_emin=30., plots=True):
    data = load_question('Q3')
    emin_values = _values(emin_values, 'E_min', 0, data.n_hours * data.load_max_kWh)
    cq_values = _values(cq_values, 'c_Q', 0, strictly_positive=True)
    if len(emin_values) > 6 or len(cq_values) > 6:
        raise ValueError('Use at most six values per experiment to keep the plots readable.')
    if not np.isfinite(fixed_cq) or fixed_cq <= 0:
        raise ValueError('Fixed c_Q must be finite and positive.')
    if not np.isfinite(fixed_emin) or not 0 <= fixed_emin <= data.n_hours * data.load_max_kWh:
        raise ValueError('Fixed E_min must be finite and feasible.')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    checks, rows, profiles, baselines = {}, [], {}, {}

    for experiment, values in [('emin', emin_values), ('cq', cq_values)]:
        for value in values:
            q = fixed_cq if experiment == 'emin' else value
            emin = value if experiment == 'emin' else fixed_emin
            d = replace(data, quadratic_disutility=q, min_daily_energy_kWh=emin)
            if q not in baselines:
                unconstrained = replace(d, min_daily_energy_kWh=None)
                baselines[q] = _solve(unconstrained, out, f'unconstrained_cq_{q:g}', checks)
            baseline = baselines[q]
            result = _solve(d, out, f'{experiment}_{value:g}', checks)
            row = _row(experiment, value, d, result, baseline)
            rows.append(row)
            frame = result.hourly.copy()
            frame['unconstrained_load'] = baseline.hourly['load'].to_numpy()
            profiles[(experiment, value)] = frame
            check_name = f'{experiment}_{value:g}_daily_energy_rule'
            error = abs(row['daily_energy'] - max(emin, baseline.meta['daily_energy']))
            checks[check_name] = {'passed': bool(error <= TOL), 'error_kWh': error}

    table = pd.DataFrame(rows)
    e = table[table.experiment == 'emin']
    checks['emin_cost_monotonicity'] = {'passed': bool(np.all(np.diff(e.total_cost) >= -TOL))}
    (out / 'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if not all(item['passed'] for item in checks.values()):
        raise RuntimeError('An experiment consistency check failed; inspect validation.json.')
    table.to_csv(out / 'sensitivity_metrics.csv', index=False)
    hourly = [frame.assign(experiment=exp, parameter_value=value)
              for (exp, value), frame in profiles.items()]
    pd.concat(hourly).to_csv(out / 'sensitivity_hourly.csv', index=False)
    snapshot = {'emin_values_kWh': emin_values, 'cq_values_DKK_per_kWh2': cq_values,
                'fixed_cq_for_emin': fixed_cq, 'fixed_emin_for_cq_kWh': fixed_emin,
                'reference_load': data.reference_load.tolist(), 'prices': data.energy_price.tolist(),
                'pv_available': data.pv_available.tolist(), 'load_min_kWh': data.load_min_kWh,
                'load_max_kWh': data.load_max_kWh, 'pv_marginal_cost': data.pv_marginal_cost,
                'import_tariff': data.import_tariff, 'export_tariff': data.export_tariff,
                'above_reference_tolerance_kWh': TOL,
                'note': 'No battery. Vary one parameter at a time. mu = -minimum_energy.Pi.'}
    (out / 'input_snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    if plots: _plots(data, table, profiles, out)
    print(table[['experiment', 'parameter_value', 'daily_energy', 'total_cost', 'mu',
                 'above_reference_hours']].to_string(index=False))
    print(f'All {len(checks)} checks passed. Outputs: {out.resolve()}')
    return table, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emin-values', type=float, nargs='+', default=DEFAULT_EMIN)
    parser.add_argument('--cq-values', type=float, nargs='+', default=DEFAULT_CQ)
    parser.add_argument('--fixed-cq', type=float, default=1.)
    parser.add_argument('--fixed-emin', type=float, default=30.)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/Q3f')
    args = parser.parse_args(argv)
    try:
        run_experiments(args.output_dir, args.emin_values, args.cq_values,
                        args.fixed_cq, args.fixed_emin)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
