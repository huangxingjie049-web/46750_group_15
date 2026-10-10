"""Q3(d): solve and validate the no-battery daily-energy model."""
import argparse,json
from dataclasses import replace
from pathlib import Path
from src.data_loader import load_question
from .model import DailyEnergyConsumerModel
from .validation import validate_solution
ROOT=Path(__file__).resolve().parents[2]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emin',type=float,default=None);p.add_argument('--cq',type=float,default=None)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'Q3d')
    a=p.parse_args(argv);d=load_question('Q3')
    if a.emin is not None:d=replace(d,min_daily_energy_kWh=a.emin)
    if a.cq is not None:d=replace(d,quadratic_disutility=a.cq)
    m=DailyEnergyConsumerModel(d).build();r=m.solve();v=validate_solution(r,d)
    if not v['passed']:raise RuntimeError(v)
    a.output_dir.mkdir(parents=True,exist_ok=True);r.save(a.output_dir)
    m.m.write(str(a.output_dir/'model.lp'))
    (a.output_dir/'validation.json').write_text(json.dumps(v,indent=2))
    print(f'Daily load: {r.meta["daily_energy"]:.6f} kWh; total cost: {r.meta["total_cost"]:.6f} DKK')
    print(f'Validation passed. Outputs: {a.output_dir.resolve()}')
