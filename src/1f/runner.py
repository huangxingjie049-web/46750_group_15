"""Q1(f): reproduce the Case A / Case B numerical comparison."""
import argparse
from pathlib import Path
from importlib import import_module
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'Q1f')
    a=p.parse_args(argv);a.output_dir.mkdir(parents=True,exist_ok=True)
    runner=import_module('src.1e.runner');rows=[]
    for case in ['Q1_caseA','Q1_caseB']:
        out=a.output_dir/case;out.mkdir(exist_ok=True)
        r=runner.run_base_case(case,out,False)
        rows.append({'case':case,'objective_DKK':r.objective,'load_kWh':float(r.hourly['load'].sum()),'pv_kWh':float(r.hourly['pv'].sum()),'import_kWh':float(r.hourly['import'].sum()),'export_kWh':float(r.hourly['export'].sum())})
    pd.DataFrame(rows).to_csv(a.output_dir/'comparison_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
