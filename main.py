"""Unified assignment entry point.

Examples:
  python main.py --list-tasks
  python main.py --task 1e --case Q1_caseA
  python main.py --task 1f
  python main.py --task 2c --sweep
  python main.py --task 3d
  python main.py --task 3e
  python main.py --task 3g --mode both --pi 2 --sweep
  python main.py --task 3g-pi --run-today
  python main.py --task 3g --help

--question Q1_caseA/Q1_caseB/Q2_quadratic/Q3/Q3_battery remains supported.
Without task/question, run Q1 Case A. Task-specific options follow --task.
"""
import argparse
from importlib import import_module
import sys

TASKS={
 '1e':('src.1e.runner','Q1 model implementation; choose Case A or B'),
 '1f':('src.1f.runner','Q1 comparison: solve both Case A and Case B'),
 '2c':('src.2c.runner','Quadratic-disutility model and optional coefficient sweep'),
 '3d':('src.3d.runner','Daily-minimum model implementation and validation'),
 '3e':('src.3e.runner','Same-data Q2(c)/Q3 comparison, validation and figures'),
 '3g':('src.3g.runner','Battery: cyclic/value modes, comparisons and optional sweeps'),
 '3g-pi':('src.3g.estimate_pi','Estimate terminal value from next-day forecasts'),
}
ALIASES={'Q1_caseA':'1e','Q1_caseB':'1e','Q2_quadratic':'2c','Q3':'3e','Q3_battery':'3g'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,add_help=False,allow_abbrev=False)
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--question',help='Backward-compatible data-case selector')
    p.add_argument('--case',choices=['Q1_caseA','Q1_caseB'])
    p.add_argument('--list-tasks',action='store_true')
    p.add_argument('-h','--help',action='store_true')
    a,rest=p.parse_known_args(argv)
    if a.list_tasks:
        for key,(_,description) in TASKS.items():print(f'{key:7} {description}')
        return
    if a.help and a.task is None and a.question is None:
        p.print_help();return
    if a.question is not None and a.question not in ALIASES:
        p.error(f'{a.question} has input data but no implemented task in this uploaded working tree. Use --list-tasks.')
    task=a.task or ALIASES.get(a.question,'1e')
    if a.task is not None and a.question is not None and ALIASES[a.question]!=a.task:
        p.error('--task and --question conflict; choose one selector.')
    if a.case is not None and task!='1e':p.error('--case is only for task 1e; task 1f runs both cases.')
    if task=='1e':
        case=a.case or a.question or 'Q1_caseA'
        if a.case and a.question and a.case!=a.question:p.error('--case and --question conflict.')
        rest=['--case',case]+rest
    if task=='2c' and not a.help:
        # Keep original main.py behaviour: base only unless --sweep requested.
        if '--sweep' not in rest and '--base-only' not in rest:rest.append('--base-only')
        if '--sweep' in rest:
            i=rest.index('--sweep')
            if i+1==len(rest) or rest[i+1].startswith('--'):
                values=import_module('src.2c.runner').DEFAULT_SWEEP
                rest[i+1:i+1]=[str(v) for v in values]
    if a.help:rest.append('--help')
    import_module(TASKS[task][0]).main(rest)

if __name__=='__main__':main()
