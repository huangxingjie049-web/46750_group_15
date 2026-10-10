import subprocess
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('task',['1e','1f','2c','3d','3e','3g','3g-pi'])
def test_task_help(task):
    r=subprocess.run([sys.executable,str(ROOT/'main.py'),'--task',task,'--help'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert 'usage:' in r.stdout.lower()
    if task=='3g': assert '--pi' in r.stdout
    if task=='3g-pi': assert '--next-day-end-soc' in r.stdout
def test_list_tasks_without_solving():
    r=subprocess.run([sys.executable,str(ROOT/'main.py'),'--list-tasks'],capture_output=True,text=True)
    assert r.returncode==0
    assert '3g-pi' in r.stdout
def test_conflicting_case_is_rejected():
    r=subprocess.run([sys.executable,str(ROOT/'main.py'),'--task','3g','--question','Q2_quadratic'],capture_output=True,text=True)
    assert r.returncode!=0
    assert 'conflict' in r.stderr.lower()

def test_original_input_data_hashes_unchanged():
    import json,hashlib
    hashes=json.loads((ROOT/'docs/data_preservation.json').read_text())
    for rel,expected in hashes.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
