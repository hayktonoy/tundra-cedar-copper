"""Shared paths and upstream replay for boundary rules."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='2'
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCAL=HERE/'.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'free-stoppages'))
spec=importlib.util.spec_from_file_location('stoppages',HERE.parent/'free-stoppages/run.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
sys.path.insert(0,str(HERE))
N=up.N
BLOCKS=up.BLOCKS
HALVES=up.HALVES
runs=up.runs
write=up.write
save=up.save

def prepare():
    f=up.Features()
    z=np.load(HERE.parent/'free-stoppages/.local/upstream.npz')
    recipe=json.loads((HERE.parent/'free-stoppages/recipe.json').read_text())
    outputs={}
    for k,row in enumerate(recipe['folds']):
        p=z[f'fold{k}'].copy()
        for r in row['rules']:p=up.apply(f,p,r['stage'],r['config'])
        outputs[f'fold{k}']=p
    np.savez_compressed(LOCAL/'upstream.npz',**outputs)
