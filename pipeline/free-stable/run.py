"""Replay saved rule configurations and provide rule dispatch."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
LOCAL = HERE / '.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0, str(EXP / 'free-timing'))
import common as timing
import boundary
import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


timing_run = module('stable_timing_run', EXP / 'free-timing/run.py')
stop = timing.up
merge = module('stable_merge', EXP / 'free-merge/run.py')
sys.modules['common'] = module('owner_common', EXP / 'free-owner/common.py')
stack = module('stable_stack', EXP / 'free-stack/run.py')
sys.modules['common'] = timing
N, BLOCKS = timing.N, timing.BLOCKS
write, save = timing.write, timing.save
STACK = ('spot_islands', 'event_stops', 'trim_dead', 'snap_events')
STOP = ('keeper_veto', 'goal_clean', 'short_throw', 'brief_control', 'box_restart', 'whistle_snap')
END = ('kickoff', 'foul_switch', 'receive_snap', 'line_snap')


class Engine:
    def __init__(self):
        self.f = merge.Inputs()
        self.ev = boundary.Evidence.__new__(boundary.Evidence)
        self.ev.f, self.ev.cache = self.f, {}
        self.rule_ev = {}

    def apply(self, k, p, stage, cfg):
        if not cfg:
            return p.copy()
        if stage == 'decoder':
            return stack.Decoder(k).run(cfg)
        if stage in STACK:
            if k not in self.rule_ev:
                self.rule_ev[k] = stack.rule_inputs(k)
            return stack.apply(self.rule_ev[k], p, stage, cfg)
        if stage in STOP:
            return stop.apply(self.f, p, stage, cfg)
        if stage in ('kickoff', 'foul_switch', 'views'):
            return merge.apply(self.f, p, stage, cfg)
        return boundary.apply(self.ev, p, stage, cfg)


def candidates(stage):
    if stage in STACK:
        return stack.candidates(stage)
    if stage in STOP:
        return stop.candidates(stage)
    if stage in ('kickoff', 'foul_switch', 'views'):
        return merge.candidates(stage)
    return timing_run.candidates(stage)


def prepare():
    eng = Engine()
    sr = json.loads((EXP / 'free-stack/recipe.json').read_text())
    tr = json.loads((EXP / 'free-stoppages/recipe.json').read_text())
    mr = json.loads((EXP / 'free-merge/recipe.json').read_text())
    outputs = {}
    for k in range(4):
        p = stack.Decoder(k).run(sr['folds'][k]['decoder'])
        outputs[f'decoder{k}'] = p.copy()
        for row in sr['folds'][k]['rules']:
            p = eng.apply(k, p, row['stage'], row['config'])
        outputs[f'stack{k}'] = p.copy()
        for row in tr['folds'][k]['rules']:
            p = eng.apply(k, p, row['stage'], row['config'])
        outputs[f'stop{k}'] = p.copy()
        for row in mr['folds'][k]['rules']:
            p = eng.apply(k, p, row['stage'], row['config'])
        outputs[f'merge{k}'] = p.copy()
    np.savez_compressed(LOCAL / 'upstream.npz', **outputs)


def stitch(arrays):
    cv = np.full(N, 2, 'i1')
    for k, (a, b) in enumerate(BLOCKS):
        cv[a:b] = arrays[k][a:b]
    return cv


def vote(arrays, weights, base):
    counts = np.zeros((N, 3), dtype='f4')
    for p, w in zip(arrays, weights):
        for c in range(3):
            counts[:, c] += w * (p == c)
    p = counts.argmax(1).astype('i1')
    tied = counts[np.arange(N), base] >= counts.max(1) - 1e-6
    p[tied] = base[tied]
    return p


def replay(eng, k, base, rows):
    p = base.copy()
    for row in rows:
        if 'recipes' in row:
            rs = row['recipes']
            p = vote([replay(eng, k, p, r['rules']) for r in rs], [r['weight'] for r in rs], p)
        elif 'ensemble' in row:
            es = row['ensemble']
            p = vote([eng.apply(k, p, row['stage'], e['config']) for e in es], [e['weight'] for e in es], p)
        else:
            p = eng.apply(k, p, row['stage'], row['config'])
    return p


def export(tag, base, policy, folds, arrays):
    cv = stitch(arrays)
    write(LOCAL/f'best_{tag}.csv', arrays[3])
    write(LOCAL/f'cv_{tag}.csv', cv)
    np.savez_compressed(LOCAL/f'{tag}.npz', **{f'fold{k}':p for k,p in enumerate(arrays)}, cv=cv)
    r = dict(tag=tag, base=base, policy=policy, folds=folds)
    save(LOCAL/f'{tag}.json', r)
    return r


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--replay', type=Path)
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return
    recipe = json.loads(args.replay.read_text())
    eng = Engine()
    z = np.load(LOCAL/'upstream.npz')
    arrays = [replay(eng, k, z[f'{recipe["base"]}{k}'], recipe['folds'][k]['rules']) for k in range(4)]
    export('selected', recipe['base'], recipe['policy'], recipe['folds'], arrays)


if __name__ == '__main__':
    main()
