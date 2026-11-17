"""Replay saved boundary rules."""
from common import *
from boundary import Evidence,apply
import argparse
import itertools

def grid(**options):return [dict(zip(options,x)) for x in itertools.product(*options.values())]

def candidates(stage):
    if stage=='contact_snap':return grid(cap=[25,50,100],distance=[.2,.4],minimum=[3],view=['both','tactical','broadcast'])
    if stage=='contact_decode':return grid(cap=[25,50,100],distance=[.2,.4],penalty=[.02,.1])
    if stage=='line_snap':return grid(cap=[25,50,100],view=['both','tactical','broadcast'])
    if stage in ('launch_snap','restart_contact'):return grid(cap=[25,50,100])
    if stage=='signal_decode':return grid(cap=[25,50,100],source=['carrier','team'],penalty=[.2,1.])
    if stage=='action_snap':return grid(cap=[25,50,100],threshold=[.7,.9])
    if stage=='handover_snap':return grid(cap=[25,50,100],view=['both','tactical','broadcast'],minimum=[3,5])
    if stage=='supported_snap':return grid(cap=[50,100],view=['both','tactical'],support=[3,6])
    if stage in ('flight_snap','receive_snap'):return grid(cap=[75,125],distance=[.3,.5])
    if stage=='shift':return grid(cap=[25],kind=['team','entry','restart'],lag=[-14,-7,7,14])
    raise ValueError(stage)

def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--replay', type=Path)
    ap.add_argument('--tag', default='initial')
    o = ap.parse_args()
    if o.prepare:
        prepare()
        return
    z=np.load(LOCAL/'upstream.npz');ev=Evidence()
    recipe=json.loads(o.replay.read_text())
    cv=np.full(N,2,'i1');outputs={};folds=[]
    for k,(a,b) in enumerate(BLOCKS):
        p=z[f'fold{k}'].copy()
        rows=recipe['folds'][k]['rules']
        for row in rows:p=apply(ev,p,row['stage'],row['config'])
        row=dict(block=k+1,rules=rows)
        folds.append(row);cv[a:b]=p[a:b];outputs[f'fold{k}']=p
    outputs['cv']=cv
    np.savez_compressed(LOCAL/f'{o.tag}.npz',**outputs)
    write(LOCAL/f'best_{o.tag}.csv',outputs['fold3']);write(LOCAL/f'cv_{o.tag}.csv',cv)
    save(LOCAL/f'{o.tag}.json', dict(tag=o.tag, folds=folds))

if __name__=='__main__':main()
