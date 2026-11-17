"""Replay saved rule configurations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parent
LOCAL=HERE/'.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0,str(EXP/'free-stoppages'))
from features import Features, N, HALVES, runs, groups  # noqa: E402
sys.path.insert(0,str(ROOT/'official/evaluation'))
BLOCKS=((11185,50919),(50919,90652),(110808,150503),(150503,190198))
CENTRE=(52.65,34.)
SHOTS=['SHOT','Shots on target','Shots off target']


def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def write(p,q):Path(p).write_text('frame,label\n'+''.join(f'{i},{"HAD"[int(v)]}\n' for i,v in enumerate(q)),newline='\n')
def prepare():
    """Snapshot free-stoppages' four fold outputs and the two label-free team views."""
    z=np.load(EXP/'free-stoppages/.local/selected.npz')
    outputs={f'fold{k}':z[f'fold{k}'] for k in range(4)}
    np.savez_compressed(LOCAL/'upstream.npz',**outputs)
    # Only the label-free arrays are copied; the label keys in these files are never accessed.
    car=np.load(EXP/'free-carrier/.local/evidence.npz')
    np.savez_compressed(LOCAL/'team_views.npz',local=np.load(EXP/'free-team/.local/q_local.npy').astype('f4'),
                        carrier=car['p_home_live'].astype('f4'))


class Inputs(Features):
    @property
    def local_team(self):return np.load(LOCAL/'team_views.npz')['local']

    @property
    def carrier(self):return np.load(LOCAL/'team_views.npz')['carrier']

    def peaks(self,names,threshold,gap=25):
        ix=np.flatnonzero(np.isin(self.names,names)&(self.ev['score']>=threshold))
        sel=[]
        for j in ix[np.argsort(-self.ev['score'][ix])]:
            if all(abs(self.times[j]-self.times[k])>gap for k in sel):sel.append(int(j))
        return sorted(int(self.times[j]) for j in sel)


def kickoff(f):
    """Frames before each half's kick-off: the ball has not been put into play yet.

    A kick-off is an observed ball departure (>5 m/s, sustained) after it has rested in the
    centre circle while the broadcast shows play. Returns (half start, kick) pairs.
    """
    out=[]
    for lo,hi in HALVES:
        kick=None
        for tr in (f.t,f.b):
            x,y,sp,src=tr['x'],tr['y'],tr['speed'],tr['source']
            rest=(src==1)&(sp<1)&(np.hypot(x-CENTRE[0],y-CENTRE[1])<9.15)
            motion=(src==1)&(sp>5)&(sp<40)
            motion&=np.convolve(motion.astype(int),np.ones(7,dtype=int),'same')>=3
            for g in groups(rest[lo:lo+7500],10,25):
                b=lo+int(g[-1])+1
                hits=np.flatnonzero(motion[b:b+250])
                if not len(hits):continue
                k=b+int(hits[0])
                if f.z['broad'][k:k+50].mean()>=.5:continue
                kick=k if kick is None else min(kick,k)
                break
        if kick is not None:out.append((lo,kick))
    return out


def views(f,width=9):
    return {k:uniform_filter1d(np.nan_to_num(v,nan=.5).astype(float),width) for k,v in
            (('team',f.z['team']),('local',f.local_team),('carrier',f.carrier))}


def apply(f,base,stage,cfg):
    p=base.copy()
    if not cfg:return p
    if stage=='kickoff':
        for lo,k in kickoff(f):p[lo:k]=2
    elif stage=='foul_switch':
        # A foul stops play; a team change at or just after the foul that runs straight into
        # the stoppage is the fouler's touch or the restart taker, not a new possession.
        fouls=f.peaks(['Foul'],cfg['score'])
        for lo,hi in HALVES:
            rr=runs(base[lo:hi])
            for i in range(1,len(rr)-1):
                a,b,t=rr[i];a+=lo;b+=lo;old=rr[i-1][2]
                if t>1 or old>1 or old==t or rr[i+1][2]!=2 or b-a>cfg['maximum']:continue
                if not any(a-cfg['before']<=x<=a+10 for x in fouls):continue
                p[a:b]=old if cfg['mode']=='old' else 2
    elif stage=='views':
        # Two independent label-free team views agreeing on the other team inside one of our
        # possessions. Vetoes: the moments before our stoppage (restart taker retrieving the
        # ball) and the moments after a shot (keeper parry/deflection is not control).
        v=views(f);names=cfg['views']
        ix=np.flatnonzero(np.isin(f.names,SHOTS)&(f.ev['score']>=.3))
        shots=[(int(f.times[j]),int(f.ev['p_home'][j]<.5)) for j in ix]
        for lo,hi in HALVES:
            for a,b,t in runs(base[lo:hi]):
                a+=lo;b+=lo
                if t>1:continue
                opp=np.min([v[n][a:b] if t==1 else 1-v[n][a:b] for n in names],axis=0)
                isl=[(a+x,a+y) for x,y,s in runs((opp>=cfg['threshold']).astype(int)) if s]
                merged=[]
                for x,y in isl:
                    if merged and x-merged[-1][1]<=cfg['gap']:merged[-1]=(merged[-1][0],y)
                    else:merged.append((x,y))
                for x,y in merged:
                    if y-x<cfg['minimum'] or y-x>cfg['maximum']:continue
                    if cfg['extend']:
                        while x>a and opp[x-1-a]>=.5:x-=1
                        while y<b and opp[y-a]>=.5:y+=1
                    if (base[y:y+cfg['veto']]==2).any() or (base[max(lo,x-cfg['veto']):x]==2).any():continue
                    if any(x-50<=s<y and team==t for s,team in shots):continue
                    p[x:y]=1-t
    else:raise ValueError(stage)
    for a,b in ((0,11185),(90652,110808),(190198,N)):p[a:b]=2
    return p


def grid(**opts):return [dict(zip(opts,v)) for v in itertools.product(*opts.values())]
def candidates(stage):
    if stage=='kickoff':return [dict(on=True)]
    if stage=='foul_switch':return grid(score=[.5,.7],before=[50,100],maximum=[150],mode=['old','dead'])
    if stage=='views':return grid(views=[('team','local')],threshold=[.6,.7],minimum=[25,50],maximum=[400],
                                  gap=[25],extend=[False,True],veto=[50])
    raise ValueError(stage)




def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--replay', type=Path)
    o = ap.parse_args()
    if o.prepare:
        prepare()
        return
    f = Inputs()
    z = np.load(LOCAL/'upstream.npz')
    recipe = json.loads(o.replay.read_text())
    folds, tag = recipe['folds'], recipe['tag']
    outputs={};cv=np.full(N,2,'i1')
    for k,(a,b) in enumerate(BLOCKS):
        p=z[f'fold{k}'].copy()
        rows=folds[k]['rules']
        for row in rows:p=apply(f,p,row['stage'],row['config'])
        cv[a:b]=p[a:b];outputs[f'fold{k}']=p
    write(LOCAL/f'best_{tag}.csv',outputs['fold3']);write(LOCAL/f'cv_{tag}.csv',cv)
    np.savez_compressed(LOCAL/f'{tag}.npz',**outputs,cv=cv)
    save(LOCAL/f'{tag}.json', dict(tag=tag, folds=folds))


if __name__=='__main__':main()
