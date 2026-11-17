"""Replay the decoder and its sequence of rule stages."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import logit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE.parent
LOCAL = HERE/'.local'
LOCAL.mkdir(exist_ok=True)
sys.path.append(str(EXP/'free-grammar'))
import grammar
sys.path.append(str(EXP/'free-rules'))
import evidence
from rules import apply as upstream_apply
sys.path.insert(0,str(EXP/'free-owner'))
from decode import decode, posterior
sys.path.insert(0,str(ROOT/'official/evaluation'))

N = 199319
BLOCKS = evidence.BLOCKS
STAGES = ('spot_islands','event_stops','trim_dead')


def apply(ev,base,stage,cfg):
    if stage not in ('terminal_touch','event_flight'):
        return upstream_apply(ev,base,stage,cfg)
    if not cfg:return base.copy()
    te=ev.touches;tt=te['t'];co=te['contact'];lo=te['local']
    good=(abs(co)>=.6)&(np.sign(co)*lo>=cfg['confidence'])
    p=base.copy()
    for fa,fb in BLOCKS:
        rr=evidence.runs(base[fa:fb])
        for i,(aa,bb,tm) in enumerate(rr):
            a,b=fa+aa,fa+bb
            if tm>=2 or b-a<25:continue
            if stage=='terminal_touch':
                if i==len(rr)-1 or rr[i+1][2]!=2:continue
                ix=np.flatnonzero(good&(tt>=max(a,b-cfg['maximum']))&(tt<b))
                if not len(ix):continue
                j=ix[-1];owner=int(co[j]<0)
                if owner==tm:continue
                s=int(tt[j])
                # A recent opposing spotter call vetoes a geometric contact.
                if np.any(te['spotter'][ix]*np.sign(co[j]) < -1.5):continue
                q=ev.local_team if owner==0 else 1-ev.local_team
                while s>max(a,b-cfg['maximum']) and q[s-1]>.5:s-=1
                if b-s>=10:p[s:b]=owner
            else:
                ix=np.flatnonzero(good&(tt>=a)&(tt<b))
                for j in ix:
                    owner=int(co[j]<0)
                    if owner==tm:continue
                    s=int(tt[j]);end=min(b,s+cfg['maximum'])
                    # Flight ends at the next independently attributed opposing touch.
                    nxt=ix[(ix>j)&(np.sign(co[ix])!=np.sign(co[j]))]
                    if len(nxt):end=min(end,int(tt[nxt[0]]))
                    q=ev.local_team if owner==0 else 1-ev.local_team
                    while s>max(a,int(tt[j])-25) and q[s-1]>.5:s-=1
                    if end-s>=15:p[s:end]=owner
    return p


def rule_inputs(k):
    ev=evidence.Inputs(k)
    ev.touches=dict(np.load(LOCAL/'touches.npz'))
    ev.local_team=np.load(EXP/'free-team/.local/q_local.npy')
    return ev


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def prepare():
    """Snapshot only permitted label-free caches and scalar recipes."""
    paths = [EXP/'free-grammar/.local'/f'fold{k+1}.npz' for k in range(4)]
    paths += [EXP/'free-grammar/.local'/n for n in ('touches.npz','em_prior.json')]
    paths += [EXP/'free-rules/.local/inputs.npz',EXP/'free-brief/.local/detect.npz']
    for path in paths:
        shutil.copyfile(path,LOCAL/path.name)


grammar.LOCAL = LOCAL
evidence.LOCAL = LOCAL


def write(path,p):
    Path(path).write_text('frame,label\n'+''.join(f'{i},{"HAD"[int(c)]}\n' for i,c in enumerate(p)),newline='\n')


def grid(base,**opts):
    return [dict(base,**dict(zip(opts,v))) for v in itertools.product(*opts.values())]


def candidates(stage):
    if stage=='spot_islands':
        return grid(dict(spot=.3,minimum=15,maximum=200,veto=.9),dead=[.3,.5,.7],grow=[.3,.5,.7],confidence=[.5,.7])
    if stage=='event_stops':
        return grid(dict(maximum=500),score=[.3,.5,.7],dead=[0.,.05,.15,.3])
    if stage=='trim_dead':
        return grid({},threshold=[.03,.1,.3],maximum=[50,100,200],entry=[False,True])
    # Original ports, reused code and grids only; no label-trained inputs.
    if stage=='contacts':return grid(dict(margin=1.),minimum=[3,5,10],team=[.4,.5,.6],near=[.8,1.5])
    if stage=='contact_chains':return grid(dict(hold=20),minimum=[3,5,10],team=[.5,.6,.7,.8])
    if stage=='brief_owner':return grid(dict(model='state',minimum=5,grow=.5,dead_fraction=0),seed=[.7,.85,.95],maximum=[50,100,200])
    if stage=='remove_flicker':return grid({},maximum=[25,50,100,200],confidence=[.4,.5,.6,.7])
    if stage=='keeper_holds':return grid(dict(maximum=750),fraction=[.1,.2,.4],dead=[.1,.3,.5])
    if stage=='snap_events':return grid({},back=[50,100,200],offset=[-10,0,10],entry=[False,True])
    if stage in ('terminal_touch','event_flight'):return grid({},confidence=[1.,2.,3.],maximum=[50,100,200])
    raise ValueError(stage)


class Decoder:
    def __init__(self,k):
        self.k=k; self.gg=grammar.Grammar(k); self.z=self.gg.z
        self.memo={}
        self.det=dict(np.load(LOCAL/'detect.npz'))
        iv=self.det['out_iv']>0
        rs=np.zeros(N);rs[np.flatnonzero(iv[:-1]&~iv[1:])+1]=1
        self.rs=maximum_filter1d(rs,11)
        self.on=maximum_filter1d(self.det['out_on'],11)

    def grammar(self,source,pars=None):
        assert pars is not None
        key=source+json.dumps(pars,sort_keys=True)
        if key not in self.memo:
            # Bound memory even during scalar searches.
            if len(self.memo)>3:self.memo.clear()
            self.memo[key]=self.gg.run(pars,source=='joint',True)
        return self.memo[key]

    def run(self,cfg,marginals=False):
        kind=cfg['kind']
        if kind in ('team','joint','team_sparse'):
            p,m=self.grammar(kind)
            return (p.copy(),m.copy()) if marginals else p.copy()
        s=self.z['s'].copy();d=self.z['dead'].copy()
        if kind=='fusion':
            wt=cfg['wt'];wd=cfg['wd']
            _,m=self.grammar('team',cfg.get('grammar_params'))
            ph=np.divide(m[:,0],m[:,:2].sum(1),out=np.full(N,.5),where=m[:,:2].sum(1)>1e-30)
            s=(1-wt)*s+wt*logit(np.clip(ph,.005,.995))
            d=(1-wd)*d+wd*logit(np.clip(m[:,2],.005,.995))
        cin=self.z['cin'];cout=self.z['cout']
        k=cfg.get('brief',0.)
        if k:
            d=d+k*self.det['out_iv']
            cin=np.maximum(.5,cin-40*k*self.on)
            cout=np.maximum(.5,cout-40*k*self.rs)
        if cfg.get('local',0.):
            w=cfg['local'];q=np.load(EXP/'free-team/.local/q_local.npy')
            s=(1-w)*s+w*logit(np.clip(q,.005,.995))
        pot=(s,d+.5*np.abs(s),cfg.get('beta',20.),cin,cout)
        p=decode(*pot);p[p==3]=2
        if not marginals:return p
        m=np.nan_to_num(posterior(*pot))
        m[:11185]=[0,0,1];m[90652:110808]=[0,0,1];m[190198:]=[0,0,1]
        return p,m


def base_candidates(name,k,up):
    if name in ('pretrained','fusion','team','joint','team_sparse'):
        return [dict(kind=name)]
    if name=='brief':return [dict(kind='pretrained',brief=v) for v in (0.,.25,.5,1.,1.5,2.)]
    if name=='fusion_brief':return [dict(kind='fusion',brief=v) for v in (0.,.25,.5,1.,1.5,2.)]
    if name=='blend':return [dict(kind='fusion',wt=v,wd=0.) for v in (0.,.1,.25,.5,.75,1.)]
    if name=='blend_brief':return [dict(kind='fusion',wt=w,wd=0.,brief=v) for w in (0.,.1,.25,.5) for v in (0.,.5,1.,2.)]
    if name=='family':return [dict(kind=v) for v in ('pretrained','fusion','team','joint','team_sparse')]
    if name=='local':return [dict(kind='fusion',local=w) for w in (0.,.1,.25,.5,.75,1.)]
    if name=='beta':return [dict(kind='fusion',beta=w) for w in (20.,5.,10.,40.,80.)]
    raise ValueError(name)


def export(tag,folds):
    cv=np.full(N,2,'i1'); arrays={}; cm=np.zeros((N,3));cm[:,2]=1
    for k,(a,b) in enumerate(BLOCKS):
        dec=Decoder(k);ev=rule_inputs(k)
        base,m=dec.run(folds[k]['decoder'],True);p=base.copy()
        for row in folds[k]['rules']:p=apply(ev,p,row['stage'],row['config'])
        cv[a:b]=p[a:b];cm[a:b]=m[a:b];arrays[f'fold{k}']=p
        arrays[f'base{k}']=base
        if k==3:best,dm=p,m
    write(LOCAL/f'cv_{tag}.csv',cv);write(LOCAL/f'best_{tag}.csv',best)
    arrays['cv']=cv
    np.savez_compressed(LOCAL/f'{tag}.npz',**arrays)
    def ph(m):
        return np.divide(m[:,0],m[:,:2].sum(1),out=np.full(N,.5),where=m[:,:2].sum(1)>1e-30).astype('f4')
    # These are decoder marginals BEFORE discrete rules, not calibrated rule probabilities.
    np.savez_compressed(LOCAL/f'evidence_{tag}.npz',label=best,label_oob=cv,
                        p_dead=dm[:,2].astype('f4'),p_home_live=ph(dm),
                        p_dead_oob=cm[:,2].astype('f4'),p_home_live_oob=ph(cm))
    res=dict(tag=tag,folds=folds)
    save_json(LOCAL/f'{tag}.json',res)
    return res


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--replay', type=Path)
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return
    recipe = load_json(args.replay)
    export(recipe['tag'], recipe['folds'])


if __name__=='__main__':main()
