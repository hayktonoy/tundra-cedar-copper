"""Replay saved rule configurations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from features import Features, N, HALVES, runs, groups

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCAL=HERE/'.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'official/evaluation'))
BLOCKS=((11185,50919),(50919,90652),(110808,150503),(150503,190198))
RESTARTS=['THROW IN','FREE KICK','Throw-in','Corner','Indirect free-kick','Direct free-kick','Kick-off']
EXITS=['OUT','Ball out of play','Foul','Offside']


def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def write(p,q):Path(p).write_text('frame,label\n'+''.join(f'{i},{"HAD"[int(v)]}\n' for i,v in enumerate(q)),newline='\n')
def prepare():
    spec=importlib.util.spec_from_file_location('free_stack_run',HERE.parent/'free-stack/run.py')
    stack=importlib.util.module_from_spec(spec);spec.loader.exec_module(stack)
    recipe=json.loads((HERE.parent/'free-stack/recipe.json').read_text())
    outputs={}
    for k,row in enumerate(recipe['folds']):
        dec=stack.Decoder(k);ev=stack.rule_inputs(k);p=dec.run(row['decoder'])
        for rule in row['rules']:p=stack.apply(ev,p,rule['stage'],rule['config'])
        outputs[f'fold{k}']=p
    np.savez_compressed(LOCAL/'upstream.npz',**outputs)


def peaks(f,names,threshold=.5,bro=True):
    ix=f.events(names,threshold,bro=bro)
    ix=ix[np.argsort(f.times[ix])]
    # NMS across heads/classes in one physical event, strongest score wins.
    selected=[]
    for j in ix[np.argsort(-f.ev['score'][ix])]:
        if all(abs(f.times[j]-f.times[k])>25 for k in selected):selected.append(int(j))
    return sorted(selected,key=lambda k:f.times[k])


def apply(f,base,stage,cfg):
    p=base.copy()
    if not cfg:return p
    clean=stage in ('goal_clean','out_clean','out_long')
    if stage=='goal_clean':stage='tag_goal_line'
    if stage in ('out_clean','out_long'):stage='out_confirmed'
    if stage=='out_physical':
        for a,b,t in runs(f.det['out_iv']>0):
            if not t:continue
            if b-a<cfg['minimum']:continue
            p[a:b]=2
    elif stage=='out_confirmed':
        outs=peaks(f,['OUT','Ball out of play'],cfg['score'])
        for a,b,t in runs(f.det['out_iv']>0):
            if not t or b-a<cfg.get('minimum',25):continue
            if (base[max(0,a-25):b+50]==2).any():continue
            near=[j for j in outs if abs(f.times[j]-a)<=50]
            if not near:continue
            s=int(f.times[min(near,key=lambda j:abs(f.times[j]-a))])
            e=b+cfg['tail']
            if e>s+15:p[s:e]=2
    elif stage in ('event_stop','short_throw'):
        starts=peaks(f,['OUT','Ball out of play'] if stage=='short_throw' else EXITS,cfg['score'])
        ends=peaks(f,['THROW IN','Throw-in'] if stage=='short_throw' else RESTARTS,cfg['score'])
        for j in starts:
            a=int(f.times[j]);future=[k for k in ends if a+10<f.times[k]<a+cfg.get('maximum',500)]
            if not future:continue
            b=int(f.times[future[0]])
            if (base[a:b]==2).any():continue
            if f.z['local'][a:b].mean()<cfg.get('dead',0):continue
            p[a:b]=2
    elif stage in ('brief_restart','brief_views','brief_control'):
        starts=peaks(f,RESTARTS,cfg['score'],bro=stage=='brief_restart');ends=peaks(f,['OUT','Ball out of play'],cfg['out'])
        for j in starts:
            a=int(f.times[j]);future=[k for k in ends if a+15<=f.times[k]<=a+cfg['maximum']]
            if not future:continue
            b=int(f.times[future[0]])
            if not any(lo+25<a<b<hi-25 for lo,hi in HALVES):continue
            if (base[a-25:a]==2).mean()<.8 or (base[a:b]==2).mean()<.5:continue
            # The out event must lead into an existing dead phase, preventing
            # an isolated spotting error from cutting a long live phase.
            if (base[b:b+25]==2).mean()<.8:continue
            if cfg['physical'] and not (f.det['out_on'][max(a,b-30):b+30].any() or
                                      f.z['local'][a:b].mean()<.9):continue
            ph=float(f.ev['p_home'][j])
            if stage in ('brief_views','brief_control'):ph=float(f.z['team'][a:b].mean())
            if not np.isfinite(ph) or abs(ph-.5)<.2:ph=float(f.z['team'][a:b].mean())
            p[a:b]=int(ph<.5)
            if stage=='brief_control':
                tm=int(ph<.5);sgn=1 if tm else -1
                for s in range(a+10,b-10):
                    near=f.z['contact'][max(a,s-5):s+1]*sgn
                    own=f.z['team'][s:s+10].mean()
                    own=own if tm else 1-own
                    if (near>0).sum()>=3 and own>.7:
                        p[s:b]=1-tm;break
    elif stage=='rest_goal':
        # Placed, stationary ball near a goal followed by a sustained departure.
        # A preceding ball-out is mandatory; a keeper's catch is not a goal kick.
        exits=peaks(f,['OUT','Ball out of play'],cfg['score'])
        for tr in (f.t,f.b):
            x,y,sp,src=tr['x'],tr['y'],tr['speed'],tr['source']
            rest=(src==1)&(sp<1)&(np.minimum(abs(x),abs(x-105.3))<16.5)&(abs(y-34)<20)
            for g in groups(rest,10,cfg['rest']):
                a,b=int(g[0]),int(g[-1])+1
                if b-a<cfg['rest'] or b-a>1000:continue
                if not any(lo<a<b+30<hi for lo,hi in HALVES):continue
                prev=[j for j in exits if a-750<f.times[j]<a+25]
                if not prev:continue
                s=int(f.times[prev[-1]])
                if b-s<50 or b-s>1000:continue
                # Nearest launch after the final stationary detection. Do not
                # use an unrelated first rest/kick cue halfway through setup.
                hits=np.flatnonzero((src[b:b+60]>0)&(sp[b:b+60]>5))
                if not len(hits):continue
                e=b+int(hits[0])
                if cfg['off'] and np.nanmean(f.off[s:e])<.3:continue
                p[s:e]=2
    elif stage=='box_restart':
        starts=peaks(f,['OUT','Ball out of play'],cfg['score'])
        kicks=peaks(f,['HIGH PASS','FREE KICK','Indirect free-kick'],.7,bro=False)
        for j in kicks:
            b=int(f.times[j])
            if not any(lo+1000<b<hi for lo,hi in HALVES):continue
            rest=False
            for tr in (f.t,f.b):
                x,y,sp,src=(tr[k][b-50:b] for k in ('x','y','speed','source'))
                rest|=int(((src==1)&(sp<1.5)&(np.minimum(abs(x),abs(x-105.3))<16.5)&(abs(y-34)<20)).sum())>=cfg['rest']
            if not rest:continue
            prev=[k for k in starts if b-750<f.times[k]<b-50]
            if not prev:continue
            a=int(f.times[prev[-1]])
            # A spotted shot/pass during the proposed interval contradicts
            # a single continuous dead phase, unless it is the final restart.
            live=peaks(f,['SHOT','Shots on target','CROSS','THROW IN','Throw-in'],.7)
            if any(a+25<f.times[k]<b-50 for k in live):continue
            p[a:b]=2
    elif stage=='whistle_snap':
        from scipy.signal import find_peaks
        au=np.load(LOCAL/'audio.npz');v=au['tonality']*au['dominance']
        wh=find_peaks(v,height=cfg['threshold'],distance=50)[0]
        for lo,hi in HALVES:
            rr=runs(base[lo:hi])
            for i,(aa,bb,t) in enumerate(rr):
                a,b=lo+aa,lo+bb
                if t!=2 or i==0:continue
                near=wh[(wh>=max(lo,a-cfg['cap']))&(wh<min(b,a+cfg['cap']))]
                if not len(near):continue
                w=int(near[np.argmax(v[near])])
                if w<a:p[w:a]=2
                else:p[a:w]=rr[i-1][2]
    elif stage in ('tag_goal','tag_goal_observed','tag_goal_line'):
        import re
        xml=(ROOT/'official/data/tactical/BAR-ATH_tactical_smart_tagging.xml').read_text(encoding='utf-8')
        tags=[]
        for inst in re.findall(r'<instance>(.*?)</instance>',xml,re.S):
            if '<text>Goal Kick</text>' in inst:
                tags.append(round(float(re.search(r'<start>(.*?)</start>',inst).group(1))*25))
        exits=peaks(f,['OUT','Ball out of play'],cfg['score'])
        for tag in tags:
            for tr in (f.t,f.b):
                lo,hi=max(0,tag-300),min(N,tag+300)
                x,y,sp,src=(tr[k] for k in ('x','y','speed','source'))
                rest=(src==1)&(sp<1)&(np.minimum(abs(x),abs(x-105.3))<8)&(abs(y-34)<15)
                gg=groups(rest[lo:hi],15,cfg['rest'])
                if not gg:continue
                g=max(gg,key=len);a,b=lo+int(g[0]),lo+int(g[-1])+1
                prev=[j for j in exits if a-750<f.times[j]<a]
                if not prev:continue
                s=int(f.times[prev[-1]])
                if stage=='tag_goal_line':
                    # A one-view goal-line crossing becomes trustworthy when
                    # a tagged goal kick and stationary goal-area ball agree.
                    cross=(src==1)&((x<0)|(x>105.3))&(y>0)&(y<68)
                    ix=np.flatnonzero(cross[max(0,s-50):s+51])+max(0,s-50)
                    if len(ix):s=int(ix[np.argmin(abs(ix-s))])
                if cfg['off'] and np.nanmean(f.off[s:b])<.3:continue
                motion=(src==1 if stage!='tag_goal' else src>0)&(sp>5)&(sp<40)
                if stage!='tag_goal':
                    motion&=np.convolve(motion.astype(int),np.ones(7,dtype=int),'same')>=3
                hits=np.flatnonzero(motion[b:b+150])
                if not len(hits):continue
                e=b+int(hits[0]);p[s:e]=2
    elif stage in ('keeper','keeper_early','keeper_veto'):
        # Weak exploratory geometric hold rule; every setting has a no-op rival.
        outs=peaks(f,['OUT','Ball out of play'],.5)
        for lo,hi in HALVES:
            for aa,bb,t in runs(base[lo:hi]):
                a,b=lo+aa,lo+bb
                if t!=2 or a==lo or b==hi or not 50<=b-a<=750:continue
                if any(a-50<=f.times[j]<b for j in outs):continue
                if stage=='keeper_veto':
                    restarts=peaks(f,['THROW IN','Throw-in','Corner','Direct free-kick'],.5)
                    if any(a-50<=f.times[j]<b+75 for j in restarts):continue
                tm=int(base[b]);goal=(0 if tm==0 else 105.3) if lo<100000 else (105.3 if tm==0 else 0)
                count=0
                for tr,c in ((f.t,f.c),(f.b,f.cb)):
                    nm='h' if tm==0 else 'a'
                    end=min(b,a+100) if stage=='keeper_early' else b
                    radius=20 if stage=='keeper' else 16.5
                    width=22 if stage=='keeper' else 20
                    hold=(tr['source'][a:end]==1)&(abs(tr['x'][a:end]-goal)<radius)&(abs(tr['y'][a:end]-34)<width)
                    hold&=(c['dn_'+nm][a:end]<.2)&(c['rel_'+nm][a:end]>.25)&(c['rel_'+nm][a:end]<1.3)
                    count+=int(hold.sum())
                if count>=cfg['minimum'] and f.z['local'][a:b].mean()<cfg['dead']:p[a:b]=tm
    elif stage=='restart_lag':
        for lo,hi in HALVES:
            rr=runs(base[lo:hi])
            for j,(a,b,t) in enumerate(rr[:-1]):
                if t==2 and rr[j+1][2]<2:
                    end=lo+b+min(cfg['lag'],(rr[j+1][1]-b)//2)
                    p[lo+b:end]=2
    else:raise ValueError(stage)
    if clean and cfg['fringe']:
        # A new stoppage can expose a tiny decoder team fragment. This often
        # reflects early ownership assigned to the next restart's team. Merge
        # only a short fragment immediately adjacent to the newly inserted D.
        for a,b,t in runs((p==2)&(base!=2)):
            if not t:continue
            s=a
            while s>0 and p[s-1]==base[a-1] and a-s<=cfg['fringe']:s-=1
            if 0<a-s<=cfg['fringe'] and s>0 and p[s-1]<2:p[s:a]=p[s-1]
    for a,b in ((0,11185),(90652,110808),(190198,N)):p[a:b]=2
    return p


def grid(**opts):return [dict(zip(opts,v)) for v in itertools.product(*opts.values())]
def candidates(stage):
    if stage=='out_physical':return grid(minimum=[25,50])
    if stage=='out_confirmed':return grid(score=[.3,.5],tail=[0,25,50])
    if stage=='out_clean':return grid(score=[.3,.5],tail=[25,50],fringe=[0,50])
    if stage=='out_long':return grid(score=[.3,.5],tail=[25,50],fringe=[0,50],minimum=[50])
    if stage=='goal_clean':return grid(score=[.3,.5],rest=[20,40],off=[False,True],fringe=[0,50])
    if stage=='event_stop':return grid(score=[.3,.5,.7],dead=[0.,.05])
    if stage=='short_throw':return grid(score=[.5,.7],maximum=[150,300])
    if stage=='box_restart':return grid(score=[.3,.5],rest=[10,20])
    if stage=='whistle_snap':return grid(threshold=[.2,.3],cap=[100,250])
    if stage in ('brief_restart','brief_views','brief_control'):return grid(score=[.5,.7,.9],out=[.5,.7],maximum=[125,200],physical=[False,True])
    if stage in ('tag_goal','tag_goal_observed','tag_goal_line'):return grid(score=[.3,.5],rest=[20,40],off=[False,True])
    if stage=='rest_goal':return grid(score=[.3,.5,.7],rest=[20,40],off=[False,True])
    if stage in ('keeper','keeper_early','keeper_veto'):return grid(minimum=[5,10,20],dead=[.7,1.])
    if stage=='restart_lag':return grid(lag=[7,14])
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
    f = Features()
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
