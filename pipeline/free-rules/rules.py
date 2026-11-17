"""Status ports and new local repairs. All decisions are evidence-based."""
import numpy as np
from evidence import BLOCKS, N, runs
from ports import Stoppages, Edges


def islands(ev,base,cfg):
    """hc-status hysteresis, duration, separation, owner and global-dead veto."""
    p=base.copy(); q=ev.smooth(cfg.get('model','local'),5)
    for fa,fb in BLOCKS:
        ix=np.flatnonzero((p[fa:fb]==2)&(q[fa:fb]<cfg.get('seed',.05)))+fa
        for g in np.split(ix,np.flatnonzero(np.diff(ix)>10)+1):
            if len(g)==0:continue
            a,b=int(g[0]),int(g[-1])+1
            if b-a<cfg.get('minimum',8) or not np.all(p[a:b]==2):continue
            while a>fa and p[a-1]==2 and q[a-1]<cfg.get('grow',.25):a-=1
            while b<fb and p[b]==2 and q[b]<cfg.get('grow',.25):b+=1
            if b-a>cfg.get('maximum',75):continue
            if ev.broad[a:b].mean()>cfg.get('veto',.5):continue
            owner=ev.team if cfg.get('owner','raw')=='raw' else ev.post_team
            qh=owner[a:b].mean();tm=int(qh<.5)
            if max(qh,1-qh)<cfg.get('confidence',.78):continue
            margin=cfg.get('separate',1)
            if margin and (np.any(p[max(fa,a-margin):a]==tm) or np.any(p[b:min(fb,b+margin)]==tm)):continue
            p[a:b]=tm
    return p


def goal_status(ev,base,cfg):
    """hc-status goal exit/rest/kick with physical keeper support."""
    p=base.copy();ball=ev.ball['provided'];x,y,sp=ball['p_x'],ball['p_y'],ball['p_speed']
    outside=((x < -1)|(x>106.3))&(abs(y-34)<34)
    for fa,fb in BLOCKS:
        ix=np.flatnonzero(outside[fa:fb])+fa
        for g in np.split(ix,np.flatnonzero(np.diff(ix)>3)+1):
            if len(g)<12:continue
            a,b=int(g[0]),int(g[-1])+1
            if ev.support[b:min(fb,b+200)].max(initial=0)<cfg.get('support',.7):continue
            lo=max(fa,a-75);side=(x[lo:a]<0) if x[a]<0 else (x[lo:a]>105.3)
            prev=np.flatnonzero(~side&np.isfinite(x[lo:a]));start=lo+int(prev[-1])+1 if len(prev) else a
            for t in range(b+25,min(fb,a+1000)):
                past=sp[max(b,t-40):t];future=sp[t:min(fb,t+10)]
                if len(past)<20 or np.isfinite(past).sum()<15 or np.isfinite(future).sum()<5:continue
                if np.nanmedian(past)>1.5 or np.nanmedian(future)<cfg.get('kick',5) or not 0<=x[t]<=105.3:continue
                if t-start>=50:p[start:t]=2
                break
    return p


def contacts(ev,base,cfg):
    """Contact-supported brief opponent control, with optional spotted action.

    Generic extension of hc-edges novelty tests to the interior of a live run.
    A contact is not automatically a possession: require duration + corroboration.
    """
    p=base.copy();src=cfg.get('source','provided');pre={'provided':'p','video':'v','broadcast':'b'}[src]
    ball=ev.ball[src]
    for fa,fb in BLOCKS:
        for aa,bb,t in runs(p[fa:fb]):
            a,b=fa+aa,fa+bb
            if t>=2 or b-a<100:continue
            tm=1-t;near=ball[pre+('_dh' if tm==0 else '_da')];far=ball[pre+('_da' if tm==0 else '_dh')]
            good=(near<cfg.get('near',1.5))&(far-near>cfg.get('margin',1))&(ev.local<cfg.get('dead',.5))
            ix=np.flatnonzero(good[a:b])+a
            for g in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',5))+1):
                if len(g)<cfg.get('minimum',5):continue
                s,e=int(g[0]),int(g[-1])+1
                if s-a<15 or b-e<15 or e-s>cfg.get('maximum',150):continue
                own=ev.team[s:e].mean() if tm==0 else 1-ev.team[s:e].mean()
                signed=ev.spot_team[s:min(b,e+25)]*(1 if tm==0 else -1)
                if own<cfg.get('team',.5) and signed.max(initial=0)<cfg.get('spot',.5):continue
                while e<b and e-s<cfg.get('maximum',150) and ev.team[e]*(1 if tm==0 else -1)+(0 if tm==0 else 1)>.5:e+=1
                if e-s<cfg.get('hold',20):continue
                p[s:e]=tm
    return p


def spot_islands(ev,base,cfg):
    """A detected kick/delivery within D, followed by independently live evidence."""
    p=base.copy()
    for fa,fb in BLOCKS:
        ix=np.flatnonzero((p[fa:fb]==2)&(ev.spot_live[fa:fb]>cfg.get('spot',.3)))+fa
        for g in np.split(ix,np.flatnonzero(np.diff(ix)>25)+1):
            if not len(g):continue
            s,e=int(g[0]),int(g[-1])+1
            if s<=fa or p[s-1]!=2:continue
            if ev.local[s:e].mean()>cfg.get('dead',.7):continue
            while e<fb and p[e]==2 and ev.local[e]<cfg.get('grow',.5) and e-s<250:e+=1
            if e-s<cfg.get('minimum',15) or e-s>cfg.get('maximum',200):continue
            if ev.broad[s:e].mean()>cfg.get('veto',.9):continue
            qh=ev.team[s:e].mean()
            if max(qh,1-qh)<cfg.get('confidence',.6):continue
            p[s:e]=int(qh<.5)
    return p


def event_stops(ev,base,cfg):
    """Spotted out/foul followed by a restart, corroborated by status/ball rest."""
    from evidence import HERE
    z=np.load(HERE.parent/'free-pretrained/.local/tdeed_events.npz')
    names=np.where(z['sn'],z['names17'][np.clip(z['cls'],0,16)],z['names12'][np.clip(z['cls'],0,11)])
    tt=np.rint(z['t']).astype(int);ss=z['score'];bro=z['broadcast']
    exits=np.isin(names,['OUT','Ball out of play','Foul','Offside'])
    restarts=np.isin(names,['THROW IN','FREE KICK','Throw-in','Corner','Indirect free-kick','Direct free-kick','Kick-off'])
    p=base.copy()
    for fa,fb in BLOCKS:
        starts=tt[exits&(ss>=cfg.get('score',.5))&(tt>=fa)&(tt<fb)&bro]
        ends=np.sort(tt[restarts&(ss>=cfg.get('score',.5))&(tt>=fa)&(tt<fb)&bro])
        for s in starts:
            future=ends[(ends>s+10)&(ends<s+cfg.get('maximum',500))]
            if not len(future):continue
            e=int(future[0]);s=int(s)
            if np.any(p[s:e]==2):continue
            q=ev.local[s:e].mean()
            if q<cfg.get('dead',.1):continue
            p[s:e]=2
    return p


def contact_chains(ev,base,cfg):
    """Repeated video foot contacts plus independent team support inside live play."""
    p=base.copy()
    for fa,fb in BLOCKS:
        for tm in (0,1):
            contact=ev.contact*(1 if tm==0 else -1)
            ix=np.flatnonzero((contact[fa:fb]>0)&(p[fa:fb]<2)&(p[fa:fb]!=tm))+fa
            for g in np.split(ix,np.flatnonzero(np.diff(ix)>15)+1):
                if len(g)<cfg.get('minimum',5):continue
                a,b=int(g[0]),int(g[-1])+1
                if b-a>200 or not np.all(p[a:b]==1-tm):continue
                owner=ev.team if tm==0 else 1-ev.team
                if owner[a:b].mean()<cfg.get('team',.6):continue
                # Grow within the independent team excursion, then hold to the
                # next observed opposing contact (at most half a second).
                while a>fa and p[a-1]==1-tm and owner[a-1]>.5 and b-a<200:a-=1
                while b<fb and p[b]==1-tm and owner[b]>.5 and b-a<200:b+=1
                if b-a<cfg.get('hold',20):continue
                p[a:b]=tm
    return p


def snap_events(ev,base,cfg):
    """Align existing dead/live edges with nearby pretrained out/restart peaks."""
    from evidence import HERE
    z=np.load(HERE.parent/'free-pretrained/.local/tdeed_events.npz')
    names=np.where(z['sn'],z['names17'][np.clip(z['cls'],0,16)],z['names12'][np.clip(z['cls'],0,11)])
    tt=np.rint(z['t']).astype(int)
    ok=(z['score']>=.5)&z['broadcast']
    restarts=np.sort(tt[ok&np.isin(names,['THROW IN','FREE KICK','Throw-in','Corner','Indirect free-kick','Direct free-kick','Kick-off'])])
    exits=np.sort(tt[ok&np.isin(names,['OUT','Ball out of play','Foul','Offside'])])
    p=base.copy()
    for fa,fb in BLOCKS:
        rr=runs(base[fa:fb])
        for i,(a,b,t) in enumerate(rr[:-1]):
            a,b=a+fa,b+fa;nt=rr[i+1][2];ne=fa+rr[i+1][1]
            if t==2 and nt<2:
                ix=restarts[(restarts>=max(a+10,b-cfg.get('back',100)))&(restarts<=min(ne-10,b+25))]
                if not len(ix):continue
                v=int(ix[np.argmin(abs(ix-b))])+cfg.get('offset',0)
                v=max(a+10,min(ne-10,v))
                if ev.local[v:min(ne,v+25)].mean()>.5:continue
                if v<b:p[v:b]=nt
                else:p[b:v]=2
            if t<2 and nt==2 and cfg.get('entry',False):
                ix=exits[(exits>=max(a+10,b-75))&(exits<=min(ne-10,b+75))]
                if not len(ix):continue
                v=int(ix[np.argmin(abs(ix-b))])+cfg.get('offset',0)
                v=max(a+10,min(ne-10,v))
                if v<b:p[v:b]=2
                else:p[b:v]=t
    return p


def trim_dead(ev,base,cfg):
    """Low local-dead evidence at an existing restart/whistle corrects lag."""
    p=base.copy();q=ev.local
    for fa,fb in BLOCKS:
        rr=runs(base[fa:fb])
        for i,(a,b,t) in enumerate(rr[:-1]):
            a,b=a+fa,b+fa;nt=rr[i+1][2];ne=fa+rr[i+1][1]
            if t==2 and nt<2:
                v=b
                while v>max(a+15,b-cfg.get('maximum',150)) and q[v-1]<cfg.get('threshold',.1):v-=1
                p[v:b]=nt
            if t<2 and nt==2 and cfg.get('entry',False):
                v=b
                while v<min(ne-15,b+cfg.get('maximum',150)) and q[v]<cfg.get('threshold',.1):v+=1
                p[b:v]=t
    return p


def contact_islands(ev,base,cfg):
    """Foot-contact episodes bound short live phases without long hysteresis tails."""
    p=base.copy()
    for fa,fb in BLOCKS:
        for aa,bb,t in runs(base[fa:fb]):
            if t!=2:continue
            a,b=fa+aa,fa+bb
            for tm in (0,1):
                ct=ev.contact[a:b]*(1 if tm==0 else -1)
                ix=np.flatnonzero(ct>0)+a
                for g in np.split(ix,np.flatnonzero(np.diff(ix)>15)+1):
                    if len(g)<cfg.get('minimum',5):continue
                    s,e=int(g[0]),int(g[-1])+1
                    if e-s>150:continue
                    if ev.local[s:e].mean()>cfg.get('dead',.3) or ev.broad[s:e].mean()>.5:continue
                    owner=ev.team[s:e].mean() if tm==0 else 1-ev.team[s:e].mean()
                    if owner<.7:continue
                    sp=ev.ball['video']['v_speed'][g]
                    if np.nanmedian(sp)<cfg.get('speed',2):continue
                    # Half-second lookahead covers the pass/flight after last contact.
                    e=min(b,e+cfg.get('tail',15));s=max(a,s-3)
                    if e-s<15:continue
                    p[s:e]=tm
    return p


def remove_flicker(ev,base,cfg):
    """Merge brief unsupported opponent excursions inside one live possession."""
    p=base.copy()
    for fa,fb in BLOCKS:
        rr=runs(base[fa:fb])
        for i,(a,b,t) in enumerate(rr):
            a,b=a+fa,b+fa
            if t>=2 or i==0 or i==len(rr)-1 or b-a>cfg.get('maximum',75):continue
            if rr[i-1][2]!=1-t or rr[i+1][2]!=1-t:continue
            owner=ev.team[a:b].mean() if t==0 else 1-ev.team[a:b].mean()
            if owner>cfg.get('confidence',.5):continue
            p[a:b]=1-t
    return p


def keeper_holds(ev,base,cfg):
    """Recover a keeper holding a live ball, vetoed by an observed goal exit."""
    p=base.copy();ball=ev.ball['provided'];x,y,sp=ball['p_x'],ball['p_y'],ball['p_speed']
    for fa,fb in BLOCKS:
        for aa,bb,t in runs(base[fa:fb]):
            a,b=fa+aa,fa+bb
            if t!=2 or a==fa or b==fb or b-a<50 or b-a>cfg.get('maximum',750):continue
            tm=int(base[b]);goal=(0 if tm==0 else 105.3) if fa<100000 else (105.3 if tm==0 else 0)
            near=ball['p_dh' if tm==0 else 'p_da'];far=ball['p_da' if tm==0 else 'p_dh']
            good=(abs(x-goal)<16)&(abs(y-34)<20)&(sp<5)&(near<2)&(far>4)&(x>0)&(x<105.3)
            if good[a:b].sum()<10 or good[a:b].mean()<cfg.get('fraction',.2):continue
            if not good[a:min(b,a+50)].any():continue
            if ev.local[a:b].mean()>cfg.get('dead',.3):continue
            if np.any((x[max(fa,a-50):b]<0)|(x[max(fa,a-50):b]>105.3)):continue
            p[a:b]=tm
    return p


def apply(ev,base,stage,cfg):
    if not cfg:return base.copy()
    if stage=='status_port':
        p=islands(ev,base,dict(model='broad',seed=cfg['seed'],grow=cfg['grow'],minimum=20,
                              maximum=cfg['maximum'],confidence=.78,veto=1,separate=0))
        p=islands(ev,p,dict(model='local',seed=cfg['seed'],grow=cfg['grow'],minimum=8,
                           maximum=cfg['maximum'],confidence=.78,veto=.5,separate=1))
        return goal_status(ev,p,dict(support=.3,kick=5))
    if stage=='stoppages_port':
        ev.base=base
        return Stoppages.predict(ev,dict(
            suppress=[dict(model='local',mean=cfg['mean'],maximum=1500,restart=True,rest=1.5,kick=3,
                           restart_owner=True,contact=2,contact_min=3,touch_lag=10)],
            state_intervals=[dict(model='state',seed=cfg['state'],minimum=10,grow=.5,dead_fraction=.1,maximum=50)],
            ending_touches=[dict(source='provided',near=.8,margin=.3,grow=.4,hold=35,
                                 local_max=cfg['local'],team_min=.5,minimum=3)]))
    if stage=='edges_port':
        ev.base=base
        return Edges.predict(ev,dict(goals=[dict(distance=0,restart_dead=.5,support=.3,local=cfg['local'])],
            contacts=[dict(sources=['provided'],votes=1,near=.8,state=cfg['state'],rise=.15,new_contact=2,
                           local=cfg['local'])],set_pieces=[dict(kind='penalty',source='broadcast',minimum=50),
            dict(kind='corner',radius=2,minimum=5,launch_gap=50,minimum_span=75)]))[0]
    if stage in ('islands','islands_broad'):return islands(ev,base,cfg)
    if stage=='goal_status':return goal_status(ev,base,cfg)
    if stage=='contacts':return contacts(ev,base,cfg)
    if stage=='spot_islands':return spot_islands(ev,base,cfg)
    if stage=='event_stops':return event_stops(ev,base,cfg)
    if stage=='contact_chains':return contact_chains(ev,base,cfg)
    if stage=='snap_events':return snap_events(ev,base,cfg)
    if stage=='trim_dead':return trim_dead(ev,base,cfg)
    if stage=='contact_islands':return contact_islands(ev,base,cfg)
    if stage=='remove_flicker':return remove_flicker(ev,base,cfg)
    if stage=='keeper_holds':return keeper_holds(ev,base,cfg)
    ev.base=base
    if stage=='brief_owner':return Stoppages.predict(ev,{'state_intervals':[cfg]})
    if stage=='suppress_broad':return Stoppages.predict(ev,{'suppress':[cfg]})
    if stage in ('suppress','state_intervals','ending_touches','dead','live','outside'):
        return Stoppages.predict(ev,{stage:[cfg]})
    key={'goal_edges':'goals','new_contact':'contacts','penalty':'set_pieces','corner':'set_pieces'}[stage]
    return Edges.predict(ev,{key:[cfg]})[0]
