"""Faithful generic rule ports from hc-stoppages and hc-edges; no source-model imports.
All evidence is supplied by evidence.Inputs. Names broad/local/state are label-free replacements.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from evidence import Inputs, BLOCKS, N, runs

class Stoppages(Inputs):
    def predict(self,spec):
        codes=self.base.copy();dead=np.zeros(N,bool);live=np.zeros(N,bool)
        for cfg in spec.get('outside',[]):
            ball=self.balls[cfg['source']];pre='p_' if cfg['source']=='provided' else 'v_'
            x,y,sp=ball[pre+'x'],ball[pre+'y'],ball[pre+'speed']
            sides=[(y,0,68)]
            if cfg.get('goal_lines'):sides.append((x,0,105.3))
            for pos,lo,hi in sides:
                outside=(pos<lo-cfg['distance'])|(pos>hi+cfg['distance'])
                for fa,fb in BLOCKS:
                    ix=np.flatnonzero(outside[fa:fb])+fa
                    for group in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',3))+1):
                        if len(group)<cfg['minimum']:continue
                        a,b=int(group[0]),int(group[-1])+1
                        # Back to the last observed in-bounds sample on this side.
                        left=max(fa,a-75)
                        outside0=(pos[left:a]<lo) if pos[a]<lo else (pos[left:a]>hi)
                        previous=np.flatnonzero(~outside0&np.isfinite(pos[left:a]))
                        start=left+int(previous[-1])+1 if len(previous) else a
                        stop=min(fb,a+cfg.get('maximum',1500))
                        sideinside=(pos>lo+cfg.get('inside',1))&(pos<hi-cfg.get('inside',1))
                        good=sideinside&(sp>cfg.get('kick',3))
                        ends=np.flatnonzero(good[b:stop])
                        if not len(ends):continue
                        end=b+int(ends[0])
                        if end-start<cfg.get('min_span',25) or end-start>cfg.get('maximum',1500):continue
                        if cfg.get('new_only') and np.any(codes[start:end]>=2):continue
                        if cfg.get('veto') and self.q[cfg['veto']][start:end].mean()<cfg['veto_min']:continue
                        codes[start:end]=2;dead[start:end]=True
        for cfg in spec.get('suppress',[]):
            q=self.smooth(cfg['model'],cfg.get('smooth',5))
            for a,b,t in runs(codes):
                if t>=2 or b-a>cfg.get('maximum',1500) or b-a<cfg.get('minimum',15):continue
                if q[a:b].mean()<cfg['mean']:continue
                if np.mean(q[a:b]>cfg.get('threshold',.5))<cfg.get('fraction',.5):continue
                end=b
                if cfg.get('restart'):
                    sp=self.balls['provided']['p_speed'];launches=[]
                    for t0 in range(a+50,b-10):
                        past=sp[t0-40:t0];future=sp[t0:t0+10]
                        if np.isfinite(past).sum()<25 or np.isfinite(future).sum()<5:continue
                        if np.nanmedian(past)<cfg.get('rest',1.5) and np.nanmedian(future)>cfg.get('kick',5):launches.append(t0)
                    if launches:end=launches[-1]+cfg.get('offset',0)
                codes[a:end]=2;dead[a:end]=True
                if cfg.get('restart_owner') and end<b:
                    ball=self.balls['provided'];opp=1-t
                    near=ball['p_dh' if opp==0 else 'p_da']
                    far=ball['p_da' if opp==0 else 'p_dh']
                    good=(near[end:b]<cfg.get('contact',1.5))&(far[end:b]>near[end:b])
                    touches=[(end+c,end+d) for c,d,v in runs(good.astype('i1')) if v and d-c>=cfg.get('contact_min',5)]
                    if touches:
                        start=touches[0][0]+cfg.get('touch_lag',0)
                        codes[start:b]=opp;live[start:b]=True
        for cfg in spec.get('ending_touches',[]):
            ball=self.balls[cfg.get('source','provided')];pre='p_' if cfg.get('source','provided')=='provided' else 'v_'
            q=self.smooth(cfg.get('status','broad'),5)
            for a,b,t in runs(codes):
                if t>=2 or b==N or codes[b]!=2 or b-a<50:continue
                tm=1-t
                dh=ball[pre+('dh' if tm==0 else 'da')]
                da=ball[pre+('da' if tm==0 else 'dh')]
                lo=max(a,b-cfg.get('lookback',30));hi=min(N,b+cfg.get('ahead',15))
                contact=(dh[lo:hi]<cfg.get('near',1.2))&(da[lo:hi]-dh[lo:hi]>cfg.get('margin',.2))&(q[lo:hi]<cfg.get('seed_dead',.4))
                contacts=[(lo+c,lo+d) for c,d,v in runs(contact.astype('i1')) if v and d-c>=cfg.get('minimum',3)]
                if not contacts:continue
                start,last=contacts[-1]
                if start==lo:continue
                if b-start>cfg.get('max_before',1000) or start-b>cfg.get('max_after',1000):continue
                if cfg.get('state') and self.states[cfg['state']][start:last,tm].mean()<cfg['state_min']:continue
                if cfg.get('local_max') is not None and self.q['local'][start:last].mean()>cfg['local_max']:continue
                if cfg.get('team_min') is not None:
                    confidence=self.team[start:last].mean() if tm==0 else 1-self.team[start:last].mean()
                    if confidence<cfg['team_min']:continue
                end=max(b,last)
                while end<min(N,b+cfg.get('maximum',75)) and codes[end]==2 and q[end]<cfg.get('grow',.7):end+=1
                end=max(end,start+cfg.get('hold',20))
                if end-start>cfg.get('maximum',75) or dead[start:end].any():continue
                codes[start:end]=tm;live[start:end]=True
        for cfg in spec.get('dead',[]):
            q=self.smooth(cfg['model'],cfg.get('smooth',15))
            for fa,fb in BLOCKS:
                ix=np.flatnonzero(q[fa:fb]>cfg['seed'])+fa
                for group in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',10))+1):
                    if len(group)<cfg['minimum']:continue
                    a,b=int(group[0]),int(group[-1])+1
                    while a>fa and q[a-1]>cfg['grow']:a-=1
                    while b<fb and q[b]>cfg['grow']:b+=1
                    if b-a>cfg.get('maximum',2000):continue
                    if cfg.get('veto') and self.q[cfg['veto']][a:b].mean()<cfg['veto_min']:continue
                    if cfg.get('interior',False) and np.any(codes[max(fa,a-25):min(fb,b+25)]>=2):continue
                    if cfg.get('new_only',False) and not np.all(codes[a:b]<2):continue
                    a+=cfg.get('start_shift',0);b+=cfg.get('end_shift',0)
                    if b-a<cfg.get('min_span',25):continue
                    codes[a:b]=2;dead[a:b]=True
        for cfg in spec.get('live',[]):
            q=self.smooth(cfg['model'],cfg.get('smooth',5))
            for fa,fb in BLOCKS:
                ix=np.flatnonzero((q[fa:fb]<cfg['seed'])&(codes[fa:fb]==2))+fa
                for group in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',10))+1):
                    if len(group)<cfg['minimum']:continue
                    a,b=int(group[0]),int(group[-1])+1
                    while a>fa and q[a-1]<cfg['grow'] and codes[a-1]==2:a-=1
                    while b<fb and q[b]<cfg['grow'] and codes[b]==2:b+=1
                    if b-a>cfg.get('maximum',500) or not np.all(codes[a:b]==2):continue
                    if dead[a:b].any():continue
                    if cfg.get('veto') and self.q[cfg['veto']][a:b].mean()>cfg['veto_max']:continue
                    owner=self.team if not cfg.get('owner') else self.states[cfg['owner']][:,0]/np.maximum(self.states[cfg['owner']][:,:2].sum(1),1e-6)
                    tm=int(owner[a:b].mean()<.5)
                    if max(owner[a:b].mean(),1-owner[a:b].mean())<cfg.get('confidence',.5):continue
                    margin=cfg.get('separate',0)
                    if margin and (np.any(codes[max(fa,a-margin):a]==tm) or np.any(codes[b:min(fb,b+margin)]==tm)):continue
                    codes[a:b]=tm;live[a:b]=True
        for cfg in spec.get('state_intervals',[]):
            state=self.states[cfg['model']]
            for tm in (0,1):
                q=state[:,tm].copy()
                for fa,fb in BLOCKS:
                    q[fa:fb]=uniform_filter1d(q[fa:fb],cfg.get('smooth',5),mode='nearest')
                    ix=np.flatnonzero((q[fa:fb]>cfg['seed'])&(codes[fa:fb]!=tm))+fa
                    for group in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',10))+1):
                        if len(group)<cfg['minimum']:continue
                        a,b=int(group[0]),int(group[-1])+1
                        while a>fa and q[a-1]>cfg['grow'] and codes[a-1]!=tm:a-=1
                        while b<fb and q[b]>cfg['grow'] and codes[b]!=tm:b+=1
                        if b-a>cfg.get('maximum',200):continue
                        if dead[a:b].any():continue
                        if (codes[a:b]==2).mean()<cfg.get('dead_fraction',.5):continue
                        if state[a:b,2].mean()>cfg.get('dead_max',.5):continue
                        codes[a:b]=tm;live[a:b]=True
        return codes

class Edges(Inputs):
    def predict(self,spec):
        p=self.base.copy();events=[]
        for cfg in spec.get('goals',[]):
            ball=self.ball[cfg.get('source','provided')];pre={'provided':'p','video':'v','broadcast':'b'}[cfg.get('source','provided')]
            x,y,sp=ball[pre+'_x'],ball[pre+'_y'],self.ball['provided']['p_speed']
            outside=((x < -cfg.get('distance',0.))|(x > 105.3+cfg.get('distance',0.)))&(abs(y-34)<cfg.get('width',20))
            for fa,fb in BLOCKS:
                ix=np.flatnonzero(outside[fa:fb])+fa
                for group in np.split(ix,np.flatnonzero(np.diff(ix)>3)+1):
                    if len(group)<cfg.get('minimum',5):continue
                    a,b=int(group[0]),int(group[-1])+1
                    lo=max(fa,a-50);pos=x[lo:a]
                    inside=(pos>=0)&(pos<=105.3)
                    prev=np.flatnonzero(inside)
                    if not len(prev):continue
                    start=lo+int(prev[-1])+1
                    hi=min(fb,b+cfg.get('context',250))
                    if self.support[b:hi].max()<cfg.get('support',.7):continue
                    if self.local[b:hi].mean()<cfg.get('local',.7):continue
                    launches=[]
                    for t in range(b+20,min(fb,a+1000)):
                        past=sp[max(b,t-40):t];future=sp[t:t+10]
                        if np.isfinite(past).sum()<25 or np.isfinite(future).sum()<5:continue
                        if self.local[t:t+10].mean()>cfg.get('restart_dead',1):continue
                        if np.nanmedian(past)<1.5 and np.nanmedian(future)>cfg.get('kick',3):launches.append(t)
                    if not launches:continue
                    end=launches[0]
                    if end-start<50:continue
                    p[start:end]=2;events.append(dict(kind='goal',start=start,end=end,seed=[a,b]))
        for cfg in spec.get('contacts',[]):
            # A neighbouring fold's fitted probability may have seen this
            # fold's labels. Never read it across a fold seam.
            fold_runs=[(fa,fb,fa+a,fa+b,t) for fa,fb in BLOCKS for a,b,t in runs(p[fa:fb])]
            for fa,fb,a,b,t in fold_runs:
                if t>=2 or b==fb or p[b]!=2 or b-a<50:continue
                tm=1-t;lo=max(a,b-cfg.get('lookback',50));hi=min(fb,b+15)
                votes=np.zeros(hi-lo,'i1')
                for s in cfg.get('sources',['provided','video']):
                    pre={'provided':'p','video':'v','broadcast':'b'}[s];ball=self.ball[s]
                    near=ball[pre+('_dh' if tm==0 else '_da')][lo:hi]
                    far=ball[pre+('_da' if tm==0 else '_dh')][lo:hi]
                    votes+=(near<cfg.get('near',1.2))&(far-near>cfg.get('margin',.2))
                good=(votes>=cfg.get('votes',2))&(self.broad[lo:hi]<cfg.get('dead',.4))&(self.local[lo:hi]<cfg.get('local',.15))
                good&=self.state[lo:hi,tm]>=cfg.get('state',.15)
                groups=[(lo+c,lo+d) for c,d,v in runs(good.astype('i1')) if v and d-c>=cfg.get('minimum',3)]
                if not groups:continue
                start,last=groups[-1]
                if start==lo:continue
                if cfg.get('new_contact'):
                    ball=self.ball['provided'];near=ball['p_dh' if tm==0 else 'p_da']
                    prev=near[max(a,start-50):max(a,start-10)]
                    if np.isfinite(prev).sum()<10 or np.nanmedian(prev)<cfg['new_contact']:continue
                if cfg.get('rise'):
                    past=self.state[max(a,start-60):max(a,start-15),tm]
                    if not len(past) or self.state[start:last,tm].mean()-past.mean()<cfg['rise']:continue
                end=max(b,last)
                while end<min(fb,b+75) and p[end]==2 and self.broad[end]<cfg.get('grow',.5):end+=1
                end=min(fb,max(end,start+cfg.get('hold',35)))
                if end-start>100:continue
                p[start:end]=tm;events.append(dict(kind='contact',start=start,end=end,team=tm,votes=int(votes[start-lo:last-lo].min())))
        for cfg in spec.get('set_pieces',[]):
            kind=cfg['kind'];s=cfg.get('source','video');pre={'provided':'p','video':'v','broadcast':'b'}[s]
            ball=self.ball[s];x,y,sp=ball[pre+'_x'],ball[pre+'_y'],ball[pre+'_speed']
            if kind=='penalty':anchors=[(11.,34.),(94.3,34.)]
            else:anchors=[(xx,yy) for xx in (0.,105.3) for yy in (0.,68.)]
            proposals=[]
            for xx,yy in anchors:
                dist=np.hypot(x-xx,y-yy)
                rest=(dist<cfg.get('radius',2))&(sp<cfg.get('rest',2.5))
                for fa,fb in BLOCKS:
                    ix=np.flatnonzero(rest[fa:fb])+fa
                    for group in np.split(ix,np.flatnonzero(np.diff(ix)>cfg.get('gap',15))+1):
                        if len(group)<cfg.get('minimum',20):continue
                        a,b=int(group[0]),int(group[-1])+1
                        if (p[max(fa,b-100):b]==2).mean()<.8:continue
                        if kind=='penalty':flight=(dist>3)&(sp>8)&((x-xx)*(1 if xx<50 else -1)<-2)
                        else:flight=(dist>4)&(sp>8)&(x>0)&(x<105.3)&(y>0)&(y<68)
                        hits=np.flatnonzero(flight[b:min(fb,b+cfg.get('launch_gap',40))])
                        if not len(hits):continue
                        t=b+int(hits[0]);start=(b+t)//2
                        nearh=ball[pre+'_dh'][max(a,b-25):b];neara=ball[pre+'_da'][max(a,b-25):b]
                        if np.isfinite(nearh).sum()<5 or np.isfinite(neara).sum()<5:continue
                        tm=int(np.nanmedian(neara)<np.nanmedian(nearh))
                        confidence=abs(np.nanmedian(neara)-np.nanmedian(nearh))
                        if confidence<cfg.get('margin',2):continue
                        if p[start]!=2:continue
                        if kind=='penalty':
                            other=self.ball['broadcast'];bx,by=other['b_x'],other['b_y']
                            cross=((bx<0) if xx<50 else (bx>105.3))&(by>0)&(by<68)
                            stops=np.flatnonzero(cross[start+10:min(fb,start+100)])
                            end=start+10+int(stops[0]) if len(stops) else min(fb,start+25)
                        else:
                            stop=min(fb,start+cfg.get('maximum',250));end=stop
                            for j in range(start+30,stop):
                                if self.state[j:min(fb,j+5),1-tm].mean()>.75 or self.broad[j:min(fb,j+15)].mean()>.9:
                                    end=j;break
                        if end-start<10:continue
                        proposals.append(dict(kind=kind,start=start,end=end,team=tm,rest=[a,b],launch=t))
            for ev in proposals:
                if cfg.get('minimum_span') and ev['end']-ev['start']<cfg['minimum_span']:continue
                p[ev['start']:ev['end']]=ev['team'];events.append(ev)
        return p,events
