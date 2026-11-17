"""Physical events from raw video ball/player geometry and permitted raw channels."""
from common import *
from scipy.ndimage import uniform_filter1d

class Evidence:
    def __init__(self):
        self.f=up.Features()
        self.cache={}

    def contact(self,distance=.2,view='both',exclusive=True,foot=False):
        key=(distance,view,exclusive,foot)
        if key in self.cache:return self.cache[key]
        f=self.f;out=np.zeros((N,2),bool)
        for c,tr,name in ((f.c,f.t,'tactical'),(f.cb,f.b,'broadcast')):
            if view not in ('both',name):continue
            for tm,nm,op in [(0,'h','a'),(1,'a','h')]:
                dn=c['foot_'+nm] if foot else c['dn_'+nm]
                good=(tr['source']==1)&(dn<distance)&(c['p_'+nm]>.6)
                good&=(c['rel_'+nm]>-.3)&(c['rel_'+nm]<1.3)
                if exclusive:good&=c['dn_'+op]>c['dn_'+nm]+.1
                out[:,tm]|=good
        self.cache[key]=out
        return out

    def onsets(self,distance=.2,minimum=3,view='both',exclusive=True,foot=False):
        key=('onsets',distance,minimum,view,exclusive,foot)
        if key in self.cache:return self.cache[key]
        co=self.contact(distance,view,exclusive,foot);out=[]
        for tm in (0,1):
            ix=np.flatnonzero(co[:,tm]);g=np.split(ix,np.flatnonzero(np.diff(ix)>5)+1)
            out.append(np.array([v[0] for v in g if len(v)>=minimum],dtype=int))
        self.cache[key]=out
        return out

    def exits(self,view='both'):
        if ('exits',view) in self.cache:return self.cache['exits',view]
        f=self.f
        # A two-camera exit detector rejects elevated balls projecting out.
        if view=='both':ix=np.flatnonzero(f.det['out_on']>0)
        else:
            tr=f.t if view=='tactical' else f.b
            x,y=tr['x'],tr['y']
            outside=(tr['source']==1)&((x<-.3)|(x>105.6)|(y<-.3)|(y>68.3))
            ix=np.flatnonzero(outside & ~np.r_[False,outside[:-1]])
            # Consecutive observed samples beyond a line corroborate a crossing.
            good=uniform_filter1d(outside.astype(float),size=11)>.18
            ix=ix[good[ix]]
        self.cache['exits',view]=ix
        return ix

    def launches(self):
        if 'launches' in self.cache:return self.cache['launches']
        ix=[]
        for tr in (self.f.t,self.f.b):
            src=tr['source'];sp=tr['speed']
            # Rest in at least 10 of the last 20 frames, then observed motion.
            rest=(src==1)&(sp<1.5)
            count=np.convolve(rest.astype(int),np.ones(20,int),'full')[:N]
            fast=(src==1)&(sp>4)&(sp<40)
            sustained=np.convolve(fast.astype(int),np.ones(7,int),'same')>=3
            ix.extend(np.flatnonzero((count>=10)&fast&sustained))
        self.cache['launches']=np.unique(ix)
        return self.cache['launches']

    def carrier(self):
        if 'carrier' not in self.cache:
            z=np.load(HERE.parent/'free-carrier/.local/carrier_s.5.npz')
            self.cache['carrier']=z['p_home']
        return self.cache['carrier']

    def actions(self,threshold=.7):
        key=('actions',threshold)
        if key not in self.cache:
            f=self.f;e=f.ev
            good=(e['score']>=threshold)&np.isin(f.names,['PASS','DRIVE','HEADER','CROSS','SHOT','TACKLE','BALL PLAYER','Pass','Drive','Header','Cross','Shot','Tackle','Interception','Ball player'])
            good&=np.isfinite(e['p_home'])&(abs(e['p_home']-.5)>.3)
            self.cache[key]=[np.unique(f.times[good&(e['p_home']>.5)]),np.unique(f.times[good&(e['p_home']<.5)])]
        return self.cache[key]

    def handovers(self,view='both',minimum=3):
        key=('handovers',view,minimum)
        if key in self.cache:return self.cache[key]
        co=self.contact(.2,view,True)
        # Keep only changes in the last observed, unambiguous contacting team.
        # Do not snap an already-established carry to another same-team touch.
        valid=np.flatnonzero(co.any(1)&~co.all(1));team=co[valid,1].astype(int)
        out=[[],[]]
        for i in np.flatnonzero(team[1:]!=team[:-1])+1:
            t=int(valid[i]);tm=int(team[i])
            if t-valid[i-1]>100:continue
            if co[t:t+10,tm].sum()<minimum:continue
            out[tm].append(t)
        self.cache[key]=[np.array(x,int) for x in out]
        return self.cache[key]

def transitions(p):
    ix=np.flatnonzero(p[1:]!=p[:-1])+1
    return [(int(t),int(p[t-1]),int(p[t])) for t in ix]

def apply(ev,base,stage,cfg):
    if not cfg:return base.copy()
    p=base.copy();tt=transitions(base)
    if stage=='contact_snap':
        events=ev.onsets(cfg['distance'],cfg.get('minimum',3),cfg.get('view','both'),cfg.get('exclusive',True),cfg.get('foot',False))
    elif stage in ('flight_snap','receive_snap'):
        events=ev.onsets(cfg['distance'],2,'tactical',True)
        co=ev.contact(cfg['distance'],'tactical',True)
    elif stage=='line_snap':events=ev.exits(cfg.get('view','both'))
    elif stage=='launch_snap':events=ev.launches()
    elif stage=='action_snap':events=ev.actions(cfg['threshold'])
    elif stage in ('handover_snap','supported_snap'):
        events=ev.handovers(cfg['view'],cfg.get('minimum',3))
        co=ev.contact(.2,cfg['view'],True)
    elif stage=='signal_decode':
        from scipy.special import logit
        q=ev.carrier() if cfg['source']=='carrier' else ev.f.z['team']
        signal=logit(np.clip(np.nan_to_num(q,nan=.5),.005,.995))
    elif stage=='contact_decode':co=ev.contact(cfg['distance'],exclusive=cfg.get('exclusive',True)).astype(float)
    for i,(t,a,b) in enumerate(tt):
        if t in (11185,90652,110808,190198):continue
        if stage in ('contact_snap','contact_decode','carrier_snap','signal_decode','action_snap','handover_snap','supported_snap','flight_snap','receive_snap') and not(a<2 and b<2):continue
        if stage in ('flight_snap','receive_snap'):
            if co[max(0,t-10):t+11].any():continue
            if not 6<ev.f.t['speed'][t]<40:continue
        if stage=='shift' and (('team' if a<2 and b<2 else 'entry' if b==2 else 'restart')!=cfg['kind']):continue
        if stage=='line_snap' and not(a<2 and b==2):continue
        if stage in ('launch_snap','restart_contact') and not(a==2 and b<2):continue
        if not any(lo<t<hi for lo,hi in HALVES):continue
        # Preserve topology and at least half of every neighbouring short run.
        prev=tt[i-1][0] if i else 0;nxt=tt[i+1][0] if i+1<len(tt) else N
        lo=max(t-cfg['cap'],prev+(t-prev)//2);hi=min(t+cfg['cap'],nxt-(nxt-t)//2)
        if stage in ('contact_snap','line_snap','launch_snap','carrier_snap','action_snap','handover_snap','supported_snap','flight_snap','receive_snap'):
            es=events[b] if stage in ('contact_snap','carrier_snap','action_snap','handover_snap','supported_snap','flight_snap','receive_snap') else events
            near=es[(es>=lo)&(es<=hi)]
            if stage=='receive_snap':near=near[near>t]
            if not len(near):continue
            v=int(near[np.argmin(abs(near-t))])+cfg.get('lag',0)
            if stage=='supported_snap':
                # Move only when continued contact contradicts the old boundary.
                s,e=sorted((v,t));tm=a if v>t else b
                counts=co[s:e].sum(0)
                if counts[tm]<cfg['support'] or counts[1-tm]>1:continue
        elif stage=='contact_decode':
            sig=co[lo:hi+1,b]-co[lo:hi+1,a]
            # Single changepoint: old contacts before, new contacts after.
            cost=np.r_[0,np.cumsum(sig)]+cfg['penalty']*abs(np.arange(lo,hi+2)-t)
            v=int(lo+np.argmin(cost))
        elif stage=='signal_decode':
            sig=signal[lo:hi+1]*(1 if b==0 else -1)
            cost=np.r_[0,np.cumsum(sig)]+cfg['penalty']*abs(np.arange(lo,hi+2)-t)
            v=int(lo+np.argmin(cost))
        elif stage=='shift':v=t+cfg['lag']
        elif stage=='restart_contact':
            co=ev.contact(.2)[:,b]
            launches=ev.launches()
            near=launches[(launches>=lo)&(launches<=hi)]
            if not len(near):continue
            near=np.array([v for v in near if co[max(0,v-5):v+3].any()])
            if not len(near):continue
            v=int(near[np.argmin(abs(near-t))])
        else:raise ValueError(stage)
        v=int(np.clip(v,lo,hi))
        if v<t:p[v:t]=b
        elif v>t:p[t:v]=a
    return p
