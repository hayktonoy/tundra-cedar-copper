"""Geometry, model signals, and pretrained events for the stoppage rules."""
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
LOCAL=HERE/'.local'
N=199319
HALVES=((11185,90652),(110808,190198))


def runs(c):
    edges=np.r_[0,np.flatnonzero(c[1:]!=c[:-1])+1,len(c)]
    return [(int(a),int(b),int(c[a])) for a,b in zip(edges[:-1],edges[1:])]


def groups(mask,gap=10,minimum=1):
    ix=np.flatnonzero(mask)
    return [g for g in np.split(ix,np.flatnonzero(np.diff(ix)>gap)+1) if len(g)>=minimum]


class Features:
    def __init__(self):
        self.t=dict(np.load(EXP/'hc-ball/.local/track_tactical.npz'))
        # Broadcast detector/tracker is raw geometry, not a fitted status head.
        self.b=dict(np.load(EXP/'hc-ball/.local/track_broadcast.npz'))
        self.c=dict(np.load(EXP/'hc-ball/.local/contact_tactical.npz'))
        self.cb=dict(np.load(EXP/'free-brief/.local/contact_broadcast.npz'))
        self.det=dict(np.load(EXP/'free-brief/.local/detect.npz'))
        self.z=dict(np.load(EXP/'free-rules/.local/inputs.npz'))
        ev=dict(np.load(EXP/'free-pretrained/.local/tdeed_events.npz'))
        self.ev=ev
        self.names=np.where(ev['sn'],ev['names17'][np.clip(ev['cls'],0,16)],ev['names12'][np.clip(ev['cls'],0,11)])
        self.times=np.rint(ev['t']).astype(int)
        f=np.load(EXP/'free-owner/.local/features.npz')
        self.off=np.ones(N)
        self.off[f['frames']]=f['Xs'][:,list(f['names_s']).index('bc_off')]
        self.owner_events=dict(np.load(EXP/'free-owner/.local/events.npz'))

    def events(self,names,threshold=.5,bro=True):
        e=self.ev
        good=np.isin(self.names,names)&(e['score']>=threshold)
        if bro:good&=e['broadcast']
        return np.flatnonzero(good)
