"""Event-time owner grammar: optional touches, persistent flights, and stoppages.

An event changes owner only if its evidence overcomes the self-transition prior.
Each event's team observation is scored once, independently of its frame duration.
The interval after it inherits that owner.
"""
from pathlib import Path
import json

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
LOCAL = HERE/'.local'
N = 199319
HALVES = [(11185,90652), (110808,190198)]


@numba.njit(cache=True)
def solve(em, tr, want_marginal=False):
    n, m = em.shape
    v = np.zeros(m)
    bp = np.zeros((n,m), np.int8)
    for t in range(n):
        nv = np.empty(m)
        for j in range(m):
            z = v + tr[t,:,j]
            b = np.argmax(z)
            bp[t,j] = b
            nv[j] = z[b]+em[t,j]
        v = nv-np.max(nv)
    out = np.empty(n, np.int8)
    st = np.argmax(v)
    for t in range(n-1,-1,-1):
        out[t] = st
        st = bp[t,st]
    probs = np.zeros((n,m))
    if want_marginal:
        f = np.zeros((n,m))
        b = np.zeros((n,m))
        f[0] = em[0]-np.max(em[0])
        for t in range(1,n):
            for j in range(m):
                z = f[t-1]+tr[t,:,j]
                mx = np.max(z)
                f[t,j] = mx+np.log(np.exp(z-mx).sum())+em[t,j]
            f[t] -= np.max(f[t])
        for t in range(n-2,-1,-1):
            for j in range(m):
                z = b[t+1]+tr[t+1,j,:]+em[t+1]
                mx = np.max(z)
                b[t,j] = mx+np.log(np.exp(z-mx).sum())
            b[t] -= np.max(b[t])
        for t in range(n):
            z = f[t]+b[t]
            z = np.exp(z-np.max(z))
            probs[t] = z/z.sum()
    return out, probs


class Grammar:
    def __init__(self, fold):
        self.z = dict(np.load(LOCAL/f'fold{fold+1}.npz'))
        self.ev = dict(np.load(LOCAL/'touches.npz'))
        prior = LOCAL/'em_prior.json'
        self.prior = json.loads(prior.read_text()) if prior.exists() else None
        self.nodes = []
        for a,b in HALVES:
            te = self.ev['t']
            te = te[(te >= a) & (te < b)]
            # Preserve every inherited status edge; add a 10-frame status grid.
            # Team transitions remain restricted to touches.
            status = np.flatnonzero(np.diff(self.z['base'][a:b] == 2))+a+1
            self.nodes.append(np.unique(np.r_[a,te,status,np.arange(a,b,10),b]))

    def potentials(self, nodes, p, joint):
        t = nodes[:-1]
        ne = len(t)
        inds = np.searchsorted(self.ev['t'],t)
        match = (inds < len(self.ev['t'])) & (self.ev['t'][np.minimum(inds,len(self.ev['t'])-1)] == t)
        def event(name):
            return np.where(match,self.ev[name][np.minimum(inds,len(self.ev['t'])-1)],0.)
        # Sparse soft observation: contact's reliability saturates with episode length.
        contact_weight = 2. if self.prior is None else np.log(self.prior['accuracy'][0]/(1-self.prior['accuracy'][0]))
        touch = (1-p['local'])*contact_weight*event('contact') + p['local']*event('local')
        if p.get('corroborated',False):
            co = event('contact')
            # Opposing contact contributes only where the independent ball-local
            # view supports it. An uncertain isolated body proximity is skipped.
            ctl = np.sign(co)*np.maximum(0.,np.sign(co)*event('local')-1.)*np.minimum(np.abs(co),1.)
            touch = (1-p['local'])*contact_weight*co + p['local']*ctl
        touch += event('spotter')
        touch *= p['event']
        c = np.r_[0,np.cumsum(np.nan_to_num(self.z['s']))]
        shape = p['shape']*(c[nodes[1:]]-c[t])/25.
        if p.get('sparse',False):
            shape = p['shape']*np.where(match,self.z['shape'][t],0.)
            shape[0] = p['shape']*self.z['shape'][t[0]]
        team = shape+touch
        em = np.zeros((ne,3))
        em[:,0] = .5*team
        em[:,1] = -.5*team
        # Status score is on the original time scale; uncertain team cancels equally.
        c = np.r_[0,np.cumsum(np.nan_to_num(self.z['dead']))]
        em[:,2] = c[nodes[1:]]-c[t]+.5*np.abs(team)
        em[:,2] -= p.get('live',0.)*event('live')
        tr = np.zeros((ne,3,3))
        cost = p['switch']*(self.prior['switch_cost']/4 if self.prior is not None else 1.)
        tr[:,0,1] = tr[:,1,0] = np.where(match,-cost,-1e9)
        tr[:,:2,2] = -self.z['cin'][t,None]
        tr[:,2,:2] = -self.z['cout'][t,None]
        if not joint:
            dead = self.z['base'][t] == 2
            em[dead,:2] = -1e9
            em[~dead,2] = -1e9
            # Restart has no previous owner.
            tr[:,2,:2] = 0.
            tr[:,:2,2] = 0.
        return em,tr

    def run(self,p,joint=False,marginals=False):
        out = np.full(N,2,np.int8)
        marg = np.zeros((N,3))
        marg[:,2] = 1.
        for nodes in self.nodes:
            em,tr = self.potentials(nodes,p,joint)
            y,m = solve(em,tr,marginals)
            dt = np.diff(nodes)
            a,b = nodes[0],nodes[-1]
            out[a:b] = np.repeat(y,dt)
            if marginals:
                marg[a:b] = np.repeat(m,dt,axis=0)
        return (out,marg) if marginals else out
