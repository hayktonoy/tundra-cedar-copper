"""Baum-Welch owner grammar on unlabelled match touch observations, no label access.

Estimate source accuracies and touch-to-touch persistence jointly. A source is a
fractional Bernoulli vote; all teams are treated symmetrically. Raw DFL P(dead)
excludes likely stoppages. Gaps >10 s start a new independent event chain.
"""
import json

import numba
import numpy as np

from grammar import LOCAL, HERE


@numba.njit(cache=True)
def expectation(obs, seen, reliability, rate, reset):
    n = len(obs)
    em = np.zeros((n,2))
    for c in range(obs.shape[1]):
        w = np.log(reliability[c]/(1-reliability[c]))
        em[:,0] += .5*w*obs[:,c]
        em[:,1] -= .5*w*obs[:,c]
    f = np.empty((n,2))
    b = np.zeros((n,2))
    stay,switch = np.log(1-rate),np.log(rate)
    for t in range(n):
        for j in range(2):
            if t == 0 or reset[t]:
                f[t,j] = em[t,j]
            else:
                aa,bb = f[t-1,j]+stay,f[t-1,1-j]+switch
                mx = max(aa,bb)
                f[t,j] = mx+np.log(np.exp(aa-mx)+np.exp(bb-mx))+em[t,j]
        f[t] -= np.max(f[t])
    for t in range(n-2,-1,-1):
        if reset[t+1]:
            continue
        for j in range(2):
            aa = b[t+1,j]+em[t+1,j]+stay
            bb = b[t+1,1-j]+em[t+1,1-j]+switch
            mx = max(aa,bb)
            b[t,j] = mx+np.log(np.exp(aa-mx)+np.exp(bb-mx))
        b[t] -= np.max(b[t])
    g = np.empty(n)
    transitions = 0.
    pairs = 0
    for t in range(n):
        z = f[t]+b[t]
        g[t] = 1/(1+np.exp(z[1]-z[0]))
        if t and not reset[t]:
            z = np.empty((2,2))
            for i in range(2):
                for j in range(2):
                    z[i,j] = f[t-1,i]+(stay if i == j else switch)+em[t,j]+b[t,j]
            z = np.exp(z-z.max()); z /= z.sum()
            transitions += z[0,1]+z[1,0]
            pairs += 1
    r = np.empty(obs.shape[1])
    for c in range(obs.shape[1]):
        # 10 pseudo observations give a mild prior of .8, independent of labels.
        r[c] = (8+np.sum(.5*(1+(2*g-1)*obs[:,c])*seen[:,c]))/(10+seen[:,c].sum())
    return g, np.clip(r,.55,.995), (transitions+1)/(pairs+20)


def main():
    ev = dict(np.load(LOCAL/'touches.npz'))
    raw = np.load(HERE.parent/'free-dfl/.local/dfl_only_evidence.npz')
    keep = raw['p_dead'][ev['t']] < .3
    t = ev['t'][keep]
    shape = np.load(HERE.parent/'free-owner/.local/q_team1.npy')[t]
    obs = np.c_[np.clip(ev['contact'][keep],-1,1),np.tanh(ev['spotter'][keep]/2),
                np.tanh(ev['local'][keep]/2),2*np.nan_to_num(shape,nan=.5)-1]
    seen = (np.abs(obs) > .01).astype(float)
    # Soft signs: attenuation describes uncertainty, never a target/sample weight from labels.
    reset = np.r_[True,np.diff(t)>250]
    r = np.array([.85,.9,.8,.85])
    rate = .07
    trace = []
    for it in range(30):
        g,nr,nrate = expectation(obs,seen,r,rate,reset)
        trace.append(dict(iteration=it,accuracy=nr.tolist(),switch_probability=float(nrate)))
        if np.max(np.abs(r-nr)) < 1e-5 and abs(rate-nrate) < 1e-5:
            break
        r,rate = nr,nrate
    prior = dict(source_names=['contact','spotter','local','shape'],accuracy=r.tolist(),
                 switch_probability=float(rate),switch_cost=float(np.log((1-rate)/rate)),
                 n_events=int(keep.sum()),n_sequences=int(reset.sum()),
                 mean_dwell_events=float(1/rate),
                 mean_dwell_seconds=float(np.diff(t)[~reset[1:]].mean()/25/rate),trace=trace)
    (LOCAL/'em_prior.json').write_text(json.dumps(prior,indent=2))
    np.savez_compressed(LOCAL/'em_events.npz',t=t,p_home=g)
    print({k:v for k,v in prior.items() if k != 'trace'},flush=True)


if __name__ == '__main__':
    main()
