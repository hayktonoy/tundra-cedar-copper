"""Build physical touch candidates and snapshot permitted fold-specific evidence. No labels.

Only detection geometry from hc-ball is read; no hc possession model/output is used.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logit

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = HERE / '.local'
sys.path.insert(0, str(EXP / 'free-pretrained'))
from fuse import PEngine, N  # noqa: E402


def odds(p):
    return logit(np.clip(np.nan_to_num(p, nan=.5), .01, .99))


def main():
    LOCAL.mkdir(exist_ok=True)
    paths = [EXP/'hc-ball/.local/contact_tactical.npz',
             EXP/'free-team/.local/contact_broadcast.npz',
             EXP/'hc-ball/.local/track_tactical.npz',
             EXP/'free-team/.local/q_local.npy',
             EXP/'free-pretrained/.local/tdeed_events.npz']
    c, cb, track = [dict(np.load(p)) for p in paths[:3]]
    local = odds(np.load(paths[3]))
    # Per-camera evidence requires a detected (not interpolated) ball.
    votes = []
    for z in (c, cb):
        near = []
        for nm in 'ha':
            foot = np.nan_to_num(z['foot_'+nm], nan=9.)
            body = np.nan_to_num(z['dn_'+nm], nan=9.)
            ground = np.nan_to_num(z['dg_'+nm], nan=99.)
            near.append((((foot < .65) & (ground < 2.)) | (body < .15)) & (z['source'] == 1))
        votes.append(near[0].astype(float)-near[1].astype(float))
    contact = votes[0] + .7*votes[1]
    team = np.sign(contact).astype(int)
    rows = []
    idx = np.flatnonzero(team)
    start = prev = idx[0]
    current = team[start]

    def append(a, b, k):
        # A long dribble supplies repeated evidence, at most once per second.
        for lo in range(a, b, 25):
            hi = min(lo+25, b)
            cnt = np.count_nonzero(team[lo:hi] == k)
            if cnt < 2:
                continue
            confidence = np.clip(np.log1p(cnt)/np.log(9), .25, 1.5)
            both = np.mean((votes[0][lo:hi] == k) & (votes[1][lo:hi] == k))
            rows.append((lo, confidence*k*(1+.5*both), 0., float(np.mean(local[lo:hi])), 1, 0.))

    for j in idx[1:]:
        if j-prev > 4 or team[j] != current:
            append(start, prev+1, current)
            start, current = j, team[j]
        prev = j
    append(start, prev+1, current)
    # Brief trajectory changes can be touches even without a long slow-ball episode.
    from scipy.signal import find_peaks
    kink = np.nan_to_num(c['kink'], nan=0.)
    kink[(kink > 45) | (track['source'] != 1)] = 0
    pk, _ = find_peaks(kink, height=6., distance=10)
    for t in pk:
        v = contact[max(0,t-3):t+4]
        k = np.sign(v.sum())
        if k and np.count_nonzero(v) >= 2:
            rows.append((t, .6*k, 0., float(np.mean(local[t-3:t+4])), 2, 0.))
    spot = dict(np.load(paths[4]))
    names = spot['names12']
    for i, t in enumerate(spot['t']):
        if spot['sn'][i] or not spot['broadcast'][i] or spot['score'][i] < .3:
            continue
        if names[spot['cls'][i]] in ('OUT', 'GOAL'):
            continue
        t = int(round(t))
        if 0 <= t < N:
            live = spot['score'][i] if names[spot['cls'][i]] in ('CROSS','SHOT','FREE KICK','THROW IN') else 0.
            rows.append((t, 0., float(spot['score'][i]*odds(spot['p_home'][i])), local[t], 4, live))
    # Merge cross-view/spotter detections within 0.24 s. Keep the earliest contact onset.
    rows.sort()
    groups = []
    for r in rows:
        if groups and r[0]-groups[-1][0][0] <= 6:
            groups[-1].append(r)
        else:
            groups.append([r])
    events = []
    for g in groups:
        a = np.array(g)
        # Max magnitude prevents double counting a kink and a foot in the same view.
        co = a[np.argmax(np.abs(a[:,1])),1]
        sp = a[np.argmax(np.abs(a[:,2])),2]
        events.append((a[0,0], co, sp, a[:,3].mean(), int(np.bitwise_or.reduce(a[:,4].astype(int))),a[:,5].max()))
    ev = np.array(events)
    np.savez_compressed(LOCAL/'touches.npz', t=ev[:,0].astype(int), contact=ev[:,1],
                        spotter=ev[:,2], local=ev[:,3], kind=ev[:,4].astype(int),live=ev[:,5])

if __name__ == '__main__':
    main()
