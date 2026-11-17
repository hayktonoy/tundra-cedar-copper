"""Agreed touches.

A touch by team k at a candidate has confidence P(touch) x P(k | touch):
- P(touch) is touches-video's physical formula (a player close to the ball's kink point, a real
  velocity change or an arrival, a ball both views agree on). No learning.
- P(k | touch) is the nearer team at the kink point: sigmoid((d_other - d_k) / NEAR). No learning.

Inside each possession of the base, a confident touch by the other team starts a brief spell of
that team (touches-video's rule: from LAG frames after the touch to END_LAG frames after the next
touch by the possession's own team, or HOLD frames if none comes). Each such spell is judged on
its own: it is kept only if free-carrier's player-level model favours its team over it (mean
probability >= AGREE), and only a kept spell blocks later candidates. A kept spell that was cut
short only because the base itself hands over to the same team within MARGIN frames runs on into
that hand-over (MERGE): the touch shows the change happened earlier than the base has it. The
carrier model judges the spell as proposed; the extension to the hand-over is not judged again.
With TIDY (off by default, on in last-touch), the stage leaves no run shorter than MIN_GAP: a
spell clipped below MIN_GAP by the possession's end is not inserted, and an own-team sliver
shorter than MIN_GAP before a kept spell is given to that spell.
Dead-ball frames are never written, and set pieces are skipped by the physical cue only (the ball
rested, then was struck).

Used by last-touch/touches.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TV = ROOT / 'pipeline' / 'touches-video'
LOCAL = HERE / '.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0, str(TV))
sys.path.insert(0, str(ROOT / 'official' / 'evaluation'))
import possession_metrics as pm  # noqa: E402
import aggressive as AG  # noqa: E402

H, A = 0, 1
P = dict(level=.5,      # confidence a candidate needs
         agree=.5,      # the carrier model must favour the inserted team
         near=.3,       # metres: softness of "the nearer team touched it"
         lag=14, end_lag=14, hold=25, max_hold=75, margin=25, min_gap=15,   # touches-video's defaults
         merge=True)


def best(df, *names):
    out = np.full(len(df), np.nan)
    for n in names:
        if n in df:
            out = np.where(np.isfinite(out), out, df[n].to_numpy(np.float64))
    return out


class Inputs:
    def __init__(self, carrier):
        df = pd.read_parquet(TV / '.local' / 'events.parquet').sort_values('frame')
        self.fr = df['frame'].to_numpy()
        self.pt = np.asarray(AG.touch_probability(df), float)
        self.dh = np.nan_to_num(np.fmin(best(df, 'kpx_d_h', 'kp_d_h'), best(df, 'px_dmin_h', 'c_dmin_h') + .2), nan=4.)
        self.da = np.nan_to_num(np.fmin(best(df, 'kpx_d_a', 'kp_d_a'), best(df, 'px_dmin_a', 'c_dmin_a') + .2), nan=4.)
        self.setpiece = (df['rest_before'].fillna(0) > .8).to_numpy()
        self.carrier = np.load(carrier)['p_home'].astype(float)


def agreement(x, a, b, k):
    """Mean carrier probability of team k over [a, b); None where the carrier says nothing."""
    seg = x.carrier[a:b]
    if not np.isfinite(seg).any():
        return None
    p = float(np.nanmean(seg))
    return p if k == H else 1 - p


def stage(base, x, p=P):
    q_home = 1 / (1 + np.exp(-(x.da - x.dh) / p['near']))
    ph, pa = x.pt * q_home, x.pt * (1 - q_home)
    fr = x.fr
    out, kept, proposed = base.copy(), [], 0
    n = len(base)
    for s, e, T in AG.spells(base):
        p_opp, p_own = (pa, ph) if T == H else (ph, pa)
        m_s = p['margin'] if s > 0 and base[s - 1] == 1 - T else 0
        m_e = p['margin'] if e < n and base[e] == 1 - T else 0
        cand = np.flatnonzero((fr + p['lag'] >= s + m_s) & (fr + p['lag'] < e - m_e) & (p_opp >= p['level']) & ~x.setpiece)
        busy = -1
        tidy = p.get('tidy', False)
        last = s                                         # where the possession's own team was last left in place
        for i in cand:
            t = fr[i]
            if t + p['lag'] < busy:
                continue
            nxt = np.flatnonzero((fr >= t + p['min_gap']) & (fr <= t + p['max_hold']) & (p_own >= .5))
            end = fr[nxt[0]] + p['end_lag'] if len(nxt) else t + p['lag'] + p['hold']
            a, b = t + p['lag'], min(end, e - m_e)
            cut = m_e > 0 and end >= e - m_e            # stopped only by the base's own hand-over
            if m_e == 0 and e - b < p['margin']:
                b = e
            # The returning touch comes at least min_gap after the touch, so a shorter spell exists
            # only because the possession's end clipped it; with tidy such a remnant is not inserted.
            if b - a < (p['min_gap'] if tidy and not (p['merge'] and cut) else 3):
                continue
            proposed += 1
            agree = agreement(x, a, b, 1 - T)            # judged on the spell as proposed, before any extension
            if agree is None or agree < p['agree']:
                continue                                 # rejected: it does not block later candidates
            if p['merge'] and cut:
                b = e                                    # the base hands over to this team anyway
            if tidy and a - last < p['min_gap'] and (last > s or m_s == 0):
                a = last                                 # no own-team sliver shorter than min_gap is left behind
            out[a:b] = 1 - T
            busy = last = b
            kept.append(dict(touch=int(t), start=int(a), end=int(b), team='HA'[1 - T], carrier=round(agree, 3),
                             merged=bool(p['merge'] and cut)))
    return out, kept, proposed
