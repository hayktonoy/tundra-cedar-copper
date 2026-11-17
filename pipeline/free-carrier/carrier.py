"""Player-level ball-carrier HMM, fitted by Baum-Welch on the unlabelled match. No labels are read.

    python carrier.py [--tag base] [--temp 1] [--gate .1] [--iters 30]
        -> .local/carrier_<tag>.npz, .local/params_<tag>.json

Hidden state per frame: one of the 46 tracking slots carries the ball (only slots tracked at that
frame), or the ball is in flight / loose with the team of its last carrier (F_H, F_A).
Transitions, tied across players:
  carrier stays; releases into a flight of its own team; hands over directly to a teammate or an
  opponent (tackle, one-touch); a flight continues or is received by a teammate of the last
  carrier (pass) or by an opponent (interception). A slot that stops being tracked releases.
Physics:
  every release, reception or hand-over is an event at which the ball's velocity changes: the
  kink of the detected video ball (both cameras) is observed on the transition, with tied
  P(kink | event) and P(kink | no event) tables;
  a player cannot receive a moving ball that is going away from them (gate on receptions).
Emissions, tied naive-Bayes tables over binned observations:
  per slot (carrier vs not): ground distance to the tactical and broadcast video ball, the ball
  pixel's distance to the linked box's foot in both cameras, ball-minus-player velocity;
  per frame (carried vs flight): ball speed in both tracks, ball height.
Opposing receptions are the only way possession changes team, so a teammate pass cannot flip it.
P(H) = sum of Barca carrier posteriors + P(F_H). M-step statistics are weighted by free-owner's
label-free EM P(live), so stoppages do not shape the tables.
"""
import argparse
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = HERE / '.local'
sys.path.insert(0, str(HERE))

N = 199319
HALVES = [(11185, 90652), (110808, 190198)]
S = 46
FH, FA = S, S + 1
K = S + 2

DIST = [0, .75, 1.5, 2.5, 4, 7, 1e9]          # m
FOOT = [0, .15, .3, .5, .8, 1.5, 1e9]         # box heights
RVEL = [0, 1.5, 3, 5, 8, 1e9]                 # m/s
SPEED = [0, 2, 4, 7, 11, 16, 25, 1e9]         # m/s
HEIGHT = [-1e9, .4, 1.2, 2.5, 1e9]            # m
KINK = [0, 1.5, 3, 5, 8, 12, 1e9]             # m/s velocity change over +-4 frames
SLOT_BINS = [len(DIST), len(DIST), len(FOOT), len(FOOT), len(RVEL)]
# Physical priors (observed bins; the missing bin stays learned). A carried ball is within ~1.5 m
# of its carrier and moves at running speed; an event changes the ball's velocity.
PRIOR_DIST = [.40, .33, .15, .06, .03, .02, .01][:len(DIST) - 1]
PRIOR_SPEED = [.30, .30, .24, .12, .03, .008, .002][:len(SPEED) - 1]
PRIOR_Q1 = [.05, .10, .15, .20, .20, .30]
PRIOR_Q0 = [.45, .25, .14, .08, .05, .03]
FRAME_BINS = [len(SPEED), len(SPEED), len(HEIGHT)]


def binned(x, edges):
    """Bin index, len(edges)-1 for missing."""
    x = np.asarray(x, np.float32)
    b = np.clip(np.searchsorted(np.asarray(edges[1:-1], np.float32), x, side='right'), 0, len(edges) - 2)
    return np.where(np.isfinite(x), b, len(edges) - 1).astype(np.int8)


def kink(x, y, det, k=4):
    """|v_after - v_before| over k frames where the three anchor frames are detections."""
    X = np.where(det, x, np.nan).astype(np.float64)
    Y = np.where(det, y, np.nan).astype(np.float64)
    out = np.full(len(x), np.nan)
    vbx = (X[k:-k] - X[:-2 * k]) / k * 25
    vby = (Y[k:-k] - Y[:-2 * k]) / k * 25
    vax = (X[2 * k:] - X[k:-k]) / k * 25
    vay = (Y[2 * k:] - Y[k:-k]) / k * 25
    out[k:-k] = np.hypot(vax - vbx, vay - vby)
    return out


def features(z):
    """Slot features (N,S,F) int8, frame features (N,G) int8, kink bin (N,), active (N,S)."""
    act = np.isfinite(z['px'])
    slot = np.stack([binned(z['dgt'], DIST), binned(z['dgb'], DIST), binned(z['cft'], FOOT),
                     binned(z['cfb'], FOOT), binned(z['rv'], RVEL)], -1)
    sp = np.hypot(z['vbxt'], z['vbyt'])
    sb = np.hypot(z['vbxb'], z['vbyb'])
    frame = np.stack([binned(sp, SPEED), binned(sb, SPEED), binned(z['height'], HEIGHT)], -1)
    kt = kink(z['bxt'], z['byt'], z['srct'] == 1)
    kb = kink(z['bxb'], z['byb'], z['srcb'] == 1)
    kk = np.where(np.isfinite(kt), kt, kb)
    return slot, frame, binned(kk, KINK), act


def gates(z, act, low):
    """Reception gate per slot: `low` where a moving ball (>4 m/s) is going away from the slot."""
    vx, vy = z['vbxt'], z['vbyt']
    vx = np.where(np.isfinite(vx), vx, z['vbxb'])
    vy = np.where(np.isfinite(vy), vy, z['vbyb'])
    bx = np.where(np.isfinite(z['bxt']), z['bxt'], z['bxb'])
    by = np.where(np.isfinite(z['byt']), z['byt'], z['byb'])
    # velocity before the frame: shift the central difference back by two frames
    vx = np.r_[np.full(2, np.nan), vx[:-2]]
    vy = np.r_[np.full(2, np.nan), vy[:-2]]
    g = np.ones((N, S), np.float32)
    for s in range(0, N, 20000):
        sl = slice(s, s + 20000)
        dx = z['px'][sl] - bx[sl, None]
        dy = z['py'][sl] - by[sl, None]
        d = np.hypot(dx, dy)
        v = np.hypot(vx[sl], vy[sl])
        cos = (dx * vx[sl, None] + dy * vy[sl, None]) / np.maximum(d * v[:, None], 1e-6)
        away = (v[:, None] > 4) & (d > 1.5) & (cos < -.3)
        g[sl][away] = low
    g[~act] = 0
    return g


def init_params():
    def ramp(n, hi_first, k=1.):
        v = np.exp(-k * np.arange(n - 1)) if hi_first else np.exp(k * np.arange(n - 1))
        v = np.r_[v / v.sum(), 1.]
        return v / v.sum()
    pc = [ramp(n, True) for n in SLOT_BINS]
    pn = [ramp(n, False, .3) for n in SLOT_BINS]
    pcar = [ramp(n, True, .5) for n in FRAME_BINS]
    pfl = [ramp(n, False, .2) for n in FRAME_BINS]
    q1 = ramp(len(KINK), False, .4)
    q0 = ramp(len(KINK), True, .6)
    trans = dict(rel=.02, ds=.0005, do=.0003, as_=.004, ao=.002)
    return dict(pc=pc, pn=pn, pcar=pcar, pfl=pfl, q0=q0, q1=q1, trans=trans)


def pin(table, prior):
    """Keep the learned missing-bin share, impose the prior shape on the observed bins."""
    t = np.asarray(table, np.float64).copy()
    pr = np.asarray(prior, np.float64)
    t[:-1] = (1 - t[-1]) * pr / pr.sum()
    return t


def constrain(par, physics):
    if not physics:
        return par
    par = dict(par)
    par['pc'] = [pin(t, PRIOR_DIST) if f < 2 else t for f, t in enumerate(par['pc'])]
    par['pcar'] = [pin(t, PRIOR_SPEED) if f < 2 else t for f, t in enumerate(par['pcar'])]
    miss = .5 * (par['q0'][-1] + par['q1'][-1])
    par['q1'] = pin(np.r_[PRIOR_Q1, miss], PRIOR_Q1)
    par['q0'] = pin(np.r_[PRIOR_Q0, miss], PRIOR_Q0)
    return par


def spotter_team():
    """Team log-odds of the SoccerNet spotter's on-ball actions (free-pretrained's T-DEED events:
    pass, drive, header, high pass, cross, throw-in, shot, block, tackle, free kick; broadcast
    team head, score > .3), spread over +-6 frames around each action. Positive = Barca acts."""
    ev = np.load(EXP / 'free-pretrained/.local/tdeed_events.npz')
    m = np.isin(ev['cls'], [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]) & ~ev['sn'] & (ev['score'] > .3) & ev['broadcast']
    t = np.round(ev['t'][m]).astype(int)
    v = ev['score'][m] * np.log(np.clip(ev['p_home'][m], .05, .95) / np.clip(1 - ev['p_home'][m], .05, .95))
    out = np.zeros(N, np.float32)
    for k in range(-6, 7):
        np.add.at(out, np.clip(t + k, 0, N - 1), v)
    return out


def context_team():
    """free-team's label-free ball-local team view (co-trained on the shape view), as log-odds."""
    q = np.load(EXP / 'free-team/.local/q_local.npy')
    q = np.clip(np.nan_to_num(q, nan=.5), .02, .98)
    return np.log(q / (1 - q)).astype(np.float32)


def log_emissions(slot, frame, act, par, temp=1., team=None, team_term=None):
    """(N,K) log emissions relative to 'nobody carries'."""
    E = np.zeros((N, K), np.float32)
    for f in range(slot.shape[-1]):
        r = (np.log(par['pc'][f]) - np.log(par['pn'][f])).astype(np.float32)
        E[:, :S] += r[slot[..., f]]
    car = np.zeros(N, np.float32)
    fl = np.zeros(N, np.float32)
    for f in range(frame.shape[-1]):
        car += np.log(par['pcar'][f])[frame[:, f]]
        fl += np.log(par['pfl'][f])[frame[:, f]]
    E[:, :S] += car[:, None]
    E[:, FH] = fl
    E[:, FA] = fl
    E *= temp
    if team_term is not None:
        # Barca's carriers and flight get +t/2, Athletic's -t/2 (a team log-odds term per frame)
        h = np.r_[team == 0, True, False]
        E[:, h] += .5 * team_term[:, None]
        E[:, ~h] -= .5 * team_term[:, None]
    E[:, :S][~act] = -np.inf
    return E


@njit(cache=True)
def fb(logE, act, team, gate, q0, q1, a, b, rel, ds, do, as_, ao):
    """Scaled forward-backward on [a,b). Returns gamma, transition counts, P(event) per frame, loglik."""
    n = b - a
    alpha = np.zeros((n, K))
    beta = np.zeros((n, K))
    E = np.zeros((n, K))
    c = np.zeros(n)
    GT = np.zeros((n, 2))
    msum = 0.
    for t in range(n):
        m = -1e300
        for k in range(K):
            if logE[a + t, k] > m:
                m = logE[a + t, k]
        msum += m
        for k in range(K):
            E[t, k] = np.exp(logE[a + t, k] - m) if np.isfinite(logE[a + t, k]) else 0.
        for j in range(S):
            if act[a + t, j]:
                GT[t, team[j]] += gate[a + t, j]
    nst = 0.
    for k in range(S):
        if act[a, k]:
            nst += 1.
    for k in range(K):
        alpha[0, k] = E[0, k] / (nst + 2.)
    s = 0.
    for k in range(K):
        s += alpha[0, k]
    c[0] = s
    for k in range(K):
        alpha[0, k] /= s
    for t in range(1, n):
        Q0, Q1 = q0[a + t], q1[a + t]
        car = np.zeros(2)
        for j in range(S):
            if act[a + t - 1, j]:
                car[team[j]] += alpha[t - 1, j]
        relm = np.zeros(2)
        for j in range(S):
            if act[a + t - 1, j]:
                T = team[j]
                if act[a + t, j]:
                    relm[T] += alpha[t - 1, j] * rel * Q1
                else:
                    relm[T] += alpha[t - 1, j] * (1. - GT[t, T] * ds - GT[t, 1 - T] * do) * Q0
        tot = 0.
        for k in range(S):
            if not act[a + t, k]:
                alpha[t, k] = 0.
                continue
            T = team[k]
            g = gate[a + t, k]
            own = alpha[t - 1, k] if act[a + t - 1, k] else 0.
            stay = 1. - rel - (GT[t, T] - g) * ds - GT[t, 1 - T] * do
            inc = own * stay * Q0 + g * Q1 * ((car[T] - own) * ds + car[1 - T] * do
                                              + alpha[t - 1, S + T] * as_ + alpha[t - 1, S + 1 - T] * ao)
            alpha[t, k] = inc * E[t, k]
            tot += alpha[t, k]
        for T in range(2):
            stayf = 1. - GT[t, T] * as_ - GT[t, 1 - T] * ao
            alpha[t, S + T] = (alpha[t - 1, S + T] * stayf * Q0 + relm[T]) * E[t, S + T]
            tot += alpha[t, S + T]
        c[t] = tot
        for k in range(K):
            alpha[t, k] /= tot
    for k in range(K):
        beta[n - 1, k] = 1.
    cnt = np.zeros(7)   # stay_c, rel, ds, do, as, ao, stay_f
    pev = np.zeros(n)
    for t in range(n - 1, 0, -1):
        Q0, Q1 = q0[a + t], q1[a + t]
        eb = np.zeros(K)
        sT = np.zeros(2)
        for k in range(S):
            if act[a + t, k]:
                eb[k] = E[t, k] * beta[t, k]
                sT[team[k]] += gate[a + t, k] * eb[k]
        eb[FH] = E[t, FH] * beta[t, FH]
        eb[FA] = E[t, FA] * beta[t, FA]
        ev = 0.
        for j in range(S):
            if not act[a + t - 1, j]:
                beta[t - 1, j] = 0.
                continue
            T = team[j]
            if act[a + t, j]:
                g = gate[a + t, j]
                stay = 1. - rel - (GT[t, T] - g) * ds - GT[t, 1 - T] * do
                v_stay = stay * eb[j] * Q0
                v_rel = rel * eb[S + T] * Q1
                v_ds = ds * (sT[T] - g * eb[j]) * Q1
            else:
                v_stay = (1. - GT[t, T] * ds - GT[t, 1 - T] * do) * eb[S + T] * Q0
                v_rel = 0.
                v_ds = ds * sT[T] * Q1
            v_do = do * sT[1 - T] * Q1
            beta[t - 1, j] = (v_stay + v_rel + v_ds + v_do) / c[t]
            w = alpha[t - 1, j] / c[t]
            cnt[0] += w * v_stay
            cnt[1] += w * v_rel
            cnt[2] += w * v_ds
            cnt[3] += w * v_do
            ev += w * (v_rel + v_ds + v_do)
        for T in range(2):
            stayf = 1. - GT[t, T] * as_ - GT[t, 1 - T] * ao
            v_s = stayf * eb[S + T] * Q0
            v_as = as_ * sT[T] * Q1
            v_ao = ao * sT[1 - T] * Q1
            beta[t - 1, S + T] = (v_s + v_as + v_ao) / c[t]
            w = alpha[t - 1, S + T] / c[t]
            cnt[4] += w * v_as
            cnt[5] += w * v_ao
            cnt[6] += w * v_s
            ev += w * (v_as + v_ao)
        pev[t] = ev
    g = alpha * beta
    for t in range(n):
        s = 0.
        for k in range(K):
            s += g[t, k]
        for k in range(K):
            g[t, k] /= s
    ll = msum
    for t in range(n):
        ll += np.log(c[t])
    return g, cnt, pev, ll


@njit(cache=True)
def viterbi(logE, act, team, gate, q0, q1, a, b, rel, ds, do, as_, ao):
    """Most likely carrier path on [a,b): state index per frame."""
    n = b - a
    lrel, lds, ldo, las, lao = np.log(rel), np.log(ds), np.log(do), np.log(as_), np.log(ao)
    V = np.full(K, -np.inf)
    for k in range(K):
        V[k] = logE[a, k]
    back = np.zeros((n, K), np.int16)
    for t in range(1, n):
        L0, L1 = np.log(q0[a + t]), np.log(q1[a + t])
        GT = np.zeros(2)
        for j in range(S):
            if act[a + t, j]:
                GT[team[j]] += gate[a + t, j]
        bestc = np.full(2, -np.inf)
        argc = np.zeros(2, np.int64)
        second = np.full(2, -np.inf)
        args = np.zeros(2, np.int64)
        for j in range(S):
            if act[a + t - 1, j] and np.isfinite(V[j]):
                T = team[j]
                if V[j] > bestc[T]:
                    second[T], args[T] = bestc[T], argc[T]
                    bestc[T], argc[T] = V[j], j
                elif V[j] > second[T]:
                    second[T], args[T] = V[j], j
        W = np.full(K, -np.inf)
        for k in range(S):
            if not act[a + t, k]:
                continue
            T = team[k]
            g = gate[a + t, k]
            lg = np.log(g) if g > 0 else -np.inf
            bv, bi = -np.inf, 0
            if act[a + t - 1, k]:
                stay = 1. - rel - (GT[T] - g) * ds - GT[1 - T] * do
                bv, bi = V[k] + np.log(stay) + L0, k
            if argc[T] != k:
                cand, ci = bestc[T], argc[T]
            else:
                cand, ci = second[T], args[T]
            for val, idx in ((cand + lds, ci), (bestc[1 - T] + ldo, argc[1 - T]),
                             (V[S + T] + las, S + T), (V[S + 1 - T] + lao, S + 1 - T)):
                if val + lg + L1 > bv:
                    bv, bi = val + lg + L1, idx
            W[k] = bv + logE[a + t, k]
            back[t, k] = bi
        for T in range(2):
            stayf = 1. - GT[T] * as_ - GT[1 - T] * ao
            bv, bi = V[S + T] + np.log(stayf) + L0, S + T
            for j in range(S):
                if act[a + t - 1, j] and team[j] == T and np.isfinite(V[j]):
                    if act[a + t, j]:
                        lr = lrel + L1
                    else:
                        lr = np.log(1. - GT[T] * ds - GT[1 - T] * do) + L0
                    if V[j] + lr > bv:
                        bv, bi = V[j] + lr, j
            W[S + T] = bv + logE[a + t, S + T]
            back[t, S + T] = bi
        V = W
    path = np.zeros(n, np.int16)
    best = -np.inf
    for k in range(K):
        if V[k] > best:
            best, path[n - 1] = V[k], k
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class Model:
    def __init__(self, gate_low=.1, wspot=0., wctx=0.):
        z = np.load(LOCAL / 'inputs.npz')
        self.slot, self.frame, self.kink, self.act = features(z)
        self.team = z['team'].astype(np.int64)
        self.gate_low = gate_low
        self.gate = gates(z, self.act, gate_low)
        pd = np.load(EXP / 'free-owner/.local/evidence.npz')['p_dead_em']
        self.w = np.nan_to_num(1 - pd, nan=0.).astype(np.float32)
        self.wspot, self.wctx = wspot, wctx
        self.team_term = None
        if wspot or wctx:
            # spotter actions and, optionally, the ball-local team view, as one team log-odds term
            self.team_term = (wspot * spotter_team() if wspot else 0) + (wctx * context_team() if wctx else 0)

    def run(self, par, temp=1., want_path=False):
        logE = log_emissions(self.slot, self.frame, self.act, par, temp, self.team, self.team_term)
        q0 = np.asarray(par['q0'])[self.kink]
        q1 = np.asarray(par['q1'])[self.kink]
        tr = par['trans']
        G = np.zeros((N, K), np.float32)
        pev = np.zeros(N, np.float32)
        path = np.full(N, -1, np.int16)
        cnt = np.zeros(7)
        ll = 0.
        for f in range(self.slot.shape[-1]):
            lp = (temp * np.log(par['pn'][f])).astype(np.float32)
            for a, b in HALVES:
                ll += float(np.where(self.act[a:b], lp[self.slot[a:b, :, f]], 0).sum())
        args = (tr['rel'], tr['ds'], tr['do'], tr['as_'], tr['ao'])
        for a, b in HALVES:
            g, c, pe, l = fb(logE, self.act, self.team, self.gate, q0, q1, a, b, *args)
            G[a:b] = g
            pev[a:b] = pe
            cnt += c
            ll += l
            if want_path:
                path[a:b] = viterbi(logE, self.act, self.team, self.gate, q0, q1, a, b, *args)
        return G, cnt, pev, ll, path

    def mstep(self, G, cnt, pev):
        w, act, team = self.w, self.act, self.team
        gs = G[:, :S] * w[:, None]
        notc = (act * w[:, None]) - gs
        new = dict(pc=[], pn=[], pcar=[], pfl=[])
        for f, nb in enumerate(SLOT_BINS):
            x = self.slot[..., f].ravel()
            pc = np.bincount(x, gs.ravel(), nb) + 1.
            pn = np.bincount(x, notc.ravel(), nb) + 1.
            new['pc'].append(pc / pc.sum())
            new['pn'].append(pn / pn.sum())
        car = gs.sum(1)
        fl = (G[:, FH] + G[:, FA]) * w
        for f, nb in enumerate(FRAME_BINS):
            x = self.frame[:, f]
            a = np.bincount(x, car, nb) + 1.
            b = np.bincount(x, fl, nb) + 1.
            new['pcar'].append(a / a.sum())
            new['pfl'].append(b / b.sum())
        q1 = np.bincount(self.kink, pev * w, len(KINK)) + 1.
        q0 = np.bincount(self.kink, (1 - pev) * w, len(KINK)) + 1.
        new['q1'], new['q0'] = q1 / q1.sum(), q0 / q0.sum()
        gH = (self.gate * (team[None] == 0)).sum(1).astype(np.float64)
        gA = (self.gate * (team[None] == 1)).sum(1).astype(np.float64)
        gc = G[:, :S].astype(np.float64)
        cH = gc[:, team == 0].sum(1)
        cA = gc[:, team == 1].sum(1)
        tr = dict(rel=cnt[1] / (cH + cA).sum(),
                  ds=cnt[2] / (cH * (gH - 1).clip(0) + cA * (gA - 1).clip(0)).sum(),
                  do=cnt[3] / (cH * gA + cA * gH).sum(),
                  as_=cnt[4] / (G[:, FH] * gH + G[:, FA] * gA).sum(),
                  ao=cnt[5] / (G[:, FH] * gA + G[:, FA] * gH).sum())
        new['trans'] = {k: float(np.clip(v, 1e-6, .2)) for k, v in tr.items()}
        return new


def p_home(G, team):
    return (G[:, :S][:, team == 0].sum(1) + G[:, FH]).astype(np.float32)


def tojson(par):
    out = {}
    for k, v in par.items():
        if k == 'trans':
            out[k] = v
        elif k in ('q0', 'q1'):
            out[k] = list(map(float, v))
        else:
            out[k] = [list(map(float, x)) for x in v]
    return out


def fit(model, temp=1., iters=30, verbose=True, physics=True):
    par = constrain(init_params(), physics)
    prev = -np.inf
    for it in range(iters):
        G, cnt, pev, ll, _ = model.run(par, temp)
        if verbose:
            print(f'iter {it} ll {ll:.1f}', {k: round(v, 5) for k, v in par['trans'].items()}, flush=True)
        if abs(ll - prev) < 1e-6 * abs(ll):
            break
        prev = ll
        par = constrain(model.mstep(G, cnt, pev), physics)
    return par


def save(model, par, temp, tag):
    G, cnt, pev, ll, path = model.run(par, temp, want_path=True)
    np.savez_compressed(LOCAL / f'carrier_{tag}.npz', p_home=p_home(G, model.team),
                        p_flight=(G[:, FH] + G[:, FA]).astype(np.float32),
                        p_max=G[:, :S].max(1).astype(np.float32), p_event=pev, path=path,
                        gamma=G.astype(np.float16))
    (LOCAL / f'params_{tag}.json').write_text(json.dumps(dict(temp=temp, gate=model.gate_low, spot=model.wspot, ctx=model.wctx,
                                                              **tojson(par)), indent=1, default=float))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--temp', type=float, default=1.)
    ap.add_argument('--gate', type=float, default=.1)
    ap.add_argument('--spot', type=float, default=0., help='weight of the spotter team term')
    ap.add_argument('--ctx', type=float, default=0., help='weight of the ball-local team view per frame')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--free', action='store_true', help='learn every table (no physical priors)')
    args = ap.parse_args()
    model = Model(args.gate, args.spot, args.ctx)
    par = fit(model, args.temp, args.iters, physics=not args.free)
    save(model, par, args.temp, args.tag)
    print('saved', args.tag, json.dumps(par['trans']), 'q1/q0', np.round(np.asarray(par['q1']) / par['q0'], 2))


if __name__ == '__main__':
    main()
