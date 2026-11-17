"""Team in possession without labels: self-trained, team-equivariant LightGBM models.

    python team.py seed        # round 1 -> .local/q_team1.npy
    python team.py round2      # round 2 -> .local/q_team2.npy (needs .local/pseudo1.npy)

Round 1 learns from contact pseudo-labels: frames where the video ball (detected, < 10 m/s) is at
one team's feet (< 0.5 box heights from the foot point, 0.15 nearer than the other team), outside
obvious stoppages (top-5 player speed over 3 s >= 1.5 m/s). Its inputs are team shape, motion,
Smart Tagging and where the ball is in each team's attacking frame, never ball-player proximity,
so it predicts the possessing team also where nobody is at the ball.

Round 2 learns from the decoded possession of round 1 (`pseudo1.npy`: a decode with fixed,
untuned constants), on live frames more than 1 s from a decoded change, and may also use
windowed proximity, since its targets are possessions rather than touches.

Both rounds are team-equivariant: each training frame is also shown with the two teams swapped
and the target flipped, and predictions average the two orientations. Each block of the match is
predicted by a model fitted on the other three (frames near a block edge are not special: no
label is involved, this only keeps a frame's own pseudo-label out of its prediction).
"""
import os
import sys

import lightgbm as lgb
import numpy as np

from common import BLOCKS, LOCAL, N

PROX = ("d1_", "d2_", "c5_", "c15_", "foot", "dn", "rel", "nv", "carry")
STATUS = ["ball_speed_w13", "ball_speed_w50", "ball_h_w25", "ball_det_w25", "top5_w75", "n_players"]


def base(name):
    for suf in ("_w", "_past", "_fut"):
        if suf in name:
            name = name[:name.index(suf)]
    return name


def load():
    z = np.load(os.path.join(LOCAL, "features.npz"))
    return z["frames"], z["Xs"], z["X0"], z["X1"], list(z["names_s"]), list(z["names_pair"])


def seeds(Xs, X0, X1, ns, npair):
    S = {k: Xs[:, i] for i, k in enumerate(ns)}
    iP = {k: i for i, k in enumerate(npair)}
    f0 = np.nan_to_num(X0[:, iP["foot"]], nan=9)
    f1 = np.nan_to_num(X1[:, iP["foot"]], nan=9)
    ok = (S["ball_det"] > 0) & (np.nan_to_num(S["ball_speed"], nan=99) < 10) & (S["top5_w75"] > 1.5)
    t0 = ok & (f0 < 0.5) & (f0 + 0.15 < f1)
    t1 = ok & (f1 < 0.5) & (f1 + 0.15 < f0)
    return t0 | t1, t1.astype(int)


def fit_predict(fr, Xs, X0, X1, ns, npair, cols, train, y, rounds=300):
    """P(team 1 = away) for every frame, out of block, team-equivariant."""
    iP = {k: i for i, k in enumerate(npair)}
    St = np.stack([Xs[:, ns.index(k)] for k in STATUS], 1)
    A = np.stack([X0[:, iP[k]] for k in cols], 1)
    B = np.stack([X1[:, iP[k]] for k in cols], 1)
    M = np.c_[A, B, St]
    Ms = np.c_[B, A, St]
    blk = np.searchsorted([b for a, b in BLOCKS], fr, side="right")
    q = np.full(len(fr), np.nan)
    for k in range(4):
        tr = np.flatnonzero(train & (blk != k))[::3]
        Xa = np.r_[M[tr], Ms[tr]]
        ya = np.r_[y[tr], 1 - y[tr]]
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=6, verbose=-1)
        m.fit(Xa, ya)
        te = blk == k
        q[te] = (m.predict_proba(M[te])[:, 1] + 1 - m.predict_proba(Ms[te])[:, 1]) / 2
        print(f"  block {k}: {len(tr)} frames x2", flush=True)
    return q


def main(mode):
    fr, Xs, X0, X1, ns, npair = load()
    shape = [k for k in npair if base(k) not in PROX]
    if mode == "seed":
        train, y = seeds(Xs, X0, X1, ns, npair)
        q = fit_predict(fr, Xs, X0, X1, ns, npair, shape, train, y)
        out = "q_team1.npy"
    else:
        pseudo = np.load(os.path.join(LOCAL, "pseudo1.npy"))[fr]      # 0 H, 1 A, 2 D
        live = pseudo < 2
        ch = np.flatnonzero(np.diff(pseudo) != 0) + 1
        near = np.zeros(len(fr), bool)
        for c in ch:
            near[max(0, c - 25):c + 25] = True
        cols = npair if mode == "round2" else shape
        q = fit_predict(fr, Xs, X0, X1, ns, npair, cols, live & ~near, (pseudo == 1).astype(int))
        out = "q_team2.npy" if mode == "round2" else "q_team2s.npy"
    full = np.full(N, 0.5)
    full[fr] = 1 - q                                                     # stored as P(home)
    np.save(os.path.join(LOCAL, out), full)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed")
