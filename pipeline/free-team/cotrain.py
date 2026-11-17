"""Co-training between disjoint views, no labels.

    python cotrain.py local  [TEACHER OUT]   # ball-local view taught by the shape view (q_team1 -> q_local)
    python cotrain.py shape  [TEACHER OUT]   # shape view re-taught by the local view   (q_local -> q_shape2)

The shape view (free-owner's team-shape model, `q_team1.npy`) never sees who is at the ball; the
local view (`local_features.py`) sees only who is at the ball and how it moves. Each view learns
from the other's confident live frames, so neither can learn its own seed rule. Live = free-owner's
EM status P(dead) < .3. Team-equivariant LightGBM (each frame also shown with the teams swapped),
every block predicted by a model fitted on the other three.
"""
import os
import sys

import lightgbm as lgb
import numpy as np

from base import BLOCKS, LOCAL, N


def equivariant_oob(fr, Xs, X0, X1, train, y, w=None, rounds=300, leaves=31, step=2):
    """P(team 1 = away) per frame, out of block; y = 1 for away."""
    M = np.c_[X0, X1, Xs]
    Ms = np.c_[X1, X0, Xs]
    blk = np.searchsorted([b for a, b in BLOCKS], fr, side="right")
    q = np.full(len(fr), np.nan)
    for k in range(4):
        tr = np.flatnonzero(train & (blk != k))[::step]
        Xa = np.r_[M[tr], Ms[tr]]
        ya = np.r_[y[tr], 1 - y[tr]]
        wa = None if w is None else np.r_[w[tr], w[tr]]
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.05, num_leaves=leaves, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=6, verbose=-1)
        m.fit(Xa, ya, sample_weight=wa)
        te = blk == k
        q[te] = (m.predict_proba(M[te])[:, 1] + 1 - m.predict_proba(Ms[te])[:, 1]) / 2
        print(f"  block {k + 1}: {len(tr)} frames x2", flush=True)
    return q


def teacher(name, fr, conf=0.35, dead=0.3):
    q = np.load(os.path.join(LOCAL, name))[fr]                  # P(home)
    pdead = np.nan_to_num(np.load(os.path.join(LOCAL, "p_dead.npy"))[fr], nan=1)
    train = (pdead < dead) & (np.abs(q - 0.5) > conf)
    return train, (q < 0.5).astype(int)


def main(mode, teacher_file=None, out=None, conf=0.35):
    if mode == "local":
        z = np.load(os.path.join(LOCAL, os.environ.get("LOCAL_FEATURES", "local.npz")))
        fr = z["frames"]
        drop = [p for p in os.environ.get("DROP", "").split(",") if p]     # feature-name prefixes to leave out
        keep = [i for i, n in enumerate(z["names_pair"]) if not any(str(n).startswith(p) for p in drop)]
        X0, X1 = z["X0"][:, keep], z["X1"][:, keep]
        train, y = teacher(teacher_file or "q_team1.npy", fr, conf)
        print(f"teacher: shape view, {train.sum()} confident live frames ({train.mean():.2f})")
        q = equivariant_oob(fr, z["Xs"], X0, X1, train, y)
        out = out or "q_local.npy"
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "free-owner"))
        import team as fo_team
        fr, Xs, X0, X1, ns, npair = fo_team.load()
        fo_team.LOCAL = LOCAL
        shape = [k for k in npair if fo_team.base(k) not in fo_team.PROX]
        train, y = teacher(teacher_file or "q_local.npy", fr, conf)
        print(f"teacher: local view, {train.sum()} confident live frames ({train.mean():.2f})")
        iP = {k: i for i, k in enumerate(npair)}
        St = np.stack([Xs[:, ns.index(k)] for k in fo_team.STATUS], 1)
        q = equivariant_oob(fr, St, X0[:, [iP[k] for k in shape]], X1[:, [iP[k] for k in shape]], train, y, step=3)
        out = out or "q_shape2.npy"
    full = np.full(N, 0.5)
    full[fr] = 1 - q
    np.save(os.path.join(LOCAL, out), full)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
