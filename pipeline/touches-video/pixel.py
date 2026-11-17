"""Detected players: precise foot points and jersey-colour teams, then ray contact with the ball.

    python pixel.py              # reads .local/players/*.npz, writes .local/pixel.npz and .local/boxteam.npz
    python pixel.py broadcast    # the broadcast boxes, re-indexed to tactical frames -> *_broadcast.npz

1. Foot point of every detected box (bottom centre) -> pitch metres through the frame's
   homography. Much closer to the player than the provided tracking (often ~1 m off).
2. Team from the torso colour: boxes whose foot point is within 1 m of exactly one tracked
   player (next one > 2 m away) take that player's team; a LightGBM on the 4x3 torso patch
   learns colour -> Barca / Athletic / referee (the detector's referee class), cross-fitted by
   half so no box is classified by a model that saw it. The team of the tracked player a box
   sits on (within 1.5 m) is then a prior (.85) on top of the colour. Keepers are a fourth colour
   class (boxes on the tracked keeper slots): both keepers wear similar kits, so a keeper-looking
   box belongs to the team defending the goal on its side of the pitch.
3. The two-camera ray contact of features.py, with detected players in place of tracked ones.

Per-frame arrays (length 199,319): px_d_h / px_d_a (contact distance to the nearest Barca /
Athletic box, metres), px_h_h / px_h_a (implied height there), px_d2_*, px_p_* (colour
confidence of that box), px_dg_* (plain ground distance from the ball's ground point),
px_nbox (boxes in the frame).
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np

from common import (HALVES, LOCAL, N, PITCH, ROOT, VIDEO_LANE, defends_left_goal_is_barca, homographies,
                    keeper_slots, load_ball, load_players)
from features import ray_contact


def to_pitch(H, px):
    Hi = np.linalg.inv(H)
    q = np.einsum("nij,nkj->nki", Hi, np.concatenate([px, np.ones(px.shape[:-1] + (1,))], -1))
    return q[..., :2] / q[..., 2:3] * np.array(PITCH)


def load_boxes(source="tactical"):
    """Boxes per tactical frame, with that frame's homography (broadcast: through the sync)."""
    d, pre = ("players", "tac") if source == "tactical" else ("players_broadcast", "bro")
    parts = sorted(glob.glob(os.path.join(LOCAL, d, f"{pre}_*.npz")), key=lambda p: int(p.split("_")[-2]))
    fr, bx, to = [], [], []
    for p in parts:
        z = np.load(p)
        fr.append(z["frames"]); bx.append(z["boxes"]); to.append(z["torso"])
    fr, bx, to = np.concatenate(fr), np.concatenate(bx), np.concatenate(to)
    if source == "tactical":
        return fr, bx, to, homographies()[fr]
    sync = json.load(open(os.path.join(ROOT, "calibration", "broadcast_sync.json")))
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync["halves"]:
        m = (T >= h["from"]) & (T < h["to"])
        B[m] = np.round((T[m] - h["a"]) / (1 + h["b"])).astype(int)
    row = np.full(B.max() + 2, -1)
    row[fr] = np.arange(len(fr))
    r = np.where(B >= 0, row[np.clip(B, 0, len(row) - 1)], -1)
    Hb = np.load(os.path.join(VIDEO_LANE, "homography_broadcast.npy"))
    keep = np.flatnonzero(r >= 0)
    keep = keep[np.isfinite(Hb[np.clip(fr[r[keep]], 0, len(Hb) - 1), 0, 0])]
    rb = r[keep]
    return keep, bx[rb], to[rb], Hb[fr[rb]]


def colour_features(torso):
    t = torso.reshape(*torso.shape[:-3], 12, 3).astype(np.float32)
    mx, mn = t.max(-1), t.min(-1)
    sat = (mx - mn) / (mx + 1)
    feats = [t.reshape(*t.shape[:-2], 36),
             t.mean(-2), np.median(t, -2),
             (mn > 150).mean(-1, keepdims=True),                       # white
             (mx < 70).mean(-1, keepdims=True),                        # dark
             ((t[..., 0] > t[..., 2] + 25) & (sat > .25)).mean(-1, keepdims=True),   # red
             ((t[..., 2] > t[..., 0] + 15) & (sat > .2)).mean(-1, keepdims=True),    # blue
             ((t[..., 1] > t[..., 0] + 10) & (t[..., 1] > t[..., 2] + 10)).mean(-1, keepdims=True),  # grass
             sat.mean(-1, keepdims=True)]
    return np.concatenate(feats, -1)


def main(source="tactical"):
    fr, bx, torso, Hfr = load_boxes(source)
    suffix = "" if source == "tactical" else "_broadcast"
    print(source, "frames with boxes", len(fr))
    valid = np.isfinite(bx[..., 0])
    fp = np.full(valid.shape + (2,), np.nan)
    target = np.full(valid.shape, -1, np.int8)
    trk_team = np.full(valid.shape, -1, np.int8)      # team of the tracked player within 1.5 m
    on_keeper = np.zeros(valid.shape, bool)          # ... and whether that player is a keeper
    ks = np.array(keeper_slots())
    pl, teams = load_players()
    step = 10000
    for s in range(0, len(fr), step):
        sl = slice(s, s + step)
        b_ = bx[sl]
        foot = np.stack([b_[..., 0], b_[..., 1] + b_[..., 3] / 2], -1).astype(np.float64)
        p = to_pitch(Hfr[sl], np.nan_to_num(foot))
        p[~valid[sl]] = np.nan
        fp[sl] = p
        P = pl[fr[sl]].astype(np.float64)
        d = np.hypot(p[:, :, None, 0] - P[:, None, :, 0], p[:, :, None, 1] - P[:, None, :, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        o = np.argsort(d, -1)[..., :2]
        d1 = np.take_along_axis(d, o[..., :1], -1)[..., 0]
        d2 = np.take_along_axis(d, o[..., 1:2], -1)[..., 0]
        good = (d1 < 1.0) & (d2 > 2.0) & valid[sl] & (b_[..., 5] == 0)
        t_ = np.full(good.shape, -1, np.int8)
        t_[good] = teams[o[..., 0]][good]
        t_[good & np.isin(o[..., 0], ks)] = 3               # a keeper: team from the goal it defends
        t_[valid[sl] & (b_[..., 5] == 1) & (b_[..., 4] > .5)] = 2
        target[sl] = t_
        tt = np.where(d1 < 1.5, teams[o[..., 0]], -1)
        tt[~valid[sl]] = -1
        trk_team[sl] = tt
        on_keeper[sl] = (tt >= 0) & np.isin(o[..., 0], ks)
    # flat list of valid boxes
    vi = np.flatnonzero(valid.ravel())
    X = np.concatenate([colour_features(torso.reshape(-1, 4, 3, 3)[vi[s:s + 500000]]) for s in range(0, len(vi), 500000)])
    y = target.ravel()[vi]
    bfr = np.repeat(fr, valid.shape[1])[vi]
    half = (bfr >= HALVES[1][0]).astype(int)
    fold = half if len(np.unique(half)) > 1 else (bfr % 2)
    pv = np.zeros((len(vi), 4), np.float32)
    params = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=31, min_data_in_leaf=50,
                  num_threads=4, verbose=-1, seed=1)
    rng = np.random.default_rng(0)
    for h in (0, 1):
        idx = np.flatnonzero((fold != h) & (y >= 0))
        idx = rng.choice(idx, min(len(idx), 400000), replace=False)
        m = lgb.train(params, lgb.Dataset(X[idx], y[idx]), 150)
        te = np.flatnonzero(fold == h)
        pv[te] = m.predict(X[te])
    ok = y >= 0
    for k, name in enumerate(("Barca", "Athletic", "official", "keeper")):
        mk = ok & (y == k)
        print(f"{name}: n {mk.sum()}, colour accuracy {(pv[mk].argmax(1) == k).mean():.4f}")
    # keepers (both wear green-ish kits) belong to the team defending the goal on their side
    fx = fp.reshape(-1, 2)[vi, 0]
    side = np.where(defends_left_goal_is_barca(bfr) == (fx < PITCH[0] / 2), 0, 1)
    pv = np.c_[pv[:, 0] + pv[:, 3] * (side == 0), pv[:, 1] + pv[:, 3] * (side == 1), pv[:, 2]].astype(np.float32)
    # jersey colour flickers on single boxes (arms, shadows; both keepers wear similar kits): the
    # tracked player each box sits on is a prior, which a confident colour still overrides
    tk = trk_team.ravel()[vi]
    for k in (0, 1):
        m = tk == k
        pv[m, k] *= 0.85
        pv[m, 1 - k] *= 0.15
    pv[tk >= 0, 2] *= 0.15
    pv /= pv.sum(1, keepdims=True)
    # a box on a tracked keeper belongs to the team defending the goal on its side
    kp = on_keeper.ravel()[vi]
    pv[kp] = np.where(side[kp, None] == np.arange(3)[None], 0.95, 0.025)
    prob = np.full(valid.shape + (3,), np.nan, np.float32)
    prob.reshape(-1, 3)[vi] = pv
    del X
    np.savez_compressed(os.path.join(LOCAL, f"boxteam{suffix}.npz"), frames=fr, foot=fp.astype(np.float32), prob=prob,
                        target=target)
    # contact with the ball, players = detected boxes classified as a team
    b = load_ball()
    gT = np.c_[b["x"], b["y"]].astype(np.float64)[fr]
    gT[~np.isin(b["src"][fr], (1, 2))] = np.nan
    gB = np.c_[b["bx"], b["by"]].astype(np.float64)[fr]
    gB[~np.isin(b["bsrc"][fr], (1, 2))] = np.nan
    g3 = np.where(np.isfinite(b["x3"][fr])[:, None], np.c_[b["x3"], b["y3"]][fr], gT)
    out = {k: np.full(N, np.nan, np.float32) for k in
           ("px_d_h", "px_d_a", "px_h_h", "px_h_a", "px_d2_h", "px_d2_a", "px_p_h", "px_p_a",
            "px_dg_h", "px_dg_a", "px_dt_h", "px_dt_a")}
    out["px_nbox"] = np.zeros(N, np.float32)
    out["px_nbox"][fr] = valid.sum(1)
    step = 20000
    for s in range(0, len(fr), step):
        sl = slice(s, s + step)
        p = fp[sl]
        dc, hc, _ = ray_contact(p, gT[sl], gB[sl])
        dt, _, _ = ray_contact(p, gT[sl], np.full_like(gB[sl], np.nan))
        dg = np.hypot(p[..., 0] - g3[sl][:, None, 0], p[..., 1] - g3[sl][:, None, 1])
        pr = prob[sl]
        for k, name in ((0, "h"), (1, "a")):
            mine = pr[..., k] > 0.5
            dd = np.where(mine & np.isfinite(dc), dc, np.inf)
            order = np.argsort(dd, 1)
            j1 = order[:, 0]
            r = np.arange(len(j1))
            f_ = fr[sl]
            d1 = dd[r, j1]
            out[f"px_d_{name}"][f_] = np.where(np.isinf(d1), np.nan, d1)
            d2 = dd[r, order[:, 1]]
            out[f"px_d2_{name}"][f_] = np.where(np.isinf(d2), np.nan, d2)
            out[f"px_h_{name}"][f_] = hc[r, j1]
            out[f"px_p_{name}"][f_] = pr[r, j1, k]
            g_ = np.where(mine & np.isfinite(dg), dg, np.inf).min(1)
            out[f"px_dg_{name}"][f_] = np.where(np.isinf(g_), np.nan, g_)
            t_ = np.where(mine & np.isfinite(dt), dt, np.inf).min(1)
            out[f"px_dt_{name}"][f_] = np.where(np.isinf(t_), np.nan, t_)
    if suffix:
        out = {k.replace("px_", "pb_"): v for k, v in out.items()}
    np.savez_compressed(os.path.join(LOCAL, f"pixel{suffix}.npz"), **out)
    print("wrote pixel.npz")


if __name__ == "__main__":
    main(*sys.argv[1:2])
