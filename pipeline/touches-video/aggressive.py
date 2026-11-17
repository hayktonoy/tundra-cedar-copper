"""Touch proposals from the video ball and the players.

The build uses three functions from this file: `touch_probability` (a physical touch score: a
player of either team close to the kink point or the observed ball, a clear kink or the ball
arriving, and a ball that is the ball), `ball_quality` and `spells`. The touch stage itself is
pipeline/agreed-touches/free.py, with the settings of pipeline/last-touch/settings.json; the
command-line script below and its defaults are an earlier version and are not run.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import BASE, LOCAL, N, codes, write


LEVELS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.8)


def sig(x):
    return 1 / (1 + np.exp(-x))


def touch_probability(df):
    """Physical P(touch): someone close enough to the ball at the touch, and a real velocity change
    (or the ball arriving at a player). Contact is the better of the kink point's and the observed
    ball's within +-3 frames (the kink point is wrong when the hinge spans a detector gap)."""
    def col(*names):
        out = np.full(len(df), np.inf)
        for n in names:
            if n in df:
                out = np.fmin(out, df[n].to_numpy(np.float64))
        return np.where(np.isnan(out), np.inf, out)
    kp = col("kpx_d_h", "kpx_d_a")
    kp = np.where(np.isfinite(kp), kp, col("kp_d_h", "kp_d_a"))
    obs = col("px_dmin_h", "px_dmin_a")
    obs = np.where(np.isfinite(obs), obs, col("c_dmin_h", "c_dmin_a"))
    d = np.fmin(kp, obs + 0.2)
    K = np.fmax(df["kmax_t"].fillna(0), df["kmax_b"].fillna(0)).to_numpy()
    K = np.where(K > 0, K, np.fmax(df["kmax_tw"].fillna(0), df["kmax_bw"].fillna(0)).to_numpy())
    onset = (K < 1.5) & (obs < 0.6)            # a reception without a sharp kink
    kq = np.where(onset, 0.7, sig((K - 2.5) / 0.8))
    return sig((1.2 - d) / 0.25) * kq * ball_quality(df)


def ball_quality(df):
    """Is the ball we see the ball? The two views agree on it (aerial balls project a few metres
    apart, a wrong object tens of metres), it does not jump, and the tactical view has it."""
    dis = df["view_disagree"].to_numpy(np.float64)
    tc = df["tac_cov"].fillna(0).to_numpy()
    q1 = np.where(np.isfinite(dis), sig((4.0 - dis) / 0.7), np.where(tc >= 0.5, 0.8, 0.3))
    q2 = sig((5.0 - df["max_step"].fillna(0).to_numpy()) / 1.0)
    q3 = np.where(tc >= 0.3, 1.0, 0.5)
    # the tactical view is the one that sits on the visible ball: without it at the touch itself,
    # the touch rests on the broadcast track or the provided ball, which follow boots more often
    q4 = np.where(df["tac_seen"].fillna(0).to_numpy() >= 0.6, 1.0, 0.75) if "tac_seen" in df else 1.0
    return q1 * q2 * q3 * q4


def spells(codes_):
    """(start, end, team) maximal runs of H or A."""
    out = []
    c = codes_
    edge = np.flatnonzero(np.diff(c) != 0) + 1
    st = np.r_[0, edge]
    en = np.r_[edge, len(c)]
    for s, e in zip(st, en):
        if c[s] < 2:
            out.append((s, e, int(c[s])))
    return out


def insert(base, fr, ph, pa, level, lag, hold, max_hold, margin, end_lag=14, min_gap=10, end_level=0.5, setpiece=None):
    """Brief opposite spells inside the base's spells. The margin keeps clear of ends where the
    base already hands over to the other team; at an end next to a dead ball the brief spell may
    run into the stoppage (a last touch before the ball goes out), and a remnant of the spell's
    team shorter than the margin before it is absorbed."""
    out = base.copy()
    rows = []
    n = len(base)
    for s, e, T in spells(base):
        p_opp = pa if T == 0 else ph
        p_own = ph if T == 0 else pa
        m_s = margin if s > 0 and base[s - 1] == 1 - T else 0
        m_e = margin if e < n and base[e] == 1 - T else 0
        m = np.flatnonzero((fr + lag >= s + m_s) & (fr + lag < e - m_e) & (p_opp >= level))
        if setpiece is not None:
            m = m[~setpiece[m]]
        busy = -1
        for i in m:
            t = fr[i]
            if t + lag < busy:
                continue
            nxt = np.flatnonzero((fr >= t + min_gap) & (fr <= t + max_hold) & (p_own >= end_level))
            end = fr[nxt[0]] + end_lag if len(nxt) else t + lag + hold
            a, b = t + lag, min(end, e - m_e)
            if m_e == 0 and e - b < margin:
                b = e
            if b - a < 3:
                continue
            out[a:b] = 1 - T
            busy = b
            rows.append((t, a, b, T))
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="all")
    ap.add_argument("--lag", type=int, default=14)
    ap.add_argument("--end-lag", type=int, default=14)
    ap.add_argument("--min-gap", type=int, default=15)
    ap.add_argument("--hold", type=int, default=25)
    ap.add_argument("--max-hold", type=int, default=75)
    ap.add_argument("--margin", type=int, default=25)
    ap.add_argument("--prefix", default="aggressive")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(LOCAL, "events.parquet"))
    q = pd.read_parquet(os.path.join(LOCAL, f"q_{a.q}.parquet"))
    assert (q["frame"].to_numpy() == df["frame"].to_numpy()).all()
    fr = df["frame"].to_numpy()
    pt = touch_probability(df)
    ph = pt * q["q_h"].to_numpy()
    pa = pt * (1 - q["q_h"].to_numpy())
    base = codes(BASE)
    # a ball struck after resting is a restart: the ball was dead, so no brief possession there
    setpiece = ((df["restart_kick"].fillna(0) > 0) | (df["rest_before"].fillna(0) > 0.8)).to_numpy()
    # per-frame arrays and the candidate list
    P = np.zeros((2, N), np.float32)
    P[0, fr], P[1, fr] = ph, pa
    np.savez_compressed(os.path.join(LOCAL, "touches.npz"), p_touch_h=P[0], p_touch_a=P[1])
    team = np.where(ph >= pa, "H", "A")
    zz = df["z"].to_numpy()
    zh = np.where(np.isfinite(zz), zz, np.where(team == "H", df.get("kpx_h_h", df["kp_h_h"]), df.get("kpx_h_a", df["kp_h_a"])))
    bt = base[fr]
    cand = pd.DataFrame({
        "frame": fr, "team": team, "confidence": np.maximum(ph, pa).round(3),
        "p_touch": pt.round(3), "p_barca": q["q_h"].round(3),
        "ball_height": np.round(zh, 2), "height_from": np.where(np.isfinite(zz), "triangulated", "implied"),
        "dist_barca": df.get("kpx_d_h", df["kp_d_h"]).round(2), "dist_athletic": df.get("kpx_d_a", df["kp_d_a"]).round(2),
        "dist_barca_tracking": df["kp_d_h"].round(2), "dist_athletic_tracking": df["kp_d_a"].round(2),
        "kink_ms": np.fmax(df["kmax_t"].fillna(0), df["kmax_b"].fillna(0)).round(1),
        "base_label": np.array(list("HADX"))[bt],
    })
    cand["opposite_in_base"] = ((cand["base_label"] == "H") & (cand["team"] == "A")) | ((cand["base_label"] == "A") & (cand["team"] == "H"))
    cand.to_csv(os.path.join(LOCAL, "candidates.csv"), index=False)
    res = []
    for lv in LEVELS:
        pred, rows = insert(base, fr, ph, pa, lv, a.lag, a.hold, a.max_hold, a.margin, a.end_lag, a.min_gap,
                            setpiece=setpiece)
        res.append(dict(level=lv, inserted=len(rows)))
        write(os.path.join(LOCAL, f"{a.prefix}_{lv:g}.csv"), pred)
        conf = dict(zip(fr, np.maximum(ph, pa)))
        pd.DataFrame([dict(touch_frame=t, start=x, end=y, base_team="HA"[T], inserted_team="HA"[1 - T],
                           confidence=round(float(conf[t]), 3))
                      for t, x, y, T in rows]).to_csv(os.path.join(LOCAL, f"inserted_{lv:g}.csv"), index=False)
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
