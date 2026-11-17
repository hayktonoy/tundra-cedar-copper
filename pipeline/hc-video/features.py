"""Per-frame possession features from a ball track and the provided player tracking.

features(ball_xy, prefix) -> dict of float32 arrays of length N. Computed independently within
each half. Nothing here reads labels.
"""
import numpy as np
import pandas as pd

from common import HALVES, N, TRACKING, players_teams

CONTROL_M = 1.5


def _players():
    z = np.load(TRACKING)
    return z["players"][..., :2].astype(np.float32), players_teams()


def nearest(ball, pl, teams):
    """Distance from the ball to the nearest player of each team, and the overall nearest slot."""
    d = np.hypot(pl[..., 0] - ball[:, None, 0], pl[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    dh = d[:, teams == 0].min(1)
    da = d[:, teams == 1].min(1)
    second = {}
    for t, name in ((0, "h"), (1, "a")):
        s = np.sort(d[:, teams == t], 1)
        second[name] = s[:, 1] if s.shape[1] > 1 else np.full(len(d), np.inf)
    unknown = np.isnan(ball[:, 0])
    for arr in (dh, da, second["h"], second["a"]):
        arr[unknown] = np.nan
        arr[np.isinf(arr)] = np.nan
    return dh, da, second["h"], second["a"]


def _ffill_last(values, valid):
    """For each frame, the last valid value and frames since it (within the array)."""
    idx = np.where(valid, np.arange(len(values)), -1)
    idx = np.maximum.accumulate(idx)
    last = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
    age = np.where(idx >= 0, np.arange(len(values)) - idx, np.nan)
    return last, age


def chains(ball, pl, teams, known):
    """Control events (nearest player within CONTROL_M) grouped into same-team chains.

    Returns per-frame: team of the chain of the last control, distinct players in that chain so
    far, its age in frames; and the same for the chain of the next control."""
    d = np.hypot(pl[..., 0] - ball[:, None, 0], pl[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    slot = d.argmin(1)
    ok = known & (d.min(1) < CONTROL_M)
    n = len(ball)
    ev = np.flatnonzero(ok)
    team_e = teams[slot[ev]]
    # chain id increments when the controlling team changes
    cid = np.r_[0, np.cumsum(team_e[1:] != team_e[:-1])] if len(ev) else np.zeros(0, int)
    distinct = np.zeros(len(ev))
    start_f = np.zeros(len(ev))
    seen, cur, first = set(), -1, 0
    for i, e in enumerate(ev):
        if cid[i] != cur:
            seen, cur, first = set(), cid[i], e
        seen.add(slot[e])
        distinct[i] = len(seen)
        start_f[i] = first
    # total distinct players per chain (for the "next" view)
    total = np.zeros(len(ev))
    if len(ev):
        last_of = np.r_[np.flatnonzero(cid[1:] != cid[:-1]), len(ev) - 1]
        tot_by_chain = distinct[last_of]
        total = tot_by_chain[cid]
    pos = np.searchsorted(ev, np.arange(n), side="right") - 1
    out = {}
    has = pos >= 0
    q = np.maximum(pos, 0)
    out["ch_team"] = np.where(has, team_e[q] if len(ev) else np.nan, np.nan)
    out["ch_npl"] = np.where(has, distinct[q] if len(ev) else np.nan, np.nan)
    out["ch_age"] = np.where(has, np.minimum(np.arange(n) - (start_f[q] if len(ev) else 0), 750), np.nan)
    nxt = np.searchsorted(ev, np.arange(n), side="left")
    hasn = nxt < len(ev)
    r = np.minimum(nxt, max(len(ev) - 1, 0))
    out["chn_team"] = np.where(hasn, team_e[r] if len(ev) else np.nan, np.nan)
    out["chn_npl"] = np.where(hasn, total[r] if len(ev) else np.nan, np.nan)
    return out


def half_features(ball, pl, teams, p):
    out = {}
    dh, da, dh2, da2 = nearest(ball, pl, teams)
    known = ~np.isnan(ball[:, 0])
    out[p + "known"] = known.astype(np.float32)
    out[p + "dh"] = np.minimum(dh, 40)
    out[p + "da"] = np.minimum(da, 40)
    margin = np.clip(np.log1p(np.nan_to_num(da, nan=40)) - np.log1p(np.nan_to_num(dh, nan=40)), -3, 3)
    margin[~known] = np.nan
    out[p + "margin"] = margin
    v = np.full(len(ball), np.nan)
    v[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * 25
    speed = pd.Series(v).rolling(5, center=True, min_periods=2).median().to_numpy()
    out[p + "speed"] = np.minimum(speed, 40)
    ctrl_h = known & (dh < CONTROL_M) & (np.nan_to_num(da, nan=99) > dh)
    ctrl_a = known & (da < CONTROL_M) & (np.nan_to_num(dh, nan=99) > da)
    ctrl = np.where(ctrl_h, 0., np.where(ctrl_a, 1., np.nan))
    # last/next controlling team and time since/until
    valid = ~np.isnan(ctrl)
    last, age = _ffill_last(ctrl, valid)
    nxt, ahead = _ffill_last(ctrl[::-1], valid[::-1])
    out[p + "last_ctrl"] = last
    out[p + "last_age"] = np.minimum(age, 500)
    out[p + "next_ctrl"] = nxt[::-1]
    out[p + "next_age"] = np.minimum(ahead[::-1], 500)
    # where the ball is: outside the lines means a stoppage is near
    out[p + "x"] = ball[:, 0]
    out[p + "y"] = ball[:, 1]
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - 105.3, -ball[:, 1], ball[:, 1] - 68.0])
    out[p + "outside"] = np.clip(outside, -20, 10)
    is_out = known & (outside > 0.5)
    _, age_o = _ffill_last(is_out.astype(float), is_out)
    _, ahead_o = _ffill_last(is_out[::-1].astype(float), is_out[::-1])
    out[p + "out_age"] = np.minimum(age_o, 750)
    out[p + "out_ahead"] = np.minimum(ahead_o[::-1], 750)
    for k, v in chains(ball, pl, teams, known).items():
        out[p + k] = v
    s_h = pd.Series(ctrl_h.astype(float))
    s_a = pd.Series(ctrl_a.astype(float))
    s_k = pd.Series(known.astype(float))
    s_m = pd.Series(margin)
    for w in (13, 25, 50, 100, 200):
        for name, s in (("ch", s_h), ("ca", s_a), ("k", s_k), ("m", s_m)):
            past = s.rolling(w, min_periods=1).mean().to_numpy()
            fut = s[::-1].rolling(w, min_periods=1).mean().to_numpy()[::-1]
            out[f"{p}{name}_p{w}"] = past
            out[f"{p}{name}_f{w}"] = fut
            if w in (25, 100):
                out[f"{p}{name}_c{w}"] = s.rolling(w, center=True, min_periods=1).mean().to_numpy()
    # share of control by H in the surrounding windows (who keeps the ball)
    for w in (50, 125, 250):
        h = s_h.rolling(w, center=True, min_periods=1).sum().to_numpy()
        a = s_a.rolling(w, center=True, min_periods=1).sum().to_numpy()
        out[f"{p}share{w}"] = (h + 1) / (h + a + 2)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def features(ball_xy, prefix):
    pl, teams = _players()
    cols = None
    for a, b in HALVES:
        f = half_features(ball_xy[a:b].astype(np.float32), pl[a:b], teams, prefix)
        if cols is None:
            cols = {k: np.full(N, np.nan, np.float32) for k in f}
        for k, v in f.items():
            cols[k][a:b] = v
    return cols


def height_features(p="h_"):
    """Features from the triangulated ball: the true ground position under the ball where both views
    see it (tactical video ball elsewhere), its height, and control only when the ball is low."""
    import os
    from common import LOCAL
    v = np.load(os.path.join(LOCAL, "ball_video.npz"))
    z3 = np.load(os.path.join(LOCAL, "ball_3d.npz"))
    xy = np.c_[v["x"], v["y"]].astype(np.float32)
    has3 = ~np.isnan(z3["x"])
    xy[has3] = np.c_[z3["x"], z3["y"]][has3]
    z = pd.Series(z3["zs"]).interpolate(limit=12, limit_area="inside").to_numpy(np.float32)
    pl, teams = _players()
    cols = {}
    for a, b in HALVES:
        ball = xy[a:b].copy()
        zz = z[a:b]
        dh, da, _, _ = nearest(ball, pl[a:b], teams)
        low = ~(zz > 1.0)          # unknown height counts as low
        f = {}
        f["z"] = zz
        f["air"] = (zz > 1.0).astype(np.float32)
        f["air"][np.isnan(zz)] = np.nan
        f["dh"] = np.minimum(dh, 40)
        f["da"] = np.minimum(da, 40)
        ch = (low & (dh < CONTROL_M) & (np.nan_to_num(da, nan=99) > dh)).astype(float)
        ca = (low & (da < CONTROL_M) & (np.nan_to_num(dh, nan=99) > da)).astype(float)
        head_h = ((zz > 1.4) & (zz < 3.0) & (dh < 1.2)).astype(float)
        head_a = ((zz > 1.4) & (zz < 3.0) & (da < 1.2)).astype(float)
        ctrl = np.where(ch > 0, 0., np.where(ca > 0, 1., np.nan))
        valid = ~np.isnan(ctrl)
        last, age = _ffill_last(ctrl, valid)
        nxt, ahead = _ffill_last(ctrl[::-1], valid[::-1])
        f["last_ctrl"], f["last_age"] = last, np.minimum(age, 500)
        f["next_ctrl"], f["next_age"] = nxt[::-1], np.minimum(ahead[::-1], 500)
        for name, arr in (("ch", ch), ("ca", ca), ("hh", head_h), ("ha", head_a), ("air", f["air"])):
            s = pd.Series(arr)
            for w in (13, 50, 125):
                f[f"{name}_p{w}"] = s.rolling(w, min_periods=1).mean().to_numpy()
                f[f"{name}_f{w}"] = s[::-1].rolling(w, min_periods=1).mean().to_numpy()[::-1]
        for k, val in chains(ball, pl[a:b], teams, ~np.isnan(ball[:, 0]) & low).items():
            f[k] = val
        if not cols:
            cols = {p + k: np.full(N, np.nan, np.float32) for k in f}
        for k, val in f.items():
            cols[p + k][a:b] = val
    return cols
