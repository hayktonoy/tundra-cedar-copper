"""Stoppage entry and exit events from ball and player physics, no labels: .local/events.npz.

    python events.py

Arrays of length 199,319, each a non-negative strength at the frame where the event happens:

out     the ball crosses a line: inside (>= 0.3 m in) within the previous second, then >= 3 frames
        more than 0.3 m outside. Position from the two-view 3D ball where both cameras see it
        (a ball in the air projects off the pitch from one camera), else the tactical ground point.
lost    the ball is last seen within 2 m of a line moving outwards, then unseen for >= 1 s.
decel   collective deceleration: mean top-5 player speed over the next 2 s minus the previous 2 s,
        negated (m/s; players stop within a couple of seconds of a whistle).
accel   the reverse: players start running after a restart.
kick    the ball is struck from rest: ground speed >= 6 m/s within the next 4 frames after a
        detected ball moving < 1.5 m/s over the previous 0.5 s, or after >= 1 s without any ball.
setp    a Smart Tagging set piece starts 1-10 s before (the restart lies in that window for most).
drop    sharp collective stop: mean top-5 speed over the last second minus over the next second,
        kept at its local maxima (+-1 s) where >= 0.3 m/s (peaks 3 frames after a stoppage starts).
entry   the ball comes back onto the pitch: >= 3 frames more than 0.3 m outside, then inside
        (throw-ins, corners taken from the arc, a ball brought back into play).
fut     mean top-5 speed 0.5-2 s ahead (m/s; a look-ahead status signal, not an event).
"""
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from common import BALL_LANE, LOCAL, N, PITCH, tags, tracking


def ball_xy():
    t = np.load(os.path.join(BALL_LANE, "track_tactical.npz"))
    z = np.load(os.path.join(BALL_LANE, "ball3d.npz"))
    src = t["source"]
    x = np.where(src > 0, t["x"], np.nan).astype(float)
    y = np.where(src > 0, t["y"], np.nan).astype(float)
    x3, y3 = z["x"].astype(float), z["y"].astype(float)
    u = np.isfinite(x3)
    return np.where(u, x3, x), np.where(u, y3, y), src, t["speed"].astype(float)


def outside(x, y):
    return np.maximum(np.maximum(-x, x - PITCH[0]), np.maximum(-y, y - PITCH[1]))


def crossings(o):
    ev = np.zeros(N)
    on = np.nan_to_num(o, nan=-99)
    isout = on > 0.3
    i = 0
    while i < N:
        if isout[i]:
            j = i
            while j < N and (isout[j] or np.isnan(o[j])):
                j += 1
            if isout[i:j].sum() >= 3 and (on[max(0, i - 25):i] < -0.3).any():
                ev[i] = 1.0
            i = j
        else:
            i += 1
    return ev


def lost_near_line(x, y):
    ev = np.zeros(N)
    seen = np.isfinite(x)
    idx = np.flatnonzero(seen)
    gaps = np.diff(idx)
    for k in np.flatnonzero(gaps > 25):
        i = idx[k]
        if i < 5 or not seen[i - 4]:
            continue
        dx, dy = x[i] - x[i - 4], y[i] - y[i - 4]
        dist = min(x[i], PITCH[0] - x[i], y[i], PITCH[1] - y[i])
        outward = ((x[i] < 2 and dx < 0) or (x[i] > PITCH[0] - 2 and dx > 0) or
                   (y[i] < 2 and dy < 0) or (y[i] > PITCH[1] - 2 and dy > 0))
        if dist < 2 and outward:
            ev[i] = 1.0
    return ev


def past_mean(x, w):
    """mean of x[t - w : t] (0 at the start)."""
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    return (c[i] - c[lo]) / np.maximum(i - lo, 1)


def future_max(x, w):
    """max of x[t : t + w]."""
    return maximum_filter1d(np.r_[x, np.zeros(w)], w, origin=-(w // 2))[:len(x)]


def window_mean(x, a, b):
    """mean of x[t + a : t + b], clipped at the ends."""
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    lo = np.clip(i + a, 0, len(x))
    hi = np.clip(i + b, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def entries(o):
    ev = np.zeros(N)
    on = np.nan_to_num(o, nan=-99)
    isin = on < -0.3
    isout = on > 0.3
    for i in np.flatnonzero(isin[1:] & ~isin[:-1]) + 1:
        if isout[max(0, i - 25):i].sum() >= 3:
            ev[i] = 1.0
    return ev


def main():
    P, _ = tracking()
    sp = np.nan_to_num(P[..., 2].astype(float), nan=0)
    top5 = -np.sort(-sp, 1)[:, :5].mean(1)
    m50 = uniform_filter1d(top5, 50)
    fut = np.r_[m50[25:], np.full(25, m50[-1])]
    past = np.r_[np.full(25, m50[0]), m50[:-25]]
    x, y, src, bsp = ball_xy()
    o = outside(x, y)
    det = src == 1
    v = np.where(src > 0, bsp, np.nan)
    v = np.where(v > 45, np.nan, v)
    vmax4 = future_max(np.nan_to_num(v, nan=0), 5)                         # max over t..t+4
    vprev = past_mean(np.where(det, np.nan_to_num(v, nan=0), 0.0), 12)
    nprev = past_mean(det.astype(float), 12)                               # share detected in t-12..t-1
    anyprev = past_mean((src > 0).astype(float), 25)
    rest = (nprev > 0.6) & (vprev / np.maximum(nprev, 1e-6) < 1.5)
    absent = anyprev < 1e-6
    kick = ((rest | absent) & (vmax4 >= 6)).astype(float)
    kick = kick * (past_mean(kick, 12) == 0)                               # first frame of a run
    setp = np.zeros(N)
    for e in tags():
        if e["code"] == "SET PIECES":
            a = max(0, e["start_frame"] + 25)
            setp[a:min(N, e["start_frame"] + 250)] = 1.0
    drop = window_mean(top5, -25, 0) - window_mean(top5, 0, 25)
    peak = (drop >= 0.3) & (drop == maximum_filter1d(drop, 51))
    out = dict(out=crossings(o), lost=lost_near_line(x, y), decel=np.maximum(0, past - fut),
               accel=np.maximum(0, fut - past), kick=kick, setp=setp, drop=np.where(peak, drop, 0.0),
               entry=entries(o), fut=window_mean(top5, 12, 50))
    np.savez_compressed(os.path.join(LOCAL, "events.npz"), **{k: v.astype(np.float32) for k, v in out.items()})
    print({k: int((v > 0).sum()) if k not in ("decel", "accel", "fut") else round(float(v.mean()), 3) for k, v in out.items()})


if __name__ == "__main__":
    main()
