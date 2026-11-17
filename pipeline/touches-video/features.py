"""Per-frame touch evidence from the two video ball tracks and the tracked players.

    python features.py        # writes .local/frames.npz

Kinematics. For each view (tactical, broadcast) and each frame t, a continuous hinge line
p(tau) = a + b*tau + c*max(tau, 0) is fitted to the detected ground projections in t-W..t+W
(one outlier pass). c is the velocity change at t (m/s): a touch, a bounce, or noise. The same
window also gets a smooth quadratic; the hinge's error reduction over it measures how kink-like
the motion is. Projections of an aerial ball are smooth curves, so kinks survive projection.

Contact. A ball seen in one view lies somewhere on the ray from that camera through its ground
projection. For every player, the ball could be at that player at height h if the ray passes
over the player's ground position at height h. With both views, h is the height that brings
both rays closest to the player; the residual is the contact distance. This finds headers that
the ground projection puts metres away. Camera centres from pipeline/multiview-ball/.
"""
import os

import numba
import numpy as np

from common import HALVES, LOCAL, N, load_ball, load_players

W = 6            # frames each side of the hinge
HMAX = 2.8       # highest touch (headers, keeper hands)
C_T = np.array([53.36, 115.52, 29.97])
C_B = np.array([52.88, 113.08, 21.33])


@numba.njit(cache=True)
def _solve3(M, r):
    # 3x3 solve by Cramer's rule; returns nan on singular
    d = (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
         + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))
    out = np.full(3, np.nan)
    if abs(d) < 1e-9:
        return out
    for k in range(3):
        Mk = M.copy()
        for i in range(3):
            Mk[i, k] = r[i]
        out[k] = (Mk[0, 0] * (Mk[1, 1] * Mk[2, 2] - Mk[1, 2] * Mk[2, 1]) - Mk[0, 1] * (Mk[1, 0] * Mk[2, 2] - Mk[1, 2] * Mk[2, 0])
                  + Mk[0, 2] * (Mk[1, 0] * Mk[2, 1] - Mk[1, 1] * Mk[2, 0])) / d
    return out


@numba.njit(cache=True)
def _fit(tau, x, y, w, hinge):
    """Weighted LS of x,y on [1, tau, hinge ? max(tau,0) : tau^2]. Returns coefs (2,3), sse."""
    M = np.zeros((3, 3))
    rx = np.zeros(3)
    ry = np.zeros(3)
    for i in range(len(tau)):
        if w[i] == 0:
            continue
        f = np.array([1.0, tau[i], max(tau[i], 0.0) if hinge else tau[i] * tau[i]])
        for a in range(3):
            rx[a] += f[a] * x[i]
            ry[a] += f[a] * y[i]
            for b in range(3):
                M[a, b] += f[a] * f[b]
    cx = _solve3(M, rx)
    cy = _solve3(M, ry)
    sse = 0.0
    for i in range(len(tau)):
        if w[i] == 0:
            continue
        f2 = max(tau[i], 0.0) if hinge else tau[i] * tau[i]
        ex = x[i] - (cx[0] + cx[1] * tau[i] + cx[2] * f2)
        ey = y[i] - (cy[0] + cy[1] * tau[i] + cy[2] * f2)
        sse += ex * ex + ey * ey
    out = np.empty((2, 3))
    out[0] = cx
    out[1] = cy
    return out, sse


@numba.njit(cache=True)
def kinks(x, y, ok, lo, hi, W):
    """Hinge fit at every frame in [lo, hi). Returns dvx, dvy, vbx, vby, gain, nb, na, ax, ay."""
    n = len(x)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    dvx = np.full(n, np.nan)
    dvy = np.full(n, np.nan)
    vbx = np.full(n, np.nan)
    vby = np.full(n, np.nan)
    gain = np.full(n, np.nan)
    nb = np.zeros(n, np.int16)
    na = np.zeros(n, np.int16)
    for t in range(lo + W, hi - W):
        m = 2 * W + 1
        tau = np.empty(m)
        xs = np.empty(m)
        ys = np.empty(m)
        w = np.zeros(m)
        cb = 0
        ca = 0
        for k in range(m):
            s = t - W + k
            tau[k] = (k - W) / 25.0
            xs[k] = x[s]
            ys[k] = y[s]
            if ok[s]:
                w[k] = 1.0
                if k < W:
                    cb += 1
                elif k > W:
                    ca += 1
        if cb < 3 or ca < 3:
            continue
        for it in range(2):
            c, sse = _fit(tau, xs, ys, w, True)
            if it == 1:
                break
            # drop the worst point if it is far off the hinge line
            worst = -1
            wv = 0.6 * 0.6
            for k in range(m):
                if w[k] == 0:
                    continue
                f2 = max(tau[k], 0.0)
                ex = xs[k] - (c[0, 0] + c[0, 1] * tau[k] + c[0, 2] * f2)
                ey = ys[k] - (c[1, 0] + c[1, 1] * tau[k] + c[1, 2] * f2)
                if ex * ex + ey * ey > wv:
                    wv = ex * ex + ey * ey
                    worst = k
            if worst < 0:
                break
            w[worst] = 0.0
            if worst < W:
                cb -= 1
            elif worst > W:
                ca -= 1
            if cb < 3 or ca < 3:
                break
        if cb < 3 or ca < 3:
            continue
        c2, sse2 = _fit(tau, xs, ys, w, False)
        dvx[t] = c[0, 2]
        dvy[t] = c[1, 2]
        ax[t] = c[0, 0]
        ay[t] = c[1, 0]
        vbx[t] = c[0, 1]
        vby[t] = c[1, 1]
        gain[t] = (sse2 - sse) / (sse + 0.05 * 0.05 * (cb + ca + 1))
        nb[t] = cb
        na[t] = ca
    return dvx, dvy, vbx, vby, gain, nb, na, ax, ay


def view_kinematics(x, y, ok, w=W):
    out = {k: np.full(N, np.nan, np.float32) for k in ("dvx", "dvy", "vbx", "vby", "gain", "ax", "ay")}
    out["nb"] = np.zeros(N, np.int16)
    out["na"] = np.zeros(N, np.int16)
    xx = np.nan_to_num(x.astype(np.float64))
    yy = np.nan_to_num(y.astype(np.float64))
    for lo, hi in HALVES:
        r = kinks(xx, yy, ok, lo, hi, w)
        for k, a in zip(("dvx", "dvy", "vbx", "vby", "gain", "nb", "na", "ax", "ay"), r):
            out[k][lo:hi] = a[lo:hi]
    return out


def ray_contact(pl, gT, gB):
    """Per frame and player: contact distance and implied height from one or both camera rays.

    pl (n,P,2) players; gT, gB (n,2) ground projections (NaN where that view has no ball).
    The horizontal ray point at height h is C + (g - C)(1 - h/Cz) = g + h*u with u = (C - g)/Cz.
    Minimise sum over views of |g_v + h u_v - p|^2 over h in [0, HMAX]."""
    num = np.zeros(pl.shape[:2])
    den = np.zeros(pl.shape[:2])
    views = []
    for g, C in ((gT, C_T), (gB, C_B)):
        have = np.isfinite(g[:, 0])
        u = (C[:2][None] - g) / C[2]
        e = g[:, None, :] - pl            # (n,P,2) offset at h = 0
        num += np.where(have[:, None], -(e * u[:, None, :]).sum(-1), 0)
        den += np.where(have[:, None], (u * u).sum(-1)[:, None], 0)
        views.append((have, u, e))
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.clip(num / den, 0, HMAX)
    J = np.zeros(pl.shape[:2])
    nv = np.zeros(len(pl))
    for have, u, e in views:
        r = e + h[..., None] * u[:, None, :]
        J += np.where(have[:, None], (r * r).sum(-1), 0)
        nv += have
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(J / nv[:, None])
    d[nv == 0] = np.nan
    return d.astype(np.float32), h.astype(np.float32), nv.astype(np.int8)


def per_team(d, h, teams, extra=None):
    """Nearest player of each team: distance, implied height, slot, and second-nearest distance."""
    out = {}
    dd = np.where(np.isfinite(d), d, np.inf)
    for t, name in ((0, "h"), (1, "a")):
        idx = np.flatnonzero(teams == t)
        sub = dd[:, idx]
        o = np.argsort(sub, 1)
        j1 = idx[o[:, 0]]
        d1 = np.take_along_axis(sub, o[:, :1], 1)[:, 0]
        d2 = np.take_along_axis(sub, o[:, 1:2], 1)[:, 0]
        out[f"d_{name}"] = np.where(np.isinf(d1), np.nan, d1).astype(np.float32)
        out[f"d2_{name}"] = np.where(np.isinf(d2), np.nan, d2).astype(np.float32)
        out[f"h_{name}"] = h[np.arange(len(h)), j1].astype(np.float32)
        out[f"slot_{name}"] = j1.astype(np.int16)
    return out


def main():
    b = load_ball()
    pl, teams = load_players()
    feats = {}
    # tactical: detections only (interpolated frames hide kinks, provided fallback is another sensor)
    okT = b["src"] == 1
    okB = b["bsrc"] == 1
    for name, x, y, ok in (("t", b["x"], b["y"], okT), ("b", b["bx"], b["by"], okB)):
        k = view_kinematics(x, y, ok)
        for key, a in k.items():
            feats[f"{name}_{key}"] = a
        print(name, "kink frames", np.isfinite(k["dvx"]).sum())
        # a wider hinge bridges the detector's gaps in duels (ball hidden by bodies)
        k = view_kinematics(x, y, ok, 2 * W)
        for key in ("dvx", "dvy", "vbx", "vby", "gain", "ax", "ay"):
            feats[f"{name}w_{key}"] = k[key]
        print(name, "wide kink frames", np.isfinite(k["dvx"]).sum())
    # contact geometry, chunked to stay small in memory
    gT = np.c_[b["x"], b["y"]].astype(np.float64)
    gT[~np.isin(b["src"], (1, 2))] = np.nan
    gB = np.c_[b["bx"], b["by"]].astype(np.float64)
    gB[~np.isin(b["bsrc"], (1, 2))] = np.nan
    cols = {}
    step = 20000
    for s in range(0, N, step):
        sl = slice(s, min(N, s + step))
        p = pl[sl].astype(np.float64)
        for tag, a, bb in (("c", gT[sl], gB[sl]), ("ct", gT[sl], np.full_like(gB[sl], np.nan))):
            d, h, nv = ray_contact(p, a, bb)
            pt = per_team(d, h, teams)
            for k, v in pt.items():
                cols.setdefault(f"{tag}_{k}", []).append(v)
            if tag == "c":
                cols.setdefault("nviews", []).append(nv)
        # ground distance from the best ground point (3D where triangulated, else tactical)
        g = np.where(np.isfinite(b["x3"][sl])[:, None], np.c_[b["x3"][sl], b["y3"][sl]], gT[sl])
        dg = np.hypot(p[..., 0] - g[:, None, 0], p[..., 1] - g[:, None, 1])
        pt = per_team(dg, np.zeros_like(dg), teams)
        for k in ("d_h", "d_a", "d2_h", "d2_a", "slot_h", "slot_a"):
            cols.setdefault(f"g_{k}", []).append(pt[k])
    for k, v in cols.items():
        feats[k] = np.concatenate(v)
    np.savez_compressed(os.path.join(LOCAL, "frames.npz"), **feats)
    print("wrote", len(feats), "arrays")


if __name__ == "__main__":
    main()
