"""One ball track per view from every detector candidate, by dynamic programming with physics.

    python track.py tactical            # native + zoom candidates -> .local/track_tactical.npz
    python track.py broadcast           # native (+ zoom if present), on tactical frames -> .local/track_broadcast.npz
    python track.py tactical --no-zoom  # native candidates only (ablation)

Candidates: the native full-frame detections of hc-video (top 8 per frame) and the zoomed ones of
detect.py, merged within 4 px. Dropped: inside a confident referee box (assistant referees' boots
and flags), on the broadcast graphics. Each candidate goes to the ground plane through its frame's
homography. Both cameras sit on fixed mounts, so the ground point of the ray through the ball
does not move when the camera pans or zooms: motion is measured there, in metres, for aerial balls
too (a ball 5 m up moves ~1.2x faster on the ground plane).

The path through (frame, candidate) nodes maximises
  sum of rewards  log(score / TAU) + bonuses
  - SKIP per frame without a node
  - motion cost: speed above VMAX is impossible, acceleration above ACC0 costs ACC per m/frame
  - TELEPORT to jump anywhere (re-acquisition after a long loss or a wrong track).
Acceleration uses the velocity stored at the predecessor on its best path (greedy second order).
Gaps of up to FILL frames between physically linked nodes are interpolated.

Output per tactical frame: px, py (view pixels), x, y (ground plane, metres), score, zoom (bool:
the zoom pass confirmed it), source (0 none, 1 detected, 2 interpolated), speed (m/s, ground plane).
"""
import argparse
import os

import numba
import numpy as np

from common import HALVES, LOCAL, N, PITCH, TRACKING, VIDEO_LANE, TOUCH_LANE, broadcast_index, homographies, to_pitch

TAU = 0.3
SKIP = 0.15
TELEPORT = 6.0
VMAX = 1.8        # m per frame on the ground plane (45 m/s)
ACC0 = 0.25       # m/frame of free velocity change per step
ACC = 1.5
G = 20            # longest physically linked gap (frames)
OUT_PEN = 1.5     # reward penalty: candidate more than 1 m outside the lines
STATIC_PEN = 1.5  # reward penalty: still for +-1 s and more than 6 m from every tracked player
FILL = 20
MASKS = {"tactical": [(0, 0, 400, 100), (1640, 0, 1920, 180), (1640, 972, 1920, 1080)],
         "broadcast": [(0, 0, 1920, 130), (0, 940, 1920, 1080)]}


def merge_candidates(nat, zoom, k=12):
    """(n,8,5) native and zoom candidates -> (n,k,6): x, y, w, h, score, zoom flag; merged within 4 px."""
    n = len(nat)
    out = np.full((n, k, 6), np.nan, np.float32)
    allc = np.concatenate([np.c_[nat.reshape(-1, 5), np.zeros(nat.shape[0] * nat.shape[1])].reshape(n, -1, 6)]
                          + ([] if zoom is None else
                             [np.c_[zoom.reshape(-1, 5), np.ones(zoom.shape[0] * zoom.shape[1])].reshape(n, -1, 6)]), 1)
    return _merge(allc, out, k)


@numba.njit(cache=True)
def _merge(allc, out, k):
    n, m, _ = allc.shape
    for t in range(n):
        order = np.argsort(-np.nan_to_num(allc[t, :, 4], nan=-1.0))
        cnt = 0
        for q in order:
            c = allc[t, q]
            if not (c[4] >= 0.02):
                continue
            dup = -1
            for j in range(cnt):
                if abs(out[t, j, 0] - c[0]) < 4 and abs(out[t, j, 1] - c[1]) < 4:
                    dup = j
                    break
            if dup >= 0:
                if c[5] > 0:
                    out[t, dup, 5] = 1.0
                    # a candidate both passes see: keep the higher score
                    out[t, dup, 4] = max(out[t, dup, 4], c[4])
                continue
            if cnt < k:
                out[t, cnt, :] = c
                cnt += 1
    return out


@numba.njit(cache=True)
def dp(gx, gy, rew, valid, SKIP, TELEPORT, VMAX, ACC0, ACC, G):
    n, k = gx.shape
    V = np.full((n, k), -1e18)
    bt = np.full((n, k), -1, np.int64)     # predecessor frame
    bi = np.full((n, k), -1, np.int64)     # predecessor slot (-1 with frame>=0 means teleport/start)
    vx = np.zeros((n, k))
    vy = np.zeros((n, k))
    linked = np.zeros((n, k), np.bool_)
    # best value of any node up to t-1, minus SKIP for the frames since: for teleports
    Bv = 0.0
    Bt, Bi = -1, -1
    bestv_prev = np.full(n, -1e18)
    bestj_prev = np.full(n, -1, np.int64)
    for t in range(n):
        if t > 0:
            if bestv_prev[t - 1] > Bv - SKIP:
                Bv, Bt, Bi = bestv_prev[t - 1], t - 1, bestj_prev[t - 1]
            else:
                Bv -= SKIP
        else:
            Bv = 0.0
        for i in range(k):
            if not valid[t, i]:
                continue
            best = Bv - TELEPORT
            pt, pi, lk = Bt, Bi, False
            bvx, bvy = 0.0, 0.0
            for g in range(1, G + 1):
                s = t - g
                if s < 0:
                    break
                for j in range(k):
                    if not valid[s, j] or V[s, j] < -1e17:
                        continue
                    dx = (gx[t, i] - gx[s, j]) / g
                    dy = (gy[t, i] - gy[s, j]) / g
                    sp = np.sqrt(dx * dx + dy * dy)
                    if sp > VMAX + 0.5 / g:
                        continue
                    ax = dx - vx[s, j]
                    ay = dy - vy[s, j]
                    acc = np.sqrt(ax * ax + ay * ay)
                    c = V[s, j] - SKIP * (g - 1) - ACC * max(0.0, acc - ACC0 * g)
                    if c > best:
                        best, pt, pi, lk = c, s, j, True
                        bvx, bvy = dx, dy
            V[t, i] = best + rew[t, i]
            bt[t, i], bi[t, i] = pt, pi
            linked[t, i] = lk
            vx[t, i], vy[t, i] = bvx, bvy
        mv, mj = -1e18, -1
        for i in range(k):
            if V[t, i] > mv:
                mv, mj = V[t, i], i
        bestv_prev[t], bestj_prev[t] = mv, mj
    # end: best node, paying SKIP for the frames after it
    endv, et, ei = -1e18, -1, -1
    for t in range(n):
        v = bestv_prev[t] - SKIP * (n - 1 - t)
        if v > endv:
            endv, et, ei = v, t, bestj_prev[t]
    choice = np.full(n, -1, np.int64)
    link = np.zeros(n, np.bool_)
    t, i = et, ei
    while t >= 0 and i >= 0:
        choice[t] = i
        link[t] = linked[t, i]
        t, i = bt[t, i], bi[t, i]
    return choice, link


def static_mask(g, ok, lag=25, r=0.5):
    """Candidates with another candidate within r metres at both t - lag and t + lag (a still object)."""
    n, k = ok.shape
    gx = np.where(ok, g[..., 0], np.nan)
    gy = np.where(ok, g[..., 1], np.nan)
    out = np.ones((n, k), bool)
    for sh in (-lag, lag):
        ox = np.full_like(gx, np.nan)
        oy = np.full_like(gy, np.nan)
        if sh < 0:
            ox[-sh:], oy[-sh:] = gx[:sh], gy[:sh]
        else:
            ox[:-sh], oy[:-sh] = gx[sh:], gy[sh:]
        d = np.hypot(gx[:, :, None] - ox[:, None, :], gy[:, :, None] - oy[:, None, :])
        out &= np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2) < r
    return out & ok


def load_native(source):
    tag = "tac" if source == "tactical" else "bro"
    fr, ba, rf = [], [], []
    for h in (1, 2):
        z = np.load(os.path.join(VIDEO_LANE, f"det_{tag}_h{h}.npz"))
        fr.append(z["frames"]); ba.append(z["balls"]); rf.append(z["refs"])
    return np.concatenate(fr), np.concatenate(ba), np.concatenate(rf)


def load_zoom(source, frames, pattern="zoom_{tag}_h{h}.npz", chunk_glob="zoom_{tag}_h*.npz"):
    tag = "tac" if source == "tactical" else "bro"
    out = np.full((len(frames), 8, 5), np.nan, np.float32)
    pos = {int(f): i for i, f in enumerate(frames)}
    got = 0
    import glob
    paths = [os.path.join(LOCAL, pattern.format(tag=tag, h=h)) for h in (1, 2)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:   # merged files not written yet: whatever chunks exist
        paths = sorted(glob.glob(os.path.join(LOCAL, "chunks", chunk_glob.format(tag=tag))))
    for p in paths:
        z = np.load(p)
        rows = np.array([pos.get(int(f), -1) for f in z["frames"]])
        ok = rows >= 0
        out[rows[ok]] = z["balls"][ok]
        got += ok.sum()
    return out if got else None


def build(source, use_zoom=True, params=None):
    if params:
        globals().update(params)
    fr, nat, refs = load_native(source)
    zoom = load_zoom(source, fr) if use_zoom else None
    gap = load_zoom(source, fr, "gap3_{tag}_h{h}.npz", "gap3_{tag}_h*.npz") if use_zoom else None
    if gap is not None:   # the 3x pass on frames the first track had to interpolate
        zoom = gap if zoom is None else np.concatenate([zoom, gap], 1)
    cand = merge_candidates(nat, zoom)
    # drop candidates on graphics or inside a confident referee box
    x, y = cand[..., 0], cand[..., 1]
    ok = np.isfinite(cand[..., 4])
    for x0, y0, x1, y1 in MASKS[source]:
        ok &= ~((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    conf = (refs[..., 4] > 0.3)[:, None, :]
    inside = ((np.abs(x[..., None] - refs[:, None, :, 0]) < 0.6 * refs[:, None, :, 2]) &
              (np.abs(y[..., None] - refs[:, None, :, 1]) < 0.6 * refs[:, None, :, 3]) & conf).any(-1)
    ok &= ~inside
    # homographies per row (broadcast: by broadcast frame), ground plane
    if source == "tactical":
        Hm = homographies("tactical")[fr]
        tframes = fr
    else:
        Hb = homographies("broadcast")
        B = broadcast_index()
        row = np.full(fr.max() + 2, -1)
        row[fr] = np.arange(len(fr))
        tframes = np.flatnonzero(B >= 0)
        bb = B[tframes]
        r = np.where(bb < len(row), row[np.clip(bb, 0, len(row) - 1)], -1)
        keep = r >= 0
        tframes, r = tframes[keep], r[keep]
        cand, ok, fr = cand[r], ok[r], fr[r]
        Hm = Hb[fr]
        hv = np.isfinite(Hm[:, 0, 0])
        ok &= hv[:, None]
        Hm = np.where(hv[:, None, None], Hm, np.eye(3)[None])
    g = to_pitch(Hm, np.nan_to_num(cand[..., :2]).astype(np.float64))
    ok &= (g[..., 0] > -6) & (g[..., 0] < 111) & (g[..., 1] > -6) & (g[..., 1] < 74)
    # rewards: detector confidence, agreement with the provided ball, distance to the nearest player
    s = np.nan_to_num(cand[..., 4], nan=0.01).astype(np.float64)
    rew = np.log(np.maximum(s, 0.01) / TAU)
    prov = np.load(TRACKING)["ball"][tframes, :2].astype(np.float64)
    dprov = np.hypot(g[..., 0] - prov[:, None, 0], g[..., 1] - prov[:, None, 1])
    rew += 0.3 * np.nan_to_num(dprov < 3.0)
    pl = np.load(TRACKING)["players"][tframes, :, :2].astype(np.float64)
    near = np.full(ok.shape, np.inf)
    for q in range(cand.shape[1]):
        d = np.hypot(pl[:, :, 0] - g[:, q, None, 0], pl[:, :, 1] - g[:, q, None, 1])
        near[:, q] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    rew -= 0.6 * (near > 20)
    # spare balls rest beyond the touchlines (by the technical areas) and are confidently detected
    # for seconds: penalise candidates outside the lines, and static objects away from the players
    outside = np.maximum(np.maximum(-g[..., 0], g[..., 0] - PITCH[0]), np.maximum(-g[..., 1], g[..., 1] - PITCH[1]))
    rew -= OUT_PEN * (outside > 1.0)
    rew -= STATIC_PEN * (static_mask(g, ok) & (near > 6))
    out = {k: np.full(N, np.nan, np.float32) for k in ("px", "py", "x", "y", "score", "speed")}
    zflag = np.zeros(N, bool)
    src = np.zeros(N, np.int8)
    for a, b in HALVES:
        m = (tframes >= a) & (tframes < b)
        idx = np.flatnonzero(m)
        # one DP over contiguous tactical frames of this half (absent rows: no valid candidates)
        T = np.arange(a, b)
        pos = np.full(b - a, -1)
        pos[tframes[idx] - a] = idx
        k = cand.shape[1]
        GX = np.zeros((b - a, k)); GY = np.zeros((b - a, k)); R = np.zeros((b - a, k)); VA = np.zeros((b - a, k), bool)
        have = pos >= 0
        GX[have], GY[have] = g[pos[have], :, 0], g[pos[have], :, 1]
        R[have] = rew[pos[have]]
        VA[have] = ok[pos[have]]
        choice, link = dp(GX, GY, R, VA, SKIP, TELEPORT, VMAX, ACC0, ACC, int(G))
        sel = np.flatnonzero(choice >= 0)
        rows = pos[sel]
        f = T[sel]
        c = choice[sel]
        out["px"][f], out["py"][f] = cand[rows, c, 0], cand[rows, c, 1]
        out["x"][f], out["y"][f] = g[rows, c, 0], g[rows, c, 1]
        out["score"][f] = cand[rows, c, 4]
        zflag[f] = cand[rows, c, 5] > 0
        src[f] = 1
        # interpolate physically linked gaps
        for p0, p1, lk in zip(sel[:-1], sel[1:], link[sel[1:]]):
            if lk and 1 < p1 - p0 <= FILL:
                ff = np.arange(a + p0 + 1, a + p1)
                for key in ("px", "py", "x", "y"):
                    out[key][ff] = np.interp(ff, [a + p0, a + p1], [out[key][a + p0], out[key][a + p1]])
                src[ff] = 2
    sp = np.full(N, np.nan, np.float32)
    v = np.hypot(np.gradient(out["x"]), np.gradient(out["y"])) * 25
    sp[:] = v
    out["speed"] = sp
    return dict(out, zoom=zflag, source=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["tactical", "broadcast"])
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = build(a.source, not a.no_zoom)
    out = a.out or os.path.join(LOCAL, f"track_{a.source}{'_native' if a.no_zoom else ''}.npz")
    np.savez_compressed(out, **res)
    g = np.zeros(N, bool)
    for s, e in HALVES:
        g[s:e] = True
    src = res["source"][g]
    print(out, "graded frames: detected %.3f interpolated %.3f none %.3f zoom-confirmed %.3f" % (
        (src == 1).mean(), (src == 2).mean(), (src == 0).mean(), res["zoom"][g].mean()))


if __name__ == "__main__":
    main()
