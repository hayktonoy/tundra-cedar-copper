"""Turn per-frame ball detections into one cleaned ball track in pitch metres.

    python clean.py            # reads .local/det_tac_h*.npz, writes .local/ball_video.npz

1. Candidates: detections with score >= MIN_SCORE, mapped to the ground plane through the frame's
   homography, kept if they land on or near the pitch.
2. Tracklets: link candidates frame to frame (gaps up to MAX_GAP frames) when the ground
   displacement is plausible for a ball, highest score first.
3. Selection: each tracklet gets a value (summed score, bonus for agreeing with the provided ball
   and for being near a player, penalty when it sits still far from everyone, which is how paint,
   spots and boots on the touchline look). Weighted interval scheduling then picks the best set of
   tracklets that do not overlap in time.
4. Gaps up to FILL frames between chosen tracklets are linearly interpolated; elsewhere the
   provided ball is used and flagged.

Output arrays (length 199,319, NaN where unknown): x, y (metres), px, py (pixels), score,
source (0 none, 1 detection, 2 interpolated, 3 provided ball).
"""
import os

import numpy as np

from common import HALVES, LOCAL, N, TRACKING, homographies, players_teams
from geom import to_pitch, to_pixels

MIN_SCORE = 0.15
MAX_GAP = 4
STEP_M = 2.2       # metres per frame a candidate may move (fast shots, aerial projection jitter)
FILL = 15


def candidates(det, H=None):
    f, b = det["frames"], det["balls"]
    H = homographies()[f] if H is None else H
    xy = to_pitch(H, b[..., :2].astype(np.float64))
    ok = (b[..., 4] >= MIN_SCORE) & (xy[..., 0] > -4) & (xy[..., 0] < 109.3) & (xy[..., 1] > -4) & (xy[..., 1] < 72)
    # officials' boots and flags look like a ball: drop candidates inside a confident referee box
    r = det["refs"]
    conf = (r[..., 4] > 0.3)[:, None, :]
    inside = ((np.abs(b[..., None, 0] - r[:, None, :, 0]) < 0.6 * r[:, None, :, 2]) &
              (np.abs(b[..., None, 1] - r[:, None, :, 1]) < 0.6 * r[:, None, :, 3]) & conf).any(-1)
    ok &= ~inside
    return f, b, xy, ok


def tracklets(f, b, xy, ok):
    """Greedy frame-to-frame linking. Returns list of index arrays (frame row, slot)."""
    active = []   # [last_row, last_xy, members]
    done = []
    for r in range(len(f)):
        slots = np.flatnonzero(ok[r])
        slots = slots[np.argsort(-b[r, slots, 4])]
        claimed = set()
        nxt = []
        # extend existing tracks by best-scoring nearby candidate
        order = sorted(range(len(active)), key=lambda k: -len(active[k][2]))
        for k in order:
            last_r, last_xy, mem = active[k]
            gap = r - last_r
            if gap > MAX_GAP + 1:
                done.append(mem)
                continue
            best, bd = None, STEP_M * gap
            for s in slots:
                if s in claimed:
                    continue
                d = np.hypot(*(xy[r, s] - last_xy))
                if d <= bd:
                    best, bd = s, d
            if best is not None:
                claimed.add(best)
                mem.append((r, best))
                nxt.append([r, xy[r, best], mem])
            else:
                nxt.append(active[k])
        for s in slots:
            if s not in claimed:
                nxt.append([r, xy[r, s], [(r, s)]])
        active = nxt
    done.extend(m for _, _, m in active)
    return [np.array(m) for m in done]


def tracklet_value(m, f, b, xy, prov, near):
    rows, slots = m[:, 0], m[:, 1]
    sc = b[rows, slots, 4]
    p = xy[rows, slots]
    value = float(np.sum(sc))
    pv = prov[rows]
    agree = np.hypot(*(p - pv).T) < 2.5
    value += 0.3 * float(np.nansum(agree))
    nd = near[rows, slots]
    span = np.ptp(p, axis=0).max() if len(p) > 1 else 0.
    if len(m) > 50 and span < 1.0 and np.nanmedian(nd) > 6:
        value *= 0.05   # static and far from all players: paint, spot, equipment
    value -= 0.15 * float(np.sum(nd > 25))
    return value


def select(tracks, values, f):
    """Weighted interval scheduling over tracklets' frame spans."""
    spans = [(int(m[0, 0]), int(m[-1, 0])) for m in tracks]
    order = np.argsort([e for _, e in spans])
    ends = np.array([spans[i][1] for i in order])
    best = np.zeros(len(order) + 1)
    take = np.zeros(len(order), bool)
    prev = np.zeros(len(order), int)
    for k, i in enumerate(order):
        s = spans[i][0]
        prev[k] = np.searchsorted(ends, s, side="left")   # tracks ending strictly before s
        a = values[i] + best[prev[k]]
        take[k] = a > best[k]
        best[k + 1] = max(a, best[k])
    chosen = []
    k = len(order)
    while k > 0:
        if take[k - 1]:
            chosen.append(order[k - 1])
            k = prev[k - 1]
        else:
            k -= 1
    return chosen


def clean_half(det, prov_all, players, teams, H=None):
    f, b, xy, ok = candidates(det, H)
    xy = np.nan_to_num(xy, nan=-999.)
    prov = prov_all[f, :2].astype(np.float64)
    pl = players[f, :, :2].astype(np.float64)
    near = np.full(ok.shape, np.inf)
    for s in range(ok.shape[1]):
        d = np.hypot(pl[:, :, 0] - xy[:, s, None, 0], pl[:, :, 1] - xy[:, s, None, 1])
        near[:, s] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    tracks = [m for m in tracklets(f, b, xy, ok) if len(m) >= 3]
    values = np.array([tracklet_value(m, f, b, xy, prov, near) for m in tracks])
    keep = values > 0.8
    tracks = [m for m, k in zip(tracks, keep) if k]
    values = values[keep]
    chosen = select(tracks, values, f)
    n = len(f)
    out = {k: np.full(n, np.nan) for k in ("x", "y", "px", "py", "score")}
    src = np.zeros(n, np.int8)
    for i in chosen:
        m = tracks[i]
        r, s = m[:, 0], m[:, 1]
        out["x"][r], out["y"][r] = xy[r, s, 0], xy[r, s, 1]
        out["px"][r], out["py"][r] = b[r, s, 0], b[r, s, 1]
        out["score"][r] = b[r, s, 4]
        src[r] = 1
    have = np.flatnonzero(src == 1)
    for a, c in zip(have[:-1], have[1:]):
        if 1 < c - a <= FILL:
            for k in ("x", "y", "px", "py"):
                out[k][a + 1:c] = np.interp(np.arange(a + 1, c), [a, c], [out[k][a], out[k][c]])
            src[a + 1:c] = 2
    if H is not None:   # broadcast view: no fallback, pixels are broadcast pixels
        return f, out, src, len(tracks), len(chosen)
    miss = (src == 0) & ~np.isnan(prov[:, 0])
    out["x"][miss], out["y"][miss] = prov[miss, 0], prov[miss, 1]
    pp = to_pixels(homographies()[f[miss]], prov[miss][:, None, :])[:, 0]
    out["px"][miss], out["py"][miss] = pp[:, 0], pp[:, 1]
    src[miss] = 3
    return f, out, src, len(tracks), len(chosen)


def main(paths=None):
    z = np.load(TRACKING)
    prov, players = z["ball"], z["players"]
    teams = players_teams()
    full = {k: np.full(N, np.nan, np.float32) for k in ("x", "y", "px", "py", "score")}
    source = np.zeros(N, np.int8)
    paths = paths or [os.path.join(LOCAL, f"det_tac_h{h}.npz") for h in (1, 2)]
    for p in paths:
        det = np.load(p)
        f, out, src, nt, nc = clean_half(det, prov, players, teams)
        for k in full:
            full[k][f] = out[k]
        source[f] = src
        print(os.path.basename(p), "tracklets", nt, "chosen", nc,
              "sources", np.bincount(src, minlength=4) / len(src))
    np.savez_compressed(os.path.join(LOCAL, "ball_video.npz"), source=source, **full)


def broadcast_det(path):
    """Broadcast detections re-indexed to tactical frames (one broadcast frame per tactical frame)."""
    import json
    from common import ROOT
    sync = json.load(open(os.path.join(ROOT, "calibration", "broadcast_sync.json")))
    det = np.load(path)
    fb = det["frames"]
    Hb = homographies("broadcast")
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync["halves"]:
        m = (T >= h["from"]) & (T < h["to"])
        B[m] = np.round((T[m] - h["a"]) / (1 + h["b"])).astype(int)
    pos = {int(b): i for i, b in enumerate(fb)}
    rows = np.array([pos.get(int(b), -1) if b >= 0 else -1 for b in B])
    keep = rows >= 0
    t = T[keep]
    r = rows[keep]
    Hs = Hb[np.clip(fb[r], 0, len(Hb) - 1)]
    return {"frames": t, "balls": det["balls"][r], "refs": det["refs"][r]}, Hs


def main_broadcast():
    z = np.load(TRACKING)
    prov, players = z["ball"], z["players"]
    teams = players_teams()
    full = {k: np.full(N, np.nan, np.float32) for k in ("x", "y", "px", "py", "score")}
    source = np.zeros(N, np.int8)
    for h in (1, 2):
        det, Hs = broadcast_det(os.path.join(LOCAL, f"det_bro_h{h}.npz"))
        f, out, src, nt, nc = clean_half(det, prov, players, teams, H=Hs)
        for k in full:
            full[k][f] = out[k]
        source[f] = src
        print("broadcast half", h, "tracklets", nt, "chosen", nc, "sources", np.bincount(src, minlength=4) / len(src))
    np.savez_compressed(os.path.join(LOCAL, "ball_broadcast.npz"), source=source, **full)


if __name__ == "__main__":
    import sys
    main_broadcast() if "broadcast" in sys.argv else main()
