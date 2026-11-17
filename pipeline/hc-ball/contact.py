"""Who is at the ball: player-ball contact per frame from the ball track and the detected players.

    python contact.py [--track .local/track_tactical.npz] [--view tactical|broadcast]

Players are touches-video's detector boxes (pixels) with their jersey-colour team (Barca, Athletic,
official; keepers by the goal they defend). Contact is measured in pixels, relative to the box
height h (~1.8 m), so it needs no ball height: a ball at the feet sits at the bottom of the box, a
header at the top. For each team, per frame:
  dn   distance from the ball pixel to the player's body (a vertical segment from head to feet,
       0.3 w wide), divided by h; 0 = the ball overlaps the body
  foot distance from the ball pixel to the foot point / h (a ground ball being played)
  dg   ground-plane distance (m) from the ball's ground point to the foot point
  p    jersey-colour confidence of that box
Ball motion from the track's ground point: speed before / after (m/s over 4 frames), direction
change, and a kink (velocity change, m/s).

Writes .local/contact_<view>.npz (length 199,319, NaN where unknown).
"""
import argparse
import glob
import os

import numpy as np

from common import HALVES, LOCAL, N, TOUCH_LANE, broadcast_index


def load_boxes(view):
    d, pre = ("players", "tac") if view == "tactical" else ("players_broadcast", "bro")
    parts = sorted(glob.glob(os.path.join(TOUCH_LANE, d, f"{pre}_*.npz")), key=lambda p: int(p.split("_")[-2]))
    fr = np.concatenate([np.load(p)["frames"] for p in parts])
    bx = np.concatenate([np.load(p)["boxes"] for p in parts])
    bt = np.load(os.path.join(TOUCH_LANE, "boxteam.npz" if view == "tactical" else "boxteam_broadcast.npz"))
    return fr, bx, bt


def per_team(bpx, bpy, boxes, prob, foot_m, bx_m, by_m):
    """bpx, bpy (n,), boxes (n,32,6), prob (n,32,3) -> dict of per-team nearest features."""
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    top, bot = y - h / 2, y + h / 2
    ox = np.maximum(0, np.abs(bpx[:, None] - x) - 0.3 * w)
    oy = np.where(bpy[:, None] > bot, bpy[:, None] - bot, np.where(bpy[:, None] < top, top - bpy[:, None], 0))
    dn = np.hypot(ox, oy) / h
    foot = np.hypot(bpx[:, None] - x, bpy[:, None] - bot) / h
    rel = (bot - bpy[:, None]) / h            # 0 at the feet, 1 at the head
    dg = np.hypot(foot_m[..., 0] - bx_m[:, None], foot_m[..., 1] - by_m[:, None])
    team = np.argmax(np.nan_to_num(prob, nan=-1), -1)
    pt = np.nanmax(prob, -1)
    valid = np.isfinite(dn) & np.isfinite(pt) & (team < 2)
    out = {}
    for k, nm in ((0, "h"), (1, "a")):
        m = valid & (team == k)
        d = np.where(m, dn, np.inf)
        j = np.argmin(d, 1)
        r = np.arange(len(j))
        ok = np.isfinite(d[r, j])
        out[f"dn_{nm}"] = np.where(ok, d[r, j], np.nan)
        out[f"foot_{nm}"] = np.where(ok, foot[r, j], np.nan)
        out[f"rel_{nm}"] = np.where(ok, rel[r, j], np.nan)
        out[f"dg_{nm}"] = np.where(ok, dg[r, j], np.nan)
        out[f"p_{nm}"] = np.where(ok, pt[r, j], np.nan)
        out[f"h_{nm}"] = np.where(ok, h[r, j], np.nan)
        # second nearest of the same team (crowding)
        d2 = d.copy()
        d2[r, j] = np.inf
        out[f"dn2_{nm}"] = np.where(np.isfinite(d2.min(1)), d2.min(1), np.nan)
        out[f"n_{nm}"] = (m & (dn < 1.0)).sum(1).astype(np.float32)
    return out


def motion(x, y, src):
    """Speed before/after (m/s over 4 frames), kink (m/s velocity change) from the ground track."""
    ok = src > 0
    X = np.where(ok, x, np.nan).astype(np.float64)
    Y = np.where(ok, y, np.nan).astype(np.float64)
    k = 4
    vbx = (X - np.roll(X, k)) / k * 25
    vby = (Y - np.roll(Y, k)) / k * 25
    vax = (np.roll(X, -k) - X) / k * 25
    vay = (np.roll(Y, -k) - Y) / k * 25
    kink = np.hypot(vax - vbx, vay - vby)
    sb, sa = np.hypot(vbx, vby), np.hypot(vax, vay)
    cosang = (vbx * vax + vby * vay) / np.maximum(sb * sa, 1e-6)
    return {"speed_b": sb, "speed_a": sa, "kink": kink, "cos": cosang, "vax": vax, "vay": vay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="tactical")
    ap.add_argument("--track", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    trk = np.load(a.track or os.path.join(LOCAL, f"track_{a.view}.npz"))
    fr, bx, bt = load_boxes(a.view)
    if a.view == "tactical":
        tf = fr
        rows = np.arange(len(fr))
    else:   # boxteam_broadcast is already re-indexed to tactical frames
        tf = bt["frames"]
        B = broadcast_index()
        pos = np.full(fr.max() + 2, -1)
        pos[fr] = np.arange(len(fr))
        bb = B[tf]
        rows = np.where((bb >= 0) & (bb < len(pos)), pos[np.clip(bb, 0, len(pos) - 1)], -1)
    keep = rows >= 0
    tf, rows = tf[keep], rows[keep]
    trow = np.searchsorted(bt["frames"], tf)
    out = {}
    n = len(tf)
    chunk = 20000
    acc = {}
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        f = tf[sl]
        r = per_team(trk["px"][f].astype(np.float64), trk["py"][f].astype(np.float64), bx[rows[sl]].astype(np.float64),
                     bt["prob"][trow[sl]].astype(np.float64), bt["foot"][trow[sl]].astype(np.float64),
                     trk["x"][f].astype(np.float64), trk["y"][f].astype(np.float64))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    for k, v in acc.items():
        arr = np.full(N, np.nan, np.float32)
        arr[tf] = np.concatenate(v)
        out[k] = arr
    out.update({k: v.astype(np.float32) for k, v in motion(trk["x"], trk["y"], trk["source"]).items()})
    out["source"] = trk["source"]
    out["score"] = trk["score"]
    path = os.path.join(LOCAL, f"contact_{a.tag or a.view}.npz")
    np.savez_compressed(path, **out)
    g = np.zeros(N, bool)
    for s0, e0 in HALVES:
        g[s0:e0] = True
    print(path, "frames with a player of each team measured:", np.isfinite(out["dn_h"][g]).mean().round(3),
          np.isfinite(out["dn_a"][g]).mean().round(3))


if __name__ == "__main__":
    main()
