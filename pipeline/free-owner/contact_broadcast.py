"""Ball-player contact seen by the broadcast camera, on tactical frames -> .local/contact_broadcast.npz.

    python contact_broadcast.py

hc-ball's contact measure (ball pixel to the nearest Barca / Athletic body and feet, per box
height) applied to hc-ball's broadcast ball track and touches-video's broadcast players (jersey
colour teams), on tactical frames where the broadcast shows the pitch. The broadcast is the
closer camera, so it separates players in duels the tactical view merges. No labels.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hc-ball"))
import contact as hc  # noqa: E402  (hc-ball/contact.py)
from common import LOCAL as BALL_LANE, N, broadcast_index  # noqa: E402  (hc-ball/common.py)


def main():
    trk = np.load(os.path.join(BALL_LANE, "track_broadcast.npz"))
    fr, bx, bt = hc.load_boxes("broadcast")
    tf = bt["frames"]                                   # boxteam_broadcast is on tactical frames
    B = broadcast_index()
    pos = np.full(fr.max() + 2, -1)
    pos[fr] = np.arange(len(fr))
    bb = B[tf]
    rows = np.where((bb >= 0) & (bb < len(pos)), pos[np.clip(bb, 0, len(pos) - 1)], -1)
    keep = rows >= 0
    tf, rows = tf[keep], rows[keep]
    trow = np.searchsorted(bt["frames"], tf)
    acc = {}
    for s in range(0, len(tf), 20000):
        sl = slice(s, s + 20000)
        f = tf[sl]
        r = hc.per_team(trk["px"][f].astype(np.float64), trk["py"][f].astype(np.float64), bx[rows[sl]].astype(np.float64),
                        bt["prob"][trow[sl]].astype(np.float64), bt["foot"][trow[sl]].astype(np.float64),
                        trk["x"][f].astype(np.float64), trk["y"][f].astype(np.float64))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        arr = np.full(N, np.nan, np.float32)
        arr[tf] = np.concatenate(v)
        out[k] = arr
    out["source"] = trk["source"]
    np.savez_compressed(os.path.join(HERE, ".local", "contact_broadcast.npz"), **out)
    print("frames with broadcast contact measured:", int(np.isfinite(out["foot_h"]).sum()))


if __name__ == "__main__":
    main()
