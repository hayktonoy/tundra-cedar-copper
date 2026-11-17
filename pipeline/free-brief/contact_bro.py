"""hc-ball's pixel contact measured on the broadcast view, written to this lane's .local. No labels.

    python contact_bro.py    -> .local/contact_broadcast.npz

Reuses hc-ball's contact.py functions unchanged (ball pixel vs touches-video's broadcast player boxes
and jersey teams, indexed by tactical frame through the algorithmic sync).
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hc-ball"))
import contact as hc  # noqa: E402
from common import HALVES, LOCAL as BALL, N, broadcast_index  # noqa: E402  hc-ball's


def main():
    trk = np.load(os.path.join(BALL, "track_broadcast.npz"))
    fr, bx, bt = hc.load_boxes("broadcast")
    tf = bt["frames"]
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
        r = hc.per_team(trk["px"][f].astype(np.float64), trk["py"][f].astype(np.float64),
                        bx[rows[sl]].astype(np.float64), bt["prob"][trow[sl]].astype(np.float64),
                        bt["foot"][trow[sl]].astype(np.float64), trk["x"][f].astype(np.float64),
                        trk["y"][f].astype(np.float64))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        arr = np.full(N, np.nan, np.float32)
        arr[tf] = np.concatenate(v)
        out[k] = arr
    out.update({k: v.astype(np.float32) for k, v in hc.motion(trk["x"], trk["y"], trk["source"]).items()})
    out["source"] = trk["source"]
    path = HERE / ".local" / "contact_broadcast.npz"
    np.savez_compressed(path, **out)
    g = np.zeros(N, bool)
    for a, b in HALVES:
        g[a:b] = True
    print(path, np.isfinite(out["dn_h"][g]).mean().round(3), np.isfinite(out["dn_a"][g]).mean().round(3))


if __name__ == "__main__":
    main()
