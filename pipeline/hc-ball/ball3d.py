"""Ball in 3D from the two views' tracks, and the ball arrays the touch pipeline reads.

    python ball3d.py [--tactical .local/track_tactical.npz] [--broadcast .local/track_broadcast.npz]

Both cameras sit on fixed mounts, so each view's ground point lies on the ray from its camera
centre through the ball; the ball is the midpoint of the shortest segment between the two rays,
kept when the rays pass within 1.5 m of each other. Uses detected or interpolated positions in
both views (interpolation only over physically linked gaps of <= 20 frames).

Writes .local/ball3d.npz: x, y, z (metres), gap (ray distance), zs (5-frame median height).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import LOCAL, N, triangulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tactical", default=os.path.join(LOCAL, "track_tactical.npz"))
    ap.add_argument("--broadcast", default=os.path.join(LOCAL, "track_broadcast.npz"))
    ap.add_argument("--out", default=os.path.join(LOCAL, "ball3d.npz"))
    a = ap.parse_args()
    t, b = np.load(a.tactical), np.load(a.broadcast)
    ok = (t["source"] > 0) & (b["source"] > 0) & np.isfinite(t["x"]) & np.isfinite(b["x"])
    p, gap = triangulate(np.c_[t["x"], t["y"]][ok].astype(np.float64), np.c_[b["x"], b["y"]][ok].astype(np.float64))
    out = {k: np.full(N, np.nan, np.float32) for k in ("x", "y", "z", "gap")}
    out["x"][ok], out["y"][ok], out["z"][ok], out["gap"][ok] = p[:, 0], p[:, 1], p[:, 2], gap
    bad = (out["gap"] > 1.5) | (out["z"] < -1) | (out["z"] > 25)
    for k in ("x", "y", "z"):
        out[k][bad] = np.nan
    out["zs"] = pd.Series(out["z"]).rolling(5, center=True, min_periods=2).median().to_numpy(np.float32)
    np.savez_compressed(a.out, **out)
    z = out["z"][np.isfinite(out["z"])]
    print(a.out, "both views:", int(ok.sum()), "kept:", len(z), "height pct 5/25/50/75/95:",
          np.percentile(z, [5, 25, 50, 75, 95]).round(2))


if __name__ == "__main__":
    main()
