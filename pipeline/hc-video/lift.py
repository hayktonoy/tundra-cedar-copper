"""Ball height by triangulating the tactical and broadcast detections.

Each camera's ground-plane projection g of the ball lies on the ray from the camera centre C
through the ball. Camera centres are fixed stadium mounts, estimated from the homographies in
pipeline/multiview-ball/ (tactical (53.36, 115.52, 29.97), broadcast (52.88, 113.08, 21.33) m).
The ball is the midpoint of the shortest segment between the two rays.

Writes .local/ball_3d.npz: x, y, z (metres; NaN unless both views detect the ball), gap (ray
distance, a quality measure), and zs (height smoothed over 5 frames).
"""
import os

import numpy as np
import pandas as pd

from common import LOCAL, N

C_T = np.array([53.36, 115.52, 29.97])
C_B = np.array([52.88, 113.08, 21.33])


def triangulate(gt, gb):
    """gt, gb (n,2) ground projections in each view -> (n,3) points and (n,) ray gaps."""
    d1 = np.c_[gt, np.zeros(len(gt))] - C_T
    d2 = np.c_[gb, np.zeros(len(gb))] - C_B
    w0 = C_T - C_B
    a, b, c = (d1 * d1).sum(1), (d1 * d2).sum(1), (d2 * d2).sum(1)
    d, e = (d1 * w0).sum(1), (d2 * w0).sum(1)
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1 = C_T + s[:, None] * d1
    p2 = C_B + t[:, None] * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2, axis=1)


def main():
    v = np.load(os.path.join(LOCAL, "ball_video.npz"))
    b = np.load(os.path.join(LOCAL, "ball_broadcast.npz"))
    ok = (v["source"] == 1) & (b["source"] == 1)
    p, gap = triangulate(np.c_[v["x"], v["y"]][ok], np.c_[b["x"], b["y"]][ok])
    out = {k: np.full(N, np.nan, np.float32) for k in ("x", "y", "z", "gap")}
    out["x"][ok], out["y"][ok], out["z"][ok], out["gap"][ok] = p[:, 0], p[:, 1], p[:, 2], gap
    bad = (out["gap"] > 1.5) | (out["z"] < -1) | (out["z"] > 25)
    for k in ("x", "y", "z"):
        out[k][bad] = np.nan
    out["zs"] = pd.Series(out["z"]).rolling(5, center=True, min_periods=2).median().to_numpy(np.float32)
    np.savez_compressed(os.path.join(LOCAL, "ball_3d.npz"), **out)
    z = out["z"][~np.isnan(out["z"])]
    print("both views:", ok.sum(), "kept:", len(z), "height percentiles 5/25/50/75/95:", np.percentile(z, [5, 25, 50, 75, 95]).round(2))


if __name__ == "__main__":
    main()
