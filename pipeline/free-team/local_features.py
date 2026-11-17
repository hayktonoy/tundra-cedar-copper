"""Ball-local features, per team in pairs, for the co-trained local view: .local/local.npz.

    python local_features.py

Only what happens at the ball, nothing about the teams' shape: for each team, the nearest
detected player's contact with the ball in the tactical and the broadcast view (pixel distance to
the feet and the body per box height, ground distance, height on the body, jersey confidence,
crowding), the same from the provided tracking (nearest player in metres, carrying), and the
ball's motion: speed before / after, kink, and its velocity towards each team's attacking goal
before and after (a kick goes forward). Short windows (0.25-1 s, past and future separately).
No labels.
"""
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from base import LOCAL, N, PITCH, fo

BALL = fo.BALL_LANE
LONG = os.environ.get("LONG", "") == "1"     # add 2-4 s contact and ball-travel context


def nanmean_w(x, w, origin=0):
    ok = np.isfinite(x)
    s = uniform_filter1d(np.where(ok, x, 0.0), w, mode="nearest", origin=origin)
    c = uniform_filter1d(ok.astype(float), w, mode="nearest", origin=origin)
    return np.where(c > 1e-6, s / np.maximum(c, 1e-9), np.nan)


def past(x, w, fn=uniform_filter1d):
    """Statistic over x[t - w + 1 .. t]."""
    return fn(x, w, mode="nearest", origin=(w - 1) // 2)


def future(x, w, fn=uniform_filter1d):
    """Statistic over x[t .. t + w - 1]."""
    return fn(x, w, mode="nearest", origin=-(w // 2))


def main():
    t = np.load(os.path.join(BALL, "track_tactical.npz"))
    c = np.load(os.path.join(BALL, "contact_tactical.npz"))
    cb = np.load(os.path.join(LOCAL, "contact_broadcast.npz"))
    z3 = np.load(os.path.join(BALL, "ball3d.npz"))
    fz = np.load(os.path.join(LOCAL, "features.npz"))
    frames, sgn_f = fz["frames"], fz["sgn"].astype(float)
    sgn = np.zeros((N, 2))
    sgn[frames] = sgn_f
    iP = {k: i for i, k in enumerate(fz["names_pair"])}
    src = t["source"]
    bx = np.where(src > 0, t["x"], np.nan).astype(float)
    by = np.where(src > 0, t["y"], np.nan).astype(float)
    k = 4
    vbx = (bx - np.roll(bx, k)) / k * 25
    vby = (by - np.roll(by, k)) / k * 25
    vax = (np.roll(bx, -k) - bx) / k * 25
    vay = (np.roll(by, -k) - by) / k * 25
    for v in (vbx, vby, vax, vay):
        v[np.abs(v) > 45] = np.nan
    S = {}                                  # team-free ball motion
    sb, sa = np.hypot(vbx, vby), np.hypot(vax, vay)
    S["sp_b"], S["sp_a"] = sb, sa
    S["kink"] = np.hypot(vax - vbx, vay - vby)
    S["kink_max6"] = maximum_filter1d(np.nan_to_num(S["kink"], nan=0), 13)
    S["det"] = (src == 1).astype(float)
    S["det_w13"] = uniform_filter1d(S["det"], 13)
    S["h"] = z3["zs"].astype(float) if "zs" in z3.files else z3["z"].astype(float)
    S["bc_seen"] = np.isfinite(cb["foot_h"]).astype(float) + np.isfinite(cb["foot_a"]).astype(float)
    T = {}
    for kk, nm in ((0, "h"), (1, "a")):
        s_ = sgn[:, kk]
        T[f"vfb@{kk}"] = vbx * s_            # ball velocity towards team's attacking goal, before / after
        T[f"vfa@{kk}"] = vax * s_
        T[f"vfa_w13@{kk}"] = future(np.nan_to_num(vax * s_, nan=0), 13)
        T[f"vfa_w38@{kk}"] = future(np.nan_to_num(vax * s_, nan=0), 38)
        T[f"vfb_w13@{kk}"] = past(np.nan_to_num(vbx * s_, nan=0), 13)
        if LONG:
            T[f"vfa_w75@{kk}"] = future(np.nan_to_num(vax * s_, nan=0), 75)
            T[f"vfb_w50@{kk}"] = past(np.nan_to_num(vbx * s_, nan=0), 50)
        for view, cc in (("t", c), ("b", cb)):
            foot = cc[f"foot_{nm}"].astype(float)
            dn = cc[f"dn_{nm}"].astype(float)
            dg = cc[f"dg_{nm}"].astype(float)
            T[f"{view}foot@{kk}"] = foot
            T[f"{view}dn@{kk}"] = dn
            T[f"{view}dg@{kk}"] = dg
            T[f"{view}rel@{kk}"] = cc[f"rel_{nm}"].astype(float)
            T[f"{view}p@{kk}"] = cc[f"p_{nm}"].astype(float)
            T[f"{view}n@{kk}"] = cc[f"n_{nm}"].astype(float)
            T[f"{view}dn2@{kk}"] = cc[f"dn2_{nm}"].astype(float)
            f9 = np.nan_to_num(foot, nan=9)
            d9 = np.nan_to_num(dn, nan=9)
            g9 = np.nan_to_num(dg, nan=30)
            T[f"{view}foot_min6@{kk}"] = minimum_filter1d(f9, 13)
            T[f"{view}dn_min6@{kk}"] = minimum_filter1d(d9, 13)
            T[f"{view}dg_min6@{kk}"] = minimum_filter1d(g9, 13)
            near = ((f9 < 0.5) & (g9 < 2.0)).astype(float)
            T[f"{view}near_p12@{kk}"] = past(near, 12)
            T[f"{view}near_f12@{kk}"] = future(near, 12)
            T[f"{view}near_p25@{kk}"] = past(near, 25)
            T[f"{view}near_f25@{kk}"] = future(near, 25)
            if LONG:
                for w in (50, 100):
                    T[f"{view}near_p{w}@{kk}"] = past(near, w)
                    T[f"{view}near_f{w}@{kk}"] = future(near, w)
        X0 = fz["X0"] if kk == 0 else fz["X1"]
        for nmf in ("d1_", "d2_", "c5_", "carry"):
            v = np.full(N, np.nan)
            v[frames] = X0[:, iP[nmf]]
            T[f"tr_{nmf}@{kk}"] = v
            T[f"tr_{nmf}_w13@{kk}"] = nanmean_w(v, 13)
    pair = sorted({key.split("@")[0] for key in T})
    Xs = np.stack([S[k][frames] for k in sorted(S)], 1).astype(np.float32)
    X0 = np.stack([T[f"{p}@0"][frames] for p in pair], 1).astype(np.float32)
    X1 = np.stack([T[f"{p}@1"][frames] for p in pair], 1).astype(np.float32)
    np.savez_compressed(os.path.join(LOCAL, "local_long.npz" if LONG else "local.npz"), frames=frames, Xs=Xs, X0=X0, X1=X1,
                        names_s=np.array(sorted(S)), names_pair=np.array(pair))
    print("local features:", Xs.shape[1], "ball,", X0.shape[1], "per-team pairs")


if __name__ == "__main__":
    main()
