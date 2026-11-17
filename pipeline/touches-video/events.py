"""Touch candidates and their local features.

    python events.py          # reads .local/frames.npz (+ .local/pixel.npz if present), writes .local/events.parquet

A candidate is a frame where the ball's velocity changes: a local maximum (+-3 frames) of the
hinge velocity change in either view, at least 1.5 m/s. Every feature is local (a few frames
around the candidate) and describes the ball and the players near it; nothing about which team
has had the ball, and no position on the pitch, so a model fitted on it can only learn who is
touching the ball, not who usually has it.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from common import HALVES, LOCAL, N, ROOT, load_ball, load_players
from features import ray_contact

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

MIN_KINK = 1.5
MAX_KINK = 45.0     # above this the "kink" is the detector jumping to another object


def kink(f, v, raw=False):
    k = np.hypot(f[f"{v}_dvx"], f[f"{v}_dvy"])
    return k if raw else np.where(k > MAX_KINK, np.nan, k)


def window(a, t, lo, hi, fn=np.nanmin):
    """fn over a[t+lo .. t+hi] for every candidate t."""
    idx = t[:, None] + np.arange(lo, hi + 1)[None]
    idx = np.clip(idx, 0, N - 1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(a[idx], axis=1)


def candidates(f, src_t, src_b):
    K = np.fmax(kink(f, "t"), kink(f, "b"))
    # inside the detector's gaps only the wide hinge exists
    Kw = np.fmax(kink(f, "tw"), kink(f, "bw"))
    K = np.where(np.isfinite(K), K, Kw)
    Kz = np.nan_to_num(K)
    live = np.zeros(N, bool)
    for a, b in HALVES:
        live[a:b] = True
    pk = (Kz == maximum_filter1d(Kz, 7)) & (Kz >= MIN_KINK) & live
    t = np.flatnonzero(pk)
    # flat plateaus give several equal maxima: keep the first
    t = t[np.r_[True, np.diff(t) > 3]]
    # the velocity change is large for a few frames around a sharp kick; the touch is where the
    # hinge explains the motion best (its gain over a smooth curve peaks at the true kink)
    G = np.fmax(np.nan_to_num(f["t_gain"], nan=-1), np.nan_to_num(f["b_gain"], nan=-1))
    idx = np.clip(t[:, None] + np.arange(-3, 4)[None], 0, N - 1)
    g = G[idx]
    ok = g.max(1) > 0
    t = np.where(ok, idx[np.arange(len(t)), g.argmax(1)], t)
    t = np.unique(t)
    # the ball arriving at a player (a reception may have no sharp kink): contact distance drops
    # below .6 m after being over 1.5 m in the previous 10 frames
    px = os.path.join(LOCAL, "pixel.npz")
    if os.path.exists(px):
        p = np.load(px)
        d = np.fmin(np.nan_to_num(p["px_d_h"], nan=9), np.nan_to_num(p["px_d_a"], nan=9))
        mf = maximum_filter1d(d, 10)                               # max over t-5 .. t+4
        far_before = np.r_[np.zeros(5, bool), mf[:-5] > 1.5]      # max over t-10 .. t-1
        onset = (d < 0.6) & far_before & live
        on = np.flatnonzero(onset)
        on = on[np.r_[True, np.diff(on) > 5]]
        # skip onsets that already have a kink candidate within 3 frames
        if len(t):
            dist = np.abs(on[:, None] - t[None, :]).min(1)
            on = on[dist > 3]
        t = np.unique(np.r_[t, on])
    # a ball must actually be seen near the touch in at least one view
    seen = ((src_t == 1) | (src_b == 1)).astype(np.float64)
    near = np.convolve(seen, np.ones(9), "same")
    return t[near[t] >= 3]


def build():
    f = dict(np.load(os.path.join(LOCAL, "frames.npz")))
    b = load_ball()
    t = candidates(f, b["src"], b["bsrc"])
    df = pd.DataFrame({"frame": t})
    for v in ("t", "b"):
        K = kink(f, v)
        df[f"k_{v}"] = K[t]
        df[f"kmax_{v}"] = window(K, t, -2, 2, np.nanmax)
        df[f"gain_{v}"] = f[f"{v}_gain"][t]
        vb = np.c_[f[f"{v}_vbx"], f[f"{v}_vby"]]
        va = vb + np.c_[f[f"{v}_dvx"], f[f"{v}_dvy"]]
        sb, sa = np.hypot(*vb.T), np.hypot(*va.T)
        df[f"sb_{v}"], df[f"sa_{v}"] = sb[t], sa[t]
        with np.errstate(all="ignore"):
            cosang = (vb * va).sum(1) / (sb * sa)
        df[f"cos_{v}"] = cosang[t]
        df[f"n_{v}"] = (f[f"{v}_nb"] + f[f"{v}_na"])[t]
    for v in ("tw", "bw"):
        K = kink(f, v)
        df[f"k_{v}"] = K[t]
        df[f"kmax_{v}"] = window(K, t, -3, 3, np.nanmax)
        df[f"gain_{v}"] = f[f"{v}_gain"][t]
    # do the two views see the same velocity change?
    dvt = np.c_[f["t_dvx"], f["t_dvy"]][t]
    dvb = np.c_[f["b_dvx"], f["b_dvy"]][t]
    with np.errstate(all="ignore"):
        df["dv_cos_tb"] = (dvt * dvb).sum(1) / (np.hypot(*dvt.T) * np.hypot(*dvb.T))
    df["k_both"] = np.fmin(df["kmax_t"], df["kmax_b"])
    df["wild"] = window(np.fmax(kink(f, "t", True), kink(f, "b", True)) > MAX_KINK, t, -6, 6, np.nanmean)
    df["gapfrac"] = window((b["src"] != 1).astype(np.float64), t, -6, 6, np.nanmean)
    # set pieces: the ball rested before it was struck (the ball was dead, not a brief possession)
    sp = np.full(N, np.nan)
    xy = np.c_[b["x"], b["y"]].astype(np.float64)
    xy[~np.isin(b["src"], (1, 2))] = np.nan
    sp[1:] = np.hypot(*(xy[1:] - xy[:-1]).T) * 25
    df["rest_before"] = window((sp < 1.0).astype(np.float64) + np.where(np.isnan(sp), np.nan, 0), t, -30, -3, np.nanmean)
    ev = np.load(os.path.join(ROOT, "pipeline", "hc-video", ".local", "evidence.npz"))
    df["restart_kick"] = window(ev["restart_kick"].astype(np.float64), t, -8, 8, np.nanmax)
    # is the ball we see the ball? both views should agree on it, and it should not jump
    both = (b["src"] == 1) & (b["bsrc"] == 1)
    dis = np.where(both, np.hypot(b["x"] - b["bx"], b["y"] - b["by"]), np.nan).astype(np.float64)
    df["view_disagree"] = window(dis, t, -8, 8, np.nanmedian)
    tv = np.isin(b["src"], (1, 2))
    stp = np.full(N, np.nan)
    stp[1:] = np.where(tv[1:] & tv[:-1], np.hypot(np.diff(b["x"]), np.diff(b["y"])), np.nan)
    df["max_step"] = window(stp, t, -8, 8, np.nanmax)
    df["tac_cov"] = window(tv.astype(np.float64), t, -8, 8, np.nanmean)
    df["tac_seen"] = window(tv.astype(np.float64), t, -2, 2, np.nanmean)
    df["bro_cov"] = window(np.isin(b["bsrc"], (1, 2)).astype(np.float64), t, -8, 8, np.nanmean)
    df["src_t"] = b["src"][t]
    df["src_b"] = b["bsrc"][t]
    df["z"] = window(b["z"], t, -3, 3, np.nanmedian)
    df["zmax"] = window(b["z"], t, -6, 6, np.nanmax)
    df["have_z"] = np.isfinite(df["z"]).astype(np.int8)
    df["nviews"] = f["nviews"][t]
    for tag in ("c", "ct", "g"):
        for s in ("h", "a"):
            d = f[f"{tag}_d_{s}"]
            df[f"{tag}_d_{s}"] = d[t]
            df[f"{tag}_dmin_{s}"] = window(d, t, -3, 1)
            df[f"{tag}_dpre_{s}"] = window(d, t, -6, -1, np.nanmean)
            df[f"{tag}_dpost_{s}"] = window(d, t, 1, 6, np.nanmean)
            df[f"{tag}_d2_{s}"] = f[f"{tag}_d2_{s}"][t]
            if tag != "g":
                df[f"{tag}_h_{s}"] = f[f"{tag}_h_{s}"][t]
    audio = os.path.join(ROOT, "pipeline", "hc-audio", ".local", "classical.npz")
    if os.path.exists(audio):
        z = np.load(audio)
        imp = np.where(z["audio_available"], z["impact"], np.nan).astype(np.float64)
        df["impact"] = window(imp, t, -2, 2, np.nanmax)
        df["impact_rel"] = df["impact"] - window(imp, t, -25, 25, np.nanmedian)
    for px in (os.path.join(LOCAL, "pixel.npz"), os.path.join(LOCAL, "pixel_broadcast.npz")):
        if not os.path.exists(px):
            continue
        p = np.load(px)
        for k in p.files:
            a = p[k]
            if k.startswith(("px_", "pb_")):
                df[k] = a[t]
                if "_d_" in k:
                    df[k.replace("_d_", "_dmin_")] = window(a, t, -3, 1)
                    # when is this team closest to the ball, relative to the kink?
                    idx = np.clip(t[:, None] + np.arange(-8, 9)[None], 0, N - 1)
                    w = np.where(np.isfinite(a[idx]), a[idx], np.inf)
                    df[k.replace("_d_", "_tmin_")] = np.where(np.isfinite(w.min(1)), w.argmin(1) - 8, np.nan)
                    df[k.replace("_d_", "_dwin_")] = np.where(np.isfinite(w.min(1)), w.min(1), np.nan)
    kink_point_features(df, f, b)
    df.to_parquet(os.path.join(LOCAL, "events.parquet"))
    print(len(df), "candidates,", df.shape[1], "columns")
    return df


def attack_sign():
    """+1 where Barca attacks towards x = 105 in that half (its players' mean x is the lower)."""
    pl, teams = load_players()
    sign = np.zeros(N, np.int8)
    for a, b in HALVES:
        mh = np.nanmean(pl[a:b:25][:, teams == 0, 0])
        ma = np.nanmean(pl[a:b:25][:, teams == 1, 0])
        sign[a:b] = 1 if mh < ma else -1
    return sign


def contact_at(points_t, points_b, players, team_prob, prefix, df, k):
    """Ray contact from kink points to a set of players (n,P,2), per team (team_prob (n,P,2))."""
    d, h, _ = ray_contact(players, points_t, points_b)
    for j, name in ((0, "h"), (1, "a")):
        dd = np.where((team_prob[..., j] > 0.5) & np.isfinite(d), d, np.inf)
        i1 = dd.argmin(1)
        r = np.arange(len(dd))
        d1 = dd[r, i1]
        col = f"{prefix}_d_{name}"
        val = np.where(np.isinf(d1), np.nan, d1)
        df[col] = np.fmin(df[col], val) if k else val
        if not k:
            df[f"{prefix}_h_{name}"] = h[r, i1]
        n2 = (dd < 2.0).sum(1)
        df[f"{prefix}_n2_{name}"] = np.maximum(df[f"{prefix}_n2_{name}"], n2) if k else n2


def kink_point_features(df, f, b):
    """Contact measured from the fitted kink point (where the before and after lines meet),
    which is where the ball was at the touch even when the detector lost it, and the direction
    of the touch relative to each team's attack."""
    t = df["frame"].to_numpy()
    pick = lambda a, b_: np.where(np.isfinite(a), a, b_).astype(np.float64)
    ktx, kty = pick(f["t_ax"][t], f["tw_ax"][t]), pick(f["t_ay"][t], f["tw_ay"][t])
    kbx, kby = pick(f["b_ax"][t], f["bw_ax"][t]), pick(f["b_ay"][t], f["bw_ay"][t])
    pt = np.c_[np.where(np.isfinite(ktx), ktx, b["x"][t]), np.where(np.isfinite(kty), kty, b["y"][t])]
    pt[~np.isfinite(ktx) & ~np.isin(b["src"][t], (1, 2))] = np.nan
    pb = np.c_[np.where(np.isfinite(kbx), kbx, b["bx"][t]), np.where(np.isfinite(kby), kby, b["by"][t])]
    pb[~np.isfinite(kbx) & ~np.isin(b["bsrc"][t], (1, 2))] = np.nan
    pl, teams = load_players()
    tp = np.stack([teams == 0, teams == 1], -1).astype(np.float32)
    for k, off in enumerate((-2, -1, 0, 1)):
        tt = np.clip(t + off, 0, N - 1)
        contact_at(pt, pb, pl[tt].astype(np.float64), np.broadcast_to(tp, (len(t),) + tp.shape), "kp", df, k)
    for name, prefix in (("boxteam.npz", "kpx"), ("boxteam_broadcast.npz", "kpb")):
        bt = os.path.join(LOCAL, name)
        if not os.path.exists(bt):
            continue
        z = np.load(bt)
        row = np.full(N, -1)
        row[z["frames"]] = np.arange(len(z["frames"]))
        for k, off in enumerate((-2, -1, 0, 1)):
            r = row[np.clip(t + off, 0, N - 1)]
            have = r >= 0
            foot = np.full((len(t),) + z["foot"].shape[1:], np.nan)
            prob = np.zeros((len(t),) + z["foot"].shape[1:2] + (2,), np.float32)
            foot[have] = z["foot"][r[have]]
            prob[have] = np.nan_to_num(z["prob"][r[have]][..., :2])
            contact_at(pt, pb, foot, prob, prefix, df, k)
            if prefix == "kpx":   # tactical ray alone: the ball's offset from the player in the image
                contact_at(pt, np.full_like(pb, np.nan), foot, prob, "kpt", df, k)
            else:                 # broadcast ray alone, broadcast players
                contact_at(np.full_like(pt, np.nan), pb, foot, prob, "kpbb", df, k)
    sgn = attack_sign()[t]
    # velocities of the view with the larger kink, along Barca's attacking direction
    use_t = np.nan_to_num(df["k_t"].to_numpy()) >= np.nan_to_num(df["k_b"].to_numpy())
    for name, key in (("vb", ("vbx",)), ("dv", ("dvx",))):
        vt, vb_ = f[f"t_{key[0]}"][t], f[f"b_{key[0]}"][t]
        df[f"{name}_fwd_h"] = np.where(use_t, vt, vb_) * sgn
    df["va_fwd_h"] = df["vb_fwd_h"] + df["dv_fwd_h"]
    vy = np.where(use_t, f["t_vby"][t], f["b_vby"][t])
    dvy = np.where(use_t, f["t_dvy"][t], f["b_dvy"][t])
    df["vb_side"] = np.abs(vy)
    df["va_side"] = np.abs(vy + dvy)
    # aerial duel: ball at head height with both teams close; who won it shows in the impulse
    hk = "kpx" if "kpx_h_h" in df else "kp"
    zz = np.where(np.isfinite(df["z"]), df["z"], np.fmin(df[f"{hk}_h_h"], df[f"{hk}_h_a"]))
    contested = (df[f"{hk}_d_h"] < 1.5) & (df[f"{hk}_d_a"] < 1.5)
    df["aerial"] = (zz > 1.3).astype(np.int8)
    df["duel_dv_fwd"] = np.where(contested, df["dv_fwd_h"], np.nan)
    df["header_dv_fwd"] = np.where(contested & (zz > 1.3), df["dv_fwd_h"], np.nan)


if __name__ == "__main__":
    build()
