"""Per-frame features for the free-owner lane, no labels: .local/features.npz.

    python features.py

Inputs: the provided tracking (positions, speeds, team per slot), hc-ball's video ball track
(ground point, speed, detected/interpolated), its two-view height and its contact measures (ball
pixel to the nearest Barca / Athletic box from touches-video's jersey-colour players), and Smart
Tagging. Every feature is either team-free (status) or comes as a pair (team 0, team 1; key `base@k`) in each
team's own attacking frame, so a team model can be made team-equivariant by swapping the pair.

Attacking direction per half: a team's mean x over the half lies nearer its own goal (keeper
and back line), so the team with the lower mean x defends the left goal.
"""
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

from common import BALL_LANE, EXP, HALVES, LOCAL, N, PITCH, ROOT, graded, tags, track_teams, tracking

HOMOGRAPHY = os.path.join(EXP, "hc-video", ".local", "homography_{}.npy")   # official CSVs as (n, 3, 3)
SYNC = os.path.join(ROOT, "calibration", "broadcast_sync.json")          # algorithmic ball-track sync

POSS_CODES = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL", "BALL IN FINAL THIRD",
              "BALL IN THE BOX", "CREATING CHANCES", "PLAYERS IN THE BOX"}
DEF_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
             "DEFENSIVE TRANSITION"}
KINDS = {"Throw-in": 1, "Corner Kick": 2, "Goal Kick": 3, "Free Kick": 4, "Kick Off (After Goal)": 5,
         "Kick Off (Start)": 5}


def nanmean_w(x, w):
    """Centred moving mean over w frames ignoring NaN (NaN where the window has no value)."""
    ok = np.isfinite(x)
    s = uniform_filter1d(np.where(ok, x, 0.0), w, mode="nearest")
    c = uniform_filter1d(ok.astype(float), w, mode="nearest")
    return np.where(c > 1e-6, s / np.maximum(c, 1e-9), np.nan)


def shifted(x, k):
    """y[t] = x[t + k] (NaN beyond the ends)."""
    out = np.full(x.shape, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def attack_sign(P, teams):
    """(N, 2): +1 where the team attacks towards x = 105 m."""
    s = np.zeros((N, 2))
    for a, b in HALVES:
        m0 = np.nanmean(P[a:b, teams == 0, 0])
        m1 = np.nanmean(P[a:b, teams == 1, 0])
        s[a:b, 0] = 1 if m0 < m1 else -1
        s[a:b, 1] = -s[a:b, 0]
    s[:HALVES[0][0]] = s[HALVES[0][0]]
    s[HALVES[0][1]:HALVES[1][0]] = s[HALVES[0][1] - 1]
    s[HALVES[1][1]:] = s[HALVES[1][1] - 1]
    return s


def broadcast_index():
    """Broadcast frame shown at each tactical frame (-1 where the sync does not cover it)."""
    sync = json.load(open(SYNC))
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync["halves"]:
        m = (T >= h["from"]) & (T < h["to"])
        B[m] = np.round((T[m] - h["a"]) / (1 + h["b"])).astype(int)
    return B


def view_width(Hm):
    """Metres of pitch spanned by the image's middle row, and the pitch point at the image centre."""
    Hi = np.linalg.inv(np.where(np.isfinite(Hm), Hm, np.eye(3)))
    pts = np.array([[0, 540, 1], [1920, 540, 1], [960, 540, 1]], float)
    v = np.einsum("nij,kj->nki", Hi, pts)
    p = v[..., :2] / v[..., 2:] * np.array(PITCH)
    ok = np.isfinite(Hm[:, 0, 0]) & (v[..., 2] > 0).all(1)
    w = np.where(ok, np.hypot(*(p[:, 1] - p[:, 0]).T), np.nan)
    return w, np.where(ok[:, None], p[:, 2], np.nan)


def camera_features():
    """Broadcast off the pitch (replays, close-ups, crowd) and zoom / pan of both cameras."""
    F = {}
    B = broadcast_index()
    Hb = np.load(HOMOGRAPHY.format("broadcast"))
    ok = (B >= 0) & (B < len(Hb))
    wb, _ = view_width(Hb)
    off = np.full(N, np.nan)
    off[ok] = ~np.isfinite(Hb[B[ok], 0, 0])
    width_b = np.full(N, np.nan)
    width_b[ok] = wb[B[ok]]
    F["bc_off"] = off
    for w in (25, 75, 250):
        F[f"bc_off_w{w}"] = nanmean_w(off, w)
    F["bc_off_past50"] = shifted(nanmean_w(off, 50), -25)
    F["bc_off_fut50"] = shifted(nanmean_w(off, 50), 25)
    F["bc_width"] = np.log(np.clip(width_b, 1, 500))
    F["bc_width_w75"] = nanmean_w(F["bc_width"], 75)
    wt, ct = view_width(np.load(HOMOGRAPHY.format("tactical")))
    F["tac_width"] = np.log(np.clip(wt, 1, 500))
    F["tac_width_w75"] = nanmean_w(F["tac_width"], 75)
    pan = np.r_[0, np.hypot(*np.diff(ct, axis=0).T)] * 25
    F["tac_pan_w25"] = nanmean_w(np.where(pan < 100, pan, np.nan), 25)
    F["tac_pan_w75"] = nanmean_w(np.where(pan < 100, pan, np.nan), 75)
    return F


def tag_arrays():
    tag_team = np.zeros((N, 2))
    setp = np.zeros(N)
    kind = np.zeros(N)
    for e in tags():
        a, b = max(0, e["start_frame"]), min(N, e["end_frame"])
        team = e["tags"].get("Team", "N/A")
        team = team[0] if isinstance(team, list) else team
        k = 0 if team == "Barcelona" else 1 if str(team).startswith("At") else -1
        if e["code"] == "SET PIECES":
            setp[a:b] = 1
            ty = e["tags"].get("Type", "")
            ty = ty[0] if isinstance(ty, list) else ty
            kind[a:b] = KINDS.get(ty, 6)
        if k < 0:
            continue
        if e["code"] in POSS_CODES or e["code"] == "SET PIECES":
            tag_team[a:b, k] += 1
        elif e["code"] in DEF_CODES:
            tag_team[a:b, 1 - k] += 1
    return tag_team, setp, kind


def main():
    P, _ = tracking()
    teams = track_teams()
    t = np.load(os.path.join(BALL_LANE, "track_tactical.npz"))
    c = np.load(os.path.join(BALL_LANE, "contact_tactical.npz"))
    z = np.load(os.path.join(BALL_LANE, "ball3d.npz"))
    sgn = attack_sign(P, teams)
    F = {}
    # ---- ball (team-free)
    bx, by = t["x"].astype(np.float64), t["y"].astype(np.float64)
    src = t["source"]
    det = (src == 1).astype(float)
    F["ball_det"] = det
    F["ball_any"] = (src > 0).astype(float)
    for w in (25, 75, 250):
        F[f"ball_det_w{w}"] = uniform_filter1d(det, w)
    bsp = np.where(src > 0, t["speed"], np.nan).astype(np.float64)
    bsp = np.where(bsp > 45, np.nan, bsp)
    F["ball_speed"] = bsp
    for w in (13, 50, 125):
        F[f"ball_speed_w{w}"] = nanmean_w(bsp, w)
    F["ball_speed_past50"] = shifted(nanmean_w(bsp, 50), -25)
    F["ball_speed_fut50"] = shifted(nanmean_w(bsp, 50), 25)
    still = np.where(np.isfinite(bsp), (bsp < 1.0).astype(float), np.nan)
    F["ball_still_w50"] = nanmean_w(still, 50)
    F["ball_still_w125"] = nanmean_w(still, 125)
    bxf = nanmean_w(np.where(src > 0, bx, np.nan), 13)
    byf = nanmean_w(np.where(src > 0, by, np.nan), 13)
    out = np.maximum(np.maximum(-bxf, bxf - PITCH[0]), np.maximum(-byf, byf - PITCH[1]))
    F["ball_out"] = out
    F["ball_out_w50"] = nanmean_w(out, 50)
    F["ball_edge_x"] = np.minimum(bxf, PITCH[0] - bxf)
    F["ball_edge_y"] = np.minimum(byf, PITCH[1] - byf)
    F["ball_centre"] = np.hypot(bxf - PITCH[0] / 2, byf - PITCH[1] / 2)
    # where a still ball lies: 1 corner, 2 goal area, 3 touchline (or outside), 4 centre spot, 5 elsewhere
    ex, ey = F["ball_edge_x"], F["ball_edge_y"]
    zone = np.full(N, 5.0)
    zone[np.hypot(ex, ey) < 2.5] = 1
    zone[(ex < 6.5) & (np.abs(byf - PITCH[1] / 2) < 10.5) & (zone == 5)] = 2
    zone[((ey < 1.5) | (ex < 0) | (ey < 0)) & (zone == 5)] = 3
    zone[(F["ball_centre"] < 2.0) & (zone == 5)] = 4
    zone[~np.isfinite(bxf)] = np.nan
    F["ball_zone"] = zone
    hz = z["zs"].astype(np.float64) if "zs" in z.files else z["z"].astype(np.float64)
    F["ball_h"] = hz
    F["ball_h_w25"] = nanmean_w(hz, 25)
    kink = np.where(c["kink"] < 45, c["kink"], np.nan).astype(np.float64)
    F["kink"] = kink
    F["kink_w25"] = nanmean_w(kink, 25)
    # ---- players (team-free)
    sp = P[..., 2].astype(np.float64)
    spn = np.nan_to_num(sp, nan=0)
    top5 = -np.sort(-spn, 1)[:, :5].mean(1)
    n_pl = np.isfinite(sp).sum(1).astype(float)
    with np.errstate(all="ignore"):
        meansp = np.nanmean(sp, 1)
    F["n_players"] = n_pl
    F["n_players_w75"] = uniform_filter1d(n_pl, 75)
    for w in (1, 25, 75, 250):
        F[f"top5_w{w}"] = uniform_filter1d(top5, w) if w > 1 else top5
        F[f"meansp_w{w}"] = nanmean_w(meansp, w) if w > 1 else meansp
    F["top5_past75"] = shifted(uniform_filter1d(top5, 75), -50)
    F["top5_fut75"] = shifted(uniform_filter1d(top5, 75), 50)
    F["top5_d"] = F["top5_fut75"] - F["top5_past75"]
    F.update(camera_features())
    # ---- tags
    tag_team, setp, kind = tag_arrays()
    F["setpiece"] = setp
    F["setpiece_kind"] = kind
    # ---- per-team pairs, in each team's attacking frame
    bvx = (shifted(bxf, 3) - shifted(bxf, -3)) / 6 * 25
    bvy = (shifted(byf, 3) - shifted(byf, -3)) / 6 * 25
    T = {}
    for k in (0, 1):
        m = teams == k
        x = P[:, m, 0].astype(np.float64)
        y = P[:, m, 1].astype(np.float64)
        s = sp[:, m]
        ax = (x - PITCH[0] / 2) * sgn[:, k:k + 1]
        vx = (shifted(x, 3) - shifted(x, -3)) / 6 * 25 * sgn[:, k:k + 1]
        cnt = np.isfinite(x).sum(1).astype(float)
        with np.errstate(all="ignore"):
            T[f"n@{k}"] = cnt
            T[f"cx@{k}"] = np.nanmean(ax, 1)
            T[f"sx@{k}"] = np.nanstd(ax, 1)
            T[f"sy@{k}"] = np.nanstd(y, 1)
            T[f"front@{k}"] = np.nanmax(ax, 1)
            T[f"back@{k}"] = np.nanmin(np.where(ax > -PITCH[0] / 2 + 8, ax, np.nan), 1)
            T[f"vx@{k}"] = np.nanmean(np.clip(vx, -12, 12), 1)
            T[f"sp@{k}"] = np.nanmean(s, 1)
        d = np.hypot(x - bxf[:, None], y - byf[:, None])
        dn = np.where(np.isfinite(d), d, np.inf)
        ds = np.sort(dn, 1)
        T[f"d1_@{k}"] = np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan)
        T[f"d2_@{k}"] = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan)
        T[f"c5_@{k}"] = (dn < 5).sum(1).astype(float)
        T[f"c15_@{k}"] = (dn < 15).sum(1).astype(float)
        # pressing: how fast this team's three nearest players close on the ball (m/s, + = closing)
        close = -(shifted(d, 3) - shifted(d, -3)) / 6 * 25
        order = np.argsort(dn, 1)[:, :3]
        cl3 = np.take_along_axis(close, order, 1)
        ok3 = np.take_along_axis(np.isfinite(dn), order, 1)
        with np.errstate(all="ignore"):
            T[f"press@{k}"] = np.nanmean(np.where(ok3 & (np.abs(cl3) < 12), cl3, np.nan), 1)
            T[f"press1@{k}"] = np.where(ok3[:, 0] & (np.abs(cl3[:, 0]) < 12), cl3[:, 0], np.nan)
        # carrying: the ball moves with this team's nearest player (m/s of relative velocity)
        j = order[:, 0]
        r_ = np.arange(N)
        pvx = (shifted(x, 3) - shifted(x, -3))[r_, j] / 6 * 25
        pvy = (shifted(y, 3) - shifted(y, -3))[r_, j] / 6 * 25
        T[f"carry@{k}"] = np.where(ok3[:, 0] & (ds[:, 0] < 3), np.hypot(pvx - bvx, pvy - bvy), np.nan)
        bax = (bxf - PITCH[0] / 2) * sgn[:, k]
        T[f"goalside@{k}"] = np.sum(np.isfinite(ax) & (ax > bax[:, None]), 1) / np.maximum(cnt, 1)
        T[f"tag@{k}"] = tag_team[:, k]
        T[f"bax@{k}"] = bax
        nm = "ha"[k]
        T[f"foot@{k}"] = c[f"foot_{nm}"].astype(np.float64)
        T[f"dn@{k}"] = c[f"dn_{nm}"].astype(np.float64)
        T[f"rel@{k}"] = c[f"rel_{nm}"].astype(np.float64)
        T[f"nv@{k}"] = c[f"n_{nm}"].astype(np.float64)
    for k in (0, 1):
        for base, ws in (("cx", (75, 250)), ("vx", (25, 75, 250)), ("sx", (75,)), ("d1_", (13, 50)),
                         ("foot", (13, 50)), ("goalside", (75,)), ("c5_", (25,)), ("bax", (75, 250)),
                         ("press", (25, 75)), ("press1", (13, 50)), ("carry", (13, 50))):
            for w in ws:
                T[f"{base}@{k}_w{w}"] = nanmean_w(T[f"{base}@{k}"], w)
        for base in ("vx", "d1_", "foot", "press"):
            T[f"{base}@{k}_past"] = shifted(nanmean_w(T[f"{base}@{k}"], 50), -25)
            T[f"{base}@{k}_fut"] = shifted(nanmean_w(T[f"{base}@{k}"], 50), 25)
    # pair names: the team index is the first digit after the base name
    pair = [tuple(key.split("@0")) for key in T if "@0" in key]
    frames = np.flatnonzero(graded())
    names_s = sorted(F)
    Xs = np.stack([F[k][frames] for k in names_s], 1).astype(np.float32)
    X0 = np.stack([T[f"{a}@0{b}"][frames] for a, b in pair], 1).astype(np.float32)
    X1 = np.stack([T[f"{a}@1{b}"][frames] for a, b in pair], 1).astype(np.float32)
    np.savez_compressed(os.path.join(LOCAL, "features.npz"), frames=frames, Xs=Xs, X0=X0, X1=X1,
                        names_s=np.array(names_s), names_pair=np.array([f"{a}{b}" for a, b in pair]),
                        sgn=sgn[frames].astype(np.int8))
    print("features:", Xs.shape[1], "status,", X0.shape[1], "per-team pairs;", len(frames), "frames")


if __name__ == "__main__":
    main()
