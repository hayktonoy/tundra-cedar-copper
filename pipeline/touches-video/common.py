"""Paths and loaders for the touch detector. Arrays are indexed by tactical frame (199,319)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
VIDEO_LANE = os.path.join(ROOT, "pipeline", "hc-video", ".local")
TACTICAL = os.path.join(ROOT, "official", "data", "tactical")
TRACKING = os.path.join(ROOT, "pipeline", "possession-engine", ".local", "tracking.npz")
BASE = os.path.join(ROOT, "pipeline", "hc-fusion", ".local", "best.csv")
sys.path.insert(0, os.path.join(ROOT, "official", "evaluation"))
import possession_metrics as pm  # noqa: E402

N = 199319
FPS = 25
HALVES = [(11185, 90652), (110808, 190198)]
H, A, D, X = 0, 1, 2, 3
PITCH = (105.3, 68.0)


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def write(path, pred):
    names = np.array(list("HADX"))
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{c}\n" for i, c in enumerate(names[pred]))


def players_teams():
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(TACTICAL, "BAR-ATH_tactical_atd.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def keeper_slots():
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(TACTICAL, "BAR-ATH_tactical_atd.xml")).getroot()
    out = []
    for i, p in enumerate(root.iter("Player")):
        for par in p.iter("ProviderParameter"):
            if par.findtext("Name") == "position_type" and par.findtext("Value") == "Goalkeeper":
                out.append(i)
    return out


def defends_left_goal_is_barca(frames):
    """Barca defends the left goal in the first half and the right goal in the second."""
    return np.asarray(frames) < (HALVES[0][1] + HALVES[1][0]) // 2


def load_players():
    """Tracked player positions (n,46,2) and each slot's team. The provider's two keeper slots can
    swap or both sit at one goal. A keeper slot belongs to the
    team defending the goal on its side. Barca's keeper slot gets the keeper on Barca's side,
    Athletic's the one on Athletic's side (the one nearer the goal line if both are on one side)."""
    z = np.load(TRACKING)
    pl = z["players"][..., :2].astype(np.float32)
    teams = players_teams()
    ks = keeper_slots()
    kh = [k for k in ks if teams[k] == 0]
    ka = [k for k in ks if teams[k] == 1]
    if len(kh) == 1 and len(ka) == 1:
        h, a = kh[0], ka[0]
        mid = PITCH[0] / 2
        left_h = defends_left_goal_is_barca(np.arange(N))
        P = pl[:, [h, a]].copy()                                    # (N,2,2)
        on_h = np.where(left_h[:, None], P[..., 0] < mid, P[..., 0] >= mid)
        goal_x = np.where(left_h[:, None] == on_h, 0.0, PITCH[0])   # the goal on each one's side
        depth = np.abs(P[..., 0] - goal_x)
        have = np.isfinite(P[..., 0])
        newh = np.full((N, 2), np.nan, np.float32)
        newa = np.full((N, 2), np.nan, np.float32)
        for side, out in ((True, newh), (False, newa)):
            cand = have & (on_h == side)
            dd = np.where(cand, depth, np.inf)
            j = dd.argmin(1)
            ok = np.isfinite(dd.min(1))
            out[ok] = P[np.flatnonzero(ok), j[ok]]
        pl[:, h], pl[:, a] = newh, newa
    return pl, teams


def homographies():
    return np.load(os.path.join(VIDEO_LANE, "homography_tactical.npy"))


def to_pixels(H, xy):
    """H (n,3,3) normalised-pitch -> pixel homographies; xy (n,k,2) metres -> (n,k,2) pixels."""
    p = xy / np.array(PITCH)
    q = np.einsum("nij,nkj->nki", H, np.concatenate([p, np.ones(p.shape[:-1] + (1,))], -1))
    return q[..., :2] / q[..., 2:3]


def load_ball():
    """The video lane's tracks: tactical ground projection, broadcast, and triangulated 3D."""
    v = np.load(os.path.join(VIDEO_LANE, "ball_video.npz"))
    b = np.load(os.path.join(VIDEO_LANE, "ball_broadcast.npz"))
    z = np.load(os.path.join(VIDEO_LANE, "ball_3d.npz"))
    return {"x": v["x"], "y": v["y"], "src": v["source"], "score": v["score"],
            "bx": b["x"], "by": b["y"], "bsrc": b["source"],
            "x3": z["x"], "y3": z["y"], "z": z["z"], "zs": z["zs"], "gap": z["gap"]}
