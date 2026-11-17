"""Shared paths, and loaders for the free-owner lane. Arrays are indexed by tactical frame."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
EXP = os.path.join(ROOT, "pipeline")
BALL_LANE = os.path.join(EXP, "hc-ball", ".local")          # label-free ball track, contact, 3D
TRACKING = os.path.join(EXP, "possession-engine", ".local", "tracking.npz")   # parsed ATD
sys.path.insert(0, os.path.join(ROOT, "official", "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "official"))
sys.path.insert(0, os.path.join(ROOT, "official", "starter"))
import possession_metrics as pm  # noqa: E402

N = 199319
FPS = 25
PITCH = (105.3, 68.0)
HALVES = [(11185, 90652), (110808, 190198)]
BLOCKS = [(11185, 50919), (50919, 90652), (110808, 150503), (150503, 190198)]
H, A, D, X = 0, 1, 2, 3


def graded():
    g = np.zeros(N, bool)
    for a, b in HALVES:
        g[a:b] = True
    return g


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def write(path, pred):
    names = np.array(list("HADX"))
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{c}\n" for i, c in enumerate(names[pred]))


def within(c, a, b):
    out = c.copy()
    out[:a] = X
    out[b:] = X
    return out


def runs(c):
    c = np.asarray(c)
    cut = np.flatnonzero(np.diff(c)) + 1
    st = np.r_[0, cut]
    en = np.r_[cut, len(c)]
    return [(int(a), int(b), int(c[a])) for a, b in zip(st, en)]


def tracking():
    z = np.load(TRACKING)
    return z["players"], z["ball"]


def track_teams(source="tactical"):
    import xml.etree.ElementTree as ET
    from games import asset_paths
    root = ET.parse(asset_paths(source=source)["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def tags():
    from load_smart_tagging import load
    return load()
