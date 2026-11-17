"""Shared paths, and loaders for the hc-video experiment."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
sys.path.insert(0, os.path.join(ROOT, "official", "evaluation"))
import possession_metrics as pm  # noqa: E402

N = 199319
FPS = 25
PITCH = (105.3, 68.0)
HALVES = [(11185, 90652), (110808, 190198)]
BLOCKS = [(11185, 50919), (50919, 90652), (110808, 150503), (150503, 190198)]
TACTICAL = os.path.join(ROOT, "official", "data", "tactical")
BROADCAST = os.path.join(ROOT, "official", "data", "broadcast")
TRACKING = os.path.join(ROOT, "pipeline", "possession-engine", ".local", "tracking.npz")


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def write(path, pred):
    names = np.array(list("HADX"))
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{c}\n" for i, c in enumerate(names[pred]))


def block_of(frame):
    for k, (a, b) in enumerate(BLOCKS):
        if a <= frame < b:
            return k
    return -1


def players_teams():
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(TACTICAL, "BAR-ATH_tactical_atd.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def homographies(source="tactical"):
    import pandas as pd
    path = os.path.join(TACTICAL if source == "tactical" else BROADCAST, f"BAR-ATH_{source}_homography.csv")
    cache = os.path.join(LOCAL, f"homography_{source}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    df = pd.read_csv(path)
    H = df.iloc[:, 1:10].to_numpy(np.float64).reshape(-1, 3, 3)
    np.save(cache, H)
    return H
