"""Shared loaders for the free-brief lane. Arrays are indexed by tactical frame.

The decoder is free-pretrained's (free-status's Engine plus the spotter's live2/team2 terms),
reused unchanged; this lane only adds terms to its scores.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
LOCAL = HERE / ".local"
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0, str(EXP / "free-pretrained"))
import fuse  # noqa: E402  (puts free-status and free-owner on sys.path)
from common import BLOCKS, HALVES, N, X, codes, write, runs  # noqa: E402,F401  free-owner's
from decode import decode, posterior  # noqa: E402,F401

H, A, D = 0, 1, 2


def ball():
    """hc-ball's tactical and broadcast tracks (label-free detection + physics tracker)."""
    t = np.load(EXP / "hc-ball/.local/track_tactical.npz")
    b = np.load(EXP / "hc-ball/.local/track_broadcast.npz")
    return t, b


def tracking():
    z = np.load(EXP / "possession-engine/.local/tracking.npz")
    return z["players"], z["ball"]


def teams():
    sys.path.insert(0, str(ROOT / "official" / "starter"))
    import xml.etree.ElementTree as ET
    from games import asset_paths
    root = ET.parse(asset_paths(source="tactical")["atd_meta"]).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == ids[0] else 1 for p in root.iter("Player")], np.int8)


def stitched(ev, folds, extra=None):
    """Held-out prediction: block k decoded with fold k's constants (+ `extra` params)."""
    out = np.full(N, 2, dtype="i1")
    for (a, b), p in zip(BLOCKS, folds):
        q = dict(p, **(extra or {}))
        out[a:b] = ev.run(q)[a:b]
    return out
