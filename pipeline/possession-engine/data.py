"""Load tactical tracking and Smart Tagging as video-frame arrays.

The parsed tracking is cached in .local/ because reading the 195 MB text file takes a while.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OFFICIAL = os.path.join(REPO, "official")
LOCAL = os.path.join(HERE, ".local")
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.join(OFFICIAL, "starter"))
sys.path.insert(0, os.path.join(OFFICIAL, "evaluation"))

from games import GAMES, asset_paths  # noqa: E402

N_FRAMES = GAMES["BAR-ATH"]["n_video_frames"]
FPS = GAMES["BAR-ATH"]["fps"]
PITCH = GAMES["BAR-ATH"]["pitch"]  # metres (length, width)
HALVES = [(11185, 90652), (110808, 190198)]  # graded, half-open, video frames
KICKOFFS = [11185, 113742]


def track_teams(source="tactical"):
    """0 = home (Barcelona), 1 = away, per track slot. First team in the metadata is home."""
    root = ET.parse(asset_paths(source=source)["atd_meta"]).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def tracking(source="tactical"):
    """(players, ball): players is (N, slots, 3) x/y in metres and speed; ball is (N, 3).

    Indexed by video frame (ATD frame_id - 1). Missing observations are NaN. The broadcast
    source has its own frame clock and its own player slots (see track_teams).
    """
    name = "tracking.npz" if source == "tactical" else f"tracking_{source}.npz"
    cache = os.path.join(LOCAL, name)
    if os.path.exists(cache):
        data = np.load(cache)
        return data["players"], data["ball"]
    with open(asset_paths(source=source)["atd_raw"]) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    values = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).to_numpy()
    frame = values[:, 0].astype(np.int64) - 1
    slots = (values.shape[1] - 4) // 3
    n = N_FRAMES if source == "tactical" else int(frame.max()) + 1
    players = np.full((n, slots, 3), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    players[frame] = values[:, 1:1 + slots * 3].reshape(-1, slots, 3)
    ball[frame] = values[:, 1 + slots * 3:1 + (slots + 1) * 3]
    for array in (players[..., :2], ball[..., :2]):
        array[..., 0] *= PITCH[0]
        array[..., 1] *= PITCH[1]
    os.makedirs(LOCAL, exist_ok=True)
    np.savez(cache, players=players, ball=ball)
    return players, ball


def smart_tagging():
    """DataFrame of tags with start_frame, end_frame, code, Team, Type."""
    from load_smart_tagging import load
    rows = [{"code": e["code"], "start": e["start_frame"], "end": e["end_frame"], **e["tags"]}
            for e in load()]
    return pd.DataFrame(rows)


def graded_mask():
    mask = np.zeros(N_FRAMES, bool)
    for start, stop in HALVES:
        mask[start:stop] = True
    return mask


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
    return path
