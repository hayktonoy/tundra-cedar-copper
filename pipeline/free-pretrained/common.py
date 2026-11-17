"""Shared paths and helpers for the free-pretrained lane. Arrays are indexed by tactical frame."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
EXP = os.path.join(ROOT, "pipeline")
VIDEOS = {"tactical": os.path.join(ROOT, "official", "data", "tactical", "BAR-ATH_tactical.mp4"),
          "broadcast": os.path.join(ROOT, "official", "data", "broadcast", "BAR-ATH_broadcast.mp4")}
N = 199319
HALVES = [(11185, 90652), (110808, 190198)]
BLOCKS = [(11185, 50919), (50919, 90652), (110808, 150503), (150503, 190198)]
W, HGT = 1920, 1080
sys.path.insert(0, os.path.join(ROOT, "official", "evaluation"))


def sync():
    return json.load(open(os.path.join(ROOT, "calibration", "broadcast_sync.json")))["halves"]


def broadcast_index():
    """Broadcast frame shown at each tactical frame (-1 where the sync does not cover it)."""
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync():
        m = (T >= h["from"]) & (T < h["to"])
        B[m] = np.round((T[m] - h["a"]) / (1 + h["b"])).astype(int)
    return B


def tactical_of_broadcast(b):
    """Tactical frame (float) showing the moment of broadcast frame b; NaN outside the sync."""
    b = np.asarray(b, float)
    out = np.full(b.shape, np.nan)
    for h in sync():
        t = h["a"] + b * (1 + h["b"])
        m = (t >= h["from"]) & (t < h["to"])
        out[m] = t[m]
    return out


def half_sign():
    """+1 where Barca defends the image-left goal (x = 0), -1 where it defends the right one.

    From the provided tracking alone: team 0 (Barcelona in the metadata) has the smaller mean x in
    the first half, and pitch x = 0 projects to the left of both cameras' images.
    """
    s = np.zeros(N)
    s[:100730] = 1.0
    s[100730:] = -1.0
    return s


def read_frame(source, frame, scale=None):
    vf = ["-vf", f"scale={scale[0]}:{scale[1]}"] if scale else []
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{frame / 25:.4f}", "-i", VIDEOS[source], "-frames:v", "1",
           *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    w, h = scale if scale else (W, HGT)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)
