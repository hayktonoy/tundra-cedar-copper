"""Shared paths, loaders and helpers for the hc-ball experiment. Arrays are indexed by tactical frame."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
VIDEO_LANE = os.path.join(ROOT, "pipeline", "hc-video", ".local")
TOUCH_LANE = os.path.join(ROOT, "pipeline", "touches-video", ".local")
TACTICAL = os.path.join(ROOT, "official", "data", "tactical")
BROADCAST = os.path.join(ROOT, "official", "data", "broadcast")
VIDEOS = {"tactical": os.path.join(TACTICAL, "BAR-ATH_tactical.mp4"),
          "broadcast": os.path.join(BROADCAST, "BAR-ATH_broadcast.mp4")}
TRACKING = os.path.join(ROOT, "pipeline", "possession-engine", ".local", "tracking.npz")
BASE = os.path.join(ROOT, "pipeline", "hc-stoppages", ".local", "best.csv")
WEIGHTS = os.path.join(ROOT, "pipeline", "football-ball-detector", ".local", "football_best.pt")
sys.path.insert(0, os.path.join(ROOT, "official", "evaluation"))
import possession_metrics as pm  # noqa: E402

N = 199319
FPS = 25
PITCH = (105.3, 68.0)
HALVES = [(11185, 90652), (110808, 190198)]
BLOCKS = [(11185, 50919), (50919, 90652), (110808, 150503), (150503, 190198)]
PURGE = 750
H, A, D, X = 0, 1, 2, 3
W, HGT = 1920, 1080
# fixed camera centres estimated in pipeline/multiview-ball (metres; z up)
C_T = np.array([53.36, 115.52, 29.97])
C_B = np.array([52.88, 113.08, 21.33])


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def write(path, pred):
    names = np.array(list("HADX"))
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{c}\n" for i, c in enumerate(names[pred]))


def block_of(frames):
    frames = np.asarray(frames)
    out = np.full(frames.shape, -1)
    for k, (a, b) in enumerate(BLOCKS):
        out[(frames >= a) & (frames < b)] = k
    return out


def train_mask(k, n=N):
    """Frames a model predicting block k may train on: graded, outside block k and its purge."""
    a, b = BLOCKS[k]
    m = np.zeros(n, bool)
    for s, e in HALVES:
        m[s:e] = True
    m[max(0, a - PURGE):b + PURGE] = False
    return m


def runs(c):
    """Maximal runs (start, end, code)."""
    c = np.asarray(c)
    cut = np.flatnonzero(np.diff(c)) + 1
    st = np.r_[0, cut]
    en = np.r_[cut, len(c)]
    return [(int(a), int(b), int(c[a])) for a, b in zip(st, en)]


def possessions(c):
    return [(a, b, t) for a, b, t in runs(c) if t in (H, A)]


def homographies(source="tactical"):
    return np.load(os.path.join(VIDEO_LANE, f"homography_{source}.npy"))


def to_pixels(Hm, xy):
    """Hm (n,3,3), xy (n,...,2) metres -> pixels."""
    shp = xy.shape
    p = xy.reshape(shp[0], -1, 2) / np.array(PITCH)
    v = np.einsum("nij,nkj->nki", Hm, np.concatenate([p, np.ones(p.shape[:2] + (1,))], -1))
    return (v[..., :2] / v[..., 2:]).reshape(shp)


def to_pitch(Hm, px):
    """Hm (n,3,3), px (n,...,2) pixels -> metres on the ground plane."""
    shp = px.shape
    Hi = np.linalg.inv(Hm)
    q = px.reshape(shp[0], -1, 2)
    v = np.einsum("nij,nkj->nki", Hi, np.concatenate([q, np.ones(q.shape[:2] + (1,))], -1))
    return (v[..., :2] / v[..., 2:] * np.array(PITCH)).reshape(shp)


def broadcast_index():
    """Broadcast frame shown at each tactical frame (-1 where the sync does not cover it)."""
    sync = json.load(open(os.path.join(ROOT, "calibration", "broadcast_sync.json")))
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync["halves"]:
        m = (T >= h["from"]) & (T < h["to"])
        B[m] = np.round((T[m] - h["a"]) / (1 + h["b"])).astype(int)
    return B


def triangulate(gt, gb):
    """Ground projections in the tactical / broadcast view (n,2) -> 3D points (n,3), ray gap (n,)."""
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


def read_frames(source, start, n, step=1):
    """Decode n consecutive frames from `start` of one video as BGR uint8 (n,1080,1920,3) on the CPU."""
    vf = [] if step == 1 else ["-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "0"]
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start / 25:.4f}", "-i", VIDEOS[source],
           "-frames:v", str((n + step - 1) // step), *vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, HGT, W, 3)
