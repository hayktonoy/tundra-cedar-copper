"""Build hc-video's ball features and physical evidence.

    cd pipeline/hc-video
    .local/venv/Scripts/python.exe ../from-raw/hcvideo_free.py

After clean.py, clean.py broadcast and lift.py, write the three feat_*.npz files
and the physical inputs in evidence.npz.
"""
import os
import sys

import numpy as np

HV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hc-video")
sys.path.insert(0, HV)
os.chdir(HV)

from common import LOCAL, N  # noqa: E402
from evidence import control  # noqa: E402
from features import _players  # noqa: E402
from fit import ball_features  # noqa: E402
from kicks import kick_evidence  # noqa: E402


def evidence():
    """Write the physical evidence channels."""
    pl, teams = _players()
    ev = {}
    v = np.load(os.path.join(LOCAL, "ball_video.npz"))
    for k in ("x", "y", "px", "py", "score"):
        ev[f"video_ball_{k}"] = v[k].astype(np.float32)
    ev["video_ball_source"] = v["source"]
    dh, da, c = control(np.c_[v["x"], v["y"]], pl, teams)
    ev["video_dist_h"], ev["video_dist_a"], ev["video_control"] = dh.astype(np.float32), da.astype(np.float32), c
    b = np.load(os.path.join(LOCAL, "ball_broadcast.npz"))
    for k in ("x", "y", "px", "py", "score"):
        ev[f"broadcast_ball_{k}"] = b[k].astype(np.float32)
    ev["broadcast_ball_source"] = b["source"]
    dh, da, c = control(np.c_[b["x"], b["y"]], pl, teams)
    ev["broadcast_dist_h"], ev["broadcast_dist_a"], ev["broadcast_control"] = dh.astype(np.float32), da.astype(np.float32), c
    z3 = np.load(os.path.join(LOCAL, "ball_3d.npz"))
    ev["ball3d_x"], ev["ball3d_y"], ev["ball_height"] = z3["x"], z3["y"], z3["zs"]
    ev["restart_kick"] = kick_evidence().astype(np.int8)
    for k, arr in ev.items():
        assert arr.shape == (N,), (k, arr.shape)
    np.savez_compressed(os.path.join(LOCAL, "evidence.npz"), **ev)
    print("wrote evidence.npz without team_q:", ", ".join(ev), flush=True)


if __name__ == "__main__":
    for kind in ("provided", "video", "broadcast"):
        path = os.path.join(LOCAL, f"feat_{kind}.npz")
        if os.path.exists(path):
            os.remove(path)          # ball_features returns a cached file as is
        ball_features(kind)
        print("wrote", path, flush=True)
    evidence()
