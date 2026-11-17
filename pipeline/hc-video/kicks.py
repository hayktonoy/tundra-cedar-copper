"""Restart kicks from the video ball: at rest for about a second, then struck.

kick_evidence() -> (N,) float in [0, 1], 1 at frames where the cleaned tactical ball was resting
(speed < REST m/s on most of the previous REST_WIN frames) and reaches > KICK m/s within KICK_WIN
frames. Fed into R7's restart-event channel to make leaving dead ball cheap at that moment.
"""
import os

import numpy as np
import pandas as pd

from common import HALVES, LOCAL, N

REST, REST_WIN, KICK, KICK_WIN = 1.5, 25, 7.0, 6


def kick_evidence(rest=REST, rest_win=REST_WIN, kick=KICK, kick_win=KICK_WIN, path="ball_video.npz"):
    v = np.load(os.path.join(LOCAL, path))
    xy = np.c_[v["x"], v["y"]].astype(np.float64)
    xy[~np.isin(v["source"], (1, 2))] = np.nan
    out = np.zeros(N, np.float32)
    for a, b in HALVES:
        p = xy[a:b]
        sp = np.full(b - a, np.nan)
        sp[1:] = np.hypot(*(p[1:] - p[:-1]).T) * 25
        sp = pd.Series(sp).rolling(3, center=True, min_periods=1).median()
        still = (sp < rest).astype(float).where(sp.notna())
        rest_frac = still.rolling(rest_win, min_periods=rest_win // 2).mean().shift(1)
        fast = sp[::-1].rolling(kick_win, min_periods=1).max()[::-1]
        ev = (rest_frac > 0.8) & (fast > kick)
        out[a:b] = ev.to_numpy(np.float32)
    return out


if __name__ == "__main__":
    k = kick_evidence()
    on = np.flatnonzero(np.diff(np.r_[0, k]) > 0)
    print("kick onsets:", len(on))
