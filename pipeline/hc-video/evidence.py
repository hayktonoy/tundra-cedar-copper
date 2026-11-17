"""Measure team control from player distances to the ball."""
import numpy as np

from common import N
from features import nearest


def control(xy, pl, teams):
    dh, da, _, _ = nearest(xy.astype(np.float32), pl, teams)
    c = np.full(N, -1, np.int8)
    c[(dh < 1.5) & (dh < da)] = 0
    c[(da < 1.5) & (da < dh)] = 1
    return dh, da, c
