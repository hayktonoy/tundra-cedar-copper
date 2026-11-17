"""Free-owner evidence and decoder settings."""
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d

from common import LOCAL, N
from decode import decode, posterior

# Default settings for the pseudo-label decoder.
DEFAULT = dict(wq=0.4, wc=2.0, beta=20.0, wd=0.2, bd=-8.0, wf=0.0, tf=2.5, cin=40.0, g_out=10.0, g_drop=10.0,
               cout=20.0, h_kick=10.0, h_setp=3.0, h_acc=5.0)
GRID = dict(wq=[0.2, 0.4, 0.6, 0.8, 1.2], wc=[0.5, 1.0, 2.0, 4.0, 8.0], beta=[5, 10, 20, 40, 80],
            wd=[0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], bd=[-14, -11, -8, -6, -4, -2, 0],
            wf=[0, 0.1, 0.2, 0.4, 0.8, 1.6], tf=[1.5, 2.0, 2.5, 3.0, 3.5],
            cin=[20, 40, 80, 160, 320], g_out=[0, 5, 10, 20, 40, 80, 160], g_drop=[0, 5, 10, 20, 40, 80, 160],
            cout=[5, 10, 20, 40, 80, 160], h_kick=[0, 5, 10, 20, 40], h_acc=[0, 2, 5, 10, 20])


def spread(x, w):
    return maximum_filter1d(x, 2 * w + 1)


class Evidence:
    def __init__(self, q_team, q_dead, contact, events):
        """q_team: P(home | live), q_dead: P(dead), contact: +1 home / -1 away contact frames, all length N."""
        self.lq = np.log(np.clip(q_team, 1e-4, 1 - 1e-4) / np.clip(1 - q_team, 1e-4, 1 - 1e-4))
        self.ld = np.log(np.clip(q_dead, 1e-4, 1 - 1e-4) / np.clip(1 - q_dead, 1e-4, 1 - 1e-4))
        self.ct = contact
        self.out = spread(np.maximum(events["out"], events["lost"]), 5)
        self.drop = spread(events["drop"], 3)
        self.fut = events["fut"]
        self.kick = spread(events["kick"], 3)
        self.setp = events["setp"]
        self.acc = events["accel"]

    def scores(self, p):
        s = p["wq"] * self.lq + p["wc"] * self.ct
        e = p["wd"] * (self.ld - p["bd"]) + p["wf"] * (p["tf"] - self.fut)
        cin = p["cin"] - p["g_out"] * self.out - p["g_drop"] * self.drop
        cout = p["cout"] - p["h_kick"] * self.kick - p["h_setp"] * self.setp - p["h_acc"] * self.acc
        return s, e, p["beta"], np.maximum(cin, 0.5), np.maximum(cout, 0.5)

    def run(self, p):
        return decode(*self.scores(p))

    def marginals(self, p):
        return posterior(*self.scores(p))


def contact(fr, Xs, X0, X1, ns, npair):
    """+1 / -1 where the detected, slow video ball is at a Barca / Athletic player's feet."""
    S = {k: Xs[:, i] for i, k in enumerate(ns)}
    iP = {k: i for i, k in enumerate(npair)}
    f0 = np.nan_to_num(X0[:, iP["foot"]], nan=9)
    f1 = np.nan_to_num(X1[:, iP["foot"]], nan=9)
    ok = (S["ball_det"] > 0) & (np.nan_to_num(S["ball_speed"], nan=99) < 10)
    ct = np.zeros(N)
    ct[fr] = (ok & (f0 < 0.5) & (f0 + 0.15 < f1)).astype(float) - (ok & (f1 < 0.5) & (f1 + 0.15 < f0)).astype(float)
    return ct


def build(team_file="q_team1.npy", dead_file="llr_dead.npy"):
    z = np.load(os.path.join(LOCAL, "features.npz"))
    fr, Xs, X0, X1 = z["frames"], z["Xs"], z["X0"], z["X1"]
    ns, npair = list(z["names_s"]), list(z["names_pair"])
    q_team = np.load(os.path.join(LOCAL, team_file))
    ll = np.nan_to_num(np.load(os.path.join(LOCAL, dead_file)), nan=0)
    q_dead = 1 / (1 + np.exp(-np.clip(ll, -9, 9)))
    return Evidence(q_team, q_dead, contact(fr, Xs, X0, X1, ns, npair), dict(np.load(os.path.join(LOCAL, "events.npz"))))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pseudo":          # decode with the fixed default constants: pseudo-labels for round 2
        ev = build(sys.argv[2] if len(sys.argv) > 2 else "q_team1.npy")
        p = ev.run(DEFAULT)
        np.save(os.path.join(LOCAL, "pseudo1.npy"), p.astype(np.int8))
        print("pseudo-labels from DEFAULT:", {c: int((p == i).sum()) for i, c in enumerate("HADX")})
