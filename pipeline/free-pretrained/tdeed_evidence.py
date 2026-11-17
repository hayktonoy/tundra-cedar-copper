"""SoccerNet spotter outputs -> events and per-frame channels on the tactical timeline. No labels.

    python tdeed_evidence.py

Reads .local/tdeed_<video>_<start>_<end>.npz (tdeed_run.py). For each video:
  - sampled frames are mapped to tactical frames (broadcast through the algorithmic sync); broadcast
    frames without a valid homography (replays, close-ups, crowd) are dropped;
  - class probabilities of the two models are averaged, shifted by the displacement head (as the
    T-DEED post-processing does, max aggregation) and peak-picked per class;
  - each event takes the team head's P(right team) at its frame, turned into P(Barcelona) by the
    side Barcelona defends in that half (common.half_sign, from the tracking).
Writes .local/tdeed_events.npz (one row per event) and .local/tdeed_channels.npz (length-N arrays):
  onball_<v>   sum of score * conf(P(home)) of on-ball actions (pass, drive, high pass, cross,
               shot, successful tackle, throw-in, free kick), spread over +-3 frames, where
               conf = sign(2P - 1) * clip((|2P - 1| - .6) / .4, 0, 1): only confident team calls
               (P > .8 or < .2) count. The .8 was chosen after diag_events.py showed that these are
               99% right on the broadcast while the unconfident ones are not better than our base;
  onlin_<v>    the same with the plain 2P - 1 (all calls);
  live_<v>     sum of scores of on-ball actions (any team, score > .3) held from 2 frames before to
               25 frames after the action: a pass, cross, shot or throw-in means the ball is in play;
  teamw_<v>    onball over the window from 2 frames before to 12 after;
  live2_<v>    as live_ but only deliveries and restarts (cross, shot, free kick, throw-in): inside
               predicted stoppages these are followed by live play, passes and drives mostly are not
               (players pass the ball during stoppages). Class choice made after diag on all blocks;
  team2_<v>    score * (2 P(home) - 1) of the same events over the same window;
  out_<v>      ball out of play (OUT, SN ball out of play), spread over +-5 frames;
  restart_<v>  restarts (THROW IN, FREE KICK, SN throw-in / corner / free kicks / kick-off / penalty);
  act_<v>      1 - background probability, the spotter's "something is happening with the ball";
  team_<v>     raw team-head P(home) at every frame (no event gating), NaN where not seen.
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import LOCAL, N, half_sign, tactical_of_broadcast  # noqa: E402

C12 = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
       "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]
C17 = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
       "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
       "Corner", "Yellow card", "Red card", "Yellow->red card"]
ONBALL = ["PASS", "DRIVE", "HIGH PASS", "CROSS", "SHOT", "PLAYER SUCCESSFUL TACKLE", "THROW IN", "FREE KICK"]
OUT12, OUT17 = ["OUT"], ["Ball out of play"]
RESTART12 = ["THROW IN", "FREE KICK"]
RESTART17 = ["Throw-in", "Corner", "Indirect free-kick", "Direct free-kick", "Kick-off", "Penalty"]
HEIGHT, DIST = 0.15, 4


def load(video):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(LOCAL, f"tdeed_{video}_*_*.npz")))
             if not p.endswith("_7000_10000.npz")]
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    f = z["frame"]
    if video == "broadcast":
        T = tactical_of_broadcast(f)
        Hb = np.load(os.path.join(HERE, "..", "hc-video", ".local", "homography_broadcast.npy"))
        ok = np.isfinite(T) & np.isfinite(Hb[np.clip(f, 0, len(Hb) - 1), 0, 0])
    else:
        T = f.astype(float)
        ok = np.ones(len(f), bool)
    return z, T, ok


def shift(p, d):
    """Max-aggregate p[i] onto index i - round(d[i]) (T-DEED's displacement post-processing)."""
    n = len(p)
    j = np.clip(np.arange(n) - np.round(d).astype(int), 0, n - 1)
    out = np.zeros_like(p)
    np.maximum.at(out, j, p)
    return out


def spread_add(arr, t, v, w, w2=None):
    for k in range(-w, (w if w2 is None else w2) + 1):
        np.add.at(arr, np.clip(t + k, 0, N - 1), v)


def main():
    events, ch = [], {}
    for video in ("broadcast", "tactical"):
        z, T, ok = load(video)
        # breaks between halves / files: never let a peak or shift run across them
        d = 0.5 * (z["teamd"] + z["balld"])
        p12 = shift(0.5 * (z["team13"][:, 1:] + z["ball13"][:, 1:]), d)
        p17 = shift(0.5 * (z["team18"][:, 1:] + z["ball18"][:, 1:]), d)
        r = z["teamr"]
        Ti = np.round(np.nan_to_num(T, nan=0)).astype(int)
        ph = np.where(half_sign()[Ti] > 0, 1 - r, r)
        for head, P, names in (("bas", p12, C12), ("sn", p17, C17)):
            for c, name in enumerate(names):
                pk, prop = find_peaks(P[:, c], height=HEIGHT, distance=DIST)
                pk = pk[ok[pk]]
                for i in pk:
                    events.append((T[i], video == "broadcast", head == "sn", c, P[i, c], ph[i]))
        onball = np.zeros(N)
        onlin = np.zeros(N)
        live = np.zeros(N)
        teamw = np.zeros(N)
        live2 = np.zeros(N)
        team2 = np.zeros(N)
        out = np.zeros(N)
        restart = np.zeros(N)
        ev = [e for e in events if e[1] == (video == "broadcast")]
        for t, _, sn, c, s, h in ev:
            t = int(round(t))
            name = (C17 if sn else C12)[c]
            if not sn and name in ONBALL:
                c2 = 2 * h - 1
                conf = np.sign(c2) * np.clip((abs(c2) - .6) / .4, 0, 1)
                spread_add(onball, np.array([t]), np.array([s * conf]), 3)
                spread_add(onlin, np.array([t]), np.array([s * c2]), 3)
                spread_add(teamw, np.array([t]), np.array([s * conf]), 2, 12)
                if s > .3:
                    spread_add(live, np.array([t]), np.array([s]), 2, 25)
                if s > .3 and name in ("CROSS", "SHOT", "FREE KICK", "THROW IN"):
                    spread_add(live2, np.array([t]), np.array([s]), 2, 25)
                    spread_add(team2, np.array([t]), np.array([s * c2]), 2, 25)
            if (not sn and name in OUT12) or (sn and name in OUT17):
                spread_add(out, np.array([t]), np.array([s]), 5)
            if (not sn and name in RESTART12) or (sn and name in RESTART17):
                spread_add(restart, np.array([t]), np.array([s]), 5)
        act = np.full(N, np.nan)
        team = np.full(N, np.nan)
        a = 1 - 0.5 * (z["team13"][:, 0] + z["ball13"][:, 0])
        g = ok & np.isfinite(T)
        # sampled every 2nd frame: fill each tactical frame from the nearest sample
        tt = T[g]
        order = np.argsort(tt)
        tt, av, hv = tt[order], a[g][order], ph[g][order]
        idx = np.clip(np.searchsorted(tt, np.arange(N)), 1, len(tt) - 1)
        near = np.where(np.abs(tt[idx - 1] - np.arange(N)) < np.abs(tt[idx] - np.arange(N)), idx - 1, idx)
        close = np.abs(tt[near] - np.arange(N)) <= 1.5
        act[close] = av[near][close]
        team[close] = hv[near][close]
        v = video[:3]
        ch.update({f"onball_{v}": onball, f"onlin_{v}": onlin, f"live_{v}": live, f"teamw_{v}": teamw, f"live2_{v}": live2, f"team2_{v}": team2, f"out_{v}": out, f"restart_{v}": restart, f"act_{v}": act,
                   f"team_{v}": team})
    e = np.array(events)
    np.savez_compressed(os.path.join(LOCAL, "tdeed_events.npz"), t=e[:, 0], broadcast=e[:, 1].astype(bool),
                        sn=e[:, 2].astype(bool), cls=e[:, 3].astype(int), score=e[:, 4], p_home=e[:, 5],
                        names12=np.array(C12), names17=np.array(C17))
    np.savez_compressed(os.path.join(LOCAL, "tdeed_channels.npz"), **{k: v.astype(np.float32) for k, v in ch.items()})
    print(len(e), "events;", {k: float(np.nanmean(np.abs(v))) for k, v in ch.items()})


if __name__ == "__main__":
    main()
