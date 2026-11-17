"""Label-free detectors of the physical moments around brief phases. No labels are read.

    python detect.py            -> .local/detect.npz (per-frame channels, length 199,319)

Inputs: three ball tracks (hc-ball's tactical and broadcast video tracks, the provided ball), pixel
contact on both views (ball pixel vs touches-video's jersey-colour player boxes; contact_bro.py for
the broadcast), the provided players' speeds, free-owner's broadcast-off-pitch feature, and the
SoccerNet spotter's events (free-pretrained). All thresholds are hand-set physical values.

outside   the ball is beyond a touch or goal line: at least two of the three tracks put it more than
          OUT_M metres outside, each sustained over 2 of 5 frames. One track alone is often wrong
          (a spare ball by the boards, a lost track); two agreeing almost never are.
out_on    onset of `outside` (the ball has just left the pitch).
held      the ball in an outfield player's hands: detected ball pixel inside one player's body box
          at chest-to-overhead height (0 at the feet, 1 at the head), nearly still, for at least
          HOLD frames, outside both penalty areas (where a keeper's catch is live play). A throw-in
          being prepared, or the ball picked up to restart.
cue       restart cues: the throw that ends a `held` episode, a kick from rest (ball still for
          20 of 25 frames, then > 6 m/s), a re-entry (the ball goes from the line to > 3 m inside
          the pitch at > 5 m/s), the spotter's restarts and deliveries (throw-in, free kick,
          corner, kick-off, cross, shot; score > .3).
out_iv    from an `out_on` to the first restart cue at least 1 s later (at most 30 s; 3 s if there
          is none): the ball is dead until it is put back in play.
live_iv   from a restart cue that ends an `out_iv` to the next `out_on`, at most 4 s: the ball is in
          play right after a restart, however short the phase.
blive     a brief live phase: a restart cue followed 1-8 s later by an `out_on`, with the broadcast on
          the pitch and the five fastest players faster after the cue than in the 3 s before it.
          `bcue` marks its restart.
"""
import numpy as np
from scipy.ndimage import binary_closing, label, maximum_filter1d, uniform_filter1d

from fb import EXP, HALVES, LOCAL, N, ball, tracking

PITCH = (105.3, 68.0)
OUT_M = .5
HOLD = 15
MIN_OUT = 25          # an out interval lasts at least 1 s before a restart cue can end it
SPOTTER_VOTE = False  # the spotter's OUT as a third vote: more onsets, but false ones in play (README)


def episodes(mask, close=5, min_len=12):
    """Runs of `mask` after closing gaps of up to 2*close frames, at least `min_len` long: [(a, b)]."""
    m = binary_closing(mask, np.ones(close * 2 + 1)) | mask if close else mask
    lab, n = label(m)
    out = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(lab == i)
        if idx[-1] + 1 - idx[0] >= min_len:
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def in_box(x, y):
    """Inside either penalty area (16.5 m deep, 40.3 m wide), with a 1 m margin."""
    yy = np.abs(y - PITCH[1] / 2) < 20.15 + 1
    return ((x < 16.5 + 1) | (x > PITCH[0] - 16.5 - 1)) & yy


def beyond(x, y):
    """Metres beyond the nearest touch or goal line (negative inside the pitch)."""
    return np.maximum(np.maximum(-x, x - PITCH[0]), np.maximum(-y, y - PITCH[1]))


def onsets(mask, gap=50):
    ons = np.flatnonzero(mask[1:] & ~mask[:-1]) + 1
    out = []
    for o in ons:
        if not out or o - out[-1] > gap:
            out.append(int(o))
    return np.array(out, int)


def outside_channel(t, b, prov, spotter=None):
    cnt = np.zeros(N) if spotter is None else (spotter > .3).astype(float)
    for src, x, y in ((t["source"] == 1, t["x"], t["y"]), (b["source"] == 1, b["x"], b["y"]),
                      (np.isfinite(prov[:, 0]), prov[:, 0], prov[:, 1])):
        o = np.where(src, beyond(x.astype(float), y.astype(float)), np.nan)
        v = np.nan_to_num(o, nan=-9.) > OUT_M
        cnt += np.convolve(v, np.ones(5), "same") >= 2
    return cnt >= 2


def held_channel(views):
    one = np.zeros(N, bool)
    team = np.zeros(N)
    H = np.zeros((N, 2), bool)
    for c, tr in views:
        ok = tr["source"] >= 1
        spd = np.nan_to_num(c["speed_b"], nan=99.)
        for k, nm in ((0, "h"), (1, "a")):
            dn = np.nan_to_num(c[f"dn_{nm}"], nan=9.)
            rel = np.nan_to_num(c[f"rel_{nm}"], nan=-9.)
            H[:, k] |= ok & (dn < .12) & (rel > .55) & (rel < 1.25) & (spd < 3.)
    one = H[:, 0] ^ H[:, 1]
    return one, np.where(H[:, 0], 1., np.where(H[:, 1], -1., 0.))


def prov_players():
    return tracking()[0]


def main():
    t, b = ball()
    _, prov = tracking()
    c_t = np.load(EXP / "hc-ball/.local/contact_tactical.npz")
    c_b = np.load(LOCAL / "contact_broadcast.npz")
    td = np.load(EXP / "free-pretrained/.local/tdeed_events.npz")
    x = np.where(t["source"] > 0, t["x"], b["x"]).astype(float)
    y = np.where(t["source"] > 0, t["y"], b["y"]).astype(float)
    graded = np.zeros(N, bool)
    for a, e in HALVES:
        graded[a:e] = True

    sp = None
    if SPOTTER_VOTE:
        z = np.load(EXP / "free-pretrained/.local/tdeed_channels.npz")
        sp = maximum_filter1d(np.nan_to_num(z["out_bro"]) + np.nan_to_num(z["out_tac"]), 51)
    outside = outside_channel(t, b, prov, sp) & graded
    out_on = np.zeros(N, np.float32)
    on = onsets(outside)
    out_on[on] = 1

    one, team = held_channel([(c_t, t), (c_b, b)])
    held = np.zeros(N, np.float32)
    held_team = np.zeros(N, np.float32)
    fast = ((t["source"] > 0) & (np.nan_to_num(t["speed"]) > 5)) | ((b["source"] > 0) & (np.nan_to_num(b["speed"]) > 5))
    cue = np.zeros(N, bool)
    for a, e in episodes(one, close=5, min_len=HOLD):
        if one[a:e].mean() < .5 or in_box(np.array([np.nanmedian(x[a:e])]), np.array([np.nanmedian(y[a:e])]))[0]:
            continue
        held[a:e] = 1
        held_team[a:e] = np.sign(team[a:e].sum())
        nxt = np.flatnonzero(fast[e - 3:e + 25])
        if len(nxt):
            cue[e - 3 + nxt[0]] = True
    # kick from rest, either view
    for tr in (t, b):
        det = tr["source"] == 1
        sp = np.nan_to_num(tr["speed"], nan=99.)
        still = (det & (sp < 1.)).astype(float)
        prev = np.r_[np.zeros(1), np.convolve(still, np.ones(25), "full")[:N - 1]]   # count over [t-25, t)
        k = (prev >= 20) & det & (sp > 6)
        cue[onsets(k, gap=50)] = True
    # spotter restarts and deliveries (names12: THROW IN, FREE KICK, CROSS, SHOT; names17: restarts)
    n12, n17 = list(td["names12"]), list(td["names17"])
    want12 = {n12.index(k) for k in ("THROW IN", "FREE KICK", "CROSS", "SHOT")}
    want17 = {n17.index(k) for k in ("Throw-in", "Corner", "Indirect free-kick", "Direct free-kick", "Kick-off", "Penalty")}
    for tt, cl, sn, sc in zip(td["t"], td["cls"], td["sn"], td["score"]):
        if sc > .3 and ((not sn and cl in want12) or (sn and cl in want17)) and 0 <= tt < N:
            cue[int(round(tt))] = True
    # re-entry: the ball goes from the line (within 1 m, or beyond it) to > 3 m inside at speed
    # (interpolated track frames count: a long throw or kick is often only seen at its ends)
    for src, bx, by, sp in ((t["source"] >= 1, t["x"], t["y"], t["speed"]), (b["source"] >= 1, b["x"], b["y"], b["speed"])):
        o = np.where(src, beyond(bx.astype(float), by.astype(float)), np.nan)
        near = maximum_filter1d(np.nan_to_num(o, nan=-99.), 26, origin=12) > -1      # over [t-25, t]
        k = src & (np.nan_to_num(o, nan=0.) < -3) & near & (np.nan_to_num(sp) > 5)
        k &= np.r_[False, k[:-1]] & np.r_[False, False, k[:-2]]
        cue[onsets(k, gap=50)] = True
    cue &= graded

    out_iv = np.zeros(N, np.float32)
    live_iv = np.zeros(N, np.float32)
    cues = np.flatnonzero(cue)
    ends = []
    for o in on:
        nxt = cues[(cues >= o + MIN_OUT) & (cues < o + 750)]
        e = int(nxt[0]) if len(nxt) else o + 75
        out_iv[o:e] = 1
        if len(nxt):
            ends.append(e)
    for e in ends:
        nxt = on[(on > e) & (on < e + 100)]
        stop = int(nxt[0]) if len(nxt) else e + 100
        live_iv[e:stop] = np.maximum(live_iv[e:stop], 1 - out_iv[e:stop])
    # brief live phases: a restart cue, then the ball out again 1-8 s later, with the broadcast on
    # the pitch and the players speeding up after the cue (in a stoppage they walk, or the
    # broadcast shows a replay or a close-up while the ball is moved around)
    sp = np.nan_to_num(prov_players()[:, :, 2], nan=0.)
    top5 = np.sort(sp, 1)[:, -5:].mean(1)
    F = np.load(EXP / "free-owner/.local/features.npz")
    off = np.full(N, np.nan)
    off[F["frames"]] = F["Xs"][:, list(F["names_s"]).index("bc_off")]
    blive = np.zeros(N, np.float32)
    bcue = np.zeros(N, np.float32)
    for o in on:
        prev = cues[(cues < o - 25) & (cues > o - 200)]
        if not len(prev):
            continue
        r = int(prev[-1])
        if np.nanmean(off[r:o]) < .3 and top5[r:o].mean() >= top5[r - 75:r].mean() + .3:
            blive[r:o] = 1
            bcue[r] = 1
    np.savez_compressed(LOCAL / "detect.npz", outside=outside.astype(np.float32), out_on=out_on, held=held,
                        held_team=held_team, cue=cue.astype(np.float32), out_iv=out_iv * graded,
                        live_iv=live_iv * graded, blive=blive * graded, bcue=bcue * graded)
    print(f"out onsets {len(on)}, outside frames {int(outside.sum())}, held frames {int(held.sum())}, "
          f"cues {int(cue.sum())}, out_iv frames {int(out_iv.sum())}, live_iv frames {int(live_iv.sum())}")


if __name__ == "__main__":
    main()
