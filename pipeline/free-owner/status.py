"""Live / dead status without labels: a two-state hidden Markov model fitted by EM on this match.

    python status.py            # -> .local/p_dead.npy (length 199,319)

Each frame emits a handful of discretised channels, independent given the state (naive Bayes):
collective player speed now and 0.5-2 s ahead (joint), the broadcast showing the pitch or not
(now and over 10 s, joint), ball detection rate, ball at rest, ball outside the lines, number of
tracked players, a Smart Tagging set piece, the tactical camera's pan speed and zoom, the
broadcast zoom. Bins are quantiles of the match's own values (or fixed physical cut points);
missing values are their own category.

Baum-Welch: start from a physical guess (players walking: dead; running: live), then alternate
forward-backward posteriors and categorical / transition re-estimates. Which state is "dead" is
fixed by the start (the slow state). No label is read.
"""
import os

import numba
import numpy as np

from common import HALVES, LOCAL, N, graded

ITER = 40


def qbins(x, k):
    """Quantile bin index 1..k, 0 where missing."""
    ok = np.isfinite(x)
    edges = np.nanquantile(x[ok], np.linspace(0, 1, k + 1)[1:-1])
    out = np.zeros(len(x), np.int64)
    out[ok] = 1 + np.searchsorted(edges, x[ok])
    return out


def cbins(x, cuts):
    ok = np.isfinite(x)
    out = np.zeros(len(x), np.int64)
    out[ok] = 1 + np.searchsorted(np.asarray(cuts, float), x[ok])
    return out


def channels(S, ev, frames):
    fut = ev["fut"][frames]
    ch = {
        "speed": qbins(S["top5_w25"], 12) * 9 + qbins(fut, 8),
        "speed_past": qbins(S["top5_past75"], 8),
        "bcast": cbins(S["bc_off_w25"], [1e-6, 0.5, 1 - 1e-6]) * 8 + cbins(S["bc_off_w250"], [1e-6, .1, .3, .6, .9, 1 - 1e-6]),
        "ball_det": cbins(S["ball_det_w75"], [.1, .3, .5, .7, .9]),
        "ball_still": cbins(S["ball_still_w50"], [1e-6, .2, .5, .8, 1 - 1e-6]),
        "ball_out": cbins(S["ball_out_w50"], [-10, -3, -1, 0, 1]),
        "n_players": cbins(S["n_players_w75"], [14, 16, 18, 19, 20, 21, 22]),
        "setpiece": cbins(S["setpiece"], [0.5]),
        "spot": cbins(S["ball_still_w50"], [0.3, 0.8]) * 7 + np.nan_to_num(S["ball_zone"], nan=0).astype(np.int64),
        "pan": qbins(S["tac_pan_w75"], 6),
        "tac_zoom": qbins(S["tac_width_w75"], 6),
        "bc_zoom": qbins(S["bc_width_w75"], 6),
    }
    return {k: np.ascontiguousarray(v) for k, v in ch.items()}


@numba.njit(cache=True)
def forward_backward(logb, seg, a_stay):
    """logb (n, 2) emission log-likelihoods; seg: segment bounds (contiguous halves). Returns gamma (n, 2)."""
    n = logb.shape[0]
    gam = np.zeros((n, 2))
    A = np.array([[a_stay[0], 1 - a_stay[0]], [1 - a_stay[1], a_stay[1]]])
    for s in range(len(seg) - 1):
        a, b = seg[s], seg[s + 1]
        m = b - a
        al = np.zeros((m, 2))
        sc = np.zeros(m)
        for t in range(m):
            e0 = np.exp(logb[a + t, 0] - max(logb[a + t, 0], logb[a + t, 1]))
            e1 = np.exp(logb[a + t, 1] - max(logb[a + t, 0], logb[a + t, 1]))
            if t == 0:
                p0, p1 = 0.5 * e0, 0.5 * e1
            else:
                p0 = (al[t - 1, 0] * A[0, 0] + al[t - 1, 1] * A[1, 0]) * e0
                p1 = (al[t - 1, 0] * A[0, 1] + al[t - 1, 1] * A[1, 1]) * e1
            z = p0 + p1
            al[t, 0], al[t, 1] = p0 / z, p1 / z
            sc[t] = z
        be = np.ones((m, 2))
        for t in range(m - 2, -1, -1):
            e0 = np.exp(logb[a + t + 1, 0] - max(logb[a + t + 1, 0], logb[a + t + 1, 1]))
            e1 = np.exp(logb[a + t + 1, 1] - max(logb[a + t + 1, 0], logb[a + t + 1, 1]))
            b0 = A[0, 0] * e0 * be[t + 1, 0] + A[0, 1] * e1 * be[t + 1, 1]
            b1 = A[1, 0] * e0 * be[t + 1, 0] + A[1, 1] * e1 * be[t + 1, 1]
            z = b0 + b1
            be[t, 0], be[t, 1] = b0 / z, b1 / z
        for t in range(m):
            g0 = al[t, 0] * be[t, 0]
            g1 = al[t, 1] * be[t, 1]
            gam[a + t, 0] = g0 / (g0 + g1)
            gam[a + t, 1] = g1 / (g0 + g1)
    return gam


def fit(ch, seg, g0, iters=ITER, verbose=True):
    """g0: initial P(dead) per frame. Returns posterior P(dead) and the per-channel log-likelihood ratios."""
    gd = g0.copy()
    for it in range(iters):
        logb = np.zeros((len(gd), 2))
        llr = {}
        for k, c in ch.items():
            K = c.max() + 1
            cnt_d = np.bincount(c, weights=gd, minlength=K) + 1.0
            cnt_l = np.bincount(c, weights=1 - gd, minlength=K) + 1.0
            pd_, pl = cnt_d / cnt_d.sum(), cnt_l / cnt_l.sum()
            logb[:, 1] += np.log(pd_[c])
            logb[:, 0] += np.log(pl[c])
            llr[k] = np.log(pd_ / pl)
        # transitions from the posterior run lengths: expected switches ~ sum |gamma_t - gamma_{t-1}| / 2
        sw = np.abs(np.diff(gd))
        for s in seg[1:-1]:
            sw[s - 1] = 0
        n_dead, n_live = gd.sum(), (1 - gd).sum()
        k_sw = max(1.0, sw.sum() / 2)
        a_stay = np.array([1 - k_sw / n_live, 1 - k_sw / n_dead])
        gam = forward_backward(logb, seg, a_stay)
        new = gam[:, 1]
        delta = np.abs(new - gd).mean()
        gd = new
        if verbose and (it % 5 == 0 or it == iters - 1):
            print(f"  EM {it}: dead share {gd.mean():.3f}, mean run live {n_live / k_sw / 25:.1f}s dead {n_dead / k_sw / 25:.1f}s, change {delta:.4f}", flush=True)
    return gd, llr, logb


def main():
    z = np.load(os.path.join(LOCAL, "features.npz"))
    fr, Xs = z["frames"], z["Xs"]
    S = {k: Xs[:, i].astype(np.float64) for i, k in enumerate(z["names_s"])}
    ev = dict(np.load(os.path.join(LOCAL, "events.npz")))
    ch = channels(S, ev, fr)
    seg = np.r_[0, np.flatnonzero(np.diff(fr) > 1) + 1, len(fr)]
    t5 = S["top5_w75"]
    g0 = np.where(t5 < 1.5, 0.95, np.where(t5 > 3.0, 0.05, 0.4))
    gd, llr, logb = fit(ch, seg, g0)
    out = np.full(N, np.nan)
    out[fr] = gd
    np.save(os.path.join(LOCAL, "p_dead.npy"), out)
    ll = np.full(N, np.nan)
    ll[fr] = logb[:, 1] - logb[:, 0]
    np.save(os.path.join(LOCAL, "llr_dead.npy"), ll)
    for k, v in llr.items():
        print(f"  {k:11s} LLR range {v.min():+.2f} .. {v.max():+.2f}")


if __name__ == "__main__":
    main()
