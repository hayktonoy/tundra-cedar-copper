"""Three-state Viterbi decoder (H, A, D) with event-dependent stoppage entry and exit costs.

Per frame t the path collects an emission score: +s[t] / 2 in H, -s[t] / 2 in A (s = log-odds
of home vs away from the team evidence) and e[t] in D (log-odds of dead vs live). Transitions
cost: H <-> A `beta`; live -> D `cin[t]`; D -> live `cout[t]`, into either team (the restart team
is free, decided by what follows).
"""
import numba
import numpy as np

from common import D, HALVES, N, X


@numba.njit(cache=True)
def _viterbi(s, e, beta, cin, cout):
    n = len(s)
    V = np.zeros(3)
    bp = np.zeros((n, 3), np.int8)
    for t in range(n):
        h, a, d = V[0], V[1], V[2]
        # into H
        best, arg = h, 0
        if a - beta > best:
            best, arg = a - beta, 1
        if d - cout[t] > best:
            best, arg = d - cout[t], 2
        nh, bh = best, arg
        # into A
        best, arg = a, 1
        if h - beta > best:
            best, arg = h - beta, 0
        if d - cout[t] > best:
            best, arg = d - cout[t], 2
        na, ba = best, arg
        # into D
        best, arg = d, 2
        if h - cin[t] > best:
            best, arg = h - cin[t], 0
        if a - cin[t] > best:
            best, arg = a - cin[t], 1
        nd, bd = best, arg
        bp[t, 0], bp[t, 1], bp[t, 2] = bh, ba, bd
        V[0] = nh + 0.5 * s[t]
        V[1] = na - 0.5 * s[t]
        V[2] = nd + e[t]
        m = max(V[0], max(V[1], V[2]))
        V -= m
    out = np.zeros(n, np.int8)
    st = 0
    if V[1] > V[st]:
        st = 1
    if V[2] > V[st]:
        st = 2
    for t in range(n - 1, -1, -1):
        out[t] = st
        st = bp[t, st]
    return out


def decode(s, e, beta, cin, cout):
    """s, e, cin, cout: arrays of length N (cin/cout may be scalars). Returns codes, X outside the halves."""
    cin = np.broadcast_to(np.asarray(cin, np.float64), (N,))
    cout = np.broadcast_to(np.asarray(cout, np.float64), (N,))
    pred = np.full(N, X, np.int64)
    for a, b in HALVES:
        pred[a:b] = _viterbi(np.ascontiguousarray(s[a:b], np.float64), np.ascontiguousarray(e[a:b], np.float64),
                             float(beta), np.ascontiguousarray(cin[a:b]), np.ascontiguousarray(cout[a:b]))
    return pred


@numba.njit(cache=True)
def _posterior(s, e, beta, cin, cout):
    """Forward-backward with the decoder's scores as log-potentials: per-frame marginals (n, 3)."""
    n = len(s)
    la = np.zeros((n, 3))
    lb = np.zeros((n, 3))
    em = np.empty((n, 3))
    for t in range(n):
        em[t, 0] = 0.5 * s[t]
        em[t, 1] = -0.5 * s[t]
        em[t, 2] = e[t]
    tr = np.empty((3, 3))

    def lse3(a, b, c):
        m = max(a, max(b, c))
        return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))

    for k in range(3):
        la[0, k] = em[0, k]
    for t in range(1, n):
        tr[0, 0], tr[0, 1], tr[0, 2] = 0.0, -beta, -cin[t]
        tr[1, 0], tr[1, 1], tr[1, 2] = -beta, 0.0, -cin[t]
        tr[2, 0], tr[2, 1], tr[2, 2] = -cout[t], -cout[t], 0.0
        for j in range(3):
            la[t, j] = em[t, j] + lse3(la[t - 1, 0] + tr[0, j], la[t - 1, 1] + tr[1, j], la[t - 1, 2] + tr[2, j])
        m = max(la[t, 0], max(la[t, 1], la[t, 2]))
        for j in range(3):
            la[t, j] -= m
    for t in range(n - 2, -1, -1):
        tr[0, 0], tr[0, 1], tr[0, 2] = 0.0, -beta, -cin[t + 1]
        tr[1, 0], tr[1, 1], tr[1, 2] = -beta, 0.0, -cin[t + 1]
        tr[2, 0], tr[2, 1], tr[2, 2] = -cout[t + 1], -cout[t + 1], 0.0
        for i in range(3):
            lb[t, i] = lse3(tr[i, 0] + em[t + 1, 0] + lb[t + 1, 0], tr[i, 1] + em[t + 1, 1] + lb[t + 1, 1],
                            tr[i, 2] + em[t + 1, 2] + lb[t + 1, 2])
        m = max(lb[t, 0], max(lb[t, 1], lb[t, 2]))
        for i in range(3):
            lb[t, i] -= m
    out = np.empty((n, 3))
    for t in range(n):
        g = la[t] + lb[t]
        m = max(g[0], max(g[1], g[2]))
        z = np.exp(g[0] - m) + np.exp(g[1] - m) + np.exp(g[2] - m)
        for k in range(3):
            out[t, k] = np.exp(g[k] - m) / z
    return out


def posterior(s, e, beta, cin, cout):
    """Per-frame marginals P(H), P(A), P(D) (n, 3); NaN outside the halves."""
    cin = np.broadcast_to(np.asarray(cin, np.float64), (N,))
    cout = np.broadcast_to(np.asarray(cout, np.float64), (N,))
    out = np.full((N, 3), np.nan)
    for a, b in HALVES:
        out[a:b] = _posterior(np.ascontiguousarray(s[a:b], np.float64), np.ascontiguousarray(e[a:b], np.float64),
                              float(beta), np.ascontiguousarray(cin[a:b]), np.ascontiguousarray(cout[a:b]))
    return out
