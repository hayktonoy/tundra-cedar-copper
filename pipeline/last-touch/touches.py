"""The touch stage on the base, with the settings of settings.json. No labels are read.

    uv run --no-project --with numpy --with pandas --with scipy --with numba --with pyarrow python pipeline/last-touch/touches.py

Fits free-carrier's player-level model at the externally chosen temperature without its spotter
term, moves the base's stoppage starts to after the possessing team's last confident touch (the
stoppage-start rule of settings.json), moves the base's team changes and restarts later by the
lag of settings.json, then runs agreed-touches' stage on the lagged base. Writes
.local/last_touch.csv (the pipeline's output), .local/stoppage_start.json and .local/kept.json.

The stoppage-start rule: where the team in possession before a stoppage still
touches the ball with confidence `level` or more (touch probability times the probability that
the toucher is of that team; set pieces excluded) within `window` frames of the stoppage's start,
the stoppage starts `pad` frames after the last such touch instead, never leaving it shorter than
`min_dead` frames.

The lag: every team change of the base is moved `team`
frames later and every restart `restart` frames later; stoppage starts stay. A boundary is moved
less where the run after it would otherwise fall under 15 frames (or under its own length, if it
was shorter). The touch stage then places each brief possession its own delay after the touch,
so a confident touch next to a hand-over times that hand-over itself.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = HERE / '.local'
S = json.loads((HERE / 'settings.json').read_text())
TAG = 'last_touch'
HALVES = ((11185, 90652), (110808, 190198))
MIN_RUN = 15


def stoppage_start(base, x, window, level, pad, min_dead):
    """The base with stoppage starts moved to just after the possessing team's last confident touch."""
    q = 1 / (1 + np.exp(-(x.da - x.dh) / .3))
    conf = (x.pt * q, x.pt * (1 - q))
    out, moved = base.copy(), []
    for lo, hi in HALVES:
        c = base[lo:hi]
        edges = np.flatnonzero(np.diff(c) != 0) + 1
        for i, s in enumerate(edges):
            if c[s] != 2 or c[s - 1] > 1:
                continue
            team = int(c[s - 1])
            end = int(edges[i + 1]) if i + 1 < len(edges) else len(c)
            a, b = lo + int(s), lo + end
            m = np.flatnonzero((x.fr >= a) & (x.fr <= min(a + window, b - min_dead - pad)) & (conf[team] >= level) & ~x.setpiece)
            if not len(m):
                continue
            new = int(x.fr[m[-1]]) + pad
            if new > a:
                out[a:new] = team
                moved.append(dict(start=a, new_start=new, team='HA'[team]))
    return out, moved


def lagged(codes, team, start, restart):
    """The timeline with team changes, stoppage starts and restarts each moved later by their own number of frames."""
    out = codes.copy()
    for a, b in HALVES:
        c = codes[a:b]
        edges = np.flatnonzero(np.diff(c) != 0) + 1
        starts, ends = np.r_[0, edges], np.r_[edges, len(c)]
        bounds = [0] + [int(s + (start if c[s] == 2 else restart if c[s - 1] == 2 else team)) for s in edges] + [len(c)]
        for i in range(len(bounds) - 2, 0, -1):          # run i keeps min(its length, MIN_RUN) frames
            keep = min(int(ends[i] - starts[i]), MIN_RUN)
            bounds[i] = max(min(bounds[i], bounds[i + 1] - keep), int(starts[i]))
        o = np.empty_like(c)
        for i, s in enumerate(starts):
            o[bounds[i]:bounds[i + 1]] = c[s]
        out[a:b] = o
    return out


def main():
    c = S['carrier']
    py = [sys.executable, '-B', '-W', 'ignore']
    subprocess.run(py + [str(EXP / 'free-carrier/prepare.py')], check=True)     # always: it depends on the tracker
    subprocess.run(py + [str(EXP / 'free-carrier/carrier.py'), '--tag', TAG, '--temp', str(c['temp']),
                         '--spot', str(c['spot']), '--gate', str(c['gate'])], check=True)
    sys.path.insert(0, str(EXP / 'agreed-touches'))
    os.chdir(EXP / 'touches-video')
    import free as F
    unknown = set(S['stage']) - set(F.P) - {'tidy'}
    assert not unknown, f'settings.json has stage keys the stage does not know: {unknown}'
    x = F.Inputs(EXP / f'free-carrier/.local/carrier_{TAG}.npz')
    base = F.pm.label_codes(F.pm.read_label_csv(str(LOCAL / 'base.csv')))
    ss = S['stoppage_start']
    assert set(ss) == {'window', 'level', 'pad', 'min_dead'}, ss
    base, moved = stoppage_start(base, x, int(ss['window']), float(ss['level']), int(ss['pad']), int(ss['min_dead']))
    (LOCAL / 'stoppage_start.json').write_text(json.dumps(moved, indent=1) + '\n')
    shift = [m['new_start'] - m['start'] for m in moved]
    print(f'stoppage-start: {len(moved)} stoppages start later (median {np.median(shift) if shift else 0:.0f} frames)')
    lag = S.get('lag', {})
    assert set(lag) <= {'team', 'start', 'restart'}, lag
    base = lagged(base, int(lag.get('team', 0)), int(lag.get('start', 0)), int(lag.get('restart', 0)))
    out, kept, proposed = F.stage(base, x, dict(F.P, **S['stage']))
    (LOCAL / 'last_touch.csv').write_text('frame,label\n' + ''.join(f'{i},{"HADX"[int(v)]}\n' for i, v in enumerate(out)), newline='\n')
    (LOCAL / 'kept.json').write_text(json.dumps(kept, indent=1) + '\n')
    print(f'proposed {proposed}, kept {len(kept)} ({sum(k["merged"] for k in kept)} merged) -> {LOCAL / "last_touch.csv"}')


if __name__ == '__main__':
    main()
