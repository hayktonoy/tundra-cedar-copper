"""Label-free XY features and sensor degradation. No fitted artefact imports."""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

warnings.filterwarnings('ignore', category=RuntimeWarning)
FPS = 25


def shift(a, k):
    out = np.full_like(a, np.nan, dtype='f4')
    if k == 0:
        return a.copy()
    if 0 < k < len(a):
        out[:-k] = a[k:]
    elif -len(a) < k < 0:
        out[-k:] = a[:k]
    return out


def ages(hit):
    t = np.arange(len(hit))
    prev = np.maximum.accumulate(np.where(hit, t, -1000000))
    nxt = np.minimum.accumulate(np.where(hit, t, 1000000)[::-1])[::-1]
    return np.minimum((t-prev)/25, 60), np.minimum((nxt-t)/25, 60)


def interp_ball(b):
    t = np.arange(len(b)); ok = np.isfinite(b).all(1)
    out = np.full_like(b, np.nan)
    if ok.any():
        for j in range(2):
            out[:, j] = np.interp(t, t[ok], b[ok, j])
    return out


def smooth(a, w):
    valid = np.isfinite(a)
    mass = uniform_filter1d(valid.astype('f4'), w, axis=0, mode='nearest')
    out = uniform_filter1d(np.nan_to_num(a).astype('f4'), w, axis=0, mode='nearest')
    return np.where(mass > .05, out/np.maximum(mass, .001), np.nan)


def degrade(p, b, sensor, seed):
    """Only XY and unlabeled target sensor stats are inputs; no DFL targets."""
    rng = np.random.default_rng(seed)
    p = p.copy(); b = b.copy(); n = len(b)
    ids = (np.arange(n)+rng.integers(len(sensor['mask']))) % len(sensor['mask'])
    # Curvature residual is an upper-bound noise proxy, NOT true tracking error.
    drift = sensor['residual'][ids].copy()
    angle = rng.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    b += (drift @ rot).astype('f4')
    b += rng.normal(0, .08, b.shape).astype('f4')
    p += gaussian_filter1d(rng.normal(0, .45, p.shape).astype('f4'), 3, axis=0)
    center = interp_ball(b)
    # Retain nearby players when the camera sees fewer than a full team. The
    # empirical count paths preserve contiguous missing and duplicate intervals.
    for team in (0, 1):
        count_ids = (np.arange(n)+rng.integers(len(sensor['counts']))) % len(sensor['counts'])
        desired = np.clip(sensor['counts'][count_ids, team], 0, 14)
        valid = np.isfinite(p[:, team]).all(2)
        dist = np.linalg.norm(p[:, team]-center[:, None], axis=2)
        order = np.argsort(np.where(valid, dist, np.inf), axis=1)
        rank = np.argsort(order, axis=1)
        p[:, team][rank >= desired[:, None]] = np.nan
        # Duplicate into an unused slot, retaining coordinates close to source.
        for extra in range(3):
            source = order[:, extra]
            rows = np.flatnonzero(desired > valid.sum(1)+extra)
            for row in rows:
                empty = np.flatnonzero(~np.isfinite(p[row, team]).all(1))
                if len(empty):
                    p[row, team, empty[0]] = p[row, team, source[row]] + rng.normal(0, .3, 2)
    b[~sensor['mask'][ids]] = np.nan
    return p, b


def raw_features(p, b):
    """Coordinates in metres, centered pitch; no identities, clock or half ID."""
    cols = {}
    def put(name, a):
        cols[name] = np.asarray(a, dtype='f4')
    n = len(b); present = np.isfinite(b).all(1)
    filled = interp_ball(b)
    before, after = ages(present)
    put('ball_observed', present)
    put('ball_prev_age', before); put('ball_next_age', after)
    put('ball_gap', before+after)
    for label, bb in [('raw', b), ('fill', filled)]:
        bv = (shift(bb, 5)-shift(bb, -5))*2.5
        bs = np.minimum(np.linalg.norm(bv, axis=1), 100)
        put(label+'_ball_speed', bs)
        put(label+'_ball_edge_x', np.abs(bb[:, 0])-52.5)
        put(label+'_ball_edge_y', np.abs(bb[:, 1])-34)
        for w in (25, 75, 251):
            put(f'{label}_speed_w{w}', smooth(bs, w))
        for k in (12, 25, 75):
            put(f'{label}_ball_travel{k}', np.minimum(np.linalg.norm(shift(bb, k)-shift(bb, -k), axis=1), 100))
        for team, tag in enumerate(('h', 'a')):
            pp = p[:, team]
            d = np.linalg.norm(pp-bb[:, None], axis=2)
            d[~np.isfinite(d)] = np.inf
            order = np.argsort(d, axis=1)
            dd = np.take_along_axis(d, order, axis=1)
            dd[~np.isfinite(dd)] = np.nan
            for rank in (0, 1, 2):
                put(f'{tag}_{label}_d{rank}', np.minimum(dd[:, rank], 70))
            for r in (1, 2, 4, 8):
                put(f'{tag}_{label}_count{r}', np.where(np.isfinite(bb).all(1), (d<r).sum(1), np.nan))
            if label == 'raw':
                vel = (shift(pp, 6)-shift(pp, -6))*(25/12)
                vel[np.linalg.norm(vel, axis=2)>14] = np.nan
                nearv = np.take_along_axis(vel, order[:, :1, None], axis=1)[:, 0]
                put(f'{tag}_near_speed', np.where(present, np.linalg.norm(nearv, axis=1), np.nan))
                put(f'{tag}_near_relative', np.linalg.norm(nearv-bv, axis=1))
                center = np.nanmean(pp, axis=1); delta = pp-center[:, None]
                speed = np.linalg.norm(vel, axis=2)
                put(f'{tag}_observed', np.isfinite(pp).all(2).sum(1))
                put(f'{tag}_speed', np.nanmean(speed, axis=1))
                put(f'{tag}_speed_median', np.nanmedian(speed, axis=1))
                put(f'{tag}_fast', np.nanmean(speed>3, axis=1))
                put(f'{tag}_expand', np.nanmean(np.sum(vel*delta, axis=2)/np.maximum(np.linalg.norm(delta, axis=2), 1), axis=1))
                put(f'{tag}_spread_x', np.nanstd(pp[:, :, 0], axis=1))
                put(f'{tag}_spread_y', np.nanstd(pp[:, :, 1], axis=1))
                put(f'{tag}_center_ball', np.linalg.norm(center-bb, axis=1))
                put(f'{tag}_ball_approach', np.nanmean(np.sum(vel*(bb[:, None]-pp), axis=2)/np.maximum(d, 1), axis=1))
                for k in (12, 25, 75):
                    put(f'{tag}_center_speed{k}', np.linalg.norm(shift(center, k)-shift(center, -k), axis=1)*(25/(2*k)))
    for tag, other in [('h', 'a'), ('a', 'h')]:
        for label in ('raw', 'fill'):
            d = cols[f'{tag}_{label}_d0']; od = cols[f'{other}_{label}_d0']
            for r in (1.5, 3):
                contact = (d<r)&(d+.3<od)
                if label == 'fill':
                    contact &= (before+after)<3
                for key, a in zip(('prev', 'next'), ages(contact)):
                    put(f'{tag}_{label}_contact{r}_{key}', a)
                for w in (25, 75, 251):
                    put(f'{tag}_{label}_contact{r}_w{w}', uniform_filter1d(contact.astype('f4'), w, mode='nearest'))
        for name in ('raw_d0', 'fill_d0', 'speed', 'expand', 'ball_approach', 'near_relative'):
            a = cols[f'{tag}_{name}']
            for k in (-125, -50, -25, 25, 50, 125):
                put(f'{tag}_{name}_lag{k}', shift(a, k))
            for w in (25, 75, 251):
                put(f'{tag}_{name}_w{w}', smooth(a, w))
    names = list(cols)
    x = np.column_stack([cols[k] for k in names]).astype('f4')
    x[~np.isfinite(x)] = np.nan
    return x, names


def swap_index(names):
    return np.array([names.index(('a_'+n[2:]) if n.startswith('h_') else ('h_'+n[2:]) if n.startswith('a_') else n) for n in names])


def status_features(x, names):
    """Exactly team invariant dead/live head, including both min and max."""
    cols = []
    for j, name in enumerate(names):
        if name.startswith('a_'):
            continue
        if name.startswith('h_'):
            other = x[:, names.index('a_'+name[2:])]
            cols.extend((np.fmin(x[:, j], other), np.fmax(x[:, j], other)))
        else:
            cols.append(x[:, j])
    return np.column_stack(cols).astype('f4')
