"""Independent DFL fits, nested validation, and deterministic challenge replay."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '3'
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import lightgbm as lgb
from scipy.ndimage import uniform_filter1d
from features import raw_features, degrade, shift, swap_index, status_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = HERE/'.local'
DATA = ROOT/'pipeline/dfl-owner/.local/data'
KEYS = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')
HALVES = ('firstHalf', 'secondHalf')
STEP = 5
sys.path.insert(0, str(ROOT/'official/evaluation'))
import possession_metrics as pm
sys.path.insert(0, str(ROOT/'official'))
from games import half_video_blocks
BLOCKS = [(a, b+1) for _, a, b in half_video_blocks('BAR-ATH')]
warnings.filterwarnings('ignore', category=RuntimeWarning)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda a: a.item() if isinstance(a, np.generic) else a.tolist())+'\n', encoding='utf-8')


def bar_parse():
    cache = LOCAL/'bar.npz'
    if not cache.exists():
        tac = ROOT/'official/data/tactical'
        meta = ET.parse(tac/'BAR-ATH_tactical_atd.xml').getroot()
        tids = [t.get('id') for t in meta.iter('Team')]
        team = np.array([tids.index(p.get('teamId')) for p in meta.iter('Player')])
        players = np.full((199319, 2, 32, 2), np.nan, 'f4')
        ball = np.full((199319, 2), np.nan, 'f4')
        seen = np.zeros(199319, bool)
        with (tac/'BAR-ATH_tactical_atd.txt').open(encoding='utf-8-sig') as f:
            for line in f:
                row = np.fromstring(line.replace(':', ',').replace(';', ','), sep=',', dtype='f4')
                assert len(row) == 142
                i = int(row[0])-1
                assert not seen[i]
                seen[i] = True
                pp = row[1:139].reshape(46, 3)[:, :2]*[105.3, 68]-[52.65, 34]
                for t in (0, 1):
                    players[i, t, :(team==t).sum()] = pp[team==t]
                ball[i] = row[139:141]*[105.3, 68]-[52.65, 34]
        assert seen.all()
        np.savez_compressed(cache, players=players, ball=ball)
    return np.load(cache)


def prepare():
    LOCAL.mkdir(parents=True, exist_ok=True)
    z = bar_parse(); p, b = z['players'], z['ball']
    ix = np.concatenate([np.arange(a, e) for a, e in BLOCKS])
    mask = np.isfinite(b[ix]).all(1)
    counts = np.isfinite(p[ix]).all(3).sum(2).astype('i1')
    residual = np.zeros_like(b)
    for a, e in BLOCKS:
        bb = b[a:e]
        rr = bb-(shift(bb, 5)+shift(bb, -5))/2
        rr[~np.isfinite(rr).all(1)] = 0
        # Long tails contain real kicks; a bounded proxy avoids teaching that
        # every native DFL frame should have a severe projection error.
        rr *= np.minimum(1, 3/np.maximum(np.linalg.norm(rr, axis=1), .001))[:, None]
        residual[a:e] = rr
    np.savez_compressed(LOCAL/'sensor.npz', mask=mask, counts=counts, residual=residual[ix])
    save(LOCAL/'sensor_summary.json', dict(availability=mask.mean(), frames=len(ix),
        player_counts_quantiles=np.quantile(counts, [.05, .5, .95], axis=0),
        curvature_residual_m=np.quantile(np.linalg.norm(residual[ix], axis=1), [.5, .9, .95, .99]),
        limitation='Short-time curvature is a label-free corruption proxy, not measured sensor error.'))
    del p, b, z
    sensor = np.load(LOCAL/'sensor.npz')
    for mi, key in enumerate(KEYS):
        source = DATA/f'{key}.npz'
        z = np.load(source)
        allp, allb = z['players'], z['ball'][:, :2]
        allowner, allstatus, halfid = z['possession'], z['status'], z['half']
        for hi, half in enumerate(HALVES):
            p, b = allp[halfid==hi], allb[halfid==hi]
            owner, status = allowner[halfid==hi], allstatus[halfid==hi]
            y = np.where(status==0, 2, np.where((status==1)&np.isin(owner, [1, 2]), owner-1, 3)).astype('i1')
            out = LOCAL/'features'/key/half; out.mkdir(parents=True, exist_ok=True)
            np.save(out/'y.npy', y)
            for view in ('train0', 'train1', 'test'):
                target = out/f'{view}.npy'
                if target.exists():
                    continue
                vi = ('train0', 'train1', 'test').index(view)
                pp, bb = degrade(p, b, sensor, 20261029+100*mi+10*hi+vi)
                x, names = raw_features(pp, bb)
                np.save(target, x[::STEP])
                save(LOCAL/'feature_names.json', names)
                print('features', key, half, view, x.shape, flush=True)
                del pp, bb, x; gc.collect()


def load_x(key, hi, view):
    return np.load(LOCAL/'features'/key/HALVES[hi]/f'{view}.npy', mmap_mode='r')


def load_y(key, hi):
    return np.load(LOCAL/'features'/key/HALVES[hi]/'y.npy')


def fit(keys, tag):
    out = LOCAL/'models'/tag; out.mkdir(parents=True, exist_ok=True)
    if (out/'done.json').exists():
        return [lgb.Booster(model_file=str(out/f'{head}.txt')) for head in ('dead', 'owner')]
    names = json.loads((LOCAL/'feature_names.json').read_text())
    xs, ys = [], []
    for key in keys:
        for hi in range(2):
            y = load_y(key, hi)[::STEP]
            for view in ('train0', 'train1'):
                xs.append(np.array(load_x(key, hi, view))); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    del xs, ys; gc.collect()
    params = dict(objective='binary', metric='binary_logloss', learning_rate=.055,
                  num_leaves=31, min_data_in_leaf=100, max_bin=127, num_threads=3,
                  feature_fraction=.9, bagging_fraction=.85, bagging_freq=1,
                  lambda_l2=2, seed=20261029, verbosity=-1, force_col_wise=True)
    heads = []; timings = {}
    for head in ('dead', 'owner'):
        t = time.perf_counter()
        if head == 'dead':
            valid = y<3; xx = status_features(x[valid], names); yy = (y[valid]==2).astype('i1')
        else:
            valid = y<2; xx = x[valid]; yy = (y[valid]==0).astype('i1')
            xx = np.concatenate([xx, xx[:, swap_index(names)]])
            yy = np.r_[yy, 1-yy]
        ds = lgb.Dataset(xx, label=yy, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=350)
        model.save_model(str(out/f'{head}.txt'))
        timings[head] = dict(seconds=time.perf_counter()-t, rows=len(yy), features=xx.shape[1])
        heads.append(model)
        del ds, xx, yy; gc.collect()
        print('fit', tag, head, timings[head], flush=True)
    del x, y; gc.collect()
    save(out/'done.json', dict(keys=keys, timings=timings, params=params, rounds=350))
    return heads


def predict(models, x, names):
    dead = models[0].predict(status_features(x, names), num_threads=3)
    h1 = models[1].predict(x, num_threads=3)
    h2 = models[1].predict(x[:, swap_index(names)], num_threads=3)
    home = .5*(h1+1-h2)
    return np.column_stack([dead, home]).astype('f4')


def runs(y):
    starts = np.r_[0, np.flatnonzero(y[1:]!=y[:-1])+1]
    return list(zip(starts, np.r_[starts[1:], len(y)], y[starts]))


def decode(prob, cfg):
    width, threshold, minimum = cfg
    pp = uniform_filter1d(prob, width, axis=0, mode='nearest') if width>1 else prob
    dead, home = pp.T
    y = np.where(dead>threshold, 2, np.where(home>.5, 0, 1)).astype('i1')
    # Remove shortest unsupported excursions first; compare integrated head
    # confidence if the two flanking states differ. Never use reference status.
    for _ in range(20):
        rr = runs(y); changed = False
        for j, (a, b, c) in enumerate(rr):
            if b-a >= minimum:
                continue
            neighbors = [rr[k][2] for k in (j-1, j+1) if 0 <= k < len(rr)]
            if not neighbors:
                continue
            scores = [np.mean(np.log(np.clip(dead[a:b] if v==2 else (1-dead[a:b])*(home[a:b] if v==0 else 1-home[a:b]), 1e-5, 1))) for v in neighbors]
            y[a:b] = neighbors[int(np.argmax(scores))]; changed = True
        if not changed:
            break
    return y


def native(prob, n):
    return np.column_stack([np.interp(np.arange(n), np.arange(len(prob))*STEP, prob[:, j]) for j in range(2)]).astype('f4')


def score(y, pred):
    vr = pm.valid_runs(y)
    sp = pm.spell_scores(y, pm.fill_abstentions(pred, vr), vr)
    confusion = pm.scored_confusion(pm.confusion(y, pred))
    macro = float(pm.macro_f1(confusion))
    return dict(spell=float(sp['mean_f1']), macro=macro)


def evaluate_match(key, probs, cfg):
    # X separator prevents linking the two source halves.
    ys, ps = [], []
    for hi in range(2):
        y = load_y(key, hi)
        pred5 = decode(probs[hi], cfg)
        pred = pred5[np.minimum(np.arange(len(y))//STEP, len(pred5)-1)]
        ys.extend([y, np.array([3], 'i1')]); ps.extend([pred, np.array([3], 'i1')])
    return score(np.concatenate(ys), np.concatenate(ps))


GRID = [(w, t, m) for w in (1, 3, 7, 11) for t in (.35, .5, .65) for m in (1, 3, 5)]


def validation():
    names = json.loads((LOCAL/'feature_names.json').read_text()); oof = {}
    for outer, key in enumerate(KEYS):
        inner_key = KEYS[(outer+1)%len(KEYS)]
        inner_tag = 'inner_'+key
        models = fit([k for k in KEYS if k not in (key, inner_key)], inner_tag)
        probs = [predict(models, load_x(inner_key, hi, 'test'), names) for hi in range(2)]
        grid = [dict(cfg=cfg, **evaluate_match(inner_key, probs, cfg)) for cfg in GRID]
        selected = max(grid, key=lambda r: (r['spell'], r['macro']))
        save(LOCAL/'validation'/f'{key}_inner.json', dict(outer=key, validation=inner_key, selected=selected['cfg']))
        del models, probs; gc.collect()
        models = fit([k for k in KEYS if k!=key], 'outer_'+key)
        probs = [predict(models, load_x(key, hi, 'test'), names) for hi in range(2)]
        for hi, pp in enumerate(probs):
            (LOCAL/'oof').mkdir(exist_ok=True)
            np.save(LOCAL/'oof'/f'{key}_{hi}.npy', pp)
        oof[key] = probs
        print('OUTER', key, flush=True)
        del models; gc.collect()
    grid = []
    for cfg in GRID:
        results = [evaluate_match(key, oof[key], cfg) for key in KEYS]
        grid.append(dict(cfg=cfg, spell=np.mean([r['spell'] for r in results]), macro=np.mean([r['macro'] for r in results])))
    best = max(grid, key=lambda r: (r['spell'], r['macro']))
    save(LOCAL/'decoder.json', dict(selected=dict(cfg=best['cfg'])))
    print('DFL decoder settings', best['cfg'], flush=True)


def write_csv(path, y):
    with path.open('w', newline='\n') as f:
        f.write('frame,label\n')
        f.writelines(f'{i},{"HAD"[v]}\n' for i, v in enumerate(y))


def transfer():
    names = json.loads((LOCAL/'feature_names.json').read_text())
    cfg = json.loads((LOCAL/'decoder.json').read_text())['selected']['cfg']
    models = fit(list(KEYS), 'all')
    z = bar_parse(); players, ball = z['players'], z['ball']
    evidence = np.zeros((199319, 2), 'f4'); evidence[:, 0] = 1; evidence[:, 1] = .5
    calibrated = evidence.copy()
    y = np.full(199319, 2, 'i1')
    # Quantile calibration uses motion inputs only, pooled teams; no target
    # labels. It is a separate inference recipe, not a new learned head.
    motion = [i for i, name in enumerate(names) if name.startswith('raw_ball_speed') or name.startswith('raw_speed_w') or name.startswith('raw_ball_travel') or 'near_relative' in name]
    source_motion = np.concatenate([np.array(load_x(k, h, 'test')[:, motion]) for k in KEYS for h in range(2)])
    half_x = []
    for a, b in BLOCKS:
        x, nn = raw_features(players[a:b], ball[a:b]); assert nn == names
        half_x.append(x[::STEP].copy())
        del x
    target_motion = np.concatenate([x[:, motion] for x in half_x])
    maps = {}; levels = np.linspace(.001, .999, 201)
    for j, column in enumerate(motion):
        name = names[column]
        paired = [j]
        if name.startswith(('h_', 'a_')):
            other = names.index(('a_' if name.startswith('h_') else 'h_')+name[2:])
            paired.append(motion.index(other))
        src = source_motion[:, paired].ravel(); tgt = target_motion[:, paired].ravel()
        sq = np.quantile(src[np.isfinite(src)], levels)
        tq = np.quantile(tgt[np.isfinite(tgt)], levels)
        unique, ix = np.unique(tq, return_index=True)
        maps[column] = (unique, sq[ix])
    for (a, b), x in zip(BLOCKS, half_x):
        # Predict at the same grid as all outer evaluations.
        pp = predict(models, x, names)
        evidence[a:b] = native(pp, b-a)
        decoded = decode(pp, cfg)
        y[a:b] = decoded[np.minimum(np.arange(b-a)//STEP, len(decoded)-1)]
        cx = x.copy()
        for column, (tq, sq) in maps.items():
            valid = np.isfinite(cx[:, column])
            cx[valid, column] = np.interp(cx[valid, column], tq, sq)
        calibrated[a:b] = native(predict(models, cx, names), b-a)
    np.savez_compressed(LOCAL/'evidence.npz', p_dead=evidence[:, 0], p_home_live=evidence[:, 1],
        p_home_given_live=evidence[:, 1], fps=np.array(25), frame=np.arange(len(y)))
    np.savez_compressed(LOCAL/'calibrated_evidence.npz', p_dead=calibrated[:, 0], p_home_live=calibrated[:, 1])
    save(LOCAL/'calibration.json', {names[k]:dict(target_quantiles=v[0], source_quantiles=v[1]) for k,v in maps.items()})
    write_csv(LOCAL/'best.csv', y)
    save(LOCAL/'frozen.json', dict(decoder=cfg))


def prefit():
    """Optional second 3-thread worker; populate only the final three folds.

    Start beside validate, before it reaches these folds; wait for this worker
    before running transfer. Existing done markers make the normal path reuse
    these exact fits.
    """
    for key in reversed(KEYS[-3:]):
        i = KEYS.index(key); inner_key = KEYS[(i+1)%len(KEYS)]
        for tag, keys in [('inner_'+key, [k for k in KEYS if k not in (key, inner_key)]),
                          ('outer_'+key, [k for k in KEYS if k!=key])]:
            models = fit(keys, tag); del models; gc.collect()


def refit():
    models = fit(list(KEYS), 'all')
    del models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'validate', 'transfer', 'prefit', 'refit'])
    args = parser.parse_args(); t = time.perf_counter()
    {'prepare':prepare, 'validate':validation, 'transfer':transfer, 'prefit':prefit, 'refit':refit}[args.stage]()
    if os.name == 'nt':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)]+[(k, ctypes.c_size_t) for k in ('peak', 'rss', 'pp', 'p', 'pn', 'nn', 'pf', 'peakpf')]
        value = Memory(); value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb)
        peak = value.peak
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    save(LOCAL/f'{args.stage}_resources.json', dict(seconds=time.perf_counter()-t, peak_rss_bytes=peak, pid=os.getpid()))
