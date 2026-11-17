"""Replay the free-merge chain using the constants in recipe/*.json.

    uv run --no-project --python 3.13 --with numpy==2.3.2 --with pandas==2.3.2 --with scipy==1.18.1 \
        --with numba==0.68.1 python pipeline/fm-rebuild/replay.py [--out DIR] [--swap LANEFILE=FILE ...]

Three processes, as the lanes run them (their modules share names such as run, common, decode):

1. derive: the label-free caches that sit between the sensor stages and the decoder, rebuilt with
   the lanes' own code into DIR: free-rules' 25 fold-independent channels (rules_inputs.npz, the
   code of agreed-touches/rules_inputs_check.py), free-brief's detectors (detect.npz), free-grammar's
   touch candidates (touches.npz, prepare.py up to its decoder snapshots) and its Baum-Welch owner
   prior (em_prior.json, learn_prior.py). --cached skips this and copies the lanes' saved files.
2. decode: free-pretrained's decoder (free-status Engine + spotter terms, free-owner Viterbi) with
   recipe/decoder.json: the fold-4 caches free-grammar/prepare.py and free-rules/evidence.py would
   write (fold4.npz; base3/post_dead3/post_team3 in inputs.npz).
3. rules: free-stack's Decoder (free-grammar's team grammar fused into the decoder,
   recipe/grammar.json), then every rule stage of free-stack, free-stoppages and free-merge
   (recipe/rules.json) through free-stable's Engine, which calls each lane's own apply().

--swap A=B makes every np.load of lane file A read B instead (A relative to pipeline/), e.g.
empty spotter stand-ins.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'

HERE = Path(__file__).resolve().parent
# numba's on-disk cache records the module a function was compiled in; this replay loads lane files
# under other names (fm_learn_prior, ...), so its caches go to its own folder, or the lanes' own
# scripts would later load them and fail to import those names. The phases inherit this setting.
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.local' / 'numba-cache'))
EXP = HERE.parent
ROOT = EXP.parent
RECIPE = HERE / 'recipe'
N = 199319
sys.path.insert(0, str(HERE))

# Derived caches: the lane file a later stage reads -> this replay's rebuilt copy.
DERIVED = {'free-rules/.local/inputs.npz': 'rules_inputs.npz', 'free-brief/.local/detect.npz': 'detect.npz',
           'free-grammar/.local/touches.npz': 'touches.npz', 'free-grammar/.local/em_prior.json': 'em_prior.json'}


class Stop(Exception):
    pass


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read(name):
    return json.loads((RECIPE / f'{name}.json').read_text())


def install_swaps(swaps):
    """Redirect np.load of the given lane files (the lanes read fixed paths)."""
    import numpy as np
    table = {Path(EXP / a).resolve(): Path(b).resolve() for a, b in swaps.items()}
    if not table:
        return
    real = np.load

    def patched(path, *a, **k):
        if isinstance(path, (str, Path)):
            path = table.get(Path(path).resolve(), path)
        return real(path, *a, **k)
    np.load = patched


def derive_phase(out):
    import numpy as np
    # Build the fold-independent rule inputs from the sensor caches.
    src = (EXP / 'agreed-touches/rules_inputs_check.py').read_text(encoding='utf-8')
    ns = {'__file__': str(EXP / 'agreed-touches/rules_inputs_check.py'), '__name__': 'fm_rules_inputs'}
    exec(compile(src, 'rules_inputs_check.py', 'exec'), ns)
    np.savez_compressed(out / 'rules_inputs.npz', **ns['out'])
    # free-brief's physical detectors, written to this replay's folder.
    shutil.copyfile(EXP / 'free-brief/.local/contact_broadcast.npz', out / 'contact_broadcast.npz')
    sys.path.insert(0, str(EXP / 'free-brief'))
    fb = load('fb', EXP / 'free-brief/fb.py')
    fb.LOCAL = out
    load('fm_detect', EXP / 'free-brief/detect.py').main()
    # Build touch candidates here and write the deployment decoder snapshot in decode.
    sys.path.insert(0, str(EXP / 'free-grammar'))
    gp = load('fm_grammar_prepare', EXP / 'free-grammar/prepare.py')
    gp.LOCAL = out

    def stop():
        raise Stop
    gp.PEngine = stop
    try:
        gp.main()
    except Stop:
        pass
    lp = load('fm_learn_prior', EXP / 'free-grammar/learn_prior.py')
    lp.LOCAL = out
    lp.main()


def decode_phase(out, derived):
    import numpy as np
    handset = load('handset_prepare', EXP / 'free-handset/prepare.py')  # free-pretrained PEngine + bonus rescale
    dec = read('decoder')
    eng = handset.HEngine(dec['bonus'])
    s, d, beta, cin, cout = eng.scores(dec['decoder'])
    base = handset.decode(s, d, beta, cin, cout).astype('i1')
    base[base == 3] = 2
    # As free-grammar/prepare.py writes fold{k+1}.npz for fold k = 3.
    np.savez_compressed(out / 'fold4.npz', s=s, dead=d - .5 * np.abs(s), cin=cin, cout=cout, base=base,
                        shape=eng.owner.lq)
    m = handset.posterior(s, d, beta, cin, cout)
    # free-rules/evidence.py's inputs.npz: the fold-independent channels plus fold 3's decode.
    z = {k: v for k, v in np.load(EXP / 'free-rules/.local/inputs.npz').items() if k[-1] not in '0123'}
    z['base3'] = base
    z['post_dead3'] = np.nan_to_num(m[:, 2], nan=1.).astype('f4')
    z['post_team3'] = np.divide(m[:, 0], m[:, :2].sum(1), out=np.full(N, .5),
                                where=m[:, :2].sum(1) > 1e-30).astype('f4')
    np.savez_compressed(out / 'inputs.npz', **z)
    if not derived:
        for src in ('free-grammar/.local/touches.npz', 'free-grammar/.local/em_prior.json', 'free-brief/.local/detect.npz'):
            shutil.copyfile(EXP / src, out / Path(src).name)


def rules_phase(out):
    stable = load('fm_stable', EXP / 'free-stable/run.py')
    for mod in (sys.modules['grammar'], sys.modules['evidence'], stable.stack):
        mod.LOCAL = out  # read this replay's caches, never the lanes' saved fold files
    g = read('grammar')
    st = g['stack']
    cfg = dict(kind=st['kind'], beta=st['beta'], brief=st['brief'], local=st['local_mix'],
               wt=g['fusion']['wt'], wd=g['fusion']['wd'], grammar_params=g['grammar'])
    k = 3  # block-4 fold slot: constants chosen on blocks 1-3 are the deployment
    p = stable.stack.Decoder(k).run(cfg)
    stable.write(out / 'stack_decoder.csv', p)
    eng = stable.Engine()
    for row in read('rules')['rules']:
        p = eng.apply(k, p, row['stage'], row['config'])
    stable.write(out / 'best.csv', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('derive', 'decode', 'rules'))
    ap.add_argument('--out', type=Path, default=HERE / '.local/replay')
    ap.add_argument('--swap', action='append', default=[], help='LANEFILE=FILE, LANEFILE relative to pipeline/')
    ap.add_argument('--cached', action='store_true', help="use the lanes' saved derived caches")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    swaps = dict(s.split('=', 1) for s in a.swap)
    if a.phase:
        derived = not a.cached
        if derived and a.phase != 'derive':
            swaps.update({k: str(out / v) for k, v in DERIVED.items() if k.endswith('.npz')})
        install_swaps(swaps)
        if a.phase == 'derive':
            derive_phase(out)
        elif a.phase == 'decode':
            decode_phase(out, derived)
        else:
            rules_phase(out)
        return
    t0 = time.monotonic()
    times = {}
    phases = ('decode', 'rules') if a.cached else ('derive', 'decode', 'rules')
    for phase in phases:
        t = time.monotonic()
        cmd = [sys.executable, __file__, '--phase', phase, '--out', str(out)] + sum([['--swap', s] for s in a.swap], [])
        if a.cached:
            cmd.append('--cached')
        subprocess.run(cmd, check=True, cwd=ROOT)
        times[phase] = round(time.monotonic() - t, 1)
    receipt = dict(seconds=times, total_seconds=round(time.monotonic() - t0, 1),
                   swaps=swaps, derived=not a.cached)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('best.csv written in', receipt['total_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
