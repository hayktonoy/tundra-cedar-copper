"""Build the last-touch pipeline's output from the raw match data, in one command.

    uv run --no-project python pipeline/last-touch/build.py                # everything, in order
    uv run --no-project python pipeline/last-touch/build.py --from stubs   # resume at a stage
    uv run --no-project python pipeline/last-touch/build.py --list

Needs the match data in official/data/{tactical,broadcast}, uv, ffmpeg on PATH and a CUDA GPU.
The sensor, Bundesliga and action-spotter stages are the ones pipeline/from-raw/build.py runs (the
spotter with deterministic cuDNN kernels); the external model is replaced by an empty stand-in (its
weight is 0), and the dead/live head is free-dfl's LightGBM head. The base is the free-merge
chain replayed from the frozen recipe files by pipeline/fm-rebuild/replay.py. The last stage,
touches.py, applies the stoppage-start rule, the lag and the touch stage.
Stage times and output paths go to .local/manifest.json.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
spec = importlib.util.spec_from_file_location('from_raw_build', EXP / 'from-raw/build.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
B.LOCAL = HERE / '.local'
B.LOGS = B.LOCAL / 'logs'
B.MANIFEST = B.LOCAL / 'manifest.json'
L = 'pipeline/'
# Pinned dependencies for the CPU stages.
PINS = {'numpy': '2.3.2', 'pandas': '2.3.2', 'scipy': '1.18.1', 'numba': '0.68.1', 'lightgbm': '4.7.0', 'pyarrow': '26.0.1',
        'scikit-learn': '1.9.1'}


def uvw(*packages):
    cmd = ['uv', 'run', '--no-project', '--python', '3.13']
    for p in packages:
        cmd += ['--with', f'{p}=={PINS[p]}' if p in PINS else p]
    return cmd + ['python']


B.uvw = uvw
B.UV_NP = uvw('numpy')[:-1]


def s_stubs(log):
    B.run([*B.UV_NP, 'python', 'pipeline/last-touch/stubs.py'], B.ROOT, log)


def s_status(log):
    """free-status: the Bundesliga heads on the video ball, then the physical cues."""
    B.run([*B.uvw('numpy', 'scipy', 'lightgbm'), '-W', 'ignore', L + 'free-status/video_transfer.py'], B.ROOT, log)
    B.run([*B.uvw(*B.SCI), '-W', 'ignore', L + 'free-status/physics.py'], B.ROOT, log)


REPLAY = EXP / 'fm-rebuild/.local/build'


def s_base(log):
    """Replay the free-merge chain using the frozen recipes."""
    B.run([*B.uvw(*B.SCI), 'pipeline/fm-rebuild/replay.py', '--out', REPLAY], B.ROOT, log)
    shutil.copyfile(REPLAY / 'best.csv', HERE / '.local/base.csv')


def s_touches(log):
    B.run([*B.uvw(*B.SCI, 'pyarrow'), 'pipeline/last-touch/touches.py'], B.ROOT, log)


keep = {s[0]: s for s in B.STAGES}
ORDER = ['tracking', 'weights', 'hc-video-detect', 'hc-video-free', 'touches-players', 'touches-cpu', 'hc-ball-detect',
         'hc-ball-cpu', 'free-owner', 'dfl-ingest', 'free-dfl', 'free-pretrained']
B.STAGES = [keep[n] for n in ORDER] + [
    ('stubs', 'CPU', s_stubs, [L + 'free-external/.local/model-equal-sc-metrica/target.npz']),
    ('free-status', 'CPU', s_status, [L + 'free-dfl/.local/dfl_only_evidence.npz', L + 'free-status/.local/video_dfl_detected.npz',
                                      L + 'free-status/.local/physics.npz']),
    keep['free-team'], keep['rules'], keep['brief-grammar'], keep['stoppages-audio'],
    ('base', 'CPU', s_base, [L + 'fm-rebuild/.local/build/best.csv', L + 'fm-rebuild/.local/build/receipt.json',
                             L + 'last-touch/.local/base.csv']),
    ('touches', 'CPU', s_touches, [L + 'free-carrier/.local/inputs.npz', L + 'free-carrier/.local/carrier_last_touch.npz',
                                   L + 'last-touch/.local/stoppage_start.json', L + 'last-touch/.local/last_touch.csv']),
]

if __name__ == '__main__':
    B.main()
