"""Build the fold-independent inputs for the rule stages."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rules_evidence', EXP / 'free-rules/evidence.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
sys.path.insert(0, str(EXP / 'free-pretrained'))
from fuse import PEngine  # noqa: E402
from scipy.special import expit  # noqa: E402

ev = PEngine()
own = np.load(EXP / 'free-owner/.local/evidence.npz')
out = {}
out['broad'] = R.smooth(own['p_dead_em']).astype('f4')
out['local'] = R.smooth(expit((ev.dfl + ev.video) * .5)).astype('f4')
out['contact'] = own['contact']
out['team'] = expit(.5 * R.odds(own['p_home_shape']) + .25 * ev.dfl_team + .25 * ev.video_team
                    + 2 * own['contact'] + 2 * ev.onball).astype('f4')
out['state'] = np.c_[(1 - out['local']) * out['team'], (1 - out['local']) * (1 - out['team']), out['local']].astype('f4')
out['spot_team'] = ev.onball.astype('f4')
out['spot_live'] = ev.live2.astype('f4')
out['spot_out'] = ev.out.astype('f4')
out['spot_restart'] = ev.restart.astype('f4')
for source, pre in (('provided', 'p'), ('video', 'v'), ('broadcast', 'b')):
    z = np.load(EXP / f'hc-video/.local/feat_{source}.npz')
    for key in ('x', 'y', 'speed', 'dh', 'da'):
        out[f'{pre}_{key}'] = z[f'{pre}_{key}']
x, y, sp = out['p_x'], out['p_y'], out['p_speed']
goal = np.minimum(abs(x), abs(x - 105.3))
near = np.fmin(out['p_dh'], out['p_da'])
out['support'] = np.nan_to_num(expit((16 - goal) / 3) * expit((14 - abs(y - 34)) / 3) * expit((5 - sp) / 2) * expit((5 - near) / 1.5))
