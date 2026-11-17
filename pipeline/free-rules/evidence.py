"""Label-free inputs. No reference reads or training occur in this module.

Each outer fold replays ONE upstream recipe over all four blocks: using the
stitched upstream CV output to select rules would leak upstream selection.
"""
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = HERE / '.local'
LOCAL.mkdir(exist_ok=True)
N = 199319
BLOCKS = ((11185,50919),(50919,90652),(110808,150503),(150503,190198))


def runs(c):
    edges = np.r_[0, np.flatnonzero(c[1:] != c[:-1])+1, len(c)]
    return [(int(a),int(b),int(c[a])) for a,b in zip(edges[:-1],edges[1:])]


def smooth(q, width=5):
    out = np.nan_to_num(q, nan=.5).copy()
    for a,b in BLOCKS:
        out[a:b] = uniform_filter1d(out[a:b], width, mode='nearest')
    return out


def odds(q):
    return logit(np.clip(np.nan_to_num(q,nan=.5),1e-4,1-1e-4))


def prepare():
    """Cache small arrays; models and trained hc-* artifacts are never opened."""
    sys.path.insert(0,str(HERE.parent/'free-pretrained'))
    from fuse import PEngine, posterior
    ev = PEngine()
    recipe = json.loads((HERE.parent/'free-pretrained/recipe.json').read_text())
    outputs = {}
    for k,row in enumerate(recipe['folds']):
        p = row['all_params']
        base = ev.run(p).astype('i1'); base[base==3]=2
        m = posterior(*ev.scores(p))
        outputs[f'base{k}'] = base
        outputs[f'post_dead{k}'] = np.nan_to_num(m[:,2],nan=1.).astype('f4')
        outputs[f'post_team{k}'] = np.divide(m[:,0],m[:,:2].sum(1),out=np.full(N,.5),where=m[:,:2].sum(1)>1e-30).astype('f4')
    own = np.load(HERE.parent/'free-owner/.local/evidence.npz')
    outputs['broad'] = smooth(own['p_dead_em']).astype('f4')
    outputs['local'] = smooth(expit((ev.dfl+ev.video)*.5)).astype('f4')
    outputs['contact'] = own['contact']
    # Fixed evidence blend, not a fitted head. Spotter is broadcast-only.
    outputs['team'] = expit(.5*odds(own['p_home_shape']) + .25*ev.dfl_team + .25*ev.video_team
                            + 2*own['contact'] + 2*ev.onball).astype('f4')
    outputs['state'] = np.c_[(1-outputs['local'])*outputs['team'],
                            (1-outputs['local'])*(1-outputs['team']),outputs['local']].astype('f4')
    outputs['spot_team'] = ev.onball.astype('f4')
    outputs['spot_live'] = ev.live2.astype('f4')
    outputs['spot_out'] = ev.out.astype('f4')
    outputs['spot_restart'] = ev.restart.astype('f4')
    for source,pre in (('provided','p'),('video','v'),('broadcast','b')):
        path = HERE.parent/f'hc-video/.local/feat_{source}.npz'
        z = np.load(path)
        for key in ('x','y','speed','dh','da'):
            outputs[f'{pre}_{key}'] = z[f'{pre}_{key}']
    # A physical keeper-support replacement: slow ball near goal and a player.
    x,y,sp = outputs['p_x'],outputs['p_y'],outputs['p_speed']
    goal = np.minimum(abs(x),abs(x-105.3))
    near = np.fmin(outputs['p_dh'],outputs['p_da'])
    outputs['support'] = np.nan_to_num(expit((16-goal)/3)*expit((14-abs(y-34))/3)
                                     *expit((5-sp)/2)*expit((5-near)/1.5))
    np.savez_compressed(LOCAL/'inputs.npz',**outputs)
    print('Prepared label-free inputs',flush=True)


class Inputs:
    def __init__(self,k):
        z = np.load(LOCAL/'inputs.npz')
        self.base = z[f'base{k}']; self.team=z['team']; self.state=z['state']; self.support=z['support']
        self.post_dead=z[f'post_dead{k}']; self.post_team=z[f'post_team{k}']
        self.broad=z['broad']; self.local=z['local']; self.contact=z['contact']
        self.q={'broad':self.broad,'local':self.local,'state':self.state[:,2],'posterior':self.post_dead}
        self.states={'state':self.state}; self.smoothed={}
        self.ball={s:{f'{pre}_{key}':z[f'{pre}_{key}'] for key in ('x','y','speed','dh','da')}
                   for s,pre in (('provided','p'),('video','v'),('broadcast','b'))}
        self.balls=self.ball
        for name in ('spot_team','spot_live','spot_out','spot_restart'):
            setattr(self,name,z[name])

    def smooth(self,name,width):
        if (name,width) not in self.smoothed:
            self.smoothed[name,width]=smooth(self.q[name],width)
        return self.smoothed[name,width]


if __name__=='__main__':
    prepare()
