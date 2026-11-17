"""Combine dead/live evidence and replay saved decoder settings."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOCAL = HERE / '.local'
OWNER = HERE.parent / 'free-owner'
sys.path.insert(0, str(OWNER))
from common import BLOCKS, N, X, write
from decode import decode, posterior
from tune import build


def odds(p):
    p = np.clip(np.nan_to_num(p, nan=.5), 1e-4, 1-1e-4)
    return np.log(p) - np.log1p(-p)


DEFAULT = dict(wq=.6, wc=2., beta=20., dfl=.75, ext=0., team=0., video=.5,
               bias=0., scale=1., cost=40., event=1., active=0., launch=0., goal=0.,edge=0.)
GRID = dict(dfl=[0., .25, .5, .75, 1.], ext=[0., .25, .5], video=[0., .5, 1.],
            bias=[-3., -2., -1., 0., 1., 2., 3.], scale=[.2, .5, 1., 2.],
            cost=[5., 10., 20., 40., 80., 160.], event=[0., .25, .5, 1., 2.],
            active=[0., 1., 2., 4.], team=[0., .25, .5, .75],
            wq=[.2, .4, .6, .8, 1.2], wc=[.5, 1., 2., 4.],
            beta=[5., 10., 20., 40., 80.],launch=[0.,2.,4.,8.],goal=[0.,2.,4.,8.],edge=[0.,.5,1.,2.])


class Engine:
    def __init__(self):
        ev = build()
        self.owner = ev
        self.ld = ev.ld
        raw = np.load(HERE.parent / 'free-dfl/.local/dfl_only_evidence.npz')
        self.dfl = odds(raw['p_dead'])
        self.dfl_team = odds(raw['p_home_live'])
        raw = np.load(LOCAL/'video_dfl_detected.npz')
        self.video = odds(raw['p_dead'])
        self.video_team = odds(raw['p_home_live'])
        def delta(q):
            sm=uniform_filter1d(np.nan_to_num(q,nan=.5),13)
            return np.r_[np.full(12,sm[0]),sm[:-12]]-np.r_[sm[12:],np.full(12,sm[-1])]
        self.video_edge=delta(raw['p_dead'])
        self.dfl_edge=delta(expit(self.dfl))
        raw = np.load(HERE.parent / 'free-external/.local/model-equal-sc-metrica/target.npz')
        # Fixed equal mean of external models.
        idx = raw['original_index']
        up = np.clip(np.searchsorted(idx, np.arange(N), side='right')-1,0,len(idx)-1)
        p = raw['p'][up]
        self.ext = odds(p[:, 2])
        z = np.load(OWNER / '.local/features.npz')
        fr = z['frames']
        S = {}
        for i, name in enumerate(z['names_s']):
            a = np.zeros(N)
            a[fr] = np.nan_to_num(z['Xs'][:, i])
            S[str(name)] = a
        # Ball flight inside the pitch, seen in the tactical view while the
        # broadcast shows the field. A physical live cue, no label masks.
        self.active = ((S['ball_speed_w13'] > 6) & (S['ball_speed_w13'] < 35)
                       & (S['ball_out'] < -.3) & (S['bc_off_w25'] < .2)
                       & (S['ball_det_w25'] > .15)).astype(float)
        self.physics=dict(np.load(LOCAL/'physics.npz'))

    def scores(self, p):
        e = self.owner
        vt = (1-p['video'])*self.dfl_team+p['video']*self.video_team
        vd = (1-p['video'])*self.dfl+p['video']*self.video
        team = (1-p['team'])*e.lq + p['team']*vt
        s = p['wq']*team + p['wc']*e.ct
        dead = (1-p['dfl'])*self.ld + p['dfl']*vd
        dead = (1-p['ext'])*dead + p['ext']*self.ext
        d = p['scale']*(dead+p['bias']) - p['active']*self.active
        d += p['goal']*self.physics['dead']-p['launch']*self.physics['live']
        if p.get('student_dead',0) or p.get('student_team',0):
            students=self.students()
            s=(1-p['student_team'])*s+p['student_team']*p['wq']*students['team']
            d=(1-p['student_dead'])*d+p['student_dead']*p['scale']*(students['dead']+p['bias'])
        # Status should not change simply because the owning team is certain.
        d += np.abs(s)*.5
        edge=(1-p['video'])*self.dfl_edge+p['video']*self.video_edge
        cin = np.maximum(.5, p['cost']*(1-p['edge']*np.maximum(-edge,0))-p['event']*(20*e.out+80*e.drop))
        cout = np.maximum(.5, p['cost']*(1-p['edge']*np.maximum(edge,0))-p['event']*(40*e.kick+3*e.setp+10*e.acc))
        return s, d, p['beta'], cin, cout

    def run(self, p):
        return decode(*self.scores(p))

    def students(self):
        path=LOCAL/'students.npz'
        if not path.exists():
            source=HERE.parent/'free-distill/.local'
            files=[source/f'{n}.npy' for n in ('local','broad','movement')]
            dead=np.mean([odds(np.load(p)) for p in files[:2]],axis=0)
            team=odds(np.load(files[2]))
            np.savez_compressed(path,dead=dead,team=team)
        if not hasattr(self,'_students'):
            self._students=dict(np.load(path))
        return self._students


def export_oob(ev,folds):
    """Replay already selected fold constants without opening any labels."""
    out=np.full(N,2,dtype='i1')
    m=np.full((N,3),np.nan)
    for (a,b),row in zip(BLOCKS,folds):
        p=row['params']
        out[a:b]=ev.run(p)[a:b]
        m[a:b]=posterior(*ev.scores(p))[a:b]
    saved=dict(np.load(LOCAL/'evidence.npz'))
    saved.update(p_dead_oob=np.nan_to_num(m[:,2],nan=1.).astype('f4'),
        p_home_live_oob=np.divide(m[:,0],m[:,:2].sum(1),out=np.full(N,.5),
                                 where=m[:,:2].sum(1)>1e-30).astype('f4'),label_oob=out)
    np.savez_compressed(LOCAL/'evidence.npz',**saved)
    write(LOCAL/'cv.csv',out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', action='store_true', required=True)
    ap.parse_args()
    LOCAL.mkdir(exist_ok=True)
    ev = Engine()
    recipe = json.loads((HERE/'recipe.json').read_text())
    params = recipe['params']
    pred = ev.run(params)
    pred[pred==X]=2
    write(LOCAL/'best.csv', pred)
    m = posterior(*ev.scores(params))
    np.savez_compressed(LOCAL/'evidence.npz', p_dead=np.nan_to_num(m[:,2],nan=1.).astype('f4'),
        p_home_live=np.divide(m[:,0], m[:,:2].sum(1), out=np.full(N,.5),
                              where=m[:,:2].sum(1)>1e-30).astype('f4'),
        p_dead_raw=np.nan_to_num(expit(ev.scores(params)[1]-np.abs(ev.scores(params)[0])*.5),nan=1.).astype('f4'),
        label=pred.astype('i1'))
    if 'folds' in recipe:
        export_oob(ev, recipe['folds'])


if __name__ == '__main__':
    main()
