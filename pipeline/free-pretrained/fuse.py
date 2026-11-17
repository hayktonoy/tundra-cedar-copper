"""Fuse the pretrained-model channels into free-status's decoder.

Free-status's Engine (free-owner Viterbi; EM, DFL, video-DFL, external and goal-line status; shape,
contact and DFL team) is reused unchanged. New terms, each with one weight:
  tb    team: SoccerNet spotter on-ball actions with a confident team call, broadcast video
        (tdeed_channels.npz; the tactical view's team head is much weaker and is not used);
  tout  stoppage entry cheaper at spotted ball-out-of-play;
  trs   stoppage exit cheaper at spotted restarts (throw-in, free kick, corner, kick-off);
  tact  dead emission lowered where the spotter sees ball actions;
  tl    dead emission lowered for 1 s after spotted on-ball actions (any team, both videos);
  tw    team: as tb but held 0.5 s after the action;
  tl2   as tl, deliveries and restarts only (cross, shot, free kick, throw-in);
  tw2   team of those deliveries and restarts (broadcast), held 1 s;
With every new weight at 0 the scores equal free-status's.

"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = HERE / ".local"
sys.path.insert(0, str(EXP / "free-status"))
from run import Engine, odds  # noqa: E402  (also puts free-owner first on sys.path)
from common import N  # noqa: E402  free-owner's
from decode import decode, posterior  # noqa: E402

NEW = dict(tb=0., tout=0., trs=0., tact=0., tl=0., tw=0., tl2=0., tw2=0.)
GRIDS = {
    "tdeed": dict(tb=[0., .5, 1., 2., 4.], tout=[0., 10., 20., 40.], trs=[0., 10., 20., 40.],
                  tact=[0., .5, 1., 2.]),
    "team": dict(tb=[0., .5, 1., 2., 4.]),
    "ev": dict(tl=[0., .5, 1., 2., 4., 8.], tw=[0., .5, 1., 2., 4.]),
    "live2": dict(tl2=[0., .5, 1., 2., 4., 8.], tw2=[0., .5, 1., 2., 4.]),
    "tl2": dict(tl2=[0., .5, 1., 2., 4., 8.]),
    "tl": dict(tl=[0., .25, .5, 1., 2., 4., 8.]),
    "tw": dict(tw=[0., .5, 1., 2., 4., 8., 16.]),
    "tout": dict(tout=[0., 5., 10., 20., 40., 80.]),
    "trs": dict(trs=[0., 5., 10., 20., 40., 80.]),
    "tact": dict(tact=[0., .25, .5, 1., 2., 4.]),
}
TEAM_GRID = dict(wq=[.2, .4, .6, .8, 1.2], wc=[.5, 1., 2., 4.], beta=[5., 10., 20., 40., 80.])


class PEngine(Engine):
    def __init__(self):
        super().__init__()
        z = np.load(LOCAL / "tdeed_channels.npz")
        self.onball = np.nan_to_num(z["onball_bro"])     # the tactical view's team calls are not used
        self.out = np.clip(np.nan_to_num(z["out_bro"]) + np.nan_to_num(z["out_tac"]), 0, 2)
        self.restart = np.clip(np.nan_to_num(z["restart_bro"]) + np.nan_to_num(z["restart_tac"]), 0, 2)
        self.live = np.clip(np.nan_to_num(z["live_bro"]) + np.nan_to_num(z["live_tac"]), 0, 2)
        self.teamw = np.nan_to_num(z["teamw_bro"])
        self.live2 = np.clip(np.nan_to_num(z["live2_bro"]) + np.nan_to_num(z["live2_tac"]), 0, 2)
        self.team2 = np.nan_to_num(z["team2_bro"])
        act = np.nanmean(np.c_[z["act_bro"], z["act_tac"]], axis=1)
        self.act = np.nan_to_num(act - np.nanmedian(act))

    def scores(self, p):
        """free-status's Engine.scores with the new terms added before its |s| coupling."""
        e = self.owner
        vt = (1-p['video'])*self.dfl_team+p['video']*self.video_team
        vd = (1-p['video'])*self.dfl+p['video']*self.video
        team = (1-p['team'])*e.lq + p['team']*vt
        s = p['wq']*team + p['wc']*e.ct
        dead = (1-p['dfl'])*self.ld + p['dfl']*vd
        dead = (1-p['ext'])*dead + p['ext']*self.ext
        d = p['scale']*(dead+p['bias']) - p['active']*self.active
        d += p['goal']*self.physics['dead']-p['launch']*self.physics['live']
        if p.get('student_dead', 0) or p.get('student_team', 0):
            students = self.students()
            s = (1-p['student_team'])*s+p['student_team']*p['wq']*students['team']
            d = (1-p['student_dead'])*d+p['student_dead']*p['scale']*(students['dead']+p['bias'])
        s = (s + p.get('tb', 0)*self.onball + p.get('tw', 0)*self.teamw
             + p.get('tw2', 0)*self.team2)
        d = d - p.get('tact', 0)*self.act - p.get('tl', 0)*self.live - p.get('tl2', 0)*self.live2
        d += np.abs(s)*.5
        edge = (1-p['video'])*self.dfl_edge+p['video']*self.video_edge
        cin = np.maximum(.5, p['cost']*(1-p['edge']*np.maximum(-edge, 0))-p['event']*(20*e.out+80*e.drop)
                         - p.get('tout', 0)*self.out)
        cout = np.maximum(.5, p['cost']*(1-p['edge']*np.maximum(edge, 0))-p['event']*(40*e.kick+3*e.setp+10*e.acc)
                          - p.get('trs', 0)*self.restart)
        return s, d, p['beta'], cin, cout
