"""Apply the existing external-only DFL heads to the improved video ball.

Inference only, three CPU threads; this process never opens any labels.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOCAL = HERE/'.local'
DFL = HERE.parent/'free-dfl/.local'
spec = importlib.util.spec_from_file_location('dfl_features', HERE.parent/'free-dfl/features.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    players = np.load(DFL/'bar.npz')['players']
    ball = np.load(HERE.parent/'hc-ball/.local/track_tactical.npz')
    models = [lgb.Booster(model_file=str(DFL/'models/all'/f'{head}.txt')) for head in ('dead','owner')]
    names = json.loads((DFL/'feature_names.json').read_text())
    for mode in ('detected','interpolated','three_d'):
        b = np.c_[ball['x'],ball['y']].astype('f4')
        if mode=='detected':
            b[ball['source']!=1] = np.nan
        else:
            b[ball['source']==0] = np.nan
        if mode=='three_d':
            z = np.load(HERE.parent/'hc-ball/.local/ball3d.npz')
            ok = np.isfinite(z['x']) & np.isfinite(z['y'])
            b[ok] = np.c_[z['x'][ok],z['y'][ok]]
        b -= [52.65,34]
        result = np.full((199319,2),np.nan,'f4')
        for a,e in [(11185,90652),(110808,190198)]:
            x, nn = f.raw_features(players[a:e],b[a:e])
            assert names==nn
            x = x[::5]
            dd = models[0].predict(f.status_features(x,names),num_threads=3)
            hh = (models[1].predict(x,num_threads=3)+1-models[1].predict(x[:,f.swap_index(names)],num_threads=3))/2
            ix=np.arange(0,e-a,5)
            result[a:e,0] = np.interp(np.arange(e-a),ix,dd)
            result[a:e,1] = np.interp(np.arange(e-a),ix,hh)
        np.savez_compressed(LOCAL/f'video_dfl_{mode}.npz',p_dead=result[:,0],p_home_live=result[:,1])
        print(mode,'finished',flush=True)


if __name__=='__main__':
    main()
