"""Set-piece launches and goal-line exits from label-free ball trajectories.

All thresholds below are physical, hand-set constants. No labels are opened.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'free-owner'))
from common import HALVES, N
from events import ball_xy, window_mean


def groups(mask, gap=3, minimum=3):
    ix=np.flatnonzero(mask)
    return [g for g in np.split(ix,np.flatnonzero(np.diff(ix)>gap)+1) if len(g)>=minimum]


def main():
    (HERE/'.local').mkdir(parents=True,exist_ok=True)
    tr = np.load(HERE.parent/'hc-ball/.local/track_tactical.npz')
    x,y,src,sp = ball_xy()
    # Keep the ground projection for launch spots; 3D improves line crossings.
    bx,by = tr['x'],tr['y']
    vd=np.load(HERE/'.local/video_dfl_detected.npz')['p_dead']
    live=np.zeros(N);dead=np.zeros(N);owner=np.full(N,.5)
    team=np.load(HERE/'.local/video_dfl_detected.npz')['p_home_live']
    records=[]
    for kind in ('corner','penalty'):
        anchors = ([(xx,yy) for xx in (0.,105.3) for yy in (0.,68.)] if kind=='corner'
                   else [(11.,34.),(94.3,34.)] if kind=='penalty' else [])
        for xx,yy in anchors:
            dist=np.hypot(bx-xx,by-yy)
            rest=(dist<2.5)&(src==1)&(sp<2.)
            flight=(dist>4)&(src>0)&(sp>8)&(sp<40)
            if kind=='corner':flight &= (bx>.5)&(bx<104.8)&(by>.5)&(by<67.5)
            else:flight &= ((bx-xx)*(1 if xx<50 else -1)<-2)
            for fa,fb in HALVES:
                for g in groups(rest[fa:fb],15,12):
                    a,b=fa+int(g[0]),fa+int(g[-1])+1
                    if np.nanmean(vd[max(fa,a-75):b])<.5:continue
                    hits=np.flatnonzero(flight[b:min(fb,b+50)])
                    if not len(hits):continue
                    launch=b+int(hits[0]);start=(b+launch)//2
                    # Interpolated trajectories can invent a flight between
                    # two false detections. Require repeated observed motion.
                    observed=(src==1)&(sp>6)&(sp<40)&(dist>3)
                    if observed[launch:min(fb,launch+50)].sum()<3:continue
                    # The next visible exit terminates the local live phase.
                    out=(x<-.3)|(x>105.6)|(y<-.3)|(y>68.3)
                    # One projected air sample beyond a line is insufficient.
                    certain=window_mean(out.astype(float),0,5)>.5
                    if kind=='penalty':
                        certain |= ((bx<0)|(bx>105.3))&(src>0)
                    crossing=np.flatnonzero(certain[start+10:min(fb,start+200)])
                    if len(crossing):end=start+10+int(crossing[0])
                    else:
                        high=window_mean((vd>.9).astype(float),0,15)>.8
                        stops=np.flatnonzero(high[start+30:min(fb,start+200)])
                        end=start+30+int(stops[0]) if len(stops) else min(fb,start+200)
                    if end-start<10:continue
                    live[start:end]=1
                    owner[start:end]=float(np.nanmean(team[max(fa,b-30):b]))
                    records.append(dict(kind=kind,start=start,end=end,rest=[a,b],flight=launch))
    # A goal-line exit followed by a stationary ball and a kick is a goal kick.
    # Reject air-projection exits with a high future live probability.
    for fa,fb in HALVES:
        out=((x<-.3)|(x>105.6))&(y>0)&(y<68)&(src>0)
        for g in groups(out[fa:fb],3,5):
            a,b=fa+int(g[0]),fa+int(g[-1])+1
            if b>=fb-50 or np.nanmean(vd[b:min(fb,b+150)])<.7:continue
            prev=np.flatnonzero((x[max(fa,a-75):a]>=0)&(x[max(fa,a-75):a]<=105.3))
            start=max(fa,a-75)+int(prev[-1])+1 if len(prev) else a
            rest=window_mean(((sp<2)&(src>0)).astype(float),-40,0)>.6
            launch=(sp>6)&(sp<40)&rest&(x>0)&(x<105.3)
            hits=np.flatnonzero(launch[b+20:min(fb,start+1000)])
            if not len(hits):continue
            end=b+20+int(hits[0])
            if end-start<50:continue
            dead[start:end]=1
            records.append(dict(kind='goal_line',start=start,end=end))
    np.savez_compressed(HERE/'.local/physics.npz',live=live.astype('f4'),dead=dead.astype('f4'),owner=owner.astype('f4'))
    (HERE/'.local/physics.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=1),flush=True)


if __name__=='__main__':main()
