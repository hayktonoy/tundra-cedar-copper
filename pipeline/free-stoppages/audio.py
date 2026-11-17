"""Stream a fixed narrow-band whistle heuristic; no learned audio model."""
from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
from features import N, LOCAL, HERE


def main():
    rate=16000;hop=640;nfft=1024
    p=HERE.parents[1]/'official/data/broadcast/BAR-ATH_broadcast.mp4'
    cmd=['ffmpeg','-v','error','-threads','1','-i',str(p),'-vn','-ac','1','-ar',str(rate),'-f','s16le','-']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    vals=[];win=np.hanning(nfft);freq=np.fft.rfftfreq(nfft,1/rate)
    band=(freq>=2500)&(freq<5000);wide=(freq>=500)&(freq<7500)
    rem=np.empty(0,np.int16)
    while True:
        raw=proc.stdout.read(2*rate*20)
        if not raw:break
        x=np.r_[rem,np.frombuffer(raw,dtype='<i2')]
        count=max(0,(len(x)-nfft)//hop+1)
        if count:
            chunks=np.lib.stride_tricks.sliding_window_view(x,nfft)[::hop][:count]
            power=abs(np.fft.rfft(chunks*win,axis=1))**2
            local=power[:,band]
            narrow=maximum_filter1d(local,5,axis=1).max(1)
            # Tonality and high-frequency dominance. Commentary/crowd can
            # trigger this too; corroborating physics is required downstream.
            sharp=narrow/(local.sum(1)+1)
            dominance=local.sum(1)/(power[:,wide].sum(1)+1)
            vals.append(np.c_[sharp,dominance,np.log1p(local.sum(1))])
        rem=x[count*hop:]
    assert proc.wait()==0
    v=np.concatenate(vals);out=np.zeros((N,3),np.float32)
    for a,b,offset,drift in [(4014,88809,4013.904,-6.1885e-06),(113465,191406,26921.591,-1.65973e-05)]:
        frames=np.arange(a,b);idx=np.rint((frames-offset)/(1+drift)).astype(int)
        good=(idx>=0)&(idx<len(v));out[frames[good]]=v[idx[good]]
    np.savez_compressed(LOCAL/'audio.npz',tonality=out[:,0],dominance=out[:,1],energy=out[:,2])
    print('Audio frames',len(v),'cache',LOCAL/'audio.npz',flush=True)


if __name__=='__main__':main()
