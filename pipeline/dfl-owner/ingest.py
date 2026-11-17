"""Pinned IDSSE downloads and streaming parse; writes only this experiment's .local."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import time
import urllib.request
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = HERE / '.local'
KEYS = ('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9')

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()), encoding='utf-8')

def fetch(entry):
    target = LOCAL / 'raw' / entry['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry['size']:
        return target
    # Resumable disjoint 32 MB ranges avoid long-lived stalled transfers.
    part = target.with_suffix('.part')
    statefile = target.with_suffix('.ranges.json')
    step = 32*1024*1024
    ranges = [(s,min(s+step,entry['size'])-1) for s in range(0,entry['size'],step)]
    done = set(json.loads(statefile.read_text()) if statefile.exists() else [])
    if not part.exists():
        done = set()
        with part.open('wb') as f: f.truncate(entry['size'])
    for i,(s,e) in enumerate(ranges):
        if i in done: continue
        for attempt in range(6):
            try:
                req = urllib.request.Request(entry['download_url']+f'?owner_range={s}',
                        headers={'Range':f'bytes={s}-{e}', 'Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=90) as r, part.open('r+b') as f:
                    assert r.status == 206 and r.headers['Content-Range'] == f'bytes {s}-{e}/{entry["size"]}', dict(r.headers)
                    f.seek(s)
                    remaining = e-s+1
                    while remaining:
                        chunk = r.read(min(1024*1024,remaining))
                        if not chunk: raise IOError('short read')
                        f.write(chunk); remaining -= len(chunk)
                break
            except Exception as exc:
                print('retry',entry['id'],i,attempt,str(exc),flush=True)
                if attempt == 5: raise
        done.add(i); save(statefile,sorted(done))
        print('download',entry['id'],len(done),len(ranges),flush=True)
    digest = hashlib.md5()
    with part.open('rb') as f:
        while block := f.read(4*1024*1024): digest.update(block)
    assert digest.hexdigest() == entry['computed_md5']
    part.replace(target)
    return target

def parse(key, entries):
    output = LOCAL / 'data' / f'{key}.npz'
    if output.exists(): return
    spec = importlib.util.spec_from_file_location('old_ingest',ROOT/'pipeline/dfl-possession-transfer/ingest.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    class Guard:
        def check(self,*a,**kw): pass
    roster = old.roster_metadata(LOCAL/'raw'/entries['matchinformation']['name'],key,Guard())
    # Use the earlier lightweight scan if retained; independently check every N
    # below. On a fresh checkout rebuild bounds without writing upstream caches.
    scanpath=ROOT/f'pipeline/dfl-possession-transfer/.local/audits/{key}_catalog.json'
    if scanpath.exists():catalog=json.loads(scanpath.read_text())
    else:
        halves={}; keep=False
        for ev,name,a in old.xml_events(LOCAL/'raw'/entries['positions']['name']):
            if ev=='start' and name=='FrameSet':
                section=a['GameSection'];keep=a.get('TeamId')!='referee'
            if ev=='end' and name=='Frame' and keep:
                num=int(a['N']);h=halves.setdefault(section,dict(min_n=num,max_n=num,grid_origin_timestamp=a['T']))
                if num<h['min_n']:h['grid_origin_timestamp']=a['T']
                h['min_n']=min(h['min_n'],num);h['max_n']=max(h['max_n'],num)
        catalog=dict(halves=halves)
    save(LOCAL/'data'/f'{key}_catalog.json',catalog)
    arrays={}; offsets={}; total=0
    for section in ('firstHalf','secondHalf'):
        half=catalog['halves'][section]; n=half['max_n']-half['min_n']+1
        offsets[section]=(total,half['min_n'],n); total+=n
    players=np.full((total,2,32,2),np.nan,'f4'); ball=np.full((total,3),np.nan,'f4')
    poss=np.full(total,-128,'i1'); status=poss.copy(); halfid=np.empty(total,'i1')
    for k,(section,(off,origin,n)) in enumerate(offsets.items()): halfid[off:off+n]=k
    current=None
    for ev,name,a in old.xml_events(LOCAL/'raw'/entries['positions']['name']):
        if ev=='start' and name=='FrameSet':
            section,isball,pid,axis,slot=old.identity(a,roster,key)
            off,origin,n=offsets[section]; current=(isball,axis,slot)
        elif ev=='end' and name=='Frame' and current:
            isball,axis,slot=current
            if axis == -1: continue
            ix=int(a['N'])-origin
            assert 0<=ix<n
            ix+=off
            xy=[float(a.get(c,'nan')) for c in ('X','Y')]
            if isball:
                ball[ix]=xy+[float(a.get('Z','nan'))]
                poss[ix]=int(a.get('BallPossession',-128)); status[ix]=int(a.get('BallStatus',-128))
            else: players[ix,axis,slot]=xy
        elif ev=='end' and name=='FrameSet': current=None
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,players=players,ball=ball,possession=poss,status=status,half=halfid)
    save(LOCAL/'data'/f'{key}.json',dict(offsets=offsets,teams=roster['teams'],team_ids=roster['team_ids'],
          poss_values=np.unique(poss,return_counts=True)[0].tolist(),status_values=np.unique(status).tolist()))
    print('parsed',key,total,'possession',np.unique(poss,return_counts=True),'status',np.unique(status,return_counts=True),flush=True)

def main():
    # The seven matches' files, pinned by figshare file id, size and MD5 (sources.json beside this
    # script; article https://doi.org/10.6084/m9.figshare.28196177).
    pinned=json.loads((HERE/'sources.json').read_text())
    pairs=pinned['files']
    assert tuple(pairs)==KEYS
    save(LOCAL/'sources.json',pinned)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,[e for p in pairs.values() for e in p.values()]))
    for key,pair in pairs.items(): parse(key,pair)

if __name__=='__main__': main()
