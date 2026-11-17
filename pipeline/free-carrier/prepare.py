"""Per-player ball evidence on the tactical timeline. No labels are read.

    python prepare.py          -> .local/inputs.npz

For every frame and each of the 46 tracking slots:
  dgt, dgb  ground distance (m) from the slot to hc-ball's tactical / broadcast video ball
  cnt, cnb  pixel contact in the tactical / broadcast camera: distance from the ball pixel to the
            body of the detected box linked to this slot, per box height (0 = overlapping)
  cft, cfb  ball pixel to that box's foot point, per box height
  rv        |ball velocity - slot velocity| (m/s, 4-frame differences, tactical ball)
Keeper slots take the team defending the goal on their side. Boxes are touches-video's detections with their label-free jersey team; each box is linked to
the nearest tracked slot of the same team within 2.5 m of its foot point.
Ball-only channels: ball speed (tactical, broadcast), source, 3D height.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE.parent
LOCAL = HERE / '.local'

N = 199319
HALVES = [(11185, 90652), (110808, 190198)]
TOUCH = EXP / 'touches-video/.local'
BALL = EXP / 'hc-ball/.local'


def track_teams():
    import xml.etree.ElementTree as ET
    root = ET.parse(ROOT / 'official/data/tactical/BAR-ATH_tactical_atd.xml').getroot()
    ids = [t.get('id') for t in root.iter('Team')]
    return np.array([0 if p.get('teamId') == ids[0] else 1 for p in root.iter('Player')], np.int8)


def keeper_slots():
    import xml.etree.ElementTree as ET
    root = ET.parse(ROOT / 'official/data/tactical/BAR-ATH_tactical_atd.xml').getroot()
    out = []
    for i, p in enumerate(root.iter('Player')):
        for par in p.iter('ProviderParameter'):
            if par.findtext('Name') == 'position_type' and par.findtext('Value') == 'Goalkeeper':
                out.append(i)
    return out


def fix_keepers(pl, team):
    """The provider's two keeper slots can swap or sit at one goal: as in touches-video, a keeper
    slot belongs to the team defending the goal on its side (Barca defends x = 0 in the first half)."""
    ks = keeper_slots()
    kh = [k for k in ks if team[k] == 0]
    ka = [k for k in ks if team[k] == 1]
    if not (len(kh) == 1 and len(ka) == 1):
        return pl
    h, a = kh[0], ka[0]
    mid = 105.3 / 2
    left_h = np.arange(N) < (HALVES[0][1] + HALVES[1][0]) // 2
    P = pl[:, [h, a]].copy()
    on_h = np.where(left_h[:, None], P[..., 0] < mid, P[..., 0] >= mid)
    goal_x = np.where(left_h[:, None] == on_h, 0.0, 105.3)
    depth = np.abs(P[..., 0] - goal_x)
    have = np.isfinite(P[..., 0])
    newh = np.full((N, 2), np.nan, np.float32)
    newa = np.full((N, 2), np.nan, np.float32)
    for side, out in ((True, newh), (False, newa)):
        dd = np.where(have & (on_h == side), depth, np.inf)
        j = dd.argmin(1)
        ok = np.isfinite(dd.min(1))
        out[ok] = P[np.flatnonzero(ok), j[ok]]
    pl[:, h], pl[:, a] = newh, newa
    return pl


def broadcast_index():
    sync = json.load(open(ROOT / 'calibration/broadcast_sync.json'))
    T = np.arange(N)
    B = np.full(N, -1)
    for h in sync['halves']:
        m = (T >= h['from']) & (T < h['to'])
        B[m] = np.round((T[m] - h['a']) / (1 + h['b'])).astype(int)
    return B


def load_boxes(view):
    d, pre = ('players', 'tac') if view == 'tactical' else ('players_broadcast', 'bro')
    parts = sorted(glob.glob(str(TOUCH / d / f'{pre}_*.npz')), key=lambda p: int(p.split('_')[-2]))
    fr = np.concatenate([np.load(p)['frames'] for p in parts])
    bx = np.concatenate([np.load(p)['boxes'] for p in parts])
    bt = np.load(TOUCH / ('boxteam.npz' if view == 'tactical' else 'boxteam_broadcast.npz'))
    return fr, bx, bt


def velocity(x, k=2):
    """Central difference over 2k frames, m/s; NaN propagates."""
    v = np.full_like(x, np.nan)
    v[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k) * 25
    return v


def box_contact(view, P, team, trk):
    """(N,46) min body distance and foot distance per box height, NaN where no linked box."""
    fr, bx, bt = load_boxes(view)
    if view == 'tactical':
        tf, rows = fr, np.arange(len(fr))
    else:
        tf = bt['frames']
        B = broadcast_index()
        pos = np.full(fr.max() + 2, -1)
        pos[fr] = np.arange(len(fr))
        bb = B[tf]
        rows = np.where((bb >= 0) & (bb < len(pos)), pos[np.clip(bb, 0, len(pos) - 1)], -1)
    keep = rows >= 0
    tf, rows = tf[keep], rows[keep]
    trow = np.searchsorted(bt['frames'], tf)
    cn = np.full((N, 46), np.nan, np.float16)
    cf = np.full((N, 46), np.nan, np.float16)
    cr = np.full((N, 46), np.nan, np.float16)
    for s in range(0, len(tf), 5000):
        f = tf[s:s + 5000]
        b = bx[rows[s:s + 5000]].astype(np.float64)
        prob = bt['prob'][trow[s:s + 5000]].astype(np.float64)
        foot = bt['foot'][trow[s:s + 5000]].astype(np.float64)
        bpx, bpy = trk['px'][f].astype(np.float64), trk['py'][f].astype(np.float64)
        x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        top, bot = y - h / 2, y + h / 2
        ox = np.maximum(0, np.abs(bpx[:, None] - x) - 0.3 * w)
        oy = np.where(bpy[:, None] > bot, bpy[:, None] - bot, np.where(bpy[:, None] < top, top - bpy[:, None], 0))
        dn = np.hypot(ox, oy) / h
        fd = np.hypot(bpx[:, None] - x, bpy[:, None] - bot) / h
        rel = (bot - bpy[:, None]) / h
        bteam = np.argmax(np.nan_to_num(prob, nan=-1), -1)
        conf = np.nanmax(prob, -1)
        ok = np.isfinite(dn) & np.isfinite(conf) & (conf > .6) & (bteam < 2) & (b[..., 5] == 0)
        Pp = P[f]                                                   # (n,46,2)
        d = np.hypot(foot[:, :, None, 0] - Pp[:, None, :, 0], foot[:, :, None, 1] - Pp[:, None, :, 1])
        d = np.where(np.isfinite(d) & (team[None, None, :] == bteam[:, :, None]), d, np.inf)
        j = np.argmin(d, -1)                                        # (n,32)
        dj = np.take_along_axis(d, j[..., None], -1)[..., 0]
        ok &= dj < 2.5
        n = len(f)
        acc_n = np.full((n, 46), np.inf)
        acc_f = np.full((n, 46), np.inf)
        acc_r = np.full((n, 46), np.nan)
        ii, kk = np.nonzero(ok)
        order = np.argsort(-dn[ii, kk])                            # smallest written last wins
        ii, kk = ii[order], kk[order]
        jj = j[ii, kk]
        acc_n[ii, jj] = dn[ii, kk]
        acc_r[ii, jj] = rel[ii, kk]
        np.minimum.at(acc_f, (ii, jj), fd[ii, kk])
        cn[f] = np.where(np.isfinite(acc_n), acc_n, np.nan)
        cf[f] = np.where(np.isfinite(acc_f), acc_f, np.nan)
        cr[f] = acc_r
    return cn, cf, cr


def main():
    z = np.load(EXP / 'possession-engine/.local/tracking.npz')
    team = track_teams()
    P = fix_keepers(z['players'][..., :2].astype(np.float32), team)
    tt = dict(np.load(BALL / 'track_tactical.npz'))
    tb = dict(np.load(BALL / 'track_broadcast.npz'))
    b3 = dict(np.load(BALL / 'ball3d.npz'))
    out = {'team': team}
    for tag, trk in (('t', tt), ('b', tb)):
        ok = trk['source'] > 0
        bxy = np.where(ok[:, None], np.c_[trk['x'], trk['y']], np.nan).astype(np.float32)
        out[f'dg{tag}'] = np.hypot(P[..., 0] - bxy[:, None, 0], P[..., 1] - bxy[:, None, 1]).astype(np.float16)
        out[f'bx{tag}'], out[f'by{tag}'] = bxy[:, 0], bxy[:, 1]
        out[f'src{tag}'] = trk['source']
        vb = np.c_[velocity(bxy[:, 0]), velocity(bxy[:, 1])]
        out[f'vbx{tag}'], out[f'vby{tag}'] = vb[:, 0].astype(np.float32), vb[:, 1].astype(np.float32)
        if tag == 't':
            vp = np.stack([velocity(P[..., 0]), velocity(P[..., 1])], -1)
            out['rv'] = np.hypot(vp[..., 0] - vb[:, None, 0], vp[..., 1] - vb[:, None, 1]).astype(np.float16)
            out['vpx'], out['vpy'] = vp[..., 0].astype(np.float16), vp[..., 1].astype(np.float16)
        cn, cf, cr = box_contact('tactical' if tag == 't' else 'broadcast', P, team, trk)
        out[f'cn{tag}'], out[f'cf{tag}'], out[f'cr{tag}'] = cn, cf, cr
        print(tag, 'frames with a linked box near the ball (<.25 h):',
              int((np.nanmin(cn.astype(np.float32), 1) < .25).sum()), flush=True)
    out['height'] = np.where(b3['gap'] < 1.5, b3['z'], np.nan).astype(np.float32)
    out['px'] = P[..., 0].astype(np.float32)
    out['py'] = P[..., 1].astype(np.float32)
    np.savez_compressed(LOCAL / 'inputs.npz', **out)


if __name__ == '__main__':
    main()
