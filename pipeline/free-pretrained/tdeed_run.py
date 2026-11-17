"""Run SoccerNet-trained T-DEED spotters over one of our videos, zero-shot.

    python tdeed_run.py broadcast|tactical [START END]

Two public checkpoints from https://github.com/arturxe2/T-DEED (Google Drive folder in its
README), both trained only on SoccerNet games:
  - teamball: the SoccerNet 2025 Team Ball Action Spotting baseline (sn-teamspotting), 12 ball
    actions + which team (left / right of the video) did it;
  - ball: the 2024 Ball Action Spotting winner (SoccerNetBall_challenge1), no team head.
Both also carry a second head for the 17 SoccerNet-v2 actions (ball out of play, throw-in, ...).

The video is decoded by FFmpeg at every 2nd frame (12.5 fps, the models' rate), resized to
796x448, cut into 100-frame (8 s) clips every 50 sampled frames, and the per-frame softmax /
sigmoid outputs of the overlapping clips are averaged. Output .local/tdeed_<video>.npz, indexed by
the video's own frame number (frame = 2 * sampled index + START):
  frame (n,), ball13/team13 (n,13) action softmax, ball18/team18 (n,18) SN-v2 softmax,
  balld/teamd (n,) displacement, teamr (n,) P(right team).
"""
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL = os.path.join(HERE, ".local")
sys.path.insert(0, os.path.join(LOCAL, "ext", "sn-teamspotting"))
import timm  # noqa: E402

_create = timm.create_model
timm.create_model = lambda name, pretrained=False, **kw: _create(name, pretrained=False, **kw)
from model.model import TDEEDModel  # noqa: E402

VIDEOS = {"tactical": os.path.join(ROOT, "official", "data", "tactical", "BAR-ATH_tactical.mp4"),
          "broadcast": os.path.join(ROOT, "official", "data", "broadcast", "BAR-ATH_broadcast.mp4")}
NFRAMES = {"tactical": 199319, "broadcast": 164488}
W, H, L, HOP = 796, 448, 100, 50


def load(name, team):
    args = SimpleNamespace(modality="rgb", temporal_arch="ed_sgp_mixer", radi_displacement=4,
                           feature_arch="rny002_gsf", event_team=team, clip_len=L, n_layers=2,
                           sgp_ks=9, sgp_r=4, num_classes=12, crop_dim=None)
    m = TDEEDModel(device="cuda", args=args)
    m._model.update_pred_head([13, 18])
    sd = torch.load(os.path.join(LOCAL, "weights", name), map_location="cpu", weights_only=True)
    m._model.load_state_dict(sd)
    m._model.eval()
    return m._model


def frames(video, start, end):
    """Yield uint8 frames (H, W, 3) of every 2nd video frame in [start, end)."""
    n = (end - start + 1) // 2
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-threads", "2", "-ss", f"{start / 25:.4f}",
           "-i", VIDEOS[video], "-frames:v", str(end - start),
           "-vf", f"framestep=2,scale={W}:{H}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 8)
    size = W * H * 3
    for _ in range(n):
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.stdout.close()
    proc.wait()


@torch.no_grad()
def run(models, clip):
    x = torch.from_numpy(clip).cuda().permute(0, 3, 1, 2)[None].float()   # (1, L, 3, H, W)
    out = {}
    for key, m in models.items():
        with torch.amp.autocast("cuda"):
            d, _ = m(x, inference=True)
        p = d["im_feat"][0].float()
        out[key + "13"] = torch.softmax(p[:, :13], -1).cpu().numpy()
        out[key + "18"] = torch.softmax(p[:, 13:], -1).cpu().numpy()
        out[key + "d"] = d["displ_feat"][0].float().cpu().numpy()
        if "team_feat" in d:
            out[key + "r"] = torch.sigmoid(d["team_feat"][0].float()).cpu().numpy()
    return out


def main():
    video = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else NFRAMES[video]
    start -= start % 2
    out_path = os.path.join(LOCAL, f"tdeed_{video}" + (f"_{start}_{end}" if len(sys.argv) > 2 else "") + ".npz")
    torch.backends.cudnn.benchmark = True
    models = {"team": load("teamball_baseline.pt", True), "ball": load("ball_challenge1.pt", False)}
    n = (end - start + 1) // 2
    acc, cnt = {}, np.zeros(n, np.float32)
    ring = np.zeros((L, H, W, 3), np.uint8)
    t0, got, pos = time.time(), 0, 0      # pos = sampled index of ring[0]
    fill = 0
    gen = frames(video, start, end)
    done = False
    while not done:
        while fill < L:
            try:
                ring[fill] = next(gen)
                fill += 1
                got += 1
            except StopIteration:
                done = True
                break
        if fill == 0:
            break
        clip = ring if fill == L else np.concatenate([ring[:fill], np.repeat(ring[fill - 1:fill], L - fill, 0)])
        o = run(models, clip)
        k = min(L, n - pos, fill)
        for key, v in o.items():
            if key not in acc:
                acc[key] = np.zeros((n,) + v.shape[1:], np.float32)
            acc[key][pos:pos + k] += v[:k]
        cnt[pos:pos + k] += 1
        if done:
            break
        ring[:L - HOP] = ring[HOP:]
        fill = L - HOP
        pos += HOP
        if (pos // HOP) % 40 == 0:
            el = time.time() - t0
            print(f"{video} {start + 2 * pos}: {got} frames, {got / el:.1f}/s, eta {(n - got) / max(got / el, 1e-6) / 60:.1f} min", flush=True)
    m = cnt > 0
    res = {k: v[m] / cnt[m].reshape((-1,) + (1,) * (v.ndim - 1)) for k, v in acc.items()}
    res["frame"] = (start + 2 * np.arange(n))[m]
    np.savez_compressed(out_path, **res)
    print("wrote", out_path, len(res["frame"]), f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
