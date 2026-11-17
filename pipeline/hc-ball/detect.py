"""Zoomed ball detection: the Soccana YOLO11n (or a fine-tuned copy) on an upscaled crop around
where the ball should be, on the GPU.

    python detect.py tactical START END OUT.npz [--crop 640] [--size 1280] [--weights PATH] [--centres PATH]

The tactical ball is ~5 px wide; the detector was trained mostly on broadcast footage where it is
10-20 px. A CROP x CROP window around the expected ball, resized to SIZE x SIZE (2x for 640 ->
1280), puts it back in the detector's range for less compute than a full native frame.
Crop centres come from `--centres` (an npz with px, py per tactical frame; default the hc-video
cleaned ball, else the provided ball projected, else the previous centre, else the frame centre).
Frames are decoded by FFmpeg on the GPU as nv12 and converted with torch, as in hc-video.

Output: frames (n,), balls (n, 8, 5) x, y, w, h, score in native pixels, centre (n, 2).
"""
import argparse
import importlib
import os
import queue
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import LOCAL, N, TRACKING, VIDEO_LANE, VIDEOS, WEIGHTS, homographies, to_pixels, broadcast_index

W, H = 1920, 1080
K = 8


def load_model(path=WEIGHTS):
    """Restricted loading (weights_only) that only admits torch / ultralytics module classes."""
    names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    objects = []
    for name in names:
        module, symbol = name.rsplit(".", 1)
        obj = getattr(importlib.import_module(module), symbol)
        assert isinstance(obj, type) and issubclass(obj, torch.nn.Module), name
        objects.append(obj)
    with torch.serialization.safe_globals(objects):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ckpt["model"].float().eval().cuda().half()
    return model, model.names


def reader(video, start, n, out):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-ss", f"{start / 25:.4f}",
           "-i", video, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3 // 2
    while True:
        buf = bytearray(size)
        view, got = memoryview(buf), 0
        while got < size:
            r = proc.stdout.readinto(view[got:])
            if not r:
                break
            got += r
        if got < size:
            break
        out.put(buf)
    out.put(None)
    proc.wait()


def nv12_to_rgb(batch):
    """uint8 (B, H*3/2, W) nv12 on GPU -> float16 (B, 3, H, W) RGB in [0, 1]."""
    y = batch[:, :H].float()
    uv = batch[:, H:].reshape(batch.shape[0], H // 2, W // 2, 2).float() - 128
    u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2)
    v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2)
    y = (y - 16) * 1.164
    r = y + 1.793 * v
    g = y - 0.213 * u - 0.533 * v
    b = y + 2.112 * u
    return (torch.stack([r, g, b], 1).clamp_(0, 255) / 255).half()


def default_centres(source="tactical"):
    """Expected ball pixel per frame: cleaned video ball, else provided ball, carried forward."""
    if source == "tactical":
        v = np.load(os.path.join(VIDEO_LANE, "ball_video.npz"))
        px, py = v["px"].astype(np.float64), v["py"].astype(np.float64)
    else:
        v = np.load(os.path.join(VIDEO_LANE, "ball_broadcast.npz"))
        px = np.full(N, np.nan)
        py = np.full(N, np.nan)
        B = broadcast_index()
        ok = (B >= 0) & np.isfinite(v["px"])
        out_x = np.full(B.max() + 1, np.nan)
        out_y = np.full(B.max() + 1, np.nan)
        out_x[B[ok]], out_y[B[ok]] = v["px"][ok], v["py"][ok]
        return out_x, out_y   # indexed by broadcast frame
    miss = np.isnan(px)
    prov = np.load(TRACKING)["ball"][:, :2].astype(np.float64)
    pp = to_pixels(homographies(), np.nan_to_num(prov, nan=0)[:, None, :])[:, 0]
    use = miss & np.isfinite(prov[:, 0])
    px[use], py[use] = pp[use, 0], pp[use, 1]
    return px, py


def fill_forward(px, py):
    last = (W / 2, H / 2)
    px, py = px.copy(), py.copy()
    for i in range(len(px)):
        if np.isfinite(px[i]) and -200 < px[i] < W + 200 and -200 < py[i] < H + 200:
            last = (px[i], py[i])
        else:
            px[i], py[i] = last
    return px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("out")
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--centres", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mask", default="", help="npy bool per frame of this video: infer only where True")
    a = ap.parse_args()
    torch.set_num_threads(2)
    model, names = load_model(a.weights)
    ball_c = [k for k, v in names.items() if v.lower() == "ball"][0]
    if a.centres:
        z = np.load(a.centres)
        cx, cy = z["px"].astype(np.float64), z["py"].astype(np.float64)
    else:
        cx, cy = default_centres(a.source)
    frames = np.arange(a.start, a.end)
    cx, cy = fill_forward(cx[frames], cy[frames])
    c = a.crop
    x0 = np.clip(np.round(cx - c / 2), 0, W - c).astype(int)
    y0 = np.clip(np.round(cy - c / 2), 0, H - c).astype(int)
    s = a.size / c
    balls = np.full((len(frames), K, 5), np.nan, np.float32)
    todo = np.load(a.mask)[frames] if a.mask else np.ones(len(frames), bool)
    fq = queue.Queue(maxsize=48)
    th = threading.Thread(target=reader, args=(VIDEOS[a.source], a.start, len(frames), fq), daemon=True)
    th.start()
    i, t0, done = 0, time.time(), False
    while not done:
        bufs = []
        while len(bufs) < a.batch:
            item = fq.get()
            if item is None:
                done = True
                break
            bufs.append(item)
        if not bufs:
            break
        n = len(bufs)
        sel = [j for j in range(n) if i + j < len(frames) and todo[i + j]]
        if not sel:
            i += n
            continue
        arr = torch.from_numpy(np.frombuffer(b"".join(bufs[j] for j in sel), np.uint8).copy().reshape(len(sel), H * 3 // 2, W)).cuda()
        rgb = nv12_to_rgb(arr)
        crops = torch.stack([rgb[q, :, y0[i + j]:y0[i + j] + c, x0[i + j]:x0[i + j] + c] for q, j in enumerate(sel)])
        if a.size != c:
            crops = F.interpolate(crops, size=(a.size, a.size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = model(crops)
            out = out[0] if isinstance(out, (list, tuple)) else out
        out = out.float()
        sc, idx = out[:, 4 + ball_c].topk(64, dim=1)
        bx = torch.gather(out[:, :4], 2, idx[:, None].expand(-1, 4, -1)).cpu().numpy()
        sc = sc.cpu().numpy()
        for q, j in enumerate(sel):
            kept = []
            for m in range(64):
                if sc[q, m] < 0.02 or len(kept) >= K:
                    break
                bxm = bx[q, :, m]
                X = x0[i + j] + bxm[0] / s
                Y = y0[i + j] + bxm[1] / s
                if any(abs(X - p[0]) < 4 and abs(Y - p[1]) < 4 for p in kept):
                    continue
                kept.append((X, Y, bxm[2] / s, bxm[3] / s, sc[q, m]))
            for slot, row in enumerate(kept):
                balls[i + j, slot] = row
        i += n
        if i % 4000 < a.batch:
            print(f"{i}/{len(frames)} {i / (time.time() - t0):.1f} fps", flush=True)
    print(f"done {i} frames in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, frames=frames[:i], balls=balls[:i], centre=np.c_[x0 + c / 2, y0 + c / 2][:i])


if __name__ == "__main__":
    main()
