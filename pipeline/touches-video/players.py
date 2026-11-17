"""Player boxes with jersey colours from the tactical video, on the GPU.

    python players.py START END OUT.npz [broadcast]   # one chunk of frames
    python players.py all [broadcast]                 # every graded frame, resumable 10k chunks

Same detector and decoding as pipeline/hc-video/detect.py (Soccana YOLO11n, FFmpeg nv12 on
the GPU, native 1920x1088, FP16), but keeps the player and referee boxes (NMS IoU .5, score
>= .25, at most 32 per frame) and, for each, a 4x3 RGB patch of the torso (roi_align over the
box's 15-55% rows and middle half), which is enough to tell Barca's stripes from Athletic's
white. Output: frames (n,), boxes (n,32,6) x, y, w, h, score, class (0 player, 1 referee),
torso (n,32,4,3,3) uint8.
"""
import importlib
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS = os.path.join(ROOT, "pipeline", "football-ball-detector", ".local", "football_best.pt")
VIDEOS = {"tactical": os.path.join(ROOT, "official", "data", "tactical", "BAR-ATH_tactical.mp4"),
          "broadcast": os.path.join(ROOT, "official", "data", "broadcast", "BAR-ATH_broadcast.mp4")}
W, H, HP = 1920, 1080, 1088
K = 32
RANGES = {"tactical": [(11185, 90652), (110808, 190198)],
          "broadcast": [(6980, 84800), (86540, 163500)]}   # broadcast frames covering both halves
CHUNK = 10000


def load_model():
    names = torch.serialization.get_unsafe_globals_in_checkpoint(WEIGHTS)
    objects = []
    for name in names:
        module, symbol = name.rsplit(".", 1)
        obj = getattr(importlib.import_module(module), symbol)
        assert isinstance(obj, type) and issubclass(obj, torch.nn.Module), name
        objects.append(obj)
    with torch.serialization.safe_globals(objects):
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = ckpt["model"].float().eval().cuda().half()
    return model, model.names


def reader(start, n, out, video):
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
    y = batch[:, :H].float()
    uv = batch[:, H:].reshape(batch.shape[0], H // 2, W // 2, 2).float() - 128
    u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2)
    v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2)
    y = (y - 16) * 1.164
    r = y + 1.793 * v
    g = y - 0.213 * u - 0.533 * v
    b = y + 2.112 * u
    rgb = torch.stack([r, g, b], 1).clamp_(0, 255) / 255
    out = torch.full((batch.shape[0], 3, HP, W), 114 / 255, device=batch.device, dtype=torch.float16)
    out[:, :, :H] = rgb.half()
    return out


def run(start, end, path, source="tactical", batch=8):
    torch.set_num_threads(2)
    model, names = load_model()
    pc = [k for k, v in names.items() if v.lower() == "player"][0]
    rc = [k for k, v in names.items() if v.lower() == "referee"][0]
    n = end - start
    boxes = np.full((n, K, 6), np.nan, np.float32)
    torso = np.zeros((n, K, 4, 3, 3), np.uint8)
    q = queue.Queue(maxsize=32)
    threading.Thread(target=reader, args=(start, n, q, VIDEOS[source]), daemon=True).start()
    i, t0, done = 0, time.time(), False
    while not done:
        bufs = []
        while len(bufs) < batch:
            item = q.get()
            if item is None:
                done = True
                break
            bufs.append(item)
        if not bufs:
            break
        arr = torch.from_numpy(np.frombuffer(b"".join(bufs), np.uint8).copy().reshape(len(bufs), H * 3 // 2, W)).cuda()
        with torch.no_grad():
            img = nv12_to_rgb(arr)
            out = model(img)
            out = out[0] if isinstance(out, (list, tuple)) else out
            out = out.float()                                  # (B, 4+nc, M)
            sc, cls = torch.stack([out[:, 4 + pc], out[:, 4 + rc]]).max(0)   # (B, M)
            bi, mi = torch.nonzero(sc > 0.25, as_tuple=True)
            bx = out[bi, :4, mi]                               # (L, 4) cx, cy, w, h
            s1, c1 = sc[bi, mi], cls[bi, mi]
            xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1)
            k = torchvision.ops.batched_nms(xyxy, s1, bi, 0.5)   # sorted by score
            bi, bx, s1, c1, xyxy = bi[k], bx[k], s1[k], c1[k], xyxy[k]
            w = xyxy[:, 2] - xyxy[:, 0]
            hgt = xyxy[:, 3] - xyxy[:, 1]
            t_box = torch.stack([bi.float(), xyxy[:, 0] + .25 * w, xyxy[:, 1] + .15 * hgt,
                                 xyxy[:, 2] - .25 * w, xyxy[:, 1] + .55 * hgt], 1)
            patches = torchvision.ops.roi_align(img, t_box.half(), (4, 3), 1.0, 2, True)
            patches = (patches.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            meta = torch.cat([bx, s1[:, None], c1[:, None].float()], 1).cpu().numpy()
            bi = bi.cpu().numpy()
        slot = np.zeros(len(bufs), int)
        for r in range(len(bi)):
            j = bi[r]
            if slot[j] < K and i + j < n:
                boxes[i + j, slot[j]] = meta[r]
                torso[i + j, slot[j]] = patches[r]
                slot[j] += 1
        i += len(bufs)
        if i % 2000 < batch:
            print(f"{start + i} {i / (time.time() - t0):.1f} fps", flush=True)
    print(f"done {i}/{n} frames in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(path, frames=np.arange(start, start + i), boxes=boxes[:i], torso=torso[:i])


def run_all(source="tactical"):
    d = os.path.join(HERE, ".local", "players" if source == "tactical" else "players_broadcast")
    os.makedirs(d, exist_ok=True)
    for lo, hi in RANGES[source]:
        for a in range(lo, hi, CHUNK):
            b = min(hi, a + CHUNK)
            p = os.path.join(d, f"{source[:3]}_{a}_{b}.npz")
            if not os.path.exists(p):
                subprocess.run([sys.executable, __file__, str(a), str(b), p, source], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        run_all(*sys.argv[2:3])
    else:
        run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], *sys.argv[4:5])
