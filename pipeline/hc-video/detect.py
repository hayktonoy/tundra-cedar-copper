"""Run the public Soccana YOLO11n football detector (player / ball / referee) over a video.

    python detect.py tactical 11185 90652 out.npz [--stride 1]

Frames are decoded by FFmpeg on the GPU as nv12, piped raw, converted to RGB with torch and fed
to the network at native resolution (1920x1088) in FP16, because the ball is only ~5 px wide in
the tactical view and down-scaling loses it. For every frame we keep the K best ball candidates
(centre x, y, width, height, score) and the referee boxes, in original pixels.
"""
import argparse
import importlib
import os
import subprocess
import threading
import queue
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS = os.path.join(ROOT, "pipeline", "football-ball-detector", ".local", "football_best.pt")
VIDEOS = {"tactical": os.path.join(ROOT, "official", "data", "tactical", "BAR-ATH_tactical.mp4"),
          "broadcast": os.path.join(ROOT, "official", "data", "broadcast", "BAR-ATH_broadcast.mp4")}
W, H, HP = 1920, 1080, 1088
K_BALL, K_REF = 8, 6


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


def reader(video, start, n, stride, out):
    vf = [] if stride == 1 else ["-vf", f"select='not(mod(n\\,{stride}))'", "-vsync", "0"]
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-ss", f"{start / 25:.4f}",
           "-i", video, "-frames:v", str((n + stride - 1) // stride), *vf,
           "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
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
    """uint8 (B, H*3/2, W) nv12 on GPU -> float16 (B, 3, HP, W) RGB in [0, 1], padded to HP rows."""
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


def top_boxes(boxes, scores, k, nms_dist):
    """Top-k boxes of one class by score with a simple centre-distance suppression.

    boxes (4, M) and scores (M,) are numpy arrays already sorted by descending score."""
    kept = []
    for i in range(len(scores)):
        s = scores[i]
        if s < 0.02 or len(kept) >= k:
            break
        x, y, w, h = boxes[:, i]
        if any(abs(x - a) < nms_dist and abs(y - b) < nms_dist for a, b, *_ in kept):
            continue
        kept.append((x, y, w, h, s))
    return kept


def candidates(out, cls, m):
    """For each image, the m highest-scoring anchors of one class: (B, 4, m) boxes, (B, m) scores."""
    sc, idx = out[:, 4 + cls].topk(m, dim=1)
    bx = torch.gather(out[:, :4], 2, idx[:, None].expand(-1, 4, -1))
    return bx.cpu().numpy(), sc.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(2)
    model, names = load_model()
    ball_c = [k for k, v in names.items() if v.lower() == "ball"][0]
    ref_c = [k for k, v in names.items() if v.lower() == "referee"][0]
    frames = np.arange(a.start, a.end, a.stride)
    balls = np.full((len(frames), K_BALL, 5), np.nan, np.float32)
    refs = np.full((len(frames), K_REF, 5), np.nan, np.float32)
    q = queue.Queue(maxsize=32)
    th = threading.Thread(target=reader, args=(VIDEOS[a.source], a.start, a.end - a.start, a.stride, q), daemon=True)
    th.start()
    i, t0, done = 0, time.time(), False
    while not done:
        bufs = []
        while len(bufs) < a.batch:
            item = q.get()
            if item is None:
                done = True
                break
            bufs.append(item)
        if not bufs:
            break
        arr = torch.from_numpy(np.frombuffer(b"".join(bufs), np.uint8).copy().reshape(len(bufs), H * 3 // 2, W)).cuda()
        with torch.no_grad():
            out = model(nv12_to_rgb(arr))
            out = out[0] if isinstance(out, (list, tuple)) else out
        out = out.float()
        bb, bs = candidates(out, ball_c, 60)
        rb, rs = candidates(out, ref_c, 120)
        for j in range(len(bufs)):
            if i + j >= len(frames):
                break
            for slot, row in enumerate(top_boxes(bb[j], bs[j], K_BALL, 6)):
                balls[i + j, slot] = row
            for slot, row in enumerate(top_boxes(rb[j], rs[j], K_REF, 20)):
                refs[i + j, slot] = row
        i += len(bufs)
        if i % 2000 < a.batch:
            print(f"{i}/{len(frames)} {i / (time.time() - t0):.1f} fps", flush=True)
    print(f"done {i} frames (expected {len(frames)}) in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, frames=frames[:i], balls=balls[:i], refs=refs[:i])


if __name__ == "__main__":
    main()
