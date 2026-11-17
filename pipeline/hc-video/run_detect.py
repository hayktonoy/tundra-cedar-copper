"""Run detect.py in resumable 10,000-frame chunks and merge them.

    python run_detect.py tactical 110600 190400 .local/det_tac_h2.npz
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 10000


def main(source, start, end, out):
    start, end = int(start), int(end)
    parts = []
    for a in range(start, end, CHUNK):
        b = min(end, a + CHUNK)
        part = os.path.join(HERE, ".local", "chunks", f"{source}_{a}_{b}.npz")
        os.makedirs(os.path.dirname(part), exist_ok=True)
        if not os.path.exists(part):
            subprocess.run([sys.executable, os.path.join(HERE, "detect.py"), source, str(a), str(b), part], check=True)
        parts.append(np.load(part))
    np.savez_compressed(out, **{k: np.concatenate([p[k] for p in parts]) for k in ("frames", "balls", "refs")})
    print("merged", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
