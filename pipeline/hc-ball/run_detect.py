"""Run detect.py over a frame range in resumable 10,000-frame chunks, then merge.

    python run_detect.py tactical 11185 90652 .local/zoom_tac_h1.npz [detect.py options]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 10000


def main(source, start, end, out, *opts):
    start, end = int(start), int(end)
    tag = os.path.splitext(os.path.basename(out))[0]
    parts = []
    for a in range(start, end, CHUNK):
        b = min(end, a + CHUNK)
        part = os.path.join(HERE, ".local", "chunks", f"{tag}_{a}_{b}.npz")
        os.makedirs(os.path.dirname(part), exist_ok=True)
        if not os.path.exists(part):
            subprocess.run([sys.executable, os.path.join(HERE, "detect.py"), source, str(a), str(b), part, *opts], check=True)
        parts.append(np.load(part))
    np.savez_compressed(out, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0].files})
    print("merged", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
