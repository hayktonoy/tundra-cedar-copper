"""free-owner's evidence.npz with only its label-free keys.

    cd pipeline/free-owner
    uv run --no-project --with numpy --with pandas --with scipy --with numba --with lightgbm --with scikit-learn python -W ignore ../from-raw/owner_evidence.py

This writes the EM status, team shape and contact channels used by the later lanes.
"""
import os
import sys

import numpy as np

OWNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "free-owner")
sys.path.insert(0, OWNER)
os.chdir(OWNER)

from common import LOCAL  # noqa: E402
from tune import build  # noqa: E402

ev = build("q_team1.npy", "llr_dead.npy")
np.savez_compressed(os.path.join(LOCAL, "evidence.npz"),
                    p_dead_em=np.load(os.path.join(LOCAL, "p_dead.npy")).astype(np.float32),
                    p_home_shape=np.load(os.path.join(LOCAL, "q_team1.npy")).astype(np.float32),
                    contact=ev.ct.astype(np.int8))
print("wrote evidence.npz with p_dead_em, p_home_shape, contact", flush=True)
