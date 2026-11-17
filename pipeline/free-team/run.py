"""Snapshot the inputs and build the co-trained team views."""
import os
import shutil
import subprocess
import sys

from base import HERE, LOCAL

EXP = os.path.dirname(HERE)
SNAPSHOT = {  # source -> name in .local (or .local/snap)
    "free-owner/.local/features.npz": "features.npz", "free-owner/.local/events.npz": "events.npz",
    "free-owner/.local/llr_dead.npy": "llr_dead.npy", "free-owner/.local/p_dead.npy": "p_dead.npy",
    "free-owner/.local/q_team1.npy": "q_team1.npy", "free-owner/.local/contact_broadcast.npz": "contact_broadcast.npz",
    "free-status/.local/video_dfl_detected.npz": "snap/video_dfl_detected.npz",
    "free-status/.local/physics.npz": "snap/physics.npz",
    "free-dfl/.local/dfl_only_evidence.npz": "snap/dfl_only_evidence.npz",
    "free-external/.local/model-equal-sc-metrica/target.npz": "snap/external_target.npz",
}


def snapshot():
    os.makedirs(os.path.join(LOCAL, "snap"), exist_ok=True)
    for src, dst in SNAPSHOT.items():
        d = os.path.join(LOCAL, dst)
        if not os.path.exists(d):
            shutil.copyfile(os.path.join(EXP, src), d)


def step(*args, env=None):
    print("==", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], check=True, cwd=HERE, env=dict(os.environ, **(env or {})))


def main():
    snapshot()
    step("local_features.py")
    step("cotrain.py", "local", "q_team1.npy", "q_local.npy")
    step("cotrain.py", "local", "q_team1.npy", "q_local_nodir.npy", env=dict(DROP="vf"))
    step("cotrain.py", "shape", "q_local.npy", "q_shape2.npy")


if __name__ == "__main__":
    main()
