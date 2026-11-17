"""Give the lanes an empty stand-in for the external model, which this pipeline does not use.

    python pipeline/last-touch/stubs.py [--force]

The base's code still opens the external (Metrica/SkillCorner) model's output. Its weight in the
frozen recipe is 0 (pipeline/fm-rebuild/recipe/decoder.json, `ext`), and replacing the real file
with a uniform one leaves the base byte-identical, so it is written here as "no opinion": a
uniform three-class probability. The action spotter is no longer stubbed: the free-pretrained
stage runs it. An existing file that is not such a stand-in is left alone unless --force is given.
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
N = 199319


def blank(path):
    """True if the file is missing or already a stand-in."""
    if not path.exists():
        return True
    z = np.load(path)
    return bool(np.allclose(z['p'], 1 / 3))


def main():
    force = '--force' in sys.argv
    path = EXP / 'free-external/.local/model-equal-sc-metrica/target.npz'
    if not blank(path) and not force:
        raise SystemExit(f'{path} holds real output; pass --force to replace it with a stand-in')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=np.full((len(range(0, N, 5)), 3), 1 / 3, 'f4'), original_index=np.arange(0, N, 5))
    print('stand-in written:', path.relative_to(EXP))


if __name__ == '__main__':
    main()
