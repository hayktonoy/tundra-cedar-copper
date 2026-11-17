"""Apply the replay's event bonuses to the pretrained decoder."""
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOCAL = HERE / '.local'
LOCAL.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

sys.path.insert(0, str(EXP / 'free-pretrained'))
from fuse import PEngine  # noqa: E402
from decode import decode, posterior  # noqa: E402  free-owner's three-state decoder


class HEngine(PEngine):
    """PEngine.scores with configurable event bonuses."""

    def __init__(self, bonus):
        super().__init__()
        self.bonus = bonus

    def scores(self, p):
        b = self.bonus
        e = self.owner
        save = e.out, e.drop, e.kick, e.setp, e.acc
        # PEngine computes 20*out+80*drop and 40*kick+3*setp+10*acc; rescale each indicator so the
        # products equal the bonus values instead. The indicators are 0/1 spreads, so this is exact.
        e.out, e.drop = save[0] * b['g_out'] / 20, save[1] * b['g_drop'] / 80
        e.kick, e.setp, e.acc = save[2] * b['h_kick'] / 40, save[3] * b['h_setp'] / 3, save[4] * b['h_acc'] / 10
        try:
            return super().scores(p)
        finally:
            e.out, e.drop, e.kick, e.setp, e.acc = save
