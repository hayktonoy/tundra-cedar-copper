"""free-pretrained/tdeed_run.py with cuDNN's autotuner off and deterministic kernels on (test only).

    cd pipeline/free-pretrained
    .local/venv/Scripts/python.exe -I ../from-raw/tdeed_det.py broadcast 6900 84900

tdeed_run.py sets torch.backends.cudnn.benchmark = True, which times several convolution
algorithms on the first batches and keeps the fastest; which one wins can change between runs,
and different algorithms round differently in FP16. This runs tdeed_run.py's own source with
that line replaced by benchmark = False and deterministic = True (and nothing else changed), to
measure whether that makes the spotter repeat bit for bit and what it costs in time. It writes the
same output file as tdeed_run.py.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'free-pretrained' / 'tdeed_run.py'
code = SRC.read_text(encoding='utf-8')
old = 'torch.backends.cudnn.benchmark = True'
assert code.count(old) == 1
code = code.replace(old, 'torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True')
sys.argv[0] = str(SRC)
exec(compile(code, str(SRC), 'exec'), {'__file__': str(SRC), '__name__': '__main__'})
