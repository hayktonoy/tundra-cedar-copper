"""Write free-rules' fold-independent channels for the replay.

    uv run --no-project --with numpy --with pandas --with scipy --with numba --with lightgbm python pipeline/from-raw/rules_inputs.py

Build the fold-independent rule inputs and copy them to the two lanes that use them.
"""
import shutil
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
src = (EXP / 'agreed-touches/rules_inputs_check.py').read_text(encoding='utf-8')
ns = {'__file__': str(EXP / 'agreed-touches/rules_inputs_check.py'), '__name__': 'rules_inputs'}
exec(compile(src, 'rules_inputs_check.py', 'exec'), ns)
out = ns['out']
dst = EXP / 'free-rules/.local/inputs.npz'
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **out)
(EXP / 'free-stack/.local').mkdir(parents=True, exist_ok=True)
shutil.copyfile(dst, EXP / 'free-stack/.local/inputs.npz')
print(f'wrote {len(out)} channels to {dst} and free-stack/.local/inputs.npz:', ', '.join(out))
