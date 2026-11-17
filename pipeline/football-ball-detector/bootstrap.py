"""Use isolated vision dependencies and keep all runtime caches in this experiment."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = HERE / ".local"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
for key in ("MPLCONFIGDIR", "YOLO_CONFIG_DIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = str(LOCAL / "cache" / key.lower())
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
for path in (ROOT / "pipeline/temporal-possession/.local/torch",
             ROOT / "pipeline/visual-stoppages/.local/vision", LOCAL / "packages"):
    sys.path.insert(0, str(path))
