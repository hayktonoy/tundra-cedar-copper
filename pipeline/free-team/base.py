"""Shared paths for the free-team lane: free-owner's loaders and decoder, with this lane's .local."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = os.path.join(os.path.dirname(HERE), "free-owner")
sys.path.insert(0, OWNER)

import common as fo  # noqa: E402  free-owner's common: paths and loaders
from common import (A, BLOCKS, D, FPS, H, HALVES, N, PITCH, X, codes, graded,  # noqa: E402,F401
                    runs, within, write)

LOCAL = os.path.join(HERE, ".local")
