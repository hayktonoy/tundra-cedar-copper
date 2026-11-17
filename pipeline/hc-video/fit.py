"""Build the ball feature caches."""
import os

import numpy as np

from common import LOCAL, TRACKING


def ball_features(kind):
    from features import features
    cache = os.path.join(LOCAL, f"feat_{kind}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    if kind == "height":
        from features import height_features
        f = height_features()
        np.savez(cache, **f)
        return f
    if kind == "pixel":
        from pixel import pixel_features
        f = pixel_features()
        np.savez(cache, **f)
        return f
    if kind == "provided":
        xy = np.load(TRACKING)["ball"][:, :2]
        f = features(xy, "p_")
    else:
        name, p = {"video": ("ball_video.npz", "v_"), "broadcast": ("ball_broadcast.npz", "b_"),
                   "fused": ("ball_fused.npz", "f_")}[kind]
        z = np.load(os.path.join(LOCAL, name))
        xy = np.c_[z["x"], z["y"]]
        f = features(xy, p)
        f[p + "source"] = z["source"].astype(np.float32)
        f[p + "score"] = z["score"]
    np.savez(cache, **f)
    return f
