"""Pixel <-> pitch (metres) through the per-frame homography (normalised pitch -> pixels)."""
import numpy as np

from common import PITCH


def to_pixels(H, xy):
    """H (n,3,3), xy (n,...,2) metres -> pixels (n,...,2)."""
    shp = xy.shape
    p = xy.reshape(shp[0], -1, 2) / np.array(PITCH)
    v = np.einsum('nij,nkj->nki', H, np.concatenate([p, np.ones(p.shape[:2] + (1,))], -1))
    return (v[..., :2] / v[..., 2:]).reshape(shp)


def to_pitch(H, px):
    """H (n,3,3), px (n,...,2) pixels -> metres (n,...,2) on the ground plane."""
    shp = px.shape
    Hi = np.linalg.inv(H)
    q = px.reshape(shp[0], -1, 2)
    v = np.einsum('nij,nkj->nki', Hi, np.concatenate([q, np.ones(q.shape[:2] + (1,))], -1))
    return (v[..., :2] / v[..., 2:] * np.array(PITCH)).reshape(shp)
