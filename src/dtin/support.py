"""Cell means on a regular lattice anchored at the data minimum (GStatSim convention)."""

from __future__ import annotations

import numpy as np


def cell_index(points: np.ndarray, res: float):
    """Nearest lattice node for each point, as in ``gstatsim.Gridding.grid_data``."""
    origin = points.min(axis=0)
    idx = np.rint((points - origin) / res).astype(np.int64)
    nodes = origin + idx * res
    return idx, nodes, origin


def grid_mean(points: np.ndarray, values: np.ndarray, res: float):
    idx, _, origin = cell_index(points, res)
    keys, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    sums = np.bincount(inverse, weights=values, minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys))
    return origin + keys * res, sums / counts
