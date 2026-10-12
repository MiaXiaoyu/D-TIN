"""Support helpers: cell means on a regular lattice (GStatSim convention) and
the distance from a query to the nearest measurement."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


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


def nearest_distance(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Distance from each query location to the nearest measurement.

    The paper reports error by exactly this quantity: a prediction several
    kilometres from the closest observation is bridged by a long triangle and
    carries a much larger expected error than one next to a track.
    """
    points = np.asarray(points, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if len(query) == 0:
        return np.empty(0)
    return cKDTree(points).query(query, k=1)[0]
