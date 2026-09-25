"""Synthetic track-sampled point cloud used by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtin.projection import unproject

X0, Y0 = -200_000.0, -2_200_000.0   # EPSG:3413 metres, central Greenland


def truth(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return 1500.0 + 0.01 * (x - X0) + 0.02 * (y - Y0) + 50.0 * np.sin((x - X0) / 5000.0)


def make_tracks(seed: int = 0, n_tracks: int = 12, length_m: float = 60_000.0) -> pd.DataFrame:
    """Twelve north-south tracks 5 km apart, 172 m shot spacing, 30 m cross-track jitter."""
    rng = np.random.default_rng(seed)
    along = np.arange(0.0, length_m, 172.0)
    xs, ys = [], []
    for i in range(n_tracks):
        x_track = X0 + 2_500.0 + 5_000.0 * i
        xs.append(x_track + rng.normal(0.0, 30.0, size=len(along)))
        ys.append(Y0 + along)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    elev = truth(x, y) + rng.normal(0.0, 0.5, size=len(x))
    lon, lat = unproject(x, y, "EPSG:3413")
    return pd.DataFrame({"lon": lon, "lat": lat, "elev": elev})
