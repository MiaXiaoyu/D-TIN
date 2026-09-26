"""WGS84 longitude/latitude to EPSG:3413 without pyproj.

Ellipsoidal formulas from Snyder (1987), *Map Projections: A Working Manual*.
"""

from __future__ import annotations

import numpy as np

A = 6378137.0                      # WGS84 semi-major axis, m
F = 1.0 / 298.257223563            # WGS84 flattening
E2 = 2.0 * F - F * F
E = np.sqrt(E2)

LAT_TS = 70.0
LON_0 = -45.0


def _t(phi: np.ndarray) -> np.ndarray:
    s = E * np.sin(phi)
    return np.tan(np.pi / 4.0 - phi / 2.0) / ((1.0 - s) / (1.0 + s)) ** (E / 2.0)


def _m(phi: float) -> float:
    return np.cos(phi) / np.sqrt(1.0 - E2 * np.sin(phi) ** 2)


def project(lon, lat):
    """WGS84 longitude/latitude in degrees -> EPSG:3413 x/y in metres."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    phi = np.radians(lat)
    phi_c = np.radians(LAT_TS)
    rho = A * _m(phi_c) * _t(phi) / _t(phi_c)
    dlam = np.radians(lon - LON_0)
    return rho * np.sin(dlam), -rho * np.cos(dlam)


def unproject(x, y):
    """EPSG:3413 x/y in metres -> WGS84 longitude/latitude in degrees."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    phi_c = np.radians(LAT_TS)
    rho = np.hypot(x, y)
    t = rho * _t(phi_c) / (A * _m(phi_c))
    chi = np.pi / 2.0 - 2.0 * np.arctan(t)
    e2, e4, e6, e8 = E2, E2 ** 2, E2 ** 3, E2 ** 4
    phi = (
        chi
        + (e2 / 2 + 5 * e4 / 24 + e6 / 12 + 13 * e8 / 360) * np.sin(2 * chi)
        + (7 * e4 / 48 + 29 * e6 / 240 + 811 * e8 / 11520) * np.sin(4 * chi)
        + (7 * e6 / 120 + 81 * e8 / 1120) * np.sin(6 * chi)
        + (4279 * e8 / 161280) * np.sin(8 * chi)
    )
    lon = LON_0 + np.degrees(np.arctan2(x, -y))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, np.degrees(phi)
