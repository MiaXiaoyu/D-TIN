"""Polar stereographic projections (EPSG:3413, EPSG:3031) without pyproj.

Ellipsoidal formulas from Snyder (1987), *Map Projections: A Working Manual*.
Other CRS codes are passed to pyproj (the ``crs`` extra).
"""

from __future__ import annotations

import numpy as np

A = 6378137.0                      # WGS84 semi-major axis, m
F = 1.0 / 298.257223563            # WGS84 flattening
E2 = 2.0 * F - F * F
E = np.sqrt(E2)

#: crs -> (hemisphere sign, latitude of true scale, central meridian)
BUILTIN = {
    "EPSG:3413": (+1.0, 70.0, -45.0),
    "EPSG:3031": (-1.0, -71.0, 0.0),
}


def normalise_crs(crs: str) -> str:
    crs = str(crs).strip().upper()
    if crs.isdigit():
        crs = "EPSG:" + crs
    return crs


def _t(phi: np.ndarray) -> np.ndarray:
    s = E * np.sin(phi)
    return np.tan(np.pi / 4.0 - phi / 2.0) / ((1.0 - s) / (1.0 + s)) ** (E / 2.0)


def _m(phi: float) -> float:
    return np.cos(phi) / np.sqrt(1.0 - E2 * np.sin(phi) ** 2)


def project(lon, lat, crs: str = "EPSG:3413"):
    """Longitude/latitude in degrees -> (x, y) in metres of ``crs``."""
    crs = normalise_crs(crs)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if crs not in BUILTIN:
        x, y = _pyproj(crs).transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    sign, lat_ts, lon0 = BUILTIN[crs]
    phi = np.radians(lat) * sign
    phi_c = np.radians(lat_ts) * sign
    rho = A * _m(phi_c) * _t(phi) / _t(phi_c)
    dlam = np.radians(lon - lon0)
    return rho * np.sin(dlam), -sign * rho * np.cos(dlam)


def unproject(x, y, crs: str = "EPSG:3413"):
    """(x, y) in metres of ``crs`` -> longitude/latitude in degrees."""
    crs = normalise_crs(crs)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if crs not in BUILTIN:
        lon, lat = _pyproj(crs).transform(x, y, direction="INVERSE")
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    sign, lat_ts, lon0 = BUILTIN[crs]
    phi_c = np.radians(lat_ts) * sign
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
    lon = lon0 + np.degrees(np.arctan2(x, -sign * y))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, sign * np.degrees(phi)


def _pyproj(crs: str):
    try:
        import pyproj
    except ImportError:
        raise ImportError(
            f"CRS {crs} is not built in; install pyproj with: pip install 'dtin[crs]'"
        ) from None
    try:
        return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    except Exception as exc:
        raise ValueError(
            f"unknown CRS {crs!r}; use an EPSG code such as EPSG:3413 ({exc})"
        ) from None
