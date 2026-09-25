"""Read a point file (CSV or Parquet) and project it to metres.

Rows with a missing or non-numeric longitude, latitude or elevation are
dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dtin.projection import normalise_crs, project, unproject

ALIASES = {
    "lon": ("lon", "longitude"),
    "lat": ("lat", "latitude"),
    "elev": ("elev", "elevation", "h", "z"),
    "x": ("x", "easting"),
    "y": ("y", "northing"),
}


@dataclass(frozen=True)
class LoadReport:
    path: str
    rows_read: int
    rows_dropped_nan: int
    rows_dropped_min_elev: int
    columns: dict
    crs: str

    @property
    def rows_used(self) -> int:
        return self.rows_read - self.rows_dropped_nan - self.rows_dropped_min_elev


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except ImportError:
            raise ImportError(
                "Parquet input needs pyarrow: pip install 'dtin[parquet]'"
            ) from None
    elif suffix in (".csv", ".txt"):
        frame = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(
            f"unsupported input type {path.suffix!r}; use .csv or .parquet"
        )
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def resolve_column(frame: pd.DataFrame, role: str, requested: str | None = None) -> str:
    lower = {c.lower(): c for c in frame.columns}
    candidates = (requested,) if requested else ALIASES[role]
    for name in candidates:
        if name.strip().lower() in lower:
            return lower[name.strip().lower()]
    wanted = requested or "/".join(ALIASES[role])
    raise ValueError(
        f"no column for {role} (looked for {wanted}); columns in file: {list(frame.columns)}"
    )


def load_points(
    path,
    *,
    lon_col: str | None = None,
    lat_col: str | None = None,
    elev_col: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    xy: bool = False,
    crs: str = "EPSG:3413",
    min_elev: float | None = None,
    need_elev: bool = True,
) -> pd.DataFrame:
    """Read a point file and return x, y, lon, lat[, elev] with a LoadReport."""
    crs = normalise_crs(crs)
    raw = read_table(path)
    cols: dict = {}
    if xy:
        cols["x"] = resolve_column(raw, "x", x_col)
        cols["y"] = resolve_column(raw, "y", y_col)
    else:
        cols["lon"] = resolve_column(raw, "lon", lon_col)
        cols["lat"] = resolve_column(raw, "lat", lat_col)
    if need_elev:
        cols["elev"] = resolve_column(raw, "elev", elev_col)

    frame = pd.DataFrame(
        {role: pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=np.float64)
         for role, c in cols.items()}
    )
    keep = np.isfinite(frame.to_numpy()).all(axis=1)
    n_nan = int((~keep).sum())
    frame = frame[keep]

    n_min = 0
    if min_elev is not None and need_elev:
        low = frame["elev"] < float(min_elev)
        n_min = int(low.sum())
        frame = frame[~low]

    if xy:
        frame["lon"], frame["lat"] = unproject(frame["x"].to_numpy(), frame["y"].to_numpy(), crs)
    else:
        frame["x"], frame["y"] = project(frame["lon"].to_numpy(), frame["lat"].to_numpy(), crs)

    order = ["x", "y", "lon", "lat"] + (["elev"] if need_elev else [])
    frame = frame[order].reset_index(drop=True)
    frame.attrs["report"] = LoadReport(str(path), len(raw), n_nan, n_min, cols, crs)
    return frame


def describe_load(frame: pd.DataFrame) -> str:
    """Summary printed by ``dtin info``."""
    rep: LoadReport = frame.attrs["report"]
    lines = [
        f"file: {rep.path}",
        f"rows read: {rep.rows_read}",
        f"dropped (NaN): {rep.rows_dropped_nan}",
        f"dropped (below min elevation): {rep.rows_dropped_min_elev}",
        f"points used: {rep.rows_used}",
        "columns: " + ", ".join(f"{k}={v}" for k, v in rep.columns.items()),
        f"crs: {rep.crs}",
    ]
    if len(frame):
        lon0, lon1 = frame["lon"].min(), frame["lon"].max()
        lat0, lat1 = frame["lat"].min(), frame["lat"].max()
        x0, x1 = frame["x"].min(), frame["x"].max()
        y0, y1 = frame["y"].min(), frame["y"].max()
        nx = int(np.floor((x1 - x0) / 1000.0)) + 1
        ny = int(np.floor((y1 - y0) / 1000.0)) + 1
        lines += [
            f"lon/lat box: [{lon0:.4f}, {lon1:.4f}] x [{lat0:.4f}, {lat1:.4f}]",
            f"projected extent: {(x1 - x0) / 1000:.1f} km x {(y1 - y0) / 1000:.1f} km",
            f"1 km grid would have {nx} x {ny} = {nx * ny:,} cells",
        ]
        if "elev" in frame:
            lines.append(f"elevation range: {frame['elev'].min():.1f} to {frame['elev'].max():.1f} m")
    return "\n".join(lines)
