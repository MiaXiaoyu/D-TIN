"""Read a lon/lat/elev CSV and project it to EPSG:3413 metres.

Rows with a missing or non-numeric longitude, latitude or elevation are
dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dtin.projection import project

REQUIRED_COLUMNS = ("lon", "lat", "elev")


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
    if path.suffix.lower() != ".csv":
        raise ValueError(f"unsupported input type {path.suffix!r}; use a CSV file")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def load_points(
    path,
    *,
    min_elev: float | None = None,
    need_elev: bool = True,
) -> pd.DataFrame:
    """Read a CSV and return x, y, lon, lat[, elev] with a LoadReport."""
    raw = read_table(path)
    required = REQUIRED_COLUMNS if need_elev else REQUIRED_COLUMNS[:2]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(
            f"required columns are {', '.join(required)}; missing {', '.join(missing)}; "
            f"found {list(raw.columns)}"
        )

    frame = pd.DataFrame(
        {name: pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=np.float64)
         for name in required}
    )
    keep = np.isfinite(frame.to_numpy()).all(axis=1)
    n_nan = int((~keep).sum())
    frame = frame[keep]

    n_min = 0
    if min_elev is not None and need_elev:
        low = frame["elev"] < float(min_elev)
        n_min = int(low.sum())
        frame = frame[~low]

    frame["x"], frame["y"] = project(frame["lon"].to_numpy(), frame["lat"].to_numpy())

    order = ["x", "y", "lon", "lat"] + (["elev"] if need_elev else [])
    frame = frame[order].reset_index(drop=True)
    columns = {name: name for name in required}
    frame.attrs["report"] = LoadReport(str(path), len(raw), n_nan, n_min, columns, "EPSG:3413")
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
