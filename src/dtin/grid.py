"""Regular prediction grids in the working CRS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_CELLS = 50_000_000


@dataclass(frozen=True)
class GridSpec:
    xs: np.ndarray
    ys: np.ndarray
    res_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ys), len(self.xs))

    @property
    def n_cells(self) -> int:
        return len(self.xs) * len(self.ys)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.xs[0]), float(self.ys[0]), float(self.xs[-1]), float(self.ys[-1]))

    @property
    def nodes(self) -> np.ndarray:
        """(n_cells, 2) node coordinates, row-major: y outer, x inner."""
        X, Y = np.meshgrid(self.xs, self.ys)
        return np.column_stack([X.ravel(), Y.ravel()])


def make_grid(points, res_m: float, bbox=None, force: bool = False) -> GridSpec:
    """Nodes at xmin + i*res, ymin + j*res over the points' bounding box or ``bbox``."""
    res_m = float(res_m)
    if not res_m > 0.0:
        raise ValueError(f"grid resolution must be positive, got {res_m}")
    points = np.asarray(points, dtype=np.float64)
    if bbox is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
    else:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"bbox must be xmin ymin xmax ymax with xmin < xmax and ymin < ymax, got {bbox}")
    nx = int(np.floor((xmax - xmin) / res_m + 1e-9)) + 1
    ny = int(np.floor((ymax - ymin) / res_m + 1e-9)) + 1
    if nx * ny > MAX_CELLS and not force:
        raise ValueError(
            f"grid would have {nx * ny:,} cells (more than {MAX_CELLS:,}); "
            "use a coarser --grid or a --bbox, or pass --force"
        )
    xs = xmin + res_m * np.arange(nx)
    ys = ymin + res_m * np.arange(ny)
    return GridSpec(xs, ys, res_m)
