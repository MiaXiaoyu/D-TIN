"""D-TIN: Delaunay triangulation with barycentric interpolation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

try:
    from scipy.spatial import QhullError
except ImportError:  # scipy < 1.8
    from scipy.spatial.qhull import QhullError

OUTSIDE_RULES = ("nan", "nearest")


def triangulate(points: np.ndarray) -> Delaunay:
    """Delaunay triangulation with a clear error for degenerate input."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {points.shape}")
    if len(points) < 3:
        raise ValueError(
            f"the triangulation needs at least three non-collinear points, got {len(points)}"
        )
    try:
        return Delaunay(points)
    except QhullError:
        raise ValueError(
            "the triangulation needs at least three non-collinear points; "
            "the input is collinear or otherwise degenerate"
        ) from None


def hull_triangulation(points: np.ndarray) -> Delaunay:
    """Triangulation of the convex hull vertices only."""
    points = np.asarray(points, dtype=np.float64)
    try:
        vertices = ConvexHull(points).vertices
    except (QhullError, ValueError):
        return triangulate(points)
    return triangulate(points[vertices])


def hull_mask(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """True where ``query`` lies inside the convex hull of ``points``."""
    return triangulate(points).find_simplex(np.asarray(query, dtype=np.float64)) >= 0


class Surface:
    """Base class: ``fit(points, values)`` then ``predict(query, outside)``.

    Outside the convex hull of the fitted points ``predict`` returns NaN, or
    the nearest measurement with ``outside="nearest"``.
    """

    _points: np.ndarray

    def fit(self, points: np.ndarray, values: np.ndarray) -> "Surface":
        raise NotImplementedError

    def _predict_raw(self, query: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _inside(self, query: np.ndarray) -> np.ndarray:
        # Built once per fit from the hull vertices only.
        if getattr(self, "_hull_source", None) is not self._points:
            self._hull_tri = hull_triangulation(self._points)
            self._hull_source = self._points
        return self._hull_tri.find_simplex(np.asarray(query, dtype=np.float64)) >= 0

    def predict(self, query: np.ndarray, outside: str = "nan") -> np.ndarray:
        if outside not in OUTSIDE_RULES:
            raise ValueError(f"outside must be one of {OUTSIDE_RULES}, got {outside!r}")
        query = np.asarray(query, dtype=np.float64)
        out = np.array(self._predict_raw(query), dtype=np.float64, copy=True)
        if outside == "nan":
            out[~self._inside(query)] = np.nan
        return out


def barycentric_eval(
    tri: Delaunay,
    values: np.ndarray,
    query: np.ndarray,
    simplex: np.ndarray,
) -> np.ndarray:
    """Linear interpolation inside known containing simplices.

    Parameters
    ----------
    tri
        Triangulation whose ``points`` are the vertices.
    values
        Vertex values, aligned with ``tri.points``.
    query
        Query coordinates, shape (n, 2). Every row must lie in its simplex.
    simplex
        Containing simplex index per query point, all >= 0.
    """
    simplices = tri.simplices[simplex]
    verts = tri.points[simplices]
    vals = values[simplices]

    v0 = verts[:, 0]
    e1 = verts[:, 1] - v0
    e2 = verts[:, 2] - v0
    rel = query - v0

    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", rel, e1)
    d21 = np.einsum("ij,ij->i", rel, e2)

    denom = d00 * d11 - d01 * d01
    denom = np.where(denom == 0.0, 1e-12, denom)
    b = (d11 * d20 - d01 * d21) / denom
    c = (d00 * d21 - d01 * d20) / denom
    a = 1.0 - b - c
    return a * vals[:, 0] + b * vals[:, 1] + c * vals[:, 2]


class _NearestFallbackMixin:
    """Nearest-measurement values for query points a method cannot serve."""

    _points: np.ndarray
    _values: np.ndarray
    _tree: cKDTree

    @property
    def n_points(self) -> int:
        """Size of the point set this surface is built on."""
        return 0 if getattr(self, "_points", None) is None else len(self._points)

    def _nearest(self, query: np.ndarray) -> np.ndarray:
        _, idx = self._tree.query(query, k=1)
        return self._values[idx]


class DTIN(_NearestFallbackMixin, Surface):
    """Piecewise-linear interpolation on the Delaunay triangulation of the measurements."""

    def __init__(self, aniso: tuple[float, float] = (1.0, 1.0)) -> None:
        self.tri: Delaunay | None = None
        # Optional (x, y) scaling before triangulation; (1, 1) is isotropic.
        self.aniso = (float(aniso[0]), float(aniso[1]))

    def _metric(self, points: np.ndarray) -> np.ndarray:
        sx, sy = self.aniso
        if sx == 1.0 and sy == 1.0:
            return points
        return points / np.array([sx, sy])

    def fit(self, points: np.ndarray, values: np.ndarray) -> "DTIN":
        self._points = np.asarray(points, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self.tri = triangulate(self._metric(self._points))
        self._tree = cKDTree(self._points)
        return self

    @property
    def n_simplices(self) -> int:
        return 0 if self.tri is None else len(self.tri.simplices)

    def _predict_raw(self, query: np.ndarray) -> np.ndarray:
        if self.tri is None:
            raise RuntimeError("DTIN.predict called before fit")
        query = np.asarray(query, dtype=np.float64)

        metric_query = self._metric(query)
        simplex = self.tri.find_simplex(metric_query)
        out = np.empty(len(query))
        inside = simplex >= 0
        if inside.any():
            out[inside] = barycentric_eval(
                self.tri, self._values, metric_query[inside], simplex[inside]
            )
        if (~inside).any():
            out[~inside] = self._nearest(query[~inside])
        return out

    def _inside(self, query: np.ndarray) -> np.ndarray:
        if self.tri is None:
            raise RuntimeError("DTIN.predict called before fit")
        return self.tri.find_simplex(self._metric(np.asarray(query, dtype=np.float64))) >= 0
