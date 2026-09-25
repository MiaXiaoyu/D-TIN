"""Tile polynomial baseline: an independent polynomial per square tile."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from dtin.tin import Surface, _NearestFallbackMixin


class TilePolynomial(_NearestFallbackMixin, Surface):
    """Polynomial of degree up to ``degree`` fitted separately in each tile.

    Coordinates are centred and scaled per tile. Each tile gets the highest
    degree its design matrix supports (condition ratio above
    ``condition_floor``), falling back to a plane or the tile mean; a query in
    an empty tile takes the nearest measurement.
    """

    N_COEFFS = {0: 1, 1: 3, 2: 6, 3: 10}

    #: Minimum ratio of smallest to largest singular value of a tile's design matrix.
    CONDITION_FLOOR = 1e-2

    SELECTIONS = ("condition", "loocv")

    def __init__(
        self,
        tile_size_m: float = 5_000.0,
        degree: int = 3,
        condition_floor: float | None = None,
        selection: str = "condition",
    ) -> None:
        if degree not in self.N_COEFFS:
            raise ValueError(f"degree must be 0-3, got {degree}")
        if selection not in self.SELECTIONS:
            raise ValueError(
                f"unknown selection rule {selection!r}; expected one of {self.SELECTIONS}"
            )
        self.tile_size_m = float(tile_size_m)
        self.degree = int(degree)
        self.selection = selection
        self.condition_floor = (
            self.CONDITION_FLOOR if condition_floor is None else float(condition_floor)
        )
        self._models: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self._conditioning: list[float] = []

    @property
    def n_tiles(self) -> int:
        return len(self._models)

    @property
    def conditioning_quantiles(self) -> dict[str, float]:
        """Quantiles of the plane design-matrix condition ratio over tiles."""
        if not self._conditioning:
            return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        a = np.asarray(self._conditioning)
        p10, p50, p90 = np.percentile(a, [10, 50, 90])
        return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}

    @property
    def degree_histogram(self) -> dict[int, int]:
        """Number of fitted tiles per realised degree."""
        hist = {d: 0 for d in self.N_COEFFS}
        for _, degree in self._models.values():
            hist[degree] += 1
        return hist

    def _design(self, dx: np.ndarray, dy: np.ndarray, degree: int) -> np.ndarray:
        cols = [np.ones_like(dx)]
        if degree >= 1:
            cols += [dx, dy]
        if degree >= 2:
            cols += [dx * dx, dx * dy, dy * dy]
        if degree >= 3:
            cols += [dx ** 3, dx * dx * dy, dx * dy * dy, dy ** 3]
        return np.column_stack(cols)

    def _tile_index(self, points: np.ndarray) -> np.ndarray:
        return np.floor((points - self._origin) / self.tile_size_m).astype(np.int64)

    def _local(self, points: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centre = self._origin + (key + 0.5) * self.tile_size_m
        rel = (points - centre) / self.tile_size_m
        return rel[:, 0], rel[:, 1]

    def fit(self, points: np.ndarray, values: np.ndarray) -> "TilePolynomial":
        self._points = np.asarray(points, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self._tree = cKDTree(self._points)
        self._origin = self._points.min(axis=0)

        index = self._tile_index(self._points)
        keys, inverse = np.unique(index, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(keys) + 1))

        self._models = {}
        self._conditioning = []
        for k, key in enumerate(keys):
            members = order[bounds[k]:bounds[k + 1]]
            degree = self.degree
            while degree > 0 and len(members) < self.N_COEFFS[degree]:
                degree -= 1
            if len(members) < self.N_COEFFS[degree]:
                continue
            dx, dy = self._local(self._points[members], key)
            if len(members) >= 3:
                plane = self._design(dx, dy, 1)
                sv = np.linalg.svd(plane, compute_uv=False)
                if sv[0] > 0:
                    self._conditioning.append(float(sv[-1] / sv[0]))
            fitted = self._fit_tile(dx, dy, self._values[members], degree)
            if fitted is not None:
                self._models[tuple(key)] = fitted
        return self

    def _fit_tile(self, dx, dy, values, degree):
        """Highest well-conditioned degree for this tile, or None."""
        if self.selection == "loocv":
            return self._fit_tile_loocv(dx, dy, values, degree)
        while degree >= 0:
            design = self._design(dx, dy, degree)
            coeffs, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
            well_conditioned = (
                rank == design.shape[1]
                and len(singular)
                and singular[-1] > self.condition_floor * singular[0]
            )
            if well_conditioned:
                return coeffs, degree
            degree -= 1
        return None

    def _fit_tile_loocv(self, dx, dy, values, max_degree):
        """Degree with the lowest leave-one-out error (closed form e_i / (1 - h_ii))."""
        best = (np.inf, None, None)
        for degree in range(max_degree + 1):
            design = self._design(dx, dy, degree)
            if len(values) <= design.shape[1]:
                break
            try:
                q, _ = np.linalg.qr(design)
            except np.linalg.LinAlgError:
                break
            leverage = (q ** 2).sum(axis=1)
            denominator = 1.0 - leverage
            if np.any(denominator <= 1e-12):
                continue
            coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
            error = float(np.mean(((values - design @ coeffs) / denominator) ** 2))
            if np.isfinite(error) and error < best[0]:
                best = (error, coeffs, degree)
        if best[1] is None:
            return None
        return best[1], best[2]

    def _predict_raw(self, query: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("TilePolynomial.predict called before fit")
        query = np.asarray(query, dtype=np.float64)
        index = self._tile_index(query)

        out = np.empty(len(query))
        unresolved = np.ones(len(query), dtype=bool)
        keys, inverse = np.unique(index, axis=0, return_inverse=True)
        inverse = inverse.ravel()

        for k, key in enumerate(keys):
            model = self._models.get(tuple(key))
            if model is None:
                continue  # empty tile: nearest measurement below
            coeffs, degree = model
            sel = inverse == k
            dx, dy = self._local(query[sel], key)
            out[sel] = self._design(dx, dy, degree) @ coeffs
            unresolved[sel] = False

        if unresolved.any():
            out[unresolved] = self._nearest(query[unresolved])
        return out
