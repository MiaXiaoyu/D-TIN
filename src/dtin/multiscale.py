"""MS-D-TIN: D-TINs on 1, 5 and 10 km cell-mean support, combined with fixed weights."""

from __future__ import annotations

import numpy as np

from dtin.support import grid_mean
from dtin.tin import DTIN, Surface


class MSDTIN(Surface):
    """Weighted sum of D-TINs built on cell means at several scales."""

    def __init__(
        self,
        scales: tuple[float, ...] = (1_000.0, 5_000.0, 10_000.0),
        weights: tuple[float, ...] = (0.6, 0.3, 0.1),
    ) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(scales):
            raise ValueError("weights and scales must have equal length")
        self.scales = tuple(float(s) for s in scales)
        self.weights = w / w.sum()
        self.models: list = []

    def fit(self, points: np.ndarray, values: np.ndarray) -> "MSDTIN":
        points = np.asarray(points, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        self._points = points
        self.models = []
        for scale in self.scales:
            level_points, level_values = grid_mean(points, values, scale)
            self.models.append(DTIN().fit(level_points, level_values))
        return self

    @property
    def n_points(self) -> int:
        """Total support points across all scales."""
        return sum(m.n_points for m in self.models)

    def component_predictions(self, query: np.ndarray) -> list[np.ndarray]:
        """Prediction of each scale's D-TIN."""
        if not self.models:
            raise RuntimeError("MSDTIN.predict called before fit")
        query = np.asarray(query, dtype=np.float64)
        return [model.predict(query, outside="nearest") for model in self.models]

    def _predict_raw(self, query: np.ndarray) -> np.ndarray:
        components = self.component_predictions(query)
        return sum(w * p for w, p in zip(self.weights, components))

    @property
    def level_sizes(self) -> list[int]:
        """Support size per scale, finest first."""
        return [m.n_points for m in self.models]
