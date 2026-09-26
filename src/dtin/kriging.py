"""GStatSim ordinary kriging baseline. Needs the ``kriging`` extra."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree

from dtin.support import grid_mean
from dtin.tin import Surface


def require_kriging_dependencies() -> None:
    """Raise ImportError with an install hint if the kriging extra is missing."""
    missing = []
    for module in ("gstatsim", "skgstat", "sklearn"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ImportError(
            "the kriging baseline needs " + ", ".join(missing)
            + "; install them with: pip install 'dtin[kriging]'"
        )


class GridKriging(Surface):
    """Kriging with GStatSim on cell-mean support.

    Elevations are normal-score transformed, an exponential variogram is
    fitted with scikit-gstat, and each query is kriged from its ``k`` nearest
    support points within ``radius_m`` (``escalated_radius_m`` where none are
    found). ``support_m=None`` kriges on the raw points.
    """

    def __init__(
        self,
        support_m: float | None = 1000.0,
        variant: str = "ordinary",
        k: int = 100,
        radius_m: float = 100_000.0,
        escalated_radius_m: float = 200_000.0,
        seed: int = 42,
        n_quantiles: int = 500,
        variogram_maxlag_m: float = 800_000.0,
        variogram_samples: int = 5000,
        variogram_lags: int = 70,
    ) -> None:
        if variant not in ("simple", "ordinary"):
            raise ValueError(f"unknown kriging variant: {variant!r}")
        self.variant = variant
        self.support_m = None if support_m is None else float(support_m)
        self.k = k
        self.radius_m = radius_m
        self.escalated_radius_m = escalated_radius_m
        self.seed = seed
        self.n_quantiles = n_quantiles
        self.variogram_maxlag_m = variogram_maxlag_m
        self.variogram_samples = variogram_samples
        self.variogram_lags = variogram_lags

        self._frame = None
        self._nst = None
        self._vario_params: list | None = None
        self._tree: cKDTree | None = None
        self.last_sigma_normal_: np.ndarray | None = None
        self._last_predictions: np.ndarray | None = None

    def fit(self, points: np.ndarray, values: np.ndarray) -> "GridKriging":
        require_kriging_dependencies()
        import pandas as pd
        from sklearn.preprocessing import QuantileTransformer
        from skgstat import Variogram

        points = np.asarray(points, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        self._points = points
        if self.support_m is not None:
            points, values = grid_mean(points, values, self.support_m)

        frame = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "elev": values})

        self._nst = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, len(frame)),
            output_distribution="normal",
            random_state=self.seed,
        )
        frame["N_elev"] = self._nst.fit_transform(values.reshape(-1, 1))

        rng = np.random.default_rng(self.seed)
        if len(frame) > self.variogram_samples:
            sample = rng.choice(len(frame), self.variogram_samples, replace=False)
        else:
            sample = np.arange(len(frame))

        variogram = Variogram(
            points[sample],
            frame["N_elev"].to_numpy()[sample],
            n_lags=self.variogram_lags,
            maxlag=self.variogram_maxlag_m,
            normalize=False,
            model="exponential",
        )
        self._vario_params = [
            0,                        # azimuth
            variogram.parameters[2],  # nugget
            variogram.parameters[0],  # major range
            variogram.parameters[0],  # minor range
            variogram.parameters[1],  # sill
            "Exponential",
        ]

        self._frame = frame
        self._tree = cKDTree(points)
        return self

    def _krige(self, coords: np.ndarray, radius: float):
        import gstatsim as gs

        fn = gs.Interpolation.skrige if self.variant == "simple" else gs.Interpolation.okrige
        with warnings.catch_warnings():
            for message in (
                "divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul",
            ):
                warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)
            estimate, variance = fn(
                coords, self._frame, "x", "y", "N_elev", self.k, self._vario_params, radius
            )
        return np.asarray(estimate), np.asarray(variance)

    def _predict_raw(self, query: np.ndarray) -> np.ndarray:
        if self._frame is None or self._tree is None:
            raise RuntimeError("GridKriging.predict called before fit")
        query = np.asarray(query, dtype=np.float64)

        counts = self._tree.query_ball_point(query, r=self.radius_m, return_length=True)
        isolated = counts == 0

        normal_scores = np.empty(len(query))
        normal_variance = np.empty(len(query))
        if (~isolated).any():
            normal_scores[~isolated], normal_variance[~isolated] = self._krige(
                query[~isolated], self.radius_m
            )
        if isolated.any():
            normal_scores[isolated], normal_variance[isolated] = self._krige(
                query[isolated], self.escalated_radius_m
            )

        predictions = self._nst.inverse_transform(normal_scores.reshape(-1, 1)).ravel()

        self.last_sigma_normal_ = np.sqrt(np.maximum(normal_variance, 0.0))
        self._last_predictions = predictions
        return predictions

    def interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Prediction interval for the last ``predict`` call, mapped back from normal scores."""
        from scipy.stats import norm

        if self.last_sigma_normal_ is None:
            raise RuntimeError("GridKriging.interval called before predict")
        z = float(norm.ppf(1.0 - alpha / 2.0))
        centre = self._nst.transform(
            self._last_predictions.reshape(-1, 1)
        ).ravel()
        bounds = [
            self._nst.inverse_transform(
                (centre + sign * z * self.last_sigma_normal_).reshape(-1, 1)
            ).ravel()
            for sign in (-1.0, 1.0)
        ]
        return bounds[0], bounds[1]

    @property
    def n_support(self) -> int:
        """Points the variogram and the kriging system are built on."""
        return 0 if self._frame is None else len(self._frame)

    @property
    def variogram_parameters(self) -> list:
        """[azimuth, nugget, major range, minor range, sill, model] as passed to GStatSim."""
        return [] if self._vario_params is None else list(self._vario_params)
