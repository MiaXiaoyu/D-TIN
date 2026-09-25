"""Available methods and their default parameters, shared by the CLI and the notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dtin.kriging import GridKriging
from dtin.multiscale import MSDTIN
from dtin.tile import TilePolynomial
from dtin.tin import DTIN, Surface


@dataclass(frozen=True)
class MethodInfo:
    label: str
    make: Callable[..., Surface]
    params: dict = field(default_factory=dict)
    help: str = ""


def _ms_dtin(scales=(1000.0, 5000.0, 10000.0), weights=(0.6, 0.3, 0.1)) -> MSDTIN:
    return MSDTIN(scales=tuple(float(s) for s in scales), weights=tuple(float(w) for w in weights))


def _tile(tile_size=5000.0, degree=3) -> TilePolynomial:
    return TilePolynomial(tile_size_m=float(tile_size), degree=int(degree))


def _kriging(maxlag=800_000.0, support=1000.0, variant="ordinary") -> GridKriging:
    return GridKriging(
        variant=variant,
        support_m=None if support is None else float(support),
        variogram_maxlag_m=float(maxlag),
    )


METHODS: dict[str, MethodInfo] = {
    "dtin": MethodInfo("D-TIN", lambda: DTIN(), {},
                       "Delaunay triangulation of the measurements, linear interpolation"),
    "ms-dtin": MethodInfo("MS-D-TIN", _ms_dtin,
                          {"scales": (1000.0, 5000.0, 10000.0), "weights": (0.6, 0.3, 0.1)},
                          "D-TINs on 1, 5 and 10 km cell means, weights 0.6, 0.3, 0.1"),
    "tile": MethodInfo("Tile polynomial", _tile, {"tile_size": 5000.0, "degree": 3},
                       "baseline: independent cubic polynomial per 5 km tile"),
    "kriging": MethodInfo("GStatSim OK", _kriging,
                          {"maxlag": 800_000.0, "support": 1000.0, "variant": "ordinary"},
                          "baseline: GStatSim ordinary kriging on 1 km cell means (needs dtin[kriging])"),
}


def summarize_model(model: Surface) -> str:
    """One-line description of a fitted model."""
    if isinstance(model, DTIN):
        return f"D-TIN: {model.n_points} points, {model.n_simplices} triangles"
    if isinstance(model, MSDTIN):
        per = ", ".join(f"{int(s)} m -> {n}" for s, n in zip(model.scales, model.level_sizes))
        return f"MS-D-TIN: support per scale {per}; weights {tuple(round(float(w), 3) for w in model.weights)}"
    if isinstance(model, TilePolynomial):
        hist = model.degree_histogram
        return f"Tile polynomial: {model.n_tiles} tiles, degree histogram {hist}"
    if isinstance(model, GridKriging):
        vp = model.variogram_parameters
        vario = (f"nugget {vp[1]:.3g}, range {vp[2] / 1000:.1f} km, sill {vp[4]:.3g}" if vp else "unfitted")
        return f"GStatSim {model.variant} kriging: {model.n_support} support points, variogram {vario}"
    return type(model).__name__
