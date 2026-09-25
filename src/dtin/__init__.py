"""dtin: Delaunay TIN surface reconstruction for track-sampled altimetry."""

from dtin.grid import GridSpec, make_grid
from dtin.io import LoadReport, describe_load, load_points
from dtin.kriging import GridKriging
from dtin.methods import METHODS, MethodInfo, summarize_model
from dtin.multiscale import MSDTIN
from dtin.projection import project, unproject
from dtin.tile import TilePolynomial
from dtin.tin import DTIN, hull_mask

__version__ = "0.1.0"

__all__ = [
    "DTIN", "MSDTIN", "TilePolynomial", "GridKriging",
    "METHODS", "MethodInfo", "summarize_model",
    "load_points", "LoadReport", "describe_load",
    "make_grid", "GridSpec", "hull_mask",
    "project", "unproject",
    "__version__",
]
