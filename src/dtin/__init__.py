"""dtin: Delaunay TIN surface reconstruction for track-sampled altimetry."""

from dtin.grid import GridSpec, make_grid
from dtin.io import LoadReport, describe_load, load_points
from dtin.kriging import GridKriging
from dtin.methods import METHODS, MethodInfo, prediction_bounds, summarize_model
from dtin.multiscale import MSDTIN
from dtin.projection import project, unproject
from dtin.support import nearest_distance
from dtin.tin import DTIN, hull_mask

__version__ = "0.2.0"

__all__ = [
    "DTIN", "MSDTIN", "GridKriging",
    "METHODS", "MethodInfo", "summarize_model", "prediction_bounds", "nearest_distance",
    "load_points", "LoadReport", "describe_load",
    "make_grid", "GridSpec", "hull_mask",
    "project", "unproject",
    "__version__",
]
