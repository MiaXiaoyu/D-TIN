import numpy as np
import pyproj

from dtin.projection import project, unproject


def _grid(lon_range, lat_range, n=25):
    lon, lat = np.meshgrid(np.linspace(*lon_range, n), np.linspace(*lat_range, n))
    return lon.ravel(), lat.ravel()


def test_forward_matches_pyproj():
    lon, lat = _grid((-75.0, -10.0), (59.0, 84.0))
    ref = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    rx, ry = ref.transform(lon, lat)
    x, y = project(lon, lat)
    np.testing.assert_allclose(x, rx, atol=1e-6)
    np.testing.assert_allclose(y, ry, atol=1e-6)


def test_inverse_round_trips():
    lon, lat = _grid((-75.0, -10.0), (59.0, 84.0))
    x, y = project(lon, lat)
    lon2, lat2 = unproject(x, y)
    np.testing.assert_allclose(lon2, lon, atol=1e-8)
    np.testing.assert_allclose(lat2, lat, atol=1e-8)


def test_longitude_in_0_360_projects_to_the_same_place():
    x1, y1 = project([315.0], [72.0])
    x2, y2 = project([-45.0], [72.0])
    np.testing.assert_allclose([x1, y1], [x2, y2], atol=1e-6)
