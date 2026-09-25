import numpy as np
import pyproj
import pytest

from dtin.projection import normalise_crs, project, unproject


def _grid(lon_range, lat_range, n=25):
    lon, lat = np.meshgrid(np.linspace(*lon_range, n), np.linspace(*lat_range, n))
    return lon.ravel(), lat.ravel()


@pytest.mark.parametrize(
    "crs,lons,lats",
    [("EPSG:3413", (-75.0, -10.0), (59.0, 84.0)),
     ("EPSG:3031", (-180.0, 180.0), (-89.9, -60.0))],
)
def test_builtin_forward_matches_pyproj(crs, lons, lats):
    lon, lat = _grid(lons, lats)
    ref = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rx, ry = ref.transform(lon, lat)
    x, y = project(lon, lat, crs)
    np.testing.assert_allclose(x, rx, atol=1e-6)
    np.testing.assert_allclose(y, ry, atol=1e-6)


@pytest.mark.parametrize("crs,lons,lats",
    [("EPSG:3413", (-75.0, -10.0), (59.0, 84.0)),
     ("EPSG:3031", (-179.0, 179.0), (-89.0, -60.0))])
def test_builtin_inverse_round_trips(crs, lons, lats):
    lon, lat = _grid(lons, lats)
    x, y = project(lon, lat, crs)
    lon2, lat2 = unproject(x, y, crs)
    np.testing.assert_allclose(lon2, lon, atol=1e-8)
    np.testing.assert_allclose(lat2, lat, atol=1e-8)


def test_longitude_in_0_360_projects_to_the_same_place():
    x1, y1 = project([315.0], [72.0])
    x2, y2 = project([-45.0], [72.0])
    np.testing.assert_allclose([x1, y1], [x2, y2], atol=1e-6)


def test_numeric_code_and_lowercase_are_accepted():
    assert normalise_crs("3413") == "EPSG:3413"
    assert normalise_crs(" epsg:3031 ") == "EPSG:3031"
    a = project([-45.0], [72.0], "3413")
    b = project([-45.0], [72.0], "EPSG:3413")
    np.testing.assert_array_equal(a, b)


def test_other_crs_goes_through_pyproj():
    ref = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    rx, ry = ref.transform([10.0], [50.0])
    x, y = project([10.0], [50.0], "EPSG:32632")
    np.testing.assert_allclose([x, y], [rx, ry], atol=1e-6)
    lon, lat = unproject(x, y, "EPSG:32632")
    np.testing.assert_allclose([lon, lat], [[10.0], [50.0]], atol=1e-8)


def test_unknown_crs_names_an_example():
    with pytest.raises(ValueError, match="EPSG:3413"):
        project([0.0], [0.0], "EPSG:999999")
