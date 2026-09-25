import numpy as np
import pytest

from dtin.tin import DTIN, hull_mask


@pytest.fixture
def plane():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, size=(400, 2))
    vals = 3.0 + 2.0 * pts[:, 0] - 0.5 * pts[:, 1]
    return pts, vals


def test_reproduces_a_plane_exactly(plane):
    pts, vals = plane
    model = DTIN().fit(pts, vals)
    query = np.array([[400.0, 400.0], [500.0, 300.0]])
    expected = 3.0 + 2.0 * query[:, 0] - 0.5 * query[:, 1]
    np.testing.assert_allclose(model.predict(query), expected, rtol=1e-9)


def test_is_exact_at_training_points(plane):
    pts, vals = plane
    model = DTIN().fit(pts, vals)
    np.testing.assert_allclose(model.predict(pts), vals, rtol=1e-9)


def test_outside_the_hull_is_nan_by_default(plane):
    pts, vals = plane
    out = DTIN().fit(pts, vals).predict(np.array([[-5000.0, -5000.0], [500.0, 500.0]]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_outside_nearest_uses_the_nearest_measurement(plane):
    pts, vals = plane
    model = DTIN().fit(pts, vals)
    out = model.predict(np.array([[-5000.0, -5000.0]]), outside="nearest")
    nearest = np.argmin(np.hypot(pts[:, 0] + 5000.0, pts[:, 1] + 5000.0))
    assert out[0] == vals[nearest]


def test_unknown_outside_rule_is_rejected(plane):
    pts, vals = plane
    with pytest.raises(ValueError, match="outside"):
        DTIN().fit(pts, vals).predict(pts[:1], outside="zero")


def test_reports_its_sizes(plane):
    pts, vals = plane
    model = DTIN().fit(pts, vals)
    assert model.n_points == 400 and model.n_simplices > 0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        DTIN().predict(np.zeros((1, 2)))


def test_too_few_points_is_a_clear_error():
    with pytest.raises(ValueError, match="three non-collinear"):
        DTIN().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1.0, 2.0]))


def test_collinear_points_are_a_clear_error():
    pts = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 1, 20)])
    with pytest.raises(ValueError, match="three non-collinear"):
        DTIN().fit(pts, pts[:, 0])


def test_duplicate_locations_do_not_break_the_triangulation(plane):
    pts, vals = plane
    pts2 = np.vstack([pts, pts[:50]])
    vals2 = np.concatenate([vals, vals[:50]])
    out = DTIN().fit(pts2, vals2).predict(np.array([[500.0, 500.0]]))
    assert np.isfinite(out).all()


def test_hull_mask(plane):
    pts, _ = plane
    inside = hull_mask(pts, np.array([[500.0, 500.0], [-1.0, -1.0]]))
    assert inside.tolist() == [True, False]
