import numpy as np
import pytest

from dtin.tile import TilePolynomial


def _cubic(p):
    u = (p[:, 0] - 2500.0) / 5000.0
    v = (p[:, 1] - 2500.0) / 5000.0
    return 1000.0 + 20.0 * u + 30.0 * v + 5.0 * u * v + 4.0 * u ** 3 - 2.0 * v ** 2 * u


def test_defaults_are_the_papers():
    m = TilePolynomial()
    assert m.tile_size_m == 5000.0 and m.degree == 3 and m.condition_floor == 1e-2


def test_reproduces_a_cubic_inside_one_tile():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 4999.0, size=(500, 2))
    model = TilePolynomial().fit(pts, _cubic(pts))
    q = rng.uniform(500, 4500, size=(50, 2))
    np.testing.assert_allclose(model.predict(q), _cubic(q), rtol=1e-6)
    assert model.n_tiles == 1 and model.degree_histogram[3] == 1


def test_one_track_per_tile_falls_back_to_a_low_degree():
    # 200 points on one straight line across the tile: rank-full by count,
    # unconstrained across track -> the condition rule refuses the cubic.
    t = np.linspace(0, 4999.0, 200)
    pts = np.column_stack([t, 2500.0 + 1e-3 * t])
    model = TilePolynomial().fit(pts, 100.0 + 0.01 * t)
    hist = model.degree_histogram
    assert hist[3] == 0 and hist[2] == 0 and sum(hist.values()) == 1


def test_empty_tile_uses_nearest_with_outside_nearest():
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.uniform(0, 4999.0, size=(300, 2)),
                     rng.uniform(10_001, 14_999, size=(300, 2))])
    vals = np.where(pts[:, 0] < 5000, 1.0, 2.0)
    model = TilePolynomial().fit(pts, vals)
    q = np.array([[7500.0, 7500.0]])          # inside the hull, in an empty tile
    out = model.predict(q, outside="nearest")
    assert out[0] in (1.0, 2.0)
    assert np.isfinite(model.predict(q)).all()  # inside the hull -> not NaN


def test_outside_hull_is_nan_by_default():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 4999.0, size=(300, 2))
    out = TilePolynomial().fit(pts, _cubic(pts)).predict(np.array([[-1e4, -1e4]]))
    assert np.isnan(out[0])


def test_rejects_bad_degree_and_selection():
    with pytest.raises(ValueError, match="degree"):
        TilePolynomial(degree=4)
    with pytest.raises(ValueError, match="selection"):
        TilePolynomial(selection="magic")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        TilePolynomial().predict(np.zeros((1, 2)))
