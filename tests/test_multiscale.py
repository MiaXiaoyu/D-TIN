import numpy as np
import pytest

from dtin.multiscale import MSDTIN


@pytest.fixture
def cloud():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 40_000, size=(3000, 2))
    vals = 1000.0 + 0.01 * pts[:, 0] + 20.0 * np.sin(pts[:, 1] / 3000.0)
    return pts, vals


def test_defaults_are_the_papers(cloud):
    m = MSDTIN()
    assert m.scales == (1000.0, 5000.0, 10000.0)
    np.testing.assert_allclose(m.weights, [0.6, 0.3, 0.1])


def test_weights_are_normalised():
    np.testing.assert_allclose(MSDTIN(weights=(6, 3, 1)).weights, [0.6, 0.3, 0.1])


def test_rejects_mismatched_weights_and_scales():
    with pytest.raises(ValueError, match="equal length"):
        MSDTIN(scales=(1000.0, 5000.0), weights=(0.5, 0.3, 0.2))


def test_level_sizes_shrink_with_scale(cloud):
    pts, vals = cloud
    m = MSDTIN().fit(pts, vals)
    assert m.level_sizes[0] > m.level_sizes[1] > m.level_sizes[2]
    assert m.n_points == sum(m.level_sizes)


def test_component_predictions_match_scale_count(cloud):
    pts, vals = cloud
    comps = MSDTIN().fit(pts, vals).component_predictions(pts[:10])
    assert len(comps) == 3 and all(c.shape == (10,) for c in comps)


def test_constant_field_is_reproduced(cloud):
    pts, _ = cloud
    out = MSDTIN().fit(pts, np.full(len(pts), 7.5)).predict(pts[:20])
    np.testing.assert_allclose(out, 7.5, rtol=1e-12)


def test_prediction_is_the_weighted_sum(cloud):
    pts, vals = cloud
    m = MSDTIN().fit(pts, vals)
    q = pts[:25]
    comps = m.component_predictions(q)
    expected = sum(w * c for w, c in zip(m.weights, comps))
    np.testing.assert_allclose(m.predict(q), expected, rtol=1e-12)


def test_outside_hull_is_nan_by_default(cloud):
    pts, vals = cloud
    out = MSDTIN().fit(pts, vals).predict(np.array([[-1e5, -1e5], [20_000.0, 20_000.0]]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        MSDTIN().predict(np.zeros((1, 2)))
