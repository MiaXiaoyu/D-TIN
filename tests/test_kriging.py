import sys

import numpy as np
import pytest

from dtin.kriging import GridKriging, require_kriging_dependencies


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(4)
    # twelve north-south tracks 5 km apart, 172 m along track, over 60 x 60 km
    xs = 5_000.0 * np.arange(12) + 2_500.0
    ys = np.arange(0.0, 60_000.0, 172.0)
    pts = np.array([(x + rng.normal(0, 30), y) for x in xs for y in ys])
    vals = 1500.0 + 0.01 * pts[:, 0] + 0.02 * pts[:, 1] + 30.0 * np.sin(pts[:, 0] / 8000.0)
    return pts, vals


def test_defaults_are_the_papers():
    m = GridKriging()
    assert m.variant == "ordinary" and m.support_m == 1000.0
    assert m.variogram_maxlag_m == 800_000.0 and m.k == 100 and m.radius_m == 100_000.0


def test_fits_on_cell_mean_support_and_predicts_a_smooth_field(field):
    pts, vals = field
    model = GridKriging().fit(pts, vals)
    assert 0 < model.n_support < len(pts)
    assert len(model.variogram_parameters) == 6
    rng = np.random.default_rng(5)
    q = np.column_stack([rng.uniform(5_000, 55_000, 60), rng.uniform(5_000, 55_000, 60)])
    truth = 1500.0 + 0.01 * q[:, 0] + 0.02 * q[:, 1] + 30.0 * np.sin(q[:, 0] / 8000.0)
    pred = model.predict(q)
    assert np.isfinite(pred).all()
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    assert rmse < 0.15 * float(np.std(vals))


def test_interval_brackets_the_prediction(field):
    pts, vals = field
    model = GridKriging().fit(pts, vals)
    q = np.array([[30_000.0, 30_000.0], [12_000.0, 40_000.0]])
    pred = model.predict(q)
    lo, hi = model.interval(alpha=0.1)
    assert (lo <= pred).all() and (pred <= hi).all()


def test_outside_hull_is_nan_by_default(field):
    pts, vals = field
    out = GridKriging().fit(pts, vals).predict(np.array([[-50_000.0, -50_000.0]]))
    assert np.isnan(out[0])


def test_no_support_discretisation_when_support_m_is_none(field):
    pts, vals = field
    model = GridKriging(support_m=None).fit(pts, vals)
    assert model.n_support == len(pts)


def test_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        GridKriging(variant="universal")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        GridKriging().predict(np.zeros((1, 2)))


def test_missing_extra_gives_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gstatsim", None)
    with pytest.raises(ImportError, match=r"dtin\[kriging\]"):
        require_kriging_dependencies()
