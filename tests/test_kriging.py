import sys
import warnings

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


def test_prediction_does_not_emit_known_gstatsim_matmul_warnings(field):
    pts, vals = field
    model = GridKriging().fit(pts, vals)
    q = np.column_stack([
        np.linspace(5_000.0, 55_000.0, 60),
        np.linspace(55_000.0, 5_000.0, 60),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(q, outside="nearest")
    messages = [str(item.message) for item in caught]
    assert not any("encountered in matmul" in message for message in messages)


def test_outside_hull_is_nan_by_default(field):
    pts, vals = field
    out = GridKriging().fit(pts, vals).predict(np.array([[-50_000.0, -50_000.0]]))
    assert np.isnan(out[0])


def test_far_outside_hull_is_nan_without_neighbor_search_failure(field):
    pts, vals = field
    out = GridKriging().fit(pts, vals).predict(np.array([[-500_000.0, -500_000.0]]))
    assert np.isnan(out[0])


def test_unreachable_query_is_nan_and_ordinary_queries_still_krige(field):
    # A query more than escalated_radius_m from every support point cannot be
    # kriged by GStatSim; it gets NaN rather than an invented value, while
    # ordinary queries in the same call are kriged as usual.
    pts, vals = field
    model = GridKriging().fit(pts, vals)
    far = np.array([[-500_000.0, -500_000.0]])
    near = np.array([[30_000.0, 30_000.0]])
    out = model.predict(np.vstack([far, near]), outside="nearest")
    assert np.isnan(out[0]) and np.isfinite(out[1])
    lo, hi = model.interval(0.1)
    assert np.isnan(lo[0]) and np.isnan(hi[0]) and lo[1] < out[1] < hi[1]


def test_support_point_exactly_on_the_search_radius_does_not_break_the_search(field):
    # Cell-mean support and output grids are both anchored at the data minimum,
    # so a query can sit at exactly radius_m from its only neighbour. GStatSim
    # keeps neighbours strictly inside the radius; the neighbour census must
    # agree with it or GStatSim raises "Unable to find nearest neighbors".
    pts, vals = field
    pts = np.round(pts)  # integer coordinates make the lattice distances exact
    model = GridKriging().fit(pts, vals)
    support = model._frame
    row = support["x"].idxmax()
    sx, sy = float(support.at[row, "x"]), float(support.at[row, "y"])
    q = np.array([[sx + model.radius_m, sy], [sx + model.escalated_radius_m, sy]])
    d, _ = model._tree.query(q, k=1)
    assert d[0] == model.radius_m and d[1] == model.escalated_radius_m  # precondition
    out = model.predict(q, outside="nearest")  # must not raise
    assert np.isfinite(out[0])   # escalated to escalated_radius_m and kriged
    assert np.isnan(out[1])      # nothing strictly inside escalated_radius_m


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
