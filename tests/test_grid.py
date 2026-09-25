import numpy as np
import pytest

from dtin.grid import MAX_CELLS, GridSpec, make_grid


@pytest.fixture
def pts():
    return np.array([[100.0, 200.0], [2_600.0, 200.0], [100.0, 1_250.0], [2_600.0, 1_250.0]])


def test_nodes_start_at_the_minimum_and_step_by_res(pts):
    g = make_grid(pts, 1000.0)
    np.testing.assert_array_equal(g.xs, [100.0, 1100.0, 2100.0])
    np.testing.assert_array_equal(g.ys, [200.0, 1200.0])
    assert g.shape == (2, 3) and g.n_cells == 6
    assert g.nodes.shape == (6, 2)
    np.testing.assert_array_equal(g.nodes[0], [100.0, 200.0])
    np.testing.assert_array_equal(g.nodes[1], [1100.0, 200.0])   # x varies fastest


def test_bbox_overrides_the_extent(pts):
    g = make_grid(pts, 500.0, bbox=(0.0, 0.0, 1000.0, 1000.0))
    np.testing.assert_array_equal(g.xs, [0.0, 500.0, 1000.0])
    assert g.bbox == (0.0, 0.0, 1000.0, 1000.0)


def test_resolution_larger_than_extent_gives_one_cell(pts):
    g = make_grid(pts, 10_000.0)
    assert g.shape == (1, 1)


def test_non_positive_resolution_is_an_error(pts):
    with pytest.raises(ValueError, match="positive"):
        make_grid(pts, 0.0)
    with pytest.raises(ValueError, match="positive"):
        make_grid(pts, -5.0)


def test_inverted_bbox_is_an_error(pts):
    with pytest.raises(ValueError, match="bbox"):
        make_grid(pts, 100.0, bbox=(10.0, 0.0, 0.0, 10.0))


def test_cell_guard(pts, monkeypatch):
    big = np.array([[0.0, 0.0], [1e7, 1e7]])
    with pytest.raises(ValueError, match="--force"):
        make_grid(big, 1.0)                      # 1e14 cells against the real limit
    monkeypatch.setattr("dtin.grid.MAX_CELLS", 4)
    with pytest.raises(ValueError, match="--force"):
        make_grid(pts, 1000.0)                   # 6 cells against a limit of 4
    assert make_grid(pts, 1000.0, force=True).n_cells == 6


def test_gridspec_is_frozen(pts):
    g = make_grid(pts, 1000.0)
    with pytest.raises(Exception):
        g.res_m = 5.0
