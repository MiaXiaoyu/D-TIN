import numpy as np
import pytest

from dtin.support import cell_index, grid_mean


@pytest.fixture
def cloud():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10_000, size=(2000, 2))
    vals = pts[:, 0] * 0.1 + pts[:, 1] * 0.05
    return pts, vals


def test_grid_mean_reduces_point_count(cloud):
    pts, vals = cloud
    p, _ = grid_mean(pts, vals, 1000.0)
    assert len(p) < len(pts)


def test_grid_mean_values_are_cell_means():
    pts = np.array([[10.0, 10.0], [20.0, 20.0], [1500.0, 1500.0]])
    vals = np.array([100.0, 200.0, 999.0])
    _, v = grid_mean(pts, vals, 1000.0)
    assert 150.0 in set(np.round(v, 6))


def test_cells_round_to_the_nearest_node_like_gstatsim():
    # rint, not floor: 0.6 res from the origin belongs to node 1, 0.4 to node 0
    pts = np.array([[0.0, 0.0], [600.0, 0.0], [400.0, 0.0]])
    idx, nodes, origin = cell_index(pts, 1000.0)
    assert idx[:, 0].tolist() == [0, 1, 0]
    np.testing.assert_array_equal(origin, [0.0, 0.0])
    np.testing.assert_array_equal(nodes[1], [1000.0, 0.0])


def test_nodes_lie_on_the_lattice_anchored_at_the_minimum(cloud):
    pts, vals = cloud
    nodes, _ = grid_mean(pts, vals, 1000.0)
    rel = (nodes - pts.min(axis=0)) / 1000.0
    np.testing.assert_allclose(rel, np.round(rel), atol=1e-9)


def test_coarser_resolution_yields_fewer_points(cloud):
    pts, vals = cloud
    assert len(grid_mean(pts, vals, 2000.0)[0]) < len(grid_mean(pts, vals, 1000.0)[0])
