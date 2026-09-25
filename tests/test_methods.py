import numpy as np

from dtin import DTIN, MSDTIN, GridKriging, TilePolynomial, METHODS, summarize_model


def test_method_table_has_the_four_paper_methods():
    assert list(METHODS) == ["dtin", "ms-dtin", "tile", "kriging"]
    assert METHODS["dtin"].label == "D-TIN"
    assert METHODS["ms-dtin"].params == {"scales": (1000.0, 5000.0, 10000.0), "weights": (0.6, 0.3, 0.1)}
    assert METHODS["tile"].params == {"tile_size": 5000.0, "degree": 3}
    assert METHODS["kriging"].params == {"maxlag": 800_000.0, "support": 1000.0, "variant": "ordinary"}


def test_make_builds_the_right_classes():
    assert isinstance(METHODS["dtin"].make(), DTIN)
    assert isinstance(METHODS["ms-dtin"].make(**METHODS["ms-dtin"].params), MSDTIN)
    tile = METHODS["tile"].make(tile_size=2000.0, degree=1)
    assert isinstance(tile, TilePolynomial) and tile.tile_size_m == 2000.0 and tile.degree == 1
    kr = METHODS["kriging"].make(maxlag=50_000.0, support=None, variant="simple")
    assert isinstance(kr, GridKriging) and kr.variant == "simple" and kr.support_m is None


def test_summaries_name_the_sizes():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 20_000, size=(500, 2))
    vals = pts[:, 0] * 0.01
    assert "triangles" in summarize_model(DTIN().fit(pts, vals))
    assert "per scale" in summarize_model(MSDTIN().fit(pts, vals))
    assert "tiles" in summarize_model(TilePolynomial().fit(pts, vals))
