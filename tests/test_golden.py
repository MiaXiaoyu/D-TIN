"""Predictions must match the reference implementation used in the paper."""

import json
from pathlib import Path

import numpy as np
import pytest

from dtin import DTIN, MSDTIN, GridKriging, TilePolynomial, project
from synthetic import make_tracks

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture(scope="module")
def fitted_inputs():
    df = make_tracks()
    x, y = project(df["lon"].to_numpy(), df["lat"].to_numpy())
    return np.column_stack([x, y]), df["elev"].to_numpy(), np.asarray(GOLDEN["query"])


@pytest.mark.parametrize("name,factory", [
    ("dtin", DTIN),
    ("ms-dtin", MSDTIN),
    ("tile", TilePolynomial),
    ("kriging", GridKriging),
])
def test_matches_reference_predictions(fitted_inputs, name, factory):
    pts, z, query = fitted_inputs
    pred = factory().fit(pts, z).predict(query, outside="nearest")
    # Tolerance covers rounding differences between numpy/LAPACK builds.
    np.testing.assert_allclose(pred, np.asarray(GOLDEN[name]), rtol=1e-9, atol=0)
