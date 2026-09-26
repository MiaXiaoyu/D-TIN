"""Notebook behaviour, hull caching and too-few-points errors."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import dtin.tin as tin_mod
from dtin import MSDTIN
from dtin.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _cells():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_interactive_cells(monkeypatch, tmp_path, output_dir=None):
    """Execute the notebook's bootstrap and input cells in widget mode, in-process."""
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("DTIN_NB_INPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if output_dir is None:
        monkeypatch.setenv("DTIN_NB_OUTPUT_DIR", str(tmp_path / "out"))
    else:
        monkeypatch.setenv("DTIN_NB_OUTPUT_DIR", str(output_dir))
    cells = _cells()
    ns: dict = {}
    exec(cells.BOOTSTRAP, ns)
    exec(cells.INPUT_CELL, ns)
    return ns


def test_notebook_exposes_only_the_csv_contract(monkeypatch, tmp_path):
    ns = _run_interactive_cells(monkeypatch, tmp_path)
    assert ns["HEADLESS"] is False
    assert ns["upload"].accept == ".csv"
    assert all(name in ns["FORMAT_NOTE"] for name in ("lon", "lat", "elev"))
    assert "lon_box" not in ns and "xy_box" not in ns and "crs_box" not in ns


def test_uploaded_filename_is_reduced_to_basename(monkeypatch, tmp_path):
    ns = _run_interactive_cells(monkeypatch, tmp_path)
    ns["_upload_value"] = lambda: ("../outside.csv", b"lon,lat,elev\n-45,72,1\n")
    target = ns["_uploaded_path"]()
    assert target.parent == ns["OUTPUT_DIR"] / "uploads"
    assert target.name == "outside.csv"


def test_notebook_falls_back_when_output_dir_is_not_writable(monkeypatch, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        ns = _run_interactive_cells(monkeypatch, tmp_path, output_dir=locked / "output")
        assert ns["OUTPUT_DIR"] == tmp_path / "home" / "dtin_output"
        assert ns["OUTPUT_DIR"].is_dir()
    finally:
        locked.chmod(0o700)


def test_hull_test_does_not_retriangulate_all_points(monkeypatch):
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 40_000, size=(3000, 2))
    model = MSDTIN().fit(pts, pts[:, 0] * 0.01)
    sizes = []
    real = tin_mod.Delaunay

    def counting(points, *a, **k):
        sizes.append(len(points))
        return real(points, *a, **k)

    monkeypatch.setattr(tin_mod, "Delaunay", counting)
    q = rng.uniform(0, 40_000, size=(50, 2))
    model.predict(q)
    model.predict(q)
    assert all(n < 200 for n in sizes), sizes  # hull vertices only
    assert len(sizes) <= 1                      # cached after the first call


def test_cached_hull_matches_the_full_hull():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, size=(2000, 2))
    q = rng.uniform(-200, 1200, size=(5000, 2))
    model = MSDTIN(scales=(100.0, 200.0, 400.0)).fit(pts, pts[:, 0])
    expected = tin_mod.hull_mask(pts, q)
    np.testing.assert_array_equal(np.isfinite(model.predict(q)), expected)


@pytest.mark.parametrize("method", ["dtin", "ms-dtin", "tile", "kriging"])
def test_too_few_points_after_filtering_is_readable(synthetic_csv, tmp_path, capsys, method):
    rc = main(["reconstruct", "--input", str(synthetic_csv), "--grid", "5000", "--method", method,
               "--min-elev", "1e9", "--out", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "three" in err and "min-elev" in err and "zero-size" not in err
