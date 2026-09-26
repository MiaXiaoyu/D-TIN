import numpy as np
import pandas as pd
import pytest
import rasterio

from dtin.cli import main


def test_info_prints_the_report(synthetic_csv, capsys):
    assert main(["info", "--input", str(synthetic_csv)]) == 0
    out = capsys.readouterr().out
    assert "points used: 4188" in out and "1 km grid would have" in out


@pytest.mark.parametrize("method,extra", [
    ("dtin", []),
    ("ms-dtin", []),
    ("tile", ["--tile-size", "5000", "--degree", "3"]),
    ("kriging", ["--maxlag", "800000", "--support", "1000"]),
])
def test_reconstruct_on_a_grid_writes_csv(synthetic_csv, tmp_path, method, extra):
    out = tmp_path / f"{method}.csv"
    res = "5000" if method == "kriging" else "1000"
    rc = main(["reconstruct", "--input", str(synthetic_csv), "--grid", res,
               "--method", method, "--out", str(out), *extra])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["x", "y", "lon", "lat", "elev"]
    assert df["elev"].notna().sum() > 0.5 * len(df)   # most nodes are inside the hull
    assert df["elev"].isna().sum() > 0                # corners of the bbox are outside


def test_fill_nearest_leaves_no_nan(synthetic_csv, tmp_path):
    out = tmp_path / "filled.csv"
    assert main(["reconstruct", "--input", str(synthetic_csv), "--grid", "2000",
                 "--out", str(out), "--fill-nearest"]) == 0
    assert pd.read_csv(out)["elev"].notna().all()


def test_query_mode_predicts_at_given_points(synthetic_csv, tmp_path):
    q = tmp_path / "q.csv"
    pd.DataFrame({"lon": [-49.5, -49.2], "lat": [70.1, 70.2]}).to_csv(q, index=False)
    out = tmp_path / "pred.csv"
    assert main(["reconstruct", "--input", str(synthetic_csv), "--query", str(q), "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert len(df) == 2 and list(df.columns) == ["x", "y", "lon", "lat", "elev"]


def test_query_entirely_outside_the_hull_warns_but_succeeds(synthetic_csv, tmp_path, capsys):
    q = tmp_path / "far.csv"
    pd.DataFrame({"lon": [0.0], "lat": [0.0]}).to_csv(q, index=False)
    out = tmp_path / "far_out.csv"
    assert main(["reconstruct", "--input", str(synthetic_csv), "--query", str(q), "--out", str(out)]) == 0
    assert "outside the convex hull" in capsys.readouterr().err
    assert pd.read_csv(out)["elev"].isna().all()


def test_geotiff_output(synthetic_csv, tmp_path):
    out = tmp_path / "surface.tif"
    assert main(["reconstruct", "--input", str(synthetic_csv), "--grid", "2000", "--out", str(out)]) == 0
    with rasterio.open(out) as src:
        assert src.count == 1 and src.crs.to_string() == "EPSG:3413"
        band = src.read(1)
        assert np.isfinite(band).sum() > 0 and np.isnan(band).sum() > 0
        assert src.transform.a == 2000.0 and src.transform.e == -2000.0


def test_geotiff_rejected_in_query_mode(synthetic_csv, tmp_path, capsys):
    q = tmp_path / "q.csv"
    pd.DataFrame({"lon": [-49.5], "lat": [70.1]}).to_csv(q, index=False)
    assert main(["reconstruct", "--input", str(synthetic_csv), "--query", str(q), "--out", str(tmp_path / "x.tif")]) == 1
    assert ".tif" in capsys.readouterr().err


def test_errors_go_to_stderr_with_exit_1(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["info", "--input", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "required columns" in err and "Traceback" not in err


def test_grid_and_query_are_exclusive(synthetic_csv, tmp_path, capsys):
    rc = main(["reconstruct", "--input", str(synthetic_csv), "--grid", "1000", "--query", str(synthetic_csv), "--out", str(tmp_path / "o.csv")])
    assert rc == 2  # argparse usage error


@pytest.mark.parametrize("flag,value", [
    ("--xy", None),
    ("--crs", "EPSG:3413"),
    ("--lon-col", "lon"),
    ("--lat-col", "lat"),
    ("--elev-col", "elev"),
])
def test_removed_input_options_are_rejected(synthetic_csv, flag, value):
    argv = ["info", "--input", str(synthetic_csv), flag]
    if value is not None:
        argv.append(value)
    assert main(argv) == 2
