import numpy as np
import pytest

from dtin.io import LoadReport, describe_load, load_points
from dtin.projection import project


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_default_columns_are_projected(tmp_path):
    p = _write(tmp_path, "a.csv", "lon,lat,elev\n-45.0,72.0,1500.0\n-44.0,72.5,1600.0\n")
    df = load_points(p)
    assert list(df.columns) == ["x", "y", "lon", "lat", "elev"]
    x, y = project([-45.0, -44.0], [72.0, 72.5])
    np.testing.assert_allclose(df["x"], x)
    np.testing.assert_allclose(df["y"], y)
    rep = df.attrs["report"]
    assert isinstance(rep, LoadReport)
    assert rep.rows_read == 2 and rep.rows_used == 2 and rep.crs == "EPSG:3413"


@pytest.mark.parametrize("header", [
    "longitude,lat,elev",
    "lon,latitude,elev",
    "lon,lat,elevation",
])
def test_training_file_requires_exact_column_names(tmp_path, header):
    p = _write(tmp_path, "bad.csv", header + "\n-45,72,1500\n")
    with pytest.raises(ValueError, match=r"required columns.*lon.*lat.*elev.*found"):
        load_points(p)


def test_missing_column_lists_what_is_there(tmp_path):
    p = _write(tmp_path, "d.csv", "a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match=r"required columns.*lon.*lat.*elev.*found.*a.*b.*c"):
        load_points(p)


def test_nan_and_text_rows_are_dropped_and_counted(tmp_path):
    p = _write(tmp_path, "e.csv", "lon,lat,elev\n-45,72,1500\n-44,,1600\n-43,72,oops\n-42,72,1700\n")
    df = load_points(p)
    assert len(df) == 2
    assert df.attrs["report"].rows_dropped_nan == 2


def test_min_elev_filter(tmp_path):
    p = _write(tmp_path, "f.csv", "lon,lat,elev\n-45,72,1.0\n-44,72.5,1600\n")
    df = load_points(p, min_elev=5.0)
    assert len(df) == 1 and df.attrs["report"].rows_dropped_min_elev == 1


def test_query_requires_lon_and_lat_but_not_elev(tmp_path):
    p = _write(tmp_path, "h.csv", "lon,lat\n-45,72\n")
    df = load_points(p, need_elev=False)
    assert list(df.columns) == ["x", "y", "lon", "lat"]


def test_extra_columns_are_ignored(tmp_path):
    p = _write(tmp_path, "extra.csv", "time,lon,lat,elev,quality\n1,-45,72,1500,0\n")
    df = load_points(p)
    assert list(df.columns) == ["x", "y", "lon", "lat", "elev"]


@pytest.mark.parametrize("name", ["points.parquet", "points.txt", "points.h5"])
def test_only_csv_is_accepted(tmp_path, name):
    p = _write(tmp_path, name, "lon,lat,elev\n-45,72,1500\n")
    with pytest.raises(ValueError, match="CSV"):
        load_points(p)


def test_describe_load_mentions_counts_and_extent(tmp_path):
    p = _write(tmp_path, "k.csv", "lon,lat,elev\n-45,72,1500\n-44,,1600\n-44,72.5,1600\n")
    text = describe_load(load_points(p))
    assert "rows read: 3" in text and "dropped (NaN): 1" in text and "1 km grid" in text
