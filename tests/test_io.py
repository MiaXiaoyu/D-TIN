import numpy as np
import pandas as pd
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


def test_aliases_case_and_bom_and_spaces(tmp_path):
    p = _write(tmp_path, "b.csv", "\ufeffLongitude ,LATITUDE,Elevation\n-45,72,1500\n-44,72.5,1600\n")
    df = load_points(p)
    assert df.attrs["report"].columns == {"lon": "Longitude", "lat": "LATITUDE", "elev": "Elevation"}
    assert len(df) == 2


def test_explicit_column_names(tmp_path):
    p = _write(tmp_path, "c.csv", "a,b,c\n-45,72,1500\n-44,72.5,1600\n")
    df = load_points(p, lon_col="a", lat_col="b", elev_col="c")
    assert len(df) == 2


def test_missing_column_lists_what_is_there(tmp_path):
    p = _write(tmp_path, "d.csv", "a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match=r"lon.*\['a', 'b', 'c'\]"):
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


def test_xy_input_round_trips_to_lon_lat(tmp_path):
    x, y = project([-45.0, -44.0], [72.0, 72.5])
    p = _write(tmp_path, "g.csv", "x,y,elev\n" + "\n".join(f"{a},{b},1500" for a, b in zip(x, y)) + "\n")
    df = load_points(p, xy=True)
    np.testing.assert_allclose(df["lon"], [-45.0, -44.0], atol=1e-8)
    np.testing.assert_allclose(df["lat"], [72.0, 72.5], atol=1e-8)


def test_query_file_without_elevation(tmp_path):
    p = _write(tmp_path, "h.csv", "lon,lat\n-45,72\n")
    df = load_points(p, need_elev=False)
    assert list(df.columns) == ["x", "y", "lon", "lat"]


def test_parquet_input(tmp_path):
    p = tmp_path / "i.parquet"
    pd.DataFrame({"lon": [-45.0, -44.0], "lat": [72.0, 72.5], "elev": [1.0, 2.0]}).to_parquet(p)
    assert len(load_points(p)) == 2


def test_unsupported_extension(tmp_path):
    p = _write(tmp_path, "j.xyz", "lon,lat,elev\n")
    with pytest.raises(ValueError, match=r"\.csv or \.parquet"):
        load_points(p)


def test_describe_load_mentions_counts_and_extent(tmp_path):
    p = _write(tmp_path, "k.csv", "lon,lat,elev\n-45,72,1500\n-44,,1600\n-44,72.5,1600\n")
    text = describe_load(load_points(p))
    assert "rows read: 3" in text and "dropped (NaN): 1" in text and "1 km grid" in text
