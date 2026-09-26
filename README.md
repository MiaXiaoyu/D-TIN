# dtin

`dtin` reconstructs an ice-sheet surface from track-sampled satellite
altimetry with Delaunay triangulation and barycentric interpolation. It
implements D-TIN, MS-D-TIN, and the two comparison methods used in our
accepted IEEE ICDM 2026 SSTDM Workshop paper.

## Install

```bash
pip install git+https://github.com/MiaXiaoyu/dtin
```

Kriging needs additional packages. From a cloned repository, install them
with `pip install ".[kriging]"`. The Ghub notebook uses `dtin[app]`, and
GeoTIFF output uses `dtin[geotiff]`.

## Input CSV

The input is a CSV file containing these exact columns:

| Column | Meaning | Unit |
|---|---|---|
| `lon` | WGS84 longitude | degrees |
| `lat` | WGS84 latitude | degrees |
| `elev` | surface elevation | metres |

Other columns are ignored. For example:

```csv
lon,lat,elev
-49.8124,70.1053,1842.6
-49.8079,70.1068,1843.1
-49.8035,70.1084,1844.0
-49.7990,70.1100,1844.8
-49.7946,70.1115,1845.2
```

No data is included in this repository. Public source products include
[ICESat GLAS GLA12 Release 34](https://nsidc.org/data/gla12) and
[ICESat-2 ATL06](https://nsidc.org/data/atl06); both require a free NASA
Earthdata account. This tool does not read the original NASA HDF5 files
directly. First extract longitude, latitude, and elevation into a CSV and name
the columns `lon`, `lat`, and `elev`.

The coordinates are projected internally to EPSG:3413. Use `--min-elev 5` to
apply the ocean-return filter used in the paper.

## Run

Check a file before starting a reconstruction:

```bash
dtin info --input points.csv
dtin reconstruct --input points.csv --grid 1000 --out surface.csv
```

To predict only at specified locations, provide a query CSV containing `lon`
and `lat`:

```bash
dtin reconstruct --input points.csv --query targets.csv --out predictions.csv
```

The same workflow is available from Python:

```python
from dtin import DTIN, load_points, make_grid

points = load_points("points.csv")
xy = points[["x", "y"]].to_numpy()
model = DTIN().fit(xy, points["elev"].to_numpy())
grid = make_grid(xy, res_m=1000)
surface = model.predict(grid.nodes).reshape(grid.shape)
```

`main.ipynb` provides the Ghub interface: upload a CSV, check its extent,
choose a method and grid spacing, reconstruct, then download the result.
`middleware/invoke` is the Ghub launcher.

## Methods

| Name | Definition | Paper setting |
|---|---|---|
| `dtin` | Delaunay triangulation of all measurements with linear barycentric interpolation | no fitted parameter |
| `ms-dtin` | D-TINs on 1, 5, and 10 km cell means | weights 0.6, 0.3, 0.1 |
| `tile` | Independent polynomial in each square tile | 5 km tiles, degree up to 3 |
| `kriging` | GStatSim ordinary kriging on 1 km cell means | 800 km variogram maxlag |

Kriging is substantially slower than the TIN methods. Use `--bbox` for a
small first run.

## Output

CSV output contains `x`, `y`, `lon`, `lat`, and `elev`. GeoTIFF output is a
single EPSG:3413 band. Predictions outside the input convex hull are `NaN`;
`--fill-nearest` replaces them with the nearest measurement value.

## Citation

Xiaoyu Xia, Noah Bergam, and Abani Patra, “D-TINs for Track-Sampled Ice Sheet
Surface Reconstruction from Satellite Altimetry Data,” IEEE ICDM 2026 SSTDM
Workshop.

```bibtex
@inproceedings{xia2026dtin,
  author    = {Xiaoyu Xia and Noah Bergam and Abani Patra},
  title     = {D-TINs for Track-Sampled Ice Sheet Surface Reconstruction from Satellite Altimetry Data},
  booktitle = {IEEE International Conference on Data Mining Workshops},
  year      = {2026}
}
```

DOI and page numbers will be added when the proceedings publish them.

## License

MIT
