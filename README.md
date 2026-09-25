# dtin

Surface reconstruction from track-sampled satellite altimetry using a Delaunay
triangulation of the measurements (D-TIN). This is the method of

> *D-TINs for Track-Sampled Ice Sheet Surface Reconstruction from Satellite
> Altimetry Data.* SSTDM Workshop, IEEE ICDM 2026.

The package also includes the multiscale variant MS-D-TIN and the two baselines
used in the paper, a tile polynomial and GStatSim ordinary kriging.

## Install

```bash
pip install git+https://github.com/MiaXiaoyu/dtin
```

Optional extras:

| Extra | Needed for |
|---|---|
| `dtin[kriging]` | `--method kriging` |
| `dtin[geotiff]` | `.tif` output |
| `dtin[parquet]` | `.parquet` input |
| `dtin[crs]` | a CRS other than EPSG:3413 or EPSG:3031 |
| `dtin[app]` | the notebook |

## Data

No data is included. The input is a CSV or Parquet file with longitude,
latitude and elevation columns. Public sources:

- ICESat GLAS GLA12 Release 34: <https://nsidc.org/data/gla12>
- ICESat-2 ATL06: <https://nsidc.org/data/atl06>

Both require a free NASA Earthdata login.

| Column | Default name | Also accepted | Option |
|---|---|---|---|
| longitude | `lon` | `longitude` | `--lon-col` |
| latitude | `lat` | `latitude` | `--lat-col` |
| elevation | `elev` | `elevation`, `h`, `z` | `--elev-col` |

Points are projected to EPSG:3413 (Greenland) by default. Use `--crs EPSG:3031`
for Antarctica, or `--xy` if the file already has projected `x`, `y` columns.
`--min-elev 5` removes ocean returns, as in the paper.

## Usage

```bash
dtin info --input points.csv
dtin reconstruct --input points.csv --grid 1000 --out surface.csv
dtin reconstruct --input points.csv --grid 1000 --out surface.tif
dtin reconstruct --input points.csv --query targets.csv --out predictions.csv
dtin reconstruct --input points.csv --grid 1000 --method ms-dtin --out surface.csv
```

```python
from dtin import DTIN, load_points, make_grid

pts = load_points("points.csv")
xy, z = pts[["x", "y"]].to_numpy(), pts["elev"].to_numpy()
model = DTIN().fit(xy, z)
grid = make_grid(xy, res_m=1000.0)
surface = model.predict(grid.nodes).reshape(grid.shape)
```

## Methods

| `--method` | Description | Default parameters |
|---|---|---|
| `dtin` | Delaunay triangulation of all measurements, linear interpolation | none |
| `ms-dtin` | D-TINs on 1, 5 and 10 km cell means, weighted 0.6, 0.3, 0.1 | `--scales 1000 5000 10000 --weights 0.6 0.3 0.1` |
| `tile` | Independent cubic polynomial per tile | `--tile-size 5000 --degree 3` |
| `kriging` | [GStatSim](https://github.com/GatorGlaciology/GStatSim) ordinary kriging on 1 km cell means | `--maxlag 800000 --support 1000` |

Kriging is much slower than the other methods; use `--bbox` to try a small area first.

## Output

CSV output has columns `x, y, lon, lat, elev`. GeoTIFF output is a single band
in the working CRS. Points outside the convex hull of the input are `NaN`;
`--fill-nearest` uses the nearest measurement instead.

## Notebook

`main.ipynb` provides the same functions with a simple interface: load a file,
choose a method and grid spacing, and reconstruct. `middleware/invoke` launches
it as a Ghub tool.

## Citation

```bibtex
@inproceedings{dtin2026,
  title     = {D-TINs for Track-Sampled Ice Sheet Surface Reconstruction from Satellite Altimetry Data},
  booktitle = {IEEE International Conference on Data Mining Workshops (ICDMW)},
  year      = {2026}
}
```

## License

MIT
