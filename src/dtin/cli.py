"""Command-line interface: ``dtin info`` and ``dtin reconstruct``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dtin import __version__
from dtin.grid import GridSpec, make_grid
from dtin.io import describe_load, load_points
from dtin.methods import METHODS, summarize_model
from dtin.projection import unproject


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("input")
    g.add_argument("--input", required=True, help="CSV or Parquet point file")
    g.add_argument("--lon-col", help="longitude column (default: lon / longitude)")
    g.add_argument("--lat-col", help="latitude column (default: lat / latitude)")
    g.add_argument("--elev-col", help="elevation column (default: elev / elevation / h / z)")
    g.add_argument("--xy", action="store_true", help="columns are already projected x, y in --crs")
    g.add_argument("--x-col", help="x column with --xy (default: x / easting)")
    g.add_argument("--y-col", help="y column with --xy (default: y / northing)")
    g.add_argument("--crs", default="EPSG:3413", help="working CRS (default EPSG:3413; EPSG:3031 also built in)")
    g.add_argument("--min-elev", type=float, default=None, help="drop rows below this elevation (m)")


def _load(args, path=None, need_elev=True) -> pd.DataFrame:
    return load_points(
        path or args.input,
        lon_col=args.lon_col, lat_col=args.lat_col, elev_col=args.elev_col,
        x_col=args.x_col, y_col=args.y_col, xy=args.xy, crs=args.crs,
        min_elev=args.min_elev, need_elev=need_elev,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dtin", description="D-TIN surface reconstruction for track-sampled altimetry")
    parser.add_argument("--version", action="version", version=f"dtin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="parse a point file and report what was found")
    _add_input_args(info)

    rec = sub.add_parser("reconstruct", help="fit a surface and predict on a grid or at query points")
    _add_input_args(rec)
    where = rec.add_mutually_exclusive_group(required=True)
    where.add_argument("--grid", type=float, metavar="RES_M", help="regular grid spacing in metres")
    where.add_argument("--query", metavar="FILE", help="predict at the points of this file (same column rules)")
    rec.add_argument("--out", required=True, help="output .csv (or .tif with --grid)")
    rec.add_argument("--method", choices=list(METHODS), default="dtin",
                     help="; ".join(f"{k}: {v.help}" for k, v in METHODS.items()))
    rec.add_argument("--scales", type=float, nargs="+", default=None, help="ms-dtin support scales in metres, as in the paper: 1000 5000 10000")
    rec.add_argument("--weights", type=float, nargs="+", default=None, help="ms-dtin weights, normalised to sum 1")
    rec.add_argument("--tile-size", type=float, default=None, help="tile size in metres (tile)")
    rec.add_argument("--degree", type=int, default=None, help="maximum polynomial degree 0-3 (tile)")
    rec.add_argument("--maxlag", type=float, default=None, help="variogram fitting window in metres (kriging)")
    rec.add_argument("--support", type=float, default=None, help="cell-mean support resolution in metres (kriging); 0 = raw points")
    rec.add_argument("--variant", choices=["ordinary", "simple"], default=None, help="kriging variant")
    rec.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="grid extent in working-CRS metres")
    rec.add_argument("--fill-nearest", action="store_true", help="outside the convex hull use the nearest measurement instead of NaN")
    rec.add_argument("--force", action="store_true", help="allow more than 50 million grid cells")
    return parser


def _method_params(args) -> dict:
    info = METHODS[args.method]
    params = dict(info.params)
    given = {
        "scales": args.scales, "weights": args.weights,
        "tile_size": args.tile_size, "degree": args.degree,
        "maxlag": args.maxlag, "support": args.support, "variant": args.variant,
    }
    for key, value in given.items():
        if value is None:
            continue
        if key not in params:
            raise ValueError(f"--{key.replace('_', '-')} does not apply to --method {args.method}")
        params[key] = value
    if "scales" in params:
        params["scales"] = tuple(params["scales"])
        params["weights"] = tuple(params["weights"])
        if len(params["scales"]) != len(params["weights"]):
            raise ValueError("--scales and --weights must have the same length")
    if "support" in params and params["support"] is not None and params["support"] <= 0:
        params["support"] = None
    return params


def write_csv(path: Path, nodes: np.ndarray, elev: np.ndarray, crs: str) -> None:
    lon, lat = unproject(nodes[:, 0], nodes[:, 1], crs)
    pd.DataFrame({"x": nodes[:, 0], "y": nodes[:, 1], "lon": lon, "lat": lat, "elev": elev}).to_csv(path, index=False)


def write_geotiff(path: Path, spec: GridSpec, elev: np.ndarray, crs: str) -> None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError("GeoTIFF output needs rasterio: pip install 'dtin[geotiff]'") from None
    band = elev.reshape(spec.shape)[::-1].astype(np.float32)   # row 0 = north
    transform = from_origin(spec.xs[0] - spec.res_m / 2, spec.ys[-1] + spec.res_m / 2, spec.res_m, spec.res_m)
    with rasterio.open(
        path, "w", driver="GTiff", height=band.shape[0], width=band.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(band, 1)


def check_enough_points(pts: pd.DataFrame) -> None:
    """Raise a clear error when fewer than three points were loaded."""
    if len(pts) >= 3:
        return
    rep = pts.attrs["report"]
    raise ValueError(
        f"the surface needs at least three non-collinear points, but {len(pts)} remain "
        f"of {rep.rows_read} rows read ({rep.rows_dropped_nan} dropped as missing or "
        f"non-numeric, {rep.rows_dropped_min_elev} dropped by --min-elev)"
    )


def cmd_info(args) -> int:
    print(describe_load(_load(args)))
    return 0


def cmd_reconstruct(args) -> int:
    out = Path(args.out)
    if out.suffix.lower() in (".tif", ".tiff") and args.query:
        raise ValueError("a .tif output needs a --grid; use a .csv output with --query")
    params = _method_params(args)

    pts = _load(args)
    check_enough_points(pts)
    xy = pts[["x", "y"]].to_numpy()
    z = pts["elev"].to_numpy()
    print(f"points used: {len(pts)} ({pts.attrs['report'].crs})")

    t0 = time.perf_counter()
    model = METHODS[args.method].make(**params).fit(xy, z)
    t_fit = time.perf_counter() - t0
    print(f"method: {args.method} {params}")
    print(summarize_model(model))

    spec = None
    if args.grid is not None:
        spec = make_grid(xy, args.grid, bbox=args.bbox, force=args.force)
        nodes = spec.nodes
        print(f"grid: {spec.shape[1]} x {spec.shape[0]} nodes at {spec.res_m:g} m")
    else:
        nodes = _load(args, path=args.query, need_elev=False)[["x", "y"]].to_numpy()
        print(f"query points: {len(nodes)}")

    t1 = time.perf_counter()
    elev = model.predict(nodes, outside="nearest" if args.fill_nearest else "nan")
    t_pred = time.perf_counter() - t1
    inside = float(np.isfinite(elev).mean()) if len(elev) else 0.0
    print(f"inside hull: {100 * inside:.1f}%   fit {t_fit:.2f} s, predict {t_pred:.2f} s")
    if len(elev) and not np.isfinite(elev).any():
        print("warning: every output point is outside the convex hull of the input; all values are NaN", file=sys.stderr)

    if out.suffix.lower() in (".tif", ".tiff"):
        write_geotiff(out, spec, elev, pts.attrs["report"].crs)
    else:
        write_csv(out, nodes, elev, pts.attrs["report"].crs)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # usage errors and --help
        return int(exc.code or 0)
    try:
        if args.command == "info":
            return cmd_info(args)
        return cmd_reconstruct(args)
    except (ValueError, ImportError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
