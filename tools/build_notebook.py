"""Generate main.ipynb (the Ghub application) from the cell sources below."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# D-TIN surface reconstruction

Reconstructs a surface from a track-sampled altimetry point cloud using a
Delaunay triangulation of the measurements (D-TIN). MS-D-TIN and two
baselines, a tile polynomial and GStatSim ordinary kriging, are also available.

Input: a CSV with longitude, latitude and elevation columns, for example
ICESat GLA12 (<https://nsidc.org/data/gla12>) or ICESat-2 ATL06 from NSIDC.
Upload a small file, or enter the path of a larger one.

1. Choose the file and press *Check file*.
2. Choose a method and grid spacing and press *Reconstruct*.

Results are saved as CSV in `output/`.
"""

BOOTSTRAP = r'''import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "middleware").exists():
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

import dtin
from dtin import METHODS, describe_load, load_points, make_grid, summarize_model, unproject

HEADLESS = os.environ.get("DTIN_NB_INPUT") is not None
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets = None
    HEADLESS = True
try:
    import matplotlib
    if HEADLESS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def _writable_dir(*candidates):
    """First candidate directory that can be written to."""
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".dtin_write_test"
            probe.write_text("")
            probe.unlink()
            return cand
        except OSError:
            continue
    raise OSError("no writable output directory; set DTIN_NB_OUTPUT_DIR")


OUTPUT_DIR = _writable_dir(
    Path(os.environ.get("DTIN_NB_OUTPUT_DIR", ROOT / "output")),
    Path.home() / "dtin_output",
)
STATE = {"points": None}
RUNS = {}
print(f"dtin {dtin.__version__} from {SRC}; output -> {OUTPUT_DIR}")
'''

INPUT_CELL = r'''def load_from(path, lon="", lat="", elev="", xy=False, crs="EPSG:3413", min_elev=None):
    pts = load_points(path, lon_col=lon or None, lat_col=lat or None, elev_col=elev or None,
                      xy=xy, crs=crs, min_elev=min_elev)
    STATE["points"] = pts
    print(describe_load(pts))
    return pts


if HEADLESS:
    load_from(os.environ["DTIN_NB_INPUT"])
else:
    upload = widgets.FileUpload(accept=".csv", multiple=False, description="Upload CSV")
    path_box = widgets.Text(description="or path:", placeholder="/path/to/points.csv",
                            layout=widgets.Layout(width="60%"))
    # Blank means the default column names are tried.
    lon_box = widgets.Text(value="", placeholder="lon / longitude", description="lon col")
    lat_box = widgets.Text(value="", placeholder="lat / latitude", description="lat col")
    elev_box = widgets.Text(value="", placeholder="elev / elevation / h / z", description="elev col")
    xy_box = widgets.Checkbox(value=False, description="columns are projected x, y")
    crs_box = widgets.Text(value="EPSG:3413", description="CRS")
    min_elev_box = widgets.Text(value="", description="min elev (m)")
    check_btn = widgets.Button(description="Check file", button_style="info")
    check_out = widgets.Output()

    def _uploaded_path():
        value = upload.value
        if not value:
            return None
        if isinstance(value, dict):                     # ipywidgets 7
            name, item = next(iter(value.items()))
            content = item["content"]
        else:                                           # ipywidgets 8
            item = value[0]
            name, content = item["name"], item["content"]
        target = OUTPUT_DIR / "uploads" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(content))
        return target

    def _resolve_path():
        uploaded = _uploaded_path()
        if uploaded is not None:
            return uploaded
        if path_box.value.strip():
            return Path(path_box.value.strip()).expanduser()
        raise ValueError("upload a CSV or type a path")

    def _on_check(_):
        check_out.clear_output()
        with check_out:
            try:
                me = float(min_elev_box.value) if min_elev_box.value.strip() else None
                load_from(_resolve_path(), lon_box.value, lat_box.value, elev_box.value,
                          xy_box.value, crs_box.value, me)
            except Exception as exc:
                print("Error:", exc)

    check_btn.on_click(_on_check)
    display(widgets.VBox([
        upload, path_box,
        widgets.HBox([lon_box, lat_box, elev_box]),
        widgets.HBox([xy_box, crs_box, min_elev_box]),
        check_btn, check_out,
    ]))
'''

RUN_CELL = r'''import ast
import time


def reconstruct(method, params, res_m, bbox=None):
    pts = STATE["points"]
    if pts is None:
        raise ValueError("load a file first (step 1)")
    if len(pts) < 3:
        raise ValueError(f"the surface needs at least three non-collinear points, but only {len(pts)} were loaded (check the min-elev filter)")
    xy = pts[["x", "y"]].to_numpy()
    z = pts["elev"].to_numpy()
    t0 = time.perf_counter()
    model = METHODS[method].make(**params).fit(xy, z)
    spec = make_grid(xy, res_m, bbox=bbox)
    elev = model.predict(spec.nodes, outside="nan")
    seconds = time.perf_counter() - t0
    crs = pts.attrs["report"].crs
    lon, lat = unproject(spec.nodes[:, 0], spec.nodes[:, 1], crs)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = OUTPUT_DIR / f"{method}_{int(res_m)}m_{stamp}.csv"
    pd.DataFrame({"x": spec.nodes[:, 0], "y": spec.nodes[:, 1], "lon": lon, "lat": lat, "elev": elev}).to_csv(out, index=False)
    key = f"{METHODS[method].label} @ {int(res_m)} m ({stamp})"
    RUNS[key] = (spec, elev)
    print(summarize_model(model))
    print(f"{spec.shape[1]} x {spec.shape[0]} nodes, {100 * np.isfinite(elev).mean():.1f}% inside the hull, {seconds:.1f} s")
    print(f"wrote {out}")
    return key


def show(keys):
    if plt is None:
        return
    pts = STATE["points"]
    n = 1 + len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    ax = axes[0, 0]
    sc = ax.scatter(pts["x"] / 1000, pts["y"] / 1000, c=pts["elev"], s=1, cmap="viridis")
    ax.set_title(f"input ({len(pts)} points)"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="m")
    for ax, key in zip(axes[0, 1:], keys):
        spec, elev = RUNS[key]
        x0, y0, x1, y1 = spec.bbox
        im = ax.imshow(elev.reshape(spec.shape), origin="lower", cmap="viridis",
                       extent=(x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000), aspect="equal")
        ax.set_title(key); ax.set_xlabel("x [km]")
        fig.colorbar(im, ax=ax, label="m")
    fig.tight_layout()
    if HEADLESS:
        fig.savefig(OUTPUT_DIR / "last_run.png", dpi=110)
    else:
        plt.show()


if HEADLESS:
    method = os.environ.get("DTIN_NB_METHOD", "dtin")
    key = reconstruct(method, dict(METHODS[method].params), float(os.environ.get("DTIN_NB_GRID", "1000")))
    show([key])
else:
    method_dd = widgets.Dropdown(options=[(f"{v.label} - {v.help}", k) for k, v in METHODS.items()],
                                 value="dtin", description="Method", layout=widgets.Layout(width="90%"))
    grid_box = widgets.FloatText(value=1000.0, description="grid (m)")
    bbox_box = widgets.Text(value="", description="bbox", placeholder="xmin ymin xmax ymax (metres, optional)",
                            layout=widgets.Layout(width="60%"))
    params_box = widgets.VBox([])
    run_btn = widgets.Button(description="Reconstruct", button_style="primary")
    compare_dd = widgets.Dropdown(options=[], description="Compare with", layout=widgets.Layout(width="60%"))
    run_out = widgets.Output()

    def _rebuild_params(*_):
        params_box.children = [widgets.Text(value=repr(v), description=k) for k, v in METHODS[method_dd.value].params.items()]

    def _read_params():
        return {w.description: ast.literal_eval(w.value) for w in params_box.children}

    def _on_run(_):
        run_out.clear_output()
        with run_out:
            try:
                bbox = tuple(float(v) for v in bbox_box.value.split()) if bbox_box.value.strip() else None
                key = reconstruct(method_dd.value, _read_params(), float(grid_box.value), bbox)
                others = [compare_dd.value] if compare_dd.value else []
                compare_dd.options = [""] + list(RUNS)
                show([key] + others)
            except Exception as exc:
                print("Error:", exc)

    method_dd.observe(_rebuild_params, names="value")
    _rebuild_params()
    run_btn.on_click(_on_run)
    display(widgets.VBox([method_dd, params_box, widgets.HBox([grid_box, bbox_box]), compare_dd, run_btn, run_out]))
'''


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(BOOTSTRAP),
        nbformat.v4.new_markdown_cell("## 1. Input"),
        nbformat.v4.new_code_cell(INPUT_CELL),
        nbformat.v4.new_markdown_cell("## 2. Reconstruct"),
        nbformat.v4.new_code_cell(RUN_CELL),
    ]
    target = ROOT / "main.ipynb"
    nbformat.write(nb, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
