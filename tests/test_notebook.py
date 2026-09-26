import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_runs_headless_and_writes_a_surface(synthetic_csv, tmp_path):
    env = dict(os.environ, DTIN_NB_INPUT=str(synthetic_csv), DTIN_NB_OUTPUT_DIR=str(tmp_path),
               DTIN_NB_METHOD="dtin", DTIN_NB_GRID="2000")
    executed = tmp_path / "executed.ipynb"
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=600", "--ExecutePreprocessor.kernel_name=python3",
           str(ROOT / "main.ipynb"), "--output", str(executed)]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    outputs = sorted(tmp_path.glob("dtin_2000m_*.csv"))
    assert len(outputs) == 1
    df = pd.read_csv(outputs[0])
    assert list(df.columns) == ["x", "y", "lon", "lat", "elev"] and df["elev"].notna().sum() > 0
    notebook = nbformat.read(executed, as_version=4)
    rendered = "\n".join(
        str(output.get("text", "")) + str(output.get("data", {}))
        for cell in notebook.cells for output in cell.get("outputs", [])
    )
    assert "Download result:" in rendered and outputs[0].name in rendered


def test_invoke_matches_the_alps2_template():
    text = (ROOT / "middleware" / "invoke").read_text()
    assert text.startswith("#!/bin/sh")
    assert "-t dtin" in text and "start_jupyter -A -T @tool -t main.ipynb" in text and "-u anaconda-7" in text
    assert os.access(ROOT / "middleware" / "invoke", os.X_OK)
