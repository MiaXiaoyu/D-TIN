import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # makes `synthetic` importable

from synthetic import make_tracks  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "tracks.csv"
    make_tracks().to_csv(path, index=False)
    return path
