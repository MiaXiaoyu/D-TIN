from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_the_exact_csv_contract_and_no_raw_data_support():
    text = (ROOT / "README.md").read_text()
    normalized = " ".join(text.split())
    for token in ("`lon`", "`lat`", "`elev`", "CSV", "GLA12", "ATL06"):
        assert token in text
    assert "Parquet" not in text and "--xy" not in text and "--crs" not in text
    assert "does not read the original NASA HDF5 files directly" in normalized


def test_readme_and_cff_name_the_paper_authors_and_venue():
    combined = (ROOT / "README.md").read_text() + (ROOT / "CITATION.cff").read_text()
    for token in ("Xiaoyu Xia", "Noah Bergam", "Abani Patra", "ICDM", "SSTDM"):
        assert token in combined
