"""Reproducible preparation of the attributed, third-party Greek gods dataset."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/katkaypettitt/greek-gods/main/greek_gods.csv"
SELECTED_NAMES = ("Athena", "Zeus", "Hermes", "Artemis", "Apollo", "Aphrodite", "Ares", "Demeter")
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "greek_gods.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "greek_gods_clean.csv"
MVP_FILE = ROOT / "data" / "processed" / "mvp_gods.csv"


def load_raw_dataset(refresh: bool = False) -> pd.DataFrame:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists() or refresh:
        try:
            data = pd.read_csv(SOURCE_URL)
            data.to_csv(RAW_FILE, index=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the third-party dataset. Check internet access or "
                f"manually save greek_gods.csv in {RAW_FILE}."
            ) from exc
    return pd.read_csv(RAW_FILE)


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    expected = {"name-english", "name-greek", "main-type", "sub-type", "description"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(expected - set(raw.columns))}")
    data = raw.copy()
    for col in expected:
        data[col] = data[col].astype("string").str.strip()
    data = data.dropna(subset=["name-english"])
    data = data[data["name-english"] != ""]
    data = data.drop_duplicates(subset=["name-english"], keep="first")
    return data.sort_values("name-english").reset_index(drop=True)


def prepare_dataset(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = clean_dataset(load_raw_dataset(refresh=refresh))
    selected = clean[clean["name-english"].isin(SELECTED_NAMES)].copy()
    missing = set(SELECTED_NAMES) - set(selected["name-english"])
    if missing:
        raise ValueError(f"Selected Greek gods missing from source dataset: {sorted(missing)}")
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_FILE, index=False)
    selected.to_csv(MVP_FILE, index=False)
    return clean, selected


if __name__ == "__main__":
    clean, selected = prepare_dataset()
    print(f"Saved {len(clean)} cleaned figures and {len(selected)} MVP gods.")
    print(f"Files: {CLEAN_FILE} and {MVP_FILE}")
