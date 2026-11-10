import pandas as pd
import pytest
from src.data_processing import SELECTED_NAMES, clean_dataset


def test_dataset_cleaning():
    df = pd.DataFrame({
        "name-english": [" Athena ", "Athena", "Zeus", None],
        "name-greek": ["a", "a", "z", "n"],
        "main-type": ["god"] * 4,
        "sub-type": ["olympian"] * 4,
        "description": [" Wisdom ", "duplicate", " Sky ", "something"],
    })
    cleaned = clean_dataset(df)
    assert list(cleaned["name-english"]) == ["Athena", "Zeus"]
    assert cleaned.loc[0, "description"] == "Wisdom"


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        clean_dataset(pd.DataFrame({"name-english": ["Athena"]}))


def test_mvp_has_eight_names():
    assert len(SELECTED_NAMES) == 8
