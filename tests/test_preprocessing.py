import pytest
import pandas as pd
import numpy as np
from src.preprocessing import PreprocessingPipeline


def test_encode_target_maps_dropout_to_1():
    """Dropout should map to 1, Enrolled and Graduate should map to 0."""
    df = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Target": ["Dropout", "Graduate", "Enrolled", "Dropout", "Graduate"],
    })
    pipeline = PreprocessingPipeline()
    X, y = pipeline._encode_target(df)

    assert list(y) == [1, 0, 0, 1, 0]
    assert "Target" not in X.columns
    assert X.shape == (5, 2)


def test_encode_target_drops_unknown_classes():
    """Only Dropout, Graduate, Enrolled are valid target values."""
    df = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0],
        "Target": ["Dropout", "Graduate", "Unknown"],
    })
    pipeline = PreprocessingPipeline()
    X, y = pipeline._encode_target(df)

    assert len(y) == 2
    assert list(y) == [1, 0]
