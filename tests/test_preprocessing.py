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


def test_split_preserves_stratification():
    """Train/test split should maintain class proportions."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "F1": np.random.randn(n),
        "F2": np.random.randn(n),
        "Target": ["Dropout"] * 30 + ["Graduate"] * 70,
    })
    pipeline = PreprocessingPipeline(test_size=0.2, random_state=42)
    X, y = pipeline._encode_target(df)
    X_train, X_test, y_train, y_test = pipeline._split(X.values, y)

    assert len(X_train) == 80
    assert len(X_test) == 20
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - 0.3) < 0.05
    assert abs(test_ratio - 0.3) < 0.1


def test_standardize_fits_on_train_only():
    """Scaler should be fit on train data only. Both sets should be transformed."""
    X_train = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    X_test = np.array([[5.0, 500.0], [6.0, 600.0]])

    pipeline = PreprocessingPipeline()
    X_train_s, X_test_s = pipeline._standardize(X_train, X_test)

    assert abs(X_train_s[:, 0].mean()) < 1e-10
    assert abs(X_train_s[:, 0].std(ddof=0) - 1.0) < 1e-10
    # Test should NOT have mean 0 (transformed with train stats)
    assert abs(X_test_s[:, 0].mean()) > 0.5
    assert pipeline.scaler is not None
