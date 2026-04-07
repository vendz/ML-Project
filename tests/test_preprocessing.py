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


def test_class_weight_computation():
    """Class weights should follow N / (n_classes * n_k) formula."""
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # 7 vs 3
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")

    weights = pipeline._compute_class_weights(y_train)

    expected_w0 = 10 / (2 * 7)  # ~0.714
    expected_w1 = 10 / (2 * 3)  # ~1.667
    assert abs(weights[0] - expected_w0) < 1e-3
    assert abs(weights[1] - expected_w1) < 1e-3


def _make_imbalanced_data():
    """Helper: 80 class-0, 20 class-1."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.array([0] * 80 + [1] * 20)
    return X, y


def test_handle_imbalance_none_passes_through():
    X, y = _make_imbalanced_data()
    pipeline = PreprocessingPipeline(imbalance_strategy="none")
    X_out, y_out = pipeline._handle_imbalance(X, y)
    assert X_out.shape == X.shape
    assert list(y_out) == list(y)


def test_handle_imbalance_smote_balances_classes():
    X, y = _make_imbalanced_data()
    pipeline = PreprocessingPipeline(imbalance_strategy="smote")
    X_out, y_out = pipeline._handle_imbalance(X, y)
    assert (y_out == 0).sum() == (y_out == 1).sum()
    assert len(y_out) > len(y)


def test_handle_imbalance_undersample_balances_classes():
    X, y = _make_imbalanced_data()
    pipeline = PreprocessingPipeline(imbalance_strategy="undersample")
    X_out, y_out = pipeline._handle_imbalance(X, y)
    assert (y_out == 0).sum() == (y_out == 1).sum()
    assert len(y_out) < len(y)


def test_handle_imbalance_class_weight_passes_through_and_stores_weights():
    X, y = _make_imbalanced_data()
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")
    X_out, y_out = pipeline._handle_imbalance(X, y)
    assert X_out.shape == X.shape
    assert pipeline.class_weights is not None
    assert pipeline.class_weights[1] > pipeline.class_weights[0]


def test_pca_reduces_dimensions():
    """PCA should reduce feature count while retaining variance threshold."""
    np.random.seed(42)
    base = np.random.randn(100, 3)
    noise = np.random.randn(100, 7) * 0.01
    X_train = np.hstack([base, noise])
    X_test = np.hstack([np.random.randn(20, 3), np.random.randn(20, 7) * 0.01])

    pipeline = PreprocessingPipeline(use_pca=True, pca_variance=0.95)
    X_train_pca, X_test_pca = pipeline._apply_pca(X_train, X_test)

    assert X_train_pca.shape[1] < 10
    assert X_test_pca.shape[1] == X_train_pca.shape[1]
    assert pipeline.pca_transformer is not None


def test_pca_disabled_passes_through():
    """When use_pca=False, data should pass through unchanged."""
    X_train = np.random.randn(50, 5)
    X_test = np.random.randn(10, 5)

    pipeline = PreprocessingPipeline(use_pca=False)
    X_train_out, X_test_out = pipeline._apply_pca(X_train, X_test)

    np.testing.assert_array_equal(X_train_out, X_train)
    np.testing.assert_array_equal(X_test_out, X_test)
    assert pipeline.pca_transformer is None


def _make_sample_dataframe():
    """Helper: minimal DataFrame mimicking dataset structure."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Feature_A": np.random.randn(n),
        "Feature_B": np.random.randn(n) * 10 + 50,
        "Feature_C": np.random.randint(0, 5, n).astype(float),
        "Target": np.random.choice(["Dropout", "Graduate", "Enrolled"], n, p=[0.3, 0.5, 0.2]),
    })
    return df


def test_fit_transform_returns_correct_shapes():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    total = len(df[df["Target"].isin(["Dropout", "Graduate", "Enrolled"])])
    expected_train = int(total * 0.8)

    assert abs(len(X_train) - expected_train) <= 1
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_fit_transform_with_smote_increases_train_size():
    df = _make_sample_dataframe()
    pipeline_none = PreprocessingPipeline(imbalance_strategy="none", random_state=42)
    X_train_none, _, _, _ = pipeline_none.fit_transform(df)

    pipeline_smote = PreprocessingPipeline(imbalance_strategy="smote", random_state=42)
    X_train_smote, _, _, _ = pipeline_smote.fit_transform(df)

    assert len(X_train_smote) >= len(X_train_none)


def test_fit_transform_with_pca_reduces_features():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(use_pca=True, pca_variance=0.95, random_state=42)
    X_train, X_test, _, _ = pipeline.fit_transform(df)

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] <= 3


def test_transform_applies_fitted_scaler():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(random_state=42)
    pipeline.fit_transform(df)

    new_data = np.array([[1.0, 50.0, 2.0]])
    transformed = pipeline.transform(new_data)
    assert transformed.shape == (1, 3)
    assert not np.allclose(transformed, new_data)


def test_get_class_weights_returns_none_when_not_used():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(imbalance_strategy="none")
    pipeline.fit_transform(df)
    assert pipeline.get_class_weights() is None


def test_get_class_weights_returns_dict_when_used():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")
    pipeline.fit_transform(df)
    weights = pipeline.get_class_weights()
    assert isinstance(weights, dict)
    assert 0 in weights and 1 in weights


def test_get_feature_names_returns_original_names():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(use_pca=False)
    pipeline.fit_transform(df)
    names = pipeline.get_feature_names()
    assert names == ["Feature_A", "Feature_B", "Feature_C"]


def test_get_feature_names_returns_pca_names():
    df = _make_sample_dataframe()
    pipeline = PreprocessingPipeline(use_pca=True, pca_variance=0.95)
    pipeline.fit_transform(df)
    names = pipeline.get_feature_names()
    assert all(name.startswith("PC") for name in names)
