import os
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import PreprocessingPipeline


# --- Helpers ---

def _make_sample_dataframe(n=200, missing=False):
    """Helper: minimal DataFrame mimicking the actual dataset structure."""
    np.random.seed(42)
    df = pd.DataFrame({
        "Student_ID": range(1, n + 1),
        "Age": np.random.uniform(18, 30, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Family_Income": np.random.uniform(20000, 80000, n),
        "Internet_Access": np.random.choice(["Yes", "No"], n),
        "Study_Hours_per_Day": np.random.uniform(1, 8, n),
        "Attendance_Rate": np.random.uniform(50, 100, n),
        "Assignment_Delay_Days": np.random.randint(0, 10, n),
        "Travel_Time_Minutes": np.random.uniform(5, 60, n),
        "Part_Time_Job": np.random.choice(["Yes", "No"], n),
        "Scholarship": np.random.choice(["Yes", "No"], n),
        "Stress_Index": np.random.uniform(1, 10, n),
        "GPA": np.random.uniform(0.5, 4.0, n),
        "Semester_GPA": np.random.uniform(0.5, 4.0, n),
        "CGPA": np.random.uniform(0.5, 4.0, n),
        "Semester": np.random.choice(["Year 1", "Year 2", "Year 3", "Year 4"], n),
        "Department": np.random.choice(["CS", "Engineering", "Arts", "Business", "Science"], n),
        "Parental_Education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
        "Dropout": np.random.choice([0, 1], n, p=[0.76, 0.24]),
    })
    if missing:
        # Inject ~5% missing values in specific columns
        for col in ["Family_Income", "Study_Hours_per_Day", "Stress_Index", "Parental_Education"]:
            mask = np.random.rand(n) < 0.05
            df.loc[mask, col] = np.nan
    return df


def _make_imbalanced_data():
    """Helper: 80 class-0, 20 class-1."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.array([0] * 80 + [1] * 20)
    return X, y


# --- Target extraction and ID drop ---

def test_prepare_features_extracts_target():
    """Dropout column becomes y, Student_ID is dropped."""
    df = _make_sample_dataframe(n=10)
    pipeline = PreprocessingPipeline()
    X, y = pipeline._prepare_features(df)

    assert "Dropout" not in X.columns
    assert "Student_ID" not in X.columns
    assert len(y) == 10
    assert set(np.unique(y)).issubset({0, 1})


def test_prepare_features_preserves_all_feature_columns():
    df = _make_sample_dataframe(n=10)
    pipeline = PreprocessingPipeline()
    X, y = pipeline._prepare_features(df)

    # 19 total columns - Student_ID - Dropout = 17
    assert X.shape[1] == 17


# --- Imputation ---

def test_impute_fills_missing_values():
    df = _make_sample_dataframe(n=200, missing=True)
    pipeline = PreprocessingPipeline()
    X, y = pipeline._prepare_features(df)
    X_train, X_test, _, _ = pipeline._split(X, y)
    X_train, X_test = pipeline._impute(X_train, X_test)

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert pipeline.num_imputer is not None


def test_impute_no_missing_passes_through():
    df = _make_sample_dataframe(n=50, missing=False)
    pipeline = PreprocessingPipeline()
    X, y = pipeline._prepare_features(df)
    X_train, X_test, _, _ = pipeline._split(X, y)
    X_train_out, X_test_out = pipeline._impute(X_train, X_test)

    assert X_train_out.shape == X_train.shape


# --- Categorical encoding ---

def test_encode_categoricals_converts_to_numeric():
    df = _make_sample_dataframe(n=50)
    pipeline = PreprocessingPipeline()
    X, y = pipeline._prepare_features(df)
    X_train, X_test, _, _ = pipeline._split(X, y)
    X_train, X_test = pipeline._impute(X_train, X_test)
    X_train, X_test = pipeline._encode_categoricals(X_train, X_test)

    for col in pipeline.CATEGORICAL_COLS:
        assert X_train[col].dtype in [np.int32, np.int64, np.intp]
    assert len(pipeline.label_encoders) == len(pipeline.CATEGORICAL_COLS)


# --- Split ---

def test_split_preserves_stratification():
    """Train/test split should maintain class proportions."""
    df = _make_sample_dataframe(n=500)
    pipeline = PreprocessingPipeline(test_size=0.2, random_state=42)
    X, y = pipeline._prepare_features(df)
    X_train, X_test, y_train, y_test = pipeline._split(X, y)

    assert len(X_train) == 400
    assert len(X_test) == 100
    # Dropout ratio should be roughly preserved
    overall_ratio = y.mean()
    train_ratio = y_train.mean()
    assert abs(train_ratio - overall_ratio) < 0.05


# --- Standardization ---

def test_standardize_fits_on_train_only():
    X_train = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    X_test = np.array([[5.0, 500.0], [6.0, 600.0]])

    pipeline = PreprocessingPipeline()
    X_train_s, X_test_s = pipeline._standardize(X_train, X_test)

    assert abs(X_train_s[:, 0].mean()) < 1e-10
    assert abs(X_train_s[:, 0].std(ddof=0) - 1.0) < 1e-10
    assert abs(X_test_s[:, 0].mean()) > 0.5
    assert pipeline.scaler is not None


# --- Class weights ---

def test_class_weight_computation():
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")
    weights = pipeline._compute_class_weights(y_train)

    expected_w0 = 10 / (2 * 7)
    expected_w1 = 10 / (2 * 3)
    assert abs(weights[0] - expected_w0) < 1e-3
    assert abs(weights[1] - expected_w1) < 1e-3


# --- Imbalance handling ---

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


def test_handle_imbalance_class_weight_stores_weights():
    X, y = _make_imbalanced_data()
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")
    X_out, y_out = pipeline._handle_imbalance(X, y)
    assert X_out.shape == X.shape
    assert pipeline.class_weights is not None
    assert pipeline.class_weights[1] > pipeline.class_weights[0]


# --- PCA ---

def test_pca_reduces_dimensions():
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
    X_train = np.random.randn(50, 5)
    X_test = np.random.randn(10, 5)

    pipeline = PreprocessingPipeline(use_pca=False)
    X_train_out, X_test_out = pipeline._apply_pca(X_train, X_test)

    np.testing.assert_array_equal(X_train_out, X_train)
    np.testing.assert_array_equal(X_test_out, X_test)
    assert pipeline.pca_transformer is None


# --- Full pipeline (fit_transform) ---

def test_fit_transform_returns_correct_shapes():
    df = _make_sample_dataframe(n=200)
    pipeline = PreprocessingPipeline(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    assert abs(len(X_train) - 160) <= 1
    assert abs(len(X_test) - 40) <= 1
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_fit_transform_with_missing_values():
    df = _make_sample_dataframe(n=200, missing=True)
    pipeline = PreprocessingPipeline(random_state=42)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_fit_transform_with_smote_increases_train_size():
    df = _make_sample_dataframe(n=200)
    pipeline_none = PreprocessingPipeline(imbalance_strategy="none", random_state=42)
    X_train_none, _, _, _ = pipeline_none.fit_transform(df)

    pipeline_smote = PreprocessingPipeline(imbalance_strategy="smote", random_state=42)
    X_train_smote, _, _, _ = pipeline_smote.fit_transform(df)

    assert len(X_train_smote) >= len(X_train_none)


def test_fit_transform_with_pca():
    df = _make_sample_dataframe(n=200)
    pipeline = PreprocessingPipeline(use_pca=True, pca_variance=0.95, random_state=42)
    X_train, X_test, _, _ = pipeline.fit_transform(df)

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] <= 17  # at most all 17 features


def test_transform_applies_fitted_scaler():
    df = _make_sample_dataframe(n=200)
    pipeline = PreprocessingPipeline(random_state=42)
    X_train, X_test, _, _ = pipeline.fit_transform(df)

    n_features = X_train.shape[1]
    new_data = np.ones((1, n_features))
    transformed = pipeline.transform(new_data)
    assert transformed.shape == (1, n_features)
    assert not np.allclose(transformed, new_data)


# --- Accessors ---

def test_get_class_weights_returns_none_when_not_used():
    df = _make_sample_dataframe(n=100)
    pipeline = PreprocessingPipeline(imbalance_strategy="none")
    pipeline.fit_transform(df)
    assert pipeline.get_class_weights() is None


def test_get_class_weights_returns_dict_when_used():
    df = _make_sample_dataframe(n=100)
    pipeline = PreprocessingPipeline(imbalance_strategy="class_weight")
    pipeline.fit_transform(df)
    weights = pipeline.get_class_weights()
    assert isinstance(weights, dict)
    assert 0 in weights and 1 in weights


def test_get_feature_names_returns_original_names():
    df = _make_sample_dataframe(n=100)
    pipeline = PreprocessingPipeline(use_pca=False)
    pipeline.fit_transform(df)
    names = pipeline.get_feature_names()
    assert len(names) == 17
    assert "GPA" in names
    assert "Student_ID" not in names
    assert "Dropout" not in names


def test_get_feature_names_returns_pca_names():
    df = _make_sample_dataframe(n=100)
    pipeline = PreprocessingPipeline(use_pca=True, pca_variance=0.95)
    pipeline.fit_transform(df)
    names = pipeline.get_feature_names()
    assert all(name.startswith("PC") for name in names)


# --- Integration test with real dataset ---

@pytest.mark.skipif(
    not os.path.exists("data/dataset.csv")
    or "Dropout" not in open("data/dataset.csv").readline(),
    reason="Dataset not available",
)
def test_integration_with_real_dataset():
    """Full pipeline on the actual dataset."""
    df = pd.read_csv("data/dataset.csv")
    pipeline = PreprocessingPipeline(
        imbalance_strategy="smote",
        use_pca=True,
        pca_variance=0.95,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}
    assert (y_train == 0).sum() == (y_train == 1).sum()
    assert X_train.shape[1] <= 17
    names = pipeline.get_feature_names()
    assert all(n.startswith("PC") for n in names)
