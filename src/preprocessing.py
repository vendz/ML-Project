import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class PreprocessingPipeline:
    """Preprocessing pipeline for student dropout prediction.

    Parameters
    ----------
    imbalance_strategy : str
        One of "none", "class_weight", "smote", "undersample".
    use_pca : bool
        Whether to apply PCA dimensionality reduction.
    pca_variance : float
        Cumulative variance threshold for PCA components.
    test_size : float
        Fraction of data reserved for the test set.
    random_state : int
        Random seed for reproducibility.
    """

    TARGET_COL = "Dropout"
    DROP_COLS = ["Student_ID"]
    CATEGORICAL_COLS = [
        "Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
        "Semester", "Department", "Parental_Education",
    ]

    def __init__(
        self,
        imbalance_strategy: str = "none",
        use_pca: bool = False,
        pca_variance: float = 0.95,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.imbalance_strategy = imbalance_strategy
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.test_size = test_size
        self.random_state = random_state

        self.scaler: StandardScaler | None = None
        self.pca_transformer: PCA | None = None
        self.class_weights: dict | None = None
        self.feature_names: list[str] = []
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.num_imputer: SimpleImputer | None = None
        self.cat_imputer: SimpleImputer | None = None

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract target, drop ID column, return features and labels."""
        df = df.copy()
        # Drop columns that aren't features
        drop = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop)
        y = df[self.TARGET_COL].values.astype(int)
        X = df.drop(columns=[self.TARGET_COL])
        return X, y

    def _impute(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute missing values: median for numeric, most_frequent for categorical."""
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in X_train.columns]

        if num_cols:
            self.num_imputer = SimpleImputer(strategy="median")
            X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])

        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = self.cat_imputer.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = self.cat_imputer.transform(X_test[cat_cols])

        return X_train, X_test

    def _encode_categoricals(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode categorical columns. Fit on train, transform both."""
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in X_train.columns]
        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le
        return X_train, X_test

    def _split(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Stratified train/test split."""
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def _standardize(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit StandardScaler on train, transform both."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _compute_class_weights(self, y: np.ndarray) -> dict[int, float]:
        """Compute class weights: N / (n_classes * n_k) for each class k."""
        classes = np.unique(y)
        n = len(y)
        n_classes = len(classes)
        weights = {}
        for c in classes:
            n_k = (y == c).sum()
            weights[int(c)] = n / (n_classes * n_k)
        self.class_weights = weights
        return weights

    def _handle_imbalance(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the chosen class imbalance strategy to training data."""
        if self.imbalance_strategy == "none":
            return X, y
        elif self.imbalance_strategy == "class_weight":
            self._compute_class_weights(y)
            return X, y
        elif self.imbalance_strategy == "smote":
            smote = SMOTE(random_state=self.random_state)
            return smote.fit_resample(X, y)
        elif self.imbalance_strategy == "undersample":
            rus = RandomUnderSampler(random_state=self.random_state)
            return rus.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown imbalance strategy: {self.imbalance_strategy}")

    def _apply_pca(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """If PCA is enabled, fit on train and transform both."""
        if not self.use_pca:
            return X_train, X_test
        self.pca_transformer = PCA(n_components=self.pca_variance, random_state=self.random_state)
        X_train_pca = self.pca_transformer.fit_transform(X_train)
        X_test_pca = self.pca_transformer.transform(X_test)
        return X_train_pca, X_test_pca

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline.

        Pipeline order:
        1. Extract target, drop ID
        2. Train/test split (stratified)
        3. Impute missing values (fit on train)
        4. Encode categoricals (fit on train)
        5. Standardize (fit on train)
        6. Handle class imbalance (train only)
        7. PCA (fit on train, optional)

        Returns (X_train, X_test, y_train, y_test) with all transforms applied.
        """
        X, y = self._prepare_features(df)
        X_train, X_test, y_train, y_test = self._split(X, y)
        X_train, X_test = self._impute(X_train, X_test)
        X_train, X_test = self._encode_categoricals(X_train, X_test)
        self.feature_names = list(X_train.columns)
        X_train_np, X_test_np = self._standardize(X_train.values, X_test.values)
        X_train_np, y_train = self._handle_imbalance(X_train_np, y_train)
        X_train_np, X_test_np = self._apply_pca(X_train_np, X_test_np)

        if self.use_pca and self.pca_transformer is not None:
            self.feature_names = [f"PC{i+1}" for i in range(X_train_np.shape[1])]

        return X_train_np, X_test_np, y_train, y_test

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted standardization and PCA to new data."""
        if self.scaler is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit_transform first.")
        X_out = self.scaler.transform(X)
        if self.use_pca and self.pca_transformer is not None:
            X_out = self.pca_transformer.transform(X_out)
        return X_out

    def get_class_weights(self) -> dict[int, float] | None:
        """Return class weights if imbalance_strategy='class_weight', else None."""
        return self.class_weights

    def get_feature_names(self) -> list[str]:
        """Return feature names (original or PCA component names)."""
        return self.feature_names
