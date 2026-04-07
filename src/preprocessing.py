import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    TARGET_MAP = {"Dropout": 1, "Graduate": 0, "Enrolled": 0}

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

    def _encode_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Encode target: Dropout=1, Graduate/Enrolled=0. Drop unknown classes."""
        df = df.copy()
        mask = df["Target"].isin(self.TARGET_MAP.keys())
        df = df[mask]
        y = df["Target"].map(self.TARGET_MAP).values.astype(int)
        X = df.drop(columns=["Target"])
        return X, y
