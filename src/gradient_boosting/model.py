import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier

from shared.base_model import BaseModel
from shared.config import RANDOM_SEED


class GradientBoostedTrees(BaseModel):
    """
    Gradient Boosted Trees for binary dropout prediction.

    Wraps the best configuration from experimentation (Iter 8):
    - KMeans (k=3) + PCA (95% variance) pseudo-features appended to scaled input
    - Partial SMOTE (minority:majority = smote_ratio) + Tomek link cleaning
    - XGBoost with tuned hyperparameters

    Expects already-scaled input (apply StandardScaler before calling fit/predict).

    Parameters
    ----------
    n_estimators      : number of boosting rounds
    learning_rate     : shrinkage per round
    max_depth         : max tree depth
    subsample         : row subsampling fraction per round
    min_child_weight  : minimum sum of instance weight in a leaf
    reg_alpha         : L1 regularization
    reg_lambda        : L2 regularization
    colsample_bytree  : column subsampling fraction per tree
    smote_ratio       : target minority:majority ratio after SMOTE (0 = no SMOTE)
    n_clusters        : number of KMeans clusters for pseudo-features
    threshold         : decision threshold for predict()
    random_state      : reproducibility seed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        subsample: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.5,
        reg_lambda: float = 0.5,
        colsample_bytree: float = 1.0,
        smote_ratio: float = 0.4,
        n_clusters: int = 3,
        threshold: float = 0.33,
        random_state: int = RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree
        self.smote_ratio = smote_ratio
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.random_state = random_state

        self._kmeans: KMeans | None = None
        self._pca: PCA | None = None
        self._xgb: XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        # Fit unsupervised components on original scaled features
        self._kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self._kmeans.fit(X)

        self._pca = PCA(n_components=0.95, random_state=self.random_state)
        self._pca.fit(X)

        X_aug = self._augment(X)

        # Partial SMOTE then Tomek cleaning (same pipeline as notebook experiments)
        X_res, y_res = SMOTE(
            sampling_strategy=self.smote_ratio, random_state=self.random_state
        ).fit_resample(X_aug, y)
        X_res, y_res = TomekLinks().fit_resample(X_res, y_res)

        self._xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self._xgb.fit(X_res, y_res)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._xgb.predict_proba(self._augment(X))[:, 1]

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "colsample_bytree": self.colsample_bytree,
            "smote_ratio": self.smote_ratio,
            "n_clusters": self.n_clusters,
            "threshold": self.threshold,
        }

    def feature_importance(self, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
        """Return (importances, augmented_feature_names) for all 35 features."""
        n_pca = self._pca.n_components_
        aug_names = (
            list(feature_names)
            + ["cluster_id"]
            + [f"pca_{i}" for i in range(n_pca)]
            + [f"dist_cluster_{i}" for i in range(self.n_clusters)]
        )
        return self._xgb.feature_importances_, aug_names

    # ---- private helpers ----

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Append cluster ID, PCA components, and centroid distances to X."""
        cluster_ids = self._kmeans.predict(X).reshape(-1, 1).astype(float)
        pca_components = self._pca.transform(X)
        distances = self._kmeans.transform(X)  # (n_samples, n_clusters), Euclidean to each centroid
        return np.hstack([X, cluster_ids, pca_components, distances])

