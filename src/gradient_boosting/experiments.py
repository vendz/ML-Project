import numpy as np
from shared.preprocessing import load_data, StandardScaler
from shared.evaluation import classification_report, roc_auc, log_experiment
from shared.config import RANDOM_SEED
from gradient_boosting.model import GradientBoostedTrees


# Best configuration from Iter 8 experimentation
BEST_PARAMS = dict(
    learning_rate=0.05,
    n_estimators=100,
    max_depth=7,
    subsample=0.8,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=0.5,
    colsample_bytree=1.0,
    smote_ratio=0.4,
    n_clusters=3,
    threshold=0.33,
    random_state=RANDOM_SEED,
)


def run_all():
    X_train, X_test, y_train, y_test, _ = load_data()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- Best model: final test evaluation ---
    model = GradientBoostedTrees(**BEST_PARAMS).fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    report  = classification_report(y_test, y_pred)
    _, _, auc = roc_auc(y_test, y_proba)
    log_experiment("gradient_boosting", BEST_PARAMS,
                   {**report, "roc_auc": auc},
                   extra={"sweep": "best_model", "split": "test"})

    # --- Threshold sweep ---
    for t in np.arange(0.20, 0.61, 0.05):
        t = round(float(t), 2)
        params = {**BEST_PARAMS, "threshold": t}
        m = GradientBoostedTrees(**params).fit(X_train_s, y_train)
        report = classification_report(y_test, m.predict(X_test_s))
        log_experiment("gradient_boosting", params, report,
                       extra={"sweep": "threshold"})

    # --- Learning rate sweep ---
    for lr in [0.01, 0.05, 0.1, 0.3]:
        params = {**BEST_PARAMS, "learning_rate": lr}
        m = GradientBoostedTrees(**params).fit(X_train_s, y_train)
        report = classification_report(y_test, m.predict(X_test_s))
        log_experiment("gradient_boosting", params, report,
                       extra={"sweep": "learning_rate"})

    # --- Depth sweep ---
    for depth in [3, 5, 7]:
        params = {**BEST_PARAMS, "max_depth": depth}
        m = GradientBoostedTrees(**params).fit(X_train_s, y_train)
        report = classification_report(y_test, m.predict(X_test_s))
        log_experiment("gradient_boosting", params, report,
                       extra={"sweep": "max_depth"})

    # --- Subsample sweep ---
    for ss in [0.5, 0.7, 0.8, 1.0]:
        params = {**BEST_PARAMS, "subsample": ss}
        m = GradientBoostedTrees(**params).fit(X_train_s, y_train)
        report = classification_report(y_test, m.predict(X_test_s))
        log_experiment("gradient_boosting", params, report,
                       extra={"sweep": "subsample"})


if __name__ == "__main__":
    run_all()
    print("Done. Results in experiments/gradient_boosting/log.jsonl")
