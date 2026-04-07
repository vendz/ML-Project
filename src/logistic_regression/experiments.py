"""
Logistic Regression experiment runner.
Owner: Brandon Tran

Run with:  python -m logistic_regression.experiments
Results logged to:  experiments/logistic_regression/log.jsonl
"""
import numpy as np
from shared.preprocessing import load_data, StandardScaler, SMOTE, random_undersample
from shared.evaluation import cross_validate, classification_report, roc_auc, log_experiment
from shared.config import RANDOM_SEED, CV_FOLDS
from logistic_regression.model import LogisticRegression


def run_all():
    X_train, X_test, y_train, y_test, feature_names = load_data()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    sweep_lr = [0.001, 0.01, 0.1]
    sweep_lambda = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    sweep_reg = ["l2", "l1", "elasticnet"]
    sweep_batch = [1, 32, 64, None]

    # Learning rate sweep
    for lr in sweep_lr:
        params = dict(lr=lr, lambda_=0.01, reg="l2", batch_size=32,
                      max_epochs=1000, patience=10, class_weight=True)
        cv = cross_validate(LogisticRegression, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("logistic_regression", params, cv, extra={"sweep": "lr"})

    # Regularization strength sweep
    for lam in sweep_lambda:
        for reg in sweep_reg:
            params = dict(lr=0.01, lambda_=lam, reg=reg, batch_size=32,
                          max_epochs=1000, patience=10, class_weight=True)
            cv = cross_validate(LogisticRegression, params, X_train_s, y_train, CV_FOLDS)
            log_experiment("logistic_regression", params, cv, extra={"sweep": "regularization"})

    # Batch size sweep (convergence analysis)
    for bs in sweep_batch:
        params = dict(lr=0.01, lambda_=0.01, reg="l2", batch_size=bs,
                      max_epochs=1000, patience=10, class_weight=True)
        cv = cross_validate(LogisticRegression, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("logistic_regression", params, cv, extra={"sweep": "batch_size"})

    # Class imbalance strategies (on best params)
    best_params = dict(lr=0.01, lambda_=0.01, reg="l2", batch_size=32,
                       max_epochs=1000, patience=10)

    strategies = {
        "none":          (X_train_s, y_train),
        "class_weight":  (X_train_s, y_train),  # handled inside model via class_weight=True
        "smote":         SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_s, y_train),
        "undersample":   random_undersample(X_train_s, y_train, RANDOM_SEED),
    }
    for name, (Xb, yb) in strategies.items():
        use_cw = name == "class_weight"
        params = {**best_params, "class_weight": use_cw}
        model = LogisticRegression(**params).fit(Xb, yb)
        y_pred = model.predict(X_test_s)
        report = classification_report(y_test, y_pred)
        log_experiment("logistic_regression", params, report,
                       extra={"imbalance_strategy": name, "split": "test"})


if __name__ == "__main__":
    run_all()
    print("Done. Results in experiments/logistic_regression/log.jsonl")
