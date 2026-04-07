import numpy as np
from shared.preprocessing import load_data, StandardScaler, SMOTE, random_undersample
from shared.evaluation import cross_validate, classification_report, roc_auc, log_experiment
from shared.config import RANDOM_SEED, CV_FOLDS
from gradient_boosting.model import GradientBoostedTrees


def run_all():
    X_train, X_test, y_train, y_test, feature_names = load_data()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    sweep_lr = [0.01, 0.05, 0.1, 0.3]
    sweep_rounds = [50, 100, 200, 500]
    sweep_depth = [1, 3, 5, 7]
    sweep_subsample = [0.5, 0.7, 1.0]

    # Learning rate vs n_rounds (heatmap)
    for lr in sweep_lr:
        for rounds in sweep_rounds:
            params = dict(n_rounds=rounds, learning_rate=lr, max_depth=3,
                          subsample=1.0, patience=10, class_weight=True)
            cv = cross_validate(GradientBoostedTrees, params, X_train_s, y_train, CV_FOLDS)
            log_experiment("gradient_boosting", params, cv,
                           extra={"sweep": "lr_vs_rounds"})

    # Depth ablation
    for depth in sweep_depth:
        params = dict(n_rounds=100, learning_rate=0.1, max_depth=depth,
                      subsample=1.0, patience=10, class_weight=True)
        cv = cross_validate(GradientBoostedTrees, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("gradient_boosting", params, cv, extra={"sweep": "depth"})

    # Subsample (stochastic GBT)
    for ss in sweep_subsample:
        params = dict(n_rounds=100, learning_rate=0.1, max_depth=3,
                      subsample=ss, patience=10, class_weight=True)
        cv = cross_validate(GradientBoostedTrees, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("gradient_boosting", params, cv, extra={"sweep": "subsample"})

    # Class imbalance strategies
    best_params = dict(n_rounds=100, learning_rate=0.1, max_depth=3,
                       subsample=1.0, patience=10)

    strategies = {
        "none":         (X_train_s, y_train),
        "class_weight": (X_train_s, y_train),
        "smote":        SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_s, y_train),
        "undersample":  random_undersample(X_train_s, y_train, RANDOM_SEED),
    }
    for name, (Xb, yb) in strategies.items():
        use_cw = name == "class_weight"
        params = {**best_params, "class_weight": use_cw}
        model = GradientBoostedTrees(**params).fit(Xb, yb)
        y_pred = model.predict(X_test_s)
        report = classification_report(y_test, y_pred)
        log_experiment("gradient_boosting", params, report,
                       extra={"imbalance_strategy": name, "split": "test"})


if __name__ == "__main__":
    run_all()
    print("Done. Results in experiments/gradient_boosting/log.jsonl")
