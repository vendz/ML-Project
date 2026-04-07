"""
Neural Network experiment runner.
Owner: Vandit Vasa

Run with:  python -m neural_network.experiments
Results logged to:  experiments/neural_network/log.jsonl
"""
import numpy as np
from shared.preprocessing import load_data, StandardScaler, SMOTE, random_undersample
from shared.evaluation import cross_validate, classification_report, roc_auc, log_experiment
from shared.config import RANDOM_SEED, CV_FOLDS
from neural_network.model import NeuralNetwork


def run_all():
    X_train, X_test, y_train, y_test, feature_names = load_data()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Hyperparameter sweeps ──────────────────────────────────────────────────

    sweep_depth  = [[32], [64, 32], [128, 64, 32]]
    sweep_width  = [32, 64, 128, 256]
    sweep_lr     = [0.001, 0.01, 0.1]
    sweep_lambda = [0.0001, 0.001, 0.01, 0.1]
    sweep_dropout = [0.0, 0.2, 0.5]
    sweep_batch  = [32, 64, None]

    # Architecture (depth)
    for hidden in sweep_depth:
        params = dict(hidden_dims=hidden, lr=0.01, lambda_=0.001, dropout_rate=0.0,
                      batch_size=32, max_epochs=500, patience=10, class_weight=True)
        cv = cross_validate(NeuralNetwork, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("neural_network", params, cv, extra={"sweep": "depth"})

    # Width (single hidden layer)
    for width in sweep_width:
        params = dict(hidden_dims=[width], lr=0.01, lambda_=0.001, dropout_rate=0.0,
                      batch_size=32, max_epochs=500, patience=10, class_weight=True)
        cv = cross_validate(NeuralNetwork, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("neural_network", params, cv, extra={"sweep": "width"})

    # Learning rate
    for lr in sweep_lr:
        params = dict(hidden_dims=[64], lr=lr, lambda_=0.001, dropout_rate=0.0,
                      batch_size=32, max_epochs=500, patience=10, class_weight=True)
        cv = cross_validate(NeuralNetwork, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("neural_network", params, cv, extra={"sweep": "lr"})

    # L2 regularization
    for lam in sweep_lambda:
        params = dict(hidden_dims=[64], lr=0.01, lambda_=lam, dropout_rate=0.0,
                      batch_size=32, max_epochs=500, patience=10, class_weight=True)
        cv = cross_validate(NeuralNetwork, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("neural_network", params, cv, extra={"sweep": "lambda"})

    # Dropout
    for dr in sweep_dropout:
        params = dict(hidden_dims=[64], lr=0.01, lambda_=0.001, dropout_rate=dr,
                      batch_size=32, max_epochs=500, patience=10, class_weight=True)
        cv = cross_validate(NeuralNetwork, params, X_train_s, y_train, CV_FOLDS)
        log_experiment("neural_network", params, cv, extra={"sweep": "dropout"})

    # Class imbalance strategies
    best_params = dict(hidden_dims=[64], lr=0.01, lambda_=0.001, dropout_rate=0.2,
                       batch_size=32, max_epochs=500, patience=10)

    strategies = {
        "none":         (X_train_s, y_train),
        "class_weight": (X_train_s, y_train),
        "smote":        SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_s, y_train),
        "undersample":  random_undersample(X_train_s, y_train, RANDOM_SEED),
    }
    for name, (Xb, yb) in strategies.items():
        use_cw = name == "class_weight"
        params = {**best_params, "class_weight": use_cw}
        model = NeuralNetwork(**params).fit(Xb, yb)
        y_pred = model.predict(X_test_s)
        report = classification_report(y_test, y_pred)
        log_experiment("neural_network", params, report,
                       extra={"imbalance_strategy": name, "split": "test"})


if __name__ == "__main__":
    run_all()
    print("Done. Results in experiments/neural_network/log.jsonl")
