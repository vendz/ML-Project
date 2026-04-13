import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from shared.preprocessing import load_data, StandardScaler
from shared.evaluation import confusion_matrix, classification_report, roc_auc, precision_recall_auc
from logistic_regression.model import LogisticRegression
from gradient_boosting.model import GradientBoostedTrees
from neural_network.model import NeuralNetwork
from dashboard.components.plots import (
    plot_confusion_matrix, plot_roc, plot_loss_curve, plot_feature_importance,
    plot_pr_curve, plot_dead_neurons, plot_confidence_histogram,
)

st.set_page_config(page_title="Student Outcome Prediction", layout="wide")
st.title("Student Academic Outcome Prediction")

@st.cache_data
def get_data():
    return load_data()

X_train, X_test, y_train, y_test, feature_names = get_data()
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# Sidebar
model_name = st.sidebar.selectbox("Model", ["Logistic Regression",
                                             "Gradient Boosted Trees",
                                             "Neural Network"])

st.sidebar.markdown("---")

# Model specific hyperparameter controls
if model_name == "Logistic Regression":
    lr      = st.sidebar.select_slider("Learning Rate", [0.001, 0.01, 0.1], value=0.01)
    lam     = st.sidebar.select_slider("Lambda (regularization)", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], value=0.01)
    reg     = st.sidebar.selectbox("Regularization Type", ["l2", "l1", "elasticnet"])
    bs      = st.sidebar.selectbox("Batch Size", [1, 32, 64, "full-batch"])
    cw      = st.sidebar.checkbox("Class Weighting", value=True)
    params  = dict(lr=lr, lambda_=lam, reg=reg,
                   batch_size=None if bs == "full-batch" else int(bs),
                   max_epochs=1000, patience=10, class_weight=cw)
    model   = LogisticRegression(**params)

elif model_name == "Gradient Boosted Trees":
    lr        = st.sidebar.select_slider("Learning Rate", [0.01, 0.05, 0.1, 0.3], value=0.05)
    rounds    = st.sidebar.slider("Boosting Rounds", 10, 500, 100, step=10)
    depth     = st.sidebar.slider("Max Tree Depth", 1, 8, 7)
    ss        = st.sidebar.slider("Subsample Rate", 0.5, 1.0, 0.8, step=0.1)
    threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.33, step=0.01)
    params    = dict(learning_rate=lr, n_estimators=rounds, max_depth=depth,
                     subsample=ss, threshold=threshold)
    model     = GradientBoostedTrees(**params)

else:  # Neural Network
    with st.sidebar.expander("Architecture", expanded=True):
        n_layers = st.slider("Hidden Layers", 1, 5, 2, key="nn_n_layers")
        hidden_dims, activations = [], []
        _act_options = ["relu", "leaky_relu", "tanh", "sigmoid"]
        _default_sizes = [64, 32, 32, 32, 32]
        for _i in range(n_layers):
            hidden_dims.append(
                st.select_slider(f"Layer {_i + 1} Size",
                                 [4, 8, 16, 32, 64, 128, 256, 512],
                                 value=_default_sizes[_i], key=f"nn_size_{_i}")
            )
            activations.append(
                st.selectbox(f"Layer {_i + 1} Activation", _act_options,
                             key=f"nn_act_{_i}")
            )
        leaky_alpha = (
            st.slider("Leaky ReLU Alpha", 0.001, 0.5, 0.01,
                      step=0.001, key="nn_leaky_alpha")
            if "leaky_relu" in activations else 0.01
        )
        init_strategy = st.selectbox("Weight Init", ["xavier", "he"],
                                     key="nn_init")

    with st.sidebar.expander("Optimizer"):
        opt = st.selectbox("Optimizer", ["adam", "sgd"], key="nn_opt")
        lr  = st.select_slider(
            "Learning Rate",
            [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            value=0.001, key="nn_lr",
        )
        if opt == "sgd":
            momentum = st.slider("Momentum", 0.0, 0.99, 0.9,
                                 step=0.01, key="nn_mom")
            beta1, beta2 = 0.9, 0.999
        else:
            momentum = 0.9
            beta1 = st.select_slider("Adam Beta1",
                                     [0.8, 0.85, 0.9, 0.95, 0.99],
                                     value=0.9, key="nn_b1")
            beta2 = st.select_slider("Adam Beta2",
                                     [0.9, 0.95, 0.99, 0.999, 0.9999],
                                     value=0.999, key="nn_b2")
        decay = st.selectbox("LR Decay", ["none", "step", "exponential"],
                             key="nn_decay")
        if decay != "none":
            decay_rate  = st.slider("Decay Rate", 0.01, 0.9, 0.1,
                                    step=0.01, key="nn_drate")
            decay_steps = (
                st.slider("Decay Steps", 10, 500, 100,
                          step=10, key="nn_dsteps")
                if decay == "step" else 100
            )
        else:
            decay_rate, decay_steps = 0.1, 100

    with st.sidebar.expander("Regularization"):
        l1 = st.select_slider("L1 Lambda",
                              [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                              value=0.0, key="nn_l1")
        l2 = st.select_slider("L2 Lambda",
                              [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                              value=0.001, key="nn_l2")
        dr = st.slider("Dropout Rate", 0.0, 0.8, 0.0,
                       step=0.05, key="nn_dr")

    with st.sidebar.expander("Training"):
        bs_opt     = st.selectbox("Batch Size",
                                  [16, 32, 64, 128, "full-batch"],
                                  index=1, key="nn_bs")
        bs         = None if bs_opt == "full-batch" else int(bs_opt)
        max_epochs = st.slider("Max Epochs", 50, 2000, 500,
                               step=50, key="nn_epochs")
        patience   = st.slider("Patience", 5, 50, 10, key="nn_patience")
        cw         = st.checkbox("Class Weighting", value=True, key="nn_cw")

    model = NeuralNetwork(
        hidden_dims=hidden_dims,
        activations=activations,
        optimizer=opt,
        lr=lr,
        momentum=momentum,
        beta1=beta1,
        beta2=beta2,
        l1_lambda=l1,
        l2_lambda=l2,
        leaky_alpha=leaky_alpha,
        dropout_rate=dr,
        init_strategy=init_strategy,
        lr_decay=decay,
        lr_decay_rate=decay_rate,
        lr_decay_steps=decay_steps,
        batch_size=bs,
        max_epochs=max_epochs,
        patience=patience,
        class_weight=cw,
    )

if st.sidebar.button("Train & Evaluate"):
    if model_name == "Neural Network":
        st.subheader("Training Loss")
        _loss_chart = st.empty()
        _tl, _vl = [], []
        _throttle = max(1, max_epochs // 50)

        def _epoch_cb(epoch, train_loss, val_loss):
            _tl.append(train_loss)
            if val_loss is not None:
                _vl.append(val_loss)
            if epoch % _throttle == 0:
                _fig = plot_loss_curve(_tl, _vl if _vl else None)
                _loss_chart.pyplot(_fig)
                plt.close(_fig)

        with st.spinner("Training neural network..."):
            model.fit(X_train_s, y_train,
                      X_val=X_test_s, y_val=y_test,
                      epoch_callback=_epoch_cb)
        _fig = plot_loss_curve(_tl, _vl if _vl else None)
        _loss_chart.pyplot(_fig)
        plt.close(_fig)

    else:
        with st.spinner("Training..."):
            model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    report  = classification_report(y_test, y_pred)
    fpr, tpr, auc = roc_auc(y_test, y_proba)

    col1, col2, col3 = st.columns(3)
    col1.metric("Macro F1",  f"{report['macro_f1']:.4f}")
    col2.metric("Accuracy",  f"{report['accuracy']:.4f}")
    col3.metric("ROC AUC",   f"{auc:.4f}")

    st.subheader("Per-Class Results")
    st.json({k: v for k, v in report.items() if k not in ("macro_f1", "accuracy")})

    st.subheader("Confusion Matrix & ROC Curve")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        cm = confusion_matrix(y_test, y_pred)
        st.pyplot(plot_confusion_matrix(cm))
    with pcol2:
        st.pyplot(plot_roc(fpr, tpr, auc))

    if model_name == "Gradient Boosted Trees":
        st.subheader("Feature Importance")
        importances, aug_names = model.feature_importance(feature_names)
        st.pyplot(plot_feature_importance(importances, aug_names))

        st.subheader("Prediction Confidence")
        fig = plot_confidence_histogram(y_test, y_proba)
        st.pyplot(fig)
        plt.close(fig)

        st.session_state["gbt_results"] = {"y_proba": y_proba}

    if model_name == "Neural Network":
        st.session_state["nn_results"] = {
            "y_proba":      y_proba,
            "model":        model,
            "train_losses": _tl,
            "val_losses":   _vl,
            "pr_curve":     precision_recall_auc(y_test, y_proba),
            "layer_acts":   model.get_layer_activations(X_test_s),
        }

if model_name == "Gradient Boosted Trees" and "gbt_results" in st.session_state:
    _gbt_proba = st.session_state["gbt_results"]["y_proba"]
    st.markdown("---")
    st.markdown("#### Decision Threshold Explorer")
    st.caption("Adjust the threshold to see how precision, recall, and F1 change without retraining.")
    _gbt_thresh = st.slider("Threshold", 0.01, 0.99, 0.33, step=0.01, key="gbt_thresh")
    _gbt_pred = (_gbt_proba >= _gbt_thresh).astype(int)
    _gbt_report = classification_report(y_test, _gbt_pred)
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Precision (dropout)", f"{_gbt_report['dropout']['precision']:.3f}")
    tc2.metric("Recall (dropout)",    f"{_gbt_report['dropout']['recall']:.3f}")
    tc3.metric("F1 (dropout)",        f"{_gbt_report['dropout']['f1']:.3f}")
    tc4.metric("Accuracy",            f"{_gbt_report['accuracy']:.3f}")

if model_name == "Neural Network" and "nn_results" in st.session_state:
    _r  = st.session_state["nn_results"]
    _nn = _r["model"]
    _yp = _r["y_proba"]

    st.markdown("---")
    st.subheader("Neural Network Analysis")

    _epochs_run  = len(_r["train_losses"])
    _param_count = sum(w.size + b.size for w, b in zip(_nn.weights_, _nn.biases_))
    _stopped_early = _epochs_run < _nn.max_epochs
    _final_train = _r["train_losses"][-1]
    _final_val   = _r["val_losses"][-1] if _r["val_losses"] else None
    _gap_pct     = (_final_val - _final_train) / _final_train * 100 if _final_val is not None else None

    sb1, sb2, sb3, sb4 = st.columns(4)
    sb1.metric("Epochs trained",
               f"{_epochs_run}/{_nn.max_epochs}",
               delta="early stop" if _stopped_early else "max reached",
               delta_color="normal" if _stopped_early else "inverse")
    sb2.metric("Parameters", f"{_param_count:,}")
    sb3.metric("Final train loss", f"{_final_train:.4f}")
    if _final_val is not None:
        _gap_color = "normal" if _gap_pct < 10 else "inverse"
        sb4.metric("Final val loss", f"{_final_val:.4f}",
                   delta=f"gap {_gap_pct:+.1f}%", delta_color=_gap_color)

    st.markdown("#### Decision Threshold")
    st.caption("Drag to see how precision, recall, and F1 trade off without retraining.")
    _thresh = st.slider("Threshold", 0.01, 0.99, 0.50, step=0.01, key="nn_thresh")
    _yt = (_yp >= _thresh).astype(int)
    _rt = classification_report(y_test, _yt)
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Precision (dropout)", f"{_rt['dropout']['precision']:.3f}")
    tc2.metric("Recall (dropout)",    f"{_rt['dropout']['recall']:.3f}")
    tc3.metric("F1 (dropout)",        f"{_rt['dropout']['f1']:.3f}")
    tc4.metric("Accuracy",          f"{_rt['accuracy']:.3f}")

    st.markdown("#### Model Behaviour")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.caption("A well-trained model has two peaks near 0 and 1. A blob in the "
                   "middle means the model is uncertain about many samples.")
        _fig = plot_confidence_histogram(y_test, _yp)
        st.pyplot(_fig); plt.close(_fig)
    with ic2:
        _rec, _pre, _auc_pr = _r["pr_curve"]
        _baseline = float((y_test == 1).mean())
        st.caption(f"Positive class rate (random baseline): {_baseline:.2f}. "
                   f"Higher AUC-PR means better minority-class detection.")
        _fig = plot_pr_curve(_rec, _pre, _auc_pr, _baseline)
        st.pyplot(_fig); plt.close(_fig)

    st.markdown("#### Network Health")
    ic1, ic2 = st.columns(2)
    with ic1:
        _layer_acts = _r["layer_acts"]
        _dlabels, _dpcts = [], []
        for _i, (_A, _act) in enumerate(zip(_layer_acts, _nn.activations_)):
            _dlabels.append(f"L{_i + 1} ({_act})")
            if _act == "relu":
                _dpcts.append(float((_A == 0).mean()) * 100)
            elif _act == "leaky_relu":
                _dpcts.append(float((_A < 0).mean()) * 100)
            elif _act == "tanh":
                _dpcts.append(float((np.abs(_A) > 0.97).mean()) * 100)
            else:
                _dpcts.append(float(((_A < 0.05) | (_A > 0.95)).mean()) * 100)
        st.caption("Green < 20% | Orange < 50% | Red > 50%")
        _fig = plot_dead_neurons(_dlabels, _dpcts, _nn.activations_)
        st.pyplot(_fig); plt.close(_fig)

    with ic2:
        st.markdown("**Recommendations**")
        _recs = []

        if _gap_pct is not None:
            if _gap_pct > 30:
                _recs.append(f"🔴 **High overfitting** (val {_gap_pct:.0f}% above train) — "
                             "raise dropout, increase L2, or reduce layer sizes.")
            elif _gap_pct > 10:
                _recs.append(f"🟡 **Mild overfitting** (val {_gap_pct:.0f}% above train) — "
                             "try light dropout or a small L2 penalty.")
            else:
                _recs.append(f"🟢 **Good generalisation** — val/train gap is only {_gap_pct:.1f}%.")

        # Convergence
        if _stopped_early:
            if _epochs_run < _nn.max_epochs * 0.1:
                _recs.append(f"🟡 **Converged very fast** (epoch {_epochs_run}/{_nn.max_epochs}) — "
                             "model may be too simple, or LR is too high.")
            else:
                _recs.append(f"🟢 **Early stopping** at epoch {_epochs_run}/{_nn.max_epochs} "
                             f"(patience={_nn.patience}) — training converged cleanly.")
        else:
            _recs.append(f"🟡 **Max epochs reached** — model may not have converged; "
                         "try increasing max epochs or raising LR.")

        if len(_r["train_losses"]) > 1:
            _loss_drop = (_r["train_losses"][0] - _final_train) / (_r["train_losses"][0] + 1e-12)
            if _loss_drop < 0.05:
                _recs.append("🔴 **Loss barely decreased** — LR is likely too low; "
                             "try 10× higher LR or switch to Adam.")

        if _auc_pr < _baseline * 1.5:
            _recs.append("🔴 **AUC-PR near random baseline** — model struggles with the "
                         "minority class; enable class weighting or try SMOTE.")
        for _lbl, _pct, _act in zip(_dlabels, _dpcts, _nn.activations_):
            if _act == "relu" and _pct > 50:
                _recs.append(f"🔴 **{_lbl}: {_pct:.0f}% dead neurons** — "
                             "switch weight init to He or lower the learning rate.")
            elif _pct > 80:
                _recs.append(f"🟡 **{_lbl}: {_pct:.0f}% saturated** — "
                             "try a different activation or reduce LR.")

        for _rec in _recs:
            st.markdown(_rec)
