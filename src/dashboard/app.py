"""
Streamlit Dashboard
Owner: Shivnarain Sarin

Run with:  streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path

# Make src/ importable when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np

from shared.preprocessing import load_data, StandardScaler
from shared.evaluation import classification_report, roc_auc
from logistic_regression.model import LogisticRegression
from gradient_boosting.model import GradientBoostedTrees
from neural_network.model import NeuralNetwork

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Outcome Prediction", layout="wide")
st.title("Student Academic Outcome Prediction — Interactive Dashboard")

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

X_train, X_test, y_train, y_test, feature_names = get_data()
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Sidebar: model selector ────────────────────────────────────────────────────
model_name = st.sidebar.selectbox("Model", ["Logistic Regression",
                                             "Gradient Boosted Trees",
                                             "Neural Network"])

st.sidebar.markdown("---")

# ── Model-specific hyperparameter controls ─────────────────────────────────────
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
    lr      = st.sidebar.select_slider("Learning Rate", [0.01, 0.05, 0.1, 0.3], value=0.1)
    rounds  = st.sidebar.slider("Boosting Rounds", 10, 500, 100, step=10)
    depth   = st.sidebar.slider("Max Tree Depth", 1, 7, 3)
    ss      = st.sidebar.slider("Subsample Rate", 0.5, 1.0, 1.0, step=0.1)
    cw      = st.sidebar.checkbox("Class Weighting", value=True)
    params  = dict(n_rounds=rounds, learning_rate=lr, max_depth=depth,
                   subsample=ss, patience=10, class_weight=cw)
    model   = GradientBoostedTrees(**params)

else:  # Neural Network
    depth   = st.sidebar.selectbox("Hidden Layers", ["[32]", "[64, 32]", "[128, 64, 32]"])
    lr      = st.sidebar.select_slider("Learning Rate", [0.001, 0.01, 0.1], value=0.01)
    lam     = st.sidebar.select_slider("L2 Lambda", [0.0001, 0.001, 0.01, 0.1], value=0.001)
    dr      = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.0, step=0.1)
    cw      = st.sidebar.checkbox("Class Weighting", value=True)
    params  = dict(hidden_dims=eval(depth), lr=lr, lambda_=lam, dropout_rate=dr,
                   batch_size=32, max_epochs=500, patience=10, class_weight=cw)
    model   = NeuralNetwork(**params)

# ── Train & evaluate ───────────────────────────────────────────────────────────
if st.sidebar.button("Train & Evaluate"):
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

    # TODO: Shiv — add plots (confusion matrix, ROC curve, convergence curve)
    # from dashboard.components.plots import plot_confusion_matrix, plot_roc
    st.info("Plotting components coming soon — add them in src/dashboard/components/plots.py")
