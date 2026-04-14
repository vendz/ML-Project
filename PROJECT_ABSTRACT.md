# Predicting Student Academic Outcomes: A Comparative Study of Classification Methods

## Project Documentation

**Group Members:** Vandit Vasa, Brandon Tran, Shivnarain Sarin<br>
**Course:** CS6140 - Machine Learning<br>
**Institution:** Northeastern University

## 1. Project Overview

### 1.1 Objective

Predict whether a university student will **drop out** or be **retained** based on demographic, academic, behavioral, and resource-access features. This is a binary classification problem with notable class imbalance.

### 1.2 Core Philosophy

This project is not about finding the single best model. It is a **deep comparative study** where every model is thoroughly dissected, every hyperparameter is systematically varied, and every design decision is ablated. The goal is to demonstrate mastery of the underlying mechanics of each algorithm by showing exactly how and why tweaking specific knobs changes behavior.

### 1.3 Key Highlights & Differentiators

- A **Streamlit interactive dashboard** lets users manipulate hyperparameters via sliders and see results update visually in real time.
- The neural network panel includes **live training loss curves**, **neuron health analysis** (dead/saturated neuron detection), and **automated recommendations** for overfitting, convergence, and architecture issues.
- The gradient boosted trees panel includes **post-training threshold exploration** and **confidence histograms** — adjusting the decision threshold updates metrics without retraining.
- Every experiment is logged automatically to JSONL, producing reproducible records of all ablation runs.

### 1.4 Division of Work

- **Vandit Vasa:** Lead the implementation of **Neural Network**, including architecture design, multiple activation functions (ReLU, Leaky ReLU, Tanh, Sigmoid), backpropagation from scratch, Adam and SGD+momentum optimizers, regularization (L1, L2, dropout), learning rate decay schedules, and hyperparameter tuning.
- **Brandon Tran:** Lead the implementation of **Logistic Regression**, including optimization (full-batch, mini-batch, SGD), regularization (L1 via proximal gradient, L2, Elastic Net), convergence analysis, class imbalance experiments (class weighting, SMOTE, undersampling), and probability-based evaluation (ROC, PR curves).
- **Shivnarain Sarin:** Lead the implementation of **Gradient Boosted Trees**, including feature augmentation (KMeans clustering, PCA pseudo-features, centroid distances), imbalance handling (partial SMOTE + Tomek link cleaning), threshold tuning, and the Streamlit dashboard with interactive analysis panels for all three models.
- **Shared responsibilities:** All three members collaborated on dataset understanding, evaluation metrics, cross-validation, final error analysis, and writing the final report/presentation. The shared infrastructure (`src/shared/`) — preprocessing, evaluation, configuration — was maintained jointly.

## 2. Dataset

### 2.1 Source

UCI Machine Learning Repository / Kaggle: "Predict Students' Dropout and Academic Success"

- **URL:** https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset
- **Original Paper:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021)

### 2.2 Summary

| Property           | Value                                                          |
| ------------------ | -------------------------------------------------------------- |
| Instances          | ~10,000                                                        |
| Features           | 18 (excluding Student_ID)                                      |
| Target Classes     | 2 (Retained, Dropped Out)                                      |
| Class Distribution | Imbalanced (Retained is majority ~76.5%, Dropped Out ~23.5%)   |
| Feature Types      | Mix of categorical and continuous numerics                     |
| Missing Values     | ~5% in some columns                                            |

### 2.3 Feature Groups

Features are organized into four logical groups.

**Academic Features (5):**
GPA, Semester_GPA, CGPA, Semester, Department. These capture the student's academic performance and progression through the program.

**Demographic Features (4):**
Age, Gender, Family_Income, Parental_Education. These capture who the student is and their background at enrollment.

**Behavioral & Lifestyle Features (6):**
Study_Hours_per_Day, Attendance_Rate, Assignment_Delay_Days, Travel_Time_Minutes, Part_Time_Job, Stress_Index. These capture engagement patterns and external pressures on the student's time and wellbeing.

**Access & Resource Features (2):**
Internet_Access, Scholarship. These capture whether the student has the material support needed to succeed.

### 2.4 Target Variable

| Class       | Label | Description                                              |
| ----------- | ----- | -------------------------------------------------------- |
| Retained    | 0     | Student is continuing in the program (has not dropped out) |
| Dropped Out | 1     | Student left the program before completion               |

## 3. Models: Theory, Implementation & Experiments

### 3.1 Logistic Regression

**Owner: Brandon Tran**

**Theory:**

Logistic regression models the probability of the positive class (Dropped Out) given input x using the sigmoid function:

```
P(y=1 | x) = 1 / (1 + exp(-(w . x + b)))
```

The model learns weight vector w and bias b by minimizing binary cross-entropy loss:

```
L = -(1/N) * sum_i( y_i * log(P(y=1 | x_i)) + (1 - y_i) * log(1 - P(y=1 | x_i)) ) + lambda * R(w)
```

where R(w) is the regularization term.

**Implementation Details:**

- Sigmoid computation clips logits to [-500, 500] to avoid overflow in exp.
- Weight initialization: Xavier initialization (normal with std = sqrt(2 / (n_features + 1))).
- Three optimizer modes: full-batch gradient descent, mini-batch SGD, and pure SGD (batch_size=1).
- Three regularization modes: L2 (Ridge, in-gradient), L1 (Lasso via proximal soft-thresholding), and Elastic Net (combined L2 in-gradient + L1 proximal, mixed via l1_ratio).
- Class weighting: multiply each sample's loss contribution by (N / (n_classes * n_k)) where n_k is the count of that sample's class.
- Internal stratified 85/15 train/validation split for early stopping with best-weight restoration.
- Loss tracked at every epoch for convergence plotting.

**Hyperparameter Experiments (57 total runs):**

| Hyperparameter                   | Values Tested                                        | Method | What It Demonstrates                                    |
| -------------------------------- | ---------------------------------------------------- | ------ | ------------------------------------------------------- |
| Learning Rate                    | 0.001, 0.01, 0.1                                     | 5-fold CV | Too high diverges, too low converges slowly             |
| Regularization Strength (lambda) | 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0                 | 5-fold CV | Bias-variance tradeoff; underfitting at high lambda     |
| Regularization Type              | L2, L1, Elastic Net (5 l1_ratios for EN)             | 5-fold CV | L1 produces sparse weights (implicit feature selection) |
| Batch Size                       | 1 (SGD), 32, 64, full-batch                          | 5-fold CV | Noise vs. convergence speed tradeoff                    |
| Class Imbalance Strategy         | none, class_weight, SMOTE, undersample (×2 candidates) | Test set | Accuracy paradox; recall vs. F1 tradeoff               |

**Unique Analyses:**

- Convergence curves: two-panel layout (early epochs + full range) overlaying loss vs. epoch for all batch sizes. Demonstrates that SGD is noisier and stops earlier (patience exhausted by validation wobble), mini-batch converges quickly and smoothly, full-batch is slowest per epoch.
- Regularization path: two-panel figure showing weight L2 norm and sparsity fraction vs. lambda for L1, L2, and Elastic Net. Demonstrates L1 sparsity promotion, L2 smooth shrinkage, and Elastic Net interpolation.
- ROC and PR curves: four imbalance strategies compared. ROC curves cluster tightly (AUC 0.817–0.821), showing similar ranking quality; PR curves reveal more separation and are more relevant for the imbalanced task.
- Imbalance comparison bar chart: makes the accuracy paradox visually explicit — `none` wins on macro F1 and accuracy but catches only 43.5% of actual dropouts.

**Key Findings:**

- lr=0.1 wins clearly (well-conditioned loss surface)
- L1, lambda=0.001 is best regularization, but top 6 configs within 0.0005 F1 (dataset prefers weak regularization)
- batch_size=1 won CV but batch_size=32 dominated on test set across all imbalance strategies (CV variance artifact)
- class_weight is the best compromise for dropout detection; undersample maximizes recall

---

### 3.2 Gradient Boosted Trees

**Owner: Shivnarain Sarin**

**Theory:**

Gradient Boosting builds an additive model in a forward stage-wise manner. At each round t, a new weak learner (shallow decision tree) is fit to the negative gradient of the loss function with respect to the current model's predictions. For binary classification with log-loss, the negative gradient (pseudo-residual) for sample i is:

```
r_i = y_i - p_i
```

where y_i is the binary label and p_i is the current model's predicted probability.

**Implementation Details:**

- Wraps XGBoost's `XGBClassifier` with a custom feature-augmentation pipeline.
- Feature augmentation (`_augment()`): appends KMeans cluster ID, PCA components (95% variance), and centroid distances to the scaled input — expands 18 features to ~35.
- Imbalance handling built into `fit()`: partial SMOTE (configurable `smote_ratio`, default 0.4) followed by Tomek link cleaning, applied in the augmented feature space. Uses imblearn's SMOTE and TomekLinks (not the shared custom SMOTE).
- Decision threshold tuned to 0.33 (below the default 0.5) to improve minority-class recall.
- Best configuration from iterative experimentation (Iter 8): learning_rate=0.05, n_estimators=100, max_depth=7, subsample=0.8, min_child_weight=5, reg_alpha=0.5, reg_lambda=0.5.

**Hyperparameter Experiments:**

| Hyperparameter             | Values Tested                | Method   | What It Demonstrates                                                |
| -------------------------- | ---------------------------- | -------- | ------------------------------------------------------------------- |
| Decision Threshold         | 0.20–0.60 (step 0.05)       | Test set | Precision-recall tradeoff for minority class                        |
| Learning Rate (shrinkage)  | 0.01, 0.05, 0.1, 0.3        | Test set | Lower = more rounds needed but often better generalization          |
| Max Depth of Weak Learners | 3, 5, 7                      | Test set | Higher depth captures interactions                                  |
| Subsample Rate             | 0.5, 0.7, 0.8, 1.0          | Test set | Stochastic boosting; regularization effect                          |

**Unique Analyses:**

- Feature importance: XGBoost gain-based importances across all ~35 augmented features, showing which original features and which engineered features (cluster IDs, PCA components, centroid distances) contribute most.
- Threshold sensitivity: how precision, recall, and F1 for the dropout class change as the decision threshold varies from 0.20 to 0.60 — justifies the 0.33 choice.
- Confidence histogram: predicted probability distributions separated by true class, showing model calibration.
- SHAP summary plot: feature-level contribution analysis.

---

### 3.3 Neural Network

**Owner: Vandit Vasa**

**Theory:**

A feedforward neural network learns a hierarchical representation of the input by composing linear transformations with nonlinear activation functions. For binary classification, the output layer applies sigmoid to produce a probability for the positive class (Dropped Out). The model is trained end-to-end by minimizing binary cross-entropy loss via backpropagation and gradient descent.

**Implementation Details:**

- Architecture: fully connected layers with configurable depth and width via `hidden_dims` list.
- Activation functions: ReLU, Leaky ReLU (configurable alpha), Tanh, Sigmoid — per-layer configurable.
- Weight initialization: Xavier/Glorot or He initialization (selectable).
- Backpropagation: gradients computed analytically using the chain rule; no autograd libraries.
- Optimizers: Adam (with bias correction, configurable beta1/beta2) and SGD with momentum.
- Learning rate decay: step decay and exponential decay schedules.
- Regularization: combined L1 + L2 weight penalty (applied in gradient); inverted dropout during training.
- Early stopping: monitor validation loss; stop if no improvement for `patience` epochs and revert to best weights.
- Class weighting: multiply each sample's loss contribution by (N / (n_classes * n_k)).
- `fit()` accepts optional `X_val`/`y_val` and `epoch_callback` for live loss plotting in the dashboard.
- `get_layer_activations(X)`: returns per-layer post-activation outputs for neuron health analysis.

**Hyperparameter Experiments:**

| Hyperparameter        | Values Tested                | Method   | What It Demonstrates                                        |
| --------------------- | ---------------------------- | -------- | ----------------------------------------------------------- |
| Hidden Layer Depth    | [32], [64,32], [128,64,32]   | 5-fold CV | Deeper networks capture more complex interactions           |
| Hidden Layer Width    | 32, 64, 128, 256 units       | 5-fold CV | Wider layers increase capacity; risk of overfitting         |
| Learning Rate         | 0.001, 0.01, 0.1             | 5-fold CV | Too high diverges; too low converges slowly                 |
| L2 Regularization (λ) | 0.0001, 0.001, 0.01, 0.1     | 5-fold CV | Bias-variance tradeoff; high λ underfits                    |
| Dropout Rate          | 0.0, 0.2, 0.5                | 5-fold CV | Dropout as regularization; reduces co-adaptation of neurons |
| Class Imbalance Strategy | none, class_weight, SMOTE, undersample | Test set | Recall vs. F1 tradeoff for minority class              |

**Unique Analyses:**

- Live training convergence: real-time loss curve updates during training via `epoch_callback` in the dashboard.
- Neuron health analysis: per-layer detection of dead neurons (ReLU output = 0), negative activations (Leaky ReLU), or saturated neurons (Tanh/Sigmoid near extremes). Color-coded bar chart with green/orange/red thresholds.
- Automated recommendations: the dashboard diagnoses overfitting (val/train gap), convergence issues (too-fast stopping, max epochs reached, loss barely decreased), and minority-class performance (AUC-PR near baseline), generating actionable suggestions.
- Confidence histogram and PR curve: post-training visualization of prediction distribution by true class and precision-recall tradeoff.

## 4. Preprocessing Pipeline

### 4.1 Data Splitting

- Stratified 80/20 train/test split (matching the original paper's setup).
- All hyperparameter selection uses 5-fold stratified cross-validation on the training set only (LR and NN sweeps). GBT sweeps evaluate on the held-out test set.
- Random seed is fixed (seed=42) for reproducibility across all splits.

### 4.2 Feature Scaling

**Standardization (z-score normalization):**

- For each feature: x_scaled = (x - mean) / std
- Fit the scaler on the training set only. Apply the same mean/std to the test set. This prevents data leakage.
- Constant-feature safeguard: columns with std=0 have their std replaced with 1 to avoid division by zero.
- Standardization is the default for all experiments.

### 4.3 Class Imbalance Handling

Four strategies, compared in dedicated ablations for LR and NN:

| Strategy                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| None (baseline)                         | Train on the raw imbalanced data.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Class-weighted loss                     | Multiply each sample's loss by (N / (n_classes * n_k)). Applied directly in the loss function for logistic regression and neural network.                                                                                                                                                                                                                                                                                                          |
| SMOTE (Synthetic Minority Oversampling) | Generate synthetic samples for minority classes by interpolating between existing minority samples and their nearest neighbors. Implemented from scratch: for each minority sample, find its 5 nearest same-class neighbors, pick one at random, create a new sample at a random point along the line between them. Oversample until all classes have the same count as the majority class. Applied to the training set only (never the test set). |
| Random Undersampling                    | Randomly remove majority class samples until all classes have the same count as the minority class.                                                                                                                                                                                                                                                                                                                                                |

GBT uses a different imbalance pipeline: partial SMOTE (via imblearn, targeting a configurable minority:majority ratio) followed by Tomek link cleaning, applied in the augmented feature space.

### 4.4 Dimensionality Reduction (PCA)

- Implemented from scratch: center the data, compute the covariance matrix, eigendecompose, project onto the top-k eigenvectors.
- Threshold: retain enough components to explain 95% of total variance.
- Fit PCA on the training set, apply the same projection to the test set.
- Used internally by GBT's feature augmentation pipeline (via scikit-learn's PCA). The shared from-scratch PCA is available but not used in the current experiment sweeps for LR or NN.

### 4.5 Preprocessing Order

```
Raw Data
  -> Drop non-feature columns (Student_ID)
  -> Separate target (Dropout)
  -> Impute missing values (median for numeric, mode for categorical)
  -> Label-encode 7 categorical features
  -> Convert to numeric matrix
  -> Train/Test Split (stratified 80/20)
  -> Standardization (fit on train, transform both)
  -> SMOTE / Undersampling (train only, if enabled)
  -> PCA (fit on train, transform both, if enabled)
  -> Feed to model
```

## 5. Evaluation Framework

### 5.1 Primary Metrics

| Metric              | Why                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Macro F1 Score      | Primary metric. Averages F1 across both classes equally, so it is not dominated by the majority class.             |
| Per-Class Precision | Shows where the model is making false positives.                                                                   |
| Per-Class Recall    | Shows where the model is missing true positives. Critical for the Dropped Out class (students at risk of leaving the program). |
| Overall Accuracy    | Reported for completeness but NOT used for model selection (misleading under imbalance).                           |
| Confusion Matrix    | 2x2 matrix visualized as a heatmap. Reveals systematic misclassification patterns.                                 |

All metrics are implemented from scratch (no scikit-learn metric dependencies).

### 5.2 ROC Curves and AUC

- Compute ROC curve by sweeping all unique predicted probability thresholds.
- AUC computed via trapezoid rule (`np.trapezoid`).
- Used to evaluate ranking quality of probability estimates across all imbalance strategies.

### 5.3 Precision-Recall Curves and AUC-PR

- Compute PR curve focused on the dropout (positive) class.
- More relevant than ROC for imbalanced tasks — shows how well the model retrieves minority-class samples.
- Class-prevalence baseline plotted for reference.

### 5.4 Cross-Validation Protocol

- 5-fold stratified cross-validation on the training set for hyperparameter selection (LR and NN).
- Custom stratified fold construction: per-class shuffled indices distributed round-robin across folds.
- Report mean and standard deviation of macro F1 and accuracy across folds.
- GBT experiments evaluate on the held-out test set rather than CV (best configuration was identified through prior iterative experimentation).

## 6. Dashboard

### 6.1 Overview

An interactive Streamlit application (`src/dashboard/app.py`) that allows users to configure, train, and evaluate all three models through a web interface.

### 6.2 Shared Features (All Models)

- Sidebar hyperparameter controls with sliders and select boxes
- One-click "Train & Evaluate" button
- Confusion matrix heatmap
- ROC curve with AUC
- Per-class precision, recall, F1 display
- Top-level metric cards: Macro F1, Accuracy, ROC AUC

### 6.3 Logistic Regression Panel

- Controls: learning rate, lambda, regularization type (L2/L1/Elastic Net), batch size, class weighting toggle

### 6.4 Gradient Boosted Trees Panel

- Controls: learning rate, boosting rounds, max depth, subsample rate, decision threshold
- Feature importance bar chart (top 15 augmented features)
- Confidence histogram (predicted probability by true class)
- Post-training threshold explorer: slider that adjusts the decision threshold and updates dropout precision/recall/F1/accuracy without retraining

### 6.5 Neural Network Panel

- **Architecture builder**: configurable number of hidden layers (1–5), per-layer width (4–512) and activation function, weight initialization strategy (Xavier/He)
- **Optimizer config**: Adam or SGD, learning rate (1e-5 to 1.0), momentum/beta1/beta2, LR decay (none/step/exponential)
- **Regularization**: L1 lambda, L2 lambda, dropout rate
- **Training**: batch size, max epochs, patience, class weighting toggle
- **Live training**: loss curve updates in real time during training via epoch callback
- **Post-training analysis**:
  - Training summary: epochs trained, parameter count, final train/val loss, generalization gap percentage
  - Decision threshold slider with live metric updates
  - Confidence histogram and PR curve
  - Neuron health: per-layer dead/saturated neuron detection with color-coded bars (green < 20%, orange < 50%, red > 50%)
  - Automated recommendations: diagnoses overfitting, convergence issues, loss stagnation, poor minority-class performance, and dead neurons with actionable suggestions

## 7. Experiment Logging

All experiments are automatically logged to `experiments/{model_name}/log.jsonl` in append mode. Each JSON record contains:

- Timestamp
- Full parameter dictionary
- Metric dictionary (per-class and macro)
- Optional `extra` metadata (sweep type, imbalance strategy, split type)

The `extra` field serves as the filter key for separating experiment groups when analyzing results. Delete `log.jsonl` before re-running experiments to avoid duplicates.
