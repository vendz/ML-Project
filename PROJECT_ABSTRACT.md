# Predicting Student Academic Outcomes: A Comparative Study of Classification Methods

## Project Documentation

**Group Members:** Vandit Vasa, Brandon Tran, Shivnarain Sarin<br>
**Course:** CS6140 - Machine Learning<br>
**Institution:** Northeastern University

## 1. Project Overview

### 1.1 Objective

Predict whether a university student will **drop out**, remain **enrolled**, or **graduate** based on demographic, socio-economic, and academic features. This is a three class classification problem with notable class imbalance.

### 1.2 Core Philosophy

This project is not about finding the single best model. It is a **deep comparative study** where every model is thoroughly dissected, every hyperparameter is systematically varied, and every design decision is ablated. The goal is to demonstrate mastery of the underlying mechanics of each algorithm by showing exactly how and why tweaking specific knobs changes behavior.

### 1.3 Key Highlights & Differentiators

- A **Streamlit interactive dashboard** lets users manipulate hyperparameters via sliders and see results update visually in real time.
- A **"Pin & Compare" system** allows side-by-side comparison of any configurations.
- Advanced analyses (bias-variance decomposition, calibration, statistical significance testing, feature interactions, misclassification analysis)
- Every experiment is logged automatically, producing the ablation tables for the final report directly from the dashboard.

### 1.4 Division of Work

- **Vandit Vasa:** Lead the implementation of **Neural Network**, including architecture design, activation functions, backpropagation, regularization, and hyperparameter tuning. Also take primary ownership of the preprocessing components most tied to this model, especially feature scaling, PCA, and the standardized vs raw / PCA ablation studies.
- **Brandon Tran:** Lead the implementation of **Multinomial Logistic Regression**, including optimization (full-batch, mini-batch, SGD), regularization (L1, L2, Elastic Net), convergence analysis, and probability outputs. Also own the class imbalance experiments tied to model performance, including class weighting comparisons, SMOTE, and undersampling.
- **Shivnarain Sarin:** Lead the implementation of **Gradient Boosted Trees**, including split criteria, residual fitting, learning rate scheduling, feature importance analysis, and the Streamlit dashboard / experiment logging pipeline.
- **Shared responsibilities:** All three members will collaborate on dataset understanding, evaluation metrics, cross-validation, statistical significance testing, final error analysis, and writing the final report/presentation. Final model comparison, interpretation of results, and polishing the narrative of the project will be done jointly so that every member can speak to both the implementation and the experimental findings.

## 2. Dataset

### 2.1 Source

UCI Machine Learning Repository / Kaggle: "Predict Students' Dropout and Academic Success"

- **URL:** https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset
- **Original Paper:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021)

### 2.2 Summary

| Property           | Value                                                       |
| ------------------ | ----------------------------------------------------------- |
| Instances          | ~4,424                                                      |
| Features           | 37                                                          |
| Target Classes     | 2 (Enrolled, Graduate)                                      |
| Class Distribution | Imbalanced (Graduate is the majority class)                 |
| Feature Types      | Mix of integer-encoded categoricals and continuous numerics |
| Missing Values     | None (pre-cleaned by original authors)                      |

### 2.3 Feature Groups

Features are organized into four logical groups. This grouping is critical for the feature subset ablation experiments.

**Academic Features (semester performance):**
Curricular units credited, enrolled, evaluated, approved, grade, and without evaluations for both 1st and 2nd semesters (12 features total). These capture the student's actual academic trajectory.

**Socio-Economic Features:**
Scholarship holder, tuition fees up to date, debtor status, mother's qualification, father's qualification, mother's occupation, father's occupation, unemployment rate, inflation rate, GDP (10 features). These capture external economic pressures and family background.

**Demographic Features:**
Marital status, application mode, application order, course, daytime/evening attendance, previous qualification, previous qualification grade, nationality, mother's/father's qualification, displaced, gender, age at enrollment, international (approximately 10 features depending on exact grouping). These capture who the student is at the time of enrollment.

**Macroeconomic Indicators:**
Unemployment rate, inflation rate, GDP (3 features). These are shared with the socio-economic group but can be isolated to test whether external economic conditions alone have predictive power.

### 2.4 Target Variable

| Class    | Description                                                            |
| -------- | ---------------------------------------------------------------------- |
| Enrolled | Student is still in the program (has not yet graduated or dropped out) |
| Graduate | Student successfully completed the program                             |

## 3. Models: Theory, Implementation & Experiments

### 3.1 Multinomial Logistic Regression

**Owner: Brandon Tran**

**Theory:**

Logistic regression models the probability of class k given input x using the softmax function:

```
P(y=k | x) = exp(w_k . x + b_k) / sum_j(exp(w_j . x + b_j))
```

The model learns weight vectors w_k and biases b_k for each class by minimizing the cross-entropy loss:

```
L = -(1/N) * sum_i( sum_k( y_ik * log(P(y=k | x_i)) ) ) + lambda * R(w)
```

where R(w) is the regularization term.

**Implementation Details:**

- Softmax computation must use the log-sum-exp trick for numerical stability: subtract max(logits) before exponentiating.
- Weight initialization: Xavier initialization (normal with std = sqrt(2 / (n_features + n_classes))).
- Support three optimizers: full-batch gradient descent, mini-batch SGD, and pure SGD (batch_size=1).
- Support three regularization modes: L2 (Ridge), L1 (Lasso via proximal gradient), and Elastic Net.
- Class weighting: multiply each sample's loss contribution by (N / (n_classes \* n_k)) where n_k is the count of that sample's class. This upweights minority classes.
- Track and return the loss at every epoch for convergence plotting.

**Hyperparameter Experiments:**

| Hyperparameter                   | Values to Test                                      | What It Demonstrates                                    |
| -------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| Learning Rate                    | 0.001, 0.01, 0.1                                    | Too high diverges, too low converges slowly             |
| Regularization Strength (lambda) | 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0                 | Bias-variance tradeoff; underfitting at high lambda     |
| Regularization Type              | L2, L1, Elastic Net (alpha=0.5)                     | L1 produces sparse weights (implicit feature selection) |
| Batch Size                       | 1 (SGD), 32, 64, full-batch                         | Noise vs. convergence speed tradeoff                    |
| Max Epochs                       | 1000 (with early stopping on val loss, patience=10) | Prevents overfitting                                    |

**Unique Analyses for This Model:**

- Convergence curves: overlay loss vs. epoch for all batch sizes on one plot.
- Regularization path: plot magnitude of each weight coefficient as lambda varies from 0.0001 to 10. This shows which features get zeroed out first under L1.
- Inspect learned coefficients to see which features the linear model relies on most.

---

### 3.2 Gradient Boosted Trees

**Owner: Shivnarain Sarin**

**Theory:**

Gradient Boosting builds an additive model in a forward stage-wise manner. At each round t, a new weak learner (shallow decision tree) is fit to the negative gradient of the loss function with respect to the current model's predictions. For multiclass classification with cross-entropy loss, the negative gradient for class k and sample i is:

```
r_ik = y_ik - p_ik
```

where y_ik is the one-hot label and p_ik is the current model's predicted probability for class k. This means at each round, we fit K separate regression trees (one per class) to these residuals.

The update rule is:

```
F_k^(t)(x) = F_k^(t-1)(x) + learning_rate * h_k^t(x)
```

where h_k^t is the weak learner fit to residuals of class k at round t.

**Implementation Details:**

- Initialize F_k^(0) for each class to the log of the class prior probability.
- At each boosting round:
  1. Compute current probabilities via softmax over all F_k values.
  2. Compute residuals r_ik = y_ik - p_ik for each class.
  3. Fit a regression tree (using MSE on the residuals) for each class.
  4. Leaf values: for each leaf, use the Newton-Raphson step: sum(residuals) / sum(p_ik \* (1 - p_ik)) for that leaf's samples. This is the optimal leaf weight under the second-order approximation.
  5. Update: F_k += learning_rate \* tree_k predictions.
- Early stopping: after each round, compute validation loss. If it has not improved for `patience` rounds, stop and revert to the best round.
- Subsampling: optionally train each round's trees on a random fraction of the training data (stochastic gradient boosting).

**Hyperparameter Experiments:**

| Hyperparameter             | Values to Test               | What It Demonstrates                                                |
| -------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| Learning Rate (shrinkage)  | 0.01, 0.05, 0.1, 0.3         | Lower = more rounds needed but often better generalization          |
| Number of Rounds           | 50, 100, 200, 500            | Shows the learning rate / n_rounds tradeoff                         |
| Max Depth of Weak Learners | 1 (stumps), 3, 5, 7          | Depth 1 = purely additive model; higher depth captures interactions |
| Subsample Rate             | 0.5, 0.7, 1.0                | Stochastic boosting; regularization effect                          |
| Early Stopping Patience    | 10 rounds on validation loss | Prevents overfitting; shows optimal stopping point                  |

**Unique Analyses for This Model:**

- Learning rate vs. n_rounds tradeoff: create a heatmap where x-axis is learning rate, y-axis is n_rounds, and cell color is validation F1. This directly shows the classic relationship: lower learning rates need more rounds but reach higher performance.
- Staged prediction: plot validation loss after each boosting round for multiple learning rates on the same graph. Shows how faster learners overfit earlier.
- Feature importance: total gain (sum of impurity improvements) across all trees and all rounds per feature.
- Depth ablation: compare stumps (depth=1, purely additive) vs. depth=3 (captures pairwise interactions) vs. depth=5 (captures higher-order interactions). If depth=1 performs nearly as well as depth=5, the problem is mostly additive.

---

### 3.3 Neural Network

**Owner: Vandit Vasa**

**Theory:**

A feedforward neural network learns a hierarchical representation of the input by composing linear transformations with nonlinear activation functions. For multiclass classification, the output layer applies softmax to produce a probability distribution over classes. The model is trained end-to-end by minimizing cross-entropy loss via backpropagation and gradient descent.

**Implementation Details:**

- Architecture: fully connected layers with configurable depth and width.
- Activation functions: ReLU (hidden layers), Softmax (output layer).
- Weight initialization: Xavier/Glorot initialization to prevent vanishing/exploding gradients.
- Backpropagation: compute gradients analytically using the chain rule; no autograd libraries.
- Optimizers: mini-batch SGD with optional momentum.
- Regularization: L2 weight decay (applied to all weight matrices, not biases); dropout (randomly zero out hidden units during training with probability p).
- Early stopping: monitor validation loss; stop if no improvement for `patience` epochs and revert to best weights.
- Class weighting: multiply each sample's loss contribution by (N / (n_classes * n_k)) to upweight minority classes.
- Track training and validation loss at every epoch for convergence plotting.

**Hyperparameter Experiments:**

| Hyperparameter        | Values to Test               | What It Demonstrates                                        |
| --------------------- | ---------------------------- | ----------------------------------------------------------- |
| Hidden Layer Depth    | 1, 2, 3 layers               | Deeper networks capture more complex interactions           |
| Hidden Layer Width    | 32, 64, 128, 256 units       | Wider layers increase capacity; risk of overfitting         |
| Learning Rate         | 0.001, 0.01, 0.1             | Too high diverges; too low converges slowly                 |
| L2 Regularization (λ) | 0.0001, 0.001, 0.01, 0.1     | Bias-variance tradeoff; high λ underfits                    |
| Dropout Rate          | 0.0, 0.2, 0.5                | Dropout as regularization; reduces co-adaptation of neurons |
| Batch Size            | 32, 64, full-batch           | Noise vs. convergence speed tradeoff                        |
| Max Epochs            | 500 (early stopping, pat=10) | Prevents overfitting                                        |

**Unique Analyses for This Model:**

- Convergence curves: overlay training and validation loss vs. epoch for different learning rates and depths.
- Depth/width ablation: compare macro F1 across all architecture configurations to find the sweet spot.
- Dropout sensitivity: plot validation F1 vs. dropout rate to show the regularization effect.
- Weight norm evolution: track the L2 norm of weights per layer across epochs to visualize how regularization constrains learning.

## 4. Preprocessing Pipeline

### 4.1 Data Splitting

- Stratified 80/20 train/test split (matching the original paper's setup).
- All hyperparameter selection uses 5-fold stratified cross-validation on the training set only.
- Random seed is fixed (seed=42) for reproducibility across all splits.

### 4.2 Feature Scaling

**Standardization (z-score normalization):**

- For each feature: x_scaled = (x - mean) / std
- Fit the scaler on the training set only. Apply the same mean/std to the test set. This prevents data leakage.
- Standardization is the default for all experiments. One ablation compares standardized vs. raw features.

### 4.3 Class Imbalance Handling

Four strategies, compared in a dedicated ablation:

| Strategy                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| None (baseline)                         | Train on the raw imbalanced data.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Class-weighted loss                     | Multiply each sample's loss by (N / (n_classes \* n_k)). Applied directly in the loss function for logistic regression, gradient boosting, and neural network.                                                                                                                                                                                                                                                                                     |
| SMOTE (Synthetic Minority Oversampling) | Generate synthetic samples for minority classes by interpolating between existing minority samples and their nearest neighbors. Implemented from scratch: for each minority sample, find its 5 nearest same-class neighbors, pick one at random, create a new sample at a random point along the line between them. Oversample until all classes have the same count as the majority class. Applied to the training set only (never the test set). |
| Random Undersampling                    | Randomly remove majority class samples until all classes have the same count as the minority class.                                                                                                                                                                                                                                                                                                                                                |

### 4.4 Dimensionality Reduction (PCA)

- Implemented from scratch: center the data, compute the covariance matrix, eigendecompose, project onto the top-k eigenvectors.
- Threshold: retain enough components to explain 95% of total variance.
- Fit PCA on the training set, apply the same projection to the test set.
- PCA is togglable in the dashboard and tested with all models for completeness.

### 4.5 Preprocessing Order

```
Raw Data
  -> Train/Test Split (stratified)
  -> Standardization (fit on train, transform both)
  -> SMOTE / Undersampling (train only, if enabled)
  -> PCA (fit on train, transform both, if enabled)
  -> Feed to model
```

Standardization must come before SMOTE (so that interpolation happens in standardized space) and before PCA (so that covariance is computed on standardized features).

## 5. Evaluation Framework

### 5.1 Primary Metrics

| Metric              | Why                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Macro F1 Score      | Primary metric. Averages F1 across all three classes equally, so it is not dominated by the majority class.        |
| Per-Class Precision | Shows where the model is making false positives.                                                                   |
| Per-Class Recall    | Shows where the model is missing true positives. Critical for the Dropout class (we want high recall for dropout). |
| Overall Accuracy    | Reported for completeness but NOT used for model selection (misleading under imbalance).                           |
| Confusion Matrix    | 3x3 matrix visualized as a heatmap. Reveals systematic misclassification patterns.                                 |

### 5.2 ROC Curves and AUC

- Compute one-vs-rest ROC curves for each class.
- Plot all three curves (Dropout vs. rest, Enrolled vs. rest, Graduate vs. rest) on the same figure.
- Report per-class AUC and macro-averaged AUC.
- Uses predict_proba output, so it evaluates the ranking quality of probability estimates.

### 5.3 Statistical Significance

- For each pair of models (best configuration each), run a paired t-test across the 5 cross-validation fold scores.
- Report p-values in a significance matrix.
- A difference is considered significant at p < 0.05.
- This prevents over-interpreting small differences (e.g., 0.82 vs. 0.80 F1 may not be significant with only 5 folds).

### 5.4 Cross-Validation Protocol

- 5-fold stratified cross-validation on the training set for all hyperparameter selection.
- Report mean and standard deviation of each metric across folds.
- Final evaluation: retrain on the full training set with the best hyperparameters, evaluate on the held-out test set.

## 6. Advanced Analyses

### 6.1 Bias-Variance Decomposition (Learning Curves)

**What:** For each model (at two complexity levels), plot training error and validation error as a function of training set size.

**How:**

- Training set fractions: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%.
- For each fraction, subsample the training data (maintaining stratification), train the model, and evaluate on both the subsample (training error) and the full validation fold (validation error).
- Average over 5 cross-validation folds.
- Plot two curves: training error (should start low and increase) and validation error (should start high and decrease). The gap between them is the variance; the floor they converge to is the bias.

**Complexity levels to compare:**

- Logistic Regression: lambda=10 (high bias) vs. lambda=0.0001 (high variance)
- Random Forest: max_depth=3 (high bias) vs. max_depth=None (high variance)
- Gradient Boosting: 10 rounds, depth=1 (high bias) vs. 500 rounds, depth=7 (high variance)
- KNN: K=51 (high bias) vs. K=1 (high variance)

### 6.2 Convergence Analysis (Logistic Regression and Gradient Boosting)

**What:** Visualize how the training loss evolves during optimization.

**Logistic Regression convergence experiments:**

- Overlay loss vs. epoch for full-batch, mini-batch (32, 64), and SGD on one plot. Shows the noise/speed tradeoff.
- For a fixed batch size, overlay loss curves for different regularization strengths. Shows how regularization smooths the loss landscape.

**Gradient Boosting convergence experiments:**

- Plot training loss and validation loss vs. boosting round for multiple learning rates. Shows how lower learning rates converge more slowly but to a better optimum, and how higher learning rates overfit earlier (validation loss increases while training loss continues to decrease).

### 6.3 Decision Boundary Visualization

**What:** Project data onto the two most important features (expected: 1st and 2nd semester grades), plot the data points colored by true class, and overlay the decision boundaries for all four models.

**How:**

- Train each model on only two features.
- Create a dense meshgrid over the 2D feature space.
- For each point in the meshgrid, predict the class and color the background accordingly.
- Overlay the actual data points.
- Display all four models' decision boundaries in a 2x2 subplot grid.

**Expected findings:**

- Logistic Regression: straight-line boundaries.
- KNN: jagged, locally adaptive boundaries.
- Random Forest: axis-aligned step-function boundaries.
- Gradient Boosting: smoother axis-aligned boundaries (more rounds = finer granularity).

### 6.4 Feature Interaction Analysis

**What:** For the top-3 most important features (as determined by Random Forest/Gradient Boosting importance), create pairwise interaction heatmaps showing how feature combinations affect predicted dropout probability.

**How:**

- Take two features (e.g., "Scholarship holder" and "1st semester grade").
- Create a 2D grid of values spanning each feature's range.
- For each grid point, fix those two features and set all others to their training-set medians.
- Run predict_proba through the best model and plot the dropout probability as a heatmap.
- Repeat for all three pairwise combinations of the top-3 features.

**Why:** This directly tests whether nonlinear models capture compounding effects that a linear model cannot. If the heatmap shows interaction patterns (e.g., low grades + no scholarship = very high dropout, but low grades + scholarship = moderate dropout), that vindicates the project's hypothesis about feature interactions.

### 6.5 Misclassification Analysis

**What:** After training the best overall model, examine the test samples it got wrong and look for systematic patterns.

**How:**

- Pull all misclassified test samples.
- Group by confusion matrix cell (e.g., "Predicted Graduate, Actually Dropout" vs. "Predicted Enrolled, Actually Dropout").
- For each group, compute summary statistics of key features and compare to correctly classified samples.
- Identify whether certain student profiles are systematically harder (e.g., students with good 1st semester grades who drop out in the 2nd semester).

**Output:** A narrative explaining WHY the model fails where it does, supported by data tables and plots. This is the qualitative analysis that ties the quantitative results together.

### 6.6 Class Imbalance Sensitivity Study

**What:** Take the best model configuration and retrain it under all four imbalance strategies (none, class-weighted, SMOTE, undersampling). Compare per-class recall.

**Expected findings:**

- Without handling: high recall on Graduate (majority), low recall on Dropout and Enrolled.
- Class weighting: improved recall on minority classes, slight decrease on Graduate.
- SMOTE: best balance of per-class recalls, slight decrease in precision.
- Undersampling: improves minority recall but wastes data; highest variance.

**Output:** A single table with 4 rows (strategies) and columns for per-class precision, recall, F1, and macro F1.
