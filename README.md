# Predicting Student Academic Outcomes

A comparative study of classification methods for predicting student dropout, implemented from scratch in Python with an interactive Streamlit dashboard.

**Course:** CS6140 - Machine Learning | Northeastern University
**Team:** Vandit Vasa, Brandon Tran, Shivnarain Sarin

## Problem

Binary classification: predict whether a student will **drop out** or be **retained** based on demographic, academic, behavioral, and resource-access features.

- ~10,000 instances, 18 features (excluding Student_ID)
- Imbalanced: ~76.5% Retained, ~23.5% Dropped Out
- ~5% missing values in some columns (Family_Income, Study_Hours_per_Day, Stress_Index, Parental_Education)

## Models

| Model | Owner |
| --- | --- |
| Logistic Regression | Brandon Tran |
| Gradient Boosted Trees | Shivnarain Sarin |
| Neural Network | Vandit Vasa |

## Project Structure

```
src/
├── shared/                  # shared by everyone — do not modify without coordinating
│   ├── config.py            # random seed, paths, target labels
│   ├── base_model.py        # abstract interface all models implement
│   ├── preprocessing.py     # StandardScaler, SMOTE, PCA, load_data()
│   └── evaluation.py        # metrics, CV, ROC, experiment logger
├── logistic_regression/     # Brandon
│   ├── model.py
│   └── experiments.py
├── gradient_boosting/       # Shiv
│   ├── model.py
│   └── experiments.py
├── neural_network/          # Vandit
│   ├── model.py
│   └── experiments.py
└── dashboard/               # Shiv
    ├── app.py
    └── components/plots.py
data/
└── raw/                     # place dataset.csv here (gitignored)
experiments/                 # auto-generated experiment logs (gitignored)
results/                     # final figures and tables
notebooks/                   # exploratory notebooks
```

## Quick Start

```bash
# Install dependencies
uv sync

# Place the dataset
# Download from: https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset
# Save as: data/raw/dataset.csv

# Run experiments (from src/)
cd src
uv run python -m logistic_regression.experiments   # Brandon
uv run python -m gradient_boosting.experiments     # Shiv
uv run python -m neural_network.experiments        # Vandit

# Launch the dashboard (from src/)
uv run streamlit run dashboard/app.py
```

## Development Rules

- **Work in your own folder.** `src/logistic_regression/`, `src/gradient_boosting/`, and `src/neural_network/` are each one person's territory.
- **Never edit `src/shared/` unilaterally.** Changes there affect everyone — coordinate first.
- **Use `shared.config.RANDOM_SEED`** everywhere. Never hardcode seeds or paths.
- **Your model must subclass `BaseModel`** (`fit` / `predict` / `predict_proba`). This is what the dashboard calls.
- **Log every experiment** via `shared.evaluation.log_experiment()`. Results land in `experiments/<model>/log.jsonl`.
- **Run from `src/`** so that `from shared.x import y` resolves correctly.
