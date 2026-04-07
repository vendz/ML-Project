# Predicting Student Academic Outcomes

A comparative study of classification methods for predicting student dropout, implemented from scratch in Python with an interactive Streamlit dashboard.

## Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd student-dropout-prediction

# Install dependencies and manage environment
uv sync

# Download the dataset
# Place dataset.csv in the data/ directory
# Source: https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

# Launch the dashboard
uv run streamlit run app.py
```

## Data

Download the dataset from Kaggle and place it in `data/dataset.csv`:
https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

This is the "Student Dropout Prediction" dataset from Kaggle. ~10,000 instances, 18 features (excluding Student_ID), binary target (Dropout: 0 = Retained, 1 = Dropped Out). Class distribution is imbalanced (~76.5% Retained, ~23.5% Dropped Out). ~5% missing values in some columns (Family_Income, Study_Hours_per_Day, Stress_Index, Parental_Education).
