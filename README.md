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

This is the UCI "Predict Students' Dropout and Academic Success" dataset (~4,424 instances, 37 features, 3-class target). No missing values.
