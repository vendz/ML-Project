from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

RANDOM_SEED = 42

TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "Dropout"         # column name in the raw CSV (binary: 0 or 1)
DROP_COLS = ["Student_ID"]     # columns to exclude from features

CATEGORICAL_COLS = [
    "Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
    "Semester", "Department", "Parental_Education",
]
