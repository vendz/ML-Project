from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

RANDOM_SEED = 42

TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "Target"          # column name in the raw CSV
POSITIVE_CLASS = "Graduate"    # encoded as 1
NEGATIVE_CLASS = "Enrolled"    # encoded as 0
