"""
Shared configuration — single source of truth for all constants.
Everyone imports from here; never hardcode seeds or paths elsewhere.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data split ─────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
CV_FOLDS = 5

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_COL = "Target"          # column name in the raw CSV
POSITIVE_CLASS = "Graduate"    # encoded as 1
NEGATIVE_CLASS = "Enrolled"    # encoded as 0
