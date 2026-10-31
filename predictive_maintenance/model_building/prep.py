"""Clean the registered dataset and create train/test splits (job 2 of the GitHub Actions pipeline)."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("predictive_maintenance/data/engine_data.csv")   # repository data folder
SPLIT_DIR = Path("splits")                                          # uploaded as the "data-splits" artifact
TARGET = "engine_condition"
RANDOM_STATE = 42
RENAME = {"Engine rpm": "engine_rpm", "Lub oil pressure": "lub_oil_pressure", "Fuel pressure": "fuel_pressure",
          "Coolant pressure": "coolant_pressure", "lub oil temp": "lub_oil_temp", "Coolant temp": "coolant_temp",
          "Engine Condition": "engine_condition"}
# Physically plausible limits: (exclusive lower bound, inclusive upper bound)
SENSOR_LIMITS = {"engine_rpm": (0, 5000), "lub_oil_pressure": (0, 15), "fuel_pressure": (0, 30),
                 "coolant_pressure": (0, 10), "lub_oil_temp": (0, 150), "coolant_temp": (0, 150)}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


data = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {data.shape[0]:,} rows x {data.shape[1]} columns")

# 1. Standardise column names
data = data.rename(columns=RENAME)

# 2. Remove unnecessary columns (index/ID-like and constant columns)
drop_cols = [c for c in data.columns if c.lower().startswith("unnamed") or c.lower() in {"id", "index"}]
drop_cols += [c for c in data.columns if data[c].nunique() <= 1 and c != TARGET]
data = data.drop(columns=drop_cols)
print(f"Unnecessary columns removed: {drop_cols if drop_cols else 'none'}")

# 3. Duplicates and missing values
n_dup, n_na = int(data.duplicated().sum()), int(data.isna().any(axis=1).sum())
data = data.drop_duplicates().dropna()
print(f"Duplicate rows removed: {n_dup} | rows with missing values removed: {n_na}")

# 4. Physically implausible sensor readings
mask, removed = pd.Series(True, index=data.index), {}
for col, (lo, hi) in SENSOR_LIMITS.items():
    bad = ~data[col].between(lo, hi, inclusive="right")
    removed[col] = int(bad.sum())
    mask &= ~bad
data = data[mask].reset_index(drop=True)
print(f"Implausible readings removed per sensor: {removed}")
print(f"Clean dataset: {data.shape[0]:,} rows x {data.shape[1]} columns")

# 5. Stratified train/test split
X, y = data.drop(columns=TARGET), data[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# 6. Save locally (the workflow uploads this folder as the "data-splits" artifact)
SPLIT_DIR.mkdir(exist_ok=True)
X_tr.assign(**{TARGET: y_tr}).to_csv(SPLIT_DIR / "train.csv", index=False)
X_te.assign(**{TARGET: y_te}).to_csv(SPLIT_DIR / "test.csv", index=False)
manifest = {"artifact": "data-splits", "source_dataset_md5": md5(DATA_PATH), "features": list(X.columns),
            "target": TARGET, "dropped_columns": drop_cols, "implausible_rows_removed": removed,
            "clean_rows": int(len(data)),
            "files": {name: {"rows": int(len(pd.read_csv(SPLIT_DIR / name))), "md5": md5(SPLIT_DIR / name)}
                      for name in ["train.csv", "test.csv"]}}
(SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Train: {len(X_tr):,} rows (maintenance rate {y_tr.mean():.2%}) -> {SPLIT_DIR / 'train.csv'}")
print(f"Test : {len(X_te):,} rows (maintenance rate {y_te.mean():.2%}) -> {SPLIT_DIR / 'test.csv'}")
print("Manifest:", json.dumps(manifest["files"]))
