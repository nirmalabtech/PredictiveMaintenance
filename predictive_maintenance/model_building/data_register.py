"""Register and validate the engine sensor dataset (job 1 of the GitHub Actions pipeline)."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_PATH = Path("predictive_maintenance/data/engine_data.csv")
REGISTRY_PATH = Path("predictive_maintenance/data/registry.json")
TARGET = "Engine Condition"
EXPECTED_COLUMNS = {
    "Engine rpm": "Engine speed (revolutions per minute)",
    "Lub oil pressure": "Lubricating-oil pressure (bar)",
    "Fuel pressure": "Fuel supply pressure (bar)",
    "Coolant pressure": "Coolant pressure (bar)",
    "lub oil temp": "Lubricating-oil temperature (deg C)",
    "Coolant temp": "Coolant temperature (deg C)",
    "Engine Condition": "Target: 0 = normal, 1 = needs maintenance",
}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    if not DATA_PATH.exists():
        sys.exit(f"[FAIL] {DATA_PATH} not found")
    data = pd.read_csv(DATA_PATH)
    errors = []
    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    extra = [c for c in data.columns if c not in EXPECTED_COLUMNS]
    if missing:
        errors.append(f"Missing expected columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    non_numeric = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")
    if TARGET in data.columns and set(data[TARGET].dropna().unique()) - {0, 1}:
        errors.append("Target contains values other than 0/1")
    if data.empty:
        errors.append("Dataset is empty")

    print("=" * 64)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 64)
    print(f"File            : {DATA_PATH}")
    print(f"Rows x Columns  : {data.shape[0]:,} x {data.shape[1]}")
    print(f"Missing values  : {int(data.isna().sum().sum())}")
    print(f"Duplicate rows  : {int(data.duplicated().sum())}")
    print("\nColumn checks:")
    for col, desc in EXPECTED_COLUMNS.items():
        ok = col in data.columns
        dtype = str(data[col].dtype) if ok else "-"
        print(f"  [{'OK ' if ok else 'MISSING'}] {col:<18} {dtype:<8} {desc}")
    print("\nDescriptive statistics:")
    print(data.describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())
    if TARGET in data.columns:
        print("\nTarget distribution:")
        for k, v in data[TARGET].value_counts().sort_index().items():
            print(f"  {k}: {v:,} ({v / len(data):.2%})")

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    record = {"dataset": "engine_data", "path": str(DATA_PATH), "rows": int(len(data)),
              "columns": list(data.columns), "target": TARGET, "md5": md5(DATA_PATH),
              "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    REGISTRY_PATH.write_text(json.dumps(record, indent=2))
    print(f"\nVALIDATION PASSED - registry written to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
