"""Tune, track (MLflow), evaluate and save the best engine-condition model (job 3 of the pipeline)."""
import hashlib
import json
import logging
import os
import time
import warnings

os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "ERROR")   # keep MLflow/alembic info messages out of the logs
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

for name in ("mlflow", "alembic"):
    logging.getLogger(name).setLevel(logging.ERROR)

SPLIT_DIR = Path("splits")                                    # downloaded "data-splits" artifact
DEPLOY_DIR = Path("predictive_maintenance/deployment")        # committed to the repository by the pipeline
REPORT_DIR = Path("predictive_maintenance/reports")
TARGET = "engine_condition"
RANDOM_STATE = 42
THRESHOLD = 0.5
SCORING = {"roc_auc": "roc_auc", "recall": "recall", "precision": "precision", "f1": "f1", "accuracy": "accuracy"}

MODELS = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                     {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [1, 20, 50]}),
    "Bagging": (BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=20, random_state=RANDOM_STATE),
                                  random_state=RANDOM_STATE, n_jobs=-1),
                {"n_estimators": [50, 100], "max_samples": [0.5, 0.8]}),
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                     {"n_estimators": [100, 200], "max_depth": [5, 8, None], "min_samples_leaf": [1, 10]}),
    "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE),
                 {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.5]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                         {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", subsample=0.8, n_jobs=-1),
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]}),
}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"test_accuracy": accuracy_score(y, pred), "test_precision": precision_score(y, pred),
            "test_recall": recall_score(y, pred), "test_f1": f1_score(y, pred), "test_roc_auc": roc_auc_score(y, proba),
            "test_tn": int(tn), "test_fp": int(fp), "test_fn": int(fn), "test_tp": int(tp)}


# ---- 1. Load the train/test data from the workflow artifact (checksums verified) ----
manifest = json.loads((SPLIT_DIR / "manifest.json").read_text())
for name, info in manifest["files"].items():
    assert md5(SPLIT_DIR / name) == info["md5"], f"{name}: checksum mismatch - artifact corrupted"
    print(f"{name:<10} rows={info['rows']:,}  md5={info['md5']}  [verified]")
train, test = pd.read_csv(SPLIT_DIR / "train.csv"), pd.read_csv(SPLIT_DIR / "test.csv")
X_tr, y_tr = train.drop(columns=TARGET), train[TARGET]
X_te, y_te = test.drop(columns=TARGET), test[TARGET]
print(f"Train {X_tr.shape} | Test {X_te.shape}")

# ---- 2. MLflow experiment tracking (SQLite backend; uploaded as a CI artifact) -------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("engine-predictive-maintenance")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

base = np.ones_like(y_te)
baseline = {"model": "Baseline (always 1)", "cv_roc_auc": 0.5, "test_accuracy": accuracy_score(y_te, base),
            "test_precision": precision_score(y_te, base), "test_recall": 1.0, "test_f1": f1_score(y_te, base),
            "test_roc_auc": 0.5}
with mlflow.start_run(run_name="Baseline_always_maintenance"):
    mlflow.log_param("model_type", "Baseline")
    mlflow.log_metrics({k: v for k, v in baseline.items() if k != "model"})

# ---- 3. Tune every model and log EVERY tuned configuration ---------------------------
results, fitted, all_cv = [], {}, []
t_start = time.time()
for name, (estimator, grid) in MODELS.items():
    t0 = time.time()
    with mlflow.start_run(run_name=f"{name}_gridsearch") as parent:
        mlflow.log_params({"model_type": name, "cv_folds": 5, "refit_metric": "roc_auc"})
        gs = GridSearchCV(estimator, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        cvr = pd.DataFrame(gs.cv_results_)
        for i, row in cvr.iterrows():
            with mlflow.start_run(run_name=f"{name}_cfg{i:02d}", nested=True):
                mlflow.log_param("model_type", name)
                mlflow.log_params(row["params"])
                mlflow.log_metrics({f"cv_{m}_mean": row[f"mean_test_{m}"] for m in SCORING})
                mlflow.log_metrics({f"cv_{m}_std": row[f"std_test_{m}"] for m in SCORING})
        bi = gs.best_index_
        tm = test_metrics(gs.best_estimator_, X_te, y_te)
        summary = {"model": name, "cv_roc_auc": cvr.loc[bi, "mean_test_roc_auc"],
                   "cv_roc_auc_std": cvr.loc[bi, "std_test_roc_auc"], "cv_recall": cvr.loc[bi, "mean_test_recall"],
                   "cv_precision": cvr.loc[bi, "mean_test_precision"], "cv_f1": cvr.loc[bi, "mean_test_f1"], **tm,
                   "n_configs": len(cvr), "fit_seconds": round(time.time() - t0, 1),
                   "best_params": json.dumps(gs.best_params_)}
        mlflow.log_params({f"best_{k}": v for k, v in gs.best_params_.items()})
        mlflow.log_metrics({k: float(v) for k, v in summary.items()
                            if isinstance(v, (int, float, np.floating, np.integer)) and k != "n_configs"})
        results.append(summary)
        fitted[name] = (gs.best_estimator_, gs.best_params_, parent.info.run_id)
        all_cv.append(pd.DataFrame({"model": name, "params": cvr["params"].map(json.dumps),
                                    **{f"cv_{m}_mean": cvr[f"mean_test_{m}"] for m in SCORING},
                                    "cv_roc_auc_std": cvr["std_test_roc_auc"]}))
        print(f"{name:<17} configs={len(cvr):>2}  CV ROC-AUC={summary['cv_roc_auc']:.4f}  "
              f"test ROC-AUC={tm['test_roc_auc']:.4f}  recall={tm['test_recall']:.4f}  "
              f"precision={tm['test_precision']:.4f}  F1={tm['test_f1']:.4f}  ({summary['fit_seconds']}s)")
        print(f"{'':<17} best params: {gs.best_params_}")
print(f"Total tuning time: {time.time() - t_start:.0f} s")

# ---- 4. Select and save the best model (committed to the repository by the pipeline) --
comp = pd.DataFrame([baseline] + results).set_index("model").sort_values("cv_roc_auc", ascending=False)
best_name = comp.drop(index="Baseline (always 1)").index[0]
best_model, best_params, run_id = fitted[best_name]
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, DEPLOY_DIR / "best_model.joblib")
comp.to_csv(REPORT_DIR / "model_comparison.csv")
pd.concat(all_cv, ignore_index=True).to_csv(REPORT_DIR / "cv_results.csv", index=False)

row = comp.loc[best_name]
in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
meta = {"model_name": best_name, "selection_metric": "cv_roc_auc", "best_params": best_params,
        "cv_roc_auc": round(float(row["cv_roc_auc"]), 4),
        "test_metrics": {k: round(float(row[k]), 4) for k in
                         ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]},
        "decision_threshold": THRESHOLD, "features": list(X_tr.columns), "target": TARGET,
        "mlflow_run_id": run_id, "train_rows": len(X_tr), "test_rows": len(X_te),
        "data_lineage": {"dataset_md5": manifest["source_dataset_md5"],
                         "train_md5": manifest["files"]["train.csv"]["md5"],
                         "test_md5": manifest["files"]["test.csv"]["md5"]},
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                             "pandas": pd.__version__},
        "trained_by": "GitHub Actions" if in_ci else "local notebook",
        "github_run_number": os.environ.get("GITHUB_RUN_NUMBER"), "github_sha": os.environ.get("GITHUB_SHA"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
(DEPLOY_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
with mlflow.start_run(run_id=run_id):
    mlflow.set_tag("selected_best_model", "true")
    mlflow.sklearn.log_model(best_model, name="model", serialization_format="cloudpickle",
                             input_example=X_tr.head(3))
print(f"\nBEST MODEL: {best_name} (CV ROC-AUC {row['cv_roc_auc']:.4f}) -> {DEPLOY_DIR / 'best_model.joblib'}")
print(f"Trained by: {meta['trained_by']} | MLflow run ID: {run_id}")
