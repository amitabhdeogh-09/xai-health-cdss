"""
Model training and cross-validation.

Run from the project root:
    python src/models.py

What it does
  1. Compares three baseline models with repeated stratified 5-fold CV
     (results saved to reports/cv_results.csv).
  2. Trains the FINAL model (XGBoost) for each disease on all the data and
     saves it to models/ so the dashboard can load it.

Each saved file is a "bundle": the full pipeline (cleaning + model) plus the
training data, which SHAP and DiCE need later as reference data.
"""
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.preprocessing import (ckd_preprocessor, heart_preprocessor, load_ckd,
                               load_heart)

SEED = 42
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")


def new_xgb():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         random_state=SEED)


def make_baselines(preprocessor_fn):
    """Three baseline pipelines. Cleaning is INSIDE the pipeline, so it is
    learned from the training fold only (no data leakage)."""
    return {
        "Logistic Regression": Pipeline([("clean", preprocessor_fn(scale=True)),
                                         ("model", LogisticRegression(max_iter=1000))]),
        "Random Forest": Pipeline([("clean", preprocessor_fn()),
                                   ("model", RandomForestClassifier(n_estimators=300,
                                                                    random_state=SEED))]),
        "XGBoost": Pipeline([("clean", preprocessor_fn()), ("model", new_xgb())]),
    }


def cross_validate_all(datasets):
    """datasets: {name: (X, y, preprocessor_fn)} -> results DataFrame."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for ds, (X, y, fn) in datasets.items():
        for name, pipe in make_baselines(fn).items():
            r = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
            row = {"Dataset": ds, "Model": name}
            for s in scoring:
                row[s] = f"{r['test_' + s].mean():.3f} \u00b1 {r['test_' + s].std():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def train_final(X, y, preprocessor_fn, name):
    """Fit the final XGBoost pipeline on ALL data and save the bundle."""
    pipe = Pipeline([("clean", preprocessor_fn()), ("model", new_xgb())]).fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}_xgb.joblib")
    joblib.dump({"pipeline": pipe, "X_train": X, "y_train": y}, path)
    print("saved", path)
    return pipe


def load_bundle(name):
    """name: 'heart' or 'ckd'. Returns the saved bundle dict."""
    path = os.path.join(MODEL_DIR, f"{name}_xgb.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run:  python src/models.py")
    return joblib.load(path)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    X_h, y_h = load_heart(os.path.join(ROOT, "data/raw/heart.csv"))
    X_c, y_c = load_ckd(os.path.join(ROOT, "data/raw/ckd.csv"))

    print("Cross-validating (5-fold x 3 repeats). This takes a few minutes...")
    res = cross_validate_all({"Heart": (X_h, y_h, heart_preprocessor),
                              "CKD": (X_c, y_c, ckd_preprocessor)})
    os.makedirs(REPORT_DIR, exist_ok=True)
    res.to_csv(os.path.join(REPORT_DIR, "cv_results.csv"), index=False)
    print(res.to_string(index=False))

    print("\nTraining final models on all data...")
    train_final(X_h, y_h, heart_preprocessor, "heart")
    train_final(X_c, y_c, ckd_preprocessor, "ckd")