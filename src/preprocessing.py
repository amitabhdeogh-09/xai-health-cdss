"""
Preprocessing for the Heart Disease and CKD datasets.

Design rule: nothing that LEARNS from data (imputing, scaling, encoding)
happens here directly. It lives inside a scikit-learn Pipeline, so it is
fitted on the training fold only. This prevents data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------
HEART_TARGET = "HeartDisease"
HEART_NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
HEART_CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

CKD_TARGET = "class"
CKD_NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
               "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
CKD_CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# Human-readable names (used later in the dashboard and SHAP plots)
CKD_LABELS = {
    "age": "Age", "bp": "Blood pressure", "sg": "Specific gravity",
    "al": "Albumin", "su": "Sugar", "rbc": "Red blood cells",
    "pc": "Pus cell", "pcc": "Pus cell clumps", "ba": "Bacteria",
    "bgr": "Blood glucose (random)", "bu": "Blood urea",
    "sc": "Serum creatinine", "sod": "Sodium", "pot": "Potassium",
    "hemo": "Hemoglobin", "pcv": "Packed cell volume",
    "wbcc": "White blood cell count", "rbcc": "Red blood cell count",
    "htn": "Hypertension", "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease", "appet": "Appetite",
    "pe": "Pedal edema", "ane": "Anemia",
}


# ----------------------------------------------------------------------
# Loading and cleaning (row-wise fixes only, nothing learned from data)
# ----------------------------------------------------------------------
def load_heart(path="data/raw/heart.csv"):
    """Return (X, y). Zero cholesterol / zero BP are impossible values,
    so they are converted to NaN and imputed later inside the pipeline."""
    df = pd.read_csv(path)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    X = df.drop(columns=[HEART_TARGET])
    y = df[HEART_TARGET].astype(int)
    return X, y


def load_ckd(path="data/raw/ckd.csv"):
    """Return (X, y) with y = 1 for CKD and 0 for not CKD."""
    df = pd.read_csv(path)
    for col in CKD_CATEGORICAL + [CKD_TARGET]:
        df[col] = df[col].astype("string").str.strip().str.lower()
    y = (df[CKD_TARGET] == "ckd").astype(int)
    X = df.drop(columns=[CKD_TARGET])
    for col in CKD_CATEGORICAL:                     # plain object columns for sklearn
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X, y


# ----------------------------------------------------------------------
# Pipeline builder
# ----------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols, scale=False):
    """
    Numeric     -> median imputation (+ optional standard scaling, needed for MLP / LogReg)
    Categorical -> most-frequent imputation -> one-hot (binary columns stay a single column,
                   so SHAP plots remain readable, e.g. 'Sex_M')
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                 sparse_output=False)),
    ]
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric_cols),
         ("cat", Pipeline(cat_steps), categorical_cols)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def heart_preprocessor(scale=False):
    return build_preprocessor(HEART_NUMERIC, HEART_CATEGORICAL, scale)


def ckd_preprocessor(scale=False):
    return build_preprocessor(CKD_NUMERIC, CKD_CATEGORICAL, scale)
